"""
graph/sql/extraction.py — ``store.extraction`` over SQL: targets, extracted marks, the sink.

Translation of ``graph/neo4j/extraction.py``. The sink holds **one** async connection and
runs every call as its own transaction (autocommit), in the order the domain
(``entity_extraction/graph_mapper.py``) emits them, with the retries on the same
statements. ``ExtractionResult`` carries the ``_document_key`` of its StructureNode;
``ModelInstance``, ``LabeledEntity`` and ``Entity`` are deduplicated within a tenant and
carry none.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from scinr.newton.entity_extraction.state import ExtractionTarget
from scinr.newton.graph.base import ExtractionOps
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._shared import (
    delete_found,
    delete_stale_extraction_result,
    node_provenance,
    prov_values,
    provenance_on_create,
    provenance_on_match,
)
from scinr.newton.graph.sql.annotation import tree_nodes
from scinr.newton.graph.types import Provenance, RawProvenance
from scinr.newton.ingest.freeze import validate_identifier
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.retry import with_sql_retry
from scinr.newton.utils.tenancy import tenant_key

if TYPE_CHECKING:
    from scinr.newton.graph.sinks import ExtractionSink

__all__ = ["SqlExtractionOps", "SqlExtractionSink", "create_extraction_result", "has_edge"]

log = logging.getLogger(__name__)

D = t.documents
E = t.edges
IU = t.info_units
MD = t.model_decisions
CM = t.complementary_matches
SF = t.supplementary_fields
ER = t.extraction_results
MI = t.model_instances

# A closed set, checked again here (the label picks the table).
_PARENT_LABELS = frozenset({"ExtractionResult", "ModelInstance"})


def _check_label(label: str) -> None:
    """Reject any parent label other than the two a model subgraph can hang from."""
    if label not in _PARENT_LABELS:
        raise ValueError(f"parent label must be one of {sorted(_PARENT_LABELS)}, got {label!r}")


def has_edge(rel_type: str, src_label: str, src_key: Any, dst_label: str | None = None) -> Any:
    """``EXISTS ((src)-[:rel_type]->(:dst_label))``."""
    condition = sa.and_(E.c.src_label == src_label, E.c.src_key == src_key, E.c.type == rel_type)
    if dst_label is not None:
        condition = sa.and_(condition, E.c.dst_label == dst_label)
    return sa.exists().where(condition)


async def _exists(connection: AsyncConnection, label: str, key: str) -> bool:
    column = c.key_column(label)
    return (await connection.execute(sa.select(column).where(column == key))).first() is not None


async def _tenant_of(connection: AsyncConnection, label: str, key: str) -> tuple[bool, str | None]:
    """``(exists, tenant_id)`` of a data node."""
    target = c.table(label)
    row = (
        await connection.execute(sa.select(target.c.tenant_id).where(c.key_column(label) == key))
    ).first()
    return (row is not None, row[0] if row is not None else None)


async def create_extraction_result(
    connection: AsyncConnection, node_full_id: str, uid: str, values: dict[str, Any]
) -> None:
    """MATCH (n:StructureNode {id}) CREATE (er:ExtractionResult {...}) CREATE (n)-[:HAS_EXTRACTION]->(er).

    Nothing is written when the node does not exist. A duplicate uid violates the primary
    key, as it violates the uniqueness constraint in Neo4j.
    """
    found = await node_provenance(connection, node_full_id)
    if found is None:
        return
    _, document_key = found
    await connection.execute(
        c.create_node("ExtractionResult", {"uid": uid, **values}, {"_document_key": document_key})
    )
    await connection.execute(
        c.merge_edge(
            "HAS_EXTRACTION",
            "StructureNode",
            node_full_id,
            "ExtractionResult",
            uid,
            tenant=values.get("tenant_id"),
        )
    )


def _merge_dedup(
    label: str,
    uid: str,
    prov: Provenance,
    *,
    on_create: dict[str, Any],
    on_match: dict[str, Any],
) -> sa.Insert:
    """MERGE (x:<label> {uid}) ON CREATE SET on_create + provenance ON MATCH SET on_match +
    provenance (``provenance_set_clause``)."""
    target = c.table(label)
    return c.merge_node(
        label,
        {"uid": uid},
        on_create={**on_create, **provenance_on_create(prov)},
        on_match={**on_match, **provenance_on_match(target, prov)},
    )


class SqlExtractionSink:
    """:class:`~scinr.newton.graph.sinks.ExtractionSink` over one async connection.

    Each method is one transaction (the autocommit statement of the Cypher); the retried
    ones are retried whole.
    """

    def __init__(self, connection: AsyncConnection) -> None:
        self._connection = connection

    async def _tx(self, work: Callable[[], Awaitable[Any]]) -> Any:
        async with self._connection.begin():
            return await work()

    async def _retried(self, work: Callable[[], Awaitable[Any]]) -> Any:
        return await with_sql_retry(lambda: self._tx(work))

    # ── Shared ────────────────────────────────────────────────────────────

    @async_sql_errors()
    async def read_node_provenance(self, node_full_id: str) -> RawProvenance | None:
        """Cypher: MATCH (n:StructureNode {id}) RETURN its tenant_id, created_by_user_id, job_id."""

        async def work() -> RawProvenance | None:
            found = await node_provenance(self._connection, node_full_id)
            return None if found is None else found[0]

        return await self._tx(work)

    # ── write_extraction_subgraph ─────────────────────────────────────────

    @async_sql_errors()
    async def delete_stale_extraction_result(self, node_full_id: str) -> None:
        """The ER of the node and every ModelInstance within 10 hops (``_shared``)."""
        await self._tx(lambda: delete_stale_extraction_result(self._connection, node_full_id))

    @async_sql_errors()
    async def create_extraction_result(
        self,
        node_full_id: str,
        uid: str,
        document_name: str,
        model_class: str,
        timestamp: str,
        prov: Provenance,
    ) -> None:
        """MATCH n CREATE (er:ExtractionResult {uid, node_full_id, document_name, model_class,
        timestamp, provenance}) CREATE (n)-[:HAS_EXTRACTION]->(er)."""
        values = {
            "node_full_id": node_full_id,
            "document_name": document_name,
            "model_class": model_class,
            "timestamp": timestamp,
            **prov_values(prov),
        }
        await self._tx(
            lambda: create_extraction_result(self._connection, node_full_id, uid, values)
        )

    @async_sql_errors()
    async def link_catalog_model(self, er_uid: str, name: str, *, primary: bool) -> None:
        """MATCH er MERGE (cm:CatalogModel {name}) MERGE (er)-[:USES_PRIMARY|COMPLEMENTARY_MODEL]->(cm). Retried."""
        rel_type = "USES_PRIMARY_MODEL" if primary else "USES_COMPLEMENTARY_MODEL"

        async def work() -> None:
            exists, tenant = await _tenant_of(self._connection, "ExtractionResult", er_uid)
            if not exists:
                return
            await self._connection.execute(c.merge_node("CatalogModel", {"name": name}))
            await self._connection.execute(
                c.merge_edge(
                    rel_type, "ExtractionResult", er_uid, "CatalogModel", name, tenant=tenant
                )
            )

        await self._retried(work)

    async def _link_child(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        props = None if list_index is None else {"index": list_index}
        await self._connection.execute(
            c.merge_edge(
                rel_name,
                parent_label,
                parent_uid,
                "ModelInstance",
                child_uid,
                tenant=prov.tenant_id,
                props=props,
            )
        )

    @async_sql_errors()
    async def merge_keyed_instance(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        model_class: str,
        key_fields: dict[str, str],
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        """MATCH parent; MERGE (child:ModelInstance {uid}) ON CREATE SET model_class, keys,
        provenance ON MATCH SET keys, provenance; MERGE (parent)-[:<rel> [{index}]]->(child)."""
        _check_label(parent_label)
        validate_identifier(rel_name)

        async def work() -> None:
            if not await _exists(self._connection, parent_label, parent_uid):
                return
            await self._connection.execute(
                _merge_dedup(
                    "ModelInstance",
                    child_uid,
                    prov,
                    on_create={"model_class": model_class, **key_fields},
                    on_match=dict(key_fields),
                )
            )
            await self._link_child(parent_label, parent_uid, rel_name, child_uid, prov, list_index)

        await self._tx(work)

    @async_sql_errors()
    async def create_instance(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        model_class: str,
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        """MATCH parent; CREATE (child:ModelInstance {uid, model_class, tenant, [user], [job]});
        CREATE (parent)-[:<rel> [{index}]]->(child)."""
        _check_label(parent_label)
        validate_identifier(rel_name)

        async def work() -> None:
            if not await _exists(self._connection, parent_label, parent_uid):
                return
            await self._connection.execute(
                c.create_node(
                    "ModelInstance",
                    {"uid": child_uid, "model_class": model_class, **provenance_on_create(prov)},
                )
            )
            await self._link_child(parent_label, parent_uid, rel_name, child_uid, prov, list_index)

        await self._tx(work)

    @async_sql_errors()
    async def merge_labeled_entity(
        self, uid: str, label: str, value: str, normalized_value: str, prov: Provenance
    ) -> None:
        """MERGE (le:LabeledEntity {uid}) ON CREATE SET label, normalized_value, value,
        provenance ON MATCH SET provenance. Retried."""
        statement = _merge_dedup(
            "LabeledEntity",
            uid,
            prov,
            on_create={"label": label, "normalized_value": normalized_value, "value": value},
            on_match={},
        )
        await self._retried(lambda: self._connection.execute(statement))

    @async_sql_errors()
    async def link_reference(
        self,
        parent_label: str,
        parent_uid: str,
        le_uid: str,
        field_name: str,
        list_index: int | None,
    ) -> None:
        """MATCH parent, le; MERGE (parent)-[:REFERENCES {field_name[, list_index]}]->(le)."""
        _check_label(parent_label)
        props: dict[str, Any] = {"field_name": field_name}
        if list_index is not None:
            props["list_index"] = list_index

        async def work() -> None:
            exists, tenant = await _tenant_of(self._connection, parent_label, parent_uid)
            if not exists or not await _exists(self._connection, "LabeledEntity", le_uid):
                return
            await self._connection.execute(
                c.merge_edge(
                    "REFERENCES",
                    parent_label,
                    parent_uid,
                    "LabeledEntity",
                    le_uid,
                    tenant=tenant,
                    props=props,
                )
            )

        await self._tx(work)

    @async_sql_errors()
    async def merge_instance_shell(
        self, tgt_uid: str, model_class: str, key_fields: dict[str, str], prov: Provenance
    ) -> None:
        """MERGE (tgt:ModelInstance {uid}) ON CREATE SET model_class, keys, provenance ON MATCH
        SET keys, provenance. Retried."""
        statement = _merge_dedup(
            "ModelInstance",
            tgt_uid,
            prov,
            on_create={"model_class": model_class, **key_fields},
            on_match=dict(key_fields),
        )
        await self._retried(lambda: self._connection.execute(statement))

    @async_sql_errors()
    async def link_instance_relationship(
        self, src_label: str, src_uid: str, tgt_uid: str, rel_type: str
    ) -> None:
        """MATCH src, tgt:ModelInstance; MERGE (src)-[:<rel_type>]->(tgt). Retried."""
        _check_label(src_label)
        validate_identifier(rel_type)

        async def work() -> None:
            exists, tenant = await _tenant_of(self._connection, src_label, src_uid)
            if not exists or not await _exists(self._connection, "ModelInstance", tgt_uid):
                return
            await self._connection.execute(
                c.merge_edge(rel_type, src_label, src_uid, "ModelInstance", tgt_uid, tenant=tenant)
            )

        await self._retried(work)

    @async_sql_errors()
    async def set_instance_properties(
        self, parent_label: str, parent_uid: str, props: dict[str, Any]
    ) -> None:
        """MATCH (parent {uid}) SET parent.k = v, … in the order of *props* (``null`` removes)."""
        _check_label(parent_label)
        if not props:
            return
        target = c.table(parent_label)
        statement = (
            sa.update(target)
            .where(c.key_column(parent_label) == parent_uid)
            .values(**c.set_columns(target, props))
        )
        await self._tx(lambda: self._connection.execute(statement))

    @async_sql_errors()
    async def link_field_relationship(self, src_uid: str, tgt_uid: str, rel_type: str) -> None:
        """MATCH src, tgt (LabeledEntity); MERGE (src)-[:<rel_type>]->(tgt)."""
        validate_identifier(rel_type)

        async def work() -> None:
            exists, tenant = await _tenant_of(self._connection, "LabeledEntity", src_uid)
            if not exists or not await _exists(self._connection, "LabeledEntity", tgt_uid):
                return
            await self._connection.execute(
                c.merge_edge(
                    rel_type, "LabeledEntity", src_uid, "LabeledEntity", tgt_uid, tenant=tenant
                )
            )

        await self._tx(work)

    # ── write_triple_subgraph ─────────────────────────────────────────────

    @async_sql_errors()
    async def delete_triple_extraction_result(self, node_full_id: str) -> None:
        """MATCH (n {id})-[:HAS_EXTRACTION]->(er) DETACH DELETE er (the Entities stay)."""
        found = c.follow(
            c.nodes_with_keys("StructureNode", [node_full_id]),
            [("HAS_EXTRACTION", "ExtractionResult")],
        )
        await self._tx(lambda: delete_found(self._connection, found))

    @async_sql_errors()
    async def create_triple_extraction_result(
        self, node_full_id: str, uid: str, document_name: str, timestamp: str, prov: Provenance
    ) -> None:
        """MATCH n CREATE (er:ExtractionResult {..., model_class: 'Triple'}) + HAS_EXTRACTION."""
        values = {
            "node_full_id": node_full_id,
            "document_name": document_name,
            "model_class": "Triple",
            "timestamp": timestamp,
            **prov_values(prov),
        }
        await self._tx(
            lambda: create_extraction_result(self._connection, node_full_id, uid, values)
        )

    @async_sql_errors()
    async def merge_entity(
        self, uid: str, value: str, normalized_value: str, prov: Provenance
    ) -> None:
        """MERGE (e:Entity {uid}) ON CREATE SET normalized_value, value, provenance ON MATCH SET
        provenance. Retried."""
        statement = _merge_dedup(
            "Entity",
            uid,
            prov,
            on_create={"normalized_value": normalized_value, "value": value},
            on_match={},
        )
        await self._retried(lambda: self._connection.execute(statement))

    @async_sql_errors()
    async def link_triple_entity(
        self, er_uid: str, entity_uid: str, role: Literal["subject", "object"]
    ) -> None:
        """MATCH er, e; MERGE (er)-[:HAS_ENTITY {role}]->(e)."""
        if role not in ("subject", "object"):
            raise ValueError(f"role must be 'subject' or 'object', got {role!r}")

        async def work() -> None:
            exists, tenant = await _tenant_of(self._connection, "ExtractionResult", er_uid)
            if not exists or not await _exists(self._connection, "Entity", entity_uid):
                return
            await self._connection.execute(
                c.merge_edge(
                    "HAS_ENTITY",
                    "ExtractionResult",
                    er_uid,
                    "Entity",
                    entity_uid,
                    tenant=tenant,
                    props={"role": role},
                )
            )

        await self._tx(work)

    @async_sql_errors()
    async def merge_predicate(
        self, subj_uid: str, obj_uid: str, rel_type: str, predicate_raw: str
    ) -> None:
        """MATCH subj, obj; MERGE (subj)-[r:<rel_type>]->(obj) ON CREATE SET r.predicate_raw."""
        validate_identifier(rel_type)

        async def work() -> None:
            exists, tenant = await _tenant_of(self._connection, "Entity", subj_uid)
            if not exists or not await _exists(self._connection, "Entity", obj_uid):
                return
            await self._connection.execute(
                c.merge_edge(
                    rel_type,
                    "Entity",
                    subj_uid,
                    "Entity",
                    obj_uid,
                    tenant=tenant,
                    props={"predicate_raw": predicate_raw},
                )
            )

        await self._tx(work)


@asynccontextmanager
async def _open_extraction_sink() -> AsyncIterator[SqlExtractionSink]:
    async with async_sql_errors():
        async with c.async_write_engine().connect() as connection:
            yield SqlExtractionSink(connection)


def _latest_doc(tenant_id: str | None, doc_path: str) -> Any:
    return sa.and_(
        D.c.tenant_id == tenant_key(tenant_id), D.c.path == doc_path, D.c.latest == sa.true()
    )


class SqlExtractionOps(ExtractionOps):
    """``store.extraction`` over SQL (asynchronous, process-wide async engine)."""

    def sink(self) -> AbstractAsyncContextManager[ExtractionSink]:
        """One async connection; each sink call is its own transaction (autocommit)."""
        return _open_extraction_sink()

    async def fetch_targets(
        self,
        *,
        tenant_id: str | None,
        doc_path: str,
        only_unextracted: bool = False,
    ) -> list[ExtractionTarget]:
        """Extraction targets of the latest document at *doc_path*, by appearance order.

        Cypher (the APOC version): one row per (StructureNode, ModelDecision) of the tree with
        an InfoUnit and no ``source: 'tabular_raw'`` ExtractionResult (no ExtractionResult at
        all with *only_unextracted*); the distinct complementary matches and supplementary
        fields of the decision; the InfoUnits sorted by ``order``. Rows by appearance_order.
        """
        nodes = tree_nodes(_latest_doc(tenant_id, doc_path)).subquery("n")
        raw_er = sa.exists().where(
            E.c.src_label == "StructureNode",
            E.c.src_key == nodes.c.id,
            E.c.type == "HAS_EXTRACTION",
            E.c.dst_label == "ExtractionResult",
            E.c.dst_key.in_(
                sa.select(ER.c.uid).where(c.json_text(ER.c.props, "source") == "tabular_raw")
            ),
        )
        conditions = [has_edge("HAS_INFO_UNIT", "StructureNode", nodes.c.id, "InfoUnit"), ~raw_er]
        if only_unextracted:
            conditions.append(
                ~has_edge("HAS_EXTRACTION", "StructureNode", nodes.c.id, "ExtractionResult")
            )
        decision = E.alias("decision")
        d = c.dialect()
        statement = (
            sa.select(
                nodes.c.id,
                nodes.c.node_id,
                nodes.c.title,
                MD.c.uid.label("md_uid"),
                MD.c.matched_model_class,
            )
            .select_from(nodes)
            .join(
                decision,
                sa.and_(
                    decision.c.src_label == "StructureNode",
                    decision.c.src_key == nodes.c.id,
                    decision.c.type == "HAS_MODEL_DECISION",
                    decision.c.dst_label == "ModelDecision",
                ),
            )
            .join(MD, MD.c.uid == decision.c.dst_key)
            .where(*conditions)
            .order_by(
                d.order_value(nodes.c.appearance_order),
                d.order_text(nodes.c.id),
                d.order_text(MD.c.uid),
            )
        )
        targets: list[ExtractionTarget] = []
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                for row in (await connection.execute(statement)).all():
                    targets.append(
                        ExtractionTarget(
                            node_full_id=row.id,
                            node_id=row.node_id,
                            node_title=row.title,
                            model_class=row.matched_model_class,
                            complementary_models=await _complementary(connection, row.md_uid),
                            supplementary_fields=await _supplementary(connection, row.md_uid),
                            info_units=await _target_info_units(connection, row.id),
                        )
                    )
        log.info(
            "fetch_targets: found %d targets for document %r (tenant=%r)",
            len(targets),
            doc_path,
            tenant_id,
        )
        return targets

    async def mark_info_units_extracted(self, node_full_id: str) -> None:
        """Cypher: MATCH (n {id})-[:HAS_INFO_UNIT]->(iu) WHERE iu.extracted IS NULL SET
        iu.extracted = <ISO timestamp>."""
        timestamp = datetime.now(UTC).isoformat()
        hop = c.follow(
            c.nodes_with_keys("StructureNode", [node_full_id]), [("HAS_INFO_UNIT", "InfoUnit")]
        )
        keys = hop.subquery()
        statement = (
            sa.update(IU)
            .where(
                IU.c.uid.in_(sa.select(keys.c.key)), c.json_text(IU.c.props, "extracted").is_(None)
            )
            .values(props=c.patch_props(IU.c.props, {"extracted": timestamp}))
        )
        async with async_sql_errors():
            async with c.async_write_engine().begin() as connection:
                updated = (await connection.execute(statement)).rowcount
        log.info(
            "mark_info_units_extracted: marked %d InfoUnits as extracted for node %r",
            updated,
            node_full_id,
        )


async def _decision_children(
    connection: AsyncConnection, md_uid: str, rel_type: str, label: str
) -> list[str]:
    hop = c.follow(c.nodes_with_keys("ModelDecision", [md_uid]), [(rel_type, label)]).subquery()
    return c.column_values(await connection.execute(sa.select(hop.c.key)))


async def _complementary(connection: AsyncConnection, md_uid: str) -> list[dict]:
    """collect(DISTINCT {model_class, coverage_note}) of (md)-[:HAS_COMPLEMENTARY_MATCH]->(cm)."""
    keys = await _decision_children(
        connection, md_uid, "HAS_COMPLEMENTARY_MATCH", "ComplementaryMatch"
    )
    if not keys:
        return []
    out: list[dict] = []
    rows = await connection.execute(
        sa.select(CM).where(CM.c.uid.in_(keys)).order_by(c.dialect().order_text(CM.c.uid))
    )
    for row in rows.mappings():
        props = c.node_from_row("ComplementaryMatch", row)[1]
        item = {
            "model_class": props.get("model_class"),
            "coverage_note": props.get("coverage_note"),
        }
        if item not in out:
            out.append(item)
    return out


async def _supplementary(connection: AsyncConnection, md_uid: str) -> list[dict]:
    """collect(DISTINCT {field_name, field_type, description, required}) of the SupplementaryFields."""
    keys = await _decision_children(
        connection, md_uid, "HAS_SUPPLEMENTARY_FIELD", "SupplementaryField"
    )
    if not keys:
        return []
    out: list[dict] = []
    rows = await connection.execute(
        sa.select(SF).where(SF.c.uid.in_(keys)).order_by(c.dialect().order_text(SF.c.uid))
    )
    for row in rows.mappings():
        props = c.node_from_row("SupplementaryField", row)[1]
        item = {
            name: props.get(name)
            for name in ("field_name", "field_type", "description", "required")
        }
        if item not in out:
            out.append(item)
    return out


async def _target_info_units(connection: AsyncConnection, node_id: str) -> list[dict]:
    """collect(iu {.uid, .title, .description, .order}) sorted by ``order`` (``order or 0``)."""
    hop = c.follow(
        c.nodes_with_keys("StructureNode", [node_id]), [("HAS_INFO_UNIT", "InfoUnit")]
    ).subquery()
    rows = await connection.execute(
        sa.select(IU.c.uid, IU.c.title, IU.c.description, IU.c.order)
        .where(IU.c.uid.in_(sa.select(hop.c.key)))
        .order_by(c.dialect().order_text(IU.c.uid))
    )
    units = [
        {"uid": row.uid, "title": row.title, "description": row.description, "order": row.order}
        for row in rows
    ]
    units.sort(key=lambda unit: unit.get("order") or 0)
    return units
