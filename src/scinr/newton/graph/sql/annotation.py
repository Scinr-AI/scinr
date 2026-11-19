"""
graph/sql/annotation.py — ``store.annotation`` over SQL: catalog, themes, annotation.

Translation of ``graph/neo4j/annotation.py``. Every write is **autocommit** like there: each
Cypher statement is one transaction here, in the same order, with the retries on the same
statements. The catalog nodes live in ``catalog_models``, ``model_fields``,
``entity_labels`` and ``themes``, the relationships among them in ``catalog_edges``.

The decision nodes (ModelDecision and what hangs from it) carry the ``_document_key`` of
the StructureNode they hang from.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Collection
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from scinr.newton.config import get_config
from scinr.newton.graph.base import AnnotationOps
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._shared import (
    STALE_MODEL_DECISION_STEPS,
    delete_stale_extraction_result,
    node_provenance,
    prov_values,
)
from scinr.newton.graph.sql.documents import latest_documents
from scinr.newton.graph.types import DocumentSelector, RawProvenance
from scinr.newton.ingest.nodes import _make_uid
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.keys import encode_key
from scinr.newton.sql.retry import with_sql_retry
from scinr.newton.sql.schema import ensure_schema, rls_enabled
from scinr.newton.utils.tenancy import tenant_key

if TYPE_CHECKING:
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.graph.sinks import CatalogSink

__all__ = ["SqlAnnotationOps", "SqlCatalogSink", "tree_nodes"]

log = logging.getLogger(__name__)

D = t.documents
SN = t.structure_nodes
IU = t.info_units
E = t.edges

_QUALIFYING_CHILD_ROLES = ("freeform_block", "table", "field_group")


def _has_edge(rel_type: str, src_label: str, src_key: Any) -> Any:
    """``EXISTS ((src)-[:rel_type]->())``."""
    return sa.exists().where(
        E.c.src_label == src_label, E.c.src_key == src_key, E.c.type == rel_type
    )


def tree_nodes(doc_condition: Any) -> sa.Select:
    """The StructureNodes of the documents matching *doc_condition*: the
    ``(d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)`` of Cypher, through
    ``_document_key`` (one row per (document, node))."""
    return sa.select(SN).join(D, SN.c._document_key == D.c._key).where(doc_condition)


# ---------------------------------------------------------------------------
# Catalog sink
# ---------------------------------------------------------------------------


class SqlCatalogSink:
    """:class:`~scinr.newton.graph.sinks.CatalogSink` over one async connection.

    Each method is its own transaction (autocommit), with the statements of the Cypher.
    """

    def __init__(self, connection: AsyncConnection) -> None:
        self._connection = connection

    async def _run(self, *statements: Any) -> None:
        async with self._connection.begin():
            for statement in statements:
                await self._connection.execute(statement)

    async def _exists(self, label: str, key: str) -> bool:
        column = c.key_column(label)
        async with self._connection.begin():
            row = (await self._connection.execute(sa.select(column).where(column == key))).first()
        return row is not None

    @async_sql_errors()
    async def merge_catalog_model(self, name: str, description: str, selectable: bool) -> None:
        """MERGE (cm:CatalogModel {name}) SET description, selectable."""
        values = {"description": description, "selectable": selectable}
        await self._run(
            c.merge_node("CatalogModel", {"name": name}, on_create=values, on_match=values)
        )

    @async_sql_errors()
    async def merge_model_field(
        self, model_name: str, field_name: str, field_type: str, description: str, required: bool
    ) -> None:
        """MERGE (mf:ModelField {name, model}) ON CREATE SET type; MATCH cm;
        MERGE (cm)-[r:HAS_FIELD]->(mf) SET r.description, r.required."""
        key = {"name": field_name, "model": model_name}
        edge_props = {"description": description, "required": required}
        field_key = encode_key("ModelField", key)
        async with self._connection.begin():
            await self._connection.execute(
                c.merge_node("ModelField", key, on_create={"type": field_type})
            )
            model = c.key_column("CatalogModel")
            if (
                await self._connection.execute(sa.select(model).where(model == model_name))
            ).first():
                await self._connection.execute(
                    c.merge_edge(
                        "HAS_FIELD",
                        "CatalogModel",
                        model_name,
                        "ModelField",
                        field_key,
                        tenant=None,
                        props=edge_props,
                        on_match=edge_props,
                    )
                )

    @async_sql_errors()
    async def merge_aggregates(
        self, src_name: str, tgt_name: str, field_name: str, is_list: bool, required: bool
    ) -> None:
        """MATCH src, tgt; MERGE (src)-[r:AGGREGATES {field_name}]->(tgt) SET is_list, required."""
        model = c.key_column("CatalogModel")
        async with self._connection.begin():
            found = {
                row[0]
                for row in await self._connection.execute(
                    sa.select(model).where(model.in_([src_name, tgt_name]))
                )
            }
            if {src_name, tgt_name} <= found:
                on_match = {"is_list": is_list, "required": required}
                await self._connection.execute(
                    c.merge_edge(
                        "AGGREGATES",
                        "CatalogModel",
                        src_name,
                        "CatalogModel",
                        tgt_name,
                        tenant=None,
                        props={"field_name": field_name, **on_match},
                        on_match=on_match,
                    )
                )

    @async_sql_errors()
    async def set_field_metadata(
        self, model_name: str, field_name: str, is_instance_key: bool, entity_label: str | None
    ) -> None:
        """MERGE (mf:ModelField {name, model}) SET is_instance_key, entity_label (null removes)."""
        values = {"is_instance_key": is_instance_key, "entity_label": entity_label}
        key = {"name": field_name, "model": model_name}
        await self._run(c.merge_node("ModelField", key, on_create=values, on_match=values))

    async def ensure_instance_key_indexes(self, props: Collection[str]) -> None:
        """One ``(tenant_id, props-><prop>)`` index per key property, alphabetically; best effort.

        Cypher: CREATE INDEX idx_mi_key_<prop> IF NOT EXISTS FOR (mi:ModelInstance) ON
        (mi.tenant_id, mi.<prop>). A failure is logged and never raised.

        In RLS mode the runtime role cannot run DDL: the same index (same name, same
        expression) is created by the ``SECURITY DEFINER`` function
        ``newton_ensure_instance_key_index``, which can create nothing else.
        """
        d = c.dialect()
        schema = get_config().sql_schema
        await asyncio.to_thread(ensure_schema)  # the mode is read when the schema is verified
        use_function = rls_enabled()
        for prop in sorted(props):
            try:
                if use_function:
                    statement = sa.select(
                        getattr(sa.func, schema).newton_ensure_instance_key_index(prop)
                    )
                    async with self._connection.begin():
                        await self._connection.execute(statement)
                    continue
                statement = d.instance_key_index_sql(prop, schema=schema)
                async with self._connection.begin():
                    await self._connection.exec_driver_sql(statement)
            except Exception as exc:  # noqa: BLE001 — a missing index must not block ingestion
                log.warning(
                    "Could not create index model_instances_key_%s on ModelInstance.%s: %s",
                    prop,
                    prop,
                    exc,
                )

    @async_sql_errors()
    async def link_produces_entity(
        self, model_name: str, field_name: str, entity_label: str
    ) -> None:
        """MERGE (el:EntityLabel {label}); MATCH cm; MERGE (cm)-[:PRODUCES_ENTITY {field_name}]->(el)."""
        model = c.key_column("CatalogModel")
        async with self._connection.begin():
            await self._connection.execute(c.merge_node("EntityLabel", {"label": entity_label}))
            if (
                await self._connection.execute(sa.select(model).where(model == model_name))
            ).first():
                await self._connection.execute(
                    c.merge_edge(
                        "PRODUCES_ENTITY",
                        "CatalogModel",
                        model_name,
                        "EntityLabel",
                        entity_label,
                        tenant=None,
                        props={"field_name": field_name},
                    )
                )

    @async_sql_errors()
    async def merge_field_relationship(
        self,
        rel_type: str,
        source_label: str,
        target_label: str,
        via_model: str,
        from_field: str,
        to_field: str,
    ) -> None:
        """MERGE both EntityLabels; MERGE (src)-[:<rel_type> {via_model, from_field, to_field}]->(tgt)."""
        props = {"via_model": via_model, "from_field": from_field, "to_field": to_field}
        await self._run(
            c.merge_node("EntityLabel", {"label": source_label}),
            c.merge_node("EntityLabel", {"label": target_label}),
            c.merge_edge(
                rel_type,
                "EntityLabel",
                source_label,
                "EntityLabel",
                target_label,
                tenant=None,
                props=props,
            ),
        )

    @async_sql_errors()
    async def merge_instance_relationship(
        self,
        rel_type: str,
        src_name: str,
        tgt_name: str,
        via_field: str,
        join_via_json: str,
        is_composite: bool,
    ) -> None:
        """MERGE both CatalogModels; MERGE (src)-[r:<rel_type> {via_field, join_via}]->(tgt)
        SET r.composite."""
        identity = {"via_field": via_field, "join_via": join_via_json}
        await self._run(
            c.merge_node("CatalogModel", {"name": src_name}),
            c.merge_node("CatalogModel", {"name": tgt_name}),
            c.merge_edge(
                rel_type,
                "CatalogModel",
                src_name,
                "CatalogModel",
                tgt_name,
                tenant=None,
                props={**identity, "composite": is_composite},
                on_match={"composite": is_composite},
            ),
        )

    @async_sql_errors()
    async def merge_theme(self, path: str, name: str) -> None:
        """MERGE (t:Theme {path}) SET t.name."""
        await self._run(
            c.merge_node("Theme", {"path": path}, on_create={"name": name}, on_match={"name": name})
        )

    @async_sql_errors()
    async def link_subtopic(self, parent_path: str, child_path: str) -> None:
        """MATCH parent, child; MERGE (parent)-[:HAS_SUBTOPIC]->(child)."""
        theme = c.key_column("Theme")
        async with self._connection.begin():
            found = {
                row[0]
                for row in await self._connection.execute(
                    sa.select(theme).where(theme.in_([parent_path, child_path]))
                )
            }
            if {parent_path, child_path} <= found:
                await self._connection.execute(
                    c.merge_edge(
                        "HAS_SUBTOPIC", "Theme", parent_path, "Theme", child_path, tenant=None
                    )
                )

    @async_sql_errors()
    async def link_model_to_theme(self, model_name: str, path: str) -> None:
        """MERGE cm, MERGE t, MERGE (cm)-[:BELONGS_TO_THEME]->(t)."""
        await self._run(
            c.merge_node("CatalogModel", {"name": model_name}),
            c.merge_node("Theme", {"path": path}),
            c.merge_edge(
                "BELONGS_TO_THEME", "CatalogModel", model_name, "Theme", path, tenant=None
            ),
        )


@asynccontextmanager
async def _open_catalog_sink() -> AsyncIterator[SqlCatalogSink]:
    async with async_sql_errors():
        async with c.async_write_engine().connect() as connection:
            yield SqlCatalogSink(connection)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


async def _info_units_of(connection: AsyncConnection, full_node_id: str) -> list[dict]:
    """(n {id})-[:HAS_INFO_UNIT]->(iu:InfoUnit) RETURN uid, title, description, order ORDER BY uid."""
    hop = c.follow(
        c.nodes_with_keys("StructureNode", [full_node_id]), [("HAS_INFO_UNIT", "InfoUnit")]
    )
    keys = hop.subquery()
    rows = await connection.execute(
        sa.select(IU.c.uid, IU.c.title, IU.c.description, IU.c.order)
        .where(IU.c.uid.in_(sa.select(keys.c.key)))
        .order_by(c.dialect().order_text(IU.c.uid))
    )
    return [
        {
            "info_unit_id": row.uid,
            "title": row.title,
            "description": row.description,
            "order": row.order,
        }
        for row in rows
    ]


async def _qualifying_children(connection: AsyncConnection, full_node_id: str) -> list[dict]:
    """(n {id})-[:HAS_CHILD]->(c) WHERE c.role IN [...] ORDER BY c.appearance_order."""
    hop = c.follow(
        c.nodes_with_keys("StructureNode", [full_node_id]), [("HAS_CHILD", "StructureNode")]
    )
    keys = hop.subquery()
    rows = await connection.execute(
        sa.select(SN.c.id, SN.c.node_id, SN.c.title, SN.c.role, SN.c.appearance_order)
        .where(SN.c.id.in_(sa.select(keys.c.key)), SN.c.role.in_(_QUALIFYING_CHILD_ROLES))
        .order_by(c.dialect().order_value(SN.c.appearance_order), c.dialect().order_text(SN.c.id))
    )
    return [
        {
            "full_id": row.id,
            "node_id": row.node_id,
            "title": row.title,
            "role": row.role,
            "appearance_order": row.appearance_order,
        }
        for row in rows
    ]


async def _node_context(
    connection: AsyncConnection,
    full_node_id: str,
    node_id: str,
    title: str | None,
    role: str | None,
    depth: int,
    max_depth: int,
    visited: set[str] | None,
) -> dict:
    """``_fetch_node_context_with_session`` of Neo4j on one connection (sequential children)."""
    if visited is None:
        visited = set()
    if depth >= max_depth or full_node_id in visited:
        return {
            "full_id": full_node_id,
            "node_id": node_id,
            "title": title,
            "role": role,
            "depth": depth,
            "depth_limit_reached": True,
            "info_units": [],
            "children": [],
        }
    visited = visited | {full_node_id}
    info_units = await _info_units_of(connection, full_node_id)
    children = []
    for child in await _qualifying_children(connection, full_node_id):
        children.append(
            await _node_context(
                connection,
                child["full_id"],
                child["node_id"],
                child.get("title"),
                child.get("role"),
                depth + 1,
                max_depth,
                visited,
            )
        )
    return {
        "full_id": full_node_id,
        "node_id": node_id,
        "title": title,
        "role": role,
        "depth": depth,
        "depth_limit_reached": False,
        "info_units": info_units,
        "children": children,
    }


def _latest_doc(tenant_id: str | None, doc_path: str) -> Any:
    return sa.and_(
        D.c.tenant_id == tenant_key(tenant_id), D.c.path == doc_path, D.c.latest == sa.true()
    )


def _nodes_with_info_units(doc_condition: Any, *, only_unannotated: bool) -> sa.Select:
    nodes = tree_nodes(doc_condition).subquery()
    conditions = [_has_edge("HAS_INFO_UNIT", "StructureNode", nodes.c.id)]
    if only_unannotated:
        conditions.append(~_has_edge("HAS_MODEL_DECISION", "StructureNode", nodes.c.id))
    return sa.select(nodes).where(*conditions)


class SqlAnnotationOps(AnnotationOps):
    """``store.annotation`` over SQL (asynchronous, process-wide async engine)."""

    def catalog_sink(self) -> AbstractAsyncContextManager[CatalogSink]:
        """One async connection; each sink call is its own transaction (autocommit)."""
        return _open_catalog_sink()

    async def count_annotated_nodes(self, selector: DocumentSelector) -> int:
        """Cypher: MATCH <latest>-[:HAS_STRUCTURE|HAS_CHILD*1..]->(sn)-[:HAS_MODEL_DECISION]->()
        RETURN count(sn) (one row per decision link)."""
        condition = latest_documents(selector)
        nodes = tree_nodes(condition).subquery()
        statement = (
            sa.select(sa.func.count())
            .select_from(nodes)
            .join(
                E,
                sa.and_(
                    E.c.src_label == "StructureNode",
                    E.c.src_key == nodes.c.id,
                    E.c.type == "HAS_MODEL_DECISION",
                ),
            )
        )
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                return (await connection.execute(statement)).scalar_one()

    async def fetch_nodes_to_annotate(
        self, *, tenant_id: str | None, doc_path: str, only_unannotated: bool = False
    ) -> list[dict]:
        """StructureNodes with an InfoUnit of the latest document at *doc_path*.

        Cypher: MATCH (d {tenant, path, latest})-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n) WHERE
        (n)-[:HAS_INFO_UNIT]->() [AND NOT (n)-[:HAS_MODEL_DECISION]->()] RETURN DISTINCT
        id, node_id, title, role, appearance_order, coalesce(theme, 'default'). Cypher has no
        ORDER BY; here: appearance_order, then id.
        """
        nodes = _nodes_with_info_units(
            _latest_doc(tenant_id, doc_path), only_unannotated=only_unannotated
        )
        sub = nodes.subquery()
        d = c.dialect()
        statement = (
            # Each node has one _document_key, so it appears once (the DISTINCT of Cypher).
            sa.select(
                sub.c.id,
                sub.c.node_id,
                sub.c.title,
                sub.c.role,
                sub.c.appearance_order,
                sub.c.theme,
            ).order_by(d.order_value(sub.c.appearance_order), d.order_text(sub.c.id))
        )
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                rows = await connection.execute(statement)
                return [
                    {
                        "full_id": row.id,
                        "node_id": row.node_id,
                        "title": row.title,
                        "role": row.role,
                        "appearance_order": row.appearance_order,
                        "theme": row.theme if row.theme is not None else "default",
                    }
                    for row in rows
                ]

    async def fetch_node_context(
        self,
        full_node_id: str,
        node_id: str,
        title: str | None = None,
        role: str | None = None,
        depth: int = 0,
        max_depth: int = 3,
        visited: set[str] | None = None,
    ) -> dict:
        """Context of a StructureNode (InfoUnits, qualifying children, recursively) on one
        connection, as ``fetch_node_context`` of Neo4j."""
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                return await _node_context(
                    connection, full_node_id, node_id, title, role, depth, max_depth, visited
                )

    async def fetch_document_context_instructions(
        self, *, tenant_id: str | None, doc_path: str
    ) -> str | None:
        """Cypher: MATCH (d {tenant, path, latest: true}) RETURN d.context_instructions (single)."""
        statement = sa.select(D.c.props).where(_latest_doc(tenant_id, doc_path)).limit(1)
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                row = (await connection.execute(statement)).first()
        if row is None:
            return None
        return c.node_from_row("Document", {"props": row.props})[1].get("context_instructions")

    async def write_annotation(
        self, full_node_id: str, decision: AnnotationDecision, document_name: str
    ) -> None:
        """Write one automatic annotation, statement by statement (autocommit), as Neo4j.

        Order: existence guard, stale-decision delete (five statements), ModelDecision +
        HAS_MODEL_DECISION, MATCHED_MODEL (retried), each ComplementaryMatch (retried),
        ProposedModel and its ProposedFields (unmatched), SupplementaryFields (matched).

        Raises:
            RuntimeError: If the StructureNode does not exist.
            GraphError: If the engine fails.
        """
        decision_uid = _make_uid(
            "model_decision",
            full_node_id,
            decision.matched_model_class if decision.matched_model_class is not None else "null",
        )
        timestamp = datetime.now(UTC).isoformat()
        engine = c.async_write_engine()
        async with async_sql_errors():
            async with engine.connect() as connection:
                async with connection.begin():
                    found = await node_provenance(connection, full_node_id)
                if found is None:
                    raise RuntimeError(
                        f"write_annotation: StructureNode not found for full_node_id={full_node_id!r}"
                    )
                prov, document_key = found
                for step in STALE_MODEL_DECISION_STEPS:
                    async with connection.begin():
                        await step(connection, [full_node_id])
                md_values = {
                    "matched_model_class": decision.matched_model_class,
                    "confidence": decision.confidence,
                    "rationale": decision.rationale,
                    "coverage_gaps": decision.coverage_gaps,
                    "propose_new_model": decision.propose_new_model,
                    "proposed_model_description": decision.proposed_model_description,
                    "document_name": document_name,
                    "timestamp": timestamp,
                    **prov_values(prov),
                }
                async with connection.begin():
                    await _create_decision(
                        connection, full_node_id, decision_uid, md_values, prov, document_key
                    )

                if decision.matched_model_class is not None:
                    model = decision.matched_model_class
                    await _retried(
                        connection,
                        lambda: _link_catalog(
                            connection, "ModelDecision", decision_uid, "MATCHED_MODEL", model, prov
                        ),
                    )

                for cm in decision.complementary_models:
                    cm_uid = _make_uid(full_node_id, cm.model_class)
                    values = {
                        "model_class": cm.model_class,
                        "coverage_note": cm.coverage_note,
                        **prov_values(prov),
                    }

                    async def complementary(
                        cm_uid: str = cm_uid, values: dict = values, model: str = cm.model_class
                    ):
                        if not await _node_exists(connection, "ModelDecision", decision_uid):
                            return
                        await _create_child(
                            connection,
                            "ComplementaryMatch",
                            cm_uid,
                            values,
                            ("HAS_COMPLEMENTARY_MATCH", "ModelDecision", decision_uid),
                            prov,
                            document_key,
                        )
                        await _link_catalog(
                            connection, "ComplementaryMatch", cm_uid, "REFERS_TO_MODEL", model, prov
                        )

                    await _retried(connection, complementary)

                if (
                    decision.matched_model_class is None
                    and decision.proposed_schema_name is not None
                ):
                    pm_uid = _make_uid(full_node_id, "proposed_model")
                    values = {
                        "schema_name": decision.proposed_schema_name,
                        "description": decision.proposed_model_description or "",
                        **prov_values(prov),
                    }
                    async with connection.begin():
                        if await _node_exists(connection, "ModelDecision", decision_uid):
                            await _create_child(
                                connection,
                                "ProposedModel",
                                pm_uid,
                                values,
                                ("HAS_PROPOSED_MODEL", "ModelDecision", decision_uid),
                                prov,
                                document_key,
                            )
                    for pf in decision.proposed_schema_fields:
                        f_uid = _make_uid(full_node_id, "proposed_field", pf.field_name)
                        values = {
                            "field_name": pf.field_name,
                            "field_type": pf.field_type,
                            "description": pf.description,
                            "required": pf.required,
                            **prov_values(prov),
                        }
                        async with connection.begin():
                            if await _node_exists(connection, "ProposedModel", pm_uid):
                                await _create_child(
                                    connection,
                                    "ProposedField",
                                    f_uid,
                                    values,
                                    ("HAS_PROPOSED_FIELD", "ProposedModel", pm_uid),
                                    prov,
                                    document_key,
                                )

                if decision.matched_model_class is not None and decision.supplementary_fields:
                    for sf in decision.supplementary_fields:
                        s_uid = _make_uid(full_node_id, "supplementary_field", sf.field_name)
                        values = {
                            "field_name": sf.field_name,
                            "field_type": sf.field_type,
                            "description": sf.description,
                            "required": sf.required,
                            **prov_values(prov),
                        }
                        async with connection.begin():
                            if await _node_exists(connection, "ModelDecision", decision_uid):
                                await _create_child(
                                    connection,
                                    "SupplementaryField",
                                    s_uid,
                                    values,
                                    ("HAS_SUPPLEMENTARY_FIELD", "ModelDecision", decision_uid),
                                    prov,
                                    document_key,
                                )

        log.info(
            "write_annotation: subgraph written for full_node_id=%r "
            "(class=%s, confidence=%s, %d complementary, %d supplementary_fields, "
            "proposed_model=%s, %d proposed_fields)",
            full_node_id,
            decision.matched_model_class,
            decision.confidence,
            len(decision.complementary_models),
            len(decision.supplementary_fields),
            decision.proposed_schema_name,
            len(decision.proposed_schema_fields),
        )

    async def write_manual_annotation(
        self,
        document_name: str,
        matched_model_class: str,
        *,
        tenant_id: str | None,
        doc_path: str,
    ) -> int:
        """Assign a manual ModelDecision to every StructureNode with InfoUnits (autocommit).

        Cypher: the DISTINCT node ids, then per node: stale ExtractionResult delete, the five
        stale-decision deletes, CREATE ModelDecision {source: 'manual', ...} + link, MERGE
        MATCHED_MODEL. Returns the number of nodes (0, with a warning, if none).
        """
        timestamp = datetime.now(UTC).isoformat()
        nodes = _nodes_with_info_units(
            _latest_doc(tenant_id, doc_path), only_unannotated=False
        ).subquery()
        engine = c.async_write_engine()
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                node_ids = c.column_values(
                    await connection.execute(
                        sa.select(nodes.c.id).order_by(c.dialect().order_text(nodes.c.id))
                    )
                )
            if not node_ids:
                log.warning(
                    "write_manual_annotation: no StructureNodes with InfoUnits found "
                    "for document %r (tenant=%r)",
                    doc_path,
                    tenant_id,
                )
                return 0
            async with engine.connect() as connection:
                for full_node_id in node_ids:
                    decision_uid = _make_uid(
                        "manual_model_decision", full_node_id, matched_model_class
                    )
                    async with connection.begin():
                        await delete_stale_extraction_result(connection, full_node_id)
                    for step in STALE_MODEL_DECISION_STEPS:
                        async with connection.begin():
                            await step(connection, [full_node_id])
                    async with connection.begin():
                        found = await node_provenance(connection, full_node_id)
                        if found is None:
                            continue
                        prov, document_key = found
                        md_values = {
                            "matched_model_class": matched_model_class,
                            "confidence": "high",
                            "rationale": "Manually assigned override",
                            "coverage_gaps": [],
                            "propose_new_model": False,
                            "proposed_model_description": None,
                            "document_name": document_name,
                            "timestamp": timestamp,
                            "source": "manual",
                            **prov_values(prov),
                        }
                        await _create_decision(
                            connection, full_node_id, decision_uid, md_values, prov, document_key
                        )
                    async with connection.begin():
                        await _link_catalog(
                            connection,
                            "ModelDecision",
                            decision_uid,
                            "MATCHED_MODEL",
                            matched_model_class,
                            prov,
                        )
        log.info(
            "write_manual_annotation: assigned '%s' to %d nodes in document '%s'",
            matched_model_class,
            len(node_ids),
            document_name,
        )
        return len(node_ids)


# ── Statement helpers of the annotation writers ──────────────────────────────


async def _retried(connection: AsyncConnection, work: Any) -> None:
    """One autocommit statement retried on transient errors (``with_neo4j_retry`` of Cypher)."""

    async def attempt() -> None:
        async with connection.begin():
            await work()

    await with_sql_retry(attempt)


async def _node_exists(connection: AsyncConnection, label: str, key: str) -> bool:
    column = c.key_column(label)
    return (await connection.execute(sa.select(column).where(column == key))).first() is not None


def _create(label: str, key: str, values: dict, document_key: str | None) -> Any:
    """``CREATE (n:<label> {uid, ...})``: insert, or replace a node left with the same uid
    (the tables have a primary key where Neo4j would create a duplicate node)."""
    return c.merge_node(
        label,
        {"uid": key},
        on_create=values,
        on_match=values,
        aux={"_document_key": document_key},
        aux_on_match={"_document_key": document_key},
    )


async def _create_decision(
    connection: AsyncConnection,
    node_id: str,
    decision_uid: str,
    values: dict,
    prov: RawProvenance,
    document_key: str | None,
) -> None:
    """MATCH (n {id}) CREATE (md:ModelDecision {...}) CREATE (n)-[:HAS_MODEL_DECISION]->(md)."""
    await connection.execute(_create("ModelDecision", decision_uid, values, document_key))
    await connection.execute(
        c.merge_edge(
            "HAS_MODEL_DECISION",
            "StructureNode",
            node_id,
            "ModelDecision",
            decision_uid,
            tenant=prov.tenant_id,
        )
    )


async def _create_child(
    connection: AsyncConnection,
    label: str,
    uid: str,
    values: dict,
    parent: tuple[str, str, str],
    prov: RawProvenance,
    document_key: str | None,
) -> None:
    """CREATE (x:<label> {...}) CREATE (parent)-[:<rel>]->(x)."""
    rel_type, parent_label, parent_key = parent
    await connection.execute(_create(label, uid, values, document_key))
    await connection.execute(
        c.merge_edge(rel_type, parent_label, parent_key, label, uid, tenant=prov.tenant_id)
    )


async def _link_catalog(
    connection: AsyncConnection,
    label: str,
    key: str,
    rel_type: str,
    model_name: str,
    prov: RawProvenance,
) -> None:
    """MATCH (x {uid}) MERGE (cat:CatalogModel {name}) MERGE (x)-[:<rel_type>]->(cat)."""
    if not await _node_exists(connection, label, key):
        return
    await connection.execute(c.merge_node("CatalogModel", {"name": model_name}))
    await connection.execute(
        c.merge_edge(rel_type, label, key, "CatalogModel", model_name, tenant=prov.tenant_id)
    )
