"""
graph/neo4j/annotation.py — Catalog, themes and annotation over Neo4j.

:class:`Neo4jAnnotationOps` (``store.annotation``) holds the Cypher that used to
live in the old annotation Cypher module, moved whole. The domain side (model
introspection, relationship-type validation, the ``*_once`` memoization) is in
:mod:`scinr.newton.annotation.catalog`.

Every method here opens its own session on the shared async driver. They write
in **autocommit** (each statement commits on its own) and keep the retries that
wrapped the same statements before the move. Semaphores are the callers' business.
"""

from __future__ import annotations

import hashlib
import logging
import re as _re
from collections.abc import AsyncIterator, Collection
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from scinr.newton.config import get_config
from scinr.newton.graph.base import AnnotationOps
from scinr.newton.graph.neo4j import driver as _driver
from scinr.newton.graph.neo4j._shared import (
    delete_stale_extraction_result,
    delete_stale_model_decision,
)
from scinr.newton.graph.neo4j.documents import latest_document_pattern
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.retry import with_neo4j_retry
from scinr.newton.graph.types import DocumentSelector
from scinr.newton.utils.tenancy import tenant_key

if TYPE_CHECKING:
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.graph.sinks import CatalogSink

log = logging.getLogger(__name__)


def _make_uid(*parts: str) -> str:
    """16-char SHA-256 hex digest — deterministic UID from one or more string parts."""
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def _fetch_info_units_for_node(session, full_node_id: str) -> list[dict]:
    """
    Fetch all InfoUnits directly attached to a StructureNode.

    Parameters
    ----------
    session:
        An already-open Neo4j async session shared for the whole subtree traversal.
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.

    Returns
    -------
    list[dict]
        Each dict has: info_unit_id, title, description, order.
    """
    query = """
    MATCH (n:StructureNode {id: $node_id})-[:HAS_INFO_UNIT]->(iu:InfoUnit)
    RETURN iu.uid         AS info_unit_id,
           iu.title       AS title,
           iu.description AS description,
           iu.order       AS order
    ORDER BY iu.uid
    """
    result = await session.run(query, node_id=full_node_id)
    return await result.data()


async def _fetch_qualifying_children(session, full_node_id: str) -> list[dict]:
    """
    Fetch direct StructureNode children whose role is freeform_block, table, or field_group.

    Parameters
    ----------
    session:
        An already-open Neo4j async session shared for the whole subtree traversal.
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.

    Returns
    -------
    list[dict]
        Each dict has: full_id, node_id, title, role, appearance_order.
    """
    query = """
    MATCH (n:StructureNode {id: $node_id})-[:HAS_CHILD]->(c:StructureNode)
    WHERE c.role IN ['freeform_block', 'table', 'field_group']
    RETURN c.id               AS full_id,
           c.node_id          AS node_id,
           c.title            AS title,
           c.role             AS role,
           c.appearance_order AS appearance_order
    ORDER BY c.appearance_order
    """
    result = await session.run(query, node_id=full_node_id)
    return await result.data()


async def _fetch_node_context_with_session(
    session,
    full_node_id: str,
    node_id: str,
    title: str | None = None,
    role: str | None = None,
    depth: int = 0,
    max_depth: int = 3,
    visited: set[str] | None = None,
) -> dict:
    """
    Recursively build a context dict for a StructureNode using a shared session.

    This private function contains the full recursive logic and is called by
    the public ``fetch_node_context`` wrapper which owns the session lifecycle.
    Reusing a single session across the entire subtree avoids opening O(b^depth)
    concurrent connections and eliminates any risk of semaphore deadlocks.

    Parameters
    ----------
    session:
        An already-open Neo4j async session. Shared across all recursive calls
        for a single subtree traversal — never opened or closed here.
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.
    node_id:
        Short node_id (already known from parent query or node_data).
        Avoids a redundant round-trip to Neo4j.
    title:
        Node title, if already known from the parent query.
    role:
        Node role, if already known from the parent query.
    depth:
        Current recursion depth (0 = root node).
    max_depth:
        Maximum recursion depth (default 3).
    visited:
        Set of full_node_ids already visited in this traversal (cycle guard).

    Returns
    -------
    dict
        NodeContext-shaped dict with keys:
        node_id, full_id, title, role, info_units, children,
        depth, depth_limit_reached.

    .. warning::
        The ``session`` parameter is shared across all recursive calls.
        Do **not** call this function with ``asyncio.gather`` over multiple
        children — that would use the session concurrently across coroutines,
        which is not safe. The children loop must remain sequential (``for``).
    """
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

    info_units = await _fetch_info_units_for_node(session, full_node_id)
    children_meta = await _fetch_qualifying_children(session, full_node_id)

    children = []
    # NOTE: intentionally sequential — do NOT convert to asyncio.gather.
    # The session is shared across all recursive calls; concurrent use would
    # violate neo4j AsyncSession's single-transaction-at-a-time contract.
    for child in children_meta:
        child_ctx = await _fetch_node_context_with_session(
            session=session,
            full_node_id=child["full_id"],
            node_id=child["node_id"],
            title=child.get("title"),
            role=child.get("role"),
            depth=depth + 1,
            max_depth=max_depth,
            visited=visited,
        )
        children.append(child_ctx)

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


_IDENT_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def instance_key_index_name(prop: str) -> str:
    """Name of the ``(tenant_id, <prop>)`` index on :ModelInstance for key *prop*.

    Identifier-like names map to ``idx_mi_key_<prop>``. Anything else is
    sanitised to ``[A-Za-z0-9_]`` and suffixed with a short hash of the
    original, so two names that sanitise alike never share an index.
    """
    if _IDENT_RE.match(prop):
        return f"idx_mi_key_{prop}"
    safe = _re.sub(r"[^A-Za-z0-9_]", "_", prop)
    digest = hashlib.sha1(prop.encode("utf-8")).hexdigest()[:8]
    return f"idx_mi_key_{safe}_{digest}"


async def _ensure_instance_key_indexes(session, props: Collection[str]) -> None:
    """Create a ``(tenant_id, <prop>)`` index on :ModelInstance per key property.

    ``instance_key`` fields are written (normalised) as properties of the
    ModelInstance node, so a search on part of a composite key — or on a key
    without rebuilding the uid — is an index seek within the tenant. The
    label is always :ModelInstance, so one index serves every model that has
    a key field with that name. Indexes of fields that stop being keys are
    left in place (harmless; drop them by hand with ``DROP INDEX``).

    A failure (e.g. no schema privilege) is logged and does not stop the
    catalog setup.
    """
    for prop in sorted(props):
        name = instance_key_index_name(prop)
        escaped = prop.replace("`", "``")
        cypher = (
            f"CREATE INDEX {name} IF NOT EXISTS "
            f"FOR (mi:ModelInstance) ON (mi.tenant_id, mi.`{escaped}`)"
        )
        try:
            result = await session.run(cypher)
            await result.consume()
        except Exception as exc:  # noqa: BLE001 — a missing index must not block ingestion
            log.warning("Could not create index %s on ModelInstance.%s: %s", name, prop, exc)


class Neo4jCatalogSink:
    """:class:`~scinr.newton.graph.sinks.CatalogSink` over one async Neo4j session.

    Each method is one autocommit statement, with the Cypher the catalog and
    theme setup always ran. Built by :meth:`Neo4jAnnotationOps.catalog_sink`.
    """

    def __init__(self, session) -> None:
        self._session = session

    @neo4j_errors()
    async def merge_catalog_model(self, name: str, description: str, selectable: bool) -> None:
        await self._session.run(
            """
            MERGE (cm:CatalogModel {name: $name})
            SET cm.description = $description,
                cm.selectable  = $selectable
            """,
            name=name,
            description=description,
            selectable=selectable,
        )

    @neo4j_errors()
    async def merge_model_field(
        self, model_name: str, field_name: str, field_type: str, description: str, required: bool
    ) -> None:
        await self._session.run(
            """
            MERGE (mf:ModelField {name: $field_name, model: $model_name})
            ON CREATE SET mf.type = $field_type
            WITH mf
            MATCH (cm:CatalogModel {name: $model_name})
            MERGE (cm)-[r:HAS_FIELD]->(mf)
            SET r.description = $description,
                r.required    = $required
            """,
            field_name=field_name,
            field_type=field_type,
            model_name=model_name,
            description=description,
            required=required,
        )

    @neo4j_errors()
    async def merge_aggregates(
        self, src_name: str, tgt_name: str, field_name: str, is_list: bool, required: bool
    ) -> None:
        await self._session.run(
            """
            MATCH (src:CatalogModel {name: $src_name})
            MATCH (tgt:CatalogModel {name: $tgt_name})
            MERGE (src)-[r:AGGREGATES {field_name: $field_name}]->(tgt)
            SET r.is_list  = $is_list,
                r.required = $required
            """,
            src_name=src_name,
            tgt_name=tgt_name,
            field_name=field_name,
            is_list=is_list,
            required=required,
        )

    @neo4j_errors()
    async def set_field_metadata(
        self, model_name: str, field_name: str, is_instance_key: bool, entity_label: str | None
    ) -> None:
        await self._session.run(
            """
            MERGE (mf:ModelField {name: $field_name, model: $model_name})
            SET mf.is_instance_key = $is_instance_key,
                mf.entity_label    = $entity_label
            """,
            field_name=field_name,
            model_name=model_name,
            is_instance_key=is_instance_key,
            entity_label=entity_label,
        )

    async def ensure_instance_key_indexes(self, props: Collection[str]) -> None:
        # Best effort by design: a failed index is logged, never raised, so the
        # driver exception is not translated here.
        await _ensure_instance_key_indexes(self._session, props)

    @neo4j_errors()
    async def link_produces_entity(self, model_name: str, field_name: str, entity_label: str) -> None:
        await self._session.run(
            """
            MERGE (el:EntityLabel {label: $entity_label})
            WITH el
            MATCH (cm:CatalogModel {name: $model_name})
            MERGE (cm)-[:PRODUCES_ENTITY {field_name: $field_name}]->(el)
            """,
            entity_label=entity_label,
            model_name=model_name,
            field_name=field_name,
        )

    @neo4j_errors()
    async def merge_field_relationship(
        self,
        rel_type: str,
        source_label: str,
        target_label: str,
        via_model: str,
        from_field: str,
        to_field: str,
    ) -> None:
        # Neo4j cannot parameterise a relationship type: the domain has already
        # validated rel_type against [A-Z][A-Z0-9_]*.
        cypher = f"""
        MERGE (el_src:EntityLabel {{label: $source_label}})
        MERGE (el_tgt:EntityLabel {{label: $target_label}})
        MERGE (el_src)-[r:`{rel_type}` {{via_model: $via_model, from_field: $from_field, to_field: $to_field}}]->(el_tgt)
        """
        await self._session.run(
            cypher,
            source_label=source_label,
            target_label=target_label,
            via_model=via_model,
            from_field=from_field,
            to_field=to_field,
        )

    @neo4j_errors()
    async def merge_instance_relationship(
        self,
        rel_type: str,
        src_name: str,
        tgt_name: str,
        via_field: str,
        join_via_json: str,
        is_composite: bool,
    ) -> None:
        # Same as merge_field_relationship: rel_type is validated by the domain.
        cypher = f"""
        MERGE (src:CatalogModel {{name: $src_name}})
        MERGE (tgt:CatalogModel {{name: $tgt_name}})
        MERGE (src)-[r:`{rel_type}` {{via_field: $via_field, join_via: $join_via_json}}]->(tgt)
        SET r.composite = $is_composite
        """
        await self._session.run(
            cypher,
            src_name=src_name,
            tgt_name=tgt_name,
            via_field=via_field,
            join_via_json=join_via_json,
            is_composite=is_composite,
        )

    @neo4j_errors()
    async def merge_theme(self, path: str, name: str) -> None:
        await self._session.run(
            "MERGE (t:Theme {path: $path}) SET t.name = $name",
            path=path,
            name=name,
        )

    @neo4j_errors()
    async def link_subtopic(self, parent_path: str, child_path: str) -> None:
        await self._session.run(
            """
            MATCH (parent:Theme {path: $parent_path})
            MATCH (child:Theme {path: $child_path})
            MERGE (parent)-[:HAS_SUBTOPIC]->(child)
            """,
            parent_path=parent_path,
            child_path=child_path,
        )

    @neo4j_errors()
    async def link_model_to_theme(self, model_name: str, path: str) -> None:
        await self._session.run(
            """
            MERGE (cm:CatalogModel {name: $model_name})
            MERGE (t:Theme {path: $path})
            MERGE (cm)-[:BELONGS_TO_THEME]->(t)
            """,
            model_name=model_name,
            path=path,
        )


@asynccontextmanager
async def _open_catalog_sink() -> AsyncIterator[Neo4jCatalogSink]:
    """One session on the shared async driver, closed when the block ends.

    Driver errors raised while acquiring or closing the session are translated.
    """
    cfg = get_config()
    with neo4j_errors():
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            yield Neo4jCatalogSink(session)


class Neo4jAnnotationOps(AnnotationOps):
    """Annotation operations over Neo4j (asynchronous, shared async driver)."""

    def catalog_sink(self) -> AbstractAsyncContextManager[CatalogSink]:
        """Open the catalog sink: one session on the shared async driver, autocommit statements."""
        return _open_catalog_sink()

    @neo4j_errors()
    async def count_annotated_nodes(self, selector: DocumentSelector) -> int:
        """How many StructureNodes of the ``latest`` documents *selector* selects carry a model decision.

        Raises:
            ValueError: If the selector is invalid (see
                :func:`~scinr.newton.graph.neo4j.documents.latest_document_pattern`).
            GraphError: If the engine fails.
        """
        pattern, params = latest_document_pattern("d", selector)
        driver = _driver.get_async_driver()
        cfg = get_config()
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(
                f"MATCH {pattern}"
                "-[:HAS_STRUCTURE|HAS_CHILD*1..]->(sn:StructureNode)"
                "-[:HAS_MODEL_DECISION]->() RETURN count(sn) AS n",
                **params,
            )
            row = await result.single()
        return row["n"]

    @neo4j_errors()
    async def fetch_nodes_to_annotate(
        self,
        *,
        tenant_id: str | None,
        doc_path: str,
        only_unannotated: bool = False,
    ) -> list[dict]:
        """
        Fetch all StructureNodes that have at least one InfoUnit for this document.

        The document is the ``latest`` :Document of *tenant_id* (``None`` = public)
        at *doc_path* — never selected by name, which is neither unique within a
        tenant nor across tenants.

        Traverses both direct HAS_STRUCTURE children and all HAS_CHILD descendants.
        Returns nodes ordered by appearance_order.

        When *only_unannotated* is True, nodes that already have a
        :HAS_MODEL_DECISION relationship are excluded.

        Returns
        -------
        list[dict]
            Each dict has keys: full_id, node_id, title, role, appearance_order, theme
        """
        extra_filter = ""
        if only_unannotated:
            extra_filter = "AND NOT (n)-[:HAS_MODEL_DECISION]->()"

        query = f"""
        MATCH (d:Document {{tenant_id: $tenant_id, path: $doc_path, latest: true}})
              -[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
        WHERE (n)-[:HAS_INFO_UNIT]->()
        {extra_filter}
        RETURN DISTINCT
               n.id               AS full_id,
               n.node_id          AS node_id,
               n.title            AS title,
               n.role             AS role,
               n.appearance_order AS appearance_order,
               coalesce(n.theme, 'default') AS theme
        """
        cfg = get_config()
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(query, tenant_id=tenant_key(tenant_id), doc_path=doc_path)
            return await result.data()

    @neo4j_errors()
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
        """
        Recursively build a context dict for a StructureNode.

        Opens exactly **one** Neo4j session for the entire subtree traversal and
        delegates all recursive work to ``_fetch_node_context_with_session``.
        This ensures that the number of concurrent connections is bounded by the
        number of root-level calls to this function, regardless of tree depth or
        branching factor.

        For each node:
        - Fetch its InfoUnits
        - Fetch direct qualifying children (freeform_block / table / field_group)
        - Recursively apply the same logic to qualifying children up to max_depth

        Guards:
        - max_depth=3 prevents excessive query fan-out
        - visited set prevents infinite loops on graph cycles

        Parameters
        ----------
        full_node_id:
            Composite StructureNode.id as stored in Neo4j.
        node_id:
            Short node_id already known from the caller (from fetch_nodes_to_annotate
            or from the parent _fetch_qualifying_children result). Passed through to
            avoid a redundant Neo4j round-trip.
        title:
            Node title, if already known.
        role:
            Node role, if already known.
        depth:
            Starting recursion depth (always 0 for external callers).
        max_depth:
            Maximum recursion depth (default 3).
        visited:
            Cycle-guard set; should be None for external callers.

        Returns
        -------
        dict
            NodeContext-shaped dict with keys:
            node_id, full_id, title, role, info_units, children,
            depth, depth_limit_reached.
        """
        cfg = get_config()
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            return await _fetch_node_context_with_session(
                session=session,
                full_node_id=full_node_id,
                node_id=node_id,
                title=title,
                role=role,
                depth=depth,
                max_depth=max_depth,
                visited=visited,
            )

    @neo4j_errors()
    async def fetch_document_context_instructions(
        self,
        *,
        tenant_id: str | None,
        doc_path: str,
    ) -> str | None:
        """Fetches the context_instructions property from *tenant_id*'s (``None`` =
        public) latest :Document at *doc_path*."""
        query = """
            MATCH (d:Document {tenant_id: $tenant_id, path: $doc_path, latest: true})
            RETURN d.context_instructions AS context_instructions
        """
        cfg = get_config()
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(query, tenant_id=tenant_key(tenant_id), doc_path=doc_path)
            record = await result.single()
            if record is None:
                return None
            return record["context_instructions"]

    @neo4j_errors()
    async def write_annotation(
        self,
        full_node_id: str,
        decision: AnnotationDecision,
        document_name: str,
    ) -> None:
        """
        Write an AnnotationDecision subgraph to Neo4j for a StructureNode.

        Creates or replaces:
          (:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)
          (:ModelDecision)-[:MATCHED_MODEL]->(:CatalogModel)                    [when matched_model_class is set]
          (:ModelDecision)-[:HAS_COMPLEMENTARY_MATCH]->(:ComplementaryMatch)
          (:ComplementaryMatch)-[:REFERS_TO_MODEL]->(:CatalogModel)
          (:ModelDecision)-[:HAS_SUPPLEMENTARY_FIELD]->(:SupplementaryField)    [when matched and gaps exist]
          (:ModelDecision)-[:HAS_PROPOSED_MODEL]->(:ProposedModel)              [when no match]
          (:ProposedModel)-[:HAS_PROPOSED_FIELD]->(:ProposedField)

        Idempotency: existing :ProposedField, :ProposedModel, :SupplementaryField,
        :ComplementaryMatch, and :ModelDecision nodes for this StructureNode are
        deleted before recreation. :CatalogModel singletons are never deleted.

        Raises
        ------
        RuntimeError
            If the StructureNode does not exist in the database.
        """
        decision_uid = _make_uid(
            "model_decision",
            full_node_id,
            decision.matched_model_class
            if decision.matched_model_class is not None
            else "null",
        )
        timestamp = datetime.now(UTC).isoformat()
        cfg = get_config()
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            # ── Guard: verify StructureNode exists ────────────────────────────
            result = await session.run(
                "MATCH (n:StructureNode {id: $node_id}) RETURN count(n) AS cnt",
                node_id=full_node_id,
            )
            record = await result.single()
            if record["cnt"] == 0:
                raise RuntimeError(
                    f"write_annotation: StructureNode not found for full_node_id={full_node_id!r}"
                )

            # ── Idempotency: delete the stale ModelDecision subtree ───────────
            await delete_stale_model_decision(session, full_node_id)

            # ── Create ModelDecision node + HAS_MODEL_DECISION relationship ───
            await session.run(
                """
                MATCH (n:StructureNode {id: $node_id})
                CREATE (md:ModelDecision {
                    uid:                       $uid,
                    matched_model_class:       $matched_model_class,
                    confidence:                $confidence,
                    rationale:                 $rationale,
                    coverage_gaps:             $coverage_gaps,
                    propose_new_model:         $propose_new_model,
                    proposed_model_description: $proposed_model_description,
                    document_name:             $document_name,
                    timestamp:                 $timestamp,
                    tenant_id:                 n.tenant_id,
                    created_by_user_id:        n.created_by_user_id,
                    job_id:                    n.job_id
                })
                CREATE (n)-[:HAS_MODEL_DECISION]->(md)
                """,
                node_id=full_node_id,
                uid=decision_uid,
                matched_model_class=decision.matched_model_class,
                confidence=decision.confidence,
                rationale=decision.rationale,
                coverage_gaps=decision.coverage_gaps,
                propose_new_model=decision.propose_new_model,
                proposed_model_description=decision.proposed_model_description,
                document_name=document_name,
                timestamp=timestamp,
            )

            # ── MATCHED_MODEL → CatalogModel (only when a model was matched) ──
            if decision.matched_model_class is not None:
                _params = dict(
                    decision_uid=decision_uid,
                    model_name=decision.matched_model_class,
                )
                await with_neo4j_retry(lambda: session.run(
                    """
                    MATCH (md:ModelDecision {uid: $decision_uid})
                    MERGE (cat:CatalogModel {name: $model_name})
                    MERGE (md)-[:MATCHED_MODEL]->(cat)
                    """,
                    **_params,
                ))

            # ── ComplementaryMatch nodes ───────────────────────────────────────
            for cm in decision.complementary_models:
                cm_uid = _make_uid(full_node_id, cm.model_class)
                _cm_params = dict(
                    decision_uid=decision_uid,
                    cm_uid=cm_uid,
                    model_class=cm.model_class,
                    coverage_note=cm.coverage_note,
                )
                await with_neo4j_retry(lambda _cm_params=_cm_params: session.run(
                    """
                    MATCH (md:ModelDecision {uid: $decision_uid})
                    CREATE (comp:ComplementaryMatch {
                        uid:                $cm_uid,
                        model_class:        $model_class,
                        coverage_note:      $coverage_note,
                        tenant_id:          md.tenant_id,
                        created_by_user_id: md.created_by_user_id,
                        job_id:             md.job_id
                    })
                    CREATE (md)-[:HAS_COMPLEMENTARY_MATCH]->(comp)
                    WITH comp
                    MERGE (cat:CatalogModel {name: $model_class})
                    MERGE (comp)-[:REFERS_TO_MODEL]->(cat)
                    """,
                    **_cm_params,
                ))

            # ── ProposedModel + ProposedField nodes (when no model matched) ───
            if (
                decision.matched_model_class is None
                and decision.proposed_schema_name is not None
            ):
                pm_uid = _make_uid(full_node_id, "proposed_model")
                await session.run(
                    """
                    MATCH (md:ModelDecision {uid: $decision_uid})
                    CREATE (pm:ProposedModel {
                        uid:                $pm_uid,
                        schema_name:        $schema_name,
                        description:        $description,
                        tenant_id:          md.tenant_id,
                        created_by_user_id: md.created_by_user_id,
                        job_id:             md.job_id
                    })
                    CREATE (md)-[:HAS_PROPOSED_MODEL]->(pm)
                    """,
                    decision_uid=decision_uid,
                    pm_uid=pm_uid,
                    schema_name=decision.proposed_schema_name,
                    description=decision.proposed_model_description or "",
                )
                for pf in decision.proposed_schema_fields:
                    f_uid = _make_uid(full_node_id, "proposed_field", pf.field_name)
                    await session.run(
                        """
                        MATCH (pm:ProposedModel {uid: $pm_uid})
                        CREATE (f:ProposedField {
                            uid:                $f_uid,
                            field_name:         $field_name,
                            field_type:         $field_type,
                            description:        $description,
                            required:           $required,
                            tenant_id:          pm.tenant_id,
                            created_by_user_id: pm.created_by_user_id,
                            job_id:             pm.job_id
                        })
                        CREATE (pm)-[:HAS_PROPOSED_FIELD]->(f)
                        """,
                        pm_uid=pm_uid,
                        f_uid=f_uid,
                        field_name=pf.field_name,
                        field_type=pf.field_type,
                        description=pf.description,
                        required=pf.required,
                    )

            # ── SupplementaryField nodes (when a model matched and gaps exist) ─
            if (
                decision.matched_model_class is not None
                and decision.supplementary_fields
            ):
                for sf in decision.supplementary_fields:
                    s_uid = _make_uid(full_node_id, "supplementary_field", sf.field_name)
                    await session.run(
                        """
                        MATCH (md:ModelDecision {uid: $decision_uid})
                        CREATE (s:SupplementaryField {
                            uid:                $s_uid,
                            field_name:         $field_name,
                            field_type:         $field_type,
                            description:        $description,
                            required:           $required,
                            tenant_id:          md.tenant_id,
                            created_by_user_id: md.created_by_user_id,
                            job_id:             md.job_id
                        })
                        CREATE (md)-[:HAS_SUPPLEMENTARY_FIELD]->(s)
                        """,
                        decision_uid=decision_uid,
                        s_uid=s_uid,
                        field_name=sf.field_name,
                        field_type=sf.field_type,
                        description=sf.description,
                        required=sf.required,
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

    @neo4j_errors()
    async def write_manual_annotation(
        self,
        document_name: str,
        matched_model_class: str,
        *,
        tenant_id: str | None,
        doc_path: str,
    ) -> int:
        """
        Assign a manual ModelDecision with the given model class to all StructureNodes
        with InfoUnits for the specified document.

        Bypasses the LLM annotation agent and writes a simplified ModelDecision with
        source='manual'. Any existing ModelDecision and ExtractionResult subgraphs
        for each qualifying node are replaced before writing the new decision.

        The subgraph written per node is minimal:
          (:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision {source: 'manual', ...})
          (:ModelDecision)-[:MATCHED_MODEL]->(:CatalogModel)

        Parameters
        ----------
        document_name:
            Document display name, recorded on each ModelDecision (provenance
            only — not used to select the document).
        matched_model_class:
            CamelCase Pydantic model class name to assign to every qualifying node.
        tenant_id, doc_path:
            Select the ``latest`` :Document to annotate: *tenant_id*'s (``None`` =
            public) document at *doc_path*.

        Returns
        -------
        int
            Number of StructureNodes updated. Returns 0 if no qualifying nodes are found.
        """
        timestamp = datetime.now(UTC).isoformat()
        cfg = get_config()
        driver = _driver.get_async_driver()
        # ── 1. Fetch all StructureNodes with at least one InfoUnit ────────────────
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(
                """
                MATCH (d:Document {tenant_id: $tenant_id, path: $doc_path, latest: true})
                      -[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
                WHERE (n)-[:HAS_INFO_UNIT]->()
                RETURN DISTINCT n.id AS full_node_id
                """,
                tenant_id=tenant_key(tenant_id),
                doc_path=doc_path,
            )
            node_ids = [r["full_node_id"] for r in await result.data()]

        if not node_ids:
            log.warning(
                "write_manual_annotation: no StructureNodes with InfoUnits found "
                "for document %r (tenant=%r)",
                doc_path,
                tenant_id,
            )
            return 0
        cfg = get_config()
        # ── 2. For each node: clear stale subgraphs, write new ModelDecision ─────
        async with driver.session(database=cfg.neo4j_database) as session:
            for full_node_id in node_ids:
                decision_uid = _make_uid(
                    "manual_model_decision",
                    full_node_id,
                    matched_model_class,
                )

                # Delete stale ExtractionResult and its exclusive ModelInstance children
                await delete_stale_extraction_result(session, full_node_id)

                # Delete stale ModelDecision subgraph — leaf nodes first to avoid
                # dangling relationships (same helper as write_annotation)
                await delete_stale_model_decision(session, full_node_id)

                # Create new minimal ModelDecision with source='manual'
                await session.run(
                    """
                    MATCH (n:StructureNode {id: $node_id})
                    CREATE (md:ModelDecision {
                        uid:                        $uid,
                        matched_model_class:        $matched_model_class,
                        confidence:                 'high',
                        rationale:                  'Manually assigned override',
                        coverage_gaps:              [],
                        propose_new_model:          false,
                        proposed_model_description: null,
                        document_name:              $document_name,
                        timestamp:                  $timestamp,
                        source:                     'manual',
                        tenant_id:                  n.tenant_id,
                        created_by_user_id:         n.created_by_user_id,
                        job_id:                     n.job_id
                    })
                    CREATE (n)-[:HAS_MODEL_DECISION]->(md)
                    """,
                    node_id=full_node_id,
                    uid=decision_uid,
                    matched_model_class=matched_model_class,
                    document_name=document_name,
                    timestamp=timestamp,
                )

                # Link ModelDecision to CatalogModel singleton
                await session.run(
                    """
                    MATCH (md:ModelDecision {uid: $decision_uid})
                    MERGE (cm:CatalogModel {name: $model_name})
                    MERGE (md)-[:MATCHED_MODEL]->(cm)
                    """,
                    decision_uid=decision_uid,
                    model_name=matched_model_class,
                )

        log.info(
            "write_manual_annotation: assigned '%s' to %d nodes in document '%s'",
            matched_model_class,
            len(node_ids),
            document_name,
        )
        return len(node_ids)
