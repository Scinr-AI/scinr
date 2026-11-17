"""
graph/neo4j/tabular.py — Tabular subgraphs over Neo4j.

:class:`Neo4jTabularOps` (``store.tabular``) opens the tabular sink and deletes
the subgraph of a sheet. :class:`Neo4jTabularSink` is the Cypher side of the
sheet writer: the traversal (ids, batches, normalization passes, per-row
extraction) lives in :mod:`scinr.newton.tabular.graph_writer` and calls the sink
once per statement, in the order the statements were always emitted. The
Cypher-free row logic (routing, normalization keys, composite instantiation) is
in :mod:`scinr.newton.tabular.rows`.

The sink holds no session: every method opens its own on the shared async
driver, as the writers did before the split. Semaphores are the callers'
business; the row writes keep the local ``_PARALLEL_ROW_WRITES`` bound.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

from neo4j import AsyncDriver

from scinr.newton.config import get_config
from scinr.newton.graph.base import TabularOps
from scinr.newton.graph.neo4j import driver as _driver
from scinr.newton.graph.neo4j._shared import (
    delete_stale_model_decision,
    provenance_set_clause,
)
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.retry import with_neo4j_retry
from scinr.newton.graph.types import Provenance, RowData, TableNode
from scinr.newton.ingest.nodes import make_structure_node_id
from scinr.newton.utils.tenancy import require_stored_tenant

if TYPE_CHECKING:
    from scinr.newton.graph.sinks import TabularSink

logger = logging.getLogger(__name__)


class Neo4jTabularSink:
    """:class:`~scinr.newton.graph.sinks.TabularSink` over the shared async driver.

    Each method is the statement (or the group of statements that always ran
    together) of the old whole-sheet writer, with the same session and
    transaction style. Built by :meth:`Neo4jTabularOps.sink`.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    @neo4j_errors()
    async def create_table(self, table: TableNode) -> None:
        tenant_id, doc_path, version = table.doc_key
        cfg = get_config()
        # MATCH, not MERGE: the Document is created by tabular/agent.py beforehand, and a
        # MERGE here would silently create an empty one if that step had failed.
        async with self._driver.session(database=cfg.neo4j_database) as session:
            tx = await session.begin_transaction()
            try:
                result = await tx.run(
                    """
                    MATCH (d:Document {tenant_id: $tenant_id, path: $doc_path, version: $version})
                    MERGE (n:StructureNode {id: $id})
                    SET n:Table,
                        n.node_id = $node_id,
                        n.role = 'table',
                        n.title = $title,
                        n.appearance_order = $order,
                        n.theme = $theme,
                        n.column_count = $col_count,
                        n.row_count = $row_count,
                        n.source_page_ids = CASE WHEN $page_id <> '' THEN [$page_id] ELSE [] END,
                        n.tenant_id = d.tenant_id,
                        n.created_by_user_id = d.created_by_user_id,
                        n.job_id = d.job_id
                    MERGE (d)-[:HAS_STRUCTURE]->(n)
                    RETURN count(n) AS created
                    """,
                    id=table.id,
                    node_id=table.node_id,
                    title=table.title,
                    order=table.order,
                    theme=table.theme,
                    col_count=table.col_count,
                    row_count=table.row_count,
                    tenant_id=tenant_id,
                    doc_path=doc_path,
                    version=version,
                    page_id=table.page_id,
                )
                record = await result.single()
                if not record or not record["created"]:
                    raise RuntimeError(
                        f"write_tabular_subgraph: Document not found "
                        f"(tenant={tenant_id!r}, path={doc_path!r}, version={version})"
                    )
                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

    @neo4j_errors()
    async def write_row_batch(
        self, table_id: str, rows: list[RowData], md_uid: str, theme: str, page_id: str
    ) -> None:
        rows_data = [
            {
                "id": row.id,
                "node_id": row.node_id,
                "row_index": row.row_index,
                "appearance_order": row.appearance_order,
                "info_uid": row.info_uid,
                "description": row.description,
            }
            for row in rows
        ]
        # One transaction: create Row nodes + InfoUnits + HAS_MODEL_DECISION links (UNWIND)
        cfg = get_config()
        async with self._driver.session(database=cfg.neo4j_database) as session:
            tx = await session.begin_transaction()
            try:
                await tx.run(
                    """
                    UNWIND $rows AS row_data
                    MATCH (t:StructureNode {id: $table_id})
                    MERGE (r:StructureNode {id: row_data.id})
                    SET r:Row,
                        r.node_id = row_data.node_id,
                        r.role = 'row',
                        r.appearance_order = row_data.appearance_order,
                        r.theme = $theme,
                        r.row_index = row_data.row_index,
                        r.source_page_ids = CASE WHEN $page_id <> '' THEN [$page_id] ELSE [] END,
                        r.tenant_id = t.tenant_id,
                        r.created_by_user_id = t.created_by_user_id,
                        r.job_id = t.job_id
                    MERGE (t)-[:HAS_CHILD]->(r)
                    WITH r, row_data
                    MERGE (u:InfoUnit {uid: row_data.info_uid})
                    SET u.title = 'Row data',
                        u.description = row_data.description,
                        u.order = 0,
                        u.tenant_id = r.tenant_id,
                        u.created_by_user_id = r.created_by_user_id,
                        u.job_id = r.job_id
                    MERGE (r)-[:HAS_INFO_UNIT]->(u)
                    WITH r
                    MATCH (md:ModelDecision {uid: $md_uid})
                    MERGE (r)-[:HAS_MODEL_DECISION]->(md)
                    """,
                    rows=rows_data,
                    table_id=table_id,
                    md_uid=md_uid,
                    theme=theme,
                    page_id=page_id,
                )
                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

    @neo4j_errors()
    async def replace_raw_extraction_result(
        self, row_id: str, extraction_uid: str, document_name: str, timestamp: str
    ) -> Provenance | None:
        cfg = get_config()
        async with self._driver.session(database=cfg.neo4j_database) as session:
            # Idempotency: delete stale ExtractionResult only.
            # :Entity nodes are deduplicated within a tenant — never deleted here.
            # Also read back the Row's provenance (stamped at Table/Row creation
            # time from the owning :Document) — see graph_mapper.py for why this
            # is read back here rather than threaded through the call stack.
            result = await session.run(
                """
                MATCH (n:StructureNode {id: $nid})
                OPTIONAL MATCH (n)-[:HAS_EXTRACTION]->(er:ExtractionResult)
                DETACH DELETE er
                RETURN n.tenant_id AS tenant_id, n.created_by_user_id AS created_by_user_id,
                       n.job_id AS job_id
                """,
                nid=row_id,
            )
            rec = await result.single()
            if rec is None:
                return None
            tenant_id = require_stored_tenant(rec["tenant_id"], row_id)
            created_by_user_id = rec["created_by_user_id"]
            job_id = rec["job_id"]
            # Create fresh ExtractionResult node
            await session.run(
                """
                MATCH (n:StructureNode {id: $nid})
                CREATE (er:ExtractionResult {
                    uid:           $uid,
                    node_full_id:  $nid,
                    document_name: $doc_name,
                    model_class:   'GenericTabularRow',
                    source:        'tabular_raw',
                    timestamp:     $timestamp,
                    tenant_id:            $tenant_id,
                    created_by_user_id:   $created_by_user_id,
                    job_id:               $job_id
                })
                CREATE (n)-[:HAS_EXTRACTION]->(er)
                """,
                nid=row_id,
                uid=extraction_uid,
                doc_name=document_name,
                timestamp=timestamp,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
        return Provenance(
            tenant_id=tenant_id, created_by_user_id=created_by_user_id, job_id=job_id
        )

    @neo4j_errors()
    async def link_raw_column_value(
        self,
        er_uid: str,
        rel_type: str,
        entity_uid: str,
        value: str,
        normalized_value: str,
        provenance: Provenance,
    ) -> None:
        cfg = get_config()
        async with self._driver.session(database=cfg.neo4j_database) as session:
            query = f"""
                MERGE (e:Entity {{uid: $uid}})
                ON CREATE SET e.normalized_value = $normalized_value,
                          e.value = $value, {provenance_set_clause("e")}
                ON MATCH SET {provenance_set_clause("e")}
                WITH e
                MATCH (er:ExtractionResult {{uid: $er_uid}})
                MERGE (er)-[:{rel_type}]->(e)
                """
            params = dict(
                normalized_value=normalized_value,
                uid=entity_uid,
                value=value,
                er_uid=er_uid,
                tenant_id=provenance.tenant_id,
                created_by_user_id=provenance.created_by_user_id,
                job_id=provenance.job_id,
            )
            await with_neo4j_retry(lambda: session.run(query, **params))


async def _delete_tabular_subgraph(
    driver: AsyncDriver,
    doc_path: str,
    resolved_version: int,
    sheet_index: int,
    *,
    tenant_id: str,
) -> None:
    """Delete an existing Table node and all its Row descendants before re-insertion.

    *tenant_id* is the stored tenant key of the owning :Document (it prefixes
    the Table id, so only that tenant's table is touched).

    Used in update_mode. Deletes in leaf-first order:
    1. ExtractionResult ModelInstance children on Row nodes
    2. ExtractionResults on Row nodes
    3. InfoUnits on Row nodes
    4. Row nodes (DETACH DELETE removes HAS_CHILD, HAS_MODEL_DECISION, HAS_INFO_UNIT)
    5. ModelDecision subgraph on Table (leaf-first)
    6. Table node (DETACH DELETE removes HAS_STRUCTURE from Document)

    NOTE: ModelDecision deletion is handled by write_annotation()'s own idempotency
    guard, so we only need to detach-delete the Table itself.
    LabeledEntity singletons are NOT deleted (global across the graph).
    """
    table_node_id = f"table_{sheet_index + 1}"
    table_composite_id = make_structure_node_id(tenant_id, doc_path, resolved_version, table_node_id)
    cfg = get_config()
    async with driver.session(database=cfg.neo4j_database) as session:
        tx = await session.begin_transaction()
        try:
            # 1. ModelInstance children under ExtractionResults on Rows
            # Skip tabular_raw rows: their ExtractionResult links to :Entity singletons
            # which must NOT be deleted here.
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                      -[:HAS_EXTRACTION]->(er:ExtractionResult)
                WHERE er.source <> 'tabular_raw'
                WITH er
                MATCH (er)-[*1..10]->(mi:ModelInstance)
                DETACH DELETE mi
                """,
                table_id=table_composite_id,
            )
            # 2. ExtractionResults on Rows
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                      -[:HAS_EXTRACTION]->(er:ExtractionResult)
                DETACH DELETE er
                """,
                table_id=table_composite_id,
            )
            # 3. InfoUnits on Rows
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                      -[:HAS_INFO_UNIT]->(iu:InfoUnit)
                DETACH DELETE iu
                """,
                table_id=table_composite_id,
            )
            # 4. Row nodes
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                DETACH DELETE r
                """,
                table_id=table_composite_id,
            )
            # 5. ModelDecision subgraph on Table (leaf-first)
            await delete_stale_model_decision(tx, table_composite_id)
            # 6. Table node
            await tx.run(
                "MATCH (n:StructureNode {id: $id}) DETACH DELETE n",
                id=table_composite_id,
            )
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise

    logger.info("delete_tabular_subgraph: deleted table at %s", table_composite_id)


@asynccontextmanager
async def _open_tabular_sink() -> AsyncIterator[Neo4jTabularSink]:
    """The sink over the shared async driver (it opens its own sessions)."""
    yield Neo4jTabularSink(_driver.get_async_driver())


class Neo4jTabularOps(TabularOps):
    """Tabular operations over Neo4j (asynchronous, shared async driver)."""

    def sink(self) -> AbstractAsyncContextManager[TabularSink]:
        """Open the tabular sink: every statement opens its own session on the shared async driver."""
        return _open_tabular_sink()

    @neo4j_errors()
    async def delete_tabular_subgraph(
        self,
        doc_path: str,
        resolved_version: int,
        sheet_index: int,
        *,
        tenant_id: str,
    ) -> None:
        """Delete the Table of one sheet and its Row descendants (explicit transaction).

        Args:
            tenant_id: Stored tenant key of the owning :Document; it prefixes the
                Table id, so only that tenant's table is touched.

        Raises:
            GraphError: If the engine fails.
        """
        await _delete_tabular_subgraph(
            _driver.get_async_driver(),
            doc_path,
            resolved_version,
            sheet_index,
            tenant_id=tenant_id,
        )
