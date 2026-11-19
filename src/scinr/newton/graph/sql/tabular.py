"""
graph/sql/tabular.py — ``store.tabular`` over SQL: the sheet sink and the sheet delete.

Translation of ``graph/neo4j/tabular.py``. The sink holds no connection: every method opens
its own on the process-wide async engine (one transaction where the Cypher has an explicit
one, one per statement where it runs in autocommit), as the Neo4j sink opens one session per
method. A Row batch is executemany over the batch (the ``UNWIND`` of Cypher). Tables and
Rows carry the ``_document_key`` of the Document; the Row InfoUnits also carry their Row in
``_structure_node_id``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from scinr.newton.graph.base import TabularOps
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._shared import (
    STALE_EXTRACTION_MAX_HOPS,
    delete_found,
    delete_stale_model_decision,
    node_provenance,
    prov_values,
    provenance_on_create,
    provenance_on_match,
)
from scinr.newton.graph.sql._traversal import reachable
from scinr.newton.graph.sql.extraction import create_extraction_result
from scinr.newton.graph.types import Provenance, RowData, TableNode
from scinr.newton.ingest.nodes import make_structure_node_id
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.retry import with_sql_retry
from scinr.newton.utils.tenancy import require_stored_tenant

if TYPE_CHECKING:
    from scinr.newton.graph.sinks import TabularSink

__all__ = ["SqlTabularOps", "SqlTabularSink"]

logger = logging.getLogger(__name__)

D = t.documents
SN = t.structure_nodes
ER = t.extraction_results

#: Columns a Row re-merge sets (``SET r.node_id = …, …`` of Cypher, plus its document).
_ROW_SET = (
    "node_id",
    "role",
    "appearance_order",
    "theme",
    "row_index",
    "source_page_ids",
    "tenant_id",
    "created_by_user_id",
    "job_id",
    "_document_key",
)
#: Columns an InfoUnit re-merge sets.
_INFO_UNIT_SET = (
    "title",
    "description",
    "order",
    "tenant_id",
    "created_by_user_id",
    "job_id",
    "_document_key",
    "_structure_node_id",
)


def _pages(page_id: str) -> list[str]:
    """``CASE WHEN $page_id <> '' THEN [$page_id] ELSE [] END``."""
    return [page_id] if page_id != "" else []


class SqlTabularSink:
    """:class:`~scinr.newton.graph.sinks.TabularSink` over the process-wide async engine.

    Each method opens its own connection, like the Neo4j sink opens its own session.
    """

    @async_sql_errors()
    async def create_table(self, table: TableNode) -> None:
        """MATCH d {tenant, path, version}; MERGE (n:StructureNode {id}) SET n:Table, …;
        MERGE (d)-[:HAS_STRUCTURE]->(n); one transaction.

        Raises:
            RuntimeError: If the Document does not exist (nothing is written).
        """
        tenant_id, doc_path, version = table.doc_key
        async with c.async_write_engine().begin() as connection:
            doc = (
                await connection.execute(
                    sa.select(D.c._key, D.c.tenant_id, D.c.created_by_user_id, D.c.job_id).where(
                        D.c.tenant_id == tenant_id, D.c.path == doc_path, D.c.version == version
                    )
                )
            ).first()
            if doc is None:
                raise RuntimeError(
                    f"write_tabular_subgraph: Document not found "
                    f"(tenant={tenant_id!r}, path={doc_path!r}, version={version})"
                )
            values = {
                "node_id": table.node_id,
                "role": "table",
                "title": table.title,
                "appearance_order": table.order,
                "theme": table.theme,
                "column_count": table.col_count,
                "row_count": table.row_count,
                "source_page_ids": _pages(table.page_id),
                "tenant_id": doc.tenant_id,
                "created_by_user_id": doc.created_by_user_id,
                "job_id": doc.job_id,
            }
            aux = {"_document_key": doc._key}
            await connection.execute(
                c.merge_node(
                    "StructureNode",
                    {"id": table.id},
                    on_create=values,
                    on_match=values,
                    aux=aux,
                    aux_on_match=aux,
                )
            )
            await connection.execute(
                c.merge_edge(
                    "HAS_STRUCTURE",
                    "Document",
                    doc._key,
                    "StructureNode",
                    table.id,
                    tenant=doc.tenant_id,
                )
            )

    @async_sql_errors()
    async def write_row_batch(
        self, table_id: str, rows: list[RowData], md_uid: str, theme: str, page_id: str
    ) -> None:
        """The ``UNWIND`` of Cypher, one transaction: MERGE each Row (SET …) + HAS_CHILD, MERGE
        its InfoUnit (SET …) + HAS_INFO_UNIT, MERGE HAS_MODEL_DECISION to the ModelDecision.

        Nothing is written when the Table does not exist; the decision links are skipped
        when the ModelDecision does not (the Rows and InfoUnits stay, as in Cypher).
        """
        if not rows:
            return
        async with c.async_write_engine().begin() as connection:
            found = await node_provenance(connection, table_id)
            if found is None:
                return
            prov, document_key = found
            tenant = prov.tenant_id
            row_rows = [
                {
                    "id": row.id,
                    "node_id": row.node_id,
                    "role": "row",
                    "title": None,
                    "appearance_order": row.appearance_order,
                    "theme": theme,
                    "row_index": row.row_index,
                    "source_page_ids": _pages(page_id),
                    "tenant_id": prov.tenant_id,
                    "created_by_user_id": prov.created_by_user_id,
                    "job_id": prov.job_id,
                    "props": {},
                    "_document_key": document_key,
                }
                for row in rows
            ]
            info_rows = [
                {
                    "uid": row.info_uid,
                    "info_unit_id": None,
                    "title": "Row data",
                    "description": row.description,
                    "order": 0,
                    "tenant_id": prov.tenant_id,
                    "created_by_user_id": prov.created_by_user_id,
                    "job_id": prov.job_id,
                    "props": {},
                    "_document_key": document_key,
                    "_structure_node_id": row.id,
                }
                for row in rows
            ]
            await connection.execute(c.upsert_many("StructureNode", _ROW_SET), row_rows)
            await connection.execute(
                c.merge_edges_many("StructureNode", "StructureNode"),
                [
                    c.edge_row(
                        "HAS_CHILD",
                        "StructureNode",
                        table_id,
                        "StructureNode",
                        row.id,
                        tenant=tenant,
                    )[1]
                    for row in rows
                ],
            )
            await connection.execute(c.upsert_many("InfoUnit", _INFO_UNIT_SET), info_rows)
            await connection.execute(
                c.merge_edges_many("StructureNode", "InfoUnit"),
                [
                    c.edge_row(
                        "HAS_INFO_UNIT",
                        "StructureNode",
                        row.id,
                        "InfoUnit",
                        row.info_uid,
                        tenant=tenant,
                    )[1]
                    for row in rows
                ],
            )
            decision = c.key_column("ModelDecision")
            if (await connection.execute(sa.select(decision).where(decision == md_uid))).first():
                await connection.execute(
                    c.merge_edges_many("StructureNode", "ModelDecision"),
                    [
                        c.edge_row(
                            "HAS_MODEL_DECISION",
                            "StructureNode",
                            row.id,
                            "ModelDecision",
                            md_uid,
                            tenant=tenant,
                        )[1]
                        for row in rows
                    ],
                )

    @async_sql_errors()
    async def replace_raw_extraction_result(
        self, row_id: str, extraction_uid: str, document_name: str, timestamp: str
    ) -> Provenance | None:
        """Two autocommit statements: DETACH DELETE the ExtractionResults of the Row (reading
        its provenance), then CREATE the ``tabular_raw`` ExtractionResult + HAS_EXTRACTION.

        Returns ``None`` when the Row does not exist.

        Raises:
            RuntimeError: If the Row has no tenant (``require_stored_tenant``).
        """
        engine = c.async_write_engine()
        async with engine.begin() as connection:
            found = await node_provenance(connection, row_id)
            if found is None:
                return None
            await delete_found(
                connection,
                c.follow(
                    c.nodes_with_keys("StructureNode", [row_id]),
                    [("HAS_EXTRACTION", "ExtractionResult")],
                ),
            )
        raw = found[0]
        tenant_id = require_stored_tenant(raw.tenant_id, row_id)
        prov = Provenance(
            tenant_id=tenant_id, created_by_user_id=raw.created_by_user_id, job_id=raw.job_id
        )
        values = {
            "node_full_id": row_id,
            "document_name": document_name,
            "model_class": "GenericTabularRow",
            "source": "tabular_raw",
            "timestamp": timestamp,
            **prov_values(prov),
        }
        async with engine.begin() as connection:
            await create_extraction_result(connection, row_id, extraction_uid, values)
        return prov

    @async_sql_errors()
    async def link_raw_column_value(
        self,
        er_uid: str,
        rel_type: str,
        entity_uid: str,
        value: str,
        normalized_value: str,
        provenance: Provenance,
    ) -> None:
        """MERGE (e:Entity {uid}) ON CREATE SET values + provenance ON MATCH SET provenance;
        MATCH er; MERGE (er)-[:<rel_type>]->(e). One statement, retried."""
        target = c.table("Entity")
        entity = c.merge_node(
            "Entity",
            {"uid": entity_uid},
            on_create={
                "normalized_value": normalized_value,
                "value": value,
                **provenance_on_create(provenance),
            },
            on_match=provenance_on_match(target, provenance),
        )
        engine = c.async_write_engine()

        async def attempt() -> None:
            async with engine.begin() as connection:
                await connection.execute(entity)
                row = (
                    await connection.execute(sa.select(ER.c.tenant_id).where(ER.c.uid == er_uid))
                ).first()
                if row is None:
                    return
                await connection.execute(
                    c.merge_edge(
                        rel_type,
                        "ExtractionResult",
                        er_uid,
                        "Entity",
                        entity_uid,
                        tenant=row.tenant_id,
                    )
                )

        await with_sql_retry(attempt)


async def _rows_of(connection: AsyncConnection, table_id: str) -> list[str]:
    """(t {id})-[:HAS_CHILD]->(r:StructureNode:Row)."""
    hop = c.follow(
        c.nodes_with_keys("StructureNode", [table_id]), [("HAS_CHILD", "StructureNode")]
    ).subquery()
    return c.column_values(
        await connection.execute(
            sa.select(SN.c.id).where(SN.c.id.in_(sa.select(hop.c.key)), SN.c.role == "row")
        )
    )


async def _delete_tabular_subgraph(connection: AsyncConnection, table_id: str) -> None:
    """The six leaf-first steps of the Cypher, on one transaction."""
    rows = await _rows_of(connection, table_id)
    results: list[str] = []
    if rows:
        er_hop = c.follow(
            c.nodes_with_keys("StructureNode", rows), [("HAS_EXTRACTION", "ExtractionResult")]
        )
        results = c.column_values(await connection.execute(sa.select(er_hop.subquery().c.key)))
    if results:
        # 1. ModelInstances under the non-raw ExtractionResults. Cypher filters with
        #    ``er.source <> 'tabular_raw'``, which is null (so false) for an
        #    ExtractionResult without ``source``: only the ones with another source qualify.
        source = c.json_text(ER.c.props, "source")
        qualifying = c.column_values(
            await connection.execute(
                sa.select(ER.c.uid).where(
                    ER.c.uid.in_(results), source.is_not(None), source != "tabular_raw"
                )
            )
        )
        if qualifying:
            below = reachable(
                c.nodes_with_keys("ExtractionResult", qualifying),
                max_hops=STALE_EXTRACTION_MAX_HOPS,
                name="tabular_stale",
            )
            instances = c.column_values(
                await connection.execute(
                    sa.select(below.c.key).where(below.c.label == "ModelInstance").distinct()
                )
            )
            for batch in c.batched(instances, c.IN_BATCH):
                for statement in c.detach_delete("ModelInstance", batch):
                    await connection.execute(statement)
        # 2. The ExtractionResults of the Rows.
        for statement in c.detach_delete("ExtractionResult", results):
            await connection.execute(statement)
    if rows:
        # 3. The InfoUnits of the Rows.
        await delete_found(
            connection,
            c.follow(c.nodes_with_keys("StructureNode", rows), [("HAS_INFO_UNIT", "InfoUnit")]),
        )
        # 4. The Rows.
        for statement in c.detach_delete("StructureNode", rows):
            await connection.execute(statement)
    # 5. The decision subtree of the Table (leaf first).
    await delete_stale_model_decision(connection, table_id)
    # 6. The Table.
    for statement in c.detach_delete("StructureNode", [table_id]):
        await connection.execute(statement)


@asynccontextmanager
async def _open_tabular_sink() -> AsyncIterator[SqlTabularSink]:
    """The sink over the process-wide async engine (it opens its own connections)."""
    yield SqlTabularSink()


class SqlTabularOps(TabularOps):
    """``store.tabular`` over SQL (asynchronous, process-wide async engine)."""

    def sink(self) -> AbstractAsyncContextManager[TabularSink]:
        """Open the tabular sink: every statement opens its own connection."""
        return _open_tabular_sink()

    async def delete_tabular_subgraph(
        self,
        doc_path: str,
        resolved_version: int,
        sheet_index: int,
        *,
        tenant_id: str,
    ) -> None:
        """Delete the Table of one sheet and its Row descendants (one transaction).

        Args:
            tenant_id: Stored tenant key of the owning :Document; it prefixes the Table id,
                so only that tenant's table is touched.

        Raises:
            GraphError: If the engine fails.
        """
        table_id = make_structure_node_id(
            tenant_id, doc_path, resolved_version, f"table_{sheet_index + 1}"
        )
        async with async_sql_errors():
            async with c.async_write_engine().begin() as connection:
                await _delete_tabular_subgraph(connection, table_id)
        logger.info("delete_tabular_subgraph: deleted table at %s", table_id)
