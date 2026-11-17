"""
graph/neo4j/restore.py — Restore over Neo4j: batch writer, finishing queries, reads.

:class:`Neo4jRestoreOps` (``store.restore``) holds the Cypher of
``restore_document()``. What goes in each batch, and in which order, is decided
by the neutral ``_Rebuilder`` of ``ingest/restore.py``; this module only runs
the batches:

- :class:`Neo4jRestoreWriter` — runs write transactions with at most
  *concurrency* in flight, each in its own session; batches of the same *lane*
  run one after the other. The producer waits for a free slot **before**
  handing over a batch, so a fast reader never piles up more than
  *concurrency* batches in memory (``O(batch_size × concurrency)``).
- the queries that finish a restore (temporary links, unmarking the stubs,
  ``latest``), the snapshot-reference count and the stale-extraction check.

Every query that reads :Document nodes is its own method: none receives query
text.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Hashable, Iterable, Sequence
from functools import lru_cache
from typing import Any

from scinr.newton.config import get_config
from scinr.newton.graph.base import RestoreOps, RestoreWriter
from scinr.newton.graph.neo4j._scope import borrow_sync_driver
from scinr.newton.graph.neo4j._shared import (
    DELETE_BATCH_ROWS,
    DOCS_BY_KEY,
    DOCUMENTS_PER_QUERY,
    delete_stale_extraction_result,
    delete_stale_model_decision,
    run_count_query,
)
from scinr.newton.graph.neo4j.driver import get_async_driver
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.retry import with_neo4j_retry_sync
from scinr.newton.graph.types import DocumentInfo, RestoreCounters
from scinr.newton.ingest.freeze import is_catalog, node_identity, validate_identifier

logger = logging.getLogger(__name__)

# Nodes merged across documents of a tenant (business-key uid): on match they
# only gain the snapshot's provenance.
_MERGED_LABELS = frozenset({"ModelInstance", "Entity", "LabeledEntity"})


def cypher_identifier(name: str) -> str:
    """Backquote a label / relationship type after validating it (``FreezeError`` if unsafe)."""
    return f"`{validate_identifier(name)}`"


# ---------------------------------------------------------------------------
# Batch queries
# ---------------------------------------------------------------------------


def _key_map(source: str, key_fields: Sequence[str]) -> str:
    # key_fields come from node_identity() — fixed names, never snapshot data.
    return ", ".join(f"{name}: {source}.{name}" for name in key_fields)


def _union(field_name: str) -> str:
    return (
        f"n.{field_name} = reduce(acc = coalesce(n.{field_name}, []), "
        f"x IN coalesce(row.props.{field_name}, []) | "
        "CASE WHEN x IN acc THEN acc ELSE acc + x END)"
    )


@lru_cache(maxsize=256)
def _node_merge_query(labels: tuple[str, ...]) -> str:
    """``MERGE`` a batch of nodes sharing *labels*; props only on create."""
    primary, key_fields = node_identity(labels)
    extra_labels = "".join(
        f", n:{cypher_identifier(label)}" for label in labels if label != primary
    )
    query = (
        "UNWIND $rows AS row\n"
        f"MERGE (n:{cypher_identifier(primary)} {{{_key_map('row.key', key_fields)}}})\n"
        f"ON CREATE SET n += row.props{extra_labels}\n"
    )
    if primary in _MERGED_LABELS:
        query += f"ON MATCH SET {_union('created_by_user_ids')}, {_union('job_ids')}\n"
    return query


@lru_cache(maxsize=1024)
def _relationship_query(rel_type: str, from_label: str, to_label: str) -> str:
    """Create a batch of relationships unless an identical one exists.

    Catalog targets are ``MERGE``d by their key (like the pipeline does).
    """
    _, from_fields = node_identity([from_label])
    _, to_fields = node_identity([to_label])
    rel = cypher_identifier(rel_type)
    target = "MERGE" if is_catalog(to_label) else "MATCH"
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (a:{cypher_identifier(from_label)} {{{_key_map('row.from', from_fields)}}})\n"
        f"{target} (b:{cypher_identifier(to_label)} {{{_key_map('row.to', to_fields)}}})\n"
        "WITH a, b, row\n"
        f"WHERE NOT EXISTS {{ MATCH (a)-[x:{rel}]->(b) WHERE properties(x) = row.props }}\n"
        f"CREATE (a)-[r:{rel}]->(b)\n"
        "SET r = row.props\n"
    )


async def _run_batch(tx, query: str, rows: list[dict[str, Any]]) -> RestoreCounters:
    result = await tx.run(query, rows=rows)
    summary = await result.consume()
    counters = summary.counters
    return RestoreCounters(
        nodes_created=counters.nodes_created,
        relationships_created=counters.relationships_created,
    )


async def _delete_stale(tx, node_ids: list[str], model_decisions: bool, extraction_results: bool):
    if extraction_results:
        await delete_stale_extraction_result(tx, node_ids)
    if model_decisions:
        await delete_stale_model_decision(tx, node_ids)


class Neo4jRestoreWriter(RestoreWriter):
    """Runs write transactions with at most *concurrency* in flight.

    Each batch runs in its own session (an async session is not safe for
    concurrent use; sessions borrow pooled connections). Batches of the same
    *lane* run one after the other, lanes in parallel. The producer waits for
    a free slot **before** handing over a batch, so a fast reader never piles
    up more than *concurrency* batches in memory.

    Every public method may raise a :class:`~scinr.newton.exceptions.GraphError`
    (the first error of a batch is kept and raised, translated, by the next
    ``delete_stale`` / ``merge_nodes`` / ``create_relationships`` that has to
    wait, or by ``drain()``).
    """

    def __init__(self, driver, database: str, concurrency: int) -> None:
        self._driver = driver
        self._database = database
        self._slots = asyncio.Semaphore(concurrency)
        self._lanes: dict[Hashable, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()
        self._error: BaseException | None = None

    async def _run(
        self,
        lane: Hashable,
        work: Callable[..., Awaitable[Any]],
        args: tuple,
        on_done: Callable[[Any], None],
    ) -> None:
        try:
            async with self._lanes.setdefault(lane, asyncio.Lock()):
                async with self._driver.session(database=self._database) as session:
                    on_done(await session.execute_write(work, *args))
        except Exception as exc:
            # Kept for drain() to raise: the task itself never fails, so a
            # finished task nobody awaits leaves no "exception never retrieved".
            if self._error is None:
                self._error = exc
        finally:
            self._slots.release()

    async def _submit(
        self,
        lane: Hashable,
        work: Callable[..., Awaitable[Any]],
        *args,
        on_done: Callable[[Any], None] = lambda _: None,
    ) -> None:
        if self._error is not None:
            await self._drain()
        await self._slots.acquire()
        task = asyncio.create_task(self._run(lane, work, args, on_done))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._error is not None:
            raise self._error

    @neo4j_errors()
    async def delete_stale(
        self,
        lane: Hashable,
        node_ids: Sequence[str],
        *,
        model_decisions: bool,
        extraction_results: bool,
    ) -> None:
        """Delete the stale ModelDecision / ExtractionResult subtrees hanging from *node_ids*."""
        await self._submit(lane, _delete_stale, list(node_ids), model_decisions, extraction_results)

    @neo4j_errors()
    async def merge_nodes(
        self,
        lane: Hashable,
        labels: tuple[str, ...],
        rows: Sequence[dict[str, Any]],
        on_done: Callable[[RestoreCounters], None],
    ) -> None:
        """``MERGE`` a batch of nodes sharing *labels* (props only on create)."""
        await self._submit(lane, _run_batch, _node_merge_query(labels), rows, on_done=on_done)

    @neo4j_errors()
    async def create_relationships(
        self,
        lane: Hashable,
        group: tuple[str, str, str],
        rows: Sequence[dict[str, Any]],
        on_done: Callable[[RestoreCounters], None],
    ) -> None:
        """Create a batch of relationships of *group* unless an identical one exists."""
        await self._submit(lane, _run_batch, _relationship_query(*group), rows, on_done=on_done)

    @neo4j_errors()
    async def drain(self) -> None:
        """Wait for every batch in flight; re-raise (translated) the first failure."""
        await self._drain()

    @neo4j_errors()
    async def abort(self) -> None:
        """Cancel every batch in flight."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Finishing — temporary links, stubs, snapshot references
# ---------------------------------------------------------------------------

# A Document never has outgoing HAS_MODEL_DECISION / HAS_EXTRACTION outside
# a freeze, so any such link on a restored stub is a temporary one. There is
# one per kept ModelDecision / ExtractionResult, so they go in batches (an
# auto-commit query, like the deletes of graph/neo4j/deletion.py); grouping by the
# link makes the query read them all before the first batch commits.
_TEMPORARY_LINKS_DELETE_QUERY = (
    DOCS_BY_KEY
    + f"""MATCH (d)-[link:HAS_MODEL_DECISION|HAS_EXTRACTION]->()
WITH link, count(*) AS hits
CALL (link) {{
  DELETE link
}} IN TRANSACTIONS OF {DELETE_BATCH_ROWS} ROWS
RETURN count(*) AS n
"""
)

_RESTORE_FINISH_QUERY = (
    DOCS_BY_KEY
    + """SET d.frozen = false
REMOVE d.frozen_blob_id, d.frozen_at, d.frozen_keep_structure_nodes,
       d.frozen_keep_annotations, d.frozen_keep_extraction_results,
       d.frozen_cleanup_pending
RETURN count(d) AS n
"""
)

# ``latest`` is the newest version of a path: a recreated Document may be an
# older version than one ingested since, or the newest one again.
_RECOMPUTE_LATEST_QUERY = """
UNWIND $paths AS p
MATCH (d:Document {tenant_id: $tenant_id, path: p})
WITH p, collect(d) AS versions, max(d.version) AS newest
UNWIND versions AS d
SET d.latest = (d.version = newest)
"""

_EXISTING_DOCUMENTS_QUERY = DOCS_BY_KEY + "RETURN properties(d) AS properties\n"

_STUBS_OF_BLOB_QUERY = """
MATCH (d:Document {tenant_id: $tenant_id})
WHERE d.frozen = true AND d.frozen_blob_id = $blob_id
RETURN properties(d) AS properties
"""

_BLOB_REFERENCES_QUERY = """
MATCH (d:Document {tenant_id: $tenant_id})
WHERE d.frozen_blob_id = $blob_id
RETURN count(d) AS n
"""

_STALE_EXTRACTION_RESULTS_QUERY = """
UNWIND $node_ids AS nid
MATCH (:StructureNode {id: nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
RETURN count(er) AS n
"""


def _read_documents(driver, database: str, query: str, **params: Any) -> list[DocumentInfo]:
    """Run one of this module's Document-reading queries (never caller-supplied text)."""

    def _do_read() -> list[DocumentInfo]:
        with driver.session(database=database) as session:
            return [
                DocumentInfo(
                    path=record["properties"].get("path"),
                    version=record["properties"].get("version"),
                    tenant_id=record["properties"].get("tenant_id"),
                    is_seed=False,
                    properties=dict(record["properties"]),
                )
                for record in session.run(query, **params)
            ]

    return with_neo4j_retry_sync(_do_read)


class Neo4jRestoreOps(RestoreOps):
    """``store.restore`` over Neo4j. Synchronous methods open their sessions on the
    driver of the current ``sync_scope()`` (or a private one)."""

    def batch_writer(self, concurrency: int) -> RestoreWriter:
        """A :class:`Neo4jRestoreWriter` on the shared async driver.

        Must be called from the event loop that will use it.

        Args:
            concurrency: Write transactions in flight at once.

        Raises:
            GraphConfigurationError: If the credentials are not configured.
        """
        with neo4j_errors():
            return Neo4jRestoreWriter(get_async_driver(), get_config().neo4j_database, concurrency)

    @neo4j_errors()
    def existing_documents(self, tenant: str, keys: Sequence[dict[str, Any]]) -> list[DocumentInfo]:
        """The :Document nodes among *keys* that exist.

        Args:
            tenant: Stored tenant key.
            keys: ``{"path", "version"}`` of every candidate.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            return _read_documents(
                driver, database, _EXISTING_DOCUMENTS_QUERY, tenant_id=tenant, keys=list(keys)
            )

    @neo4j_errors()
    def frozen_stubs_of_blob(self, tenant: str, blob_id: str) -> list[DocumentInfo]:
        """The documents of *tenant* frozen with the snapshot *blob_id*.

        Args:
            tenant: Stored tenant key.
            blob_id: The ``frozen_blob_id`` of the snapshot.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            return _read_documents(
                driver, database, _STUBS_OF_BLOB_QUERY, tenant_id=tenant, blob_id=blob_id
            )

    @neo4j_errors()
    def finish_restore(
        self, tenant: str, docs: Sequence[DocumentInfo], recreated_paths: Sequence[str]
    ) -> None:
        """Remove the temporary links and unmark the stubs, ``DOCUMENTS_PER_QUERY``
        documents at a time, then recompute ``latest`` for the recreated paths.

        Not one transaction: a freeze that kept annotations or extraction results
        leaves one link per kept node, and deleting those of hundreds of
        documents at once is what a transaction memory pool cannot hold. A
        document is unmarked only once its links are gone; if this stops
        half-way the remaining documents are still frozen stubs, so calling
        ``restore_document()`` again finishes them.

        Args:
            tenant: Stored tenant key.
            docs: The restored documents.
            recreated_paths: Paths of the documents that were recreated.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        keys = [doc.key for doc in docs]
        with borrow_sync_driver() as driver:
            for start in range(0, len(keys), DOCUMENTS_PER_QUERY):
                chunk = keys[start : start + DOCUMENTS_PER_QUERY]
                run_count_query(
                    driver, database, _TEMPORARY_LINKS_DELETE_QUERY, tenant_id=tenant, keys=chunk
                )

                def _do_unmark(chunk: list[dict[str, Any]] = chunk) -> None:
                    with driver.session(database=database) as session:
                        session.execute_write(
                            lambda tx: tx.run(
                                _RESTORE_FINISH_QUERY, tenant_id=tenant, keys=chunk
                            ).consume()
                        )

                with_neo4j_retry_sync(_do_unmark)

            if recreated_paths:
                paths = list(recreated_paths)

                def _do_latest() -> None:
                    with driver.session(database=database) as session:
                        session.execute_write(
                            lambda tx: tx.run(
                                _RECOMPUTE_LATEST_QUERY, tenant_id=tenant, paths=paths
                            ).consume()
                        )

                with_neo4j_retry_sync(_do_latest)

    @neo4j_errors()
    def count_blob_references(self, tenant: str, blob_id: str) -> int:
        """How many documents of *tenant* still reference the snapshot *blob_id*.

        Args:
            tenant: Stored tenant key.
            blob_id: The ``frozen_blob_id`` of the snapshot.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:

            def _do_count() -> int:
                with driver.session(database=database) as session:
                    record = session.run(
                        _BLOB_REFERENCES_QUERY, tenant_id=tenant, blob_id=blob_id
                    ).single()
                    return record["n"] if record else 0

            return with_neo4j_retry_sync(_do_count)

    @neo4j_errors()
    def has_stale_extraction_results(self, node_id_groups: Iterable[Sequence[str]]) -> bool:
        """Whether any kept StructureNode still carries an ExtractionResult the
        restore is about to delete — the only thing a restore deletes that can
        orphan an :Entity / :ModelInstance / :LabeledEntity.

        Args:
            node_id_groups: One list of StructureNode ids per snapshot entry; each
                list is checked :data:`DELETE_BATCH_ROWS` ids at a time.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            for ids in node_id_groups:
                for start in range(0, len(ids), DELETE_BATCH_ROWS):
                    batch = list(ids[start : start + DELETE_BATCH_ROWS])
                    if run_count_query(
                        driver, database, _STALE_EXTRACTION_RESULTS_QUERY, node_ids=batch
                    ):
                        return True
        return False
