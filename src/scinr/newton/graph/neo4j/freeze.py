"""
graph/neo4j/freeze.py — Document freezing over Neo4j: set resolution, export reads, marks, reduction.

:class:`Neo4jFreezeOps` (``store.freeze``) holds the Cypher of ``freeze_document()``
and ``export_document_snapshot()``: the resolution of the document set, the
reads that feed a snapshot, the ``frozen`` / backup marks and the deletion of the
subtree of the stubs (the batched cascade of ``graph/neo4j/deletion.py``). The
snapshot format, the validation of what is read (tenant, business keys), the
deduplication and the orchestration stay in ``ingest/freeze.py``.

One export is one session: ``export_snapshot()`` retries the *whole* attempt
(the caller's callback opens the output file inside it), so a retry starts the
snapshot over. The :class:`ExportReader` it hands out streams — each method
returns a generator over the open result — so the callback writes as it reads
and nothing is materialised here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from scinr.newton.config import get_config
from scinr.newton.exceptions import FreezeError
from scinr.newton.graph.base import FreezeOps, OrphanCollector
from scinr.newton.graph.neo4j._scope import borrow_sync_driver
from scinr.newton.graph.neo4j._shared import DELETE_BATCH_ROWS, DOCS_BY_KEY, run_count_query
from scinr.newton.graph.neo4j.deletion import doc_match, run_subtree_steps, subtree_steps
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.gc import GC_REACHABILITY_MAX_HOPS
from scinr.newton.graph.neo4j.retry import with_neo4j_retry_sync
from scinr.newton.graph.types import (
    DocumentInfo,
    DocumentSelector,
    ExportReader,
    KeepFlags,
    RawNode,
    RawRelationship,
)
from scinr.newton.ingest.freeze import CATALOG_KEY_FIELDS

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Document set
# ---------------------------------------------------------------------------

# Matched Documents (``is_seed``) plus their IS_COMPOSED_OF* descendants,
# each once. Appended to the doc_match() prefix (tenant always set).
_DOCUMENT_SET_TAIL = """
OPTIONAL MATCH (d)-[:IS_COMPOSED_OF*]->(cd:Document)
WITH collect(DISTINCT d) AS seeds, collect(DISTINCT cd) AS descendants
UNWIND seeds + descendants AS doc
WITH DISTINCT doc, seeds
RETURN properties(doc) AS properties, doc IN seeds AS is_seed
ORDER BY doc.path, doc.version
"""

# ---------------------------------------------------------------------------
# Snapshot export — reads
# ---------------------------------------------------------------------------

_CATALOG_PREDICATE = " OR ".join(f"n:{label}" for label in CATALOG_KEY_FIELDS)

# Every node of one document's subtree, once: the whole HAS_STRUCTURE /
# HAS_CHILD tree (unbounded — a strict hierarchy), what hangs from each
# StructureNode (InfoUnit, ModelDecision and its children, ExtractionResult),
# and everything reachable from each ExtractionResult through any outgoing
# relationship within GC_REACHABILITY_MAX_HOPS hops (ModelInstance / Entity /
# LabeledEntity may form cycles), catalog nodes excluded.
_SUBTREE_NODES_FRAGMENT = f"""
MATCH (d:Document {{tenant_id: $tenant_id, path: $path, version: $version}})
CALL (d) {{
  MATCH (d)-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)
  WITH DISTINCT s
  CALL (s) {{
    RETURN s AS n
    UNION
    MATCH (s)-[:HAS_INFO_UNIT]->(n:InfoUnit)
    RETURN n
    UNION
    MATCH (s)-[:HAS_MODEL_DECISION]->(n:ModelDecision)
    RETURN n
    UNION
    MATCH (s)-[:HAS_MODEL_DECISION]->(:ModelDecision)
          -[:HAS_PROPOSED_MODEL|HAS_COMPLEMENTARY_MATCH|HAS_SUPPLEMENTARY_FIELD]->(n)
    RETURN n
    UNION
    MATCH (s)-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(:ProposedModel)
          -[:HAS_PROPOSED_FIELD]->(n:ProposedField)
    RETURN n
    UNION
    MATCH (s)-[:HAS_EXTRACTION]->(n:ExtractionResult)
    RETURN n
    UNION
    MATCH (s)-[:HAS_EXTRACTION]->(:ExtractionResult)-[*1..{GC_REACHABILITY_MAX_HOPS}]->(n)
    WHERE NOT ({_CATALOG_PREDICATE})
    RETURN n
  }}
  RETURN n
}}
WITH DISTINCT n
"""

_SUBTREE_NODES_EXPORT_QUERY = (
    _SUBTREE_NODES_FRAGMENT + "RETURN labels(n) AS labels, properties(n) AS properties\n"
)

# Only the properties any business key is made of (never the whole node).
_KEY_PROJECTION = "{.id, .uid, .name, .model, .label, .path, .tenant_id, .version}"

# Outgoing relationships of every subtree node. Targets outside the entry's
# node set (beyond the hop bound, other documents) are dropped by the caller;
# catalog targets are kept as references.
_SUBTREE_RELATIONSHIPS_EXPORT_QUERY = (
    _SUBTREE_NODES_FRAGMENT
    + f"""MATCH (n)-[r]->(m)
RETURN type(r) AS type,
       labels(n) AS from_labels, n {_KEY_PROJECTION} AS from_key,
       labels(m) AS to_labels, m {_KEY_PROJECTION} AS to_key,
       properties(r) AS properties
"""
)

# The Document's own relationships: HAS_STRUCTURE into its tree, and its
# links to other Documents of the tenant (folder IS_COMPOSED_OF, version chain
# HAS_NEWER_VERSION) in both directions — what restore_document() needs to
# re-link a Document it has to recreate (deleted, or restored from a backup).
_DOCUMENT_RELATIONSHIPS_EXPORT_QUERY = f"""
MATCH (d:Document {{tenant_id: $tenant_id, path: $path, version: $version}})
CALL (d) {{
  MATCH (d)-[r:HAS_STRUCTURE]->(m:StructureNode)
  RETURN d AS n, r, m
  UNION
  MATCH (d)-[r:IS_COMPOSED_OF|HAS_NEWER_VERSION]->(m:Document)
  WHERE m.tenant_id = $tenant_id
  RETURN d AS n, r, m
  UNION
  MATCH (n:Document)-[r:IS_COMPOSED_OF|HAS_NEWER_VERSION]->(d)
  WHERE n.tenant_id = $tenant_id
  RETURN n, r, d AS m
}}
RETURN type(r) AS type,
       labels(n) AS from_labels, n {_KEY_PROJECTION} AS from_key,
       labels(m) AS to_labels, m {_KEY_PROJECTION} AS to_key,
       properties(r) AS properties
"""


def _to_relationship(record: Any) -> RawRelationship:
    return RawRelationship(
        type=record["type"],
        from_labels=tuple(record["from_labels"]),
        from_key=record["from_key"],
        to_labels=tuple(record["to_labels"]),
        to_key=record["to_key"],
        properties=dict(record["properties"]),
    )


class _Neo4jExportReader:
    """:class:`ExportReader` over the open session of one ``export_snapshot`` attempt.

    Each method is a generator over the result of its query, so rows are handed
    out as the driver delivers them. A driver error raised while iterating is
    **not** translated here: it must reach the retry of ``export_snapshot`` raw.
    """

    def __init__(self, session: Any, tenant: str) -> None:
        self._session = session
        self._tenant = tenant

    def _params(self, doc: DocumentInfo) -> dict[str, Any]:
        return {"tenant_id": self._tenant, "path": doc.path, "version": doc.version}

    def subtree_nodes(self, doc: DocumentInfo) -> Iterator[RawNode]:
        for record in self._session.run(_SUBTREE_NODES_EXPORT_QUERY, **self._params(doc)):
            yield RawNode(labels=tuple(record["labels"]), properties=record["properties"])

    def document_relationships(self, doc: DocumentInfo) -> Iterator[RawRelationship]:
        for record in self._session.run(_DOCUMENT_RELATIONSHIPS_EXPORT_QUERY, **self._params(doc)):
            yield _to_relationship(record)

    def subtree_relationships(self, doc: DocumentInfo) -> Iterator[RawRelationship]:
        for record in self._session.run(_SUBTREE_RELATIONSHIPS_EXPORT_QUERY, **self._params(doc)):
            yield _to_relationship(record)


# ---------------------------------------------------------------------------
# Graph mutation — freeze (§7.3) and backup marking
# ---------------------------------------------------------------------------

# In-transaction guard: every document still there and none frozen yet
# (a concurrent freeze between the existence check and the mutation).
_FREEZE_GUARD_QUERY = (
    DOCS_BY_KEY + "RETURN count(d) AS found, count(CASE WHEN d.frozen = true THEN 1 END) AS frozen\n"
)

# frozen_cleanup_pending stays until the subtree is gone (see reduce_to_stubs).
_MARK_FROZEN_QUERY = (
    DOCS_BY_KEY
    + """SET d.frozen = true,
    d.frozen_blob_id = $blob_id,
    d.frozen_at = $frozen_at,
    d.frozen_keep_structure_nodes = $keep_structure_nodes,
    d.frozen_keep_annotations = $keep_annotations,
    d.frozen_keep_extraction_results = $keep_extraction_results,
    d.frozen_cleanup_pending = true
RETURN count(d) AS n
"""
)

_CLEAR_PENDING_QUERY = (
    DOCS_BY_KEY + "REMOVE d.frozen_cleanup_pending, d.deletion_pending\nRETURN count(d) AS n\n"
)

_MARK_BACKUP_QUERY = (
    DOCS_BY_KEY
    + """SET d.last_backup_blob_id = $blob_id, d.last_backup_at = $backup_at
RETURN count(d) AS n
"""
)

class Neo4jFreezeOps(FreezeOps):
    """``store.freeze`` over Neo4j. Every method opens its own sessions on the
    synchronous driver of the current ``sync_scope()`` (or a private one)."""

    @neo4j_errors()
    def fetch_document_set(self, selector: DocumentSelector) -> list[DocumentInfo]:
        """Resolve the selector to the matched Documents and their descendants.

        Args:
            selector: Filters selector; ``tenant_id`` is the *stored* key.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        match_clause, params = doc_match(selector)
        database = get_config().neo4j_database

        with borrow_sync_driver() as driver:

            def _do_query() -> list[DocumentInfo]:
                with driver.session(database=database) as session:
                    result = session.run(match_clause + _DOCUMENT_SET_TAIL, **params)
                    return [
                        DocumentInfo(
                            path=record["properties"].get("path"),
                            version=record["properties"].get("version"),
                            tenant_id=record["properties"].get("tenant_id"),
                            is_seed=record["is_seed"],
                            properties=dict(record["properties"]),
                        )
                        for record in result
                    ]

            return with_neo4j_retry_sync(_do_query)

    @neo4j_errors()
    def export_snapshot(
        self, docs: Sequence[DocumentInfo], tenant: str, attempt: Callable[[ExportReader], T]
    ) -> T:
        """Run *attempt* with an :class:`ExportReader` over one session, retrying it whole.

        The retry wraps opening the session **and** the callback (which opens
        the output file), so a retried attempt starts the snapshot over. An
        exception from *attempt* that is not a driver error passes through; a
        driver error raised while the reader is iterated is retried like any
        other and translated when it finally leaves this method.

        Args:
            docs: The documents to export (kept for the contract: the callback
                asks the reader about each one).
            tenant: Stored tenant key; every read is scoped to it.
            attempt: Reads from the reader and writes the snapshot.

        Raises:
            GraphError: If the engine fails after the retries.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:

            def _do_export() -> T:
                with driver.session(database=database) as session:
                    return attempt(_Neo4jExportReader(session, tenant))

            return with_neo4j_retry_sync(_do_export)

    @neo4j_errors()
    def mark_frozen(
        self,
        tenant: str,
        docs: Sequence[DocumentInfo],
        *,
        blob_id: str,
        frozen_at: str,
        keep_flags: KeepFlags,
    ) -> None:
        """Mark the stubs — in one write transaction, **before** anything is deleted.

        The subtree is then deleted in batches (:meth:`reduce_to_stubs`), which
        is not atomic: marking first means that whatever happens next, every
        document already points at the snapshot that holds its whole subtree.

        Args:
            tenant: Stored tenant key.
            docs: The documents to freeze.
            blob_id: The stored snapshot.
            frozen_at: ISO timestamp of the freeze.
            keep_flags: ``(keep_structure_nodes, keep_annotations, keep_extraction_results)``.

        Raises:
            FreezeError: If the documents changed since they were read (one is
                missing or already frozen); nothing is marked.
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        keys = [doc.key for doc in docs]
        base = {"tenant_id": tenant, "keys": keys}
        keep_structure_nodes, keep_annotations, keep_extraction_results = keep_flags

        with borrow_sync_driver() as driver:

            def _do_mark() -> None:
                with driver.session(database=database) as session:
                    with session.begin_transaction() as tx:
                        try:
                            guard = tx.run(_FREEZE_GUARD_QUERY, **base).single()
                            if guard["found"] != len(keys) or guard["frozen"]:
                                raise FreezeError(
                                    "The documents to freeze changed during the export "
                                    f"(expected {len(keys)} unfrozen, found {guard['found']} "
                                    f"with {guard['frozen']} already frozen); nothing was mutated."
                                )
                            tx.run(
                                _MARK_FROZEN_QUERY,
                                **base,
                                blob_id=blob_id,
                                frozen_at=frozen_at,
                                keep_structure_nodes=keep_structure_nodes,
                                keep_annotations=keep_annotations,
                                keep_extraction_results=keep_extraction_results,
                            ).single()
                            tx.commit()
                        except Exception:
                            tx.rollback()
                            logger.exception(
                                "freeze_document: stub marking rolled back (tenant=%r, keys=%r)",
                                tenant,
                                keys,
                            )
                            raise

            with_neo4j_retry_sync(_do_mark)

    @neo4j_errors()
    def mark_backup(
        self, tenant: str, docs: Sequence[DocumentInfo], blob_id: str, backup_at: str
    ) -> None:
        """Record ``last_backup_blob_id`` / ``last_backup_at`` on the documents.

        Args:
            tenant: Stored tenant key.
            docs: The documents backed up.
            blob_id: The stored snapshot.
            backup_at: ISO timestamp of the backup.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        keys = [doc.key for doc in docs]

        with borrow_sync_driver() as driver:

            def _do_mark() -> None:
                with driver.session(database=database) as session:
                    session.execute_write(
                        lambda tx: tx.run(
                            _MARK_BACKUP_QUERY,
                            tenant_id=tenant,
                            keys=keys,
                            blob_id=blob_id,
                            backup_at=backup_at,
                        ).consume()
                    )

            with_neo4j_retry_sync(_do_mark)

    @neo4j_errors()
    def reduce_to_stubs(
        self,
        tenant: str,
        docs: Sequence[DocumentInfo],
        keep_flags: KeepFlags,
        collector: OrphanCollector | None,
        *,
        caller: str,
    ) -> dict[str, int]:
        """Delete the subtree of *docs* that *keep_flags* do not keep, in bounded transactions.

        Only touches what is still there, so after a failure the same call
        finishes the job. *collector* (from ``store.gc.collector``) is handed,
        before each chunk is touched, what the ExtractionResults about to be
        deleted point at.

        Args:
            tenant: Stored tenant key.
            docs: Documents marked frozen with *keep_flags*.
            keep_flags: ``(keep_structure_nodes, keep_annotations, keep_extraction_results)``.
            collector: The orphan collector of the operation, or ``None``.
            caller: Public function name, used only as the log prefix.

        Returns:
            The summed subtree counters.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            return run_subtree_steps(
                driver,
                database,
                tenant,
                [doc.key for doc in docs],
                subtree_steps(*keep_flags),
                caller=caller,
                orphans=collector,
            )

    @neo4j_errors()
    def clear_cleanup_pending(self, tenant: str, docs: Sequence[DocumentInfo]) -> None:
        """Remove ``frozen_cleanup_pending`` / ``deletion_pending`` from *docs*,
        :data:`DELETE_BATCH_ROWS` per query. Goes last in a freeze, after the GC, so
        a failed GC is retried too.

        Args:
            tenant: Stored tenant key.
            docs: The documents whose cleanup is finished.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        keys = [doc.key for doc in docs]
        with borrow_sync_driver() as driver:
            for start in range(0, len(keys), DELETE_BATCH_ROWS):
                run_count_query(
                    driver,
                    database,
                    _CLEAR_PENDING_QUERY,
                    tenant_id=tenant,
                    keys=keys[start : start + DELETE_BATCH_ROWS],
                )
