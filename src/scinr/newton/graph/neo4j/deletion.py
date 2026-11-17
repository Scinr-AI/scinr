"""
graph/neo4j/deletion.py — Document deletion over Neo4j: selection, batched cascade, marks.

:class:`Neo4jDeletionOps` (``store.deletion``) holds the Cypher of
``delete_document()``: the selection of the documents (a
:class:`~scinr.newton.graph.types.DocumentSelector` translated to a ``MATCH``),
the existence and raw-file queries, the ``deletion_pending`` mark and the batched
cascade below. The validation and normalization of the filters, and the
orchestration (storage first, then the graph, then the orphans), stay in
``ingest/deletion.py``.

The cascade is shared by ``delete_document()`` and ``freeze_document()``. Deleting the subtree of
hundreds of documents in one transaction keeps every deleted node and
relationship in that transaction's state, which overruns the server's
transaction memory pool (``dbms.memory.transaction.total.max``, shared by
every transaction of the DBMS). So the subtree goes in bounded pieces:

- the documents are processed :data:`DOCUMENTS_PER_QUERY` at a time, which
  bounds what each query reads and collects;
- every delete runs as ``CALL { ... } IN TRANSACTIONS OF`` :data:`DELETE_BATCH_ROWS`
  ``ROWS``, which bounds what each write transaction holds.

The price is atomicity: a failure leaves part of the subtree deleted. Every
step only deletes what is still there, so the callers make the operation
resumable instead (``delete_document()`` deletes the Documents last;
``freeze_document()`` marks the stubs first).

``CALL ... IN TRANSACTIONS`` only runs as an auto-commit query, so nothing
here goes through ``execute_write()``: each query is retried as a whole with
:func:`~scinr.newton.graph.neo4j.retry.with_neo4j_retry_sync`. A retried
delete only counts what was left to delete, so after a retry the counters are
a lower bound.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from scinr.newton.config import get_config
from scinr.newton.graph.base import DeletionOps
from scinr.newton.graph.neo4j._scope import borrow_sync_driver
from scinr.newton.graph.neo4j._shared import (
    DELETE_BATCH_ROWS,
    DOCS_BY_KEY,
    DOCUMENTS_PER_QUERY,
    run_count_query,
)
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.retry import with_neo4j_retry_sync
from scinr.newton.graph.types import DocumentSelector
from scinr.newton.results import DeletedDocument

if TYPE_CHECKING:
    from scinr.newton.graph.base import OrphanCollector

logger = logging.getLogger(__name__)

# Every query below starts from explicit (tenant_id, path, version) keys
# (``DOCS_BY_KEY``): the document set resolved before any mutation.

SUBTREE_STRUCTURE = (
    DOCS_BY_KEY
    + "MATCH (d)-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)\n"
    + "WITH DISTINCT d, s\n"
)


def delete_in_batches(alias: str) -> str:
    """Tail deleting every distinct node bound to *alias*, in batches.

    The batches commit while the query is still running, so every node to
    delete must be known before the first one goes: deleting a StructureNode
    cuts the path to its children. Grouping by the node (the ``count(*)`` is
    not used) is an aggregation, which reads its whole input before it
    returns a row. Do not ``collect()`` the nodes and ``UNWIND`` the list
    instead: every row handed to the batches then carries — and is charged
    for — the whole list, which is what overran the memory pool in tests.
    """
    return (
        f"WITH {alias} AS x, count(*) AS hits\n"
        "CALL (x) {\n"
        "  DETACH DELETE x\n"
        f"}} IN TRANSACTIONS OF {DELETE_BATCH_ROWS} ROWS\n"
        "RETURN count(*) AS n\n"
    )


def subtree_query(pattern: str, action: str) -> str:
    """``pattern`` binds ``x`` from ``s``; ``action`` is ``"delete"`` or ``"count"``."""
    body = SUBTREE_STRUCTURE + f"MATCH (s){pattern}\n"
    if action == "delete":
        return body + delete_in_batches("x")
    return body + "RETURN count(DISTINCT x) AS n\n"


PROPOSED_FIELD_PATTERN = (
    "-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(:ProposedModel)"
    "-[:HAS_PROPOSED_FIELD]->(x:ProposedField)"
)
PROPOSED_MODEL_PATTERN = (
    "-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(x:ProposedModel)"
)
COMPLEMENTARY_MATCH_PATTERN = (
    "-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_COMPLEMENTARY_MATCH]->(x:ComplementaryMatch)"
)
SUPPLEMENTARY_FIELD_PATTERN = (
    "-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_SUPPLEMENTARY_FIELD]->(x:SupplementaryField)"
)
MODEL_DECISION_PATTERN = "-[:HAS_MODEL_DECISION]->(x:ModelDecision)"
EXTRACTION_RESULT_PATTERN = "-[:HAS_EXTRACTION]->(x:ExtractionResult)"
INFO_UNIT_PATTERN = "-[:HAS_INFO_UNIT]->(x:InfoUnit)"

STRUCTURE_NODES_DELETE_QUERY = SUBTREE_STRUCTURE + delete_in_batches("s")
STRUCTURE_NODES_COUNT_QUERY = SUBTREE_STRUCTURE + "RETURN count(DISTINCT s) AS n\n"

# The Documents themselves: few and, by then, bare — one plain transaction.
DOCUMENTS_DELETE_QUERY = DOCS_BY_KEY + "DETACH DELETE d\nRETURN count(d) AS n\n"

# Direct links kept ModelDecision / ExtractionResult nodes hang from while
# their StructureNode is gone (keep_structure_nodes=False). MERGE: one link
# per node even for the tabular ModelDecision shared by every row.
RELINK_MODEL_DECISIONS_QUERY = (
    SUBTREE_STRUCTURE
    + "MATCH (s)-[:HAS_MODEL_DECISION]->(md:ModelDecision)\n"
    + "WITH DISTINCT d, md\n"
    + "MERGE (d)-[:HAS_MODEL_DECISION]->(md)\n"
)

RELINK_EXTRACTION_RESULTS_QUERY = (
    SUBTREE_STRUCTURE
    + "MATCH (s)-[:HAS_EXTRACTION]->(er:ExtractionResult)\n"
    + "WITH DISTINCT d, er\n"
    + "MERGE (d)-[:HAS_EXTRACTION]->(er)\n"
)

SUBTREE_COUNTER_FIELDS = (
    "structure_nodes_deleted",
    "structure_nodes_kept",
    "info_units_deleted",
    "model_decisions_deleted",
    "model_decisions_kept",
    "proposed_models_deleted",
    "proposed_fields_deleted",
    "extraction_results_deleted",
    "extraction_results_kept",
)


def subtree_steps(
    keep_structure_nodes: bool = False,
    keep_annotations: bool = False,
    keep_extraction_results: bool = False,
) -> list[tuple[str | None, str]]:
    """Ordered ``(counter, query)`` steps removing the subtree of a Document.

    Children before parents (a deleted parent can no longer lead to them),
    and the re-links before the StructureNodes they hang from are deleted.
    Every step is idempotent, so the whole list can be run again after a
    failure. With no ``keep_*`` flag the whole subtree goes.
    """
    steps: list[tuple[str | None, str]] = []
    if not keep_structure_nodes:
        if keep_annotations:
            steps.append((None, RELINK_MODEL_DECISIONS_QUERY))
        if keep_extraction_results:
            steps.append((None, RELINK_EXTRACTION_RESULTS_QUERY))
    if keep_annotations:
        steps.append(("model_decisions_kept", subtree_query(MODEL_DECISION_PATTERN, "count")))
    else:
        steps += [
            ("proposed_fields_deleted", subtree_query(PROPOSED_FIELD_PATTERN, "delete")),
            ("proposed_models_deleted", subtree_query(PROPOSED_MODEL_PATTERN, "delete")),
            (None, subtree_query(COMPLEMENTARY_MATCH_PATTERN, "delete")),
            (None, subtree_query(SUPPLEMENTARY_FIELD_PATTERN, "delete")),
            ("model_decisions_deleted", subtree_query(MODEL_DECISION_PATTERN, "delete")),
        ]
    if keep_extraction_results:
        steps.append(
            ("extraction_results_kept", subtree_query(EXTRACTION_RESULT_PATTERN, "count"))
        )
    else:
        steps.append(
            ("extraction_results_deleted", subtree_query(EXTRACTION_RESULT_PATTERN, "delete"))
        )
    steps.append(("info_units_deleted", subtree_query(INFO_UNIT_PATTERN, "delete")))
    if keep_structure_nodes:
        steps.append(("structure_nodes_kept", STRUCTURE_NODES_COUNT_QUERY))
    else:
        steps.append(("structure_nodes_deleted", STRUCTURE_NODES_DELETE_QUERY))
    return steps


def run_subtree_steps(
    driver,
    database: str,
    tenant: str,
    keys: Sequence[dict[str, Any]],
    steps: Sequence[tuple[str | None, str]],
    *,
    caller: str,
    orphans: OrphanCollector | None = None,
) -> dict[str, int]:
    """Run *steps* over the documents at *keys*, :data:`DOCUMENTS_PER_QUERY`
    documents at a time, and return the summed counters.

    Parameters
    ----------
    tenant:
        Stored tenant key (``tenant_key()`` applied — never ``None``).
    keys:
        ``{"path", "version"}`` of every document.
    steps:
        The output of :func:`subtree_steps`.
    caller:
        Public function name, used only as the log prefix.
    orphans:
        When *steps* delete the ExtractionResults: the collector that is
        handed, before each chunk is touched, what those ExtractionResults
        point at — the nodes the chunk can orphan (``graph/neo4j/gc.py``).
    """
    counters = dict.fromkeys(SUBTREE_COUNTER_FIELDS, 0)
    for start in range(0, len(keys), DOCUMENTS_PER_QUERY):
        chunk = list(keys[start : start + DOCUMENTS_PER_QUERY])
        if orphans is not None:
            orphans.gather(chunk)
        for counter, query in steps:
            count = run_count_query(driver, database, query, tenant_id=tenant, keys=chunk)
            if counter is not None:
                counters[counter] += count
        if orphans is not None:
            orphans.chunk_done()
        logger.info(
            "%s: subtree of %d/%d document(s) processed.",
            caller,
            start + len(chunk),
            len(keys),
        )
    return counters


def delete_documents(driver, database: str, tenant: str, keys: Sequence[dict[str, Any]]) -> int:
    """``DETACH DELETE`` the Documents at *keys*, :data:`DELETE_BATCH_ROWS`
    per write transaction, in the order given."""
    deleted = 0
    for start in range(0, len(keys), DELETE_BATCH_ROWS):
        chunk = list(keys[start : start + DELETE_BATCH_ROWS])

        def _do_delete(chunk: list[dict[str, Any]] = chunk) -> int:
            with driver.session(database=database) as session:
                return session.execute_write(
                    lambda tx: tx.run(DOCUMENTS_DELETE_QUERY, tenant_id=tenant, keys=chunk).single()[
                        "n"
                    ]
                )

        deleted += with_neo4j_retry_sync(_do_delete)
    return deleted


# ---------------------------------------------------------------------------
# Document selection (the "filters" selector)
# ---------------------------------------------------------------------------

# The DocumentSelector fields that optionally narrow the :Document selection,
# in a fixed order. Each name is also the node property it filters on. The names
# are a hard-coded whitelist — never user input — so interpolating them into the
# query string below carries no injection risk. tenant_id is deliberately NOT
# here: it is never optional. ``job_id`` and ``created_by_user_id`` match with
# ``IN`` (a list of values); the others with equality.
_IN_FIELDS = frozenset({"job_id", "created_by_user_id"})
_FILTERABLE_FIELDS = ("path", "job_id", "version", "created_by_user_id")


def doc_match(selector: DocumentSelector) -> tuple[str, dict]:
    """Build the ``MATCH (d:Document) WHERE ...`` selection prefix, together
    with the matching parameter dict.

    The tenant condition is **always** emitted: ``selector.tenant_id`` must
    hold the stored tenant key (``utils.tenancy.tenant_key()`` applied, so
    never ``None`` — ``"__public__"`` selects public documents). Every other
    filter produces a condition only when it is set (not ``None``).

    Emitting a condition **only** for each supplied filter keeps the WHERE a
    plain conjunction of equality predicates, so Neo4j can use the
    :Document indexes (the ``(tenant_id, path, version)`` constraint index,
    ``idx_document_job_id``) instead of falling back to a full label scan —
    which an ``$x IS NULL OR d.x = $x`` disjunction would force.

    ``delete_document()`` guarantees at least one of ``path`` / ``job_id`` is
    set on top of the tenant.

    Raises:
        ValueError: If the selector carries no tenant.
    """
    tenant = selector.tenant_id
    if tenant is None:
        raise ValueError("doc_match requires the stored tenant key in selector.tenant_id.")
    conditions: list[str] = ["d.tenant_id = $tenant_id"]
    params: dict = {"tenant_id": tenant}
    for name in _FILTERABLE_FIELDS:
        value = getattr(selector, name)
        if value is not None:
            op = "IN" if name in _IN_FIELDS else "="
            conditions.append(f"d.{name} {op} ${name}")
            # A single value is accepted for the any-of fields; a list goes
            # through untouched.
            params[name] = [value] if name in _IN_FIELDS and isinstance(value, str) else value
    return f"MATCH (d:Document)\nWHERE {' AND '.join(conditions)}\n", params


_EXISTENCE_TAIL = "RETURN d.version AS version\n"

_RAW_FILE_IDS_TAIL = """
OPTIONAL MATCH (d)-[:IS_COMPOSED_OF*]->(cd)
WITH collect(DISTINCT d) + collect(DISTINCT cd) AS nodes
UNWIND nodes AS n
WITH DISTINCT n
WHERE n IS NOT NULL AND n.raw_file_id IS NOT NULL AND n.raw_file_id <> ''
RETURN DISTINCT n.raw_file_id AS raw_file_id, n.tenant_id AS tenant_id
"""

# The documents the cascade works on: the matched ones and their descendants.
# ``interrupted``: an earlier delete_document() or freeze_document() stopped
# half-way through this document.
_DOCUMENT_KEYS_TAIL = """
OPTIONAL MATCH (d)-[:IS_COMPOSED_OF*]->(cd:Document)
WITH collect(DISTINCT d) AS seeds, collect(DISTINCT cd) AS descendants
UNWIND seeds + descendants AS doc
WITH DISTINCT doc, seeds
RETURN doc.path AS path, doc.version AS version, doc.tenant_id AS tenant_id,
       coalesce(doc.is_folder, false) AS is_folder,
       doc IN seeds AS is_seed,
       coalesce(doc.deletion_pending, doc.frozen_cleanup_pending, false) AS interrupted
"""

# Set before anything is deleted, and gone with the Document itself: a
# Document that still carries it was left half-deleted.
_MARK_PENDING_QUERY = DOCS_BY_KEY + "SET d.deletion_pending = true\nRETURN count(d) AS n\n"


# ---------------------------------------------------------------------------
# The store group
# ---------------------------------------------------------------------------


class Neo4jDeletionOps(DeletionOps):
    """``store.deletion`` over Neo4j. Every method opens its own sessions on the
    synchronous driver of the current ``sync_scope()`` (or a private one)."""

    @neo4j_errors()
    def fetch_document_keys(
        self, selector: DocumentSelector
    ) -> tuple[list[dict], list[DeletedDocument], bool]:
        """Resolve the selector to the ``{"path", "version"}`` keys of the matched
        Documents and their ``IS_COMPOSED_OF*`` descendants — descendants first,
        matched documents last (the order they are deleted in) —, the same
        documents as :class:`DeletedDocument` (with ``is_folder``), and whether an
        earlier delete or freeze of any of them was interrupted.

        A Document that cannot be addressed by its key — no ``path`` /
        ``version``, or another tenant's (impossible under the write invariant) —
        is left alone, with a warning.

        Args:
            selector: Filters selector; ``tenant_id`` is the *stored* key.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        tenant = selector.tenant_id
        match_clause, params = doc_match(selector)

        with borrow_sync_driver() as driver:

            def _do_query() -> list[dict]:
                cfg = get_config()
                with driver.session(database=cfg.neo4j_database) as session:
                    result = session.run(match_clause + _DOCUMENT_KEYS_TAIL, **params)
                    return [dict(record) for record in result]

            rows = with_neo4j_retry_sync(_do_query)
        addressable = [
            row
            for row in rows
            if row["path"] is not None
            and row["version"] is not None
            and row["tenant_id"] == tenant
        ]
        if len(addressable) != len(rows):
            logger.warning(
                "delete_document: %d Document(s) reached by the selector have no path/version "
                "or belong to another tenant than %r; they are left untouched.",
                len(rows) - len(addressable),
                tenant,
            )
        addressable.sort(key=lambda row: row["is_seed"])
        keys = [{"path": row["path"], "version": row["version"]} for row in addressable]
        documents = [
            DeletedDocument(
                path=row["path"], version=row["version"], is_folder=bool(row["is_folder"])
            )
            for row in addressable
        ]
        return keys, documents, any(row["interrupted"] for row in addressable)

    @neo4j_errors()
    def fetch_existing_versions(self, selector: DocumentSelector) -> list[int]:
        """Run the read-only existence-check query and return the sorted list of
        matching Document versions.

        Retried like the cascade and GC queries, so a transient error on this
        first query is handled the same way as the rest of ``delete_document()``.

        Any ``None`` version values (from legacy/malformed :Document nodes) are
        filtered out defensively before sorting, since Python 3 cannot compare
        ``None`` with ``int``.

        Args:
            selector: Filters selector; ``tenant_id`` is the *stored* key.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        match_clause, params = doc_match(selector)

        with borrow_sync_driver() as driver:

            def _do_query() -> list[int | None]:
                cfg = get_config()
                with driver.session(database=cfg.neo4j_database) as session:
                    existence_result = session.run(match_clause + _EXISTENCE_TAIL, **params)
                    return [record["version"] for record in existence_result]

            raw_versions = with_neo4j_retry_sync(_do_query)
        return sorted(v for v in raw_versions if v is not None)

    @neo4j_errors()
    def fetch_raw_file_ids(self, selector: DocumentSelector) -> list[tuple[str, str | None]]:
        """Run the read-only raw_file_ids query and return the distinct
        ``(raw_file_id, tenant_id)`` pairs — non-empty ``raw_file_id`` values and
        the stored tenant of the node carrying each — for the target Document(s)
        and every descendant reached via ``IS_COMPOSED_OF*``: the same scope the
        cascade works on.

        Args:
            selector: Filters selector; ``tenant_id`` is the *stored* key.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        match_clause, params = doc_match(selector)

        with borrow_sync_driver() as driver:

            def _do_query() -> list[tuple[str, str | None]]:
                cfg = get_config()
                with driver.session(database=cfg.neo4j_database) as session:
                    result = session.run(match_clause + _RAW_FILE_IDS_TAIL, **params)
                    return [(record["raw_file_id"], record["tenant_id"]) for record in result]

            return with_neo4j_retry_sync(_do_query)

    @neo4j_errors()
    def mark_deletion_pending(self, tenant: str, keys: Sequence[dict[str, Any]]) -> None:
        """Mark the documents at *keys* ``deletion_pending``, :data:`DELETE_BATCH_ROWS`
        per query, before anything is deleted.

        Args:
            tenant: Stored tenant key.
            keys: ``{"path", "version"}`` of every document.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            for start in range(0, len(keys), DELETE_BATCH_ROWS):
                run_count_query(
                    driver,
                    database,
                    _MARK_PENDING_QUERY,
                    tenant_id=tenant,
                    keys=keys[start : start + DELETE_BATCH_ROWS],
                )

    @neo4j_errors()
    def cascade_delete(
        self,
        tenant: str,
        keys: Sequence[dict[str, Any]],
        collector: OrphanCollector | None,
        *,
        caller: str,
    ) -> dict[str, int]:
        """Delete the whole subtree of the documents at *keys* (the Documents
        themselves stay), in bounded transactions, and return the summed counters.

        Not atomic: every step only deletes what is still there, so the call can
        be repeated after a failure. *collector* (from ``store.gc.collector``)
        is handed, before each chunk is touched, what the ExtractionResults about
        to be deleted point at.

        Args:
            tenant: Stored tenant key.
            keys: ``{"path", "version"}`` of every document.
            collector: The orphan collector of the operation, or ``None``.
            caller: Public function name, used only as the log prefix.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            return run_subtree_steps(
                driver,
                database,
                tenant,
                keys,
                subtree_steps(),
                caller=caller,
                orphans=collector,
            )

    @neo4j_errors()
    def delete_documents(self, tenant: str, keys: Sequence[dict[str, Any]]) -> int:
        """``DETACH DELETE`` the Documents at *keys*, :data:`DELETE_BATCH_ROWS` per
        write transaction, in the order given. Returns how many were deleted.

        Args:
            tenant: Stored tenant key.
            keys: ``{"path", "version"}`` of every document, descendants first.

        Raises:
            GraphError: If the engine fails.
        """
        database = get_config().neo4j_database
        with borrow_sync_driver() as driver:
            return delete_documents(driver, database, tenant, keys)
