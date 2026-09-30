"""
ingest/_cascade.py — Batched deletion of the subtree of a set of Documents.

Shared by :func:`~scinr.newton.ingest.deletion.delete_document` and
:func:`~scinr.newton.ingest.freeze.freeze_document`. Deleting the subtree of
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
:func:`~scinr.newton.utils.neo4j_retry.with_neo4j_retry_sync`. A retried
delete only counts what was left to delete, so after a retry the counters are
a lower bound.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync

if TYPE_CHECKING:
    from scinr.newton.ingest._gc import OrphanCollector

logger = logging.getLogger(__name__)

DELETE_BATCH_ROWS = 1000
"""Nodes deleted per write transaction."""

DOCUMENTS_PER_QUERY = 50
"""Documents whose subtree one query walks."""

# Every query below starts from explicit (tenant_id, path, version) keys: the
# document set resolved before any mutation.
DOCS_BY_KEY = """
UNWIND $keys AS k
MATCH (d:Document {tenant_id: $tenant_id, path: k.path, version: k.version})
"""

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


def run_count_query(driver, database: str, query: str, **params: Any) -> int:
    """Run *query* as an auto-commit query and return its ``n`` (0 when it
    returns nothing), retrying transient errors."""

    def _do_query() -> int:
        with driver.session(database=database) as session:
            record = session.run(query, **params).single()
            return record["n"] if record else 0

    return with_neo4j_retry_sync(_do_query)


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
        point at — the nodes the chunk can orphan (``ingest/_gc.py``).
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
