"""
graph/neo4j/_shared.py — Cypher steps shared by several Neo4j operation groups.

These are internal steps, not part of the :class:`~scinr.newton.graph.base.GraphStore`
interface: they receive the ``runner`` (an ``AsyncSession`` or
``AsyncTransaction``) of the method that calls them, so they run inside that
method's session or transaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scinr.newton.graph.neo4j.retry import with_neo4j_retry_sync

DELETE_BATCH_ROWS = 1000
"""Nodes deleted per write transaction (deletion, GC, freeze, restore)."""

DOCUMENTS_PER_QUERY = 50
"""Documents whose subtree one query walks."""

# Every bulk query of deletion, freeze and restore starts from explicit
# (tenant_id, path, version) keys: the document set resolved before any mutation.
DOCS_BY_KEY = """
UNWIND $keys AS k
MATCH (d:Document {tenant_id: $tenant_id, path: k.path, version: k.version})
"""


def run_count_query(driver, database: str, query: str, **params: Any) -> int:
    """Run *query* as an auto-commit query and return its ``n`` (0 when it
    returns nothing), retrying transient errors."""

    def _do_query() -> int:
        with driver.session(database=database) as session:
            record = session.run(query, **params).single()
            return record["n"] if record else 0

    return with_neo4j_retry_sync(_do_query)

# Leaf-first deletion of the ModelDecision subtree hanging from StructureNodes
# (ProposedField → ProposedModel → SupplementaryField → ComplementaryMatch →
# ModelDecision). :CatalogModel singletons are never deleted.
_STALE_MODEL_DECISION_DELETE_QUERIES = (
    """
    UNWIND $node_ids AS node_id
    MATCH (:StructureNode {id: node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
          -[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)-[:HAS_PROPOSED_FIELD]->(pf)
    DETACH DELETE pf
    """,
    """
    UNWIND $node_ids AS node_id
    MATCH (:StructureNode {id: node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
          -[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)
    DETACH DELETE pm
    """,
    """
    UNWIND $node_ids AS node_id
    MATCH (:StructureNode {id: node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
          -[:HAS_SUPPLEMENTARY_FIELD]->(sf)
    DETACH DELETE sf
    """,
    """
    UNWIND $node_ids AS node_id
    MATCH (:StructureNode {id: node_id})
          -[:HAS_MODEL_DECISION]->(:ModelDecision)
          -[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
    DETACH DELETE cm
    """,
    """
    UNWIND $node_ids AS node_id
    MATCH (:StructureNode {id: node_id})-[:HAS_MODEL_DECISION]->(md:ModelDecision)
    DETACH DELETE md
    """,
)


async def delete_stale_model_decision(runner, node_ids: str | Sequence[str]) -> None:
    """Delete the ModelDecision subtree(s) hanging from the given StructureNode(s).

    The idempotency step of every ModelDecision writer (write_annotation(),
    write_manual_annotation(), the tabular subgraph delete) and of
    restore_document(): the stale decision is removed before a new one is
    created. A ModelDecision shared by several StructureNodes (the tabular
    one, shared by every row) goes as soon as one of them is passed.

    Parameters
    ----------
    runner:
        An ``AsyncSession`` or ``AsyncTransaction`` (anything with an async ``run``).
    node_ids:
        One ``StructureNode.id`` (tenant-prefixed) or several.
    """
    ids = [node_ids] if isinstance(node_ids, str) else list(node_ids)
    for query in _STALE_MODEL_DECISION_DELETE_QUERIES:
        await runner.run(query, node_ids=ids)


_STALE_EXTRACTION_RESULT_DELETE_QUERY = """
UNWIND $node_ids AS nid
MATCH (n:StructureNode {id: nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
OPTIONAL MATCH (er)-[*1..10]->(child:ModelInstance)
DETACH DELETE child
WITH DISTINCT er
DETACH DELETE er
"""


async def delete_stale_extraction_result(runner, node_ids: str | Sequence[str]) -> None:
    """Delete the ExtractionResult(s) hanging from the given StructureNode(s),
    together with every :ModelInstance reachable from them within 10 hops.

    The idempotency step of write_extraction_subgraph(),
    write_manual_annotation() and restore_document(). Note that the
    ModelInstance purge does not check whether another live ExtractionResult
    still reaches the instance (long-standing pipeline behaviour); the GC
    passes of ``graph/neo4j/gc.py`` clean up whatever else is left orphaned.

    Parameters
    ----------
    runner:
        An ``AsyncSession`` or ``AsyncTransaction`` (anything with an async ``run``).
    node_ids:
        One ``StructureNode.id`` (tenant-prefixed) or several.
    """
    ids = [node_ids] if isinstance(node_ids, str) else list(node_ids)
    await runner.run(_STALE_EXTRACTION_RESULT_DELETE_QUERY, node_ids=ids)


def provenance_set_clause(alias: str) -> str:
    """SET-clause fragment for a MERGE'd node: scalar ``tenant_id`` plus
    accumulate ``created_by_user_id`` / ``job_id`` into deduplicated arrays.

    Used identically in ``ON CREATE`` and ``ON MATCH`` branches — re-affirming
    ``tenant_id`` on every MATCH is harmless since it is already folded into the
    node's ``uid`` (a MATCH only ever happens within the same tenant). A ``None``
    provenance value is skipped rather than pushed into the array.
    """
    return (
        f"{alias}.tenant_id = $tenant_id, "
        f"{alias}.created_by_user_ids = CASE "
        f"WHEN $created_by_user_id IS NULL THEN coalesce({alias}.created_by_user_ids, []) "
        f"WHEN $created_by_user_id IN coalesce({alias}.created_by_user_ids, []) THEN {alias}.created_by_user_ids "
        f"ELSE coalesce({alias}.created_by_user_ids, []) + $created_by_user_id END, "
        f"{alias}.job_ids = CASE "
        f"WHEN $job_id IS NULL THEN coalesce({alias}.job_ids, []) "
        f"WHEN $job_id IN coalesce({alias}.job_ids, []) THEN {alias}.job_ids "
        f"ELSE coalesce({alias}.job_ids, []) + $job_id END"
    )


# ---------------------------------------------------------------------------
# Core writer
