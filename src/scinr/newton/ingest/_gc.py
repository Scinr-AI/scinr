"""
ingest/_gc.py — Garbage collection of orphaned extraction nodes.

Deleting an :ExtractionResult can orphan what hung from it: an :Entity /
:ModelInstance is an orphan when no :ExtractionResult reaches it any more
(within :data:`GC_REACHABILITY_MAX_HOPS` hops), a :LabeledEntity when nothing
points at it. Two collectors apply those two rules:

- :class:`OrphanCollector` — scoped to **one operation**. It is handed the
  nodes the operation may have orphaned (what the deleted :ExtractionResult
  nodes pointed at) and checks only those; deleting one makes the nodes it
  pointed at candidates in turn. Its cost follows what the operation deleted,
  not the size of the tenant. Used by
  :func:`~scinr.newton.ingest.deletion.delete_document` and
  :func:`~scinr.newton.ingest.freeze.freeze_document`.
- :func:`sweep_tenant` — scoped to **one tenant**. It checks every :Entity,
  :ModelInstance and :LabeledEntity of the tenant, so it also finds the
  orphans no operation knows about: the ones other write paths leave behind
  (re-ingestion with ``update_mode``, re-extraction), and the candidates of
  an operation that was interrupted — they only live in the memory of the
  process that died. It is what
  :func:`~scinr.newton.ingest.deletion.collect_orphans` runs, and what a
  resumed ``delete_document()`` / ``freeze_document()`` falls back to.

The two apply the same rules, with two differences of reach. The collector
only checks a node when something that pointed at it was deleted: a node that
hung from a deleted :ExtractionResult through a node that survives is left
alone, even in the corner case where the path it keeps to an
:ExtractionResult is longer than the hop bound (the sweep would delete it).
And the collector follows a chain of orphans to its end, where the sweep
stops after :data:`GC_MAX_PASSES` iterations.

Both are scoped to one tenant (``tenant_id = $tenant_id``, the stored key):
the subtree these operations remove belongs to a single tenant (no data
relationship crosses tenants), so only that tenant's nodes can have been
orphaned by them. Nodes without ``tenant_id`` (graphs written before
multitenancy) are never collected.

Every delete is bounded: the candidates go ``DELETE_BATCH_ROWS`` per write
transaction, and the sweep deletes with ``CALL { ... } IN TRANSACTIONS``
(which only runs as an auto-commit query, hence ``session.run()`` instead of
``execute_write()`` there).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from scinr.newton.ingest._cascade import DELETE_BATCH_ROWS, SUBTREE_STRUCTURE
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync

logger = logging.getLogger(__name__)

GC_MAX_PASSES = 7
"""Maximum number of iterations :func:`sweep_tenant` runs for each pass."""

GC_REACHABILITY_MAX_HOPS = 7
"""Hop bound of the "still reachable from an :ExtractionResult" check.

Not to be confused with :data:`GC_MAX_PASSES`. The same bound limits what
``export_document_snapshot()`` archives from each :ExtractionResult, so that
export, GC and ``keep_extraction_results`` agree on what hangs from one.
"""

GC_FLUSH_CANDIDATES = 100_000
"""Candidates an :class:`OrphanCollector` holds before it collects them.

Bounds the memory of the client on a large operation. Collecting early is
safe: a node still reachable from an :ExtractionResult not deleted yet is
left alone, and is a candidate again when that one goes.
"""

_NOT_REACHABLE = (
    f"NOT EXISTS {{ MATCH (:ExtractionResult)-[*1..{GC_REACHABILITY_MAX_HOPS}]->(mi) }}"
)
_NO_INCOMING = "NOT EXISTS { (mi)<--() }"

# ---------------------------------------------------------------------------
# Tenant-wide sweep
# ---------------------------------------------------------------------------

GC_ENTITY_MODEL_INSTANCE_QUERY = f"""
MATCH (mi:Entity|ModelInstance)
WHERE mi.tenant_id = $tenant_id AND {_NOT_REACHABLE}
CALL (mi) {{
  DETACH DELETE mi
}} IN TRANSACTIONS OF {DELETE_BATCH_ROWS} ROWS
RETURN count(*) AS borrados
"""

GC_LABELED_ENTITY_QUERY = f"""
MATCH (mi:LabeledEntity)
WHERE mi.tenant_id = $tenant_id AND {_NO_INCOMING}
CALL (mi) {{
  DETACH DELETE mi
}} IN TRANSACTIONS OF {DELETE_BATCH_ROWS} ROWS
RETURN count(*) AS borrados
"""

# ---------------------------------------------------------------------------
# Operation-scoped collection
# ---------------------------------------------------------------------------

# A label test the planner cannot start from: every candidate is reached
# from a node already in hand (an ExtractionResult, a deleted node), never
# looked up through the tenant's indexes — that lookup is the scan of the
# whole tenant these queries exist to avoid.
_IS_CANDIDATE = (
    "any(label IN labels(c) WHERE label IN ['Entity', 'ModelInstance', 'LabeledEntity'])"
    " AND c.tenant_id = $tenant_id"
)

# What the ExtractionResults of a set of Documents point at — read before
# they are deleted. One hop is enough: whatever hangs deeper becomes a
# candidate when the node it hangs from is collected.
GC_SEEDS_QUERY = (
    SUBTREE_STRUCTURE
    + f"""MATCH (s)-[:HAS_EXTRACTION]->(er:ExtractionResult)
WITH DISTINCT er
MATCH (er)-->(c)
WHERE {_IS_CANDIDATE}
RETURN DISTINCT elementId(c) AS id, c:LabeledEntity AS labeled
"""
)

# The orphans among $ids, in two steps: read them with what each points at
# (the next candidates), then delete them. Two steps because a write that is
# retried after a lost commit response finds nothing left to return: what the
# orphans pointed at would be lost with it. The delete checks again that the
# node is an orphan, so one linked in between is left alone.
_ENTITY_MODEL_INSTANCE_ORPHANS = f"""
MATCH (mi) WHERE elementId(mi) IN $ids
  AND any(label IN labels(mi) WHERE label IN ['Entity', 'ModelInstance'])
  AND mi.tenant_id = $tenant_id
  AND {_NOT_REACHABLE}
"""

_LABELED_ENTITY_ORPHANS = f"""
MATCH (mi) WHERE elementId(mi) IN $ids
  AND 'LabeledEntity' IN labels(mi)
  AND mi.tenant_id = $tenant_id
  AND {_NO_INCOMING}
"""

_READ_ORPHANS_TAIL = (
    f"RETURN elementId(mi) AS id,\n"
    f"       [(mi)-->(c) WHERE {_IS_CANDIDATE} | [elementId(c), c:LabeledEntity]] AS dependents\n"
)
_DELETE_ORPHANS_TAIL = "DETACH DELETE mi\nRETURN count(mi) AS n\n"

GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY = _ENTITY_MODEL_INSTANCE_ORPHANS + _READ_ORPHANS_TAIL
GC_ENTITY_MODEL_INSTANCE_DELETE_QUERY = _ENTITY_MODEL_INSTANCE_ORPHANS + _DELETE_ORPHANS_TAIL
GC_LABELED_ENTITY_ORPHANS_QUERY = _LABELED_ENTITY_ORPHANS + _READ_ORPHANS_TAIL
GC_LABELED_ENTITY_DELETE_QUERY = _LABELED_ENTITY_ORPHANS + _DELETE_ORPHANS_TAIL


class OrphanCollector:
    """Collects the orphans one operation leaves, from its candidates.

    ``gather()`` reads the candidates of a set of Documents **before** their
    :ExtractionResult nodes are deleted; ``chunk_done()`` tells the collector
    they are gone; ``collect()`` deletes the candidates that ended up
    orphaned, and then the orphans among what those pointed at, until a round
    deletes nothing.

    The candidates are element ids held in memory, so an operation that dies
    between deleting the :ExtractionResult nodes and ``collect()`` loses
    them: its caller must leave a mark in the graph first, and run
    :func:`sweep_tenant` instead when it finds the mark.

    Parameters
    ----------
    driver:
        An open, authenticated sync Neo4j driver instance.
    tenant_id:
        Stored tenant key (``tenant_key()`` applied — never ``None``).
    database:
        Neo4j database name.
    caller:
        Public function name, used only as the log prefix.
    """

    def __init__(self, driver, *, tenant_id: str, database: str, caller: str) -> None:
        self._driver = driver
        self._tenant = tenant_id
        self._database = database
        self._caller = caller
        self._entity_model_instance: set[str] = set()
        self._labeled_entity: set[str] = set()
        self._counts = {
            "gc_entity_model_instance_deleted": 0,
            "gc_entity_model_instance_passes": 0,
            "gc_labeled_entity_deleted": 0,
            "gc_labeled_entity_passes": 0,
        }

    def _add(self, element_id: str, labeled: bool) -> None:
        (self._labeled_entity if labeled else self._entity_model_instance).add(element_id)

    def gather(self, keys: Sequence[dict[str, Any]]) -> None:
        """Add what the :ExtractionResult nodes of the documents at *keys*
        point at. Call it before they are deleted."""

        def _do_query() -> list[tuple[str, bool]]:
            with self._driver.session(database=self._database) as session:
                result = session.run(GC_SEEDS_QUERY, tenant_id=self._tenant, keys=list(keys))
                return [(record["id"], record["labeled"]) for record in result]

        for element_id, labeled in with_neo4j_retry_sync(_do_query):
            self._add(element_id, labeled)

    def chunk_done(self) -> None:
        """The :ExtractionResult nodes of every gathered document are gone:
        collect now if enough candidates have piled up."""
        if len(self._entity_model_instance) + len(self._labeled_entity) >= GC_FLUSH_CANDIDATES:
            self.collect()

    def _run_round(self, read_query: str, delete_query: str, ids: set[str]) -> int:
        """Delete the orphans among *ids*; what they pointed at is added to
        the candidates. Returns how many were deleted."""
        ordered = sorted(ids)
        deleted = 0
        for start in range(0, len(ordered), DELETE_BATCH_ROWS):
            batch = ordered[start : start + DELETE_BATCH_ROWS]

            def _do_read(batch: list[str] = batch) -> list[tuple[str, list[list[Any]]]]:
                with self._driver.session(database=self._database) as session:
                    result = session.run(read_query, tenant_id=self._tenant, ids=batch)
                    return [(record["id"], record["dependents"]) for record in result]

            orphans = with_neo4j_retry_sync(_do_read)
            if not orphans:
                continue
            for _, dependents in orphans:
                for element_id, labeled in dependents:
                    self._add(element_id, labeled)
            orphan_ids = [element_id for element_id, _ in orphans]

            def _do_delete(orphan_ids: list[str] = orphan_ids) -> int:
                with self._driver.session(database=self._database) as session:
                    return session.execute_write(
                        lambda tx: tx.run(
                            delete_query, tenant_id=self._tenant, ids=orphan_ids
                        ).single()["n"]
                    )

            deleted += with_neo4j_retry_sync(_do_delete)
        return deleted

    def collect(self) -> None:
        """Delete every candidate that is an orphan, round after round."""
        rounds = (
            (
                "_entity_model_instance",
                GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY,
                GC_ENTITY_MODEL_INSTANCE_DELETE_QUERY,
                "gc_entity_model_instance",
                "Entity|ModelInstance",
            ),
            (
                "_labeled_entity",
                GC_LABELED_ENTITY_ORPHANS_QUERY,
                GC_LABELED_ENTITY_DELETE_QUERY,
                "gc_labeled_entity",
                "LabeledEntity",
            ),
        )
        while self._entity_model_instance or self._labeled_entity:
            for attribute, read_query, delete_query, counter, label in rounds:
                ids = getattr(self, attribute)
                if not ids:
                    continue
                setattr(self, attribute, set())
                deleted = self._run_round(read_query, delete_query, ids)
                self._counts[f"{counter}_deleted"] += deleted
                self._counts[f"{counter}_passes"] += 1
                logger.info(
                    "%s: GC (%s) deleted %d of %d candidate(s).",
                    self._caller,
                    label,
                    deleted,
                    len(ids),
                )

    def counts(self) -> dict[str, int]:
        """The ``gc_*`` counters shared by ``DeletionResult`` and ``FreezeResult``
        (``*_passes`` is the number of rounds run). A delete that had to be
        retried after its commit was lost counts nothing, so ``*_deleted`` is
        then a lower bound."""
        return dict(self._counts)


def run_gc_pass(
    driver,
    query: str,
    label: str,
    *,
    tenant_id: str,
    database: str,
    caller: str,
) -> tuple[int, int]:
    """Run a single tenant-wide garbage-collection query up to
    :data:`GC_MAX_PASSES` times, stopping as soon as an execution deletes
    zero nodes.

    Each individual execution is an auto-commit query that deletes in
    batches, wrapped in with_neo4j_retry_sync. An execution retried after
    some of its batches were committed only counts what was left, so after a
    retry the total is a lower bound.

    Parameters
    ----------
    driver:
        An open, authenticated sync Neo4j driver instance.
    query:
        :data:`GC_ENTITY_MODEL_INSTANCE_QUERY` or :data:`GC_LABELED_ENTITY_QUERY`
        (must return a single ``borrados`` count).
    label:
        Human-readable label used only for logging.
    tenant_id:
        Stored tenant key (``tenant_key()`` applied — never ``None``).
    database:
        Neo4j database name.
    caller:
        Public function name, used only as the log prefix.

    Returns
    -------
    tuple[int, int]
        (total nodes deleted across all iterations, number of iterations run).
    """
    total_deleted = 0
    passes_run = 0

    def _do_gc_iteration() -> int:
        try:
            with driver.session(database=database) as session:
                return session.run(query, tenant_id=tenant_id).single()["borrados"]
        except Exception:
            logger.exception(
                "%s: GC pass (%s) iteration %d raised an exception.",
                caller,
                label,
                passes_run + 1,
            )
            raise

    for _ in range(GC_MAX_PASSES):
        deleted = with_neo4j_retry_sync(_do_gc_iteration)
        passes_run += 1
        total_deleted += deleted
        logger.info(
            "%s: GC pass (%s) iteration %d deleted %d node(s).",
            caller,
            label,
            passes_run,
            deleted,
        )
        if deleted == 0:
            break

    return total_deleted, passes_run


def sweep_tenant(driver, *, tenant_id: str, database: str, caller: str) -> dict[str, int]:
    """Collect every orphan of the tenant: both passes in order
    (:Entity/:ModelInstance first, then :LabeledEntity, which the first pass
    can orphan), each over every node of the tenant.

    Returns the ``gc_*`` counters shared by ``DeletionResult``,
    ``FreezeResult``, ``RestoreResult`` and ``OrphanCollectionResult``.
    """
    emi_deleted, emi_passes = run_gc_pass(
        driver,
        GC_ENTITY_MODEL_INSTANCE_QUERY,
        "Entity|ModelInstance",
        tenant_id=tenant_id,
        database=database,
        caller=caller,
    )
    le_deleted, le_passes = run_gc_pass(
        driver,
        GC_LABELED_ENTITY_QUERY,
        "LabeledEntity",
        tenant_id=tenant_id,
        database=database,
        caller=caller,
    )
    return {
        "gc_entity_model_instance_deleted": emi_deleted,
        "gc_entity_model_instance_passes": emi_passes,
        "gc_labeled_entity_deleted": le_deleted,
        "gc_labeled_entity_passes": le_passes,
    }
