"""
ingest/deletion.py — Full document deletion (Document node + cascade + GC).

Unlike ``delete_document_content()`` in ``ingest/nodes.py`` (which only wipes
structure/annotation data for a single version to support in-place
re-ingestion via ``--update``, keeping the :Document node itself), the
public :func:`delete_document` here removes the :Document node(s) as well
as their entire composed/structural subtree, and then runs a two-pass
global garbage collector to remove any resulting orphaned :Entity,
:ModelInstance, and :LabeledEntity nodes.

Public API
----------
    result = delete_document(path, version=None)  # sync, opens its own driver
"""

from __future__ import annotations

import logging

from scinr.newton.ingest.config import get_driver
from scinr.newton.results import DeletionResult
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync

logger = logging.getLogger(__name__)

GC_MAX_PASSES = 7
"""Maximum number of iterations run for each garbage-collection pass."""


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_EXISTENCE_QUERY = """
MATCH (d:Document {path: $path})
WHERE $version IS NULL OR d.version = $version
RETURN d.version AS version
"""

_CASCADE_DELETE_QUERY = """
MATCH (d:Document {path: $path})
WHERE $version IS NULL OR d.version = $version
OPTIONAL MATCH (d)-[r:IS_COMPOSED_OF*]->(cd)
WITH d, r, collect(DISTINCT cd) + d AS nodes
UNWIND nodes AS documentNode
OPTIONAL MATCH (documentNode)-[rdps:HAS_STRUCTURE*..]->(parentStructureNode)
OPTIONAL MATCH (parentStructureNode)-[rpscs:HAS_CHILD*..]->(childStructureNode)
WITH nodes, documentNode, r, rdps, rpscs, collect(DISTINCT childStructureNode) + collect(DISTINCT parentStructureNode) AS structureNodes
UNWIND structureNodes AS structureNode
OPTIONAL MATCH (structureNode)-[rsiu:HAS_INFO_UNIT]->(iu)
OPTIONAL MATCH (structureNode)-[rsmd:HAS_MODEL_DECISION]->(md)
OPTIONAL MATCH (md)-[rmdpm:HAS_PROPOSED_MODEL]-(pm)
OPTIONAL MATCH (pm)-[rpmpf:HAS_PROPOSED_FIELD]->(pf)
OPTIONAL MATCH (structureNode)-[rse:HAS_EXTRACTION]->(e)
DETACH DELETE documentNode, structureNode, iu, md, pm, pf, rse, e
RETURN
  count(DISTINCT documentNode) AS documents_deleted,
  count(DISTINCT structureNode) AS structure_nodes_deleted,
  count(DISTINCT iu) AS info_units_deleted,
  count(DISTINCT md) AS model_decisions_deleted,
  count(DISTINCT pm) AS proposed_models_deleted,
  count(DISTINCT pf) AS proposed_fields_deleted,
  count(DISTINCT e) AS extraction_results_deleted
"""

_GC_ENTITY_MODEL_INSTANCE_QUERY = """
MATCH (mi:Entity|ModelInstance)
WHERE NOT EXISTS {
  MATCH (e:ExtractionResult)-[*1..7]->(mi)
}
DETACH DELETE mi
RETURN count(mi) AS borrados
"""

_GC_LABELED_ENTITY_QUERY = """
MATCH (mi:LabeledEntity)
WHERE NOT EXISTS { (mi)<--() }
DETACH DELETE mi
RETURN count(mi) AS borrados
"""

# Cascade-delete counter fields, in the order returned by _CASCADE_DELETE_QUERY.
_CASCADE_COUNTER_FIELDS = (
    "documents_deleted",
    "structure_nodes_deleted",
    "info_units_deleted",
    "model_decisions_deleted",
    "proposed_models_deleted",
    "proposed_fields_deleted",
    "extraction_results_deleted",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_cascade_delete(driver, path: str, version: int | None) -> dict[str, int]:
    """Run the cascade delete query in a single write transaction and sum
    the per-row counters returned (the query can yield multiple rows).

    Parameters
    ----------
    driver:
        An open, authenticated Neo4j driver instance.
    path:
        Document ``path`` to delete.
    version:
        Specific version to delete, or ``None`` to delete all versions.
    """
    def _do_delete() -> dict[str, int]:
        local_counters = dict.fromkeys(_CASCADE_COUNTER_FIELDS, 0)
        with driver.session() as session:
            with session.begin_transaction() as tx:
                try:
                    result = tx.run(_CASCADE_DELETE_QUERY, path=path, version=version)
                    for record in result:
                        for field_name in _CASCADE_COUNTER_FIELDS:
                            local_counters[field_name] += record[field_name]
                    tx.commit()
                except Exception:
                    tx.rollback()
                    logger.exception(
                        "delete_document: cascade delete transaction rolled back "
                        "for path=%r version=%r",
                        path,
                        version,
                    )
                    raise
        return local_counters

    return with_neo4j_retry_sync(_do_delete)


def _run_gc_pass(driver, query: str, label: str) -> tuple[int, int]:
    """Run a single garbage-collection query up to GC_MAX_PASSES times,
    stopping as soon as an execution deletes zero nodes.

    Each individual execution runs in its own write transaction, wrapped
    in with_neo4j_retry_sync.

    Parameters
    ----------
    driver:
        An open, authenticated Neo4j driver instance.
    query:
        The GC Cypher query to run (must return a single ``borrados`` count).
    label:
        Human-readable label used only for logging.

    Returns
    -------
    tuple[int, int]
        (total nodes deleted across all iterations, number of iterations run).
    """
    total_deleted = 0
    passes_run = 0

    def _do_gc_iteration() -> int:
        try:
            with driver.session() as session:
                return session.execute_write(lambda tx: tx.run(query).single()["borrados"])
        except Exception:
            logger.exception(
                "delete_document: GC pass (%s) iteration %d raised an exception.",
                label,
                passes_run + 1,
            )
            raise

    for _ in range(GC_MAX_PASSES):
        deleted = with_neo4j_retry_sync(_do_gc_iteration)
        passes_run += 1
        total_deleted += deleted
        logger.info(
            "delete_document: GC pass (%s) iteration %d deleted %d node(s).",
            label,
            passes_run,
            deleted,
        )
        if deleted == 0:
            break

    return total_deleted, passes_run


def _fetch_existing_versions(driver, path: str, version: int | None) -> list[int]:
    """Run the read-only existence-check query and return the sorted list of
    matching Document versions.

    Wrapped in with_neo4j_retry_sync for consistency with the cascade delete
    and GC passes below, so a transient Neo4j error on this first query is
    retried the same way as the rest of delete_document().

    Any ``None`` version values (from legacy/malformed :Document nodes) are
    filtered out defensively before sorting, since Python 3 cannot compare
    ``None`` with ``int`` and would raise TypeError otherwise.
    """

    def _do_query() -> list[int | None]:
        with driver.session() as session:
            existence_result = session.run(_EXISTENCE_QUERY, path=path, version=version)
            return [record["version"] for record in existence_result]

    raw_versions = with_neo4j_retry_sync(_do_query)
    return sorted(v for v in raw_versions if v is not None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delete_document(path: str, version: int | None = None) -> DeletionResult:
    """Completely delete a Document node, its entire cascade, and orphans.

    Unlike ``delete_document_content()`` (which only wipes content for
    in-place re-ingestion and keeps the :Document node), this permanently
    removes the :Document node(s) matching *path* (and, if *version* is
    given, only that version) along with:

    - Every descendant reached via ``IS_COMPOSED_OF*`` (folder-parent
      Document nodes, sibling documents, etc.).
    - All :StructureNode descendants (``HAS_STRUCTURE`` / ``HAS_CHILD``),
      their :InfoUnit, :ModelDecision, :ProposedModel, :ProposedField, and
      :ExtractionResult children.

    After the cascade delete, runs two independent garbage-collection
    passes (up to :data:`GC_MAX_PASSES` iterations each) to remove any
    :Entity/:ModelInstance and :LabeledEntity nodes left orphaned by the
    deletion.

    Opens and closes its own Neo4j driver — does not require the caller to
    manage one.

    Parameters
    ----------
    path:
        The ``path`` property of the Document to delete. Required.
    version:
        Specific integer version to delete. When ``None`` (default), every
        version of *path* is deleted (including each one's full cascade).

    Returns
    -------
    DeletionResult
        Structured counts of everything deleted. If no Document matches
        *path*/*version*, ``found`` is ``False`` and all counters are 0
        (no delete or GC queries are executed in that case).
    """
    driver = get_driver()
    try:
        versions_found = _fetch_existing_versions(driver, path, version)

        if not versions_found:
            logger.warning(
                "delete_document: no Document found for path=%r version=%r; "
                "nothing to delete.",
                path,
                version,
            )
            return DeletionResult(
                path=path,
                version=version,
                found=False,
                versions_deleted=[],
                documents_deleted=0,
                structure_nodes_deleted=0,
                info_units_deleted=0,
                model_decisions_deleted=0,
                proposed_models_deleted=0,
                proposed_fields_deleted=0,
                extraction_results_deleted=0,
                gc_entity_model_instance_deleted=0,
                gc_entity_model_instance_passes=0,
                gc_labeled_entity_deleted=0,
                gc_labeled_entity_passes=0,
            )

        logger.info(
            "delete_document: deleting path=%r version=%r (versions found: %s)",
            path,
            version,
            versions_found,
        )

        cascade_counts = _run_cascade_delete(driver, path, version)

        gc_emi_deleted, gc_emi_passes = _run_gc_pass(
            driver, _GC_ENTITY_MODEL_INSTANCE_QUERY, "Entity|ModelInstance"
        )
        gc_le_deleted, gc_le_passes = _run_gc_pass(
            driver, _GC_LABELED_ENTITY_QUERY, "LabeledEntity"
        )

        logger.info(
            "delete_document: complete for path=%r version=%r. "
            "documents_deleted=%d structure_nodes_deleted=%d "
            "gc_entity_model_instance_deleted=%d gc_labeled_entity_deleted=%d",
            path,
            version,
            cascade_counts["documents_deleted"],
            cascade_counts["structure_nodes_deleted"],
            gc_emi_deleted,
            gc_le_deleted,
        )

        return DeletionResult(
            path=path,
            version=version,
            found=True,
            versions_deleted=versions_found,
            documents_deleted=cascade_counts["documents_deleted"],
            structure_nodes_deleted=cascade_counts["structure_nodes_deleted"],
            info_units_deleted=cascade_counts["info_units_deleted"],
            model_decisions_deleted=cascade_counts["model_decisions_deleted"],
            proposed_models_deleted=cascade_counts["proposed_models_deleted"],
            proposed_fields_deleted=cascade_counts["proposed_fields_deleted"],
            extraction_results_deleted=cascade_counts["extraction_results_deleted"],
            gc_entity_model_instance_deleted=gc_emi_deleted,
            gc_entity_model_instance_passes=gc_emi_passes,
            gc_labeled_entity_deleted=gc_le_deleted,
            gc_labeled_entity_passes=gc_le_passes,
        )
    finally:
        driver.close()
