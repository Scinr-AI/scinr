"""
ingest/deletion.py — Full document deletion (Document node + cascade + GC).

Unlike ``delete_document_content()`` in ``ingest/nodes.py`` (which only wipes
structure/annotation data for a single version to support in-place
re-ingestion via ``update_mode=True``, keeping the :Document node itself), the
public :func:`delete_document` here removes the :Document node(s) as well
as their entire composed/structural subtree, and then runs a two-pass
global garbage collector to remove any resulting orphaned :Entity,
:ModelInstance, and :LabeledEntity nodes.

Before touching Neo4j, it also deletes the corresponding documental storage
records (raw binaries + converted Markdown pages) for every ``raw_file_id``
referenced by the affected :Document node(s), via the configured storage
backend (see ``storage/factory.py``). This storage cleanup is fail-fast: if
it raises, the Neo4j cascade delete never runs.

Public API
----------
    result = await delete_document(path, tenant_id="acme")               # by path
    result = await delete_document(job_id="job-123", tenant_id="acme")   # by ingestion run
    result = await delete_document(path, tenant_id=None)                 # public document
    # tenant_id is mandatory (keyword-only, no default): every deletion is
    # scoped to exactly one tenant, or to the public documents.
    # optional AND filters in either mode: version=, created_by_user_id=
    # opens its own driver
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from scinr.newton.config import get_config
from scinr.newton.ingest.config import get_driver
from scinr.newton.results import DeletionResult
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

GC_MAX_PASSES = 7
"""Maximum number of iterations run for each garbage-collection pass."""


def _as_list(name: str, value: str | Sequence[str] | None) -> list[str] | None:
    """Normalize a ``str | Sequence[str] | None`` filter to a list (``None`` = no filter)."""
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError(f"{name} must not be an empty list (omit it for no filter).")
    return values


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

# delete_document() argument names that optionally narrow the :Document
# selection, in a fixed order. Each name is also the node property it filters
# on (equality). The names are a hard-coded whitelist — never user input — so
# interpolating them into the query string below carries no injection risk.
# tenant_id is deliberately NOT here: it is never optional (see below).
# ``job_id`` and ``created_by_user_id`` match with ``IN`` (a list of values);
# the others with equality.
_IN_FIELDS = frozenset({"job_id", "created_by_user_id"})
_FILTERABLE_FIELDS = ("path", "job_id", "version", "created_by_user_id")


def _build_doc_match(filters: dict) -> tuple[str, dict]:
    """Build the ``MATCH (d:Document) WHERE ...`` selection prefix, together
    with the matching parameter dict.

    The tenant condition is **always** emitted: ``filters["tenant_id"]`` must
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
    """
    tenant = filters.get("tenant_id")
    if tenant is None:
        raise ValueError("_build_doc_match requires the stored tenant key in filters['tenant_id'].")
    conditions: list[str] = ["d.tenant_id = $tenant_id"]
    params: dict = {"tenant_id": tenant}
    for name in _FILTERABLE_FIELDS:
        value = filters.get(name)
        if value is not None:
            op = "IN" if name in _IN_FIELDS else "="
            conditions.append(f"d.{name} {op} ${name}")
            params[name] = value
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

_CASCADE_DELETE_TAIL = """
OPTIONAL MATCH (d)-[:IS_COMPOSED_OF*]->(cd)
WITH collect(DISTINCT d) + collect(DISTINCT cd) AS documentNodes
UNWIND documentNodes AS documentNode
WITH DISTINCT documentNode
OPTIONAL MATCH (documentNode)-[:HAS_STRUCTURE*..]->(parentStructureNode)
OPTIONAL MATCH (parentStructureNode)-[:HAS_CHILD*..]->(childStructureNode)
WITH documentNode, collect(DISTINCT parentStructureNode) + collect(DISTINCT childStructureNode) AS docStructureNodes
UNWIND (CASE WHEN docStructureNodes = [] THEN [NULL] ELSE docStructureNodes END) AS structureNode
OPTIONAL MATCH (structureNode)-[:HAS_INFO_UNIT]->(iu)
OPTIONAL MATCH (structureNode)-[:HAS_MODEL_DECISION]->(md)
OPTIONAL MATCH (md)-[:HAS_PROPOSED_MODEL]-(pm)
OPTIONAL MATCH (pm)-[:HAS_PROPOSED_FIELD]->(pf)
OPTIONAL MATCH (structureNode)-[:HAS_EXTRACTION]->(e)
DETACH DELETE documentNode, structureNode, iu, md, pm, pf, e
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

# Cascade-delete counter fields, in the order returned by _CASCADE_DELETE_TAIL.
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


def _run_cascade_delete(driver, filters: dict) -> dict[str, int]:
    """Run the cascade delete query in a single write transaction and sum
    the per-row counters returned (the query can yield multiple rows).

    Parameters
    ----------
    driver:
        An open, authenticated Neo4j driver instance.
    filters:
        The target-selection parameter dict (``path``, ``version``,
        ``tenant_id``, ``created_by_user_id``, ``job_id``); ``tenant_id`` (the
        stored key) always becomes a WHERE condition, the others only when
        not None — see :func:`_build_doc_match`.
    """

    def _do_delete() -> dict[str, int]:
        local_counters = dict.fromkeys(_CASCADE_COUNTER_FIELDS, 0)
        cfg = get_config()
        match_clause, params = _build_doc_match(filters)
        with driver.session(database=cfg.neo4j_database) as session:
            with session.begin_transaction() as tx:
                try:
                    result = tx.run(match_clause + _CASCADE_DELETE_TAIL, **params)
                    for record in result:
                        for field_name in _CASCADE_COUNTER_FIELDS:
                            local_counters[field_name] += record[field_name]
                    tx.commit()
                except Exception:
                    tx.rollback()
                    logger.exception(
                        "delete_document: cascade delete transaction rolled back for filters=%r",
                        filters,
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
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
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


def _fetch_existing_versions(driver, filters: dict) -> list[int]:
    """Run the read-only existence-check query and return the sorted list of
    matching Document versions.

    Wrapped in with_neo4j_retry_sync for consistency with the cascade delete
    and GC passes below, so a transient Neo4j error on this first query is
    retried the same way as the rest of delete_document().

    Any ``None`` version values (from legacy/malformed :Document nodes) are
    filtered out defensively before sorting, since Python 3 cannot compare
    ``None`` with ``int`` and would raise TypeError otherwise.

    *filters* is the full target-selection parameter dict (see
    :func:`_run_cascade_delete`).
    """

    def _do_query() -> list[int | None]:
        cfg = get_config()
        match_clause, params = _build_doc_match(filters)
        with driver.session(database=cfg.neo4j_database) as session:
            existence_result = session.run(match_clause + _EXISTENCE_TAIL, **params)
            return [record["version"] for record in existence_result]

    raw_versions = with_neo4j_retry_sync(_do_query)
    return sorted(v for v in raw_versions if v is not None)


def _fetch_raw_file_ids(driver, filters: dict) -> list[tuple[str, str | None]]:
    """Run the read-only raw_file_ids query and return the distinct
    ``(raw_file_id, tenant_id)`` pairs — non-empty ``raw_file_id`` values and
    the stored tenant of the node carrying each — for the target Document(s)
    and every descendant reached via ``IS_COMPOSED_OF*``: the same scope used
    by the cascade delete query below.

    Wrapped in with_neo4j_retry_sync for consistency with the other queries
    in this module. *filters* is the full target-selection parameter dict
    (see :func:`_run_cascade_delete`).
    """

    def _do_query() -> list[tuple[str, str | None]]:
        cfg = get_config()
        match_clause, params = _build_doc_match(filters)
        with driver.session(database=cfg.neo4j_database) as session:
            result = session.run(match_clause + _RAW_FILE_IDS_TAIL, **params)
            return [(record["raw_file_id"], record["tenant_id"]) for record in result]

    return with_neo4j_retry_sync(_do_query)


async def _delete_storage_for_raw_file_ids(
    raw_file_ids: list[tuple[str, str | None]],
) -> tuple[int, int]:
    """Delete storage records (raw binaries + converted pages) for every
    given ``(raw_file_id, tenant_id)`` pair, via the configured storage backend.

    Each delete is scoped to the tenant of the graph node that carries the
    id, so a ``raw_file_id`` forged to point at another tenant's upload
    deletes nothing. No user / job filter here: ``delete_document``'s filters
    already selected the Documents in the graph, and their original must go
    even when another job of the same tenant uploaded it. A legacy node
    without ``tenant_id`` deletes unfiltered (its records have no tenant
    either).

    Fail-fast: no exception raised here is caught — any unexpected error
    (e.g. StorageError, a dropped connection) propagates to the caller so
    that the Neo4j cascade delete is never reached. Backend implementations
    are expected to be idempotent for "already gone" cases (missing
    metadata, missing GridFS binary, invalid ObjectId, no matching pages,
    out of scope) and to not raise for those.

    Parameters
    ----------
    raw_file_ids:
        Distinct ``(raw_file_id, stored tenant_id)`` pairs to delete storage for.

    Returns
    -------
    tuple[int, int]
        (raw_files_deleted, converted_pages_deleted).
    """
    if not raw_file_ids:
        return 0, 0

    from scinr.newton.storage.factory import get_storage

    raw_file_repo, page_repo = get_storage()
    raw_files_deleted = 0
    converted_pages_deleted = 0
    for rid, tenant in raw_file_ids:
        converted_pages_deleted += await page_repo.delete_pages(rid, tenant_id=tenant)
        await raw_file_repo.delete(rid, tenant_id=tenant)
        raw_files_deleted += 1
    return raw_files_deleted, converted_pages_deleted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def delete_document(
    path: str | None = None,
    version: int | None = None,
    *,
    tenant_id: str | None,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> DeletionResult:
    """Completely delete Document node(s), their entire cascade, and orphans.

    Unlike ``delete_document_content()`` (which only wipes content for
    in-place re-ingestion and keeps the :Document node), this permanently
    removes every :Document node matching the selector below along with:

    - Every descendant reached via ``IS_COMPOSED_OF*`` (folder-parent
      Document nodes, sibling documents, etc.).
    - All :StructureNode descendants (``HAS_STRUCTURE`` / ``HAS_CHILD``),
      their :InfoUnit, :ModelDecision, :ProposedModel, :ProposedField, and
      :ExtractionResult children.

    Target selection
    ----------------
    *tenant_id* is **mandatory** (keyword-only, no default — omitting it is a
    ``TypeError``) and always applied: the deletion never leaves that
    tenant's documents. ``tenant_id=None`` explicitly means "public
    documents" (stored as ``"__public__"``, and ``tenant_id="__public__"``
    means the same); there is no way to delete across tenants. Because every tenant has its own folder documents, the
    ``IS_COMPOSED_OF*`` cascade below stays within the tenant too.

    On top of the tenant, exactly one of *path* or *job_id* must be provided
    (``ValueError`` otherwise):

    - *path*: delete the tenant's Document at that ``path`` (all versions,
      or only *version* when given).
    - *job_id*: delete every Document of the tenant whose ``job_id``
      property is any of these values (one ``str`` or several) — across all
      paths and versions of those ingestion runs.

    *created_by_user_id* (one ``str`` or several, matched with ``IN``), when
    given, is an additional AND filter applied on top of either selector. Left as ``None`` it means "do not filter on this
    property" (it does **not** mean "the property must be null"). *version*
    is also accepted as an extra filter in *job_id* mode.

    Who may delete a public document is an authorization decision for the
    calling API layer; this function only enforces the scope.

    Before any Neo4j deletion happens, this also deletes the documental
    storage records (raw binary + converted Markdown pages) for every
    non-empty ``raw_file_id`` found on the target Document(s) and their
    ``IS_COMPOSED_OF*`` descendants, via the configured storage backend
    (``storage/factory.py::get_storage()``). This step is fail-fast: if
    deleting storage for any raw_file_id raises an unexpected exception,
    it propagates immediately and the Neo4j cascade delete is never run.

    After the cascade delete, runs two independent garbage-collection
    passes (up to :data:`GC_MAX_PASSES` iterations each) to remove any
    :Entity/:ModelInstance and :LabeledEntity nodes left orphaned by the
    deletion.

    Opens and closes its own Neo4j driver — does not require the caller to
    manage one. The Neo4j-specific work (existence check, raw_file_id
    lookup, cascade delete, GC passes) uses the existing synchronous Neo4j
    driver under the hood, dispatched via ``asyncio.to_thread()``; the
    storage deletion calls are awaited directly since storage repositories
    (Motor-backed) are natively async.

    Returns
    -------
    DeletionResult
        Structured counts of everything deleted. If no Document matches the
        selector, ``found`` is ``False`` and all counters are 0 (no storage,
        delete, or GC queries are executed in that case).

    Raises
    ------
    TypeError
        If *tenant_id* is not passed.
    ValueError
        If neither or both of *path* and *job_id* are provided, or
        *tenant_id* is empty, or *job_id* / *created_by_user_id* is an
        empty list.
    """
    if (path is None) == (job_id is None):
        raise ValueError(
            "delete_document requires exactly one of 'path' or 'job_id' "
            f"(got path={path!r}, job_id={job_id!r})."
        )
    job_ids = _as_list("job_id", job_id)
    user_ids = _as_list("created_by_user_id", created_by_user_id)

    filters = {
        "path": path,
        "version": version,
        "tenant_id": tenant_key(tenant_id),
        "created_by_user_id": user_ids,
        "job_id": job_ids,
    }
    selector_repr = ", ".join(f"{k}={v!r}" for k, v in filters.items() if v is not None)

    driver = get_driver()
    try:
        versions_found = await asyncio.to_thread(_fetch_existing_versions, driver, filters)

        if not versions_found:
            logger.warning(
                "delete_document: no Document found for %s; nothing to delete.",
                selector_repr,
            )
            return DeletionResult(
                path=path,
                version=version,
                job_id=job_id,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
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
                raw_files_deleted=0,
                converted_pages_deleted=0,
            )

        logger.info(
            "delete_document: deleting %s (versions found: %s)",
            selector_repr,
            versions_found,
        )

        raw_file_ids = await asyncio.to_thread(_fetch_raw_file_ids, driver, filters)
        raw_files_deleted, converted_pages_deleted = await _delete_storage_for_raw_file_ids(
            raw_file_ids
        )

        logger.info(
            "delete_document: storage cleanup complete for %s. "
            "raw_files_deleted=%d converted_pages_deleted=%d",
            selector_repr,
            raw_files_deleted,
            converted_pages_deleted,
        )

        cascade_counts = await asyncio.to_thread(_run_cascade_delete, driver, filters)

        gc_emi_deleted, gc_emi_passes = await asyncio.to_thread(
            _run_gc_pass, driver, _GC_ENTITY_MODEL_INSTANCE_QUERY, "Entity|ModelInstance"
        )
        gc_le_deleted, gc_le_passes = await asyncio.to_thread(
            _run_gc_pass, driver, _GC_LABELED_ENTITY_QUERY, "LabeledEntity"
        )

        logger.info(
            "delete_document: complete for %s. "
            "documents_deleted=%d structure_nodes_deleted=%d "
            "gc_entity_model_instance_deleted=%d gc_labeled_entity_deleted=%d",
            selector_repr,
            cascade_counts["documents_deleted"],
            cascade_counts["structure_nodes_deleted"],
            gc_emi_deleted,
            gc_le_deleted,
        )

        return DeletionResult(
            path=path,
            version=version,
            job_id=job_id,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
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
            raw_files_deleted=raw_files_deleted,
            converted_pages_deleted=converted_pages_deleted,
        )
    finally:
        driver.close()
