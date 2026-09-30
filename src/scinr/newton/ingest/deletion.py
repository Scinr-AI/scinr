"""
ingest/deletion.py — Full document deletion (Document node + cascade + GC).

Unlike ``delete_document_content()`` in ``ingest/nodes.py`` (which only wipes
structure/annotation data for a single version to support in-place
re-ingestion via ``update_mode=True``, keeping the :Document node itself), the
public :func:`delete_document` here removes the :Document node(s) as well
as their entire composed/structural subtree, and collects the :Entity,
:ModelInstance and :LabeledEntity nodes that deletion leaves orphaned
(``ingest/_gc.py``). :func:`collect_orphans` sweeps every orphan of a tenant,
whatever left it.

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

    result = await collect_orphans(tenant_id="acme")    # maintenance sweep
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from scinr.newton.config import get_config
from scinr.newton.ingest._cascade import (
    DELETE_BATCH_ROWS,
    DOCS_BY_KEY,
    delete_documents,
    run_count_query,
    run_subtree_steps,
    subtree_steps,
)
from scinr.newton.ingest._gc import GC_MAX_PASSES, OrphanCollector, sweep_tenant
from scinr.newton.ingest.config import get_driver
from scinr.newton.results import DeletionResult, OrphanCollectionResult
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

__all__ = ["GC_MAX_PASSES", "collect_orphans", "delete_document"]


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

# The documents the cascade works on: the matched ones and their descendants.
# ``interrupted``: an earlier delete_document() or freeze_document() stopped
# half-way through this document.
_DOCUMENT_KEYS_TAIL = """
OPTIONAL MATCH (d)-[:IS_COMPOSED_OF*]->(cd:Document)
WITH collect(DISTINCT d) AS seeds, collect(DISTINCT cd) AS descendants
UNWIND seeds + descendants AS doc
WITH DISTINCT doc, seeds
RETURN doc.path AS path, doc.version AS version, doc.tenant_id AS tenant_id,
       doc IN seeds AS is_seed,
       coalesce(doc.deletion_pending, doc.frozen_cleanup_pending, false) AS interrupted
"""

# Set before anything is deleted, and gone with the Document itself: a
# Document that still carries it was left half-deleted.
_MARK_PENDING_QUERY = DOCS_BY_KEY + "SET d.deletion_pending = true\nRETURN count(d) AS n\n"

# Cascade-delete counter fields of DeletionResult.
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


def _fetch_document_keys(driver, filters: dict) -> tuple[list[dict], bool]:
    """Resolve the selector to the ``{"path", "version"}`` keys of the matched
    Documents and their ``IS_COMPOSED_OF*`` descendants — descendants first,
    matched documents last (the order they are deleted in) — and whether an
    earlier delete or freeze of any of them was interrupted.

    A Document that cannot be addressed by its key — no ``path`` /
    ``version``, or another tenant's (impossible under the write invariant) —
    is left alone, with a warning.
    """
    tenant = filters["tenant_id"]

    def _do_query() -> list[dict]:
        cfg = get_config()
        match_clause, params = _build_doc_match(filters)
        with driver.session(database=cfg.neo4j_database) as session:
            result = session.run(match_clause + _DOCUMENT_KEYS_TAIL, **params)
            return [dict(record) for record in result]

    rows = with_neo4j_retry_sync(_do_query)
    addressable = [
        row
        for row in rows
        if row["path"] is not None and row["version"] is not None and row["tenant_id"] == tenant
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
    return keys, any(row["interrupted"] for row in addressable)


def _run_cascade_delete(driver, filters: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Delete the subtree of the selected Documents, collect the orphans and
    then delete the Documents themselves, in bounded transactions
    (``ingest/_cascade.py``). Returns the cascade counters and the ``gc_*``
    counters.

    Not atomic — a single transaction holding the subtree of hundreds of
    documents overruns the server's transaction memory. The Documents are
    marked ``deletion_pending`` first and go last (descendants before the
    matched documents), so after a failure the selector still matches and the
    same call deletes what is left.

    The orphans are collected among what the deleted :ExtractionResult nodes
    pointed at (``OrphanCollector``). Those candidates only live in this
    process: when a Document carries the mark of an interrupted delete or
    freeze, the candidates of that run are lost and the whole tenant is swept
    instead.

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
    database = get_config().neo4j_database
    tenant = filters["tenant_id"]
    keys, interrupted = _fetch_document_keys(driver, filters)
    orphans = None
    if interrupted:
        logger.warning(
            "delete_document: an earlier delete or freeze of filters=%r was interrupted; "
            "collecting the orphans of the whole tenant %r instead of only those of "
            "this deletion.",
            filters,
            tenant,
        )
    else:
        orphans = OrphanCollector(
            driver, tenant_id=tenant, database=database, caller="delete_document"
        )
    try:
        for start in range(0, len(keys), DELETE_BATCH_ROWS):
            run_count_query(
                driver,
                database,
                _MARK_PENDING_QUERY,
                tenant_id=tenant,
                keys=keys[start : start + DELETE_BATCH_ROWS],
            )
        subtree_counts = run_subtree_steps(
            driver,
            database,
            tenant,
            keys,
            subtree_steps(),
            caller="delete_document",
            orphans=orphans,
        )
        if orphans is not None:
            orphans.collect()
            gc_counts = orphans.counts()
        else:
            gc_counts = sweep_tenant(
                driver, tenant_id=tenant, database=database, caller="delete_document"
            )
        documents_deleted = delete_documents(driver, database, tenant, keys)
    except Exception:
        logger.exception(
            "delete_document: cascade delete interrupted for filters=%r. Part of the "
            "subtree may be gone; the Document node(s) still there are deleted by calling "
            "delete_document() again with the same selector.",
            filters,
        )
        raise
    counters = {name: subtree_counts.get(name, 0) for name in _CASCADE_COUNTER_FIELDS}
    counters["documents_deleted"] = documents_deleted
    return counters, gc_counts


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

    - Every descendant reached via ``IS_COMPOSED_OF*`` **downwards** from
      the matched nodes: deleting a folder deletes its whole subtree, while
      deleting a leaf by *path* leaves its parent folder and siblings
      untouched (they only go too when they match the selector themselves,
      e.g. the same *job_id*).
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

    The cascade runs in bounded transactions (``ingest/_cascade.py``), not
    in a single one: one transaction holding the subtree of hundreds of
    documents overruns the server's transaction memory. It is therefore not
    atomic — the :Document nodes are marked ``deletion_pending`` first and
    deleted last, so if it fails half-way, calling ``delete_document()``
    again with the same selector deletes what is left. It also removes the
    :ComplementaryMatch and :SupplementaryField nodes of the annotations.

    Before the :Document nodes go, the :Entity/:ModelInstance and
    :LabeledEntity nodes **of the same tenant** this deletion leaves orphaned
    are deleted too (``ingest/_gc.py``). Only what hung from the deleted
    :ExtractionResult nodes is checked, so the cost follows the size of the
    deletion, not of the tenant; orphans left by anything else stay until
    :func:`collect_orphans` is run. The exception is a call that finishes an
    interrupted delete or freeze: the candidates of the interrupted run are
    lost, so it sweeps the whole tenant (up to :data:`GC_MAX_PASSES`
    iterations per pass), as ``collect_orphans()`` does.

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

        cascade_counts, gc_counts = await asyncio.to_thread(_run_cascade_delete, driver, filters)

        logger.info(
            "delete_document: complete for %s. "
            "documents_deleted=%d structure_nodes_deleted=%d "
            "gc_entity_model_instance_deleted=%d gc_labeled_entity_deleted=%d",
            selector_repr,
            cascade_counts["documents_deleted"],
            cascade_counts["structure_nodes_deleted"],
            gc_counts["gc_entity_model_instance_deleted"],
            gc_counts["gc_labeled_entity_deleted"],
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
            **gc_counts,
            raw_files_deleted=raw_files_deleted,
            converted_pages_deleted=converted_pages_deleted,
        )
    finally:
        driver.close()


async def collect_orphans(*, tenant_id: str | None) -> OrphanCollectionResult:
    """Delete every orphaned :Entity, :ModelInstance and :LabeledEntity node
    of one tenant.

    ``delete_document()`` and ``freeze_document()`` only collect the orphans
    they cause. This maintenance call collects the rest: the nodes left
    behind by the write paths that delete :ExtractionResult nodes without
    collecting (re-ingestion with ``update_mode=True``, re-extraction of a
    node), and by an interrupted delete or freeze that was never run again.

    An :Entity / :ModelInstance is an orphan when no :ExtractionResult
    reaches it within 7 hops; a :LabeledEntity, when nothing points at it.
    Every such node of the tenant is checked, so the cost grows with the
    tenant, not with what there is to collect: run it from time to time, not
    after every operation.

    Do not run it while the same tenant is being ingested: a node written
    just before the :ExtractionResult that will point at it is, for that
    instant, an orphan.

    *tenant_id* is mandatory (keyword-only, no default); ``None`` and
    ``"__public__"`` both mean the public documents. Opens and closes its own
    Neo4j driver.

    Returns
    -------
    OrphanCollectionResult
        How many nodes were deleted, and in how many iterations.

    Raises
    ------
    TypeError
        If *tenant_id* is not passed.
    ValueError
        If *tenant_id* is empty.
    """
    tenant = tenant_key(tenant_id)
    driver = get_driver()
    try:
        counts = await asyncio.to_thread(
            sweep_tenant,
            driver,
            tenant_id=tenant,
            database=get_config().neo4j_database,
            caller="collect_orphans",
        )
    finally:
        driver.close()
    logger.info(
        "collect_orphans: complete for tenant %r. gc_entity_model_instance_deleted=%d "
        "gc_labeled_entity_deleted=%d",
        tenant,
        counts["gc_entity_model_instance_deleted"],
        counts["gc_labeled_entity_deleted"],
    )
    return OrphanCollectionResult(tenant_id=tenant_id, **counts)
