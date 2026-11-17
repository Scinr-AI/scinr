"""
ingest/deletion.py — Full document deletion (Document node + cascade + GC).

Unlike ``delete_document_content()`` in ``ingest/nodes.py`` (which only wipes
structure/annotation data for a single version to support in-place
re-ingestion via ``update_mode=True``, keeping the :Document node itself), the
public :func:`delete_document` here removes the :Document node(s) as well
as their entire composed/structural subtree, and collects the :Entity,
:ModelInstance and :LabeledEntity nodes that deletion leaves orphaned
(``graph/neo4j/gc.py``). :func:`collect_orphans` sweeps every orphan of a tenant,
whatever left it.

Before touching the graph, it also deletes the corresponding documental storage
records (raw binaries + converted Markdown pages) for every ``raw_file_id``
referenced by the affected :Document node(s), via the configured storage
backend (see ``storage/factory.py``). This storage cleanup is fail-fast: if
it raises, the graph cascade delete never runs.

Public API
----------
    result = await delete_document(path, tenant_id="acme")               # by path
    result = await delete_document(job_id="job-123", tenant_id="acme")   # by ingestion run
    result = await delete_document(path, tenant_id=None)                 # public document
    # tenant_id is mandatory (keyword-only, no default): every deletion is
    # scoped to exactly one tenant, or to the public documents.
    # optional AND filters in either mode: version=, created_by_user_id=
    # opens its own graph-store scope (one synchronous driver)

    result = await collect_orphans(tenant_id="acme")    # maintenance sweep
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from scinr.newton.graph import GraphStore, get_graph_store
from scinr.newton.graph.types import GC_MAX_PASSES, DocumentSelector
from scinr.newton.results import DeletedDocument, DeletionResult, OrphanCollectionResult
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


def _run_cascade_delete(
    store: GraphStore, selector: DocumentSelector, filters: dict
) -> tuple[dict[str, int], dict[str, int], list[DeletedDocument]]:
    """Delete the subtree of the selected Documents, collect the orphans and
    then delete the Documents themselves, in bounded transactions
    (``store.deletion``). Returns the cascade counters, the ``gc_*``
    counters and the documents it worked on.

    Not atomic — a single transaction holding the subtree of hundreds of
    documents overruns the server's transaction memory. The Documents are
    marked ``deletion_pending`` first and go last (descendants before the
    matched documents), so after a failure the selector still matches and the
    same call deletes what is left.

    The orphans are collected among what the deleted :ExtractionResult nodes
    pointed at (``store.gc.collector``). Those candidates only live in this
    process: when a Document carries the mark of an interrupted delete or
    freeze, the candidates of that run are lost and the whole tenant is swept
    instead.

    Parameters
    ----------
    store:
        The graph store; run inside its ``sync_scope()``.
    selector:
        The validated target selection (stored tenant key).
    filters:
        The same selection as a ``dict`` (``path``, ``version``, ``tenant_id``,
        ``created_by_user_id``, ``job_id``), only for the log messages.
    """
    tenant = selector.tenant_id
    keys, documents, interrupted = store.deletion.fetch_document_keys(selector)
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
        orphans = store.gc.collector(tenant, "delete_document")
    try:
        store.deletion.mark_deletion_pending(tenant, keys)
        subtree_counts = store.deletion.cascade_delete(
            tenant, keys, orphans, caller="delete_document"
        )
        if orphans is not None:
            orphans.collect()
            gc_counts = orphans.counts()
        else:
            gc_counts = store.gc.sweep_tenant(tenant, "delete_document")
        documents_deleted = store.deletion.delete_documents(tenant, keys)
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
    return counters, gc_counts, documents


async def _delete_storage_for_raw_file_ids(
    raw_file_ids: list[tuple[str, str | None]],
) -> tuple[int, int, int]:
    """Delete storage records (raw binaries + converted pages + stored
    images) for every given ``(raw_file_id, tenant_id)`` pair, via the
    configured storage backend.

    Each delete is scoped to the tenant of the graph node that carries the
    id, so a ``raw_file_id`` forged to point at another tenant's upload
    deletes nothing. No user / job filter here: ``delete_document``'s filters
    already selected the Documents in the graph, and their original must go
    even when another job of the same tenant uploaded it. A legacy node
    without ``tenant_id`` deletes unfiltered (its records have no tenant
    either).

    Fail-fast: no exception raised here is caught — any unexpected error
    (e.g. StorageError, a dropped connection) propagates to the caller so
    that the graph cascade delete is never reached. Backend implementations
    are expected to be idempotent for "already gone" cases (missing
    metadata, missing GridFS binary, invalid ObjectId, no matching pages or
    images, out of scope) and to not raise for those. A backend without
    image storage returns 0 from ``delete_images`` and the deletion proceeds
    unchanged.

    Parameters
    ----------
    raw_file_ids:
        Distinct ``(raw_file_id, stored tenant_id)`` pairs to delete storage for.

    Returns
    -------
    tuple[int, int, int]
        (raw_files_deleted, converted_pages_deleted, page_images_deleted).
    """
    if not raw_file_ids:
        return 0, 0, 0

    from scinr.newton.storage.factory import get_storage

    raw_file_repo, page_repo = get_storage()
    raw_files_deleted = 0
    converted_pages_deleted = 0
    page_images_deleted = 0
    for rid, tenant in raw_file_ids:
        page_images_deleted += await raw_file_repo.delete_images(rid, tenant_id=tenant)
        converted_pages_deleted += await page_repo.delete_pages(rid, tenant_id=tenant)
        await raw_file_repo.delete(rid, tenant_id=tenant)
        raw_files_deleted += 1
    return raw_files_deleted, converted_pages_deleted, page_images_deleted


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

    Before any graph deletion happens, this also deletes the documental
    storage records (raw binary + converted Markdown pages + stored images)
    for every non-empty ``raw_file_id`` found on the target Document(s) and
    their ``IS_COMPOSED_OF*`` descendants, via the configured storage backend
    (``storage/factory.py::get_storage()``). This step is fail-fast: if
    deleting storage for any raw_file_id raises an unexpected exception,
    it propagates immediately and the graph cascade delete is never run.

    The cascade runs in bounded transactions (``graph/neo4j/deletion.py``), not
    in a single one: one transaction holding the subtree of hundreds of
    documents overruns the server's transaction memory. It is therefore not
    atomic — the :Document nodes are marked ``deletion_pending`` first and
    deleted last, so if it fails half-way, calling ``delete_document()``
    again with the same selector deletes what is left. It also removes the
    :ComplementaryMatch and :SupplementaryField nodes of the annotations.

    Before the :Document nodes go, the :Entity/:ModelInstance and
    :LabeledEntity nodes **of the same tenant** this deletion leaves orphaned
    are deleted too (``graph/neo4j/gc.py``). Only what hung from the deleted
    :ExtractionResult nodes is checked, so the cost follows the size of the
    deletion, not of the tenant; orphans left by anything else stay until
    :func:`collect_orphans` is run. The exception is a call that finishes an
    interrupted delete or freeze: the candidates of the interrupted run are
    lost, so it sweeps the whole tenant (up to :data:`GC_MAX_PASSES`
    iterations per pass), as ``collect_orphans()`` does.

    Opens and closes its own synchronous graph-store scope — does not require
    the caller to manage one. The graph work (existence check, raw_file_id
    lookup, cascade delete, GC passes) goes through ``store.deletion`` and
    ``store.gc`` and is dispatched via ``asyncio.to_thread()``; the
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
    GraphError
        If the graph engine fails (see :mod:`scinr.newton.exceptions`).
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

    selector = DocumentSelector(**filters)
    store = get_graph_store()
    with store.sync_scope():
        versions_found = await asyncio.to_thread(store.deletion.fetch_existing_versions, selector)

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

        raw_file_ids = await asyncio.to_thread(store.deletion.fetch_raw_file_ids, selector)
        raw_files_deleted, converted_pages_deleted, page_images_deleted = (
            await _delete_storage_for_raw_file_ids(raw_file_ids)
        )

        logger.info(
            "delete_document: storage cleanup complete for %s. "
            "raw_files_deleted=%d converted_pages_deleted=%d page_images_deleted=%d",
            selector_repr,
            raw_files_deleted,
            converted_pages_deleted,
            page_images_deleted,
        )

        cascade_counts, gc_counts, documents = await asyncio.to_thread(
            _run_cascade_delete, store, selector, filters
        )

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
            page_images_deleted=page_images_deleted,
            documents=documents,
        )


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
    graph-store scope.

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
    GraphError
        If the graph engine fails (see :mod:`scinr.newton.exceptions`).
    """
    tenant = tenant_key(tenant_id)
    store = get_graph_store()
    with store.sync_scope():
        counts = await asyncio.to_thread(store.gc.sweep_tenant, tenant, "collect_orphans")
    logger.info(
        "collect_orphans: complete for tenant %r. gc_entity_model_instance_deleted=%d "
        "gc_labeled_entity_deleted=%d",
        tenant,
        counts["gc_entity_model_instance_deleted"],
        counts["gc_labeled_entity_deleted"],
    )
    return OrphanCollectionResult(tenant_id=tenant_id, **counts)
