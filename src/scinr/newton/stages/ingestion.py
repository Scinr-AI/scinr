"""
stages/ingestion.py — Stage 2: Load Documents into the graph database, plus replacement helpers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from scinr.newton.exceptions import IngestionError, PreconditionError
from scinr.newton.graph import get_graph_store
from scinr.newton.ingest.loader import load_documents, load_files
from scinr.newton.results import DocumentResult, StageResult
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)


async def run_ingestion(
    output_folder: str | None = None,
    files: list[Path] | None = None,
    documents: list | None = None,
    update_mode: bool = False,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> StageResult:
    """Load extracted documents into the graph database.

    Accepts documents either from disk (via *output_folder* or *files*) or
    directly as in-memory :class:`~models.document_structure.Document` objects
    (via *documents*). Exactly one source must be provided.

    Parameters
    ----------
    output_folder:
        Path to the directory containing ``extract-*.json`` files. Used when
        neither *files* nor *documents* is provided.
    files:
        Explicit list of JSON extraction file paths to ingest. Takes priority
        over *output_folder* when both are given.
    documents:
        List of in-memory Document objects from run_extraction() (in-memory
        mode). When provided, *output_folder* and *files* are ignored.
    update_mode:
        If True, wipe existing structure of the latest version and re-insert
        without creating a new version.
    tenant_id:
        Tenant owning the ingested documents (``None`` = public). Part of the
        :Document identity: versions are resolved, and ``update_mode`` wipes
        content, within this tenant only. Overrides any tenant baked into an
        ingested ``extract-*.json`` (see ``ingest.loader.load_file``).
    created_by_user_id, job_id:
        Optional caller-supplied provenance metadata written onto every
        :Document node created by this stage (see ``ingest.loader.load_file``).

    Returns
    -------
    StageResult
        Stage result with one DocumentResult per ingested document.

    Documents whose ``raw_file_id`` is not a stored raw file of their
    effective tenant are rejected before any graph write and reported as
    failed (see ``ingest.raw_file_check``); the rest of the batch is ingested.

    Raises
    ------
    ValueError
        If no source is provided.
    """
    t0 = time.monotonic()

    if documents is None and files is None and output_folder is None:
        raise ValueError(
            "At least one of documents, files, or output_folder must be provided."
        )

    from scinr.newton.ingest.raw_file_check import verify_document, verify_json_file

    # Reject, before any graph write, every document whose raw_file_id is not a
    # stored raw file of its effective tenant (see ingest/raw_file_check.py).
    rejected: dict[int, str] = {}
    if documents is not None:
        for i, doc in enumerate(documents):
            try:
                await verify_document(doc, tenant_id)
            except IngestionError as exc:
                logger.error("run_ingestion: rejecting '%s': %s", doc.document_name, exc)
                rejected[i] = str(exc)
    else:
        if files is None:
            output_path = Path(output_folder)
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Output folder not found: '{output_folder}'. "
                    f"Run run_extraction() first."
                )
            files_to_check = sorted(output_path.rglob("extract-*.json"))
            if not files_to_check:
                raise FileNotFoundError(
                    f"No 'extract-*.json' files found in '{output_folder}'. "
                    f"Run run_extraction() first."
                )
        else:
            files_to_check = list(files)
        for i, path in enumerate(files_to_check):
            try:
                await verify_json_file(path, tenant_id)
            except IngestionError as exc:
                logger.error("run_ingestion: rejecting '%s': %s", path, exc)
                rejected[i] = str(exc)

    store = get_graph_store()
    doc_results: list[DocumentResult] = []
    with store.sync_scope():
        store.schema.setup()

        if documents is not None:
            # In-memory mode
            accepted = [d for i, d in enumerate(documents) if i not in rejected]
            doc_names = load_documents(
                accepted,
                update_mode=update_mode,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            for i, doc in enumerate(documents):
                success = i not in rejected and doc.document_name in doc_names
                doc_results.append(DocumentResult(
                    document_name=doc.document_name,
                    nodes_processed=1 if success else 0,
                    nodes_failed=0 if success else 1,
                    errors=[] if success else [
                        rejected.get(i, f"Failed to ingest '{doc.document_name}'")
                    ],
                ))
        else:
            # From disk: an explicit file list, or every extract-*.json under
            # output_folder (the same files load_folder() would pick up).
            accepted_files = [p for i, p in enumerate(files_to_check) if i not in rejected]
            doc_names = load_files(
                accepted_files,
                update_mode=update_mode,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            for i, path in enumerate(files_to_check):
                doc_name = path.stem.removeprefix("extract-")
                success = i not in rejected and doc_name in doc_names
                doc_results.append(DocumentResult(
                    document_name=doc_name,
                    nodes_processed=1 if success else 0,
                    nodes_failed=0 if success else 1,
                    errors=[] if success else [
                        rejected.get(i, f"Failed to ingest '{path.name}'")
                    ],
                ))

    total_failed = sum(1 for r in doc_results if r.nodes_failed > 0)
    duration = time.monotonic() - t0
    return StageResult(
        stage="ingestion",
        success=total_failed == 0,
        documents=doc_results,
        total_processed=sum(r.nodes_processed for r in doc_results),
        total_failed=total_failed,
        duration_seconds=duration,
    )


def preflight_check_replaces(replaces_name: str, *, tenant_id: str | None = None) -> dict:
    """Verify that the document to be replaced exists in the graph.

    Queries for *tenant_id*'s Document nodes with the given name and
    latest=True — another tenant's document with the same name is never
    considered (``None`` = public documents only).
    Fails if:

    - No document is found.
    - Multiple documents with the same name and latest=True are found
      (ambiguous — user should use a more specific path).

    Opens and closes its own graph connection, unless called inside a
    ``store.sync_scope()`` block.

    Parameters
    ----------
    replaces_name:
        The ``name`` property of the document to replace.
    tenant_id:
        Tenant whose document is replaced (``None`` = public).

    Returns
    -------
    dict
        A dict with ``path`` and ``version`` of the found document.

    Raises
    ------
    PreconditionError
        If the document is not found or if the match is ambiguous.
    GraphError
        If the graph engine fails.
    """
    rows = get_graph_store().documents.preflight_replaces(replaces_name, tenant_key(tenant_id))

    if len(rows) == 0:
        raise PreconditionError(
            f"replaces={replaces_name!r} not found in the graph: no Document with "
            f"name={replaces_name!r} and latest=true exists. Check the document name."
        )
    if len(rows) > 1:
        paths_str = "\n".join(
            f"  - path={r['path']!r}, version={r['version']!r}" for r in rows
        )
        raise PreconditionError(
            f"replaces={replaces_name!r} is ambiguous: several documents with "
            f"name={replaces_name!r} and latest=true were found:\n"
            f"{paths_str}\n"
            f"Rename your documents to disambiguate before using replaces."
        )
    return rows[0]


def apply_replacement(
    replaces_name: str,
    new_root_doc_names: list[str],
    *,
    tenant_id: str | None = None,
    replaced_path: str | None = None,
) -> None:
    """After ingestion, link the old document to the new root document(s) via HAS_NEWER_VERSION.

    Sets old document latest=False and creates HAS_NEWER_VERSION relationships.
    Both the old and the new documents are looked up within *tenant_id* only,
    so a tenant can never mark another tenant's (or a public) document as
    superseded.

    Opens and closes its own graph connection, unless called inside a
    ``store.sync_scope()`` block.

    Parameters
    ----------
    replaces_name:
        The ``name`` property of the old document being replaced.
    new_root_doc_names:
        Names of the newly ingested root documents (those without an IS_COMPOSED_OF parent).
    tenant_id:
        Tenant of both the old and the new documents (``None`` = public).
    replaced_path:
        Path of the old document, as returned by :func:`preflight_check_replaces`.
        When given, the old document is selected by path instead of by name.

    Raises
    ------
    GraphError
        If the graph engine fails.
    """
    if not new_root_doc_names:
        logger.warning("apply_replacement: no new root documents found; skipping.")
        return
    get_graph_store().documents.apply_replacement(
        replaces_name,
        new_root_doc_names,
        tenant=tenant_key(tenant_id),
        replaced_path=replaced_path,
    )
