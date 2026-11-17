"""
stages/tabular.py — Stage 5: Direct CSV/XLSX ingestion into Neo4j.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from scinr.newton.graph import get_graph_store
from scinr.newton.progress import _NULL_EMITTER, _ProgressEmitter
from scinr.newton.results import DocumentResult, StageResult
from scinr.newton.tabular.agent import decide_content_type
from scinr.newton.tabular.reader import SheetScan, scan_tabular_file
from scinr.newton.utils.naming import source_document_name, tabular_doc_path

logger = logging.getLogger(__name__)


async def run_tabular_pipeline(
    input_raw: str,
    update_mode: bool = False,
    parallel_docs: int = 1,
    tabular_extensions: set | None = None,
    tabular_delimiter: str | None = None,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
    *,
    files: Sequence[tuple[Path, int]] | None = None,
    emitter: _ProgressEmitter = _NULL_EMITTER,
    batch_version: int | None = None,
) -> StageResult:
    """Ingest all tabular files (CSV/XLSX/XLS) in *input_raw* directly into Neo4j.

    Bypasses Stages 0-4 entirely. For each file:
      1. Scans it (headers, row count, 5-row preview). A file that cannot be read,
         or whose sheets have no data rows, is **skipped**: no raw file is stored
         and no :Document is created.
      2. Stores the raw file and creates the :Document node.
      3. Per sheet: one LLM call to decide the extraction model, one to map
         columns to model fields, then writes the Table + Row StructureNode
         subgraph to Neo4j directly.

    Each file is reported in the returned :class:`StageResult` as one
    :class:`DocumentResult` whose counters are **sheets**: ``nodes_processed`` =
    tables written, ``nodes_failed`` = sheets with errors (at most one per error
    the agent recorded). A file is ``"success"`` when every sheet was written
    without errors, ``"partial"`` when at least one table was written but some
    sheet failed, ``"failed"`` when no table was written, and ``"skipped"``
    (``0`` / ``0``, so it never fails the batch) when it was rejected by step 1.

    Parameters
    ----------
    input_raw:
        Folder containing raw tabular files (searched recursively).
    update_mode:
        If True, wipe existing Table/Row subgraph and re-insert.
    parallel_docs:
        Maximum number of files to process concurrently.
    tabular_extensions:
        Set of file extensions to treat as tabular. Defaults to
        {'.csv', '.xlsx', '.xls'} when None.
    tabular_delimiter:
        Field delimiter for CSV files. When None, uses the default
        delimiter of the tabular agent.
    tenant_id:
        Tenant owning every document this pipeline creates (``None`` = public).
        Part of the :Document identity: versions are resolved within this
        tenant only, and another tenant's document at the same path is never
        touched.
    created_by_user_id, job_id:
        Optional caller-supplied provenance metadata written verbatim onto
        every :Document node this pipeline creates (leaf tabular documents and
        their ancestor folder-parent nodes). Always SET (null when omitted).
    files:
        Tabular files already discovered by the caller, each with its
        ``document_index`` for progress events. ``None`` discovers them under
        *input_raw* (sorted, indices ``0..N-1``).
    emitter:
        Progress emitter of the run (``run_pipeline()`` passes its own). Each
        file emits ``document_started`` → ``stage_started`` →
        ``stage_completed`` / ``stage_failed`` → ``document_finished`` with
        ``stage="tabular"`` and its ``document_index``; a skipped file emits
        ``document_finished(status="skipped")`` without a stage outcome.
    batch_version:
        Batch version already resolved by the caller (``run_pipeline()`` shares
        one version across tabular and document units). ``None`` resolves it
        here from this run's tabular paths and *tenant_id*.

    Returns
    -------
    StageResult
        Stage result with per-document ingestion counts and errors.
    """
    from scinr.newton.annotation.catalog import (
        ensure_catalog_models_once,
        ensure_theme_structure_once,
    )
    from scinr.newton.tabular.agent import run_tabular_agent
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    t0 = time.monotonic()
    _TABULAR_EXTS = tabular_extensions if tabular_extensions is not None else {".csv", ".xlsx", ".xls"}
    input_path = Path(input_raw)

    if not input_path.exists():
        raise FileNotFoundError(f"input_raw path not found: {input_raw}")

    if files is None:
        discovered = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in _TABULAR_EXTS
        )
        files = [(f, i) for i, f in enumerate(discovered)]
    tabular_files = [f for f, _ in files]
    if not tabular_files:
        logger.warning("run_tabular_pipeline: no tabular files found in '%s'", input_raw)
        duration = time.monotonic() - t0
        return StageResult(
            stage="tabular",
            success=True,
            documents=[],
            total_processed=0,
            total_failed=0,
            duration_seconds=duration,
        )

    logger.info(
        "run_tabular_pipeline: found %d tabular file(s) in '%s'",
        len(tabular_files), input_raw,
    )

    def _file_to_doc_path(f: Path) -> tuple[str, str]:
        return source_document_name(f), tabular_doc_path(f, input_path)

    file_infos = [_file_to_doc_path(f) for f in tabular_files]
    all_doc_paths = list({dp for _, dp in file_infos})

    store = get_graph_store()
    doc_results: list[DocumentResult] = []
    with store.sync_scope():
        store.schema.setup()

        logger.info("Ensuring catalog models are loaded")

        await ensure_catalog_models_once(theme_registry)

        logger.info("Ensuring themes are loaded")

        await ensure_theme_structure_once(theme_registry)
        logger.info("Setup of theme and catalog complete")
        if batch_version is not None:
            resolved_version = batch_version
        else:
            from scinr.newton.ingest.loader import _extract_all_paths

            logger.info("Verifying previous documents")
            # Versions are numbered per tenant (tenant_id None -> public scope).
            resolved_version = await asyncio.to_thread(
                store.documents.resolve_batch_version,
                _extract_all_paths(all_doc_paths),
                [tenant_id],
                update_mode,
            )

        logger.info("run_tabular_pipeline: batch version=%d (update_mode=%s)", resolved_version, update_mode)

        semaphore = asyncio.Semaphore(parallel_docs)

        from scinr.newton.storage.factory import get_storage
        try:
            _raw_file_repo, _ = get_storage()
        except Exception as exc:
            logger.warning("Storage backend unavailable: %s. Continuing without persisting raw files.", exc)
            from scinr.newton.storage.null import NullRawFileRepository
            _raw_file_repo = NullRawFileRepository()

        async def _process_file(f: Path, doc_name: str, doc_path: str, index: int) -> DocumentResult:
            def _emit(kind: str, **fields) -> None:
                emitter.emit(
                    kind,
                    document_index=index,
                    doc_path=doc_path,
                    document_name=doc_name,
                    source_path=str(f),
                    tenant_id=tenant_id,
                    **fields,
                )

            async with semaphore:
                logger.info("run_tabular_pipeline: processing '%s'", f.name)
                file_t0 = time.monotonic()
                _emit("document_started", planned_stages=("tabular",))
                _emit("stage_started", stage="tabular")

                def _fail(error: str, nodes_processed: int = 0, nodes_failed: int = 1) -> DocumentResult:
                    duration = time.monotonic() - file_t0
                    _emit(
                        "stage_failed",
                        stage="tabular",
                        duration_s=duration,
                        metrics={"nodes_processed": nodes_processed, "nodes_failed": nodes_failed},
                        error=error,
                    )
                    _emit(
                        "document_finished",
                        status="failed",
                        stopped_at="tabular",
                        error=error,
                        duration_s=duration,
                    )
                    return DocumentResult(
                        document_name=doc_name,
                        nodes_processed=nodes_processed,
                        nodes_failed=nodes_failed,
                        errors=[error],
                    )

                try:
                    # Reject unreadable / empty files before anything is
                    # persisted (no raw file, no :Document).
                    skip_reason: str | None = None
                    scans: list[SheetScan] = []
                    try:
                        scans = await asyncio.to_thread(scan_tabular_file, f, tabular_delimiter)
                    except Exception as scan_exc:
                        skip_reason = f"unreadable tabular file: {scan_exc}"
                    else:
                        if all(sc["total_rows"] == 0 for sc in scans):
                            skip_reason = "tabular file has no data rows"
                    if skip_reason is not None:
                        logger.warning("run_tabular_pipeline: skipping '%s': %s", f, skip_reason)
                        _emit(
                            "document_finished",
                            status="skipped",
                            error=skip_reason,
                            duration_s=time.monotonic() - file_t0,
                        )
                        return DocumentResult(
                            document_name=doc_name, nodes_processed=0, nodes_failed=0
                        )

                    raw_file_id = None
                    if _raw_file_repo is not None:
                        try:
                            _content_type = decide_content_type(f.suffix.lower())
                            folder_path_str = str(f.parent.relative_to(input_path)) if f.parent != input_path else ""
                            raw_file_id = await _raw_file_repo.store_file(
                                path=f,
                                filename=f.name,
                                content_type=_content_type,
                                folder_path=folder_path_str,
                                tenant_id=tenant_id,
                                created_by_user_id=created_by_user_id,
                                job_id=job_id,
                            )
                        except Exception as store_exc:
                            logger.warning(
                                "run_tabular_pipeline: failed to store raw file '%s': %s",
                                f.name, store_exc,
                            )

                    final_state = await run_tabular_agent(
                        file_path=f,
                        document_name=doc_name,
                        doc_path=doc_path,
                        resolved_version=resolved_version,
                        update_mode=update_mode,
                        raw_file_id=raw_file_id,
                        tenant_id=tenant_id,
                        created_by_user_id=created_by_user_id,
                        job_id=job_id,
                        delimiter=tabular_delimiter,
                        scans=scans,
                    )
                except Exception as exc:
                    logger.exception("run_tabular_pipeline: failed to process '%s': %s", f.name, exc)
                    return _fail(str(exc))

                errors = [str(e) for e in final_state.get("errors") or []]
                tables = len(final_state.get("ingested_table_node_ids") or [])
                if tables == 0:
                    logger.warning(
                        "run_tabular_pipeline: '%s' wrote no table: %s", f.name, errors,
                    )
                    return _fail(
                        "; ".join(errors) or "no table was written",
                        nodes_failed=max(1, min(len(errors), len(scans))),
                    )

                nodes_failed = min(len(errors), len(scans))
                duration = time.monotonic() - file_t0
                _emit(
                    "stage_completed",
                    stage="tabular",
                    duration_s=duration,
                    metrics={"nodes_processed": tables, "nodes_failed": nodes_failed},
                )
                if errors:
                    logger.warning(
                        "run_tabular_pipeline: '%s' completed with errors: %s", f.name, errors,
                    )
                    _emit(
                        "document_finished",
                        status="partial",
                        error="; ".join(errors),
                        duration_s=duration,
                    )
                else:
                    _emit("document_finished", status="success", duration_s=duration)
                return DocumentResult(
                    document_name=doc_name,
                    nodes_processed=tables,
                    nodes_failed=nodes_failed,
                    errors=errors,
                )

        tasks = [
            _process_file(f, doc_name, doc_path, index)
            for (f, index), (doc_name, doc_path) in zip(files, file_infos, strict=True)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                doc_results.append(DocumentResult(
                    document_name=file_infos[i][0], nodes_processed=0, nodes_failed=1,
                    errors=[str(result) or type(result).__name__],
                ))
            else:
                doc_results.append(result)

        written = sum(1 for r in doc_results if r.nodes_processed > 0)
        logger.info("run_tabular_pipeline: complete. %d/%d file(s) written.", written, len(tabular_files))

    total_failed = sum(r.nodes_failed for r in doc_results)
    duration = time.monotonic() - t0
    return StageResult(
        stage="tabular",
        success=total_failed == 0,
        documents=doc_results,
        total_processed=sum(r.nodes_processed for r in doc_results),
        total_failed=total_failed,
        duration_seconds=duration,
    )
