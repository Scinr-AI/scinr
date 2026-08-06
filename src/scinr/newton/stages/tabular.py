"""
stages/tabular.py — Stage 5: Direct CSV/XLSX ingestion into Neo4j.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from scinr.newton.ingest.config import get_async_driver, get_driver
from scinr.newton.ingest.schema import setup_schema
from scinr.newton.results import DocumentResult, StageResult
from scinr.newton.tabular.agent import decide_content_type

logger = logging.getLogger(__name__)


async def run_tabular_pipeline(
    input_raw: str,
    update_mode: bool = False,
    parallel_docs: int = 1,
    tabular_extensions: set | None = None,
    tabular_delimiter: str | None = None,
) -> StageResult:
    """Ingest all tabular files (CSV/XLSX/XLS) in *input_raw* directly into Neo4j.

    Bypasses Stages 0-4 entirely. For each file:
      1. Reads headers + 5-row preview.
      2. Makes one LLM call to decide the extraction model.
      3. Makes one LLM call to map columns to model fields.
      4. Writes Table + Row StructureNode subgraph to Neo4j directly.

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

    Returns
    -------
    StageResult
        Stage result with per-document ingestion counts and errors.
    """
    from scinr.newton.annotation.neo4j_ops import (
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
        raise FileNotFoundError(f"--input-raw path not found: {input_raw}")

    tabular_files = sorted(
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _TABULAR_EXTS
    )
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
        doc_name = f.stem
        try:
            rel = f.relative_to(input_path)
            folder = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else ""
        except ValueError:
            folder = ""
        doc_path = f"{folder}/{doc_name}" if folder else doc_name
        return doc_name, doc_path

    file_infos = [_file_to_doc_path(f) for f in tabular_files]
    all_doc_paths = list({dp for _, dp in file_infos})
    expanded_paths: set[str] = set()
    for dp in all_doc_paths:
        expanded_paths.add(dp)
        parts = dp.split("/")
        for i in range(1, len(parts)):
            expanded_paths.add("/".join(parts[:i]))
    all_paths = list(expanded_paths)

    driver = get_driver()
    doc_results: list[DocumentResult] = []
    try:
        setup_schema(driver)

        # Fix: ensure_catalog_models and ensure_theme_structure are async and need AsyncDriver
        logger.info("Getting async neo4j driver")
        
        async_driver = get_async_driver()
        
        logger.info("Ensuring catalog models are loaded")
        
        await ensure_catalog_models_once(async_driver)
        
        logger.info("Ensuring themes are loaded")
        
    
        await ensure_theme_structure_once(async_driver, theme_registry)
        logger.info("Setup of theme and catalog complete")

        with driver.session() as session:
            logger.info("Verifying previous documents")
            if update_mode:
                result = session.run(
                    "MATCH (d:Document {latest: true}) WHERE d.path IN $paths "
                    "RETURN max(d.version) AS max_version",
                    paths=all_paths,
                )
            else:
                
                result = session.run(
                    "MATCH (d:Document) WHERE d.path IN $paths "
                    "RETURN max(d.version) AS max_version",
                    paths=all_paths,
                )
            record = result.single()
            max_version = record["max_version"] if record else None
            resolved_version = max_version if (update_mode and max_version is not None) else ((max_version + 1) if max_version is not None else 1)
            logger.info("Max version of ingested document selected")
            

        logger.info("run_tabular_pipeline: batch version=%d (update_mode=%s)", resolved_version, update_mode)

        semaphore = asyncio.Semaphore(parallel_docs)

        from scinr.newton.storage.factory import get_storage
        try:
            _raw_file_repo, _ = get_storage()
        except Exception as exc:
            logger.warning("Storage backend unavailable: %s. Continuing without persisting raw files.", exc)
            from scinr.newton.storage.null import NullRawFileRepository
            _raw_file_repo = NullRawFileRepository()

        async def _process_file(f: Path, doc_name: str, doc_path: str) -> str | None:
            async with semaphore:
                logger.info("run_tabular_pipeline: processing '%s'", f.name)
                try:
                    _driver = get_driver()
                    try:
                        raw_file_id = None
                        if _raw_file_repo is not None:
                            try:
                                raw_bytes = f.read_bytes()
                                _content_type = decide_content_type(f.suffix.lower())
                                folder_path_str = str(f.parent.relative_to(input_path)) if f.parent != input_path else ""
                                raw_file_id = await _raw_file_repo.store(
                                    filename=f.name,
                                    content=raw_bytes,
                                    content_type=_content_type,
                                    folder_path=folder_path_str,
                                )
                            except Exception as store_exc:
                                logger.warning(
                                    "run_tabular_pipeline: failed to store raw file '%s': %s",
                                    f.name, store_exc,
                                )

                        agent_kwargs = dict(
                            file_path=f,
                            document_name=doc_name,
                            doc_path=doc_path,
                            driver=_driver,
                            resolved_version=resolved_version,
                            update_mode=update_mode,
                            raw_file_id=raw_file_id,
                        )
                        if tabular_delimiter is not None:
                            agent_kwargs["delimiter"] = tabular_delimiter

                        final_state = await run_tabular_agent(**agent_kwargs)
                        if final_state.get("errors"):
                            logger.warning(
                                "run_tabular_pipeline: '%s' completed with errors: %s",
                                f.name, final_state["errors"],
                            )
                        return doc_name
                    finally:
                        _driver.close()
                except Exception as exc:
                    logger.exception("run_tabular_pipeline: failed to process '%s': %s", f.name, exc)
                    return None

        tasks = [
            _process_file(f, doc_name, doc_path)
            for f, (doc_name, doc_path) in zip(tabular_files, file_infos)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            doc_name = file_infos[i][0]
            if isinstance(result, Exception) or result is None:
                err = str(result) if isinstance(result, Exception) else "Processing failed"
                doc_results.append(DocumentResult(document_name=doc_name, nodes_processed=0, nodes_failed=1, errors=[err]))
            else:
                doc_results.append(DocumentResult(document_name=doc_name, nodes_processed=1, nodes_failed=0))

        success_count = sum(1 for r in doc_results if r.nodes_processed > 0)
        logger.info("run_tabular_pipeline: complete. %d/%d file(s) processed.", success_count, len(tabular_files))
    finally:
        driver.close()

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
