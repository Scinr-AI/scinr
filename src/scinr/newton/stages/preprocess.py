"""
stages/preprocess.py — Stage 0: Convert raw files to intermediate JSON.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from scinr.newton.results import DocumentResult, StageResult
from scinr.newton.storage.factory import get_storage  # noqa: F401 — imported inside function

logger = logging.getLogger(__name__)


async def run_preprocess(
    input_raw: str,
    output_dir: str | None = None,
    context_instructions: str | None = None,
) -> tuple[StageResult, list]:
    """Convert raw source files to intermediate JSON using the converters module.

    Also stores raw files (binary) and converted pages in MongoDB via the
    storage layer when a backend is configured (``STORAGE_BACKEND`` env var).

    Parameters
    ----------
    input_raw:
        Path to the folder containing raw source files (PDF, DOCX, XLSX, etc.).
    output_dir:
        Path to the output folder where intermediate JSON files will be written.
        If None, converted documents are only available in memory; no files are
        written to disk.
    context_instructions:
        Optional free-text context about the documents being processed.

    Returns
    -------
    tuple[StageResult, list[IntermediateDocument]]
        A StageResult with counts and errors, and a list of IntermediateDocument
        objects (one per successfully converted file).
    """
    import shutil
    import tempfile

    from scinr.newton.converters.main import convert_folder  # deferred import

    t0 = time.monotonic()
    raw_file_repo, page_repo = get_storage()

    # If no output_dir specified, use a temp dir and clean it up after
    _temp_dir: str | None = None
    if output_dir is None:
        _temp_dir = tempfile.mkdtemp(prefix="scinr_preprocess_")
        effective_output_dir = _temp_dir
    else:
        effective_output_dir = output_dir

    try:
        results = await convert_folder(
            Path(input_raw),
            Path(effective_output_dir),
            raw_file_repo=raw_file_repo,
            page_repo=page_repo,
            context_instructions=context_instructions,
        )
    except Exception as exc:
        duration = time.monotonic() - t0
        if _temp_dir:
            shutil.rmtree(_temp_dir, ignore_errors=True)
        return (
            StageResult(
                stage="preprocess",
                success=False,
                documents=[],
                total_processed=0,
                total_failed=0,
                duration_seconds=duration,
                errors=[str(exc)],
            ),
            [],
        )

    intermediate_docs = []
    doc_results = []
    for raw_path, _json_path, doc in results:
        intermediate_docs.append(doc)
        doc_results.append(
            DocumentResult(
                document_name=doc.document_name or raw_path.stem,
                nodes_processed=1,
                nodes_failed=0,
            )
        )

    # Clean up temp dir if we used one (docs already captured in memory)
    if _temp_dir:
        shutil.rmtree(_temp_dir, ignore_errors=True)

    duration = time.monotonic() - t0
    stage_result = StageResult(
        stage="preprocess",
        success=True,
        documents=doc_results,
        total_processed=len(doc_results),
        total_failed=0,
        duration_seconds=duration,
    )
    return stage_result, intermediate_docs
