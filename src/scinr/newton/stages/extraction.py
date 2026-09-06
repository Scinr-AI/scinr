"""
stages/extraction.py — Stage 1: LLM extraction → Document objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scinr.newton.config import get_config, get_llm, get_llm_semaphore
from scinr.newton.extraction.compact_extraction import compact_extraction, get_active_hierarchy
from scinr.newton.extraction.extraction import ExtractionMaxRetriesError, extract_chunk
from scinr.newton.extraction.structure_consolidation import (
    assemble_tree,
    consolidate_structure,
    delete_map_checkpoint,
    namespace_node_ids,
    write_map_checkpoint,
)
from scinr.newton.models.document_structure import Document
from scinr.newton.results import DocumentResult, StageResult

if TYPE_CHECKING:
    from scinr.newton.converters.base import IntermediateDocument

logger = logging.getLogger(__name__)


async def _run_chunk_extraction(
    document: Document,
    chunks: list[tuple[str, list[str]]],
    page_indices: list[int],
    page_ids_by_index: dict[int, str],
    batch_size: int,
    output_file: Path | None,
    llm,
    fast_extraction: bool,
) -> Document:
    """Runs the legacy sequential or fast-extraction parallel chunk-processing
    loop against *document*, writing intermediate/final output when
    *output_file* is set, and returns the mutated *document*.
    """
    if not fast_extraction:
        # ── Legacy sequential path (fast_extraction=False, always the
        # default) — unchanged. Do not modify a single line in this branch. ──
        for chunk_idx, (prev_page, curr_pages) in enumerate(chunks):
            active_hierarchy = get_active_hierarchy(document)
            curr_start_idx = chunk_idx * batch_size  # position in `pages`/`page_indices`
            curr_page_ids = [
                page_ids_by_index[page_indices[curr_start_idx + i]]
                for i in range(len(curr_pages))
                if page_indices[curr_start_idx + i] in page_ids_by_index
            ] or None

            try:
                async with get_llm_semaphore():
                    nodes = await extract_chunk(
                        prev_page=prev_page,
                        curr_pages=curr_pages,
                        active_hierarchy=active_hierarchy,
                        llm=llm,
                        curr_page_ids=curr_page_ids,
                        user_context=document.context_instructions or "",
                    )
            except ExtractionMaxRetriesError:
                logger.warning(
                    "Chunk %d/%d: all extraction attempts failed — skipping chunk.",
                    chunk_idx + 1, len(chunks),
                )
                continue

            document = compact_extraction(document, nodes)

            # Intermediate crash-safe write if output requested
            if output_file:
                output_file.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    else:
        # ── fast_extraction=True: parallel chunk extraction + consolidation ──
        async def _extract_one_chunk_fast(
            chunk_idx: int, prev_page: str, curr_pages: list[str], curr_page_ids: list[str] | None
        ) -> list:
            try:
                async with get_llm_semaphore():
                    return await extract_chunk(
                        prev_page=prev_page,
                        curr_pages=curr_pages,
                        active_hierarchy="",
                        defer_hierarchy=True,
                        llm=llm,
                        curr_page_ids=curr_page_ids,
                        user_context=document.context_instructions or "",
                    )
            except ExtractionMaxRetriesError:
                logger.warning(
                    "Chunk %d/%d: all extraction attempts failed — skipping chunk.",
                    chunk_idx + 1, len(chunks),
                )
                return []

        chunk_first_abs_page_idx: list[int] = []
        tasks = []
        for chunk_idx, (prev_page, curr_pages) in enumerate(chunks):
            curr_start_idx = chunk_idx * batch_size  # position in `pages`/`page_indices`
            curr_page_ids = [
                page_ids_by_index[page_indices[curr_start_idx + i]]
                for i in range(len(curr_pages))
                if page_indices[curr_start_idx + i] in page_ids_by_index
            ] or None
            chunk_first_abs_page_idx.append(page_indices[curr_start_idx])
            tasks.append(_extract_one_chunk_fast(chunk_idx, prev_page, curr_pages, curr_page_ids))

        all_chunks_nodes = await asyncio.gather(*tasks)

        for nodes, first_abs_page_idx in zip(all_chunks_nodes, chunk_first_abs_page_idx):
            namespace_node_ids(nodes, first_abs_page_idx)

        checkpoint_path: Path | None = None
        if output_file:
            checkpoint_path = output_file.with_name(f"map-checkpoint-{document.document_name}.json")
            write_map_checkpoint(checkpoint_path, list(all_chunks_nodes))

        decisions = await consolidate_structure(list(all_chunks_nodes), llm=llm)
        document.document_structure = assemble_tree(list(all_chunks_nodes), decisions)

        if checkpoint_path is not None:
            delete_map_checkpoint(checkpoint_path)

    # Final write if output requested
    if output_file:
        output_file.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Written: %s", output_file)

    return document


async def extract_one_intermediate(
    doc: IntermediateDocument,
    output_path: Path | None,
    fast_extraction: bool = False,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> Document | None:
    """Process a single IntermediateDocument already in memory.

    No document-level concurrency semaphore of its own — the caller
    (``run_extraction()`` today, a future per-document orchestration engine
    tomorrow) is responsible for bounding how many of these run concurrently.

    Each real LLM call (via ``extract_chunk()``) is bounded internally by the
    global ``get_llm_semaphore()`` so that Stage 1 extraction calls never run
    unbounded relative to the LLM provider's rate limits, regardless of how
    many documents the caller lets run concurrently.

    Parameters
    ----------
    doc:
        IntermediateDocument produced by run_preprocess() (in-memory mode).
    output_path:
        Base output directory for this stage, or None to keep the result
        in-memory only.
    fast_extraction:
        When ``False`` (default), uses the legacy sequential per-chunk
        extraction loop (unchanged). When ``True``, extracts all chunks in
        parallel with cross-chunk hierarchy resolution deferred to a single
        post-extraction consolidation LLM call. See ``run_pipeline()``'s
        docstring for the full tradeoff explanation.
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata. Stamped onto the
        produced Document so it is serialized into the ``extract-*.json``
        written here and later persisted on the :Document node at ingestion.

    Returns
    -------
    Document | None
        The extracted Document, or None if the document had no pages.
    """
    doc_name = doc.document_name or "unnamed"
    pages: list[str] = [p.markdown for p in doc.pages]
    # position -> original absolute page index. In best_effort mode, chunks
    # that were skipped leave gaps in `doc.pages[i].index` (e.g. [0, 1, 4, 5]
    # instead of [0, 1, 2, 3]), so list position no longer equals absolute
    # index. This mapping is used to translate position -> absolute index
    # before looking up page_ids_by_index below.
    page_indices: list[int] = [p.index for p in doc.pages]
    page_ids_by_index: dict[int, str] = {
        p.index: p.page_id
        for p in doc.pages
        if p.page_id
    }
    folder_path = doc.folder_path
    context_instructions = doc.context_instructions
    total_pages = len(pages)

    logger.info("Processing in-memory document: %s (%d page(s))", doc_name, total_pages)

    if total_pages == 0:
        logger.warning("Document has no pages — skipping: %s", doc_name)
        return None

    batch_size = max(1, get_config().extraction_batch_size)
    chunks: list[tuple[str, list[str]]] = [("", pages[:batch_size])]
    for i in range(batch_size, total_pages, batch_size):
        chunks.append((pages[i - 1], pages[i : i + batch_size]))

    doc_path = f"{folder_path}/{doc_name}" if folder_path else doc_name
    ext = ""  # No file extension for in-memory docs

    document = Document(
        document_name=doc_name,
        document_type=ext,
        document_structure=[],
        doc_path=doc_path,
        raw_file_id=doc.raw_file_id or "",
        context_instructions=context_instructions,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
    )
    llm = get_llm()

    # Set up output file if output_folder provided
    if output_path:
        if folder_path:
            out_subdir = output_path / Path(folder_path)
        else:
            out_subdir = output_path
        out_subdir.mkdir(parents=True, exist_ok=True)
        output_file = out_subdir / f"extract-{doc_name}.json"
    else:
        output_file = None

    return await _run_chunk_extraction(
        document, chunks, page_indices, page_ids_by_index, batch_size, output_file, llm, fast_extraction,
    )


async def extract_one_file(
    json_file: Path,
    output_path: Path | None,
    input_folder: Path | None,
    fast_extraction: bool = False,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> Document | None:
    """Process a single Stage 0 JSON intermediate file from disk.

    No document-level concurrency semaphore of its own — the caller
    (``run_extraction()`` today, a future per-document orchestration engine
    tomorrow) is responsible for bounding how many of these run concurrently.

    Each real LLM call (via ``extract_chunk()``) is bounded internally by the
    global ``get_llm_semaphore()`` so that Stage 1 extraction calls never run
    unbounded relative to the LLM provider's rate limits, regardless of how
    many documents the caller lets run concurrently.

    Parameters
    ----------
    json_file:
        Path to the Stage 0 intermediate JSON file to process.
    output_path:
        Base output directory for this stage, or None to keep the result
        in-memory only.
    input_folder:
        The root input folder, used to compute the relative subdirectory
        structure to mirror under output_path. May be None.
    fast_extraction:
        When ``False`` (default), uses the legacy sequential per-chunk
        extraction loop (unchanged). When ``True``, extracts all chunks in
        parallel with cross-chunk hierarchy resolution deferred to a single
        post-extraction consolidation LLM call. See ``run_pipeline()``'s
        docstring for the full tradeoff explanation.
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata. Stamped onto the
        produced Document so it is serialized into the ``extract-*.json``
        written here and later persisted on the :Document node at ingestion.

    Returns
    -------
    Document | None
        The extracted Document, or None if the file had no pages.
    """
    logger.info("Processing file: %s", json_file)

    raw = json.loads(json_file.read_text(encoding="utf-8"))
    pages: list[str] = [page["markdown"] for page in raw["pages"]]
    # position -> original absolute page index. See comment in
    # extract_one_intermediate() for why this is needed when best_effort
    # chunking left gaps in the original page indices.
    page_indices: list[int] = [page["index"] for page in raw["pages"]]
    page_ids_by_index: dict[int, str] = {
        page["index"]: page["page_id"]
        for page in raw["pages"]
        if page.get("page_id")
    }
    folder_path: str | None = raw.get("folder_path")
    context_instructions: str | None = raw.get("context_instructions")
    total_pages = len(pages)
    logger.info("  Total pages: %d", total_pages)

    if total_pages == 0:
        logger.warning("  File has no pages — skipping: %s", json_file)
        return None

    batch_size = max(1, get_config().extraction_batch_size)
    chunks: list[tuple[str, list[str]]] = [("", pages[:batch_size])]
    for i in range(batch_size, total_pages, batch_size):
        chunks.append((pages[i - 1], pages[i : i + batch_size]))

    name, ext = os.path.splitext(json_file)
    nombre = Path(name).name
    doc_path = f"{folder_path}/{nombre}" if folder_path else nombre

    document = Document(
        document_name=nombre,
        document_type=ext,
        document_structure=[],
        doc_path=doc_path,
        raw_file_id=raw.get("raw_file_id") or "",
        context_instructions=context_instructions,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
    )
    llm = get_llm()

    # Determine relative structure to preserve subdir layout
    if input_folder:
        try:
            rel_dir = json_file.relative_to(Path(input_folder)).parent
        except ValueError:
            rel_dir = Path(".")
    else:
        rel_dir = Path(".")

    if output_path:
        output_subdir = output_path / rel_dir
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"extract-{json_file.stem}.json"
    else:
        output_file = None

    return await _run_chunk_extraction(
        document, chunks, page_indices, page_ids_by_index, batch_size, output_file, llm, fast_extraction,
    )


async def run_extraction(
    input_folder: str | None = None,
    output_folder: str | None = None,
    intermediate_documents: list | None = None,
    parallel_docs: int = 1,
) -> tuple[StageResult, list]:
    """Process intermediate JSON files or IntermediateDocument objects through
    the LLM extraction pipeline and produce Document objects.

    Exactly one of *input_folder* or *intermediate_documents* must be provided.
    If *output_folder* is given, the extracted Document JSON is also written to
    disk (mirroring subdirectory structure). If not, documents only exist in
    memory.

    Note: this older batch entry point always calls ``extract_one_intermediate()``/
    ``extract_one_file()`` with ``fast_extraction=False`` — the ``fast_extraction``
    flag is only reachable via ``run_pipeline()``'s per-document-unit engine
    (``_process_document_unit()``). This is a deliberate scope boundary, not
    an oversight.

    Parameters
    ----------
    input_folder:
        Path to the directory containing input JSON files from Stage 0
        (searched recursively). Mutually exclusive with *intermediate_documents*.
    output_folder:
        Path to the directory where extraction output files will be written.
        If None, extraction output is only available in memory.
    intermediate_documents:
        List of IntermediateDocument objects from run_preprocess() (in-memory
        mode). Mutually exclusive with *input_folder*.
    parallel_docs:
        Maximum number of documents to process concurrently.

    Returns
    -------
    tuple[StageResult, list[Document]]
        A StageResult with counts and errors, and a list of Document objects
        (one per successfully extracted document).
    """
    t0 = time.monotonic()

    # ── Validate inputs ────────────────────────────────────────────────────────
    if input_folder is not None and intermediate_documents is not None:
        raise ValueError(
            "input_folder and intermediate_documents are mutually exclusive. "
            "Provide one or the other, not both."
        )
    if input_folder is None and intermediate_documents is None:
        raise ValueError(
            "Either input_folder or intermediate_documents must be provided."
        )

    output_path = Path(output_folder) if output_folder else None

    semaphore = asyncio.Semaphore(parallel_docs)

    async def _process_intermediate(doc: IntermediateDocument) -> Document | None:
        """Process an IntermediateDocument in-memory (no disk read)."""
        async with semaphore:
            return await extract_one_intermediate(doc, output_path)

    async def _process_file(json_file: Path) -> Document | None:
        """Process a JSON file from disk."""
        async with semaphore:
            return await extract_one_file(
                json_file,
                output_path,
                Path(input_folder) if input_folder else None,
            )

    # ── Dispatch: in-memory or from-disk ──────────────────────────────────────
    extracted_docs: list[Document] = []
    doc_results: list[DocumentResult] = []

    if intermediate_documents is not None:
        tasks = [_process_intermediate(doc) for doc in intermediate_documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            src_name = (intermediate_documents[i].document_name or f"doc_{i}")
            if isinstance(result, Exception):
                logger.error("Extraction failed for '%s': %s", src_name, result)
                doc_results.append(DocumentResult(document_name=src_name, nodes_processed=0, nodes_failed=1, errors=[str(result)]))
            elif result is None:
                doc_results.append(DocumentResult(document_name=src_name, nodes_processed=0, nodes_failed=1, errors=["No pages to process"]))
            else:
                extracted_docs.append(result)
                doc_results.append(DocumentResult(document_name=result.document_name, nodes_processed=1, nodes_failed=0))
    else:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: '{input_folder}'.")
        json_files = sorted(input_path.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in '{input_folder}'.")
        tasks = [_process_file(f) for f in json_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            src_name = json_files[i].stem
            if isinstance(result, Exception):
                logger.error("Extraction failed for '%s': %s", src_name, result)
                doc_results.append(DocumentResult(document_name=src_name, nodes_processed=0, nodes_failed=1, errors=[str(result)]))
            elif result is None:
                doc_results.append(DocumentResult(document_name=src_name, nodes_processed=0, nodes_failed=1, errors=["No pages to process"]))
            else:
                extracted_docs.append(result)
                doc_results.append(DocumentResult(document_name=result.document_name, nodes_processed=1, nodes_failed=0))

    total_failed = sum(1 for r in doc_results if r.nodes_failed > 0)
    duration = time.monotonic() - t0
    stage_result = StageResult(
        stage="extraction",
        success=total_failed == 0,
        documents=doc_results,
        total_processed=len(extracted_docs),
        total_failed=total_failed,
        duration_seconds=duration,
    )
    return stage_result, extracted_docs
