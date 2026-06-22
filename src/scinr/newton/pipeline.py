"""
pipeline.py — Unified pipeline orchestrator for scinr-ingest.

Provides run_pipeline(), a single entry point that chains all five stages:

    Stage 0: preprocess    — Convert raw files to IntermediateDocument objects
    Stage 1: extraction    — Extract structure via LLM → Document objects
    Stage 2: ingestion     — Ingest Document objects into Neo4j
    Stage 3: annotation    — Annotate StructureNodes with model decisions
    Stage 4: entity_extraction — Extract entities from annotated nodes

Usage
-----
Minimal full pipeline::

    import asyncio
    from scinr.newton.config import configure
    from scinr.newton.pipeline import run_pipeline

    configure(llm=my_llm, neo4j_user="neo4j", neo4j_password="...")
    result = asyncio.run(run_pipeline(input_raw="files/"))

Skip Stage 0 (reuse previous converter output)::

    result = asyncio.run(run_pipeline(
        extraction_input_dir="data/json/",
    ))

Annotation-only run::

    result = asyncio.run(run_pipeline(
        stages=["annotation", "entity_extraction"],
        document_names=["MyDocument"],
    ))

Control concurrency before calling (node-level and Neo4j)::

    configure(llm=my_llm, ..., llm_concurrency=8, neo4j_concurrency=20)
    result = asyncio.run(run_pipeline(input_raw="files/", parallel_docs=4))
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

from scinr.newton.results import DocumentResult, PipelineResult, StageResult

logger = logging.getLogger(__name__)

# Valid stage names
_VALID_STAGES = {
    "preprocess",
    "extraction",
    "ingestion",
    "annotation",
    "entity_extraction",
    "tabular",
}

_DEFAULT_STAGES = [
    "preprocess",
    "extraction",
    "ingestion",
    "annotation",
    "entity_extraction",
]

_DEFAULT_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}


async def run_pipeline(
    # ── Raw input (Stage 0 source) ────────────────────────────────────────────
    input_raw: str | None = None,
    # ── Directory params — control data flow and stage skipping ──────────────
    converter_output_dir: str | None = None,
    extraction_input_dir: str | None = None,
    extraction_output_dir: str | None = None,
    ingestion_input_dir: str | None = None,
    # ── Stage selection ───────────────────────────────────────────────────────
    stages: list[str] | None = None,
    # ── Document identity for annotation / entity_extraction only runs ────────
    document_names: list[str] | None = None,
    document_names_dir: str | None = None,
    # ── Annotation options ────────────────────────────────────────────────────
    manual: bool = False,
    model_class: str | None = None,
    only_unannotated: bool = False,
    only_unextracted: bool = False,
    context_instructions: str | None = None,
    # ── Versioning / replacement ──────────────────────────────────────────────
    update_mode: bool = False,
    replaces: str | None = None,
    # ── Parallelism ───────────────────────────────────────────────────────────
    parallel_docs: int = 1,
    # ── Behaviour on partial failure ─────────────────────────────────────────
    on_partial_failure: Literal["abort", "continue", "warn"] = "abort",
    # ── Tabular options (auto-detected from input_raw) ────────────────────────
    tabular_extensions: set[str] | None = None,
    tabular_delimiter: str | None = None,
) -> PipelineResult:
    """Orchestrate the scinr-ingest pipeline end-to-end.

    Chains Stages 0-4 in sequence, passing data between stages in memory when
    intermediate directory parameters are omitted. Tabular files (.csv, .xlsx,
    .xls) found in *input_raw* are automatically routed to the tabular pipeline.

    Concurrency for LLM calls (``llm_concurrency``) and Neo4j writes
    (``neo4j_concurrency``) must be configured via
    :func:`~scinr.newton.config.configure` **before** calling this function.
    Use ``parallel_docs`` to control how many documents are processed
    concurrently across all stages.

    Parameters
    ----------
    input_raw:
        Folder containing raw source files (PDF, DOCX, CSV, XLSX, …) for
        Stage 0. Required when ``stages`` includes ``"preprocess"`` and
        ``extraction_input_dir`` is not given. CSV/XLSX files found here are
        automatically processed via the tabular pipeline.
    converter_output_dir:
        Where Stage 0 **writes** its intermediate JSON files to disk. When
        ``None``, Stage 0 only returns ``IntermediateDocument`` objects in
        memory without persisting anything.
    extraction_input_dir:
        Where Stage 1 **reads** its JSON input from disk, skipping Stage 0
        entirely. Mutually exclusive with ``input_raw``.
    extraction_output_dir:
        Where Stage 1 **writes** its ``extract-*.json`` output to disk. When
        ``None``, Stage 1 only returns ``Document`` objects in memory.
    ingestion_input_dir:
        Where Stage 2 **reads** ``extract-*.json`` files from disk, skipping
        Stages 0 and 1. Mutually exclusive with ``input_raw`` and
        ``extraction_input_dir``.
    stages:
        Ordered list of stages to execute. Valid values: ``"preprocess"``,
        ``"extraction"``, ``"ingestion"``, ``"annotation"``,
        ``"entity_extraction"``, ``"tabular"``. ``None`` runs the full
        pipeline (all five non-tabular stages). ``"tabular"`` cannot be
        combined with other stages; it is handled automatically when tabular
        files are detected in ``input_raw``.
    document_names:
        Explicit list of Neo4j ``document_name`` values for Stage 3/4 when
        running annotation or entity extraction without running Stage 2.
        Cannot be combined with ``document_names_dir``.
    document_names_dir:
        Folder containing ``extract-*.json`` files whose ``document_name``
        fields are read to build the document list for Stage 3/4. Used when
        Stage 2 ran in a prior invocation. Ignored (raises ``ValueError``)
        when ``document_names`` is also provided.
    manual:
        If ``True``, annotation (Stage 3) assigns ``model_class`` to all
        qualifying nodes without calling the LLM. Requires ``model_class``.
    model_class:
        CamelCase model class name for manual annotation. Required when
        ``manual=True``. Must not be provided without ``manual=True``.
    only_unannotated:
        Stage 3 only: skip nodes that already have a ``:HAS_MODEL_DECISION``
        relationship. Useful for resuming an interrupted annotation run.
    only_unextracted:
        Stage 4 only: skip nodes that already have a ``:HAS_EXTRACTION``
        relationship. Useful for resuming an interrupted extraction run.
    context_instructions:
        Free-text context about the documents, injected into Stage 0
        (converter) and Stage 3 (annotation agent).
    update_mode:
        If ``True``, Stage 2 replaces the latest document version in Neo4j
        without creating a new version. Only one source file is allowed when
        ``update_mode=True``. Cannot be combined with ``replaces``.
    replaces:
        ``document_name`` of an existing Neo4j document being superseded by
        the newly ingested content. Verified in Neo4j before any stage runs.
        Cannot be combined with ``update_mode``.
    parallel_docs:
        Maximum number of documents processed concurrently across all stages,
        including tabular. Must be >= 1.
    on_partial_failure:
        Controls behaviour when a stage returns ``success=False``:

        - ``"abort"`` (default): stop and return with the stages executed so
          far (``PipelineResult.success=False``).
        - ``"continue"``: proceed to the next stage regardless.
        - ``"warn"``: log a warning and proceed.
    tabular_extensions:
        File extensions treated as tabular. Defaults to
        ``{'.csv', '.xlsx', '.xls'}`` when ``None``.
    tabular_delimiter:
        Field delimiter for CSV files forwarded to the tabular agent. Uses
        the agent's default when ``None``.

    Returns
    -------
    PipelineResult
        Aggregated result containing a :class:`~results.StageResult` for
        every stage that was executed, plus overall ``success`` and
        ``total_duration_seconds``.

    Raises
    ------
    ValueError
        If any parameter combination is invalid (see validation section).
    FileNotFoundError
        If a required directory does not exist.
    """
    # ── Imports (deferred to avoid circular imports at module level) ──────────
    from scinr.newton.ingest.config import get_driver
    from scinr.newton.stages import (
        apply_replacement,
        preflight_check_replaces,
        run_annotation,
        run_entity_extraction,
        run_extraction,
        run_ingestion,
        run_preprocess,
        run_tabular_pipeline,
    )

    pipeline_t0 = time.monotonic()
    effective_stages = list(stages) if stages is not None else list(_DEFAULT_STAGES)

    # ── 1. Validate stage names ───────────────────────────────────────────────
    invalid = set(effective_stages) - _VALID_STAGES
    if invalid:
        raise ValueError(
            f"Unknown stage(s): {sorted(invalid)!r}. "
            f"Valid values: {sorted(_VALID_STAGES)!r}."
        )

    # ── 2. Tabular exclusivity ────────────────────────────────────────────────
    if "tabular" in effective_stages and len(effective_stages) > 1:
        raise ValueError(
            "'tabular' stage cannot be combined with other stages. "
            "Use stages=['tabular'] alone, or omit 'tabular' and let the pipeline "
            "detect tabular files automatically from input_raw."
        )
    # ── 3. update_mode + replaces mutual exclusion ───────────────────────────
    if update_mode and replaces is not None:
        raise ValueError(
            "update_mode=True and replaces cannot be used together. "
            "update_mode fixes the current version in-place; "
            "replaces links a new document as the successor of an existing one."
        )

    # ── 4. manual / model_class cross-validation ─────────────────────────────
    if model_class is not None and not manual:
        raise ValueError(
            "model_class requires manual=True. "
            "Either set manual=True or remove model_class."
        )
    if manual and not model_class:
        raise ValueError(
            "manual=True requires model_class. "
            "Provide the CamelCase model class name via model_class=."
        )
    if manual and "annotation" not in effective_stages:
        raise ValueError(
            "manual=True is only valid when 'annotation' is in stages."
        )

    # ── 5. document_names + document_names_dir mutual exclusion ──────────────
    if document_names is not None and document_names_dir is not None:
        raise ValueError(
            "document_names and document_names_dir cannot both be provided. "
            "Provide one or the other."
        )

    # ── 6. Input source mutual exclusions ────────────────────────────────────
    if input_raw is not None and extraction_input_dir is not None:
        raise ValueError(
            "input_raw and extraction_input_dir are mutually exclusive. "
            "input_raw activates Stage 0; extraction_input_dir skips to Stage 1."
        )
    if input_raw is not None and ingestion_input_dir is not None:
        raise ValueError(
            "input_raw and ingestion_input_dir are mutually exclusive. "
            "ingestion_input_dir skips Stages 0 and 1."
        )
    if extraction_input_dir is not None and ingestion_input_dir is not None:
        raise ValueError(
            "extraction_input_dir and ingestion_input_dir are mutually exclusive. "
            "ingestion_input_dir skips both Stage 0 and Stage 1."
        )

    # ── 7. Stage-specific input availability ─────────────────────────────────
    if "extraction" in effective_stages and extraction_input_dir is None:
        if input_raw is None and converter_output_dir is None and "preprocess" not in effective_stages:
            raise ValueError(
                "'extraction' stage requires an input source if executed as first stage. Provide one of:\n"
                "  - input_raw and add preprocess stage (raw source files for Stage 0)\n"
                "  - converter_output_dir (pre-converted JSON from a prior Stage 0)\n"
                "  - extraction_input_dir (skip Stage 0 and read from this folder)"
            )

    if "ingestion" in effective_stages and ingestion_input_dir is None:
        if extraction_output_dir is None and "extraction" not in effective_stages:
            raise ValueError(
                "'ingestion' stage requires an input source. Provide one of:\n"
                "  - extraction_output_dir (persisted Stage 1 output)\n"
                "  - ingestion_input_dir (skip Stage 1 and read from this folder)\n"
                "  - include 'extraction' in stages so Stage 1 provides the input"
            )

    needs_doc_names = (
        ("annotation" in effective_stages or "entity_extraction" in effective_stages)
        and ingestion_input_dir is None
        and "ingestion" not in effective_stages
    )
    if needs_doc_names and document_names is None and document_names_dir is None:
        raise ValueError(
            "'annotation' and/or 'entity_extraction' stages require document names. "
            "Provide one of:\n"
            "  - document_names (explicit list of Neo4j document_name values)\n"
            "  - document_names_dir (folder with extract-*.json files)\n"
            "  - include 'ingestion' in stages so Stage 2 provides the names"
        )

    if "tabular" in effective_stages and input_raw is None:
        raise ValueError(
            "'tabular' stage requires input_raw (folder containing tabular files)."
        )
    if "preprocess" in effective_stages and input_raw is None:
        raise ValueError(
            "'preprocess' stage requires input_raw (folder containing tabular files)."
        )

    # ── 8. parallel_docs range guard ─────────────────────────────────────────
    if parallel_docs < 1:
        raise ValueError(f"parallel_docs must be >= 1, got {parallel_docs}.")

    # ── 9. Determine skip logic ───────────────────────────────────────────────
    if ingestion_input_dir is not None:
        effective_stages = [s for s in effective_stages if s not in ("preprocess", "extraction")]
        logger.info(
            "Skipping 'preprocess' and 'extraction': ingestion_input_dir='%s'",
            ingestion_input_dir,
        )
    elif extraction_input_dir is not None:
        effective_stages = [s for s in effective_stages if s != "preprocess"]
        logger.info(
            "Skipping 'preprocess': extraction_input_dir='%s'", extraction_input_dir
        )

    # ── 10. Replaces pre-flight check ─────────────────────────────────────────
    if replaces is not None and "ingestion" in effective_stages:
        _driver = get_driver()
        try:
            preflight_check_replaces(_driver, replaces)
            logger.info(
                "Pre-flight check passed: document '%s' found in Neo4j.", replaces
            )
        finally:
            _driver.close()

    # ── State accumulators ────────────────────────────────────────────────────
    stage_results: list[StageResult] = []
    intermediate_docs: list = []   # list[IntermediateDocument]
    extracted_docs: list = []      # list[Document]
    ingested_doc_names: list[str] = []
    pipeline_result_kwargs: dict = {
        "preprocess": None,
        "extraction": None,
        "ingestion": None,
        "annotation": None,
        "entity_extraction": None,
        "tabular": None,
    }

    _effective_tabular_extensions = tabular_extensions or _DEFAULT_TABULAR_EXTENSIONS

    def _should_abort(sr: StageResult) -> bool:
        if sr.success:
            return False
        if on_partial_failure == "abort":
            return True
        if on_partial_failure == "warn":
            logger.warning(
                "Stage '%s' completed with failures (%d failed). Continuing.",
                sr.stage, sr.total_failed,
            )
        return False

    def _build_result(success: bool) -> PipelineResult:
        total_dur = time.monotonic() - pipeline_t0
        return PipelineResult(
            success=success,
            total_duration_seconds=total_dur,
            stages_executed=[sr.stage for sr in stage_results],
            **{k: v for k, v in pipeline_result_kwargs.items()},
        )
    sr = False
    # ── Tabular short-circuit ─────────────────────────────────────────────────
    if effective_stages == ["tabular"]:
        sr = await run_tabular_pipeline(
            input_raw=input_raw,
            update_mode=update_mode,
            parallel_docs=parallel_docs,
            tabular_extensions=_effective_tabular_extensions,
            tabular_delimiter=tabular_delimiter,
        )
        stage_results.append(sr)
        pipeline_result_kwargs["tabular"] = sr
        return _build_result(sr.success)

    sr = False
    # ── Stage 0: Preprocess ───────────────────────────────────────────────────
    if "preprocess" in effective_stages:
        input_raw_path = Path(input_raw)
        has_tabular = any(
            f.suffix.lower() in _effective_tabular_extensions
            for f in input_raw_path.rglob("*")
            if f.is_file()
        )
        has_non_tabular = any(
            f.suffix.lower() not in _effective_tabular_extensions
            for f in input_raw_path.rglob("*")
            if f.is_file()
        )
        if has_tabular:
            logger.info(
                "Tabular files detected in '%s' — running tabular pipeline first.", input_raw
            )
            sr = await run_tabular_pipeline(
                input_raw=input_raw,
                update_mode=update_mode,
                parallel_docs=parallel_docs,
                tabular_extensions=_effective_tabular_extensions,
                tabular_delimiter=tabular_delimiter,
            )
            stage_results.append(sr)
            pipeline_result_kwargs["tabular"] = sr
            if _should_abort(sr):
                return _build_result(False)

        if has_non_tabular:
            logger.info("Non tabular files detected in '%s' - Running Stage 0: preprocess", input_raw)
            sr, intermediate_docs = await run_preprocess(
                input_raw=input_raw,
                output_dir=converter_output_dir,
                context_instructions=context_instructions,
            )
            stage_results.append(sr)
            pipeline_result_kwargs["preprocess"] = sr
            logger.info(
                "Stage 0 complete: %d converted, %d failed.",
                sr.total_processed, sr.total_failed,
            )
            if _should_abort(sr):
                return _build_result(False)
    sr = False
    # ── Stage 1: Extraction ───────────────────────────────────────────────────
    if "extraction" in effective_stages:
        logger.info("Stage 1: extraction")
        if extraction_input_dir is not None:
            # Read from disk (Stage 0 was skipped)
            sr, extracted_docs = await run_extraction(
                input_folder=extraction_input_dir,
                output_folder=extraction_output_dir,
                parallel_docs=parallel_docs,
            )
        elif intermediate_docs:
            # In-memory from Stage 0
            sr, extracted_docs = await run_extraction(
                intermediate_documents=intermediate_docs,
                output_folder=extraction_output_dir,
                parallel_docs=parallel_docs,
            )
        elif converter_output_dir:
            # Fallback: read from converter_output_dir if provided
            sr, extracted_docs = await run_extraction(
                input_folder=converter_output_dir,
                output_folder=extraction_output_dir,
                parallel_docs=parallel_docs,
            )
        else:
            logger.info("No documents available for extraction; skipping Stage 1.")
        if sr:
            stage_results.append(sr)
            pipeline_result_kwargs["extraction"] = sr
            logger.info(
                "Stage 1 complete: %d extracted, %d failed.",
                sr.total_processed, sr.total_failed,
            )
            if _should_abort(sr):
                return _build_result(False)
            
    sr = False
    # ── Stage 2: Ingestion ────────────────────────────────────────────────────
    if "ingestion" in effective_stages:
        logger.info("Stage 2: ingestion")
        if ingestion_input_dir is not None:
            sr = await run_ingestion(
                output_folder=ingestion_input_dir,
                update_mode=update_mode,
            )
        elif extracted_docs:
            # In-memory from Stage 1
            sr = await run_ingestion(
                documents=extracted_docs,
                update_mode=update_mode,
            )
        elif extraction_output_dir is not None:
            # Prefer disk output if Stage 1 wrote to disk
            extract_files = sorted(Path(extraction_output_dir).rglob("extract-*.json"))
            sr = await run_ingestion(
                files=extract_files if extract_files else None,
                output_folder=extraction_output_dir if not extract_files else None,
                update_mode=update_mode,
            )
        else:
            logger.info("No documents available for ingestion; skipping Stage 2.")

        # update_mode + multiple files guard
        if update_mode and len(sr.documents) > 1:
            raise ValueError(
                f"update_mode=True is not allowed when ingesting multiple documents "
                f"({len(sr.documents)} found). update_mode is designed for single-document "
                "correction runs."
            )
        if sr:
            stage_results.append(sr)
            pipeline_result_kwargs["ingestion"] = sr
            ingested_doc_names = [
                doc.document_name for doc in sr.documents if doc.nodes_processed > 0
            ]
            logger.info(
                "Stage 2 complete: %d ingested, %d failed.",
                sr.total_processed, sr.total_failed,
            )

            if replaces is not None:
                _driver = get_driver()
                try:
                    apply_replacement(_driver, replaces, ingested_doc_names)
                finally:
                    _driver.close()

            if _should_abort(sr):
                return _build_result(False)

    # ── Resolve document names for Stages 3/4 ────────────────────────────────
    doc_names_for_ann_ee: list[str] = []

    if ingested_doc_names:
        doc_names_for_ann_ee = ingested_doc_names
    elif document_names is not None:
        doc_names_for_ann_ee = document_names
    elif document_names_dir is not None:
        dir_path = Path(document_names_dir)
        extract_files = sorted(dir_path.rglob("extract-*.json"))
        doc_names_for_ann_ee = []
        for f in extract_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                name = data.get("document_name")
                if name:
                    doc_names_for_ann_ee.append(name)
            except Exception as exc:
                logger.warning("Could not read document_name from '%s': %s", f, exc)
        if not doc_names_for_ann_ee:
            raise ValueError(
                f"document_names_dir='{document_names_dir}' contains no readable "
                "'document_name' fields in extract-*.json files."
            )

    # ── Stage 3: Annotation ───────────────────────────────────────────────────
    if "annotation" in effective_stages:
        if not doc_names_for_ann_ee:
            logger.info("No documents available for annotation; skipping Stage 3.")
        else:
            logger.info(
                "Stage 3: annotation (%d document(s), parallel_docs=%d)",
                len(doc_names_for_ann_ee), parallel_docs,
            )
            semaphore = asyncio.Semaphore(parallel_docs)
            ann_stage_results: list[StageResult] = []

            async def _annotate_one(name: str) -> StageResult:
                async with semaphore:
                    result = await run_annotation(
                        document_name=name,
                        manual=manual,
                        model_class=model_class,
                        parallel_docs=parallel_docs,
                        only_unannotated=only_unannotated,
                        context_instructions_override=context_instructions,
                    )
                    logger.info(
                        "Annotation complete for '%s': %d processed, %d failed.",
                        name, result.total_processed, result.total_failed,
                    )
                    return result

            ann_gather = await asyncio.gather(
                *[_annotate_one(n) for n in doc_names_for_ann_ee],
                return_exceptions=True,
            )
            all_ann_docs: list[DocumentResult] = []
            all_ann_errors: list[str] = []
            for item in ann_gather:
                if isinstance(item, Exception):
                    logger.error("Annotation failed: %s", item)
                    all_ann_errors.append(str(item))
                else:
                    ann_stage_results.append(item)
                    all_ann_docs.extend(item.documents)

            combined_ann = StageResult(
                stage="annotation",
                success=len(all_ann_errors) == 0 and all(r.success for r in ann_stage_results),
                documents=all_ann_docs,
                total_processed=sum(r.total_processed for r in ann_stage_results),
                total_failed=sum(r.total_failed for r in ann_stage_results),
                duration_seconds=sum(r.duration_seconds for r in ann_stage_results),
                errors=all_ann_errors,
            )
            stage_results.append(combined_ann)
            pipeline_result_kwargs["annotation"] = combined_ann
            if _should_abort(combined_ann):
                return _build_result(False)

    # ── Stage 4: Entity Extraction ────────────────────────────────────────────
    if "entity_extraction" in effective_stages:
        if not doc_names_for_ann_ee:
            logger.info("No documents available for entity extraction; skipping Stage 4.")
        else:
            logger.info(
                "Stage 4: entity_extraction (%d document(s), parallel_docs=%d)",
                len(doc_names_for_ann_ee), parallel_docs,
            )
            semaphore = asyncio.Semaphore(parallel_docs)
            ee_stage_results: list[StageResult] = []

            async def _extract_one(name: str) -> StageResult:
                async with semaphore:
                    result = await run_entity_extraction(
                        document_name=name,
                        parallel_docs=parallel_docs,
                        only_unextracted=only_unextracted,
                    )
                    logger.info(
                        "Entity extraction complete for '%s': %d processed, %d failed.",
                        name, result.total_processed, result.total_failed,
                    )
                    return result

            ee_gather = await asyncio.gather(
                *[_extract_one(n) for n in doc_names_for_ann_ee],
                return_exceptions=True,
            )
            all_ee_docs: list[DocumentResult] = []
            all_ee_errors: list[str] = []
            for item in ee_gather:
                if isinstance(item, Exception):
                    logger.error("Entity extraction failed: %s", item)
                    all_ee_errors.append(str(item))
                else:
                    ee_stage_results.append(item)
                    all_ee_docs.extend(item.documents)

            combined_ee = StageResult(
                stage="entity_extraction",
                success=len(all_ee_errors) == 0 and all(r.success for r in ee_stage_results),
                documents=all_ee_docs,
                total_processed=sum(r.total_processed for r in ee_stage_results),
                total_failed=sum(r.total_failed for r in ee_stage_results),
                duration_seconds=sum(r.duration_seconds for r in ee_stage_results),
                errors=all_ee_errors,
            )
            stage_results.append(combined_ee)
            pipeline_result_kwargs["entity_extraction"] = combined_ee
            if _should_abort(combined_ee):
                return _build_result(False)

    # ── Final result ──────────────────────────────────────────────────────────
    overall_success = all(sr.success for sr in stage_results)
    return _build_result(overall_success)
