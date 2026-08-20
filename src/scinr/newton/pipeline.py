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
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from scinr.newton.results import DocumentResult, PipelineResult, StageResult

if TYPE_CHECKING:
    from scinr.newton.converters.base import IntermediateDocument
    from scinr.newton.models.document_structure import Document
    from scinr.newton.pipeline_units import DocumentUnit, UnitResult

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
    parallel_docs: int = 5,
    # ── Behaviour on partial failure ─────────────────────────────────────────
    on_partial_failure: Literal["abort", "continue", "warn"] = "warn",
    # ── Stage 1 extraction performance (opt-in) ───────────────────────────────
    fast_extraction: bool = False,
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

    Args:
        input_raw: Folder containing raw source files (PDF, DOCX, CSV, XLSX, …) for
            Stage 0. Required when `stages` includes `"preprocess"` and `extraction_input_dir`
            is not given.
        converter_output_dir: Folder where Stage 0 writes intermediate JSON files to disk.
            When `None`, intermediate files are kept in memory only.
        extraction_input_dir: Folder where Stage 1 reads JSON input from disk, skipping Stage 0.
            **Precedence:** if provided, this takes absolute priority over `document_names` /
            `document_names_dir` for document discovery, regardless of which stages are
            requested — even for a `stages=["annotation"]`-only run, discovery reads the
            documents found under this folder rather than resolving `document_names` via
            Neo4j. Passing both is not an error; `document_names`/`document_names_dir` are
            then silently ignored. Do not pass `extraction_input_dir` together with
            `document_names`/`document_names_dir` unless this precedence is intended.
        extraction_output_dir: Folder where Stage 1 writes `extract-*.json` output files.
        ingestion_input_dir: Folder where Stage 2 reads `extract-*.json` files from disk, skipping Stages 0 and 1.
            **Precedence:** same absolute-priority rule as `extraction_input_dir` above
            (takes precedence over `document_names`/`document_names_dir` regardless of
            requested stages) — see that entry for the full explanation.
        stages: Ordered list of stage names to execute (`"preprocess"`, `"extraction"`, `"ingestion"`,
            `"annotation"`, `"entity_extraction"`, `"tabular"`). Default runs full pipeline.
        document_names: Explicit list of Neo4j `document_name` values for Stage 3/4 runs.
            Ignored (silently) if `extraction_input_dir` or `ingestion_input_dir` is also
            provided — see the precedence note on those two parameters.
        document_names_dir: Directory of `extract-*.json` files to extract document names from.
            Same silent-ignore precedence rule as `document_names` above.
        manual: If `True`, Stage 3 manual annotation assigns `model_class` without LLM calls.
        model_class: CamelCase Pydantic model class name for manual annotation.
        only_unannotated: Skip nodes that already have an annotation decision.
        only_unextracted: Skip nodes that already have extracted entities.
        context_instructions: Custom instructions injected into converter and annotation prompts.
        update_mode: If `True`, Stage 2 replaces latest document version in Neo4j without incrementing version.
        replaces: `document_name` of existing document superseded by newly ingested document.
        parallel_docs: Maximum number of documents processed concurrently (default: `1`).
        on_partial_failure: Control behavior when a stage fails
            (`"abort"`, `"continue"`, or `"warn"`).

            The pipeline never stops processing OTHER documents because of
            this flag: every document in the batch is always dispatched to
            the per-document-unit engine and runs independently of its
            siblings, regardless of `on_partial_failure`'s value (outside
            the tabular short-circuit below, which still fails fast exactly
            as before).

            Within a single document's own remaining stages, the effect of
            `on_partial_failure` depends on which stage failed:

            - `"preprocess"` / `"extraction"` / `"ingestion"`: a failure in
              any of these three is a *total* failure for that document —
              no valid artifact was produced for the next stage to operate
              on. These always stop that document from advancing through
              its remaining stages, regardless of `on_partial_failure`
              (there is nothing valid to continue with).
            - `"annotation"` / `"entity_extraction"`: these operate
              per-node, so a stage reporting `nodes_failed > 0` for a
              document is only a *partial* failure — the document itself
              is still valid and can proceed to its next requested stage.
              This only stops that document's advancement when
              `on_partial_failure` is `"abort"` (the default, preserving
              the historical per-unit "soft-abort" behavior). With
              `"continue"` or `"warn"`, the document keeps advancing to its
              next requested stage even though some nodes failed in the
              previous one.

            `"warn"` behaves like `"continue"` (the document keeps
            advancing) but additionally logs at two levels:

            - Immediately, a per-document warning is emitted the moment
              *that specific document* decides to keep advancing despite a
              partial failure in `annotation` or `entity_extraction` —
              naming the document, the stage, the failed-node count, and
              the concrete error(s) reported for it.
            - At the end of the batch, the pre-existing aggregated
              per-stage warning still fires whenever a stage reports one or
              more failed documents overall (only the total failed-document
              count for that stage, not per-document detail).

            Both warnings coexist in `"warn"` mode; `"continue"` mode stays
            completely silent.

        tabular_extensions: File extensions to process via tabular pipeline (default: `.csv`, `.xlsx`, `.xls`).
        tabular_delimiter: Delimiter character for CSV tabular files.
        fast_extraction: Opt-in, resolved once per call and passed explicitly through
            every layer down to Stage 1 — never read from global config, by design,
            so that concurrent `run_pipeline()` calls with different values never
            interfere with each other. When `True`, Stage 1 extraction runs chunks
            in parallel and defers cross-chunk hierarchy resolution to a single
            post-extraction consolidation LLM call instead of incremental
            deterministic prefix-matching. This can reduce Stage 1 wall-clock time
            substantially for multi-chunk documents, but concentrates
            hierarchy-correctness risk into one LLM call — a degenerate or
            partially-failed consolidation response has a larger blast radius than
            the default per-chunk behavior. Recommended only after validating output
            quality against representative documents. Default: `False` (safe,
            unchanged legacy behavior). Raises `ValueError` if `True` while
            `"extraction"` is not in `stages`.

    Returns:
        PipelineResult containing stage metrics, execution flags, and duration.

    Raises:
        ConfigurationError: If Neo4j or LLM configuration is missing.
        PreconditionError: If invalid parameters or mutually exclusive options are supplied.
        ExtractionError: If entity extraction fails.
        IngestionError: If Neo4j graph write fails.
        ValueError: If any parameter combination is invalid (see validation section).
        FileNotFoundError: If a required directory does not exist.
    """
    # ── Imports (deferred to avoid circular imports at module level) ──────────
    from scinr.newton.ingest.config import get_driver
    from scinr.newton.stages import (
        apply_replacement,
        preflight_check_replaces,
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

    # ── 9b. fast_extraction requires 'extraction' in stages — checked after
    # the skip logic above, which may have just removed 'extraction' from
    # effective_stages (e.g. ingestion_input_dir/extraction_input_dir skips). ──
    if fast_extraction and "extraction" not in effective_stages:
        raise ValueError(
            "fast_extraction=True has no effect unless 'extraction' is included in stages."
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

    # ── Tabular files auto-detected inside input_raw (mixed folder) ─────────
    # Only the non-tabular files in `input_raw` proceed to the new
    # per-document-unit engine below; tabular files are fully handled here,
    # exactly as before.
    has_non_tabular = False
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

    # ── Discover document units for the new per-document-unit engine ────────
    # Collapses the legacy per-stage source resolution (Stage 0 <- input_raw;
    # Stage 1 <- extraction_input_dir -> in-memory Stage-0 output ->
    # converter_output_dir fallback; Stage 2 <- ingestion_input_dir ->
    # in-memory Stage-1 output -> extraction_output_dir fallback; Stage 3/4
    # <- ingested names -> document_names -> document_names_dir) into a
    # single discovery call. Precedence matches the real branches of the
    # legacy sequential code above (see Coder report for the full analysis).
    from scinr.newton.pipeline_units import _discover_units, build_all_paths_for_versioning

    if ingestion_input_dir is not None:
        units = await _discover_units(ingestion_input_dir=ingestion_input_dir)
    elif extraction_input_dir is not None:
        units = await _discover_units(extraction_input_dir=extraction_input_dir)
    elif input_raw is not None and has_non_tabular:
        units = await _discover_units(
            input_raw=input_raw, tabular_extensions=_effective_tabular_extensions
        )
    elif (
        converter_output_dir is not None
        and "extraction" in effective_stages
        and "preprocess" not in effective_stages
    ):
        units = await _discover_units(extraction_input_dir=converter_output_dir)
    elif (
        extraction_output_dir is not None
        and "ingestion" in effective_stages
        and "extraction" not in effective_stages
    ):
        units = await _discover_units(ingestion_input_dir=extraction_output_dir)
    elif document_names is not None:
        units = await _discover_units(document_names=document_names)
    elif document_names_dir is not None:
        units = await _discover_units(document_names_dir=document_names_dir)
    else:
        units = []

    if not units:
        logger.info("No documents discovered for this run; nothing to process.")

    # ── Pre-warm shared resources, then fan out one task per document unit ──
    from scinr.newton.annotation.neo4j_ops import (
        ensure_catalog_models_once,
        ensure_theme_structure_once,
    )
    from scinr.newton.ingest.config import get_async_driver
    from scinr.newton.ingest.loader import resolve_batch_version_sync
    from scinr.newton.ingest.schema import setup_schema
    from scinr.newton.utils.theme_registry import get_theme_registry

    raw_file_repo = None
    page_repo = None
    if "preprocess" in effective_stages:
        from scinr.newton.storage.factory import get_storage
        raw_file_repo, page_repo = get_storage()

    sync_driver = None
    shared_ingest_version: int | None = None
    unit_results: list = []
    unit_gather_duration = 0.0
    try:
        if "ingestion" in effective_stages and units:
            sync_driver = get_driver()
            setup_schema(sync_driver)
            all_paths = build_all_paths_for_versioning(units)
            shared_ingest_version = await asyncio.to_thread(
                resolve_batch_version_sync, sync_driver, all_paths, update_mode
            )

        if ("annotation" in effective_stages or "entity_extraction" in effective_stages) and units:
            async_driver_for_prewarm = get_async_driver()
            theme_registry = get_theme_registry()
            await ensure_catalog_models_once(async_driver_for_prewarm)
            await ensure_theme_structure_once(async_driver_for_prewarm, theme_registry)

        document_semaphore = asyncio.Semaphore(parallel_docs)
        unit_gather_t0 = time.monotonic()
        if units:
            unit_results = await asyncio.gather(
                *[
                    _process_document_unit(
                        u,
                        effective_stages=effective_stages,
                        document_semaphore=document_semaphore,
                        sync_driver=sync_driver,
                        shared_ingest_version=shared_ingest_version,
                        converter_output_dir=converter_output_dir,
                        extraction_output_dir=extraction_output_dir,
                        extraction_input_dir=extraction_input_dir,
                        raw_file_repo=raw_file_repo,
                        page_repo=page_repo,
                        context_instructions=context_instructions,
                        update_mode=update_mode,
                        manual=manual,
                        model_class=model_class,
                        only_unannotated=only_unannotated,
                        only_unextracted=only_unextracted,
                        on_partial_failure=on_partial_failure,
                        fast_extraction=fast_extraction,
                    )
                    for u in units
                ],
                return_exceptions=True,
            )
        unit_gather_duration = time.monotonic() - unit_gather_t0
    finally:
        if sync_driver is not None:
            sync_driver.close()

    # ── Per-stage aggregation (Fase 7a) ───────────────────────────────────────
    # Never returns early on failure here: once units are dispatched, every
    # requested stage is always aggregated regardless of on_partial_failure.
    # on_partial_failure additionally gates the preserved tabular
    # short-circuit above (via _should_abort()) and, inside
    # _process_document_unit() itself, whether a unit's per-unit soft-abort
    # triggers after a partial (`nodes_failed > 0`) annotation/
    # entity_extraction failure ("abort" stops that unit's remaining stages;
    # "continue"/"warn" let it keep advancing) — see that function's
    # docstring for the full breakdown, including the stages
    # (preprocess/extraction/ingestion) whose soft-abort is unconditional.
    for stage_name in ("preprocess", "extraction", "ingestion", "annotation", "entity_extraction"):
        if stage_name not in effective_stages:
            continue

        documents = [
            ur.stage_results[stage_name]
            for ur in unit_results
            if not isinstance(ur, BaseException) and stage_name in ur.stage_results
        ]
        fatal = [str(ur) for ur in unit_results if isinstance(ur, BaseException)]
        # Design decision (reviewer point 3, kept as-is): a unit's
        # fatal_error is folded into EVERY requested stage it never reached
        # (not deduplicated to only the first one). A "fatal" error here is
        # by definition not attributable to any single concrete stage (see
        # _process_document_unit()'s own catch-all docstring) — the unit
        # made it into stage_results for zero of the stages it was asked to
        # run. Each of those stages' own aggregate is therefore genuinely
        # incomplete for this unit (it is missing from `documents`, and
        # that stage's `success` must correctly flip to False to reflect
        # that), so surfacing the same diagnostic message in each of their
        # `errors` lists is accurate, not merely repetitive noise: every
        # affected stage independently needs to know it never got to
        # process this unit and why. Deduplicating to "only the first
        # requested stage" was considered and rejected — it would require
        # picking an arbitrary stage (first in *effective_stages* order,
        # not necessarily the one that "actually" failed, since fatal
        # errors are unattributable by definition) while making every
        # OTHER affected stage silently report success=True despite also
        # being missing that unit, which is strictly worse. See
        # TestFatalErrorAggregation in test_pipeline_orchestration.py for
        # the multi-stage regression test pinning this choice.
        fatal += [
            ur.fatal_error
            for ur in unit_results
            if not isinstance(ur, BaseException)
            and ur.fatal_error
            and stage_name not in ur.stage_results
        ]
        total_failed = sum(d.nodes_failed for d in documents)
        stage_sr = StageResult(
            stage=stage_name,
            success=(total_failed == 0 and not fatal),
            documents=documents,
            total_processed=sum(d.nodes_processed for d in documents),
            total_failed=total_failed,
            duration_seconds=unit_gather_duration,
            errors=fatal,
        )
        stage_results.append(stage_sr)
        pipeline_result_kwargs[stage_name] = stage_sr

        if stage_name == "ingestion":
            # update_mode + multiple documents guard (same message as before)
            if update_mode and len(stage_sr.documents) > 1:
                raise ValueError(
                    f"update_mode=True is not allowed when ingesting multiple documents "
                    f"({len(stage_sr.documents)} found). update_mode is designed for "
                    "single-document correction runs."
                )
            if replaces is not None:
                ingested_doc_names = [
                    doc.document_name for doc in stage_sr.documents if doc.nodes_failed == 0
                ]
                _driver = get_driver()
                try:
                    apply_replacement(_driver, replaces, ingested_doc_names)
                finally:
                    _driver.close()

        if on_partial_failure == "warn" and total_failed > 0:
            logger.warning(
                "Stage '%s' completed with failures (%d failed). Continuing.",
                stage_name, total_failed,
            )

    # ── Final result ──────────────────────────────────────────────────────────
    overall_success = all(sr.success for sr in stage_results)
    return _build_result(overall_success)


# ---------------------------------------------------------------------------
# Per-document-unit orchestration (Bloque B). Fully wired into run_pipeline()
# above via the discovery + asyncio.gather() fan-out block; this is not a
# standalone/unused building block.
# ---------------------------------------------------------------------------


def _combine_stage_documents(current_name: str, documents: list[DocumentResult]) -> DocumentResult:
    """Flatten a ``StageResult.documents`` list into a single ``DocumentResult``
    for one document-unit's ``stage_results`` entry.

    ``run_annotation()`` / ``run_entity_extraction()`` are always called here
    with a single leaf ``document_name``, so *documents* normally has 0 or 1
    entries. The >1 case is handled defensively (should not occur for a
    single leaf document) by summing counts and concatenating errors.
    """
    if len(documents) == 1:
        d = documents[0]
        return DocumentResult(current_name, d.nodes_processed, d.nodes_failed, d.errors)
    if not documents:
        return DocumentResult(current_name, 0, 0, [])
    total_processed = sum(d.nodes_processed for d in documents)
    total_failed = sum(d.nodes_failed for d in documents)
    all_errors = [e for d in documents for e in d.errors]
    return DocumentResult(current_name, total_processed, total_failed, all_errors)


async def _process_document_unit(
    unit: DocumentUnit,
    *,
    effective_stages: list[str],
    document_semaphore: asyncio.Semaphore,
    sync_driver,
    shared_ingest_version: int | None,
    converter_output_dir: str | None,
    extraction_output_dir: str | None,
    extraction_input_dir: str | None,
    raw_file_repo,
    page_repo,
    context_instructions: str | None,
    update_mode: bool,
    manual: bool,
    model_class: str | None,
    only_unannotated: bool,
    only_unextracted: bool,
    fast_extraction: bool,
    on_partial_failure: Literal["abort", "continue", "warn"] = "abort",
) -> UnitResult:
    """Process a single ``DocumentUnit`` end-to-end through whichever of
    *effective_stages* apply to it.

    Bounded by *document_semaphore* for its entire duration (acquired once,
    held across every stage this unit passes through). Each concrete stage
    call (``convert_one()``, ``extract_one_intermediate()``/
    ``extract_one_file()``, ``ingest_one()``/``ingest_one_from_path()``,
    ``run_annotation()``, ``run_entity_extraction()``) already bounds its own
    real LLM/Neo4j work internally (via ``get_llm_semaphore()`` /
    ``get_neo4j_sync_semaphore()``) — this function does not add another
    semaphore around those calls.

    Soft-abort semantics: as soon as one stage fails for this unit, no
    exception propagates — the caller (the per-unit ``asyncio.gather()`` in
    ``run_pipeline()`` above) can safely run many units concurrently without
    one unit's failure cancelling the others. Whether the *remaining* stages
    for *this unit only* are skipped (``stopped_at`` set to the failing
    stage's name) depends on which stage failed and, for two of them, on
    *on_partial_failure*:

    - ``"preprocess"`` / ``"extraction"`` / ``"ingestion"``: a failure in
      any of these three means there is no valid artifact
      (``intermediate_doc`` / ``doc_obj`` / a consistent ``current_name``)
      for the next stage to operate on. These **always** stop this unit's
      remaining stages, regardless of *on_partial_failure* — there is
      nothing valid to continue with.
    - ``"annotation"`` / ``"entity_extraction"``: these operate per-node, so
      ``nodes_failed > 0`` is a *partial* failure — ``current_name`` is
      still valid and the next stage can still run against it. This only
      stops the unit's remaining stages when *on_partial_failure* is
      ``"abort"`` (the default). With ``"continue"`` or ``"warn"`` the unit
      keeps advancing to its next requested stage even if some nodes failed
      in the previous one. When *on_partial_failure* is ``"warn"``, this
      unit additionally logs a warning right at the point it decides to
      keep advancing, naming this document, the failing stage, the
      failed-node count, and the concrete error(s) reported for it;
      ``"continue"`` performs the exact same advancement but logs nothing.

    Args:
        unit: The document unit to process (see ``pipeline_units.DocumentUnit``).
        effective_stages: Ordered list of stage names requested for this run (a subset of
            ``{"preprocess", "extraction", "ingestion", "annotation",
            "entity_extraction"}``).
        document_semaphore: Shared semaphore bounding how many document units run concurrently.
        sync_driver: Shared sync Neo4j ``Driver``, or ``None`` if ``"ingestion"`` is not
            in *effective_stages*.
        shared_ingest_version: Pre-computed batch version forwarded to ``ingest_one()`` /
            ``ingest_one_from_path()``.
        converter_output_dir: Folder where Stage 0 writes intermediate JSON to disk, or ``None``
            to keep the converted document in memory only (a fresh temporary
            directory, scoped to and removed immediately after this unit's
            ``convert_one()`` call via ``tempfile.TemporaryDirectory()``, is
            created per unit in that case, since ``convert_one()`` requires a
            concrete output directory on disk).
        extraction_output_dir: Folder where Stage 1 writes ``extract-*.json`` output, or ``None``
            to keep the extracted document in memory only.
        extraction_input_dir: Root input folder for ``extraction_json`` units, used to mirror the
            relative subdirectory structure under *extraction_output_dir*.
        raw_file_repo: Optional ``RawFileRepository`` forwarded to ``convert_one()``.
        page_repo: Optional ``PageRepository`` forwarded to ``convert_one()``.
        context_instructions: Free-text context forwarded to ``convert_one()`` and
            ``run_annotation()``.
        update_mode: Forwarded to ``ingest_one()`` / ``ingest_one_from_path()``.
        manual: Forwarded to ``run_annotation()``.
        model_class: Forwarded to ``run_annotation()`` (required when *manual* is
            ``True``).
        only_unannotated: Forwarded to ``run_annotation()``.
        only_unextracted: Forwarded to ``run_entity_extraction()``.
        fast_extraction: Forwarded to ``extract_one_intermediate()`` / ``extract_one_file()``
            at the Stage 1 call sites below. Resolved once per ``run_pipeline()`` call
            and passed explicitly all the way down — never read from global config.
            See ``run_pipeline()``'s docstring for the full tradeoff explanation.
        on_partial_failure: Controls whether this unit keeps advancing to its next requested
            stage after ``annotation`` or ``entity_extraction`` reports
            ``nodes_failed > 0`` for it. ``"abort"`` (default) stops the unit
            at that stage (``stopped_at`` set); ``"continue"`` and ``"warn"``
            let it keep advancing. Has no effect on the unconditional
            ``preprocess``/``extraction``/``ingestion`` stops described above.
            ``"warn"`` additionally logs a per-document warning at the exact
            moment this unit decides to keep advancing despite the partial
            failure (document name, failing stage, failed-node count, and
            concrete error detail); ``"continue"`` stays silent.

    Returns:
        ``stage_results`` contains only the stages actually reached.
        ``stopped_at`` names the stage where this unit stopped due to a
        failure, or ``None`` if it completed every requested stage (or hit
        the documented "unsupported format, nothing to convert" no-op).
        ``fatal_error`` is set only for an exception not attributable to any
        single concrete stage above (defense-in-depth — this function must
        never let an exception escape uncaught).
    """
    from scinr.newton.converters.main import convert_one
    from scinr.newton.ingest.loader import ingest_one, ingest_one_from_path
    from scinr.newton.pipeline_units import UnitResult
    from scinr.newton.stages import run_annotation, run_entity_extraction
    from scinr.newton.stages.extraction import extract_one_file, extract_one_intermediate

    stage_results: dict[str, DocumentResult] = {}
    current_name = unit.document_name_hint

    async with document_semaphore:
        try:
            intermediate_doc: IntermediateDocument | None = None
            doc_obj: Document | None = None

            # ── Stage: preprocess ──────────────────────────────────────
            if "preprocess" in effective_stages and unit.kind == "raw_file":
                relative_prefix = unit.relative_dir if unit.relative_dir != Path(".") else None
                if converter_output_dir:
                    written, failures = await convert_one(
                        unit.source_path,
                        Path(converter_output_dir),
                        raw_file_repo=raw_file_repo,
                        page_repo=page_repo,
                        _relative_prefix=relative_prefix,
                        context_instructions=context_instructions,
                    )
                else:
                    # No persistent output dir requested: scope a fresh
                    # temporary directory to just this convert_one() call so
                    # it is always removed afterwards — on the happy path AND
                    # if convert_one() raises — via TemporaryDirectory's own
                    # __exit__. The old `tempfile.mkdtemp()` here was never
                    # cleaned up, leaking one orphaned directory (with the
                    # full converted JSON inside it) per raw_file unit into
                    # /tmp forever. `written[...][2]` (the IntermediateDocument)
                    # is a plain Python object already fully materialized in
                    # memory by the time convert_one() returns, so it safely
                    # outlives the tempdir's removal below — only the on-disk
                    # JSON convert_one() wrote is discarded, which is fine
                    # since the rest of this unit's chain only needs the
                    # in-memory object, not those files.
                    with tempfile.TemporaryDirectory() as tmp_dir_str:
                        written, failures = await convert_one(
                            unit.source_path,
                            Path(tmp_dir_str),
                            raw_file_repo=raw_file_repo,
                            page_repo=page_repo,
                            _relative_prefix=relative_prefix,
                            context_instructions=context_instructions,
                        )
                if failures:
                    stage_results["preprocess"] = DocumentResult(
                        unit.document_name_hint, 0, 1, [failures[0][1]]
                    )
                    return UnitResult(current_name, stage_results, "preprocess", None)
                if not written:
                    # Unsupported format, silently skipped by convert_one()
                    # (mirrors its own behaviour) — not a failure, simply
                    # nothing to convert for this unit.
                    stage_results["preprocess"] = DocumentResult(unit.document_name_hint, 0, 0, [])
                    return UnitResult(current_name, stage_results, None, None)
                intermediate_doc = written[0][2]
                stage_results["preprocess"] = DocumentResult(unit.document_name_hint, 1, 0, [])

            # ── Stage: extraction ──────────────────────────────────────
            if "extraction" in effective_stages:
                extraction_out = Path(extraction_output_dir) if extraction_output_dir else None
                if unit.kind == "raw_file":
                    doc_obj = await extract_one_intermediate(
                        intermediate_doc, extraction_out, fast_extraction=fast_extraction
                    )
                elif unit.kind == "extraction_json":
                    extraction_in = Path(extraction_input_dir) if extraction_input_dir else None
                    doc_obj = await extract_one_file(
                        unit.source_path, extraction_out, extraction_in, fast_extraction=fast_extraction
                    )

                if doc_obj is None:
                    stage_results["extraction"] = DocumentResult(
                        current_name, 0, 1, ["No pages to process"]
                    )
                    return UnitResult(current_name, stage_results, "extraction", None)
                current_name = doc_obj.document_name
                stage_results["extraction"] = DocumentResult(current_name, 1, 0, [])

            # ── Stage: ingestion ───────────────────────────────────────
            if "ingestion" in effective_stages:
                try:
                    if unit.kind == "ingestion_json":
                        current_name = await ingest_one_from_path(
                            unit.source_path,
                            sync_driver,
                            update_mode,
                            shared_ingest_version,
                        )
                    else:
                        current_name = await ingest_one(
                            doc_obj, sync_driver, update_mode, shared_ingest_version
                        )
                except Exception as exc:
                    stage_results["ingestion"] = DocumentResult(current_name, 0, 1, [str(exc)])
                    return UnitResult(current_name, stage_results, "ingestion", None)
                stage_results["ingestion"] = DocumentResult(current_name, 1, 0, [])

            # ── pre_ingested passthrough (no preprocess/extraction/ingestion
            # applies to this kind — the ifs above simply never matched) ──
            if unit.kind == "pre_ingested":
                current_name = unit.doc_path

            # ── Stage: annotation ──────────────────────────────────────
            if "annotation" in effective_stages:
                sr = await run_annotation(
                    current_name,
                    manual=manual,
                    model_class=model_class,
                    parallel_docs=1,
                    only_unannotated=only_unannotated,
                    context_instructions_override=context_instructions,
                )
                combined = _combine_stage_documents(current_name, sr.documents)
                stage_results["annotation"] = combined
                if combined.nodes_failed > 0:
                    if on_partial_failure == "abort":
                        return UnitResult(current_name, stage_results, "annotation", None)
                    if on_partial_failure == "warn":
                        logger.warning(
                            "Document '%s': stage 'annotation' had %d failed node(s) "
                            "(%s) — continuing to remaining stages despite the failure "
                            "(on_partial_failure='warn').",
                            current_name,
                            combined.nodes_failed,
                            "; ".join(combined.errors) if combined.errors else "no error details",
                        )

            # ── Stage: entity_extraction ───────────────────────────────
            if "entity_extraction" in effective_stages:
                sr = await run_entity_extraction(
                    current_name,
                    parallel_docs=1,
                    only_unextracted=only_unextracted,
                )
                combined = _combine_stage_documents(current_name, sr.documents)
                stage_results["entity_extraction"] = combined
                if combined.nodes_failed > 0:
                    if on_partial_failure == "abort":
                        return UnitResult(current_name, stage_results, "entity_extraction", None)
                    if on_partial_failure == "warn":
                        logger.warning(
                            "Document '%s': stage 'entity_extraction' had %d failed node(s) "
                            "(%s) — continuing to remaining stages despite the failure "
                            "(on_partial_failure='warn').",
                            current_name,
                            combined.nodes_failed,
                            "; ".join(combined.errors) if combined.errors else "no error details",
                        )

            return UnitResult(current_name, stage_results, None, None)
        except Exception as exc:  # noqa: BLE001 — defense-in-depth, see docstring
            logger.exception("Unhandled error processing document unit '%s'", current_name)
            return UnitResult(current_name, stage_results, None, str(exc))
