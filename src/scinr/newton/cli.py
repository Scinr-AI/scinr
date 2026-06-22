"""
cli.py — scinr-ingest pipeline entry point.

Orchestrates five independent stages:

    Stage 0: preprocess    — Convert raw files (PDF, DOCX, XLSX, etc.) to intermediate JSON
    Stage 1: extract       — Read JSON → sliding window → LLM → write output JSON
    Stage 2: ingest        — Read output JSON → Neo4j ingestion
    Stage 3: annotate      — Annotate Neo4j StructureNodes with model decisions
    Stage 4: entity_extract — Traverse annotated StructureNodes → LLM extraction → write entity subgraph
    Stage 5: tabular       — Ingest CSV/XLSX files directly into Neo4j (no LLM extraction stage needed)

Usage examples
--------------
Full pipeline::

    python main.py --stage all --input data/json/ --output data/output/ --document "MyDocument"

Full pipeline with subdirectory structure::

    python main.py --stage all --input-raw files/ --input data/json/ --output data/output/

Update the latest ingested version in-place::

    python main.py --stage ingest --output data/output/ --update

Replace an existing document::

    python main.py --stage all --input-raw files/nuevo/ --input data/json/ --output data/output/ --replaces "NombreDocumentoAntiguo"

Only extraction::

    python main.py --stage extract --input data/json/ --output data/output/

Only ingestion::

    python main.py --stage ingest --output data/output/

Only annotation (requires prior ingestion)::

    python main.py --stage annotate --document "MyDocument"

Only entity extraction::

    python main.py --stage entity_extract --document "MyDocument"

Ingest CSV/XLSX files directly::

    python main.py --stage tabular --input-raw files/

Ingest CSV/XLSX files with update mode::

    python main.py --stage tabular --input-raw files/ --update

Defaults (stage=all, input=data/json-pruebas/, output=data/output-pruebas/)::

    python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from scinr.newton.config import configure
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
from scinr.newton.utils.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Module logger — basicConfig is deferred to main() to avoid hijacking the
# root logger on import.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_INPUT = "data/json-pruebas/"
_DEFAULT_OUTPUT = "data/output-pruebas/"
_DEFAULT_STAGE = "all"
_DEFAULT_WINDOW_SIZE = 2
_DEFAULT_PARALLEL_DOCS = 1


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main",
        description="scinr-ingest pipeline: extract → ingest → annotate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "extract", "ingest", "annotate", "entity_extract", "tabular", "all"],
        default=_DEFAULT_STAGE,
        help="Pipeline stage to run. Use 'tabular' for direct CSV/XLSX ingestion.",
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_INPUT,
        metavar="DIR",
        help="Input folder containing raw JSON files (used by the extract stage).",
    )
    parser.add_argument(
        "--input-raw",
        default=None,
        metavar="DIR",
        help=(
            "Input folder with raw source files (PDF, DOCX, XLSX, PPTX, etc.) "
            "for the preprocess stage. "
            "If provided together with --stage all, preprocess runs before extract."
        ),
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        metavar="DIR",
        help="Output folder for extracted JSON files (used by extract and ingest stages).",
    )
    parser.add_argument(
        "--document",
        default=None,
        metavar="NAME",
        help=(
            "Document name for the annotation and entity_extract stages. "
            "Required when --stage annotate or --stage entity_extract; optional when --stage all."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=_DEFAULT_WINDOW_SIZE,
        metavar="N",
        help=(
            "Sliding window size in pages. "
            "Currently hardcoded to 2 internally; kept for future flexibility."
        ),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help=(
            "Update mode: wipe the existing structure of the latest document version "
            "and re-insert it cleanly. Does not create a new version. "
            "Use to correct errors in the last ingest. "
            "Cannot be combined with --replaces."
        ),
    )
    parser.add_argument(
        "--replaces",
        default=None,
        metavar="DOC_NAME",
        help=(
            "Name of an existing document in Neo4j that the newly ingested content replaces. "
            "The old document's latest=True version will be set to latest=False and linked "
            "via HAS_NEWER_VERSION to the new root document. "
            "Aborts with error if the named document is not found."
        ),
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        default=False,
        help=(
            "Manual annotation mode: assign a fixed model class to all StructureNodes "
            "of the document without running the LLM agent. "
            "Requires --model. Only valid with --stage annotate."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="CLASS_NAME",
        help=(
            "Exact CamelCase model class name for manual annotation override "
            "(e.g. 'DrugProductComposition'). Required when --manual is specified."
        ),
    )
    parser.add_argument(
        "--parallel-docs",
        type=int,
        default=_DEFAULT_PARALLEL_DOCS,
        metavar="N",
        help=(
            "Number of documents to process concurrently in the extraction, annotation "
            "and entity extraction stages. Use 1 (default) for sequential processing. "
            "Higher values speed up multi-document runs but increase Bedrock API load."
        ),
    )
    parser.add_argument(
        "--only-unextracted",
        action="store_true",
        default=False,
        help=(
            "Entity extraction stage only: skip StructureNodes that already have a "
            ":HAS_EXTRACTION->(:ExtractionResult) relationship. "
            "Only valid with --stage entity_extract or --stage all."
        ),
    )
    parser.add_argument(
        "--only-unannotated",
        action="store_true",
        default=False,
        help=(
            "Annotation stage only: skip StructureNodes that already have a "
            ":HAS_MODEL_DECISION relationship. "
            "Only valid with --stage annotate or --stage all."
        ),
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Free-text context instructions about the document(s) being ingested. "
            "Passed to the extraction and annotation LLMs as additional guidance. "
            "Example: 'This is a regulatory dossier for product X; focus on safety data.'"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def main() -> None:
    """Orchestrate the requested pipeline stage(s)."""
    import sys
    setup_logging(log_dir=Path("logs"))

    try:
        configure()  # Initialize from environment variables
    except Exception as exc:
        print(f"[scinr-ingest] Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    args = _parse_args()

    if args.window_size != 2:
        logger.warning(
            "--window-size=%d provided but sliding window is hardcoded to 2; value ignored.",
            args.window_size,
        )

    # Validate: --stage annotate requires --document
    if args.stage == "annotate" and not args.document:
        raise SystemExit(
            "error: --document is required when --stage annotate.\n"
            "       Example: python main.py --stage annotate --document 'MyDocument'"
        )

    # Validate: --stage entity_extract requires --document
    if args.stage == "entity_extract" and not args.document:
        raise SystemExit(
            "error: --document is required when --stage entity_extract.\n"
            "       Example: python main.py --stage entity_extract --document 'MyDocument'"
        )

    # Validate: --update and --replaces are mutually exclusive
    if args.update and args.replaces:
        raise SystemExit(
            "error: --update and --replaces cannot be used together.\n"
            "       --update fixes the current latest version in-place.\n"
            "       --replaces links a new document as the successor of an existing one."
        )

    # Validate: --manual requires --model
    if args.manual and not args.model:
        raise SystemExit(
            "error: --model is required when --manual is specified.\n"
            "       Example: python main.py --stage annotate --document 'MyDocument'"
            " --manual --model 'DrugProductComposition'"
        )

    # Validate: --manual is only valid with --stage annotate
    if args.manual and args.stage != "annotate":
        raise SystemExit(
            "error: --manual is only valid with --stage annotate.\n"
            "       Example: python main.py --stage annotate --document 'MyDocument'"
            " --manual --model 'DrugProductComposition'"
        )

    # Pre-flight check for --replaces
    if args.replaces and args.stage in ("ingest", "all"):
        _driver = get_driver()
        try:
            preflight_check_replaces(_driver, args.replaces)
            logger.info(
                "Pre-flight check passed: document '%s' found and will be replaced.",
                args.replaces,
            )
        finally:
            _driver.close()

    # Validate: --stage tabular requires --input-raw
    if args.stage == "tabular" and not args.input_raw:
        raise SystemExit(
            "error: --input-raw is required when --stage tabular.\n"
            "       Example: python main.py --stage tabular --input-raw files/"
        )

    # Ensure output directory exists before any stage runs
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 5: Tabular Ingestion (CSV/XLSX) ─────────────────────────────────
    if args.stage == "tabular":
        tabular_result = await run_tabular_pipeline(
            args.input_raw,
            update_mode=args.update,
            parallel_docs=args.parallel_docs,
        )
        logger.info(
            "Tabular pipeline complete: %d file(s) processed.", tabular_result.total_processed
        )
        return

    # ── Stage 0: Preprocess (Converters) ─────────────────────────────────────
    if args.stage == "preprocess":
        if not args.input_raw:
            raise SystemExit(
                "error: --input-raw is required when --stage preprocess.\n"
                "       Example: python main.py --stage preprocess "
                "--input-raw files/ --input data/input/"
            )
        stage0_result, _ = await run_preprocess(args.input_raw, args.input, context_instructions=args.context)
        logger.info("Preprocess complete: %d file(s) converted.", stage0_result.total_processed)
        return

    raw_to_json: dict[Path, Path] = {}
    processed_json_files: list[Path] = []
    tabular_doc_names_from_all: list[str] = []
    _TABULAR_EXTS = {".csv", ".xlsx", ".xls"}

    if args.stage == "all" and args.input_raw:
        # ── Route CSV/XLSX files to the tabular pipeline ──────────────────
        input_raw_path_for_all = Path(args.input_raw)
        has_tabular = any(
            f.suffix.lower() in _TABULAR_EXTS
            for f in input_raw_path_for_all.rglob("*")
            if f.is_file()
        )
        if has_tabular:
            logger.info(
                "Found tabular (CSV/XLSX) files in '%s' — routing to tabular pipeline.",
                args.input_raw,
            )
            tabular_result_all = await run_tabular_pipeline(
                args.input_raw,
                update_mode=args.update,
                parallel_docs=args.parallel_docs,
            )
            tabular_doc_names_from_all = [
                doc.document_name for doc in tabular_result_all.documents
                if doc.nodes_processed > 0
            ]
            logger.info(
                "Tabular pipeline complete: %d file(s) processed.",
                tabular_result_all.total_processed,
            )

        # ── Preprocess non-tabular files (PDF, DOCX, etc.) ────────────────
        stage0_result, _stage0_docs = await run_preprocess(args.input_raw, args.input, context_instructions=args.context)
        logger.info("Preprocess complete: %d file(s) converted.", stage0_result.total_processed)

    # ── Stage 1: Extraction ───────────────────────────────────────────────────
    output_files: list[str] = []
    if args.stage in ("extract", "all"):
        stage1_result, _stage1_docs = await run_extraction(
            input_folder=args.input, output_folder=args.output, parallel_docs=args.parallel_docs
        )
        output_files = [
            str((Path(args.output) / doc.document_name).with_suffix(".json"))
            for doc in stage1_result.documents if doc.nodes_processed > 0
        ]
        processed_json_files = [
            Path(args.input) / f"{doc.document_name}.json"
            for doc in stage1_result.documents if doc.nodes_processed > 0
        ]
        logger.info("Extraction complete: %d file(s) written.", stage1_result.total_processed)

    # ── Stage 2: Ingestion ────────────────────────────────────────────────────
    ingested_doc_names: list[str] = []
    if args.stage in ("ingest", "all"):
        specific_files = [Path(f) for f in output_files] if args.stage == "all" else None
        stage2_result = await run_ingestion(
            output_folder=args.output, files=specific_files, update_mode=args.update
        )
        ingested_doc_names.extend([
            doc.document_name for doc in stage2_result.documents if doc.nodes_processed > 0
        ])
        logger.info("Ingestion complete. %d document(s) ingested.", stage2_result.total_processed)

        # Apply replacement if --replaces was specified
        if args.replaces:
            _driver = get_driver()
            try:
                apply_replacement(_driver, args.replaces, ingested_doc_names)
            finally:
                _driver.close()

    # ── Stage 3: Annotation ───────────────────────────────────────────────────
    if args.stage == "annotate":
        await run_annotation(
            args.document,
            manual=args.manual,
            model_class=args.model,
            parallel_docs=args.parallel_docs,
            only_unannotated=args.only_unannotated,
            context_instructions_override=args.context,
        )
        logger.info("Annotation complete for document: %r", args.document)
    elif args.stage == "all":
        doc_names_to_annotate = [args.document] if args.document else ingested_doc_names
        if not doc_names_to_annotate:
            logger.warning("No documents available for annotation; skipping.")
        if doc_names_to_annotate:
            semaphore = asyncio.Semaphore(args.parallel_docs)

            async def _annotate(name: str) -> None:
                async with semaphore:
                    await run_annotation(
                        name,
                        parallel_docs=args.parallel_docs,
                        only_unannotated=args.only_unannotated,
                        context_instructions_override=args.context,
                    )
                    logger.info("Annotation complete for document: %r", name)

            ann_results = await asyncio.gather(
                *[_annotate(n) for n in doc_names_to_annotate],
                return_exceptions=True,
            )
            for r in ann_results:
                if isinstance(r, Exception):
                    logger.error("Annotation failed for a document: %s", r)

    # ── Stage 4: Entity Extraction ────────────────────────────────────────────
    if args.stage == "entity_extract":
        await run_entity_extraction(
            args.document,
            parallel_docs=args.parallel_docs,
            only_unextracted=args.only_unextracted,
        )
        logger.info("Entity extraction complete for document: %r", args.document)
    elif args.stage == "all":
        doc_names_to_extract = [args.document] if args.document else ingested_doc_names
        if not doc_names_to_extract:
            logger.warning("No documents available for entity extraction; skipping.")
        if doc_names_to_extract:
            semaphore = asyncio.Semaphore(args.parallel_docs)

            async def _entity_extract(name: str) -> None:
                async with semaphore:
                    await run_entity_extraction(
                        name,
                        parallel_docs=args.parallel_docs,
                        only_unextracted=args.only_unextracted,
                    )
                    logger.info("Entity extraction complete for document: %r", name)

            ee_results = await asyncio.gather(
                *[_entity_extract(n) for n in doc_names_to_extract],
                return_exceptions=True,
            )
            for r in ee_results:
                if isinstance(r, Exception):
                    logger.error("Entity extraction failed for a document: %s", r)

    # ── Archive processed files ───────────────────────────────────────────────
    if args.stage == "all":
        from scinr.newton.utils.file_archiver import archive_processed_files
        try:
            if processed_json_files:
                archive_processed_files(processed_json_files, label="intermediate JSONs")
        except Exception as exc:
            logger.warning("Archiving step failed unexpectedly: %s", exc)


def main_sync() -> None:
    """Synchronous entry point for the CLI (used by pyproject.toml scripts)."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
