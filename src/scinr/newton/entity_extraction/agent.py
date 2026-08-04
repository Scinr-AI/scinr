"""
entity_extraction/agent.py — Public API for Stage 4 entity extraction.

Usage:
    # Async (single document or folder with nested documents)
    result = await run_entity_extraction_agent(document_name="MyDocument")

    # Sync
    result = run_entity_extraction_agent_sync(document_name="MyDocument")

When *document_name* refers to a folder document (one that has children via
IS_COMPOSED_OF in Neo4j), all **leaf** descendants are processed sequentially.
A failure in one leaf is logged and the remaining leaves are still processed.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper — single-document extraction
# ---------------------------------------------------------------------------


async def _run_entity_extraction_for_single_document(
    document_name: str,
    only_unextracted: bool = False,
) -> dict:
    """
    Run the entity extraction pipeline for exactly one document.

    Delegates to _run_entity_extraction_parallel which processes all targets
    concurrently, bounded by the global LLM_CONCURRENCY semaphore.

    Args:
        document_name: The exact Document.name as stored in Neo4j.
        only_unextracted: When True, only process nodes without an existing
            :HAS_EXTRACTION->(:ExtractionResult) relationship.

    Returns:
        Final state dict with keys: document_name, targets, errors.
    """
    logger.info("Starting entity extraction agent for document: %r", document_name)
    final_state = await _run_entity_extraction_parallel(document_name, only_unextracted=only_unextracted)

    n_targets = len(final_state.get("targets", []))
    n_errors = len(final_state.get("errors", []))
    logger.info(
        "Entity extraction complete: %d nodes processed, %d errors for document: %r",
        n_targets, n_errors, document_name,
    )
    if final_state.get("errors"):
        for err in final_state["errors"]:
            logger.warning("  Non-fatal error: %s", err)

    return final_state


async def _run_entity_extraction_parallel(
    document_name: str,
    only_unextracted: bool = False,
) -> dict:
    """Run entity extraction for a single document using intra-document parallelism.

    Fetches all targets once (load_targets), then processes them concurrently
    using asyncio.gather() bounded by the global Bedrock semaphore.

    Parameters
    ----------
    document_name:
        Neo4j Document.name.
    only_unextracted:
        When True, skip nodes that already have a :HAS_EXTRACTION relationship.

    Returns
    -------
    dict
        Compatible with EntityExtractionState: keys document_name, targets, errors.
    """
    from scinr.newton.config import get_llm_semaphore
    from scinr.newton.entity_extraction.neo4j_ops import fetch_extraction_targets
    from scinr.newton.entity_extraction.nodes import process_single_extraction_target
    from scinr.newton.ingest.config import get_async_driver

    driver = get_async_driver()
    targets = await fetch_extraction_targets(driver, document_name, only_unextracted=only_unextracted)

    logger.info(
        "_run_entity_extraction_parallel: %d targets for document %r",
        len(targets), document_name,
    )

    if not targets:
        return {"document_name": document_name, "targets": targets, "errors": []}

    semaphore = get_llm_semaphore()
    raw_results = await asyncio.gather(
        *[process_single_extraction_target(target, document_name, semaphore) for target in targets],
        return_exceptions=True,
    )

    errors: list[str] = []
    for target, result in zip(targets, raw_results):
        if isinstance(result, Exception):
            node_id = target.get("node_full_id", "unknown")
            logger.error(
                "_run_entity_extraction_parallel: unhandled exception for %r: %s", node_id, result
            )
            errors.append(f"[{node_id}] unhandled: {result}")
        elif result.get("error"):
            errors.append(result["error"])

    logger.info(
        "_run_entity_extraction_parallel: complete for %r — %d targets, %d errors",
        document_name, len(targets), len(errors),
    )
    return {
        "document_name": document_name,
        "targets": targets,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_entity_extraction_agent(
    document_name: str,
    parallel_docs: int = 1,
    only_unextracted: bool = False,
) -> dict:
    """
    Run the full entity extraction pipeline for a document (or folder) already annotated in Neo4j.

    If *document_name* refers to a folder document that has children via
    IS_COMPOSED_OF, all leaf descendants are resolved and extracted.  Up to
    *parallel_docs* leaves are processed concurrently; a failure on any
    individual leaf is logged and the remaining leaves continue.

    Traverses all StructureNodes that have a matched ModelDecision and at least one
    unextracted InfoUnit, runs the composite LLM extraction, and writes the entity
    subgraph to Neo4j. Marks each InfoUnit as extracted on completion.

    Args:
        document_name: The exact Document.name as stored in Neo4j.
        parallel_docs: Maximum number of leaf documents to extract concurrently.
            Defaults to ``1`` (sequential, backward-compatible behaviour).
        only_unextracted: When True, only process nodes that do NOT already have a
            :HAS_EXTRACTION->(:ExtractionResult) relationship. Defaults to False.

    Returns:
        If a single document: the final EntityExtractionState dict.
        If multiple leaf documents: an aggregated dict with keys:
            - document_name: the original name passed in
            - leaf_documents: list of resolved leaf names
            - results: list of per-leaf EntityExtractionState dicts
            - targets: all targets across all leaves (combined)
            - errors: all errors across all leaves (combined)

    Raises:
        ValueError: if document_name is empty.
    """
    if not document_name:
        raise ValueError("document_name must be a non-empty string.")

    from scinr.newton.exceptions import PreconditionError
    from scinr.newton.ingest.config import get_async_driver
    from scinr.newton.utils.document_resolver import resolve_leaf_document_names_async

    driver = get_async_driver()
    async with driver.session() as _session:
        # Check 1: document exists
        _result1 = await _session.run(
            "MATCH (d:Document {name: $name, latest: true}) RETURN count(d) AS n",
            name=document_name,
        )
        _doc_count = (await _result1.single())["n"]
        if _doc_count == 0:
            raise PreconditionError(
                f"Document '{document_name}' not found in Neo4j (latest=true). "
                f"Run run_ingestion() before run_entity_extraction_agent()."
            )
        # Check 2: at least one annotated node exists
        _result2 = await _session.run(
            "MATCH (d:Document {name: $n, latest: true})"
            "-[:HAS_STRUCTURE|HAS_CHILD*1..]->(sn:StructureNode)"
            "-[:HAS_MODEL_DECISION]->() RETURN count(sn) AS n",
            n=document_name,
        )
        _annotated_count = (await _result2.single())["n"]
        if _annotated_count == 0:
            raise PreconditionError(
                f"Document '{document_name}' has no annotated StructureNodes. "
                f"Run run_annotation_agent() before run_entity_extraction_agent()."
            )

    leaf_names = await resolve_leaf_document_names_async(driver, document_name)

    # Single document (no IS_COMPOSED_OF children): original behaviour
    if len(leaf_names) == 1 and leaf_names[0] == document_name:
        return await _run_entity_extraction_for_single_document(
            document_name, only_unextracted=only_unextracted
        )

    # Multiple leaf documents: process with bounded concurrency, accumulate results
    all_errors: list[str] = []
    all_targets: list[dict] = []
    results: list[dict] = []

    semaphore = asyncio.Semaphore(parallel_docs)

    async def _run_leaf(leaf_name: str) -> dict:
        async with semaphore:
            logger.info("Processing leaf document %r (parent: %r)", leaf_name, document_name)
            return await _run_entity_extraction_for_single_document(
                leaf_name, only_unextracted=only_unextracted
            )

    leaf_results = await asyncio.gather(
        *[_run_leaf(name) for name in leaf_names],
        return_exceptions=True,
    )

    for leaf_name, result in zip(leaf_names, leaf_results):
        if isinstance(result, Exception):
            logger.error("Entity extraction failed for leaf document %r: %s", leaf_name, result)
            all_errors.append(f"[{leaf_name}] {result}")
        else:
            results.append(result)
            all_errors.extend(result.get("errors", []))
            all_targets.extend(result.get("targets", []))

    logger.info(
        "Entity extraction complete for folder %r: %d leaves, %d total targets, %d total errors",
        document_name,
        len(leaf_names),
        len(all_targets),
        len(all_errors),
    )

    return {
        "document_name": document_name,
        "leaf_documents": leaf_names,
        "results": results,
        "targets": all_targets,
        "errors": all_errors,
    }


def run_entity_extraction_agent_sync(
    document_name: str,
    parallel_docs: int = 1,
    only_unextracted: bool = False,
) -> dict:
    """Synchronous wrapper around run_entity_extraction_agent."""
    return asyncio.run(
        run_entity_extraction_agent(
            document_name,
            parallel_docs=parallel_docs,
            only_unextracted=only_unextracted,
        )
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from scinr.newton.utils.logging_config import setup_logging
    setup_logging(log_dir=Path("logs"))

    parser = argparse.ArgumentParser(
        description=(
            "scinr-ingest entity extraction agent — Stage 4\n"
            "\n"
            "If --document refers to a folder (a document with IS_COMPOSED_OF children),\n"
            "all leaf descendants are processed automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Exact Document.name as stored in Neo4j",
    )
    parser.add_argument(
        "--parallel-docs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Maximum number of leaf documents to process concurrently. "
            "Defaults to 1 (sequential)."
        ),
    )
    parser.add_argument(
        "--only-unextracted",
        action="store_true",
        default=False,
        help=(
            "Only process StructureNodes that do not already have a "
            ":HAS_EXTRACTION->(:ExtractionResult) relationship."
        ),
    )
    args = parser.parse_args()

    result = run_entity_extraction_agent_sync(
        args.document,
        parallel_docs=args.parallel_docs,
        only_unextracted=args.only_unextracted,
    )
    errors = result.get("errors", [])
    n_targets = len(result.get("targets", []))
    logger.info("Entity extraction complete: %d nodes processed", n_targets)
    if errors:
        logger.info("Non-fatal errors (%d):", len(errors))
        for e in errors:
            logger.info("  - %s", e)
    else:
        logger.info("No errors.")
