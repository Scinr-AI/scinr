"""
Public API for the scinr-ingest annotation agent.

Usage:
    # Async — LLM agent (single document or folder with nested documents)
    result = await run_annotation_agent(document_name="MyDocument")

    # Sync — LLM agent
    result = run_annotation_agent_sync(document_name="MyDocument")

    # Async — manual override (single document or folder with nested documents)
    count = await run_manual_annotation(document_name="MyDocument", model_class="DrugProductComposition")

    # Sync — manual override
    count = run_manual_annotation_sync(document_name="MyDocument", model_class="DrugProductComposition")

When *document_name* refers to a folder document (one that has children via
IS_COMPOSED_OF in Neo4j), all **leaf** descendants are processed sequentially.
A failure in one leaf is logged and the remaining leaves are still processed.
"""
from __future__ import annotations

import asyncio
import logging

from scinr.newton.config import get_config, get_llm_semaphore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper — single-document annotation
# ---------------------------------------------------------------------------


async def _run_annotation_for_single_document(
    document_name: str,
    only_unannotated: bool = False,
    context_instructions_override: str | None = None,
) -> dict:
    """
    Run the annotation pipeline for exactly one document.

    Delegates to _run_annotation_parallel which processes all nodes
    concurrently, bounded by the global LLM_CONCURRENCY semaphore.

    Args:
        document_name: The exact Document.name as stored in Neo4j.
        only_unannotated: When True, only process nodes that do NOT already have
            a :HAS_MODEL_DECISION relationship.
        context_instructions_override: When provided, use this context string
            instead of fetching context_instructions from Neo4j.

    Returns:
        Final state dict with keys: document_name, nodes_to_annotate, errors.
    """
    logger.info("Starting annotation agent for document: %r", document_name)
    final_state = await _run_annotation_parallel(
        document_name,
        only_unannotated=only_unannotated,
        context_instructions_override=context_instructions_override,
    )

    n_nodes = len(final_state.get("nodes_to_annotate", []))
    n_errors = len(final_state.get("errors", []))
    logger.info(
        "Annotation complete: %d nodes processed, %d errors for document: %r",
        n_nodes, n_errors, document_name,
    )
    if final_state.get("errors"):
        for err in final_state["errors"]:
            logger.warning("  Non-fatal error: %s", err)

    return final_state


async def _run_annotation_parallel(
    document_name: str,
    only_unannotated: bool = False,
    context_instructions_override: str | None = None,
) -> dict:
    """Run annotation for a single document using intra-document parallelism.

    Fetches all nodes once (load_nodes), then processes them concurrently
    using asyncio.gather() bounded by the global Bedrock semaphore.

    Parameters
    ----------
    document_name:
        Neo4j Document.name.
    only_unannotated:
        When True, skip nodes that already have a :HAS_MODEL_DECISION relationship.
    context_instructions_override:
        When provided, use this context string instead of fetching
        context_instructions from Neo4j.

    Returns
    -------
    dict
        Compatible with AnnotationState: keys document_name, nodes_to_annotate, errors.
    """
    from scinr.newton.annotation.neo4j_ops import (
        ensure_catalog_models_once,
        ensure_theme_structure_once,
        fetch_document_context_instructions,
        fetch_nodes_to_annotate,
    )
    from scinr.newton.annotation.nodes import process_single_annotation_node
    from scinr.newton.ingest.config import get_async_driver
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    driver = get_async_driver()
    await ensure_catalog_models_once(driver)
    await ensure_theme_structure_once(driver, theme_registry)
    nodes = await fetch_nodes_to_annotate(driver, document_name, only_unannotated=only_unannotated)
    if context_instructions_override is not None:
        doc_context = context_instructions_override
    else:
        doc_context = await fetch_document_context_instructions(driver, document_name)

    logger.info(
        "_run_annotation_parallel: %d nodes to annotate for document %r",
        len(nodes), document_name,
    )

    if not nodes:
        return {"document_name": document_name, "nodes_to_annotate": nodes, "errors": []}

    semaphore = get_llm_semaphore()
    raw_results = await asyncio.gather(
        *[
            process_single_annotation_node(node, document_name, semaphore, user_context=doc_context or "")
            for node in nodes
        ],
        return_exceptions=True,
    )

    errors: list[str] = []
    for node, result in zip(nodes, raw_results):
        if isinstance(result, Exception):
            node_id = node.get("node_id", "unknown")
            logger.error("_run_annotation_parallel: unhandled exception for %r: %s", node_id, result)
            errors.append(f"[{node_id}] unhandled: {result}")
        elif result.get("error"):
            errors.append(result["error"])

    logger.info(
        "_run_annotation_parallel: complete for %r — %d nodes, %d errors",
        document_name, len(nodes), len(errors),
    )
    return {
        "document_name": document_name,
        "nodes_to_annotate": nodes,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_annotation_agent(
    document_name: str,
    parallel_docs: int = 1,
    only_unannotated: bool = False,
    context_instructions_override: str | None = None,
) -> dict:
    """
    Run the full annotation pipeline for a document (or folder) already ingested in Neo4j.

    If *document_name* refers to a folder document that has children via
    IS_COMPOSED_OF, all leaf descendants are resolved and annotated.  Up to
    *parallel_docs* leaves are processed concurrently; a failure on any
    individual leaf is logged and the remaining leaves continue.

    Traverses all StructureNodes that have at least one InfoUnit, calls the
    two-step LLM pipeline (decide_model → format_decision) for each, and writes
    the resulting AnnotationDecision subgraph back to Neo4j.

    Args:
        document_name: The exact Document.name as stored in Neo4j.
        parallel_docs: Maximum number of leaf documents to annotate concurrently.
            Defaults to ``1`` (sequential, backward-compatible behaviour).
        only_unannotated: When True, only process StructureNodes that do NOT already
            have a :HAS_MODEL_DECISION relationship. Defaults to False.
        context_instructions_override: When provided, use this context string instead
            of fetching context_instructions from Neo4j. Defaults to None (Neo4j
            fetch is used).

    Returns:
        If a single document: the final AnnotationState dict.
        If multiple leaf documents: an aggregated dict with keys:
            - document_name: the original name passed in
            - leaf_documents: list of resolved leaf names
            - results: list of per-leaf AnnotationState dicts
            - nodes_to_annotate: all nodes across all leaves (combined)
            - errors: all errors across all leaves (combined)

    Raises:
        ValueError: if document_name is empty.
    """
    if not document_name:
        raise ValueError("document_name must be a non-empty string.")

    from scinr.newton.exceptions import PreconditionError
    from scinr.newton.ingest.config import get_async_driver
    from scinr.newton.utils.document_resolver import resolve_leaf_document_names_async
    cfg = get_config()
    driver = get_async_driver()
    async with driver.session(database=cfg.neo4j_database) as _session:
        _result = await _session.run(
            "MATCH (d:Document {name: $name, latest: true}) RETURN count(d) AS n",
            name=document_name,
        )
        _pre_row = await _result.single()
    if _pre_row["n"] == 0:
        raise PreconditionError(
            f"Document '{document_name}' not found in Neo4j (latest=true). "
            f"Run run_ingestion() before run_annotation_agent()."
        )

    leaf_names = await resolve_leaf_document_names_async(driver, document_name)

    # Single document (no IS_COMPOSED_OF children): original behaviour
    if len(leaf_names) == 1 and leaf_names[0] == document_name:
        return await _run_annotation_for_single_document(
            document_name,
            only_unannotated=only_unannotated,
            context_instructions_override=context_instructions_override,
        )

    # Multiple leaf documents: process with bounded concurrency, accumulate results
    all_errors: list[str] = []
    all_nodes: list = []
    results: list[dict] = []

    semaphore = asyncio.Semaphore(parallel_docs)

    async def _run_leaf(leaf_name: str) -> dict:
        async with semaphore:
            logger.info("Processing leaf document %r (parent: %r)", leaf_name, document_name)
            return await _run_annotation_for_single_document(
                leaf_name,
                only_unannotated=only_unannotated,
                context_instructions_override=context_instructions_override,
            )

    leaf_results = await asyncio.gather(
        *[_run_leaf(name) for name in leaf_names],
        return_exceptions=True,
    )

    for leaf_name, result in zip(leaf_names, leaf_results):
        if isinstance(result, Exception):
            logger.error("Annotation failed for leaf document %r: %s", leaf_name, result)
            all_errors.append(f"[{leaf_name}] {result}")
        else:
            results.append(result)
            all_errors.extend(result.get("errors", []))
            all_nodes.extend(result.get("nodes_to_annotate", []))

    logger.info(
        "Annotation complete for folder %r: %d leaves, %d total nodes, %d total errors",
        document_name,
        len(leaf_names),
        len(all_nodes),
        len(all_errors),
    )

    return {
        "document_name": document_name,
        "leaf_documents": leaf_names,
        "results": results,
        "nodes_to_annotate": all_nodes,
        "errors": all_errors,
    }


def run_annotation_agent_sync(
    document_name: str,
    parallel_docs: int = 1,
    only_unannotated: bool = False,
    context_instructions_override: str | None = None,
) -> dict:
    """Synchronous wrapper around run_annotation_agent."""
    return asyncio.run(
        run_annotation_agent(
            document_name,
            parallel_docs=parallel_docs,
            only_unannotated=only_unannotated,
            context_instructions_override=context_instructions_override,
        )
    )


async def run_manual_annotation(document_name: str, model_class: str) -> int:
    """
    Manually assign a fixed ModelDecision to all StructureNodes with InfoUnits
    for the specified document (or all leaf descendants if it is a folder).

    Bypasses the LLM annotation agent entirely. Every qualifying StructureNode
    receives a new :ModelDecision {source: 'manual'} pointing to the given model
    class. Any pre-existing ModelDecision and ExtractionResult subgraphs for those
    nodes are replaced.

    If *document_name* refers to a folder, all leaf descendants are processed
    sequentially. A failure on any leaf is logged and the remaining leaves
    continue processing.

    Args:
        document_name: The exact Document.name as stored in Neo4j.
        model_class:   CamelCase name of the Pydantic model class to assign.
                       Must exist in the model registry.

    Returns:
        Total number of StructureNodes updated across all processed documents.

    Raises:
        ValueError: if document_name or model_class is empty.
        KeyError:   if model_class is not found in the model registry.
    """
    if not document_name:
        raise ValueError("document_name must be a non-empty string.")
    if not model_class:
        raise ValueError("model_class must be a non-empty string.")

    # Validate that the model exists in the registry before touching Neo4j.
    # resolve_model_class raises KeyError with the full list of available classes.
    from scinr.newton.entity_extraction.model_resolver import resolve_model_class
    resolve_model_class(model_class)

    from scinr.newton.annotation.neo4j_ops import write_manual_annotation
    from scinr.newton.ingest.config import get_async_driver
    from scinr.newton.utils.document_resolver import resolve_leaf_document_names_async

    driver = get_async_driver()
    leaf_names = await resolve_leaf_document_names_async(driver, document_name)
    total_count = 0
    for leaf_name in leaf_names:
        try:
            count = await write_manual_annotation(driver, leaf_name, model_class)
            total_count += count
            logger.info(
                "Manual annotation: assigned %r to %d nodes in %r",
                model_class,
                count,
                leaf_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Manual annotation failed for leaf document %r: %s", leaf_name, exc
            )

    return total_count


def run_manual_annotation_sync(document_name: str, model_class: str) -> int:
    """Synchronous wrapper around run_manual_annotation."""
    return asyncio.run(run_manual_annotation(document_name, model_class))


# CLI entry point
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from scinr.newton.utils.logging_config import setup_logging
    setup_logging(log_dir=Path("logs"))

    parser = argparse.ArgumentParser(
        description=(
            "scinr-ingest annotation agent — assigns extraction models to StructureNodes.\n"
            "\n"
            "By default, runs the LLM annotation agent. Use --manual to assign a fixed\n"
            "model class to all nodes without invoking the LLM.\n"
            "\n"
            "If --document refers to a folder (a document with IS_COMPOSED_OF children),\n"
            "all leaf descendants are processed automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Exact Document.name as stored in Neo4j.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        default=False,
        help=(
            "Manual override mode: assign a fixed model class to all StructureNodes "
            "instead of running the LLM agent. Requires --model."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="CLASS_NAME",
        help=(
            "Exact CamelCase model class name to assign (e.g. 'DrugProductComposition'). "
            "Required when --manual is specified."
        ),
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
        "--only-unannotated",
        action="store_true",
        default=False,
        help=(
            "Only process StructureNodes that do not already have a "
            ":HAS_MODEL_DECISION relationship."
        ),
    )
    args = parser.parse_args()

    if args.manual and not args.model:
        parser.error("--model is required when --manual is specified.")
    if args.model and not args.manual:
        logger.warning("--model is ignored without --manual; running LLM agent.")

    if args.manual:
        try:
            count = run_manual_annotation_sync(args.document, args.model)
            print(f"Assigned '{args.model}' to {count} nodes in '{args.document}'.")
        except KeyError as exc:
            raise SystemExit(f"error: {exc}") from exc
    else:
        result = run_annotation_agent_sync(
            args.document,
            parallel_docs=args.parallel_docs,
            only_unannotated=args.only_unannotated,
        )
        errors = result.get("errors", [])
        n_nodes = len(result.get("nodes_to_annotate", []))

        logger.info("Annotation complete: %d nodes processed", n_nodes)
        if errors:
            logger.info("Non-fatal errors (%d):", len(errors))
            for e in errors:
                logger.info("  - %s", e)
        else:
            logger.info("No errors.")
