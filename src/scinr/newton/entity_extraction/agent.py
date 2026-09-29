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

Documents are always selected within one tenant (``tenant_id``, ``None`` =
public): by ``doc_path`` when given, otherwise by ``document_name``. Each leaf
is then processed by its ``(tenant_id, path)`` — a name never selects nodes of
another document, or of another tenant.
"""
from __future__ import annotations

import asyncio
import logging

from scinr.newton.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper — single-document extraction
# ---------------------------------------------------------------------------


async def _run_entity_extraction_for_single_document(
    document_name: str,
    only_unextracted: bool = False,
    *,
    tenant_id: str | None,
    doc_path: str,
) -> dict:
    """
    Run the entity extraction pipeline for exactly one document.

    Delegates to _run_entity_extraction_parallel which processes all targets
    concurrently, bounded by the global LLM_CONCURRENCY semaphore.

    Args:
        document_name: Document display name (provenance on the extractions).
        only_unextracted: When True, only process nodes without an existing
            :HAS_EXTRACTION->(:ExtractionResult) relationship.
        tenant_id: Owner of the document (``None`` = public).
        doc_path: Path of the document — with *tenant_id*, the selector.

    Returns:
        Final state dict with keys: document_name, doc_path, targets, errors.
    """
    logger.info(
        "Starting entity extraction agent for document: %r (path=%r, tenant=%r)",
        document_name, doc_path, tenant_id,
    )
    final_state = await _run_entity_extraction_parallel(
        document_name, only_unextracted=only_unextracted, tenant_id=tenant_id, doc_path=doc_path
    )

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
    *,
    tenant_id: str | None,
    doc_path: str,
) -> dict:
    """Run entity extraction for a single document using intra-document parallelism.

    Fetches all targets once (load_targets), then processes them concurrently
    using asyncio.gather() bounded by the global Bedrock semaphore.

    Parameters
    ----------
    document_name:
        Neo4j Document.name (provenance only).
    only_unextracted:
        When True, skip nodes that already have a :HAS_EXTRACTION relationship.
    tenant_id, doc_path:
        Select the latest :Document to extract.

    Returns
    -------
    dict
        Compatible with EntityExtractionState: keys document_name, doc_path,
        targets, errors.
    """
    from scinr.newton.config import get_llm_semaphore
    from scinr.newton.entity_extraction.neo4j_ops import fetch_extraction_targets
    from scinr.newton.entity_extraction.nodes import process_single_extraction_target
    from scinr.newton.ingest.config import get_async_driver

    driver = get_async_driver()
    targets = await fetch_extraction_targets(
        driver, tenant_id=tenant_id, doc_path=doc_path, only_unextracted=only_unextracted
    )

    logger.info(
        "_run_entity_extraction_parallel: %d targets for document %r",
        len(targets), document_name,
    )

    if not targets:
        return {"document_name": document_name, "doc_path": doc_path, "targets": targets, "errors": []}

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
        "doc_path": doc_path,
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
    *,
    tenant_id: str | None = None,
    doc_path: str | None = None,
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
        document_name: The exact Document.name as stored in Neo4j. Selects the
            root document(s) only when *doc_path* is not given (every latest
            document of the tenant with that name); otherwise it is only a label.
        parallel_docs: Maximum number of leaf documents to extract concurrently.
            Defaults to ``1`` (sequential, backward-compatible behaviour).
        only_unextracted: When True, only process nodes that do NOT already have a
            :HAS_EXTRACTION->(:ExtractionResult) relationship. Defaults to False.
        tenant_id: Owner of the document (``None`` = public). Only that
            tenant's documents are read or written.
        doc_path: Path of the root document (preferred selector).

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
    from scinr.newton.utils.document_resolver import (
        latest_document_pattern,
        resolve_leaf_documents_async,
    )
    doc_pattern, doc_params = latest_document_pattern(
        "d", tenant_id=tenant_id, doc_path=doc_path, document_name=document_name
    )
    doc_label = f"'{doc_path or document_name}' (tenant={tenant_id!r})"
    cfg = get_config()
    driver = get_async_driver()
    async with driver.session(database=cfg.neo4j_database) as _session:
        # Check 1: document exists
        _result1 = await _session.run(f"MATCH {doc_pattern} RETURN count(d) AS n", **doc_params)
        _doc_count = (await _result1.single())["n"]
        if _doc_count == 0:
            raise PreconditionError(
                f"Document {doc_label} not found in Neo4j (latest=true). "
                f"Run run_ingestion() before run_entity_extraction_agent()."
            )
        # Check 2: at least one annotated node exists
        _result2 = await _session.run(
            f"MATCH {doc_pattern}"
            "-[:HAS_STRUCTURE|HAS_CHILD*1..]->(sn:StructureNode)"
            "-[:HAS_MODEL_DECISION]->() RETURN count(sn) AS n",
            **doc_params,
        )
        _annotated_count = (await _result2.single())["n"]
        if _annotated_count == 0:
            raise PreconditionError(
                f"Document {doc_label} has no annotated StructureNodes. "
                f"Run run_annotation_agent() before run_entity_extraction_agent()."
            )

    leaves = await resolve_leaf_documents_async(
        driver, tenant_id=tenant_id, doc_path=doc_path, document_name=document_name
    )
    leaf_names = [leaf.name for leaf in leaves]

    # Single document (no IS_COMPOSED_OF children): original behaviour
    if len(leaves) == 1 and (
        leaves[0].path == doc_path if doc_path is not None else leaves[0].name == document_name
    ):
        return await _run_entity_extraction_for_single_document(
            leaves[0].name,
            only_unextracted=only_unextracted,
            tenant_id=tenant_id,
            doc_path=leaves[0].path,
        )

    # Multiple leaf documents: process with bounded concurrency, accumulate results
    all_errors: list[str] = []
    all_targets: list[dict] = []
    results: list[dict] = []

    semaphore = asyncio.Semaphore(parallel_docs)

    async def _run_leaf(leaf) -> dict:
        async with semaphore:
            logger.info(
                "Processing leaf document %r (path=%r, parent: %r)",
                leaf.name, leaf.path, doc_path or document_name,
            )
            return await _run_entity_extraction_for_single_document(
                leaf.name,
                only_unextracted=only_unextracted,
                tenant_id=tenant_id,
                doc_path=leaf.path,
            )

    leaf_results = await asyncio.gather(
        *[_run_leaf(leaf) for leaf in leaves],
        return_exceptions=True,
    )

    for leaf, result in zip(leaves, leaf_results):
        if isinstance(result, Exception):
            logger.error("Entity extraction failed for leaf document %r: %s", leaf.path, result)
            all_errors.append(f"[{leaf.name}] {result}")
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
    *,
    tenant_id: str | None = None,
    doc_path: str | None = None,
) -> dict:
    """Synchronous wrapper around run_entity_extraction_agent."""
    return asyncio.run(
        run_entity_extraction_agent(
            document_name,
            parallel_docs=parallel_docs,
            only_unextracted=only_unextracted,
            tenant_id=tenant_id,
            doc_path=doc_path,
        )
    )
