"""
stages/entity_extraction.py — Stage 4: Entity extraction from annotated nodes.
"""

from __future__ import annotations

import logging
import time

from scinr.newton.results import DocumentResult, StageResult

logger = logging.getLogger(__name__)


async def run_entity_extraction(
    document_name: str,
    parallel_docs: int = 1,
    only_unextracted: bool = False,
) -> StageResult:
    """Run the entity extraction agent for an already-annotated document.

    Delegates to :func:`entity_extraction.agent.run_entity_extraction_agent`.

    Parameters
    ----------
    document_name:
        Name of the document node already annotated in Neo4j.
    parallel_docs:
        Maximum number of leaf documents to extract concurrently when
        *document_name* refers to a folder.
    only_unextracted:
        When True, only process StructureNodes without a :HAS_EXTRACTION
        relationship.

    Returns
    -------
    StageResult
        Stage result with per-document extraction counts and errors.

    Raises
    ------
    ValueError
        If *document_name* is empty.
    """
    if not document_name:
        raise ValueError("document_name must not be empty for the entity_extract stage.")

    t0 = time.monotonic()
    from scinr.newton.entity_extraction.agent import run_entity_extraction_agent

    agent_result = await run_entity_extraction_agent(
        document_name,
        parallel_docs=parallel_docs,
        only_unextracted=only_unextracted,
    )

    duration = time.monotonic() - t0

    if "results" in agent_result:
        doc_results = []
        for leaf in agent_result.get("results", []):
            targets = leaf.get("targets", [])
            errors = leaf.get("errors", [])
            doc_results.append(DocumentResult(
                document_name=leaf.get("document_name", document_name),
                nodes_processed=len(targets),
                nodes_failed=len(errors),
                errors=errors,
            ))
    else:
        targets = agent_result.get("targets", [])
        errors = agent_result.get("errors", [])
        doc_results = [DocumentResult(
            document_name=document_name,
            nodes_processed=len(targets),
            nodes_failed=len(errors),
            errors=errors,
        )]

    total_processed = sum(r.nodes_processed for r in doc_results)
    total_failed = sum(r.nodes_failed for r in doc_results)
    return StageResult(
        stage="entity_extraction",
        success=total_failed == 0,
        documents=doc_results,
        total_processed=total_processed,
        total_failed=total_failed,
        duration_seconds=duration,
    )
