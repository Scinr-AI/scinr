"""
stages/annotation.py — Stage 3: Annotate StructureNodes via LLM or manual override.
"""

from __future__ import annotations

import logging
import time

from scinr.newton.annotation.agent import run_annotation_agent, run_manual_annotation
from scinr.newton.results import DocumentResult, StageResult

logger = logging.getLogger(__name__)


async def run_annotation(
    document_name: str,
    manual: bool = False,
    model_class: str | None = None,
    parallel_docs: int = 1,
    only_unannotated: bool = False,
    context_instructions_override: str | None = None,
) -> StageResult:
    """Run the annotation stage for an already-ingested document.

    In normal mode delegates to :func:`annotation.agent.run_annotation_agent`.
    In manual mode delegates to :func:`annotation.agent.run_manual_annotation`,
    assigning *model_class* to every qualifying StructureNode without invoking
    the LLM.

    Parameters
    ----------
    document_name:
        Name of the document node already present in Neo4j.
    manual:
        If True, run in manual override mode instead of the LLM agent.
    model_class:
        CamelCase model class name required when *manual* is True.
    parallel_docs:
        Maximum number of leaf documents to annotate concurrently when
        *document_name* refers to a folder.
    only_unannotated:
        When True, only process StructureNodes without a :HAS_MODEL_DECISION
        relationship. Ignored when *manual* is True.
    context_instructions_override:
        When provided, use this context string instead of fetching from Neo4j.

    Returns
    -------
    StageResult
        Stage result with per-document annotation counts and errors.

    Raises
    ------
    ValueError
        If *document_name* is empty, or if *manual* is True but *model_class*
        is not provided.
    """
    if not document_name:
        raise ValueError("document_name must not be empty for the annotation stage.")
    if manual and not model_class:
        raise ValueError("model_class is required when manual=True.")

    t0 = time.monotonic()

    if manual:
        count = await run_manual_annotation(document_name, model_class)
        logger.info(
            "Manual annotation complete: assigned '%s' to %d nodes in '%s'",
            model_class, count, document_name,
        )
        duration = time.monotonic() - t0
        return StageResult(
            stage="annotation",
            success=True,
            documents=[DocumentResult(
                document_name=document_name,
                nodes_processed=count,
                nodes_failed=0,
            )],
            total_processed=count,
            total_failed=0,
            duration_seconds=duration,
        )

    agent_result = await run_annotation_agent(
        document_name,
        parallel_docs=parallel_docs,
        only_unannotated=only_unannotated,
        context_instructions_override=context_instructions_override,
    )

    duration = time.monotonic() - t0

    # Build DocumentResult(s) from agent output
    if "results" in agent_result:
        # Multi-leaf (folder) case
        doc_results = []
        for leaf in agent_result.get("results", []):
            nodes = leaf.get("nodes_to_annotate", [])
            errors = leaf.get("errors", [])
            doc_results.append(DocumentResult(
                document_name=leaf.get("document_name", document_name),
                nodes_processed=len(nodes),
                nodes_failed=len(errors),
                errors=errors,
            ))
    else:
        # Single document case
        nodes = agent_result.get("nodes_to_annotate", [])
        errors = agent_result.get("errors", [])
        doc_results = [DocumentResult(
            document_name=document_name,
            nodes_processed=len(nodes),
            nodes_failed=len(errors),
            errors=errors,
        )]

    total_processed = sum(r.nodes_processed for r in doc_results)
    total_failed = sum(r.nodes_failed for r in doc_results)
    return StageResult(
        stage="annotation",
        success=total_failed == 0,
        documents=doc_results,
        total_processed=total_processed,
        total_failed=total_failed,
        duration_seconds=duration,
    )
