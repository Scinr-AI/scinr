"""
results.py — Typed result dataclasses for scinr-ingest pipeline stages.

These dataclasses are returned by all stage functions and by run_pipeline(),
giving callers structured, type-safe access to outcomes, counts, and errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentResult:
    """Result of processing a single document through a pipeline stage.

    Attributes
    ----------
    document_name:
        The Neo4j document_name (or filename stem) of the processed document.
    nodes_processed:
        Number of nodes (or files) successfully processed within this document.
        For stages 0-2, this is 1 for success, 0 for failure.
        For stages 3-4, this is the number of StructureNodes processed.
    nodes_failed:
        Number of nodes (or files) that failed processing within this document.
    errors:
        List of error messages for this document. Empty on full success.
    """

    document_name: str
    nodes_processed: int
    nodes_failed: int
    errors: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    """Aggregated result of running a single pipeline stage.

    Attributes
    ----------
    stage:
        Stage identifier. One of: 'preprocess', 'extraction', 'ingestion',
        'annotation', 'entity_extraction', 'tabular'.
    success:
        True if total_failed == 0 and no global errors occurred.
    documents:
        Per-document results. One DocumentResult per file or document processed.
    total_processed:
        Sum of nodes_processed across all DocumentResult entries.
    total_failed:
        Sum of nodes_failed across all DocumentResult entries.
    duration_seconds:
        Wall-clock time in seconds for the entire stage.
    errors:
        Global stage-level errors not attributable to a specific document.
    """

    stage: str
    success: bool
    documents: list[DocumentResult]
    total_processed: int
    total_failed: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Aggregated result of a full run_pipeline() invocation.

    Attributes
    ----------
    success:
        True only if every executed stage succeeded (no failures or errors).
    total_duration_seconds:
        Total wall-clock time for the entire pipeline run.
    stages_executed:
        Ordered list of stage names that were actually run (skipped stages
        are not included).
    preprocess:
        StageResult for Stage 0, or None if the stage was not executed.
    extraction:
        StageResult for Stage 1, or None if the stage was not executed.
    ingestion:
        StageResult for Stage 2, or None if the stage was not executed.
    annotation:
        StageResult for Stage 3, or None if the stage was not executed.
    entity_extraction:
        StageResult for Stage 4, or None if the stage was not executed.
    tabular:
        StageResult for the tabular pipeline, or None if not executed.
    """

    success: bool
    total_duration_seconds: float
    stages_executed: list[str]
    preprocess: StageResult | None = None
    extraction: StageResult | None = None
    ingestion: StageResult | None = None
    annotation: StageResult | None = None
    entity_extraction: StageResult | None = None
    tabular: StageResult | None = None
