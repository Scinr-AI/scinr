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

    Attributes:
        document_name: The Neo4j document_name (or filename stem) of the processed document.
        nodes_processed: Number of nodes (or files) successfully processed within this document.
            For stages 0-2, this is 1 for success, 0 for failure.
            For stages 3-4, this is the number of StructureNodes processed.
        nodes_failed: Number of nodes (or files) that failed processing within this document.
        errors: List of error messages for this document. Empty on full success.
    """

    document_name: str
    nodes_processed: int
    nodes_failed: int
    errors: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    """Aggregated result of running a single pipeline stage.

    Attributes:
        stage: Stage identifier. One of: 'preprocess', 'extraction', 'ingestion',
            'annotation', 'entity_extraction', 'tabular'.
        success: True if total_failed == 0 and no global errors occurred.
        documents: Per-document results. One DocumentResult per file or document processed.
        total_processed: Sum of nodes_processed across all DocumentResult entries.
        total_failed: Sum of nodes_failed across all DocumentResult entries.
        duration_seconds: Wall-clock time in seconds for the entire stage.
        errors: Global stage-level errors not attributable to a specific document.
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

    Attributes:
        success: True only if every executed stage succeeded (no failures or errors).
        total_duration_seconds: Total wall-clock time for the entire pipeline run.
        stages_executed: Ordered list of stage names that were actually run (skipped stages
            are not included).
        preprocess: StageResult for Stage 0, or None if the stage was not executed.
        extraction: StageResult for Stage 1, or None if the stage was not executed.
        ingestion: StageResult for Stage 2, or None if the stage was not executed.
        annotation: StageResult for Stage 3, or None if the stage was not executed.
        entity_extraction: StageResult for Stage 4, or None if the stage was not executed.
        tabular: StageResult for the tabular pipeline, or None if not executed.
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


@dataclass
class DeletionResult:
    """Result of a delete_document() call — full Document + cascade + GC deletion.

    Attributes:
        path: The Document ``path`` that was targeted for deletion.
        version: The specific version requested, or None if all versions were targeted.
        found: True if at least one matching Document existed before deletion. When False,
            all counters below are 0 and no delete or GC queries were executed.
        versions_deleted: Sorted list of integer versions that matched and were deleted.
            Empty when found is False.
        documents_deleted: Number of :Document nodes deleted (the matched Document(s) plus
            any reached via IS_COMPOSED_OF*).
        structure_nodes_deleted: Number of :StructureNode nodes deleted.
        info_units_deleted: Number of :InfoUnit nodes deleted.
        model_decisions_deleted: Number of :ModelDecision nodes deleted.
        proposed_models_deleted: Number of :ProposedModel nodes deleted.
        proposed_fields_deleted: Number of :ProposedField nodes deleted.
        extraction_results_deleted: Number of :ExtractionResult nodes deleted.
        gc_entity_model_instance_deleted: Total :Entity/:ModelInstance nodes deleted across
            all garbage-collection iterations.
        gc_entity_model_instance_passes: Number of GC iterations actually run for the
            Entity/ModelInstance pass (capped at GC_MAX_PASSES).
        gc_labeled_entity_deleted: Total :LabeledEntity nodes deleted across all
            garbage-collection iterations.
        gc_labeled_entity_passes: Number of GC iterations actually run for the
            LabeledEntity pass (capped at GC_MAX_PASSES).
        raw_files_deleted: Number of raw_file_ids for which deletion was attempted
            against the storage backend (GridFS + metadata), for the raw_file_ids
            referenced by the deleted Document(s) and their descendants. Idempotent —
            includes ids that were already absent, since delete() returns None either way.
        converted_pages_deleted: Number of ConvertedPageRecord (converted Markdown
            pages) deleted from the storage layer for the same raw_file_ids.
    """

    path: str
    version: int | None
    found: bool
    versions_deleted: list[int]
    documents_deleted: int
    structure_nodes_deleted: int
    info_units_deleted: int
    model_decisions_deleted: int
    proposed_models_deleted: int
    proposed_fields_deleted: int
    extraction_results_deleted: int
    gc_entity_model_instance_deleted: int
    gc_entity_model_instance_passes: int
    gc_labeled_entity_deleted: int
    gc_labeled_entity_passes: int
    raw_files_deleted: int
    converted_pages_deleted: int
