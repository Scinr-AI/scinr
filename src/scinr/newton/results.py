"""
results.py — Typed result dataclasses for scinr-ingest pipeline stages.

These dataclasses are returned by all stage functions and by run_pipeline(),
giving callers structured, type-safe access to outcomes, counts, and errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from scinr.newton.utils.redaction import redact_secrets


@dataclass
class DocumentResult:
    """Result of processing a single document through a pipeline stage.

    Attributes:
        document_name: The Neo4j document_name of the processed document (the source file name with its extension, e.g. ``informe.pdf``, or the JSON file stem when the stage started from JSON files).
        nodes_processed: Number of nodes (or files) successfully processed within this document.
            For stages 0-2, this is 1 for success, 0 for failure.
            For stages 3-4, this is the number of StructureNodes processed.
            For the tabular stage, this is the number of tables (sheets) written.
        nodes_failed: Number of nodes (or files) that failed processing within this document.
            For the tabular stage, the number of sheets with errors.
        errors: List of error messages for this document. Empty on full success.
        missing_page_ranges: Preprocess only. Page ranges ``[start, end)`` of a PDF
            skipped in best-effort mode (``mistral_ocr_error_strategy``). None when
            nothing was skipped.
        missing_segments: Preprocess only. Time ranges ``[start_ms, end_ms)`` of an
            audio or video whose transcription was skipped in best-effort mode
            (``media_error_strategy``). None when nothing was skipped.
        missing_images: Preprocess only. Images ``(page_index, image_index)`` (embedded
            images, video frames) that could not be described and were skipped in
            best-effort mode. None when nothing was skipped.

    A document with any of the three is ingested from the parts that were
    converted: it counts as processed, not failed. See :attr:`has_omissions`
    and :attr:`PipelineResult.documents_with_omissions`.
    """

    document_name: str
    nodes_processed: int
    nodes_failed: int
    errors: list[str] = field(default_factory=list)
    missing_page_ranges: list[tuple[int, int]] | None = None
    missing_segments: list[tuple[int, int]] | None = None
    missing_images: list[tuple[int, int]] | None = None

    @property
    def has_omissions(self) -> bool:
        """True when part of the document was skipped in best-effort mode."""
        return bool(self.missing_page_ranges or self.missing_segments or self.missing_images)

    def __post_init__(self) -> None:
        # Error strings are built from driver exceptions and returned to API
        # callers verbatim: scrub any credentials they may echo.
        self.errors = [redact_secrets(e) if isinstance(e, str) else e for e in self.errors]


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

    def __post_init__(self) -> None:
        # See DocumentResult.__post_init__.
        self.errors = [redact_secrets(e) if isinstance(e, str) else e for e in self.errors]


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

    Documents converted only in part (best-effort mode) are listed by
    :attr:`documents_with_omissions`; ``success`` does not change for them.
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

    @property
    def documents_with_omissions(self) -> list[DocumentResult]:
        """Preprocess results of the documents with parts skipped in best-effort mode."""
        if self.preprocess is None:
            return []
        return [d for d in self.preprocess.documents if d.has_omissions]


@dataclass
class SkippedFile:
    """A file under ``input_raw`` that the pipeline cannot interpret.

    Attributes:
        path: The file path.
        reason: Why it is skipped, e.g. ``"unsupported format: .xyz"``.
    """

    path: str
    reason: str


@dataclass
class DiscoveredDocuments:
    """What run_pipeline() would do with every file under an ``input_raw``
    folder, without converting or reading anything (see discover_documents()).

    Attributes:
        tabular: The ``doc_path`` each tabular file will get.
        documents: The ``doc_path`` each convertible file will get.
        skipped: The files the pipeline cannot interpret; they create nothing.
    """

    tabular: list[str]
    documents: list[str]
    skipped: list[SkippedFile]

    @property
    def count(self) -> int:
        """The leaf :Document nodes the run can create, at most.

        An upper bound: an unreadable or empty tabular file and a file that
        fails before ingestion create no :Document. With ``update_mode=False``
        a file whose ``doc_path`` already exists creates a new version.
        """
        return len(self.tabular) + len(self.documents)


@dataclass
class DeletedDocument:
    """One :Document node deleted by delete_document().

    Attributes:
        path: The Document ``path`` (its ``doc_path``).
        version: The Document version.
        is_folder: True for a folder document, False for a leaf document.
    """

    path: str
    version: int
    is_folder: bool


@dataclass
class DeletionResult:
    """Result of a delete_document() call — full Document + cascade + GC deletion.

    Attributes:
        path: The Document ``path`` that was targeted for deletion, or None when
            the deletion was selected by ``job_id`` instead.
        version: The specific version requested, or None if all versions were targeted.
        job_id: The ``job_id`` selector (one value or several) that was targeted, or None when the deletion
            was selected by ``path`` instead.
        tenant_id: The tenant the deletion was scoped to (always applied), or None
            when it targeted public documents.
        created_by_user_id: The ``created_by_user_id`` filter applied to the match, or
            None if no such filter was requested.
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
        gc_entity_model_instance_deleted: :Entity/:ModelInstance nodes this deletion
            left orphaned, and deleted. Only the nodes that hung from the deleted
            :ExtractionResult nodes are checked, not the whole tenant (see
            collect_orphans() for that).
        gc_entity_model_instance_passes: Number of GC rounds run over :Entity/:ModelInstance
            candidates: 0 when there was none; one more each time a deleted node
            turned what it pointed at into candidates.
        gc_labeled_entity_deleted: :LabeledEntity nodes this deletion left orphaned,
            and deleted.
        gc_labeled_entity_passes: Number of GC rounds run over :LabeledEntity candidates.
        raw_files_deleted: Number of raw_file_ids for which deletion was attempted
            against the storage backend (GridFS + metadata), for the raw_file_ids
            referenced by the deleted Document(s) and their descendants. Idempotent —
            includes ids that were already absent, since delete() returns None either way.
        converted_pages_deleted: Number of ConvertedPageRecord (converted Markdown
            pages) deleted from the storage layer for the same raw_file_ids.
        page_images_deleted: Number of PageImageRecord (stored images: embedded
            images, video frames) deleted from the storage layer for the same
            raw_file_ids. A backend that stores no images always reports 0.
        documents: The :Document nodes the deletion worked on — the matched ones
            and their ``IS_COMPOSED_OF*`` descendants, descendants first (the
            order they are deleted in) — with their path, version and
            ``is_folder``. Empty when found is False. A call that completes an
            interrupted deletion only lists what was still there. Lets a caller
            that keeps per-document counters outside the graph adjust them
            without repeating the selection.
    """

    path: str | None
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
    job_id: str | list[str] | tuple[str, ...] | None = None
    tenant_id: str | None = None
    created_by_user_id: str | list[str] | tuple[str, ...] | None = None
    page_images_deleted: int = 0
    documents: list[DeletedDocument] = field(default_factory=list)


@dataclass
class OrphanCollectionResult:
    """Result of a collect_orphans() call — sweep of every orphan of one tenant.

    Attributes:
        tenant_id: The tenant that was swept (API value: None = public).
        gc_entity_model_instance_deleted: :Entity/:ModelInstance nodes of the tenant
            deleted because no :ExtractionResult reached them.
        gc_entity_model_instance_passes: Iterations of the Entity/ModelInstance pass
            (each one checks every node of the tenant; it stops at the first
            iteration that deletes nothing, capped at GC_MAX_PASSES).
        gc_labeled_entity_deleted: :LabeledEntity nodes of the tenant deleted because
            nothing pointed at them.
        gc_labeled_entity_passes: Iterations of the LabeledEntity pass (same rule).
    """

    tenant_id: str | None
    gc_entity_model_instance_deleted: int = 0
    gc_entity_model_instance_passes: int = 0
    gc_labeled_entity_deleted: int = 0
    gc_labeled_entity_passes: int = 0


@dataclass
class FreezeResult:
    """Result of a freeze_document() call — snapshot export + graph reduction.

    Attributes:
        path: The Document ``path`` targeted, or None when selected by ``job_id``.
        version: The specific version requested, or None if all versions were targeted.
        job_id: The ``job_id`` selector (one value or several) as passed by the
            caller, or None when selected by ``path``.
        tenant_id: The tenant the freeze was scoped to (API value: None = public).
        created_by_user_id: The ``created_by_user_id`` filter as passed, or None.
        found: True if at least one Document matched. When False, every counter
            is 0 and nothing was exported, stored or mutated.
        mode: ``"freeze"`` (``delete_after_export=True``: the subtree was
            reduced and the Document(s) marked ``frozen=true``) or ``"backup"``
            (``delete_after_export=False``: only the snapshot was stored and
            ``last_backup_blob_id``/``last_backup_at`` set; the graph is intact).
        frozen_blob_id: Id of the stored snapshot in the freeze backend (in both
            modes), or None when nothing was found.
        versions_frozen: Sorted versions of the Documents matched by the selector
            (IS_COMPOSED_OF* descendants not included).
        documents_frozen: Number of :Document nodes in the snapshot — frozen
            (``mode="freeze"``) or backed up (``mode="backup"``) — including the
            IS_COMPOSED_OF* descendants of the matched ones.
        structure_nodes_deleted / structure_nodes_kept: :StructureNode nodes removed /
            left in place (``keep_structure_nodes=True``).
        info_units_deleted: :InfoUnit nodes removed (always, in ``mode="freeze"``).
        model_decisions_deleted / model_decisions_kept: :ModelDecision nodes removed /
            kept (``keep_annotations=True``).
        proposed_models_deleted / proposed_fields_deleted: :ProposedModel /
            :ProposedField nodes removed (``keep_annotations=False``).
        extraction_results_deleted / extraction_results_kept: :ExtractionResult nodes
            removed / kept (``keep_extraction_results=True``).
        gc_*: Garbage-collection counters, same meaning as in DeletionResult
            (the orphans this freeze caused; always 0 with
            ``keep_extraction_results=True``, which deletes no :ExtractionResult).

    Every ``*_deleted`` / ``*_kept`` / ``gc_*`` counter is 0 in ``mode="backup"``.
    """

    path: str | None
    version: int | None
    found: bool
    mode: Literal["freeze", "backup"]
    frozen_blob_id: str | None
    versions_frozen: list[int]
    documents_frozen: int
    structure_nodes_deleted: int = 0
    structure_nodes_kept: int = 0
    info_units_deleted: int = 0
    model_decisions_deleted: int = 0
    model_decisions_kept: int = 0
    proposed_models_deleted: int = 0
    proposed_fields_deleted: int = 0
    extraction_results_deleted: int = 0
    extraction_results_kept: int = 0
    gc_entity_model_instance_deleted: int = 0
    gc_entity_model_instance_passes: int = 0
    gc_labeled_entity_deleted: int = 0
    gc_labeled_entity_passes: int = 0
    job_id: str | list[str] | tuple[str, ...] | None = None
    tenant_id: str | None = None
    created_by_user_id: str | list[str] | tuple[str, ...] | None = None


@dataclass
class RestoreResult:
    """Result of a restore_document() call — subtree rebuilt from its snapshot.

    Attributes:
        path: The Document ``path`` targeted, or None when selected by ``job_id``.
        version: The specific version requested, or None if all versions were targeted.
        job_id: The ``job_id`` selector as passed by the caller, or None.
        tenant_id: The tenant the restore was scoped to (API value: None = public).
        created_by_user_id: The ``created_by_user_id`` filter as passed, or None.
        found: True if at least one frozen Document matched. When False, every
            counter is 0 and nothing was read or written.
        versions_restored: Sorted versions of the frozen Documents matched by the
            selector (folder descendants restored with them not included).
        documents_restored: Number of :Document nodes restored (matched ones plus
            the IS_COMPOSED_OF* descendants frozen in the same snapshot), stubs
            and recreated ones alike.
        documents_recreated: How many of them no longer existed in the graph
            (deleted, or restored from a backup) and were recreated.
        frozen_blob_ids: Distinct snapshots the documents were restored from.
        snapshots_deleted: Snapshots deleted from the freeze backend because no
            other Document of the tenant still references them.
        nodes_created: Nodes created from the snapshot (recreated :Document
            nodes included).
        nodes_reused: Snapshot nodes that already existed and were reused via MERGE
            (kept :StructureNode trees, shared :ModelInstance/:Entity/:LabeledEntity,
            catalog nodes are not counted).
        relationships_created: Relationships recreated from the snapshot (those
            that already existed with the same properties are not duplicated).
        gc_*: Garbage-collection counters of a sweep of the whole tenant (same
            meaning as in OrphanCollectionResult); 0 when the restore deleted no
            stale :ExtractionResult (the GC is then skipped).
    """

    path: str | None
    version: int | None
    found: bool
    versions_restored: list[int]
    documents_restored: int
    documents_recreated: int = 0
    frozen_blob_ids: list[str] = field(default_factory=list)
    snapshots_deleted: int = 0
    nodes_created: int = 0
    nodes_reused: int = 0
    relationships_created: int = 0
    gc_entity_model_instance_deleted: int = 0
    gc_entity_model_instance_passes: int = 0
    gc_labeled_entity_deleted: int = 0
    gc_labeled_entity_passes: int = 0
    job_id: str | list[str] | tuple[str, ...] | None = None
    tenant_id: str | None = None
    created_by_user_id: str | list[str] | tuple[str, ...] | None = None
