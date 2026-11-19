"""
scinr-ingest — Document ingestion and entity extraction pipeline for a knowledge graph
(Neo4j, Postgres or SQLite).
"""

from scinr.newton.config import (
    StageLLMConfig,
    ThemePath,
    configure,
    get_available_themes,
    get_config,
)
from scinr.newton.exceptions import (
    ConfigurationError,
    ConversionError,
    DocumentVersionConflict,
    ExtractionError,
    FreezeError,
    GraphAuthenticationError,
    GraphConfigurationError,
    GraphConnectionError,
    GraphConstraintError,
    GraphError,
    GraphQueryError,
    GraphResourceError,
    ImageStorageError,
    IngestionError,
    ModelError,
    NavigationError,
    PreconditionError,
    ScinrError,
    StorageError,
    TenantAccessError,
    TransientGraphError,
    UnsupportedOperationError,
)
from scinr.newton.ingest.deletion import collect_orphans, delete_document
from scinr.newton.ingest.freeze import export_document_snapshot, freeze_document
from scinr.newton.ingest.restore import restore_document
from scinr.newton.media.transcriber import Transcriber, TranscriptSegment
from scinr.newton.models.transcript import TranscriptCue
from scinr.newton.navigation import (
    GraphNavigator,
    get_graph_navigator,
    graph_navigator,
)
from scinr.newton.pipeline import discover_documents, run_pipeline
from scinr.newton.progress import (
    AsyncProgressCallback,
    DocumentProgress,
    ProgressCallback,
    ProgressEvent,
    ProgressSnapshot,
    ProgressTracker,
)
from scinr.newton.results import (
    DeletedDocument,
    DeletionResult,
    DiscoveredDocuments,
    DocumentResult,
    FreezeResult,
    OrphanCollectionResult,
    PipelineResult,
    RestoreResult,
    SkippedFile,
    StageResult,
)
from scinr.newton.sql.admin import (
    SchemaStatus,
    TenantAccess,
    grant_tenant_access,
    list_tenant_access,
    revoke_tenant_access,
    setup_sql_schema,
)
from scinr.newton.stages import (
    run_annotation,
    run_entity_extraction,
    run_extraction,
    run_ingestion,
    run_preprocess,
    run_tabular_pipeline,
)
from scinr.newton.tabular.normalization import NormalizationEngine

__version__ = "0.6.0"

__all__ = [
    # Configuration
    "configure",
    "get_config",
    "get_available_themes",
    "ThemePath",
    "StageLLMConfig",
    # Media (audio / video transcription)
    "Transcriber",
    "TranscriptCue",
    "TranscriptSegment",
    # Exceptions
    "ScinrError",
    "ConfigurationError",
    "PreconditionError",
    "ExtractionError",
    "IngestionError",
    "ModelError",
    "StorageError",
    "ImageStorageError",
    "ConversionError",
    "NavigationError",
    "GraphConnectionError",
    "UnsupportedOperationError",
    "FreezeError",
    "GraphConfigurationError",
    "GraphError",
    "GraphAuthenticationError",
    "TransientGraphError",
    "GraphResourceError",
    "GraphConstraintError",
    "DocumentVersionConflict",
    "GraphQueryError",
    "TenantAccessError",
    # SQL administration (PostgreSQL)
    "setup_sql_schema",
    "grant_tenant_access",
    "revoke_tenant_access",
    "list_tenant_access",
    "SchemaStatus",
    "TenantAccess",
    # Result dataclasses
    "DocumentResult",
    "StageResult",
    "PipelineResult",
    "DeletionResult",
    "DeletedDocument",
    "DiscoveredDocuments",
    "SkippedFile",
    "OrphanCollectionResult",
    "FreezeResult",
    "RestoreResult",
    # Unified pipeline
    "run_pipeline",
    "discover_documents",
    # Progress reporting
    "ProgressEvent",
    "ProgressCallback",
    "AsyncProgressCallback",
    "ProgressTracker",
    "ProgressSnapshot",
    "DocumentProgress",
    # Progress reporting
    "ProgressEvent",
    "ProgressCallback",
    "AsyncProgressCallback",
    "ProgressTracker",
    "ProgressSnapshot",
    "DocumentProgress",
    # Individual stage functions
    "run_preprocess",
    "run_extraction",
    "run_ingestion",
    "run_annotation",
    "run_entity_extraction",
    "run_tabular_pipeline",
    # Document deletion
    "delete_document",
    "collect_orphans",
    # Document freezing
    "freeze_document",
    "restore_document",
    "export_document_snapshot",
    # Graph navigation (read-only)
    "get_graph_navigator",
    "graph_navigator",
    "GraphNavigator",
    # Normalization
    "NormalizationEngine",
    # Version
    "__version__",
]
