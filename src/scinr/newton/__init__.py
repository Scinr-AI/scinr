"""
scinr-ingest — Document ingestion and entity extraction pipeline for Neo4j.
"""
from scinr.newton.config import (
    ThemePath,
    configure,
    get_available_themes,
    get_config,
)
from scinr.newton.exceptions import (
    ConfigurationError,
    ConversionError,
    ExtractionError,
    FreezeError,
    GraphConnectionError,
    IngestionError,
    ModelError,
    NavigationError,
    PreconditionError,
    ScinrError,
    StorageError,
    UnsupportedOperationError,
)
from scinr.newton.ingest.deletion import collect_orphans, delete_document
from scinr.newton.ingest.freeze import export_document_snapshot, freeze_document
from scinr.newton.ingest.restore import restore_document
from scinr.newton.navigation import (
    GraphNavigator,
    get_graph_navigator,
    graph_navigator,
)
from scinr.newton.pipeline import run_pipeline
from scinr.newton.results import (
    DeletionResult,
    DocumentResult,
    FreezeResult,
    OrphanCollectionResult,
    PipelineResult,
    RestoreResult,
    StageResult,
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

__version__ = "0.2.0"

__all__ = [
    # Configuration
    "configure",
    "get_config",
    "get_available_themes",
    "ThemePath",
    # Exceptions
    "ScinrError",
    "ConfigurationError",
    "PreconditionError",
    "ExtractionError",
    "IngestionError",
    "ModelError",
    "StorageError",
    "ConversionError",
    "NavigationError",
    "GraphConnectionError",
    "UnsupportedOperationError",
    "FreezeError",
    # Result dataclasses
    "DocumentResult",
    "StageResult",
    "PipelineResult",
    "DeletionResult",
    "OrphanCollectionResult",
    "FreezeResult",
    "RestoreResult",
    # Unified pipeline
    "run_pipeline",
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
