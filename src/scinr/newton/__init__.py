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
    IngestionError,
    ModelError,
    PreconditionError,
    ScinrError,
    StorageError,
)
from scinr.newton.ingest.deletion import delete_document
from scinr.newton.pipeline import run_pipeline
from scinr.newton.results import (
    DeletionResult,
    DocumentResult,
    PipelineResult,
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
    # Result dataclasses
    "DocumentResult",
    "StageResult",
    "PipelineResult",
    "DeletionResult",
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
    # Normalization
    "NormalizationEngine",
    # Version
    "__version__",
]
