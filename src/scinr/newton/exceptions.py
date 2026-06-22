"""
exceptions.py — scinr-ingest exception hierarchy.

All exceptions raised by the public API inherit from ScinrError,
allowing callers to catch the entire family with a single except clause.
"""


class ScinrError(Exception):
    """Base class for all scinr-ingest exceptions."""


class ConfigurationError(ScinrError):
    """
    Raised when the library is misconfigured.

    Examples: LLM not set, Neo4j credentials missing, invalid storage backend,
    models directory not found, invalid catalog.py.
    """


class PreconditionError(ScinrError):
    """
    Raised when a pipeline function is called out of order.

    Examples: run_entity_extraction called without prior run_annotation,
    document not found in Neo4j.
    """


class ExtractionError(ScinrError):
    """
    Raised when LLM extraction fails after all retries are exhausted.
    """


class IngestionError(ScinrError):
    """
    Raised when writing to Neo4j fails.

    Examples: version conflict, schema constraint violation.
    """


class ModelError(ScinrError):
    """
    Raised when a Pydantic extraction model cannot be resolved or is invalid.

    Examples: class name not found in registry, catalog.py defines a non-BaseModel class.
    """


class StorageError(ScinrError):
    """
    Raised when the storage backend (MongoDB) is unavailable or misconfigured.
    """


class ConversionError(ScinrError):
    """
    Raised when a file converter fails to process a source file.

    This is the canonical ConversionError. converters/base.py re-exports it
    via multiple inheritance (ConverterError, ScinrError) for backward compatibility.
    """
