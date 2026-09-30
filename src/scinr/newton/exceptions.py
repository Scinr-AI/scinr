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


class FreezeError(ScinrError):
    """
    Raised when a document cannot be frozen, restored or snapshotted.

    Examples: freezing a document that is already frozen, restoring a document
    that is not frozen, a snapshot missing from the freeze backend, or a
    snapshot whose tenant does not match the document it is restored into.
    """


class ConversionError(ScinrError):
    """
    Raised when a file converter fails to process a source file.

    This is the canonical ConversionError. converters/base.py re-exports it
    via multiple inheritance (ConverterError, ScinrError) for backward compatibility.
    """


class NavigationError(ScinrError):
    """
    Raised for invalid graph-navigation input or misuse.

    Examples: a malformed identifier or property key, a malformed ``where=``
    selector, a write statement passed to ``execute_raw``, a ``dialect=``
    mismatch, or a ``strict=True`` single-get that found nothing.
    """


class GraphConnectionError(NavigationError):
    """
    Raised when the configured graph engine is unreachable.

    Surfaced by :func:`scinr.newton.navigation.get_graph_navigator` and by
    ``GraphNavigator.ping()``. Mirrors :class:`StorageError` for the storage
    layer.
    """


class UnsupportedOperationError(NavigationError):
    """
    Raised when an optional navigator capability is not implemented by the
    active backend.

    The canonical case is calling ``execute_raw`` / ``execute_raw_one`` on a
    backend that exposes no raw-query path.
    """


class ScopeError(NavigationError, StorageError):
    """
    Raised for an invalid read scope (``tenant_id`` / ``include_public`` /
    ``created_by_user_id`` / ``job_id``): an empty ``tenant_id`` or an empty
    list filter.

    The same scope is accepted by the graph navigation API and by the storage
    repositories, so it is both a :class:`NavigationError` and a
    :class:`StorageError` — callers of either layer can keep catching their
    own error type.
    """
