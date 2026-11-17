"""
graph/neo4j/store.py — :class:`Neo4jGraphStore`.

It carries the synchronous scope (``sync_scope()``), the lifecycle
(``close`` / ``aclose``) and the operation groups: ``schema``, ``documents``, ``annotation``, ``extraction``,
``tabular``, ``deletion``, ``gc``, ``freeze`` and ``restore``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from scinr.newton.graph.base import (
    AnnotationOps,
    DeletionOps,
    DocumentOps,
    ExtractionOps,
    FreezeOps,
    GcOps,
    GraphStore,
    RestoreOps,
    SchemaOps,
    TabularOps,
)
from scinr.newton.graph.neo4j import driver as _driver
from scinr.newton.graph.neo4j._scope import open_scope
from scinr.newton.graph.neo4j.annotation import Neo4jAnnotationOps
from scinr.newton.graph.neo4j.deletion import Neo4jDeletionOps
from scinr.newton.graph.neo4j.documents import Neo4jDocumentOps
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.extraction import Neo4jExtractionOps
from scinr.newton.graph.neo4j.freeze import Neo4jFreezeOps
from scinr.newton.graph.neo4j.gc import Neo4jGcOps
from scinr.newton.graph.neo4j.restore import Neo4jRestoreOps
from scinr.newton.graph.neo4j.schema import Neo4jSchemaOps
from scinr.newton.graph.neo4j.tabular import Neo4jTabularOps


class _SyncScope:
    """Context manager of ``sync_scope()``; translates driver errors of open/close only."""

    def __init__(self) -> None:
        self._scope = open_scope()

    def __enter__(self) -> None:
        with neo4j_errors():
            self._scope.__enter__()

    def __exit__(self, exc_type, exc, tb) -> bool | None:  # noqa: ANN001
        # The body's exceptions are the callers' business: they were translated
        # (or not) by the methods that raised them. Only closing the driver can
        # raise an engine error of its own.
        with neo4j_errors():
            return self._scope.__exit__(exc_type, exc, tb)


class Neo4jGraphStore(GraphStore):
    """:class:`GraphStore` over Neo4j; runs the same Cypher the pipeline always has."""

    def __init__(self) -> None:
        self._schema = Neo4jSchemaOps()
        self._documents = Neo4jDocumentOps()
        self._annotation = Neo4jAnnotationOps()
        self._extraction = Neo4jExtractionOps()
        self._tabular = Neo4jTabularOps()
        self._deletion = Neo4jDeletionOps()
        self._gc = Neo4jGcOps()
        self._freeze = Neo4jFreezeOps()
        self._restore = Neo4jRestoreOps()

    @property
    def schema(self) -> SchemaOps:
        return self._schema

    @property
    def documents(self) -> DocumentOps:
        return self._documents

    @property
    def annotation(self) -> AnnotationOps:
        return self._annotation

    @property
    def extraction(self) -> ExtractionOps:
        return self._extraction

    @property
    def tabular(self) -> TabularOps:
        return self._tabular

    @property
    def deletion(self) -> DeletionOps:
        return self._deletion

    @property
    def gc(self) -> GcOps:
        return self._gc

    @property
    def freeze(self) -> FreezeOps:
        return self._freeze

    @property
    def restore(self) -> RestoreOps:
        return self._restore

    def sync_scope(self) -> AbstractContextManager[None]:
        """One synchronous driver for everything run inside the block (see :class:`GraphStore`).

        Raises:
            GraphConfigurationError: If the credentials are not configured.
            GraphConnectionError: If closing the driver fails.
        """
        return _SyncScope()

    def close(self) -> None:
        """Nothing to release: each ``sync_scope()`` closes its own driver."""

    async def aclose(self) -> None:
        """Close the shared async driver; the next async call creates a new one."""
        with neo4j_errors():
            await _driver.close_async_driver()
