"""
graph/sql/store.py — :class:`SqlGraphStore` (Postgres and SQLite).

The same nine operation groups as :class:`~scinr.newton.graph.neo4j.store.Neo4jGraphStore`,
translated to SQLAlchemy Core over the schema of ``sql/migrations``. The engines are the
process-wide ones of :mod:`scinr.newton.sql.engine`.
"""

from __future__ import annotations

import contextlib
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
from scinr.newton.graph.sql.annotation import SqlAnnotationOps
from scinr.newton.graph.sql.deletion import SqlDeletionOps
from scinr.newton.graph.sql.documents import SqlDocumentOps
from scinr.newton.graph.sql.extraction import SqlExtractionOps
from scinr.newton.graph.sql.freeze import SqlFreezeOps
from scinr.newton.graph.sql.gc import SqlGcOps
from scinr.newton.graph.sql.restore import SqlRestoreOps
from scinr.newton.graph.sql.schema import SqlSchemaOps
from scinr.newton.graph.sql.tabular import SqlTabularOps
from scinr.newton.sql import engine as _engine
from scinr.newton.sql.errors import async_sql_errors, sql_errors

__all__ = ["SqlGraphStore"]


class SqlGraphStore(GraphStore):
    """:class:`GraphStore` over SQL; replicates the behaviour of the Neo4j store."""

    def __init__(self) -> None:
        self._schema = SqlSchemaOps()
        self._documents = SqlDocumentOps()
        self._annotation = SqlAnnotationOps()
        self._extraction = SqlExtractionOps()
        self._tabular = SqlTabularOps()
        self._deletion = SqlDeletionOps()
        self._gc = SqlGcOps()
        self._freeze = SqlFreezeOps()
        self._restore = SqlRestoreOps()

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
        """A no-op scope: the SQLAlchemy ``Engine`` is already a thread-safe pool, so the
        synchronous methods share it without a connection of their own."""
        return contextlib.nullcontext()

    def close(self) -> None:
        """Dispose the pool of the synchronous engine of the active configuration, if any."""
        with sql_errors():
            _engine.dispose_sync_engine()

    async def aclose(self) -> None:
        """Dispose both engines; the next call creates new ones."""
        async with async_sql_errors():
            await _engine.aclose_engines()
