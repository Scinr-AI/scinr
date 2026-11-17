"""
tests/unit/_graph_fakes.py — Test double for the graph-store interface.

``FakeGraphStore`` is a real :class:`GraphStore` whose sub-interfaces are
``create_autospec(<Sub>Ops, instance=True)``: the methods the interface declares
exist (a typo in a test fails), asynchronous ones are ``AsyncMock`` and
everything returns a ``MagicMock`` unless the test configures it. Its
``sync_scope()`` is a ``nullcontext``.

Inject it with the ``fake_graph_store`` fixture (``tests/conftest.py``), which
calls ``reset_graph_store()`` when the test ends.

Only the groups the interface already declares are faked; groups added by later
work packages appear here automatically because the properties are built from
the abstract ``GraphStore`` ones.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from unittest.mock import MagicMock, create_autospec

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


class FakeGraphStore(GraphStore):
    """A :class:`GraphStore` made of autospecced sub-interfaces."""

    def __init__(self) -> None:
        self._schema = create_autospec(SchemaOps, instance=True)
        self._documents = create_autospec(DocumentOps, instance=True)
        self._annotation = create_autospec(AnnotationOps, instance=True)
        self._extraction = create_autospec(ExtractionOps, instance=True)
        self._tabular = create_autospec(TabularOps, instance=True)
        self._deletion = create_autospec(DeletionOps, instance=True)
        self._gc = create_autospec(GcOps, instance=True)
        self._freeze = create_autospec(FreezeOps, instance=True)
        self._restore = create_autospec(RestoreOps, instance=True)
        self.sync_scope_calls = MagicMock(name="sync_scope")
        self.close_calls = MagicMock(name="close")

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
        self.sync_scope_calls()
        return nullcontext()

    def close(self) -> None:
        self.close_calls()

    async def aclose(self) -> None:
        self.close_calls()
