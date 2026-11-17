"""
tests/contract/test_errors.py -- Error contract: engine failures leave as ``scinr`` exceptions.

Cases any engine can reproduce: a host nobody listens on, credentials the engine rejects,
credentials not configured, and a duplicate document version. ``GraphResourceError`` is
allowed by the contract but not every engine can produce it, so it is not tested here.
"""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import (
    ConfigurationError,
    DocumentVersionConflict,
    GraphAuthenticationError,
    GraphConfigurationError,
    GraphConnectionError,
    GraphConstraintError,
    GraphError,
    GraphQueryError,
    GraphResourceError,
    TransientGraphError,
)
from scinr.newton.graph.types import DocumentSelector

from ._helpers import make_document

SELECTOR = DocumentSelector(tenant_id="t", path="a.txt", latest=True)


class TestHierarchy:
    def test_every_graph_error_is_a_graph_error(self):
        for cls in (
            GraphConnectionError,
            GraphAuthenticationError,
            TransientGraphError,
            GraphResourceError,
            GraphConstraintError,
            DocumentVersionConflict,
            GraphQueryError,
        ):
            assert issubclass(cls, GraphError), cls
        assert issubclass(GraphAuthenticationError, GraphConnectionError)
        assert issubclass(DocumentVersionConflict, GraphConstraintError)

    def test_configuration_error_is_not_an_engine_error(self):
        assert issubclass(GraphConfigurationError, ConfigurationError)
        assert not issubclass(GraphConfigurationError, GraphError)


class TestUnreachable:
    """A store configured to a host that does not answer.

    Only methods that do not retry: the ones that do (loading, schema setup) wait out the
    whole backoff (minutes) before they give up with the same error.
    """

    async def test_sync_method(self, broken_store):
        store = broken_store("unreachable")
        with pytest.raises(GraphConnectionError) as caught:
            store.documents.next_version("a.txt", "t")
        assert isinstance(caught.value, GraphError)
        assert not isinstance(caught.value, GraphAuthenticationError)
        assert caught.value.__cause__ is not None

    async def test_async_method(self, broken_store):
        store = broken_store("unreachable")
        with pytest.raises(GraphConnectionError) as caught:
            await store.documents.count_latest_documents(SELECTOR)
        assert caught.value.__cause__ is not None


class TestBadCredentials:
    async def test_sync_method(self, broken_store):
        store = broken_store("bad_credentials")
        with pytest.raises(GraphAuthenticationError) as caught:
            store.documents.next_version("a.txt", "t")
        assert isinstance(caught.value, GraphConnectionError)
        assert isinstance(caught.value, GraphError)
        assert caught.value.__cause__ is not None

    async def test_async_method(self, broken_store):
        store = broken_store("bad_credentials")
        with pytest.raises(GraphAuthenticationError) as caught:
            await store.annotation.count_annotated_nodes(SELECTOR)
        assert caught.value.__cause__ is not None

    async def test_secret_is_not_in_the_message(self, broken_store, backend):
        store = broken_store("bad_credentials")
        with pytest.raises(GraphAuthenticationError) as caught:
            store.schema.setup()
        assert backend.broken_configs["bad_credentials"]["neo4j_password"] not in str(caught.value)


class TestNoCredentials:
    async def test_sync_and_async_methods(self, broken_store):
        store = broken_store("no_credentials")
        with pytest.raises(GraphConfigurationError) as caught:
            store.schema.setup()
        assert isinstance(caught.value, ConfigurationError)
        assert not isinstance(caught.value, GraphError)
        with pytest.raises(GraphConfigurationError):
            store.documents.next_version("a.txt", "t")
        with pytest.raises(GraphConfigurationError):
            await store.documents.count_latest_documents(SELECTOR)
        with pytest.raises(GraphConfigurationError):
            store.restore.batch_writer(1)


class TestDuplicateVersion:
    async def test_same_version_twice_conflicts_or_is_a_no_op(self, graph_store, dump):
        """The one constraint failure every engine can be made to raise on load.

        A backend that detects the duplicate raises ``DocumentVersionConflict`` (a
        ``GraphConstraintError``, never ``IngestionError``) with the engine error chained;
        one that makes the second load a no-op (Neo4j ``MERGE``) leaves the graph as it was.
        """
        doc = make_document("a.txt", tenant="t")
        graph_store.documents.insert_document_graph(doc, 1, False)
        before = dump()
        try:
            graph_store.documents.insert_document_graph(doc, 1, False)
        except DocumentVersionConflict as exc:
            assert isinstance(exc, GraphConstraintError)
            assert exc.__cause__ is not None
        assert dump() == before
