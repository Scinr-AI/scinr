"""
tests/unit/graph_neo4j/test_store.py — Skeleton of the graph store: registry
(``get/set/reset_graph_store``), ``configure()`` reset, driver credentials error
and the synchronous scope of ``Neo4jGraphStore``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from scinr.newton import configure
from scinr.newton.exceptions import (
    ConfigurationError,
    GraphConfigurationError,
    GraphConnectionError,
)
from scinr.newton.graph import get_graph_store, reset_graph_store, set_graph_store
from scinr.newton.graph.base import GraphStore
from scinr.newton.graph.neo4j import driver as driver_mod
from scinr.newton.graph.neo4j.store import Neo4jGraphStore
from tests.unit._graph_fakes import FakeGraphStore

_URI = "bolt://localhost:7687"


@pytest.fixture(autouse=True)
def _clean_store():
    reset_graph_store()
    yield
    reset_graph_store()


def _configure(**overrides):
    kwargs = {
        "neo4j_uri": _URI,
        "neo4j_user": "neo4j",
        "neo4j_password": "pw",
    }
    kwargs.update(overrides)
    configure(**kwargs)


class TestRegistry:
    def test_get_returns_singleton_neo4j_store(self):
        _configure()
        store = get_graph_store()
        assert isinstance(store, Neo4jGraphStore)
        assert get_graph_store() is store

    def test_get_opens_no_connection(self, monkeypatch):
        _configure()
        monkeypatch.setattr(driver_mod, "get_driver", MagicMock(side_effect=AssertionError))
        get_graph_store()

    def test_set_replaces_and_reset_discards(self):
        _configure()
        fake = FakeGraphStore()
        set_graph_store(fake)
        assert get_graph_store() is fake
        reset_graph_store()
        assert isinstance(get_graph_store(), Neo4jGraphStore)

    def test_configure_resets_the_store(self):
        _configure()
        fake = FakeGraphStore()
        set_graph_store(fake)
        _configure()
        assert get_graph_store() is not fake

    def test_unknown_backend_is_a_configuration_error(self, monkeypatch):
        _configure()
        import scinr.newton.config as cfg_module

        monkeypatch.setattr(cfg_module.get_config(), "graph_backend", "nope")
        with pytest.raises(ConfigurationError, match="graph_backend"):
            get_graph_store()

    def test_fixture_installs_and_discards_the_fake(self, fake_graph_store):
        assert get_graph_store() is fake_graph_store

    def test_fake_is_a_graph_store_with_autospecced_groups(self):
        fake = FakeGraphStore()
        assert isinstance(fake, GraphStore)
        fake.documents.next_version("a/b", "acme")
        fake.documents.next_version.assert_called_once_with("a/b", "acme")
        with fake.sync_scope():
            pass
        fake.sync_scope_calls.assert_called_once()
        with pytest.raises(AttributeError):
            fake.documents.not_a_method  # noqa: B018

    async def test_fake_async_methods_are_awaitable(self):
        fake = FakeGraphStore()
        await fake.annotation.write_annotation("n", object(), "doc")
        fake.annotation.write_annotation.assert_awaited_once()


class TestSkeleton:
    @pytest.mark.parametrize(
        "group",
        ["schema", "documents", "annotation", "extraction", "tabular", "deletion", "gc", "freeze", "restore"],
    )
    def test_no_group_is_left_unimplemented(self, group):
        assert getattr(Neo4jGraphStore(), group) is not None

    def test_every_implemented_group_is_wired(self):
        from scinr.newton.graph.neo4j.annotation import Neo4jAnnotationOps
        from scinr.newton.graph.neo4j.deletion import Neo4jDeletionOps
        from scinr.newton.graph.neo4j.documents import Neo4jDocumentOps
        from scinr.newton.graph.neo4j.extraction import Neo4jExtractionOps
        from scinr.newton.graph.neo4j.freeze import Neo4jFreezeOps
        from scinr.newton.graph.neo4j.gc import Neo4jGcOps
        from scinr.newton.graph.neo4j.restore import Neo4jRestoreOps
        from scinr.newton.graph.neo4j.schema import Neo4jSchemaOps
        from scinr.newton.graph.neo4j.tabular import Neo4jTabularOps

        store = Neo4jGraphStore()
        assert isinstance(store.schema, Neo4jSchemaOps)
        assert isinstance(store.documents, Neo4jDocumentOps)
        assert isinstance(store.annotation, Neo4jAnnotationOps)
        assert isinstance(store.extraction, Neo4jExtractionOps)
        assert isinstance(store.tabular, Neo4jTabularOps)
        assert isinstance(store.deletion, Neo4jDeletionOps)
        assert isinstance(store.gc, Neo4jGcOps)
        assert isinstance(store.freeze, Neo4jFreezeOps)
        assert isinstance(store.restore, Neo4jRestoreOps)

    def test_close_is_a_no_op(self):
        Neo4jGraphStore().close()

    async def test_aclose_closes_the_async_singleton(self, monkeypatch):
        _configure()
        fake_async = MagicMock()

        async def _close():
            fake_async.closed = True

        fake_async.close = _close
        monkeypatch.setattr(driver_mod, "_async_driver", fake_async)
        await Neo4jGraphStore().aclose()
        assert fake_async.closed is True
        assert driver_mod._async_driver is None


class TestDriverCredentials:
    def test_missing_credentials_raise_graph_configuration_error(self, monkeypatch):
        _configure()
        import scinr.newton.config as cfg_module

        monkeypatch.setattr(cfg_module.get_config(), "neo4j_password", "")
        for factory in (driver_mod.get_driver, driver_mod.get_async_driver):
            with pytest.raises(GraphConfigurationError) as info:
                factory()
            assert not isinstance(info.value, ValueError)
            assert "Neo4j credentials not configured" in str(info.value)


class _Driver:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def created_drivers(monkeypatch):
    created: list[_Driver] = []

    def _get_driver():
        drv = _Driver()
        created.append(drv)
        return drv

    monkeypatch.setattr(driver_mod, "get_driver", _get_driver)
    return created


class TestSyncScope:
    def test_scope_creates_one_driver_and_closes_it(self, created_drivers):
        from scinr.newton.graph.neo4j._scope import borrow_sync_driver

        store = Neo4jGraphStore()
        with store.sync_scope():
            with borrow_sync_driver() as first, borrow_sync_driver() as second:
                assert first is second
            assert created_drivers[0].closed == 0
        assert len(created_drivers) == 1
        assert created_drivers[0].closed == 1

    def test_nested_scopes_reuse_the_outer_driver(self, created_drivers):
        store = Neo4jGraphStore()
        with store.sync_scope():
            with store.sync_scope():
                pass
            assert created_drivers[0].closed == 0
        assert len(created_drivers) == 1
        assert created_drivers[0].closed == 1

    def test_without_scope_each_borrow_gets_and_closes_its_own(self, created_drivers):
        from scinr.newton.graph.neo4j._scope import borrow_sync_driver

        with borrow_sync_driver() as one:
            pass
        with borrow_sync_driver() as two:
            pass
        assert one is not two
        assert [d.closed for d in created_drivers] == [1, 1]

    def test_driver_is_closed_when_the_body_raises(self, created_drivers):
        store = Neo4jGraphStore()
        with pytest.raises(RuntimeError, match="boom"):
            with store.sync_scope():
                raise RuntimeError("boom")
        assert created_drivers[0].closed == 1

    async def test_threads_started_inside_the_scope_see_the_driver(self, created_drivers):
        from scinr.newton.graph.neo4j._scope import borrow_sync_driver

        def _in_thread():
            with borrow_sync_driver() as drv:
                return drv

        store = Neo4jGraphStore()
        with store.sync_scope():
            seen = await asyncio.gather(
                asyncio.to_thread(_in_thread), asyncio.to_thread(_in_thread)
            )
        assert seen[0] is seen[1] is created_drivers[0]
        assert len(created_drivers) == 1

    async def test_concurrent_runs_do_not_share_a_scope(self, created_drivers):
        from scinr.newton.graph.neo4j._scope import borrow_sync_driver

        store = Neo4jGraphStore()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _holder():
            with store.sync_scope():
                entered.set()
                await release.wait()

        async def _outsider():
            await entered.wait()
            with borrow_sync_driver() as drv:
                return drv

        task = asyncio.create_task(_holder())
        outsider = await _outsider()
        release.set()
        await task
        assert outsider is not created_drivers[0]
        assert len(created_drivers) == 2

    def test_credentials_error_surfaces_from_the_scope(self, monkeypatch):
        _configure()
        import scinr.newton.config as cfg_module

        monkeypatch.setattr(cfg_module.get_config(), "neo4j_user", "")
        with pytest.raises(GraphConfigurationError):
            with Neo4jGraphStore().sync_scope():
                pass

    def test_engine_error_while_closing_is_translated(self, monkeypatch):
        from neo4j.exceptions import ServiceUnavailable

        class _BadDriver:
            def close(self):
                raise ServiceUnavailable("lost")

        monkeypatch.setattr(driver_mod, "get_driver", lambda: _BadDriver())
        with pytest.raises(GraphConnectionError):
            with Neo4jGraphStore().sync_scope():
                pass
