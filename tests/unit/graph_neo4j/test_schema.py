"""
tests/unit/graph_neo4j/test_schema.py — ``ensure_indexes`` and ``Neo4jSchemaOps``
(``store.schema``) against a fake driver.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scinr.newton.graph import reset_graph_store, set_graph_store
from scinr.newton.graph.neo4j import driver as driver_mod
from scinr.newton.graph.neo4j import schema as schema_mod
from scinr.newton.graph.neo4j.store import Neo4jGraphStore
from scinr.newton.ingest import restore


class TestEnsureIndexes:
    class _Session:
        def __init__(self, outer) -> None:
            self.outer = outer

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute_write(self, fn):
            return fn(self)

        def run(self, query, **params):
            outer = self.outer
            if outer.error is not None:
                raise outer.error
            outer.queries.append((query, params))
            added = 1 if query.startswith("CREATE INDEX") and outer.missing else 0
            return SimpleNamespace(
                consume=lambda: SimpleNamespace(counters=SimpleNamespace(indexes_added=added))
            )

    class _Driver:
        def __init__(self, missing: bool, error: Exception | None = None) -> None:
            self.missing = missing
            self.error = error
            self.queries: list[tuple[str, dict]] = []

        def session(self, **kwargs):
            return TestEnsureIndexes._Session(self)

    def test_existing_indexes_do_not_wait(self):
        from scinr.newton.graph.neo4j.schema import ensure_indexes

        driver = self._Driver(missing=False)
        assert ensure_indexes(driver, restore._RESTORE_INDEXES, database="neo4j") == 0
        assert len(driver.queries) == len(restore._RESTORE_INDEXES)
        assert all(q.startswith("CREATE INDEX") and "IF NOT EXISTS" in q for q, _ in driver.queries)

    def test_created_indexes_are_awaited(self):
        from scinr.newton.graph.neo4j.schema import ensure_indexes

        driver = self._Driver(missing=True)
        created = ensure_indexes(driver, ("idx_model_decision_uid",), database="neo4j")
        assert created == 1
        assert driver.queries[-1] == ("CALL db.awaitIndexes($timeout)", {"timeout": 300})

    def test_failure_is_logged_not_raised(self, caplog):
        from scinr.newton.graph.neo4j.schema import ensure_indexes

        driver = self._Driver(missing=True, error=RuntimeError("no schema privilege"))
        assert ensure_indexes(driver, ("idx_model_decision_uid",), database="neo4j") == 0
        assert "Could not ensure indexes" in caplog.text

    def test_unknown_name_is_rejected(self):
        from scinr.newton.graph.neo4j.schema import ensure_indexes

        with pytest.raises(ValueError, match="Unknown index"):
            ensure_indexes(self._Driver(missing=False), ("nope",), database="neo4j")


class TestSchemaOps:
    @pytest.fixture(autouse=True)
    def _store(self, monkeypatch):
        monkeypatch.setattr(
            schema_mod, "get_config", lambda: SimpleNamespace(neo4j_database="graphdb")
        )
        set_graph_store(Neo4jGraphStore())
        yield
        reset_graph_store()

    def test_ensure_indexes_uses_the_scope_driver_and_the_configured_database(self, monkeypatch):
        databases: list[str] = []
        driver = TestEnsureIndexes._Driver(missing=False)
        original_session = driver.session
        driver.session = lambda **kw: (databases.append(kw["database"]), original_session())[1]
        driver.close = lambda: None
        opened: list[int] = []
        monkeypatch.setattr(driver_mod, "get_driver", lambda: (opened.append(1), driver)[1])

        store = Neo4jGraphStore()
        with store.sync_scope():
            assert store.schema.ensure_indexes(("idx_model_decision_uid",)) == 0
            assert store.schema.ensure_indexes(("idx_catalog_model_name",)) == 0

        assert opened == [1]  # one driver for the whole scope
        assert set(databases) == {"graphdb"}

    def test_ensure_indexes_outside_a_scope_opens_and_closes_its_own_driver(self, monkeypatch):
        closed: list[int] = []
        driver = TestEnsureIndexes._Driver(missing=False)
        driver.close = lambda: closed.append(1)
        monkeypatch.setattr(driver_mod, "get_driver", lambda: driver)

        Neo4jGraphStore().schema.ensure_indexes(restore._RESTORE_INDEXES)

        assert closed == [1]
        assert len(driver.queries) == len(restore._RESTORE_INDEXES)

    def test_ensure_indexes_accepts_any_iterable_of_names(self, monkeypatch):
        driver = TestEnsureIndexes._Driver(missing=False)
        driver.close = lambda: None
        monkeypatch.setattr(driver_mod, "get_driver", lambda: driver)

        Neo4jGraphStore().schema.ensure_indexes(iter(["idx_model_decision_uid"]))

        assert len(driver.queries) == 1
