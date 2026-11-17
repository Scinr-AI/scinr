"""
tests/unit/graph_neo4j/test_deletion_ops.py — Unit tests for
scinr.newton.graph.neo4j.deletion (the batched subtree deletion shared by
delete_document() and freeze_document()), for the batched GC queries of
scinr.newton.graph.neo4j.gc and for the shared steps in graph.neo4j._shared.

No real Neo4j: a fake driver records what runs and how (auto-commit
``session.run`` vs. a managed transaction).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from neo4j.exceptions import Neo4jError

from scinr.newton.graph.neo4j import _shared
from scinr.newton.graph.neo4j import deletion as neo_deletion
from scinr.newton.graph.neo4j import gc as neo_gc
from scinr.newton.graph.types import DocumentSelector


class _Result:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    def single(self):
        return self._row


class _Tx:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver

    def run(self, query: str, **params) -> _Result:
        return self.driver.serve("tx.run", query, params)


class _Session:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params) -> _Result:
        return self.driver.serve("session.run", query, params)

    def execute_write(self, fn):
        return fn(_Tx(self.driver))


class _Driver:
    def __init__(self, row: dict | None = None, errors: list[Exception] | None = None) -> None:
        self.row = row
        self.errors = list(errors or [])
        self.calls: list[tuple[str, str, dict]] = []

    def session(self, **kwargs):
        return _Session(self)

    def serve(self, kind: str, query: str, params: dict) -> _Result:
        self.calls.append((kind, query, params))
        if self.errors:
            raise self.errors.pop(0)
        if self.row is not None:
            return _Result(self.row)
        return _Result({"n": len(params["keys"])})


def _keys(count: int) -> list[dict]:
    return [{"path": f"doc-{i}", "version": 1} for i in range(count)]


def _memory_pool_error() -> Neo4jError:
    return Neo4jError._hydrate_neo4j(
        code="Neo.TransientError.General.MemoryPoolOutOfMemoryError", message="pool full"
    )


class TestBatchedQueries:
    def test_every_delete_runs_in_bounded_transactions(self):
        deletes = [q for _, q in neo_deletion.subtree_steps() if "DETACH DELETE" in q]
        assert len(deletes) == 8  # pf, pm, cm, sf, md, er, iu, structure nodes
        for query in deletes:
            assert f"IN TRANSACTIONS OF {_shared.DELETE_BATCH_ROWS} ROWS" in query
            assert "RETURN count(*) AS n" in query

    def test_nodes_to_delete_are_known_before_the_first_batch_commits(self):
        """Grouping by the node is an aggregation: it reads the whole input
        (the HAS_CHILD walk) before a row reaches the batches, so deleting a
        StructureNode never hides its children from the query."""
        query = neo_deletion.STRUCTURE_NODES_DELETE_QUERY
        assert (
            query.index("HAS_CHILD*0..") < query.index("count(*) AS hits") < query.index("CALL (x)")
        )

    def test_batched_rows_never_carry_a_collected_list(self):
        # collect() + UNWIND charges every batched row for the whole list:
        # that is what overran the transaction memory pool.
        for _, query in neo_deletion.subtree_steps():
            if "DETACH DELETE" in query:
                assert "collect(" not in query

    def test_default_steps_delete_the_whole_subtree_children_first(self):
        steps = neo_deletion.subtree_steps()
        assert [counter for counter, _ in steps] == [
            "proposed_fields_deleted",
            "proposed_models_deleted",
            None,  # ComplementaryMatch
            None,  # SupplementaryField
            "model_decisions_deleted",
            "extraction_results_deleted",
            "info_units_deleted",
            "structure_nodes_deleted",
        ]
        assert all("DETACH DELETE" in query for _, query in steps)

    @pytest.mark.parametrize(
        "query", [neo_gc.GC_ENTITY_MODEL_INSTANCE_QUERY, neo_gc.GC_LABELED_ENTITY_QUERY]
    )
    def test_gc_deletes_in_bounded_transactions(self, query):
        assert f"IN TRANSACTIONS OF {_shared.DELETE_BATCH_ROWS} ROWS" in query
        assert "mi.tenant_id = $tenant_id" in query
        assert "RETURN count(*) AS borrados" in query


class TestRunners:
    def test_count_query_is_auto_commit(self):
        """CALL ... IN TRANSACTIONS is refused inside an explicit transaction."""
        driver = _Driver(row={"n": 7})
        assert _shared.run_count_query(driver, "neo4j", "Q", tenant_id="acme") == 7
        assert driver.calls == [("session.run", "Q", {"tenant_id": "acme"})]

    def test_count_query_without_a_row_counts_zero(self):
        driver = _Driver(row=None)
        driver.serve = lambda kind, query, params: _Result(None)
        assert _shared.run_count_query(driver, "neo4j", "Q") == 0

    def test_count_query_retries_a_full_memory_pool(self):
        driver = _Driver(row={"n": 3}, errors=[_memory_pool_error(), _memory_pool_error()])
        with patch("scinr.newton.graph.neo4j.retry.time.sleep") as sleep:
            assert _shared.run_count_query(driver, "neo4j", "Q") == 3
        assert len(driver.calls) == 3
        assert sleep.call_count == 2

    def test_steps_run_per_chunk_of_documents_and_sum(self):
        driver = _Driver()
        keys = _keys(_shared.DOCUMENTS_PER_QUERY * 2 + 3)
        steps = neo_deletion.subtree_steps()

        counters = neo_deletion.run_subtree_steps(driver, "neo4j", "acme", keys, steps, caller="t")

        sizes = [len(params["keys"]) for _, _, params in driver.calls]
        per_chunk = [_shared.DOCUMENTS_PER_QUERY, _shared.DOCUMENTS_PER_QUERY, 3]
        assert sizes == [size for size in per_chunk for _ in steps]
        # A chunk finishes all its steps before the next one starts.
        assert [q for _, q, _ in driver.calls[: len(steps)]] == [q for _, q in steps]
        assert all(params["tenant_id"] == "acme" for _, _, params in driver.calls)
        assert counters["structure_nodes_deleted"] == len(keys)
        assert counters["structure_nodes_kept"] == 0

    def test_the_orphan_collector_brackets_every_chunk(self):
        """What a chunk's ExtractionResults point at must be read before they
        are deleted, and may only be collected once they are gone."""
        driver = _Driver()
        keys = _keys(_shared.DOCUMENTS_PER_QUERY + 1)
        steps = neo_deletion.subtree_steps()
        events: list[tuple] = []

        class _Orphans:
            def gather(self, chunk):
                events.append(("gather", len(chunk), len(driver.calls)))

            def chunk_done(self):
                events.append(("done", len(driver.calls)))

        neo_deletion.run_subtree_steps(
            driver, "neo4j", "acme", keys, steps, caller="t", orphans=_Orphans()
        )

        assert events == [
            ("gather", _shared.DOCUMENTS_PER_QUERY, 0),
            ("done", len(steps)),
            ("gather", 1, len(steps)),
            ("done", 2 * len(steps)),
        ]

    def test_a_failing_step_stops_the_run(self):
        driver = _Driver(errors=[ValueError("boom")])
        with pytest.raises(ValueError, match="boom"):
            neo_deletion.run_subtree_steps(
                driver, "neo4j", "acme", _keys(3), neo_deletion.subtree_steps(), caller="t"
            )
        assert len(driver.calls) == 1

    def test_documents_are_deleted_in_order_in_managed_transactions(self):
        driver = _Driver()
        keys = _keys(_shared.DELETE_BATCH_ROWS + 2)

        assert neo_deletion.delete_documents(driver, "neo4j", "acme", keys) == len(keys)

        assert [kind for kind, _, _ in driver.calls] == ["tx.run", "tx.run"]
        assert all(q == neo_deletion.DOCUMENTS_DELETE_QUERY for _, q, _ in driver.calls)
        assert [k for _, _, params in driver.calls for k in params["keys"]] == keys


class TestDocMatch:
    """Unit tests for doc_match() — the dynamic WHERE builder (selector to Cypher)."""

    def test_only_supplied_filters_become_conditions(self):
        match, params = neo_deletion.doc_match(
            DocumentSelector(tenant_id="__public__", job_id=["j1"])
        )
        assert params == {"tenant_id": "__public__", "job_id": ["j1"]}
        assert (
            match.strip()
            == "MATCH (d:Document)\nWHERE d.tenant_id = $tenant_id AND d.job_id IN $job_id"
        )

    def test_multiple_filters_are_anded_in_fixed_order(self):
        match, params = neo_deletion.doc_match(
            DocumentSelector(tenant_id="t", path="p", version=2, created_by_user_id=["u", "v"])
        )
        assert params == {
            "tenant_id": "t",
            "path": "p",
            "version": 2,
            "created_by_user_id": ["u", "v"],
        }
        assert (
            "WHERE d.tenant_id = $tenant_id AND d.path = $path AND d.version = $version"
            " AND d.created_by_user_id IN $created_by_user_id" in match
        )

    def test_tenant_is_mandatory(self):
        with pytest.raises(ValueError):
            neo_deletion.doc_match(DocumentSelector(path="p", tenant_id=None))
        with pytest.raises(ValueError):
            neo_deletion.doc_match(DocumentSelector(path="p"))

    def test_no_is_null_disjunction_is_ever_emitted(self):
        match, _ = neo_deletion.doc_match(DocumentSelector(path="p", tenant_id="t"))
        assert "IS NULL" not in match
        assert " OR " not in match

    def test_falsy_but_non_none_values_are_kept(self):
        match, params = neo_deletion.doc_match(DocumentSelector(path="", version=0, tenant_id="t"))
        assert params == {"tenant_id": "t", "path": "", "version": 0}
        assert "d.path = $path" in match
        assert "d.version = $version" in match


class _GcResult(list):
    def single(self):
        return self[0] if self else None


class _GcSession:
    def __init__(self, driver: _GcDriver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):
        self.driver.calls.append((query, params))
        return _GcResult(self.driver.serve(query, params))

    def execute_write(self, fn):
        return fn(self)


class _GcDriver:
    """Serves the rows ``serve(query, params)`` returns; counts how often it is closed."""

    def __init__(self, serve) -> None:
        self.serve = serve
        self.calls: list[tuple[str, dict]] = []
        self.closed = 0

    def session(self, **kwargs):
        return _GcSession(self)

    def close(self):
        self.closed += 1


@pytest.fixture
def gc_store(monkeypatch):
    """A ``Neo4jGraphStore`` whose ``get_driver`` hands out the driver the test installs."""
    from types import SimpleNamespace

    from scinr.newton.graph.neo4j import driver as driver_mod
    from scinr.newton.graph.neo4j.store import Neo4jGraphStore

    cfg = SimpleNamespace(neo4j_database="neo4j")
    monkeypatch.setattr(neo_gc, "get_config", lambda: cfg)
    monkeypatch.setattr(neo_deletion, "get_config", lambda: cfg)
    holder: dict[str, _GcDriver] = {}
    created: list[_GcDriver] = []

    def _get_driver():
        created.append(holder["driver"])
        return holder["driver"]

    monkeypatch.setattr(driver_mod, "get_driver", _get_driver)

    def _make(serve) -> tuple[Neo4jGraphStore, _GcDriver, list[_GcDriver]]:
        holder["driver"] = _GcDriver(serve)
        return Neo4jGraphStore(), holder["driver"], created

    return _make


class TestGcOps:
    def test_collector_uses_the_driver_of_the_sync_scope(self, gc_store):
        store, driver, created = gc_store(
            lambda query, params: (
                [{"id": "4:db:1", "labeled": False}] if query == neo_gc.GC_SEEDS_QUERY else []
            )
        )
        collector = store.gc.collector("acme", "t")
        with store.sync_scope():
            collector.gather(_keys(2))
            collector.gather(_keys(1))
        assert len(created) == 1
        assert driver.closed == 1
        assert [query for query, _ in driver.calls] == [neo_gc.GC_SEEDS_QUERY] * 2
        assert driver.calls[0][1] == {"tenant_id": "acme", "keys": _keys(2)}

    def test_collector_outside_a_scope_borrows_a_private_driver_per_call(self, gc_store):
        store, driver, created = gc_store(lambda query, params: [])
        collector = store.gc.collector("acme", "t")
        collector.gather(_keys(1))
        collector.gather(_keys(1))
        assert len(created) == 2
        assert driver.closed == 2

    def test_collector_deletes_orphans_and_follows_their_dependents(self, gc_store):
        def serve(query, params):
            if query == neo_gc.GC_SEEDS_QUERY:
                return [{"id": "a", "labeled": False}]
            if query == neo_gc.GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY:
                return [
                    {"id": i, "dependents": [["b", True]] if i == "a" else []}
                    for i in params["ids"]
                ]
            if query == neo_gc.GC_LABELED_ENTITY_ORPHANS_QUERY:
                return [{"id": i, "dependents": []} for i in params["ids"]]
            return [{"n": len(params["ids"])}]

        store, driver, _ = gc_store(serve)
        collector = store.gc.collector("acme", "t")
        with store.sync_scope():
            collector.gather(_keys(1))
            collector.collect()
        assert collector.counts() == {
            "gc_entity_model_instance_deleted": 1,
            "gc_entity_model_instance_passes": 1,
            "gc_labeled_entity_deleted": 1,
            "gc_labeled_entity_passes": 1,
        }
        assert driver.calls[-1][0] == neo_gc.GC_LABELED_ENTITY_DELETE_QUERY

    def test_run_pass_stops_at_the_first_empty_iteration(self, gc_store):
        results = iter([3, 2, 0, 9])
        store, driver, _ = gc_store(lambda query, params: [{"borrados": next(results)}])
        assert store.gc.run_pass("labeled_entity", "acme", "t") == (5, 3)
        assert {query for query, _ in driver.calls} == {neo_gc.GC_LABELED_ENTITY_QUERY}

    def test_run_pass_is_bounded_by_gc_max_passes(self, gc_store):
        store, driver, _ = gc_store(lambda query, params: [{"borrados": 1}])
        assert store.gc.run_pass("entity_model_instance", "acme", "t") == (
            neo_gc.GC_MAX_PASSES,
            neo_gc.GC_MAX_PASSES,
        )
        assert {query for query, _ in driver.calls} == {neo_gc.GC_ENTITY_MODEL_INSTANCE_QUERY}

    def test_sweep_tenant_runs_both_passes_in_order(self, gc_store):
        store, driver, _ = gc_store(lambda query, params: [{"borrados": 0}])
        counts = store.gc.sweep_tenant("acme", "t")
        assert [query for query, _ in driver.calls] == [
            neo_gc.GC_ENTITY_MODEL_INSTANCE_QUERY,
            neo_gc.GC_LABELED_ENTITY_QUERY,
        ]
        assert counts == {
            "gc_entity_model_instance_deleted": 0,
            "gc_entity_model_instance_passes": 1,
            "gc_labeled_entity_deleted": 0,
            "gc_labeled_entity_passes": 1,
        }


class TestDeletionOps:
    def test_mark_deletion_pending_batches_the_keys(self, gc_store):
        keys = _keys(_shared.DELETE_BATCH_ROWS + 1)
        store, driver, _ = gc_store(lambda query, params: [{"n": len(params["keys"])}])
        store.deletion.mark_deletion_pending("acme", keys)
        assert [len(params["keys"]) for _, params in driver.calls] == [_shared.DELETE_BATCH_ROWS, 1]
        assert {query for query, _ in driver.calls} == {neo_deletion._MARK_PENDING_QUERY}

    def test_cascade_delete_feeds_the_collector_and_sums_the_steps(self, gc_store):
        store, driver, _ = gc_store(lambda query, params: [{"n": 2}])
        events: list[str] = []

        class _Collector:
            def gather(self, chunk):
                events.append("gather")

            def chunk_done(self):
                events.append("done")

        counters = store.deletion.cascade_delete(
            "acme", _keys(3), _Collector(), caller="delete_document"
        )
        assert events == ["gather", "done"]
        assert counters["structure_nodes_deleted"] == 2
        assert len(driver.calls) == len(neo_deletion.subtree_steps())
