"""
tests/unit/test_cascade.py — Unit tests for scinr.newton.ingest._cascade (the
batched subtree deletion shared by delete_document() and freeze_document())
and for the batched GC queries of scinr.newton.ingest._gc.

No real Neo4j: a fake driver records what runs and how (auto-commit
``session.run`` vs. a managed transaction).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from neo4j.exceptions import Neo4jError

from scinr.newton.ingest import _cascade, _gc


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
        deletes = [q for _, q in _cascade.subtree_steps() if "DETACH DELETE" in q]
        assert len(deletes) == 8  # pf, pm, cm, sf, md, er, iu, structure nodes
        for query in deletes:
            assert f"IN TRANSACTIONS OF {_cascade.DELETE_BATCH_ROWS} ROWS" in query
            assert "RETURN count(*) AS n" in query

    def test_nodes_to_delete_are_known_before_the_first_batch_commits(self):
        """Grouping by the node is an aggregation: it reads the whole input
        (the HAS_CHILD walk) before a row reaches the batches, so deleting a
        StructureNode never hides its children from the query."""
        query = _cascade.STRUCTURE_NODES_DELETE_QUERY
        assert query.index("HAS_CHILD*0..") < query.index("count(*) AS hits") < query.index("CALL (x)")

    def test_batched_rows_never_carry_a_collected_list(self):
        # collect() + UNWIND charges every batched row for the whole list:
        # that is what overran the transaction memory pool.
        for _, query in _cascade.subtree_steps():
            if "DETACH DELETE" in query:
                assert "collect(" not in query

    def test_default_steps_delete_the_whole_subtree_children_first(self):
        steps = _cascade.subtree_steps()
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
        "query", [_gc.GC_ENTITY_MODEL_INSTANCE_QUERY, _gc.GC_LABELED_ENTITY_QUERY]
    )
    def test_gc_deletes_in_bounded_transactions(self, query):
        assert f"IN TRANSACTIONS OF {_cascade.DELETE_BATCH_ROWS} ROWS" in query
        assert "mi.tenant_id = $tenant_id" in query
        assert "RETURN count(*) AS borrados" in query


class TestRunners:
    def test_count_query_is_auto_commit(self):
        """CALL ... IN TRANSACTIONS is refused inside an explicit transaction."""
        driver = _Driver(row={"n": 7})
        assert _cascade.run_count_query(driver, "neo4j", "Q", tenant_id="acme") == 7
        assert driver.calls == [("session.run", "Q", {"tenant_id": "acme"})]

    def test_count_query_without_a_row_counts_zero(self):
        driver = _Driver(row=None)
        driver.serve = lambda kind, query, params: _Result(None)
        assert _cascade.run_count_query(driver, "neo4j", "Q") == 0

    def test_count_query_retries_a_full_memory_pool(self):
        driver = _Driver(row={"n": 3}, errors=[_memory_pool_error(), _memory_pool_error()])
        with patch("scinr.newton.utils.neo4j_retry.time.sleep") as sleep:
            assert _cascade.run_count_query(driver, "neo4j", "Q") == 3
        assert len(driver.calls) == 3
        assert sleep.call_count == 2

    def test_steps_run_per_chunk_of_documents_and_sum(self):
        driver = _Driver()
        keys = _keys(_cascade.DOCUMENTS_PER_QUERY * 2 + 3)
        steps = _cascade.subtree_steps()

        counters = _cascade.run_subtree_steps(driver, "neo4j", "acme", keys, steps, caller="t")

        sizes = [len(params["keys"]) for _, _, params in driver.calls]
        per_chunk = [_cascade.DOCUMENTS_PER_QUERY, _cascade.DOCUMENTS_PER_QUERY, 3]
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
        keys = _keys(_cascade.DOCUMENTS_PER_QUERY + 1)
        steps = _cascade.subtree_steps()
        events: list[tuple] = []

        class _Orphans:
            def gather(self, chunk):
                events.append(("gather", len(chunk), len(driver.calls)))

            def chunk_done(self):
                events.append(("done", len(driver.calls)))

        _cascade.run_subtree_steps(
            driver, "neo4j", "acme", keys, steps, caller="t", orphans=_Orphans()
        )

        assert events == [
            ("gather", _cascade.DOCUMENTS_PER_QUERY, 0),
            ("done", len(steps)),
            ("gather", 1, len(steps)),
            ("done", 2 * len(steps)),
        ]

    def test_a_failing_step_stops_the_run(self):
        driver = _Driver(errors=[ValueError("boom")])
        with pytest.raises(ValueError, match="boom"):
            _cascade.run_subtree_steps(
                driver, "neo4j", "acme", _keys(3), _cascade.subtree_steps(), caller="t"
            )
        assert len(driver.calls) == 1

    def test_documents_are_deleted_in_order_in_managed_transactions(self):
        driver = _Driver()
        keys = _keys(_cascade.DELETE_BATCH_ROWS + 2)

        assert _cascade.delete_documents(driver, "neo4j", "acme", keys) == len(keys)

        assert [kind for kind, _, _ in driver.calls] == ["tx.run", "tx.run"]
        assert all(q == _cascade.DOCUMENTS_DELETE_QUERY for _, q, _ in driver.calls)
        assert [k for _, _, params in driver.calls for k in params["keys"]] == keys
