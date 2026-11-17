"""
tests/unit/graph_neo4j/test_tabular_ops.py — Streaming tabular writes
(``tabular/graph_writer.py``), the Cypher of the tabular sink
(``graph/neo4j/tabular.py``) and the normalization keys of ``tabular/rows.py``.

Covers compute_row_normalization_keys (single source of keys for the scan and
write passes), the streaming write path (batches of 500), the normalization
path (passes B/C), bounded normalization tasks, a live-memory bound and the
statements of ``Neo4jTabularSink``. Which sink calls the traversal emits, in
which order, is in ``tests/unit/test_tabular_graph_writer.py``.

Everything is faked: no Neo4j, no LLM, no MongoDB.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from scinr.newton.graph import reset_graph_store, set_graph_store
from scinr.newton.graph.neo4j import driver as driver_mod
from scinr.newton.graph.neo4j import tabular as neo4j_tabular
from scinr.newton.graph.neo4j.tabular import Neo4jTabularSink
from scinr.newton.graph.types import Provenance, RowData, TableNode
from scinr.newton.tabular import graph_writer as tab_ops
from scinr.newton.tabular import rows as tabular_rows
from scinr.newton.tabular.models import ColumnFieldMapping, ColumnMapping
from scinr.newton.tabular.normalization.engine import NormalizationEngine
from scinr.newton.tabular.reader import iter_sheet_batches, scan_tabular_file


def _write_csv(path: Path, n_rows: int, ncols: int = 3, delimiter: str = ",") -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow([f"col{c}" for c in range(ncols)])
        for r in range(n_rows):
            w.writerow([f"v{r}_{c}" for c in range(ncols)])
    return path


class NormalizedAddress(BaseModel):
    city: str | None = None


class Person(BaseModel):
    name: str | None = None
    address_raw: str | None = None
    address: NormalizedAddress | None = Field(
        default=None,
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["address_raw"],
        },
    )


class PersonComposite(BaseModel):
    person: Person | None = None


_MAPPING = ColumnMapping(
    mappings=[
        ColumnFieldMapping(column_name="name", model_field_name="name", target_model="primary"),
        ColumnFieldMapping(
            column_name="addr", model_field_name="address_raw", target_model="primary"
        ),
    ]
)


def _factory(rows: list[list[str]], counter: list[int] | None = None):
    """RowBatchFactory over an in-memory list; counts how many passes were started."""

    def row_batches(batch_size: int):
        if counter is not None:
            counter[0] += 1
        for i in range(0, len(rows), batch_size):
            yield rows[i : i + batch_size]

    return row_batches


class _RecordingSink:
    """Records every sink call as ``(method, args)``; ``replace_raw_extraction_result`` returns a Provenance."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        async def _record(*args):
            self.calls.append((name, args))
            if name == "replace_raw_extraction_result":
                return Provenance(tenant_id="acme", created_by_user_id=None, job_id=None)

        return _record


def _store_with(sink):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    @asynccontextmanager
    async def _open():
        yield sink

    return SimpleNamespace(
        tabular=SimpleNamespace(sink=_open, delete_tabular_subgraph=AsyncMock()),
        annotation=SimpleNamespace(write_annotation=AsyncMock()),
    )


@pytest.fixture
def recording_sink():
    sink = _RecordingSink()
    store = _store_with(sink)
    set_graph_store(store)
    sink.store = store
    try:
        yield sink
    finally:
        reset_graph_store()


class _FakeResult:
    async def single(self):
        # create_table checks the owning :Document was matched.
        return {"created": 1}


class _FakeTx:
    async def run(self, *a, **k):
        return _FakeResult()

    async def commit(self):
        return None

    async def rollback(self):
        return None


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def begin_transaction(self):
        return _FakeTx()

    async def run(self, *a, **k):
        return None


class _FakeDriver:
    def session(self, **kwargs):
        return _FakeSession()


@pytest.fixture
def fake_cfg(monkeypatch):
    cfg = SimpleNamespace(
        neo4j_database="neo4j",
        normalization_enabled=True,
        normalization_llm=object(),
        llm=None,
        normalization_batch_size=5,
        llm_concurrency=2,
    )
    monkeypatch.setattr(tab_ops, "get_config", lambda: cfg)
    monkeypatch.setattr(neo4j_tabular, "get_config", lambda: cfg)
    monkeypatch.setattr(driver_mod, "get_async_driver", lambda: _FakeDriver())
    monkeypatch.setattr("scinr.newton.config.get_llm_semaphore", lambda: asyncio.Semaphore(2))
    return cfg


@pytest.fixture
def recorded_batches(monkeypatch):
    """Replace _write_row_batch; record (batch_len, batch_start_index, composite_results)."""
    calls: list[dict] = []

    async def fake_write_row_batch(**kwargs):
        calls.append(
            {
                "n": len(kwargs["rows_batch"]),
                "start": kwargs["batch_start_index"],
                "rows": kwargs["rows_batch"],
                "composites": kwargs.get("composite_results"),
                "row_indices": kwargs.get("row_indices"),
            }
        )

    monkeypatch.setattr(tab_ops, "_write_row_batch", fake_write_row_batch)
    return calls


class TestComputeRowNormalizationKeys:
    def test_key_matches_the_engine_hash_and_is_deterministic(self):
        keys = tabular_rows.compute_row_normalization_keys(
            ["name", "addr"], ["Ann", "1 Main St"], _MAPPING, Person, {}, []
        )
        again = tabular_rows.compute_row_normalization_keys(
            ["name", "addr"], ["Ann", "1 Main St"], _MAPPING, Person, {}, []
        )

        (key,) = keys
        assert key.unique_key == (
            "NormalizedAddress:"
            + NormalizationEngine._hash_source_values({"address_raw": "1 Main St"})
        )
        assert key.is_primary and key.field_name == "address"
        assert key.target_type is NormalizedAddress
        assert key.source_values == {"address_raw": "1 Main St"}
        assert keys == again

    def test_row_without_normalizable_value_has_no_keys(self):
        assert tabular_rows.compute_row_normalization_keys(
            ["name", "addr"], ["Ann", ""], _MAPPING, Person, {}, []
        ) == []

    def test_scan_pass_and_write_pass_use_the_same_keys(self):
        headers = ["name", "addr"]
        rows = [["a", "x"], ["b", "y"], ["c", "x"], ["d", ""]]

        dedup_map, scanned = tabular_rows._build_normalization_dedup_map(
            headers, [rows[:2], rows[2:]], Person, {}, [], _MAPPING
        )
        write_pass_keys = {
            k.unique_key
            for r in rows
            for k in tabular_rows.compute_row_normalization_keys(headers, r, _MAPPING, Person, {}, [])
        }

        assert scanned == 4
        assert set(dedup_map) == write_pass_keys
        assert len(dedup_map) == 2  # "x" is deduplicated across rows 0 and 2
        entry = dedup_map[next(iter(dedup_map))]
        assert entry.row_indices == []  # no per-row bookkeeping any more


class TestStandardWritePathStreams:
    async def test_2000_rows_are_written_in_4_batches_of_500_never_the_whole_list(
        self, fake_cfg, recorded_batches, recording_sink
    ):
        rows = [[f"n{i}", f"a{i}"] for i in range(2000)]
        passes = [0]
        decision = SimpleNamespace(
            matched_model_class=None, complementary_models=[], supplementary_fields=[]
        )
        sheet = {
            "file_path": "unused",
            "sheet_name": "s",
            "headers": ["name", "addr"],
            "total_rows": 2000,
        }

        table_id = await tab_ops.write_tabular_subgraph(
            doc_path="doc",
            document_name="doc",
            resolved_version=1,
            sheet=sheet,
            row_batches=_factory(rows, passes),
            sheet_index=0,
            decision=decision,
            mapping=_MAPPING,
            tenant_id="acme",
        )

        assert table_id == "acme::doc::1::table_1"
        recording_sink.store.annotation.write_annotation.assert_awaited_once_with(
            "acme::doc::1::table_1", decision, "doc"
        )
        ((create_name, (table,)),) = recording_sink.calls
        assert create_name == "create_table"
        assert table.id == "acme::doc::1::table_1"
        assert table.doc_key == ("acme", "doc", 1)
        assert [c["n"] for c in recorded_batches] == [500, 500, 500, 500]
        assert [c["start"] for c in recorded_batches] == [0, 500, 1000, 1500]
        assert all(c["n"] < len(rows) for c in recorded_batches)
        assert passes == [1]  # a single streaming pass when there is no normalization


class TestNormalizationWritePath:
    @pytest.fixture
    def engine_calls(self, monkeypatch):
        calls: list[list[str]] = []

        async def fake_process_key_batch(self, entries):
            calls.append([e.unique_key for e in entries])
            for e in entries:
                self.result_cache[e.unique_key] = NormalizedAddress(
                    city=str(e.source_values["address_raw"]).upper()
                )
            return list(entries)

        monkeypatch.setattr(NormalizationEngine, "process_key_batch", fake_process_key_batch)
        return calls

    async def _run(self, rows, passes):
        return await tab_ops._write_tabular_with_normalization(
            sink=_RecordingSink(),
            table_composite_id="doc::1::table_1",
            headers=["name", "addr"],
            row_batches=_factory(rows, passes),
            decision_uid="uid",
            decision=SimpleNamespace(matched_model_class="Person"),
            mapping=_MAPPING,
            primary_cls=Person,
            composite_cls=PersonComposite,
            primary_field_name="person",
            comp_class_names=[],
            comp_cls_map={},
            document_name="doc",
            theme="default",
            sheet_page_id="",
        )

    async def test_scan_normalize_write_passes_apply_the_cached_results(
        self, fake_cfg, recorded_batches, engine_calls, caplog
    ):
        # 1200 rows, 7 distinct addresses, every 4th row without address.
        cities = [f"city {i}" for i in range(7)]
        rows = [
            [f"n{i}", "" if i % 4 == 3 else cities[i % 7]] for i in range(1200)
        ]
        passes = [0]

        with caplog.at_level(logging.WARNING, logger=tab_ops.logger.name):
            written = await self._run(rows, passes)

        assert written == 1200
        assert passes == [2]  # pass B (scan) + pass C (write)
        # LLM sees only the unique keys, in batches of normalization_batch_size=5.
        seen_keys = [k for batch in engine_calls for k in batch]
        assert len(seen_keys) == len(set(seen_keys)) == 7
        assert all(len(batch) <= 5 for batch in engine_calls)
        # Batches of 500 in file order, indices contiguous.
        assert [c["n"] for c in recorded_batches] == [500, 500, 200]
        assert [c["start"] for c in recorded_batches] == [0, 500, 1000]
        assert recorded_batches[1]["row_indices"] == list(range(500, 1000))
        # Normalization applied where there is a key; plain otherwise.
        for c in recorded_batches:
            for offset, row_values, composite, extraction_uid in c["composites"]:
                row_index = c["start"] + offset
                assert row_values == rows[row_index]
                assert extraction_uid  # deterministic uid
                if row_index % 4 == 3:
                    assert composite.person.address is None
                else:
                    assert composite.person.address == NormalizedAddress(
                        city=cities[row_index % 7].upper()
                    )
        assert not [r for r in caplog.records if "cache miss" in r.getMessage()]

    async def test_llm_failure_leaves_the_field_as_is_and_still_writes_every_row(
        self, fake_cfg, recorded_batches, monkeypatch, caplog
    ):
        async def failing_process_key_batch(self, entries):
            return []  # nothing cached: normalization "failed"

        monkeypatch.setattr(NormalizationEngine, "process_key_batch", failing_process_key_batch)
        rows = [[f"n{i}", "addr"] for i in range(10)]

        with caplog.at_level(logging.WARNING, logger=tab_ops.logger.name):
            written = await self._run(rows, [0])

        assert written == 10
        (call,) = recorded_batches
        assert all(c[2].person.address is None for c in call["composites"])
        assert len([r for r in caplog.records if "cache miss" in r.getMessage()]) == 10

    async def test_no_normalizable_values_falls_back_to_the_standard_path(
        self, fake_cfg, recorded_batches, engine_calls
    ):
        rows = [[f"n{i}", ""] for i in range(700)]
        passes = [0]

        written = await self._run(rows, passes)

        assert written == 700
        assert engine_calls == []
        assert passes == [2]  # scan pass + standard write pass
        assert [c["n"] for c in recorded_batches] == [500, 200]
        assert all(c["composites"] is None for c in recorded_batches)


class TestRunBounded:
    async def test_at_most_limit_tasks_alive_and_all_items_processed(self):
        alive = 0
        max_alive = 0
        done: list[int] = []

        async def worker(i: int) -> None:
            nonlocal alive, max_alive
            alive += 1
            max_alive = max(max_alive, alive)
            await asyncio.sleep(0.001)
            alive -= 1
            done.append(i)

        await tab_ops._run_bounded(range(200), worker, limit=8)

        assert max_alive <= 8
        assert sorted(done) == list(range(200))

    async def test_items_are_consumed_lazily(self):
        consumed = 0

        def items():
            nonlocal consumed
            for i in range(100):
                consumed += 1
                yield i

        started = asyncio.Event()
        release = asyncio.Event()

        async def worker(i: int) -> None:
            started.set()
            await release.wait()

        task = asyncio.create_task(tab_ops._run_bounded(items(), worker, limit=3))
        await started.wait()
        await asyncio.sleep(0.01)
        assert consumed <= 4  # 3 running + the one waiting for a free slot
        release.set()
        await task
        assert consumed == 100

    async def test_first_failure_is_raised_and_pending_tasks_are_cancelled(self):
        cancelled = 0

        async def worker(i: int) -> None:
            nonlocal cancelled
            if i == 0:
                raise RuntimeError("boom")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled += 1
                raise

        # If the failure were only surfaced once every task finished, this would take 10 s.
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(tab_ops._run_bounded(range(3), worker, limit=3), timeout=2)
        assert cancelled == 2

    async def test_empty_input_is_a_noop(self):
        async def worker(i: int) -> None:  # pragma: no cover
            raise AssertionError

        await tab_ops._run_bounded([], worker, limit=4)


class TestLiveMemoryIsIndependentOfRowCount:
    async def test_standard_write_path_peak_is_bounded(self, tmp_path, fake_cfg, monkeypatch):
        n_rows = 150_000
        p = _write_csv(tmp_path / "huge.csv", n_rows, ncols=8)

        async def sink(**kwargs):
            return None  # like a real write: the batch is dropped afterwards

        monkeypatch.setattr(tab_ops, "_write_row_batch", sink)

        def factory(batch_size):
            return iter_sheet_batches(p, "huge", batch_size)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            written = await tab_ops._write_rows_in_batches(
                sink=None,
                table_composite_id="t",
                headers=["c"],
                row_batches=factory,
                decision_uid="u",
                decision=None,
                mapping=None,
                primary_cls=None,
                composite_cls=None,
                primary_field_name=None,
                comp_class_names=[],
                comp_cls_map={},
                document_name="d",
                theme="default",
                sheet_page_id="",
            )
            (scan,) = scan_tabular_file(p)
            peak_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        finally:
            tracemalloc.stop()

        assert written == n_rows
        assert scan["total_rows"] == n_rows
        # Materialising 150k x 8 cells costs well over 100 MB; streaming stays tiny.
        assert peak_mb < 50, f"peak live memory {peak_mb:.0f} MB"


# ---------------------------------------------------------------------------
# Cypher of Neo4jTabularSink
# ---------------------------------------------------------------------------


class _Rec:
    """Result of a recorded statement: ``single()`` returns the next scripted record."""

    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _StatementDriver:
    """Async driver recording every statement of its sessions and transactions.

    *records* are the ``single()`` results handed out in order; ``events`` also
    sees ``begin``/``commit``/``rollback`` so transaction boundaries can be asserted.
    """

    def __init__(self, records=()):
        self.records = list(records)
        self.statements: list[tuple[str, dict]] = []
        self.events: list[str] = []
        self.sessions = 0

    def session(self, **kwargs):
        driver = self
        driver.sessions += 1

        class _Session(_FakeSession):
            async def run(self, query, **params):
                driver.statements.append((query, params))
                return _Rec(driver.records.pop(0) if driver.records else None)

            async def begin_transaction(self):
                driver.events.append("begin")

                class _T:
                    async def run(_self, query, **params):
                        driver.statements.append((query, params))
                        return _Rec(driver.records.pop(0) if driver.records else None)

                    async def commit(_self):
                        driver.events.append("commit")

                    async def rollback(_self):
                        driver.events.append("rollback")

                return _T()

        return _Session()


@pytest.fixture
def sink_driver(monkeypatch):
    cfg = SimpleNamespace(neo4j_database="neo4j")
    monkeypatch.setattr(neo4j_tabular, "get_config", lambda: cfg)

    def _make(records=()):
        driver = _StatementDriver(records)
        return driver, Neo4jTabularSink(driver)

    return _make


_PROV = Provenance(tenant_id="acme", created_by_user_id="u1", job_id="j1")


def _table() -> TableNode:
    return TableNode(
        id="acme::F/sheet::1::table_1",
        node_id="table_1",
        title="s",
        order=1,
        theme="t",
        col_count=2,
        row_count=3,
        doc_key=("acme", "F/sheet", 1),
        page_id="p1",
    )


class TestNeo4jTabularSinkCypher:
    async def test_create_table_matches_the_document_and_commits(self, sink_driver):
        driver, sink = sink_driver([{"created": 1}])

        await sink.create_table(_table())

        ((query, params),) = driver.statements
        assert "MATCH (d:Document {tenant_id: $tenant_id, path: $doc_path, version: $version})" in query
        assert "MERGE (d:Document" not in query
        assert params == dict(
            id="acme::F/sheet::1::table_1",
            node_id="table_1",
            title="s",
            order=1,
            theme="t",
            col_count=2,
            row_count=3,
            tenant_id="acme",
            doc_path="F/sheet",
            version=1,
            page_id="p1",
        )
        assert driver.events == ["begin", "commit"]

    async def test_missing_document_rolls_back_and_raises_the_same_error(self, sink_driver):
        driver, sink = sink_driver([{"created": 0}])

        with pytest.raises(RuntimeError, match="write_tabular_subgraph: Document not found"):
            await sink.create_table(_table())

        assert driver.events == ["begin", "rollback"]

    async def test_write_row_batch_is_one_unwind_transaction(self, sink_driver):
        driver, sink = sink_driver()
        rows = [
            RowData(
                id=f"t/row_{i + 1}",
                node_id=f"row_{i + 1}",
                row_index=i,
                appearance_order=i + 1,
                info_uid=f"iu{i}",
                description=f"d{i}",
            )
            for i in range(2)
        ]

        await sink.write_row_batch("t", rows, "md-uid", "theme", "page")

        ((query, params),) = driver.statements
        assert query.lstrip().startswith("UNWIND $rows AS row_data")
        assert "MERGE (r)-[:HAS_MODEL_DECISION]->(md)" in query
        assert params["rows"] == [
            {
                "id": "t/row_1",
                "node_id": "row_1",
                "row_index": 0,
                "appearance_order": 1,
                "info_uid": "iu0",
                "description": "d0",
            },
            {
                "id": "t/row_2",
                "node_id": "row_2",
                "row_index": 1,
                "appearance_order": 2,
                "info_uid": "iu1",
                "description": "d1",
            },
        ]
        assert (params["table_id"], params["md_uid"], params["theme"], params["page_id"]) == (
            "t",
            "md-uid",
            "theme",
            "page",
        )
        assert driver.events == ["begin", "commit"]

    async def test_replace_raw_deletes_then_creates_in_one_session(self, sink_driver):
        driver, sink = sink_driver(
            [{"tenant_id": "acme", "created_by_user_id": "u1", "job_id": "j1"}]
        )

        prov = await sink.replace_raw_extraction_result("row", "er", "doc", "ts")

        assert prov == _PROV
        assert driver.sessions == 1
        (delete_q, delete_p), (create_q, create_p) = driver.statements
        assert "DETACH DELETE er" in delete_q and delete_p == {"nid": "row"}
        assert "model_class:   'GenericTabularRow'" in create_q
        assert "source:        'tabular_raw'" in create_q
        assert create_p == dict(
            nid="row",
            uid="er",
            doc_name="doc",
            timestamp="ts",
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )

    async def test_replace_raw_returns_none_without_the_second_statement(self, sink_driver):
        driver, sink = sink_driver([None])

        assert await sink.replace_raw_extraction_result("row", "er", "doc", "ts") is None
        assert len(driver.statements) == 1

    async def test_replace_raw_refuses_a_row_without_tenant_between_the_statements(
        self, sink_driver
    ):
        driver, sink = sink_driver(
            [{"tenant_id": None, "created_by_user_id": None, "job_id": None}]
        )

        with pytest.raises(RuntimeError, match="no tenant_id"):
            await sink.replace_raw_extraction_result("row", "er", "doc", "ts")

        assert len(driver.statements) == 1

    async def test_link_raw_column_value_interpolates_the_rel_type_in_its_own_session(
        self, sink_driver
    ):
        driver, sink = sink_driver()

        await sink.link_raw_column_value("er", "HAS_NAME", "ent", "Ann", "ann", _PROV)
        await sink.link_raw_column_value("er", "HAS_CITY", "ent2", "Rome", "rome", _PROV)

        assert driver.sessions == 2
        (q1, p1), (q2, _) = driver.statements
        assert "MERGE (er)-[:HAS_NAME]->(e)" in q1
        assert "MERGE (er)-[:HAS_CITY]->(e)" in q2
        assert "MERGE (e:Entity {uid: $uid})" in q1
        assert p1 == dict(
            normalized_value="ann",
            uid="ent",
            value="Ann",
            er_uid="er",
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )
