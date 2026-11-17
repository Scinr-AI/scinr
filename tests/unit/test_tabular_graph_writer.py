"""
tests/unit/test_tabular_graph_writer.py — the domain traversal of the tabular
subgraph (``tabular/graph_writer.py``) against a recording fake sink.

The traversal has no Cypher: what matters is which sink calls it emits, with which
arguments, in which order. The Cypher of each call is in
``tests/unit/graph_neo4j/test_tabular_ops.py``; the streaming, normalization and
memory properties of the traversal are there too.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scinr.newton.graph import reset_graph_store, set_graph_store
from scinr.newton.graph.types import Provenance
from scinr.newton.tabular import graph_writer
from scinr.newton.tabular.graph_writer import write_tabular_subgraph
from scinr.newton.utils.uid import make_uid

_PROV = Provenance(tenant_id="acme", created_by_user_id="u1", job_id="j1")
_SHEET = {"sheet_name": "s", "headers": ["name", "city"], "total_rows": 3}
_DECISION = SimpleNamespace(
    matched_model_class=None, complementary_models=[], supplementary_fields=[]
)
_TABLE_ID = "acme::F/sheet::1::table_1"


class _RecordingSink:
    """Records every call as ``(method, args)``; raw extraction returns the configured provenance."""

    def __init__(self, prov: Provenance | None = _PROV) -> None:
        self.prov = prov
        self.calls: list[tuple[str, tuple]] = []
        self.fail_columns: set[str] = set()

    async def replace_raw_extraction_result(self, *args):
        self.calls.append(("replace_raw_extraction_result", args))
        return self.prov

    async def link_raw_column_value(self, er_uid, rel_type, *rest):
        self.calls.append(("link_raw_column_value", (er_uid, rel_type, *rest)))
        if rel_type in self.fail_columns:
            raise RuntimeError("engine down")

    def __getattr__(self, name):
        async def _record(*args):
            self.calls.append((name, args))

        return _record

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(
        graph_writer, "get_config", lambda: SimpleNamespace(normalization_enabled=False)
    )


@pytest.fixture
def traversal():
    holder = {}

    def _install(sink: _RecordingSink):
        @asynccontextmanager
        async def _open():
            yield sink

        store = SimpleNamespace(
            tabular=SimpleNamespace(sink=_open, delete_tabular_subgraph=AsyncMock()),
            annotation=SimpleNamespace(write_annotation=AsyncMock()),
        )
        holder["store"] = store
        set_graph_store(store)
        return store

    try:
        yield _install
    finally:
        reset_graph_store()


def _batches(rows):
    def row_batches(batch_size):
        for i in range(0, len(rows), batch_size):
            yield rows[i : i + batch_size]

    return row_batches


async def _write(rows, **kwargs):
    return await write_tabular_subgraph(
        doc_path="F/sheet",
        document_name="sheet",
        resolved_version=1,
        sheet={**_SHEET, "total_rows": len(rows)},
        row_batches=_batches(rows),
        sheet_index=0,
        decision=_DECISION,
        mapping=None,
        tenant_id="acme",
        **kwargs,
    )


async def test_raw_rows_emission_order_and_arguments(traversal):
    sink = _RecordingSink()
    store = traversal(sink)
    rows = [["Ann", "Rome"], ["Bob", ""]]

    table_id = await _write(rows, theme="th", sheet_page_id="pg")

    assert table_id == _TABLE_ID
    store.tabular.delete_tabular_subgraph.assert_not_awaited()
    store.annotation.write_annotation.assert_awaited_once_with(_TABLE_ID, _DECISION, "sheet")
    create, batch, *raw = sink.calls
    assert create[0] == "create_table"
    (table,) = create[1]
    assert (table.id, table.node_id, table.title, table.order, table.theme) == (
        _TABLE_ID,
        "table_1",
        "s",
        1,
        "th",
    )
    assert (table.col_count, table.row_count, table.doc_key, table.page_id) == (
        2,
        2,
        ("acme", "F/sheet", 1),
        "pg",
    )
    assert batch[0] == "write_row_batch"
    table_id_arg, row_data, md_uid, theme, page_id = batch[1]
    assert table_id_arg == _TABLE_ID
    assert md_uid == make_uid("model_decision", _TABLE_ID, "null")
    assert (theme, page_id) == ("th", "pg")
    assert [(r.id, r.node_id, r.row_index, r.appearance_order) for r in row_data] == [
        (f"{_TABLE_ID}/row_1", "row_1", 0, 1),
        (f"{_TABLE_ID}/row_2", "row_2", 1, 2),
    ]
    assert row_data[0].info_uid == make_uid(f"{_TABLE_ID}/row_1", "row_data")
    # Rows run concurrently, so compare per row: replace first, then one link per non-empty column.
    by_row: dict[str, list[tuple[str, tuple]]] = {}
    for name, args in raw:
        er_uid = args[1] if name == "replace_raw_extraction_result" else args[0]
        by_row.setdefault(er_uid, []).append((name, args))
    assert len(by_row) == 2
    ann_calls = by_row[make_uid("tabular_extraction", f"{_TABLE_ID}/row_1", "raw")]
    assert [n for n, _ in ann_calls] == [
        "replace_raw_extraction_result",
        "link_raw_column_value",
        "link_raw_column_value",
    ]
    replace_args = ann_calls[0][1]
    assert replace_args[0] == f"{_TABLE_ID}/row_1" and replace_args[2] == "sheet"
    er_uid, rel_type, entity_uid, value, normalized, prov = ann_calls[1][1]
    assert (rel_type, value, normalized, prov) == ("HAS_NAME", "Ann", "ann", _PROV)
    assert entity_uid == make_uid("entity", "acme", "ann")
    bob_calls = by_row[make_uid("tabular_extraction", f"{_TABLE_ID}/row_2", "raw")]
    assert [n for n, _ in bob_calls] == ["replace_raw_extraction_result", "link_raw_column_value"]


async def test_update_mode_deletes_the_old_subgraph_before_creating_the_table(traversal):
    sink = _RecordingSink()
    store = traversal(sink)

    await _write([["a", "b"]], update_mode=True)

    store.tabular.delete_tabular_subgraph.assert_awaited_once_with("F/sheet", 1, 0, tenant_id="acme")
    assert sink.names()[0] == "create_table"


async def test_rows_are_written_in_batches_of_500_in_file_order(traversal):
    sink = _RecordingSink()
    traversal(sink)
    rows = [[f"n{i}", ""] for i in range(1100)]

    await _write(rows)

    batches = [c[1][1] for c in sink.calls if c[0] == "write_row_batch"]
    assert [len(b) for b in batches] == [500, 500, 100]
    assert [b[0].row_index for b in batches] == [0, 500, 1000]
    # Each batch is fully written before the next one is sent.
    first_second_batch = sink.names().index("write_row_batch", sink.names().index("write_row_batch") + 1)
    assert sink.names()[first_second_batch - 1] == "link_raw_column_value"


async def test_row_missing_from_the_graph_is_skipped_with_a_warning(traversal, caplog):
    sink = _RecordingSink(prov=None)
    traversal(sink)

    with caplog.at_level("WARNING", logger=graph_writer.logger.name):
        await _write([["Ann", "Rome"]])

    assert "link_raw_column_value" not in sink.names()
    assert "Row StructureNode not found" in caplog.text


async def test_a_failed_column_is_logged_and_the_others_still_written(traversal, caplog):
    sink = _RecordingSink()
    sink.fail_columns = {"HAS_NAME"}
    traversal(sink)

    with caplog.at_level("WARNING", logger=graph_writer.logger.name):
        await _write([["Ann", "Rome"]])

    links = [c[1][1] for c in sink.calls if c[0] == "link_raw_column_value"]
    assert links == ["HAS_NAME", "HAS_CITY"]
    assert "failed to write column 'name'" in caplog.text


async def test_long_raw_values_are_truncated_to_500_characters(traversal):
    sink = _RecordingSink()
    traversal(sink)

    await _write([["x" * 800, ""]])

    (link,) = [c[1] for c in sink.calls if c[0] == "link_raw_column_value"]
    assert len(link[3]) == 500 and link[4] == "x" * 500


async def test_rows_with_a_model_use_the_extraction_writer_not_the_raw_calls(traversal):
    class Person:
        pass

    sink = _RecordingSink()
    traversal(sink)
    fake_write = AsyncMock()
    composite = object()

    decision = SimpleNamespace(
        matched_model_class="Person", complementary_models=[], supplementary_fields=[]
    )
    with (
        patch("scinr.newton.entity_extraction.model_resolver.resolve_model_class", return_value=Person),
        patch(
            "scinr.newton.entity_extraction.schema_composer.compose_extraction_schema",
            return_value=object,
        ),
        patch.object(graph_writer, "instance_has_normalizable_fields", return_value=False),
        patch.object(graph_writer, "_instantiate_composite_from_row", return_value=composite),
        patch.object(graph_writer, "write_extraction_subgraph", fake_write),
    ):
        await write_tabular_subgraph(
            doc_path="F/sheet",
            document_name="sheet",
            resolved_version=1,
            sheet={**_SHEET, "total_rows": 2},
            row_batches=_batches([["a", "b"], ["c", "d"]]),
            sheet_index=0,
            decision=decision,
            mapping=None,
            tenant_id="acme",
        )

    assert sink.names() == ["create_table", "write_row_batch"]
    assert fake_write.await_count == 2
    kwargs = fake_write.await_args_list[0].kwargs
    assert kwargs["node_full_id"] == f"{_TABLE_ID}/row_1"
    assert kwargs["composite_instance"] is composite
    assert kwargs["primary_model_class"] == "Person"
    assert kwargs["document_name"] == "sheet"
