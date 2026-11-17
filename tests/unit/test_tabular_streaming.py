"""
tests/unit/test_tabular_streaming.py — Streaming tabular pipeline (memory O(batch)).

Covers:
1. reader: scan_tabular_file (pass A) and iter_sheet_batches, and their equivalence
   with the materialising read_csv / read_xlsx / read_tabular_file.
2. state: load_sheets keeps no data rows in the LangGraph state.
3. The write path and its normalization keys live in
   tests/unit/graph_neo4j/test_tabular_ops.py.

Everything is faked: no Neo4j, no LLM, no MongoDB.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scinr.newton.tabular.reader import (
    iter_sheet_batches,
    preview_indices,
    read_csv,
    read_tabular_file,
    scan_tabular_file,
    select_preview_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, n_rows: int, ncols: int = 3, delimiter: str = ",") -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow([f"col{c}" for c in range(ncols)])
        for r in range(n_rows):
            w.writerow([f"v{r}_{c}" for c in range(ncols)])
    return path


def _flatten(batches) -> list[list[str]]:
    return [row for batch in batches for row in batch]


def _make_xlsx(path: Path) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "People"
    ws1.append(["name", "age"])
    for i in range(12):
        ws1.append([f"p{i}", i])
    ws1.append([None, None])  # empty row in the middle of nowhere
    ws1.append(["last", 99])
    ws2 = wb.create_sheet("Empty")  # no cells at all
    ws3 = wb.create_sheet("Tiny")
    ws3.append(["a", "b", "c"])
    ws3.append([1, 2])  # short row -> padded
    del ws2
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# 1. Reader
# ---------------------------------------------------------------------------


class TestPreviewIndices:
    @pytest.mark.parametrize("n", [0, 1, 4, 5, 6, 7, 8, 100, 1001])
    def test_matches_select_preview_rows(self, n):
        rows = [[str(i)] for i in range(n)]
        sheet = {"sheet_name": "s", "headers": ["h"], "all_rows": rows, "total_rows": n}

        assert select_preview_rows(sheet)["row_indices"] == preview_indices(n)


class TestScanTabularFile:
    def test_csv_scan_has_headers_total_and_preview_without_rows(self, tmp_path):
        p = _write_csv(tmp_path / "data.csv", 1000)

        (scan,) = scan_tabular_file(p)

        assert scan["sheet_name"] == "data"
        assert scan["headers"] == ["col0", "col1", "col2"]
        assert scan["total_rows"] == 1000
        assert "all_rows" not in scan
        preview = scan["preview"]
        assert preview["row_indices"] == [0, 250, 500, 750, 999]
        assert [r[0] for r in preview["preview_rows"]] == ["v0_0", "v250_0", "v500_0", "v750_0", "v999_0"]
        assert preview["total_rows"] == 1000

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_small_csv_previews_every_row(self, tmp_path, n):
        p = _write_csv(tmp_path / "small.csv", n)

        (scan,) = scan_tabular_file(p)

        assert scan["total_rows"] == n
        assert scan["preview"]["row_indices"] == list(range(n))
        assert len(scan["preview"]["preview_rows"]) == n

    def test_total_ignores_empty_rows_and_counts_multiline_records_once(self, tmp_path):
        p = tmp_path / "odd.csv"
        p.write_text('id,note\n1,"a\nb"\n\n,\n2,c\n', encoding="utf-8")

        (scan,) = scan_tabular_file(p)

        assert scan["total_rows"] == 2
        assert scan["preview"]["preview_rows"] == [["1", "a\nb"], ["2", "c"]]

    def test_empty_csv_gives_placeholder_sheet(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")

        (scan,) = scan_tabular_file(p)

        assert scan["sheet_name"] == "empty"
        assert scan["headers"] == []
        assert scan["total_rows"] == 0
        assert scan["preview"]["preview_rows"] == []

    def test_scan_agrees_with_read_tabular_file_for_xlsx(self, tmp_path):
        p = _make_xlsx(tmp_path / "book.xlsx")

        scans = scan_tabular_file(p)
        sheets = read_tabular_file(p)

        assert [s["sheet_name"] for s in scans] == [s["sheet_name"] for s in sheets] == ["People", "Tiny"]
        for scan, sheet in zip(scans, sheets, strict=True):
            assert scan["headers"] == sheet["headers"]
            assert scan["total_rows"] == sheet["total_rows"]
            assert scan["preview"] == select_preview_rows(sheet)

    def test_binary_csv_raises(self, tmp_path):
        # AppleDouble resource fork (__MACOSX/._name.csv): header + xattr data
        # whose value happens to contain newlines.
        p = tmp_path / "._data.csv"
        p.write_bytes(
            b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \x00\x02"
            + b"\x00" * 40 + b"hello,world\nline2,foo\n"
        )
        with pytest.raises(ValueError, match="binary content"):
            scan_tabular_file(p)
        with pytest.raises(ValueError, match="binary content"):
            list(iter_sheet_batches(p, "._data"))

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("a,b")
        with pytest.raises(ValueError):
            scan_tabular_file(p)


class TestIterSheetBatches:
    def test_concatenation_equals_read_csv_for_small_and_large_files(self, tmp_path):
        for n in (1, 7, 501, 1234):
            p = _write_csv(tmp_path / f"f{n}.csv", n, ncols=4)
            expected = read_csv(p)[0]["all_rows"]

            got = _flatten(iter_sheet_batches(p, p.stem, batch_size=500))

            assert got == expected

    def test_batches_are_bounded_ordered_and_last_one_partial(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv", 1201)

        batches = list(iter_sheet_batches(p, "f", batch_size=500))

        assert [len(b) for b in batches] == [500, 500, 201]
        assert batches[0][0][0] == "v0_0"
        assert batches[2][-1][0] == "v1200_0"

    def test_header_only_and_empty_sheets_yield_nothing(self, tmp_path):
        only_header = tmp_path / "h.csv"
        only_header.write_text("a,b\n", encoding="utf-8")
        empty = tmp_path / "e.csv"
        empty.write_text("", encoding="utf-8")

        assert list(iter_sheet_batches(only_header, "h")) == []
        assert list(iter_sheet_batches(empty, "e")) == []

    def test_short_and_long_rows_are_padded_and_trimmed_like_read_csv(self, tmp_path):
        p = tmp_path / "ragged.csv"
        p.write_text("a,b,c\n1\n1,2,3,4,5\n\n7,8,9\n", encoding="utf-8")

        assert _flatten(iter_sheet_batches(p, "ragged")) == read_csv(p)[0]["all_rows"]
        assert read_csv(p)[0]["all_rows"] == [["1", "", ""], ["1", "2", "3"], ["7", "8", "9"]]

    def test_xlsx_sheets_stream_the_same_rows_as_read_tabular_file(self, tmp_path):
        p = _make_xlsx(tmp_path / "book.xlsx")
        by_name = {s["sheet_name"]: s["all_rows"] for s in read_tabular_file(p)}

        for name in ("People", "Tiny"):
            assert _flatten(iter_sheet_batches(p, name, batch_size=5)) == by_name[name]

    def test_unknown_sheet_raises_key_error(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv", 3)
        with pytest.raises(KeyError):
            list(iter_sheet_batches(p, "nope"))

    def test_file_is_closed_when_the_iterator_is_abandoned(self, tmp_path):
        p = _write_csv(tmp_path / "f.csv", 2000)
        it = iter_sheet_batches(p, "f", batch_size=10)
        next(it)
        it.close()  # must not raise, and must release the file handle


# ---------------------------------------------------------------------------
# 2. LangGraph state carries no rows
# ---------------------------------------------------------------------------


class TestLoadSheetsKeepsNoRows:
    async def test_state_sheets_have_metadata_only(self, tmp_path, monkeypatch):
        from scinr.newton.tabular.nodes import load_sheets

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr("scinr.newton.annotation.catalog.ensure_catalog_models_once", _noop)
        monkeypatch.setattr("scinr.newton.annotation.catalog.ensure_theme_structure_once", _noop)
        monkeypatch.setattr("scinr.newton.utils.theme_registry.get_theme_registry", lambda: object())

        p = _write_csv(tmp_path / "big.csv", 3000)
        state = {"file_path": str(p), "document_name": "big", "doc_path": "big", "errors": []}

        out = await load_sheets(state)

        (sheet,) = out["sheets"]
        assert "all_rows" not in sheet
        assert sheet["file_path"] == str(p)
        assert sheet["sheet_name"] == "big"
        assert sheet["total_rows"] == 3000
        assert sheet["headers"] == ["col0", "col1", "col2"]
        assert "| v0_0 |" in sheet["preview_markdown"]
        assert "showing 5 of 3000 rows" in sheet["preview_markdown"]


    @pytest.fixture
    def _no_prewarm(self, monkeypatch):
        async def _noop(*a, **k):
            return None

        monkeypatch.setattr("scinr.newton.annotation.catalog.ensure_catalog_models_once", _noop)
        monkeypatch.setattr("scinr.newton.annotation.catalog.ensure_theme_structure_once", _noop)
        monkeypatch.setattr("scinr.newton.utils.theme_registry.get_theme_registry", lambda: object())

    async def test_prefetched_scans_are_reused_without_reading_the_file(
        self, tmp_path, monkeypatch, _no_prewarm
    ):
        from scinr.newton.tabular import nodes
        from scinr.newton.tabular.reader import scan_tabular_file

        p = _write_csv(tmp_path / "small.csv", 3)
        scans = scan_tabular_file(p)

        def _fail(*a, **k):
            raise AssertionError("scan_tabular_file must not be called with prefetched scans")

        monkeypatch.setattr(nodes, "scan_tabular_file", _fail)
        state = {
            "file_path": str(p), "document_name": "small", "doc_path": "small",
            "errors": [], "prefetched_scans": scans,
        }

        out = await nodes.load_sheets(state)

        (sheet,) = out["sheets"]
        assert sheet["total_rows"] == 3
        assert sheet["headers"] == ["col0", "col1", "col2"]

    async def test_delimiter_reaches_scan(self, tmp_path, _no_prewarm):
        from scinr.newton.tabular.nodes import load_sheets

        # Comma-free semicolon file whose sniffing would be ambiguous: force ";".
        p = tmp_path / "semi.csv"
        p.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
        state = {
            "file_path": str(p), "document_name": "semi", "doc_path": "semi",
            "errors": [], "delimiter": ";",
        }

        out = await load_sheets(state)

        assert out["sheets"][0]["headers"] == ["a", "b"]
        assert out["sheets"][0]["total_rows"] == 2


class TestWriteTabularDelimiter:
    async def test_delimiter_reaches_iter_sheet_batches(self, monkeypatch):
        from scinr.newton.tabular import nodes

        seen: list = []

        def fake_iter(path, sheet_name, batch_size=500, delimiter=None):
            seen.append(delimiter)
            return iter(())

        async def fake_write(**kwargs):
            list(kwargs["row_batches"](10))
            return "table-1"

        monkeypatch.setattr(nodes, "iter_sheet_batches", fake_iter)
        monkeypatch.setattr(
            "scinr.newton.tabular.graph_writer.write_tabular_subgraph", fake_write
        )
        state = {
            "current_sheet": {"file_path": "x.csv", "sheet_name": "x"},
            "current_decision": object(),
            "current_mapping": object(),
            "current_sheet_index": 0,
            "doc_path": "x", "document_name": "x", "resolved_version": 1,
            "tenant_id": "", "delimiter": "|", "errors": [],
            "ingested_table_node_ids": [],
        }

        out = await nodes.write_tabular(state)

        assert seen == ["|"]
        assert out["ingested_table_node_ids"] == ["table-1"]
