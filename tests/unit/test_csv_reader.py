"""
tests/unit/test_csv_reader.py — Unit tests for scinr.newton.tabular.reader

Imports directly from the submodule to avoid triggering the CLI import chain.
"""
from __future__ import annotations

import pytest

from scinr.newton.exceptions import ConversionError

# Import directly from the submodule — NOT from scinr.newton (top-level)
from scinr.newton.tabular.reader import (
    read_csv,
    read_tabular_file,
    read_xlsx,
)


class TestReadCsvCommaDelimiter:
    def test_read_csv_comma_delimiter(self, tmp_path):
        """CSV with commas is parsed correctly."""
        p = tmp_path / "comma.csv"
        p.write_text("name,age,city\nAlice,30,Madrid\nBob,25,Barcelona\n", encoding="utf-8")

        sheets = read_csv(p)
        assert len(sheets) == 1
        sheet = sheets[0]
        assert sheet["headers"] == ["name", "age", "city"]
        assert sheet["total_rows"] == 2
        assert sheet["all_rows"][0] == ["Alice", "30", "Madrid"]
        assert sheet["all_rows"][1] == ["Bob", "25", "Barcelona"]

    def test_read_csv_semicolon_delimiter(self, tmp_path):
        """CSV with semicolons is auto-detected and parsed correctly."""
        p = tmp_path / "semi.csv"
        p.write_text("name;age;city\nAlice;30;Madrid\nBob;25;Barcelona\n", encoding="utf-8")

        sheets = read_csv(p)
        assert len(sheets) == 1
        sheet = sheets[0]
        assert sheet["headers"] == ["name", "age", "city"]
        assert sheet["total_rows"] == 2
        assert sheet["all_rows"][0] == ["Alice", "30", "Madrid"]

    def test_read_csv_tab_delimiter(self, tmp_path):
        """CSV with tab separators is auto-detected and parsed correctly."""
        p = tmp_path / "tab.csv"
        p.write_text("name\tage\tcity\nAlice\t30\tMadrid\n", encoding="utf-8")

        sheets = read_csv(p)
        assert len(sheets) == 1
        sheet = sheets[0]
        assert sheet["headers"] == ["name", "age", "city"]
        assert sheet["total_rows"] == 1
        assert sheet["all_rows"][0] == ["Alice", "30", "Madrid"]

    def test_read_csv_bom(self, tmp_path):
        """CSV with UTF-8 BOM: headers must not start with the BOM character."""
        p = tmp_path / "bom.csv"
        # Write with BOM: utf-8-sig encoding adds the BOM automatically
        p.write_bytes("name,age\nAlice,30\n".encode("utf-8-sig"))

        sheets = read_csv(p)
        assert len(sheets) == 1
        headers = sheets[0]["headers"]
        assert headers[0] == "name", f"Expected 'name', got {headers[0]!r}"
        assert not headers[0].startswith("\ufeff")

    def test_read_csv_duplicate_headers(self, tmp_path):
        """Duplicate headers are deduplicated with _N suffixes."""
        p = tmp_path / "dupes.csv"
        p.write_text("a,b,a,c\n1,2,3,4\n", encoding="utf-8")

        sheets = read_csv(p)
        assert len(sheets) == 1
        headers = sheets[0]["headers"]
        assert headers == ["a", "b", "a_2", "c"]

    def test_read_csv_empty_file(self, tmp_path):
        """Empty CSV returns a sheet with no headers and no rows."""
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")

        sheets = read_csv(p)
        assert len(sheets) == 1
        assert sheets[0]["headers"] == []
        assert sheets[0]["total_rows"] == 0

    def test_read_csv_single_row(self, tmp_path):
        """CSV with only a header row returns zero data rows."""
        p = tmp_path / "header_only.csv"
        p.write_text("col1,col2,col3\n", encoding="utf-8")

        sheets = read_csv(p)
        assert sheets[0]["total_rows"] == 0
        assert sheets[0]["headers"] == ["col1", "col2", "col3"]

    def test_read_csv_sheet_name_is_stem(self, tmp_path):
        """sheet_name is set to the file stem."""
        p = tmp_path / "mydata.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")

        sheets = read_csv(p)
        assert sheets[0]["sheet_name"] == "mydata"

    def test_read_csv_skips_empty_rows(self, tmp_path):
        """Empty rows in the CSV body are skipped."""
        p = tmp_path / "gaps.csv"
        p.write_text("a,b\n1,2\n\n3,4\n", encoding="utf-8")

        sheets = read_csv(p)
        assert sheets[0]["total_rows"] == 2

    def test_read_csv_pads_short_rows(self, tmp_path):
        """Rows shorter than the header count are padded with empty strings."""
        p = tmp_path / "short.csv"
        p.write_text("a,b,c\n1,2\n", encoding="utf-8")

        sheets = read_csv(p)
        assert sheets[0]["all_rows"][0] == ["1", "2", ""]


class TestReadXlsx:
    def test_read_xlsx_xls_raises(self, tmp_path):
        """Passing a .xls file to read_xlsx raises ConversionError mentioning .xls."""
        p = tmp_path / "legacy.xls"
        p.write_bytes(b"fake xls content")

        with pytest.raises(ConversionError, match=r"\.xls"):
            read_xlsx(p)

    def test_read_tabular_file_xls_raises(self, tmp_path):
        """read_tabular_file with .xls extension raises ConversionError."""
        p = tmp_path / "legacy.xls"
        p.write_bytes(b"fake xls content")

        with pytest.raises(ConversionError, match=r"\.xls"):
            read_tabular_file(p)

    def test_read_tabular_file_unsupported_extension(self, tmp_path):
        """read_tabular_file with unsupported extension raises ValueError."""
        p = tmp_path / "file.xyz"
        p.write_text("data")

        with pytest.raises(ValueError, match="Unsupported"):
            read_tabular_file(p)

    def test_read_xlsx_basic(self, tmp_path):
        """read_xlsx parses a valid XLSX file correctly."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["name", "score"])
        ws.append(["Alice", 95])
        ws.append(["Bob", 87])
        p = tmp_path / "data.xlsx"
        wb.save(str(p))

        sheets = read_xlsx(p)
        assert len(sheets) >= 1
        sheet = sheets[0]
        assert sheet["headers"] == ["name", "score"]
        assert sheet["total_rows"] == 2
        assert sheet["all_rows"][0] == ["Alice", "95"]

    def test_read_xlsx_duplicate_headers(self, tmp_path):
        """XLSX with duplicate headers are deduplicated."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["col", "col", "other"])
        ws.append([1, 2, 3])
        p = tmp_path / "dupes.xlsx"
        wb.save(str(p))

        sheets = read_xlsx(p)
        assert sheets[0]["headers"] == ["col", "col_2", "other"]

    def test_read_tabular_file_dispatches_csv(self, tmp_path):
        """read_tabular_file dispatches to read_csv for .csv files."""
        p = tmp_path / "data.csv"
        p.write_text("x,y\n1,2\n", encoding="utf-8")

        sheets = read_tabular_file(p)
        assert sheets[0]["headers"] == ["x", "y"]

    def test_read_tabular_file_dispatches_xlsx(self, tmp_path):
        """read_tabular_file dispatches to read_xlsx for .xlsx files."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["a", "b"])
        ws.append([10, 20])
        p = tmp_path / "data.xlsx"
        wb.save(str(p))

        sheets = read_tabular_file(p)
        assert sheets[0]["headers"] == ["a", "b"]
