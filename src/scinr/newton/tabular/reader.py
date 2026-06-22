"""tabular/reader.py — Read CSV and XLSX files into structured in-memory data."""
from __future__ import annotations

import csv as csv_module
import io
import logging
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)


class TabularSheet(TypedDict):
    sheet_name: str           # "Sheet1" for CSV; actual sheet name for XLSX
    headers: list[str]        # Column headers from row 0
    all_rows: list[list[str]] # All data rows (row 1+), each as list[str]
    total_rows: int           # len(all_rows)


class TabularPreview(TypedDict):
    sheet_name: str
    headers: list[str]
    preview_rows: list[list[str]]   # Up to 5 selected rows
    row_indices: list[int]          # 0-based indices of selected rows within all_rows
    total_rows: int


def _cell_to_str(value: object) -> str:
    """Convert a spreadsheet cell value to a clean string."""
    if value is None:
        return ""
    return str(value).strip()


def _deduplicate_headers(headers: list[str], source_name: str) -> list[str]:
    """Return a copy of *headers* with duplicate names made unique by appending _N suffixes.

    If any duplicates are found, a warning is emitted and the duplicate column
    names are listed.  The first occurrence is kept as-is; subsequent occurrences
    receive an incrementing integer suffix (e.g. ``col``, ``col_2``, ``col_3``).
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    has_duplicates = False
    for h in headers:
        if h in seen:
            has_duplicates = True
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 1
            result.append(h)
    if has_duplicates:
        dupes = [h for h in headers if headers.count(h) > 1]
        log.warning(
            "tabular '%s': duplicate headers detected: %s. "
            "Suffixes added to distinguish them.",
            source_name, sorted(set(dupes)),
        )
    return result


def read_csv(path: Path) -> list[TabularSheet]:
    """Read a CSV file (UTF-8-BOM aware, errors='replace') and return a single-element list.

    The file encoding is ``utf-8-sig`` so that a leading BOM written by Windows
    tools is stripped automatically.  The column separator is auto-detected via
    :class:`csv.Sniffer`; the fallback is a comma.

    Row 0 = headers. All subsequent rows = data rows.
    Empty rows are skipped. All cell values converted to str via _cell_to_str.
    Duplicate header names are deduplicated with numeric suffixes.
    """
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        raw_content = f.read()

    # Auto-detect delimiter
    sample = raw_content[:4096]
    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
        log.debug("CSV '%s': detected delimiter=%r", path.name, delimiter)
    except csv_module.Error:
        delimiter = ","
        log.debug("CSV '%s': could not detect delimiter, falling back to comma", path.name)

    reader = csv_module.reader(io.StringIO(raw_content), delimiter=delimiter)
    rows = [row for row in reader if any(c.strip() for c in row)]

    if not rows:
        return [{"sheet_name": path.stem, "headers": [], "all_rows": [], "total_rows": 0}]
    headers = _deduplicate_headers([_cell_to_str(h) for h in rows[0]], path.name)
    data_rows = [[_cell_to_str(c) for c in r] for r in rows[1:]]
    # Pad or trim data rows to match header count
    ncols = len(headers)
    data_rows = [(r + [""] * ncols)[:ncols] for r in data_rows]
    return [{"sheet_name": path.stem, "headers": headers, "all_rows": data_rows, "total_rows": len(data_rows)}]


def read_xlsx(path: Path) -> list[TabularSheet]:
    """Read an XLSX file (openpyxl, read_only=True, data_only=True).

    Returns one TabularSheet per worksheet. Empty worksheets are skipped.
    Duplicate header names are deduplicated with numeric suffixes.

    Raises
    ------
    ConversionError
        If the file has a ``.xls`` extension (Excel 97-2003 format), which is
        not supported by openpyxl.  The caller must convert it to ``.xlsx`` first.
    """
    if path.suffix.lower() == ".xls":
        from scinr.newton.exceptions import ConversionError
        raise ConversionError(
            f"'{path.name}' is in Excel 97-2003 format (.xls), which is not supported. "
            f"Open the file in Excel or LibreOffice and save as .xlsx, then retry."
        )
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for ws in wb.worksheets:
        rows = [[_cell_to_str(cell.value) for cell in row] for row in ws.iter_rows()]
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            continue
        headers = _deduplicate_headers(rows[0], ws.title)
        ncols = len(headers)
        data_rows = [(r + [""] * ncols)[:ncols] for r in rows[1:]]
        sheets.append({"sheet_name": ws.title, "headers": headers, "all_rows": data_rows, "total_rows": len(data_rows)})
    wb.close()
    return sheets or [{"sheet_name": "Sheet1", "headers": [], "all_rows": [], "total_rows": 0}]


def read_tabular_file(path: Path) -> list[TabularSheet]:
    """Dispatch to read_csv or read_xlsx based on file extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return read_xlsx(path)
    raise ValueError(f"Unsupported tabular file extension: {ext!r}")


def select_preview_rows(sheet: TabularSheet) -> TabularPreview:
    """Select up to 5 representative rows from sheet.all_rows.

    - total_rows <= 5: use all rows (indices 0..total_rows-1)
    - total_rows > 5: row[0], rows at ~25%, ~50%, ~75%, and row[-1]
    Deduplicate and sort indices.
    """
    n = sheet["total_rows"]
    if n == 0:
        return {
            "sheet_name": sheet["sheet_name"],
            "headers": sheet["headers"],
            "preview_rows": [],
            "row_indices": [],
            "total_rows": 0,
        }
    if n <= 5:
        indices = list(range(n))
    else:
        indices = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))
    preview_rows = [sheet["all_rows"][i] for i in indices]
    return {
        "sheet_name": sheet["sheet_name"],
        "headers": sheet["headers"],
        "preview_rows": preview_rows,
        "row_indices": indices,
        "total_rows": n,
    }


def row_to_markdown(headers: list[str], row: list[str]) -> str:
    """Render a single row as a 2-row GFM Markdown table (headers + separator + values).

    Used as InfoUnit.description for each Row node.
    """
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    value_line = "| " + " | ".join(row) + " |"
    return "\n".join([header_line, sep_line, value_line])


def preview_to_markdown(preview: TabularPreview) -> str:
    """Render the full preview as a GFM Markdown table (headers + selected rows).

    Used as LLM context in prompts.
    """
    if not preview["headers"]:
        return "(empty table)"
    header_line = "| " + " | ".join(preview["headers"]) + " |"
    sep_line = "| " + " | ".join("---" for _ in preview["headers"]) + " |"
    data_lines = ["| " + " | ".join(r) + " |" for r in preview["preview_rows"]]
    lines = [header_line, sep_line] + data_lines
    if preview["total_rows"] > len(preview["preview_rows"]):
        lines.append(f"*(showing {len(preview['preview_rows'])} of {preview['total_rows']} rows)*")
    return "\n".join(lines)
