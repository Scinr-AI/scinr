"""tabular/reader.py — Read CSV and XLSX files, either whole or as a stream of row batches.

Two APIs share the same sanitising rules (empty rows skipped, first non-empty
row = headers, data rows trimmed / padded to the header width, cells stripped):

* ``read_csv`` / ``read_xlsx`` / ``read_tabular_file`` materialise every row in
  ``all_rows`` (kept for callers that need the whole sheet in memory).
* ``scan_tabular_file`` + ``iter_sheet_batches`` never hold more than one batch:
  the scan collects headers, the row count and a 5-row preview; the batches are
  then streamed on demand. The file must not change between those passes.
"""
from __future__ import annotations

import csv as csv_module
import io
import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
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


class SheetScan(TypedDict):
    """Result of :func:`scan_tabular_file` for one sheet — no data rows attached."""

    sheet_name: str
    headers: list[str]
    total_rows: int           # number of data rows (headers and empty rows excluded)
    preview: TabularPreview   # up to 5 representative rows, see select_preview_rows()


_PREVIEW_MAX_ROWS = 5


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


def _sniff_delimiter(f: io.TextIOBase, name: str) -> str:
    """Detect the CSV delimiter from the first 4096 chars of *f*, then rewind *f* to the start.

    Falls back to a comma when :class:`csv.Sniffer` cannot decide.
    """
    sample = f.read(4096)
    f.seek(0)
    try:
        delimiter = csv_module.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        log.debug("CSV '%s': detected delimiter=%r", name, delimiter)
    except csv_module.Error:
        delimiter = ","
        log.debug("CSV '%s': could not detect delimiter, falling back to comma", name)
    return delimiter


# A sheet source: (sheet_name, factory). Each call of the factory starts a NEW pass
# over that sheet and yields its raw rows as lists of stripped strings.
_SheetSource = tuple[str, Callable[[], Iterator[list[str]]]]


def _iter_csv_rows(path: Path, delimiter: str | None) -> Iterator[list[str]]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        delim = delimiter or _sniff_delimiter(f, path.name)
        for row in csv_module.reader(f, delimiter=delim):
            yield [_cell_to_str(c) for c in row]


@contextmanager
def _open_csv(path: Path, delimiter: str | None = None) -> Iterator[list[_SheetSource]]:
    yield [(path.stem, lambda: _iter_csv_rows(path, delimiter))]


def _iter_xlsx_rows(ws) -> Iterator[list[str]]:
    for values in ws.iter_rows(values_only=True):
        yield [_cell_to_str(v) for v in values]


@contextmanager
def _open_xlsx(path: Path) -> Iterator[list[_SheetSource]]:
    """Open the workbook (read-only) for the duration of the ``with`` block; always closed."""
    if path.suffix.lower() == ".xls":
        from scinr.newton.exceptions import ConversionError
        raise ConversionError(
            f"'{path.name}' is in Excel 97-2003 format (.xls), which is not supported. "
            f"Open the file in Excel or LibreOffice and save as .xlsx, then retry."
        )
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        yield [(ws.title, lambda ws=ws: _iter_xlsx_rows(ws)) for ws in wb.worksheets]
    finally:
        wb.close()


def _open_sheets(path: Path, delimiter: str | None = None):
    """Dispatch on the file extension (same rules as :func:`read_tabular_file`)."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return _open_csv(path, delimiter)
    if ext in (".xlsx", ".xls"):
        return _open_xlsx(path)
    raise ValueError(f"Unsupported tabular file extension: {ext!r}")


def _empty_sheet_name(path: Path) -> str:
    """Name of the placeholder sheet returned when a file has no rows at all."""
    return path.stem if path.suffix.lower() == ".csv" else "Sheet1"


def _pop_header(rows: Iterator[list[str]]) -> list[str] | None:
    """Consume *rows* up to and including the first non-empty row, and return it (or ``None``)."""
    for row in rows:
        if any(c.strip() for c in row):
            return row
    return None


def _iter_data_rows(rows: Iterable[list[str]], ncols: int) -> Iterator[list[str]]:
    """Yield the non-empty rows of *rows*, trimmed / padded in place to *ncols* cells."""
    for row in rows:
        if not any(c.strip() for c in row):
            continue
        del row[ncols:]
        if len(row) < ncols:
            row.extend([""] * (ncols - len(row)))
        yield row


def _batched(rows: Iterable[list[str]], size: int) -> Iterator[list[list[str]]]:
    batch: list[list[str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _collect_sheet(
    rows: Iterator[list[str]], source_name: str
) -> tuple[list[str], list[list[str]]] | None:
    """Single pass over already-stringified *rows*: pick headers and collect all data rows.

    Returns ``None`` when there is no non-empty row at all.
    """
    raw_headers = _pop_header(rows)
    if raw_headers is None:
        return None
    headers = _deduplicate_headers(raw_headers, source_name)
    return headers, list(_iter_data_rows(rows, len(headers)))


def read_csv(path: Path) -> list[TabularSheet]:
    """Read a CSV file (UTF-8-BOM aware, errors='replace') and return a single-element list.

    The file encoding is ``utf-8-sig`` so that a leading BOM written by Windows
    tools is stripped automatically.  The column separator is auto-detected via
    :class:`csv.Sniffer` on the first 4096 characters; the fallback is a comma.

    The file is read in a single streaming pass (no full-text copy, no
    intermediate row lists), so peak memory is essentially the returned
    ``all_rows`` itself.

    Row 0 = headers. All subsequent rows = data rows.
    Empty rows are skipped. All cell values converted to str via _cell_to_str.
    Duplicate header names are deduplicated with numeric suffixes.
    """
    with _open_csv(path) as sheets:
        (name, rows), = sheets
        collected = _collect_sheet(rows(), name)

    if collected is None:
        return [{"sheet_name": path.stem, "headers": [], "all_rows": [], "total_rows": 0}]
    headers, data_rows = collected
    return [{"sheet_name": path.stem, "headers": headers, "all_rows": data_rows, "total_rows": len(data_rows)}]


def read_xlsx(path: Path) -> list[TabularSheet]:
    """Read an XLSX file (openpyxl, read_only=True, data_only=True).

    Returns one TabularSheet per worksheet. Empty worksheets are skipped.
    Duplicate header names are deduplicated with numeric suffixes.
    The workbook is always closed, even if reading a sheet fails.

    Raises
    ------
    ConversionError
        If the file has a ``.xls`` extension (Excel 97-2003 format), which is
        not supported by openpyxl.  The caller must convert it to ``.xlsx`` first.
    """
    sheets: list[TabularSheet] = []
    with _open_xlsx(path) as sources:
        for name, rows in sources:
            collected = _collect_sheet(rows(), name)
            if collected is None:
                continue
            headers, data_rows = collected
            sheets.append({"sheet_name": name, "headers": headers, "all_rows": data_rows, "total_rows": len(data_rows)})
    return sheets or [{"sheet_name": "Sheet1", "headers": [], "all_rows": [], "total_rows": 0}]


def read_tabular_file(path: Path) -> list[TabularSheet]:
    """Dispatch to read_csv or read_xlsx based on file extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return read_xlsx(path)
    raise ValueError(f"Unsupported tabular file extension: {ext!r}")


# ── Streaming API ─────────────────────────────────────────────────────────────


def preview_indices(n: int) -> list[int]:
    """Indices (0-based, within the data rows) of the preview rows for a sheet of *n* rows.

    - n <= 5: all rows
    - n > 5: row 0, ~25 %, ~50 %, ~75 % and the last row (deduplicated, sorted)
    """
    if n == 0:
        return []
    if n <= _PREVIEW_MAX_ROWS:
        return list(range(n))
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})


def _make_preview(
    sheet_name: str, headers: list[str], rows: list[list[str]], indices: list[int], total_rows: int
) -> TabularPreview:
    return {
        "sheet_name": sheet_name,
        "headers": headers,
        "preview_rows": rows,
        "row_indices": indices,
        "total_rows": total_rows,
    }


def scan_tabular_file(path: Path, delimiter: str | None = None) -> list[SheetScan]:
    """Pass A: headers, row count and preview of every sheet, without keeping the data rows.

    The row count has to be known before the preview rows (``n // 4``, ``n // 2``,
    ...) can be picked, so a sheet with more than 5 rows is read twice: once to
    count, once to fetch the preview rows. Memory stays O(1) in the number of rows.

    Same sheets and sanitising as :func:`read_tabular_file` (empty sheets are
    skipped; a file with no rows at all yields a single empty placeholder sheet).
    """
    scans: list[SheetScan] = []
    with _open_sheets(path, delimiter) as sources:
        for name, rows in sources:
            it = rows()
            try:
                raw_headers = _pop_header(it)
                if raw_headers is None:
                    continue
                headers = _deduplicate_headers(raw_headers, name)
                ncols = len(headers)
                total = 0
                head: list[list[str]] = []  # first rows: the whole preview when total <= 5
                for row in _iter_data_rows(it, ncols):
                    if total < _PREVIEW_MAX_ROWS:
                        head.append(row)
                    total += 1
            finally:
                it.close()

            indices = preview_indices(total)
            if total <= _PREVIEW_MAX_ROWS:
                preview_rows = head
            else:
                wanted = set(indices)
                preview_rows = []
                it = rows()
                try:
                    _pop_header(it)
                    for i, row in enumerate(_iter_data_rows(it, ncols)):
                        if i in wanted:
                            preview_rows.append(row)
                            if len(preview_rows) == len(indices):
                                break
                finally:
                    it.close()
            scans.append({
                "sheet_name": name,
                "headers": headers,
                "total_rows": total,
                "preview": _make_preview(name, headers, preview_rows, indices, total),
            })

    if scans:
        return scans
    empty_name = _empty_sheet_name(path)
    return [{
        "sheet_name": empty_name,
        "headers": [],
        "total_rows": 0,
        "preview": _make_preview(empty_name, [], [], [], 0),
    }]


def iter_sheet_batches(
    path: Path,
    sheet_name: str,
    batch_size: int = 500,
    delimiter: str | None = None,
) -> Iterator[list[list[str]]]:
    """Stream the data rows of *sheet_name* in batches of at most *batch_size* rows.

    Same rows, order and sanitising as ``read_tabular_file(path)[...]["all_rows"]``,
    but only one batch is alive at a time. Every call starts a new pass over the
    file, so the file must not be modified between passes. The file/workbook is
    closed when the iterator is exhausted or closed.

    Yields nothing when the sheet is empty; raises ``KeyError`` for an unknown sheet.
    """
    with _open_sheets(path, delimiter) as sources:
        by_name = dict(sources)
        if sheet_name not in by_name:
            raise KeyError(f"{path.name}: no sheet named {sheet_name!r}")
        it = by_name[sheet_name]()
        try:
            raw_headers = _pop_header(it)
            if raw_headers is None:
                return
            yield from _batched(_iter_data_rows(it, len(raw_headers)), batch_size)
        finally:
            it.close()


def select_preview_rows(sheet: TabularSheet) -> TabularPreview:
    """Select up to 5 representative rows from sheet.all_rows.

    - total_rows <= 5: use all rows (indices 0..total_rows-1)
    - total_rows > 5: row[0], rows at ~25%, ~50%, ~75%, and row[-1]
    Deduplicate and sort indices.
    """
    n = sheet["total_rows"]
    indices = preview_indices(n)
    return _make_preview(
        sheet["sheet_name"], sheet["headers"], [sheet["all_rows"][i] for i in indices], indices, n
    )


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
