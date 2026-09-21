"""Peak LIVE memory (tracemalloc) of reading a 1M-row x 12-col CSV (~141 MB).

Usage: python scripts/memory_bench/bench_csv.py

Rows printed
------------
current read_csv     : scinr.newton.tabular.reader.read_csv (WP5 makes this row drop from ~12.7x to ~5x).
one-pass, list kept  : reference implementation of the WP5 fix (same output contract).
streaming 500/batch  : reference for the WP6 design (never holds more than one batch).
Measured before any change: 1780 MB (12.7x) | 741 MB (5.3x) | 1 MB.
"""

import csv
import tempfile
import time
import tracemalloc
from pathlib import Path

from scinr.newton.tabular.reader import _cell_to_str, _deduplicate_headers, read_csv

p = Path(tempfile.mkdtemp()) / "big.csv"
with open(p, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"col_{i}" for i in range(12)])
    for r in range(1_000_000):
        w.writerow([f"value_{r}_{c}" if c % 3 else str(r * c) for c in range(12)])
size = p.stat().st_size / 1024 / 1024


def read_csv_one_pass(
    path,
):  # cheap fix: same output contract (all_rows list), single pass, no StringIO, no intermediates
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        head = f.read(4096)
        f.seek(0)
        try:
            delim = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","
        rows, headers, ncols = [], None, 0
        for row in csv.reader(f, delimiter=delim):
            if not any(c.strip() for c in row):
                continue
            if headers is None:
                headers = _deduplicate_headers([_cell_to_str(h) for h in row], path.name)
                ncols = len(headers)
                continue
            r = [_cell_to_str(c) for c in row[:ncols]]
            if len(r) < ncols:
                r += [""] * (ncols - len(r))
            rows.append(r)
    return [
        {
            "sheet_name": path.stem,
            "headers": headers or [],
            "all_rows": rows,
            "total_rows": len(rows),
        }
    ]


def batches(path, n=500):  # streaming: never holds more than one batch
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        batch = []
        for row in rd:
            batch.append(row)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch


for name, fn in (
    ("current read_csv", lambda: read_csv(p)),
    ("one-pass, list kept", lambda: read_csv_one_pass(p)),
    ("streaming 500/batch", lambda: sum(len(b) for b in batches(p))),
):
    tracemalloc.start()
    t = time.time()
    out = fn()
    peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    print(
        f"{name:22} file={size:.0f} MB  peak live memory = {peak:6.0f} MB ({peak / size:5.1f}x)  {time.time() - t:.1f}s"
    )
    del out
