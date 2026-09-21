"""Peak LIVE memory (tracemalloc; allocator independent) of the PDF split path.

Usage:
    python scripts/memory_bench/bench_pdf.py make <pages>      # writes a synthetic PDF (~0.4 MB/page) to the temp dir
    python scripts/memory_bench/bench_pdf.py <mode> <pages>

Modes
-----
current     : today's flow  (read_bytes x2 + split_pdf() -> list of every chunk).
gen_no_hint : lazy generator, single shared reader, window = max_pages.
generator   : lazy generator, single shared reader, size-aware first window.
gen_clear   : generator + reader cache cleared after each window.
gen_fresh   : generator + NEW PdfReader per window  <-- the design adopted in WP4.
real        : scinr.newton.converters.pdf_splitter.iter_pdf_chunks (exists once WP4 is implemented).

Measured (235 MB / 938 MB PDF): current 1458/4283 MB | gen_no_hint 490/1243 | generator 293/1016 |
gen_clear 293/548 | gen_fresh 226/328  (only gen_fresh stops growing with the file size).
Do NOT use RSS (ps / ru_maxrss) for this: on macOS freed large blocks stay in RSS.
"""

import os
import sys
import tempfile
import time
import tracemalloc
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

P = Path(tempfile.gettempdir()) / "scinr_bench_split.pdf"
MAXB = 45 * 1024 * 1024
pages = int(sys.argv[2]) if len(sys.argv) > 2 else 600
if sys.argv[1] == "make":
    w = PdfWriter()
    for _ in range(pages):
        pg = w.add_blank_page(612, 792)
        s = DecodedStreamObject()
        s.set_data(os.urandom(400 * 1024))
        pg[NameObject("/Contents")] = w._add_object(s)
    with P.open("wb") as f:
        w.write(f)
    sys.exit()


def ser(reader, a, b):
    w = PdfWriter()
    for i in range(a, b):
        w.add_page(reader.pages[i])
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


def bisect(reader, a, b):
    d = ser(reader, a, b)
    if len(d) <= MAXB:
        yield d
        return
    m = a + (b - a) // 2
    del d
    yield from bisect(reader, a, m)
    yield from bisect(reader, m, b)


def gen(path):
    size = path.stat().st_size
    with path.open("rb") as fh:
        r = PdfReader(fh)
        total = len(r.pages)
        win = max(1, min(900, int(0.8 * MAXB / (size / total))))  # size-aware window
        for s in range(0, total, win):
            yield from bisect(r, s, min(s + win, total))


def gen_no_size_hint(path):  # same generator but window=900 like today
    with path.open("rb") as fh:
        r = PdfReader(fh)
        total = len(r.pages)
        for s in range(0, total, 900):
            yield from bisect(r, s, min(s + 900, total))


def _win(path, total):
    size = path.stat().st_size
    return max(1, min(900, int(0.8 * MAXB / (size / total))))


def gen_clear(path):  # one reader, cache dropped after each window
    with path.open("rb") as fh:
        r = PdfReader(fh)
        total = len(r.pages)
        win = _win(path, total)
        for s in range(0, total, win):
            yield from bisect(r, s, min(s + win, total))
            r.resolved_objects.clear()


def gen_fresh(path):  # new reader per window, dropped when the window is done
    with path.open("rb") as fh:
        total = len(PdfReader(fh).pages)
    win = _win(path, total)
    for s in range(0, total, win):
        with path.open("rb") as fh:
            r = PdfReader(fh)
            yield from bisect(r, s, min(s + win, total))
        del r


mode = sys.argv[1]
size_mb = P.stat().st_size / 1024 / 1024
# Import scinr BEFORE tracing starts: its import-time allocations (~100 MB) are not part of the split path.
if mode in ("current", "real"):
    from scinr.newton.converters.pdf_splitter import iter_pdf_chunks, split_pdf  # noqa: F401
tracemalloc.start()
t = time.time()
n = 0
if mode == "current":  # main.py read + pdf.py read + list of all chunks
    raw = P.read_bytes()
    pdf_bytes = P.read_bytes()
    tracemalloc.reset_peak()
    chunks = split_pdf(pdf_bytes, 900, MAXB, source_name="x")
    n = len(chunks)
else:
    if mode == "real":
        for _c in iter_pdf_chunks(P, 900, MAXB, source_name="bench"):
            n += 1
    else:
        variants = {
            "generator": gen,
            "gen_no_hint": gen_no_size_hint,
            "gen_clear": gen_clear,
            "gen_fresh": gen_fresh,
        }
        for _d in variants[mode](P):
            n += 1
peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
print(
    f"{mode:11} file={size_mb:4.0f} MB  chunks={n:3}  peak live memory = {peak:6.0f} MB  ({peak / size_mb:.2f}x file)  {time.time() - t:.1f}s"
)
