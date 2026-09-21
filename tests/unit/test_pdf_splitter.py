"""
tests/unit/test_pdf_splitter.py — Unit tests for scinr.newton.converters.pdf_splitter

Pure logic, no network. Generates synthetic PDFs with pypdf to exercise
count_pdf_pages / needs_splitting / split_pdf.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from scinr.newton.converters.base import ConversionError
from scinr.newton.converters.pdf_splitter import (
    PdfChunk,
    PdfSplitError,
    count_pdf_pages,
    initial_window_pages,
    iter_pdf_chunks,
    needs_splitting,
    probe_pdf,
    split_pdf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(num_pages: int) -> bytes:
    """Build a minimal synthetic PDF with *num_pages* blank pages."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# count_pdf_pages
# ---------------------------------------------------------------------------


class TestCountPdfPages:
    def test_counts_pages_correctly(self):
        pdf_bytes = _make_pdf(7)
        assert count_pdf_pages(pdf_bytes) == 7

    def test_corrupt_pdf_raises_conversion_error(self):
        """A garbage byte string must not leak pypdf's native exception."""
        with pytest.raises(ConversionError):
            count_pdf_pages(b"not a pdf")


# ---------------------------------------------------------------------------
# needs_splitting
# ---------------------------------------------------------------------------


class TestNeedsSplitting:
    def test_false_when_within_both_limits(self):
        pdf_bytes = _make_pdf(3)
        assert needs_splitting(pdf_bytes, max_pages=10, max_bytes=1_000_000) is False

    def test_true_when_exceeds_page_limit(self):
        pdf_bytes = _make_pdf(10)
        assert needs_splitting(pdf_bytes, max_pages=5, max_bytes=1_000_000) is True

    def test_true_when_exceeds_byte_limit(self):
        pdf_bytes = _make_pdf(10)
        assert needs_splitting(pdf_bytes, max_pages=1000, max_bytes=10) is True


# ---------------------------------------------------------------------------
# split_pdf
# ---------------------------------------------------------------------------


class TestSplitPdf:
    def test_single_chunk_when_within_limits(self):
        pdf_bytes = _make_pdf(4)
        chunks = split_pdf(pdf_bytes, max_pages=10, max_bytes=1_000_000)
        assert len(chunks) == 1
        assert isinstance(chunks[0], PdfChunk)
        assert chunks[0].start_page == 0
        assert chunks[0].end_page == 4

    def test_exact_boundary_does_not_split(self):
        """A PDF with exactly max_pages pages must not be split."""
        pdf_bytes = _make_pdf(5)
        chunks = split_pdf(pdf_bytes, max_pages=5, max_bytes=1_000_000)
        assert len(chunks) == 1
        assert chunks[0].start_page == 0
        assert chunks[0].end_page == 5

    def test_splits_into_two_contiguous_chunks_by_page_count(self):
        pdf_bytes = _make_pdf(10)
        chunks = split_pdf(pdf_bytes, max_pages=5, max_bytes=10_000_000)
        assert len(chunks) == 2
        assert chunks[0].start_page == 0
        assert chunks[0].end_page == 5
        assert chunks[1].start_page == 5
        assert chunks[1].end_page == 10
        # No gaps, no overlaps.
        assert chunks[0].end_page == chunks[1].start_page
        assert chunks[-1].end_page == 10

    def test_bisects_by_byte_size(self):
        """A small max_bytes forces recursive bisection even though the
        page count is well within max_pages."""
        pdf_bytes = _make_pdf(20)
        # Empirically, ~10 blank pages serialize to ~1.5KB — force at
        # least one bisection level with a byte budget below that.
        chunks = split_pdf(pdf_bytes, max_pages=1000, max_bytes=1500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.pdf_bytes) <= 1500
        # Coverage: contiguous, no gaps/overlap, covers [0, 20).
        chunks_sorted = sorted(chunks, key=lambda c: c.start_page)
        assert chunks_sorted[0].start_page == 0
        assert chunks_sorted[-1].end_page == 20
        for prev, nxt in zip(chunks_sorted, chunks_sorted[1:], strict=False):
            assert prev.end_page == nxt.start_page

    def test_single_page_exceeding_max_bytes_raises(self):
        pdf_bytes = _make_pdf(3)
        with pytest.raises(PdfSplitError) as exc_info:
            split_pdf(pdf_bytes, max_pages=3, max_bytes=100, source_name="doc.pdf")
        message = str(exc_info.value)
        # Must mention the absolute 0-based page number that failed.
        assert "página 0" in message or "pagina 0" in message
        assert "doc.pdf" in message

    def test_pdf_split_error_is_a_conversion_error(self):
        assert issubclass(PdfSplitError, ConversionError)

    def test_encrypted_pdf_raises_conversion_error_not_native_exception(self):
        """A password-protected PDF may construct PdfReader without error but
        fail later (e.g. pypdf.errors.FileNotDecryptedError) when accessing
        `len(reader.pages)` or individual pages. split_pdf() must not leak
        that native pypdf exception — it must be wrapped as ConversionError,
        same contract as count_pdf_pages()."""
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=200, height=200)
        writer.encrypt("some-password")
        buf = BytesIO()
        writer.write(buf)
        encrypted_bytes = buf.getvalue()

        with pytest.raises(ConversionError) as exc_info:
            split_pdf(encrypted_bytes, max_pages=10, max_bytes=1_000_000, source_name="secret.pdf")

        # Must be a plain ConversionError wrapping, not an unrelated
        # PdfSplitError (that is reserved for the "1 page too big" case).
        assert not isinstance(exc_info.value, PdfSplitError)
        assert "secret.pdf" in str(exc_info.value)


# ---------------------------------------------------------------------------
# probe_pdf / iter_pdf_chunks (path-based, lazy)
# ---------------------------------------------------------------------------


def _write_pdf(tmp_path: Path, num_pages: int, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(_make_pdf(num_pages))
    return path


def _encrypted_pdf_bytes(num_pages: int = 3) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    writer.encrypt("some-password")
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _page_count(pdf_bytes: bytes) -> int:
    return count_pdf_pages(pdf_bytes)


class TestProbePdf:
    def test_returns_size_and_page_count(self, tmp_path):
        path = _write_pdf(tmp_path, 7)
        size, pages = probe_pdf(path)
        assert size == path.stat().st_size
        assert pages == 7

    def test_corrupt_pdf_raises_conversion_error(self, tmp_path):
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"not a pdf")
        with pytest.raises(ConversionError):
            probe_pdf(path)

    def test_encrypted_pdf_raises_conversion_error(self, tmp_path):
        path = tmp_path / "secret.pdf"
        path.write_bytes(_encrypted_pdf_bytes())
        with pytest.raises(ConversionError):
            probe_pdf(path)

    def test_missing_file_raises_conversion_error(self, tmp_path):
        with pytest.raises(ConversionError, match="Cannot read PDF file"):
            probe_pdf(tmp_path / "nope.pdf")


class TestInitialWindowPages:
    def test_capped_by_max_pages_when_pages_are_small(self):
        assert initial_window_pages(1_000, 100, max_pages=900, max_bytes=45_000_000) == 900

    def test_shrinks_with_average_page_size(self):
        # 1 MB per page, 45 MB limit -> 80 % of 45 pages = 36
        assert initial_window_pages(100_000_000, 100, max_pages=900, max_bytes=45_000_000) == 36

    def test_never_below_one(self):
        assert initial_window_pages(10**9, 2, max_pages=900, max_bytes=1_000) == 1

    def test_degenerate_inputs_fall_back_to_max_pages(self):
        assert initial_window_pages(0, 0, max_pages=50, max_bytes=1_000) == 50


class TestIterPdfChunksParity:
    @pytest.mark.parametrize(
        ("num_pages", "max_pages", "max_bytes"),
        [
            (1, 900, 10_000_000),
            (5, 2, 10_000_000),
            (20, 5, 10_000_000),
            (20, 20, 10_000_000),
            (13, 4, 10_000_000),
        ],
    )
    def test_same_ranges_and_page_counts_as_split_pdf(
        self, tmp_path, num_pages, max_pages, max_bytes
    ):
        path = _write_pdf(tmp_path, num_pages)
        expected = split_pdf(path.read_bytes(), max_pages, max_bytes, source_name="doc.pdf")

        got = list(iter_pdf_chunks(path, max_pages, max_bytes, source_name="doc.pdf"))

        assert [(c.start_page, c.end_page) for c in got] == [
            (c.start_page, c.end_page) for c in expected
        ]
        assert [_page_count(c.pdf_bytes) for c in got] == [
            _page_count(c.pdf_bytes) for c in expected
        ]

    def test_chunks_cover_the_document_and_respect_byte_limit_when_bisecting(self, tmp_path):
        path = _write_pdf(tmp_path, 20)
        max_bytes = len(_make_pdf(3))  # forces windows of a few pages at most

        chunks = list(iter_pdf_chunks(path, 20, max_bytes, source_name="doc.pdf"))

        assert len(chunks) > 1
        assert chunks[0].start_page == 0
        assert chunks[-1].end_page == 20
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            assert prev.end_page == nxt.start_page
        for c in chunks:
            assert len(c.pdf_bytes) <= max_bytes
            assert _page_count(c.pdf_bytes) == c.page_count


class TestIterPdfChunksLaziness:
    def test_taking_first_chunk_serialises_only_the_first_window(self, tmp_path, monkeypatch):
        from scinr.newton.converters import pdf_splitter

        path = _write_pdf(tmp_path, 12)
        real = pdf_splitter._serialize_page_range
        calls: list[tuple[int, int]] = []

        def counting(reader, start, end):
            calls.append((start, end))
            return real(reader, start, end)

        monkeypatch.setattr(pdf_splitter, "_serialize_page_range", counting)

        gen = iter_pdf_chunks(path, 4, 10_000_000, source_name="doc.pdf")
        first = next(gen)

        assert (first.start_page, first.end_page) == (0, 4)
        assert calls == [(0, 4)]
        gen.close()
        assert calls == [(0, 4)]

    def test_uses_a_fresh_reader_per_window_and_never_reads_the_whole_file(
        self, tmp_path, monkeypatch
    ):
        from scinr.newton.converters import pdf_splitter

        path = _write_pdf(tmp_path, 9)
        opened: list[object] = []
        real_reader = pdf_splitter.PdfReader

        def tracking_reader(*args, **kwargs):
            reader = real_reader(*args, **kwargs)
            opened.append(reader)
            return reader

        def _boom(self):
            raise AssertionError("Path.read_bytes must not be called")

        monkeypatch.setattr(pdf_splitter, "PdfReader", tracking_reader)
        monkeypatch.setattr(Path, "read_bytes", _boom)

        chunks = list(iter_pdf_chunks(path, 3, 10_000_000, source_name="doc.pdf"))

        assert [(c.start_page, c.end_page) for c in chunks] == [(0, 3), (3, 6), (6, 9)]
        # 1 reader to count pages + 1 per window (3 windows), all distinct objects.
        assert len(opened) == 4
        assert len({id(r) for r in opened}) == 4


class TestIterPdfChunksErrors:
    def test_single_page_exceeding_max_bytes_raises_same_message_as_split_pdf(self, tmp_path):
        path = _write_pdf(tmp_path, 3)

        with pytest.raises(PdfSplitError) as lazy_exc:
            list(iter_pdf_chunks(path, max_pages=3, max_bytes=100, source_name="doc.pdf"))
        with pytest.raises(PdfSplitError) as eager_exc:
            split_pdf(path.read_bytes(), max_pages=3, max_bytes=100, source_name="doc.pdf")

        assert str(lazy_exc.value) == str(eager_exc.value)
        assert "página 0" in str(lazy_exc.value)

    def test_encrypted_pdf_raises_conversion_error_not_native_exception(self, tmp_path):
        path = tmp_path / "secret.pdf"
        path.write_bytes(_encrypted_pdf_bytes())

        with pytest.raises(ConversionError) as exc_info:
            list(iter_pdf_chunks(path, 10, 1_000_000, source_name="secret.pdf"))

        assert not isinstance(exc_info.value, PdfSplitError)
        assert "secret.pdf" in str(exc_info.value)

    def test_corrupt_pdf_raises_conversion_error(self, tmp_path):
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"not a pdf")

        with pytest.raises(ConversionError, match="Cannot split PDF bad.pdf"):
            list(iter_pdf_chunks(path, 10, 1_000_000, source_name="bad.pdf"))

    def test_split_pdf_error_can_surface_after_earlier_chunks_were_yielded(
        self, tmp_path, monkeypatch
    ):
        from scinr.newton.converters import pdf_splitter

        path = _write_pdf(tmp_path, 4)
        real = pdf_splitter._serialize_page_range

        def fat_third_page(reader, start, end):
            data = real(reader, start, end)
            return data + b"0" * 10_000 if start <= 2 < end else data

        monkeypatch.setattr(pdf_splitter, "_serialize_page_range", fat_third_page)
        max_bytes = len(_make_pdf(2)) + 1_000

        gen = iter_pdf_chunks(path, 2, max_bytes, source_name="doc.pdf")
        first = next(gen)

        assert (first.start_page, first.end_page) == (0, 2)
        with pytest.raises(PdfSplitError, match="página 2"):
            next(gen)
