"""
tests/unit/test_pdf_splitter.py — Unit tests for scinr.newton.converters.pdf_splitter

Pure logic, no network. Generates synthetic PDFs with pypdf to exercise
count_pdf_pages / needs_splitting / split_pdf.
"""
from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from scinr.newton.converters.base import ConversionError
from scinr.newton.converters.pdf_splitter import (
    PdfChunk,
    PdfSplitError,
    count_pdf_pages,
    needs_splitting,
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
