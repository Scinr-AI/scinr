"""
tests/unit/test_pdf_converter_chunking.py — Unit tests for PdfConverter chunking,
retry, and configurable error-strategy behaviour.

All httpx.AsyncClient.post calls are mocked — no real network access, no
real API key. PdfConverter.convert() is a native coroutine (is_async =
True), so every test in this module is an `async def` awaiting it directly.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pypdf import PdfWriter

from scinr.newton.converters.base import ConversionError
from scinr.newton.converters.pdf import PdfConverter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _write_pdf(tmp_path: Path, num_pages: int, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(_make_pdf(num_pages))
    return path


class _FakeResponse:
    """Minimal stand-in for httpx.Response used by the converter."""

    def __init__(self, status_code: int, text: str = "", json_data: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data if json_data is not None else {}

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self) -> dict:
        return self._json_data


def _ok_response(page_count: int) -> _FakeResponse:
    """Build a successful Mistral OCR response with *page_count* pages,
    each with index resetting at 0 (as Mistral would for an independent
    chunk request)."""
    pages = [
        {"index": i, "markdown": f"page-{i}", "images": [], "dimensions": {}}
        for i in range(page_count)
    ]
    return _FakeResponse(200, json_data={"pages": pages})


def _side_effect_from(responses: list):
    it = iter(responses)

    def _fn(*args, **kwargs):
        return next(it)

    return _fn


# ---------------------------------------------------------------------------
# 9. No-regression: small PDF, single call, exact legacy error message
# ---------------------------------------------------------------------------


class TestNoSplitBehaviour:
    async def test_small_pdf_single_call_and_correct_mapping(self, tmp_path, mocker):
        source = _write_pdf(tmp_path, num_pages=2)
        mock_post = mocker.patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_ok_response(2)
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=900,
            safe_max_bytes=45 * 1024 * 1024,
        )
        doc = await converter.convert(source)

        assert mock_post.call_count == 1
        assert [p.index for p in doc.pages] == [0, 1]
        assert doc.pages[0].markdown == "page-0"
        assert doc.pages[1].markdown == "page-1"
        assert doc.missing_page_ranges is None

    async def test_small_pdf_http_error_message_is_identical_to_legacy_format(self, tmp_path, mocker):
        """No 'chunk X/Y' prefix must leak into the error message when the
        document did not need splitting."""
        source = _write_pdf(tmp_path, num_pages=2)
        mocker.patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_FakeResponse(400, text="bad request"),
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=900,
            safe_max_bytes=45 * 1024 * 1024,
        )
        with pytest.raises(ConversionError) as exc_info:
            await converter.convert(source)

        assert str(exc_info.value) == "Mistral OCR API returned HTTP 400: bad request"

    async def test_map_page_index_offset_defaults_to_zero(self):
        """_map_page(page_data) without index_offset preserves the raw index —
        explicit backward-compatibility check on the method signature."""
        converter = PdfConverter(api_key="dummy-test-key")
        page = converter._map_page({"index": 3, "markdown": "x"})
        assert page.index == 3


# ---------------------------------------------------------------------------
# 10. Chunking + reassembly across 3 chunks
# ---------------------------------------------------------------------------


class TestChunkingAndReassembly:
    async def test_three_chunks_reassembled_with_continuous_index(self, tmp_path, mocker):
        source = _write_pdf(tmp_path, num_pages=5)
        # max_pages=2 over 5 pages -> chunks of sizes [2, 2, 1]
        mock_post = mocker.patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_side_effect_from(
                [_ok_response(2), _ok_response(2), _ok_response(1)]
            ),
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=2,
            safe_max_bytes=10_000_000,
            error_strategy="fail_fast",
        )
        doc = await converter.convert(source)

        assert mock_post.call_count == 3
        assert [p.index for p in doc.pages] == [0, 1, 2, 3, 4]
        assert doc.missing_page_ranges is None


# ---------------------------------------------------------------------------
# 11. fail_fast: chunk 2/3 fails after exhausting retries
# ---------------------------------------------------------------------------


class TestFailFastStrategy:
    async def test_fail_fast_raises_and_skips_remaining_chunks(self, tmp_path, mocker):
        source = _write_pdf(tmp_path, num_pages=5)
        # chunk1 success, chunk2: 3x HTTP 503 (exhausts default max_retries=15),
        # chunk3 must never be requested.
        mock_post = mocker.patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_side_effect_from(
                [
                    _ok_response(2),
                    _FakeResponse(503, text="unavailable"),
                    _FakeResponse(503, text="unavailable"),
                    _FakeResponse(503, text="unavailable"),
                ]
            ),
        )
        mocker.patch(
            "scinr.newton.converters.pdf.asyncio.sleep", new_callable=AsyncMock, return_value=None
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=2,
            safe_max_bytes=10_000_000,
            error_strategy="fail_fast",
        )
        with pytest.raises(ConversionError) as exc_info:
            await converter.convert(source)

        message = str(exc_info.value)
        assert "[2, 4)" in message
        assert "best_effort" in message
        assert "MISTRAL_OCR_ERROR_STRATEGY" in message
        # Only chunk1 (1 call) + chunk2 (3 retries) — chunk3 was never attempted.
        assert mock_post.call_count == 4


# ---------------------------------------------------------------------------
# 12. best_effort: chunk 2/3 fails, chunk 3 still processed, no exception
# ---------------------------------------------------------------------------


class TestBestEffortStrategy:
    async def test_best_effort_skips_failed_chunk_and_continues(self, tmp_path, mocker):
        source = _write_pdf(tmp_path, num_pages=5)
        mock_post = mocker.patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_side_effect_from(
                [
                    _ok_response(2),
                    _FakeResponse(503, text="unavailable"),
                    _FakeResponse(503, text="unavailable"),
                    _FakeResponse(503, text="unavailable"),
                    _ok_response(1),
                ]
            ),
        )
        mocker.patch(
            "scinr.newton.converters.pdf.asyncio.sleep", new_callable=AsyncMock, return_value=None
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=2,
            safe_max_bytes=10_000_000,
            error_strategy="best_effort",
        )
        doc = await converter.convert(source)

        # chunk1 (pages 0-1) + chunk3 (page 4, remapped with offset -> index 4)
        assert [p.index for p in doc.pages] == [0, 1, 4]
        assert doc.missing_page_ranges == [(2, 4)]
        # chunk1 (1) + chunk2 (3 retries) + chunk3 (1) == 5 calls total.
        assert mock_post.call_count == 5


# ---------------------------------------------------------------------------
# 13. Non-retryable HTTP error: no retries attempted
# ---------------------------------------------------------------------------


class TestNonRetryableError:
    async def test_http_400_does_not_retry(self, tmp_path, mocker):
        source = _write_pdf(tmp_path, num_pages=5)
        mock_post = mocker.patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_side_effect_from(
                [_FakeResponse(400, text="bad request")]
            ),
        )
        mocker.patch(
            "scinr.newton.converters.pdf.asyncio.sleep", new_callable=AsyncMock, return_value=None
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=2,
            safe_max_bytes=10_000_000,
            error_strategy="fail_fast",
        )
        with pytest.raises(ConversionError):
            await converter.convert(source)

        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# 14. Transient failure recovers on a later attempt
# ---------------------------------------------------------------------------


class TestTransientRecovery:
    async def test_recovers_after_two_transient_failures(self, tmp_path, mocker):
        source = _write_pdf(tmp_path, num_pages=2)
        mock_post = mocker.patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_side_effect_from(
                [
                    _FakeResponse(503, text="unavailable"),
                    _FakeResponse(503, text="unavailable"),
                    _ok_response(2),
                ]
            ),
        )
        mocker.patch(
            "scinr.newton.converters.pdf.asyncio.sleep", new_callable=AsyncMock, return_value=None
        )

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=900,
            safe_max_bytes=45 * 1024 * 1024,
            max_retries=3,
        )
        doc = await converter.convert(source)

        assert mock_post.call_count == 3
        assert [p.index for p in doc.pages] == [0, 1]
        assert doc.missing_page_ranges is None


# ---------------------------------------------------------------------------
# 15. Invalid error_strategy override must raise, not warn+fallback
# ---------------------------------------------------------------------------


class TestInvalidErrorStrategyOverride:
    async def test_invalid_error_strategy_constructor_override_raises(self, tmp_path, mocker):
        """An invalid error_strategy passed explicitly to the PdfConverter
        constructor must be treated as a hard configuration error
        (ConversionError), consistent with configure()'s ConfigurationError
        for the same invalid value — not a silent warning+fallback to
        'fail_fast'."""
        source = _write_pdf(tmp_path, num_pages=2)
        # httpx.AsyncClient.post must never be called — the error is raised
        # before any network I/O, while resolving limits.
        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)

        converter = PdfConverter(
            api_key="dummy-test-key",
            error_strategy="not_a_real_strategy",
        )
        with pytest.raises(ConversionError, match="mistral_ocr_error_strategy|error_strategy"):
            await converter.convert(source)

        assert mock_post.call_count == 0
