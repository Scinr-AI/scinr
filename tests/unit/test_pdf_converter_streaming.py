"""
tests/unit/test_pdf_converter_streaming.py — Streamed OCR request body and the
semaphore bounding concurrent PDF conversions.

No real network access: requests go either to a mocked httpx.AsyncClient.post or
to an httpx.MockTransport, which sees the request exactly as it would be sent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tracemalloc
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from pypdf import PdfReader, PdfWriter

from scinr.newton.converters import pdf as pdf_module
from scinr.newton.converters.base import ConversionError
from scinr.newton.converters.pdf import PdfConverter, _OcrRequestBody

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


async def _read_body(body: _OcrRequestBody) -> bytes:
    return b"".join([part async for part in body])


def _decoded_document(raw_body: bytes) -> bytes:
    """The PDF carried by an OCR request body (also checks it is the expected JSON)."""
    payload = json.loads(raw_body)
    assert payload["model"] == "mistral-ocr-latest"
    assert payload["document"]["type"] == "document_url"
    prefix = "data:application/pdf;base64,"
    url = payload["document"]["document_url"]
    assert url.startswith(prefix)
    return base64.b64decode(url[len(prefix) :], validate=True)


class _FakeResponse:
    """Minimal stand-in for httpx.Response used by the converter."""

    def __init__(self, page_count: int):
        self.status_code = 200
        self.text = ""
        self._pages = [
            {"index": i, "markdown": f"page-{i}", "images": [], "dimensions": {}}
            for i in range(page_count)
        ]

    @property
    def is_error(self) -> bool:
        return False

    def json(self) -> dict:
        return {"pages": self._pages}


# ---------------------------------------------------------------------------
# The streamed body is the same JSON the API received before
# ---------------------------------------------------------------------------


class TestOcrRequestBody:
    @pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 50])
    async def test_bytes_body_round_trips_across_block_boundaries(self, size, monkeypatch):
        # A 6-byte block exercises bodies of 0, 1 and several blocks with every padding.
        monkeypatch.setattr(pdf_module, "_BASE64_BLOCK_BYTES", 6)
        data = os.urandom(size)

        body = _OcrRequestBody(data)
        raw = await _read_body(body)

        assert len(raw) == body.content_length
        assert _decoded_document(raw) == data

    @pytest.mark.parametrize("size", [0, 5, 6, 7, 50])
    async def test_file_body_round_trips_across_block_boundaries(self, size, tmp_path, monkeypatch):
        monkeypatch.setattr(pdf_module, "_BASE64_BLOCK_BYTES", 6)
        data = os.urandom(size)
        path = tmp_path / "doc.pdf"
        path.write_bytes(data)

        body = _OcrRequestBody(path)
        raw = await _read_body(body)

        assert len(raw) == body.content_length
        assert _decoded_document(raw) == data

    async def test_body_matches_the_legacy_json_payload(self):
        data = _make_pdf(1)
        legacy = json.dumps(
            {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64,"
                    + base64.b64encode(data).decode("ascii"),
                },
            }
        ).encode("ascii")

        assert await _read_body(_OcrRequestBody(data)) == legacy

    async def test_body_can_be_streamed_again_for_a_retry(self, tmp_path):
        path = _write_pdf(tmp_path, num_pages=2)
        body = _OcrRequestBody(path)

        assert await _read_body(body) == await _read_body(body)

    async def test_missing_file_is_a_conversion_error(self, tmp_path):
        with pytest.raises(ConversionError, match="Cannot read PDF file"):
            _OcrRequestBody(tmp_path / "missing.pdf")

    async def test_file_that_shrinks_after_measuring_is_a_conversion_error(self, tmp_path):
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"x" * 100)
        body = _OcrRequestBody(path)
        path.write_bytes(b"x" * 10)

        with pytest.raises(ConversionError, match="changed size"):
            await _read_body(body)


# ---------------------------------------------------------------------------
# What actually goes over the wire (real httpx client, mock transport)
# ---------------------------------------------------------------------------


@pytest.fixture
def wire(monkeypatch):
    """Route the converter's httpx.AsyncClient to a MockTransport recording requests."""
    requests: list[tuple[httpx.Headers, bytes]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        raw = await request.aread()
        requests.append((request.headers, raw))
        pages = len(PdfReader(BytesIO(_decoded_document(raw))).pages)
        return httpx.Response(
            200,
            json={
                "pages": [
                    {"index": i, "markdown": f"page-{i}", "images": [], "dimensions": {}}
                    for i in range(pages)
                ]
            },
        )

    real_client = httpx.AsyncClient

    class _Client(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return requests


class TestWireFormat:
    async def test_unsplit_pdf_is_sent_with_content_length_not_chunked(self, tmp_path, wire):
        source = _write_pdf(tmp_path, num_pages=2)

        doc = await PdfConverter(api_key="dummy-test-key").convert(source)

        assert [p.index for p in doc.pages] == [0, 1]
        [(headers, raw)] = wire
        assert "transfer-encoding" not in headers
        assert int(headers["content-length"]) == len(raw)
        assert headers["content-type"] == "application/json"
        assert headers["authorization"] == "Bearer dummy-test-key"
        assert _decoded_document(raw) == source.read_bytes()

    async def test_split_chunks_are_each_a_valid_streamed_pdf(self, tmp_path, wire):
        source = _write_pdf(tmp_path, num_pages=3)

        doc = await PdfConverter(api_key="dummy-test-key", safe_max_pages=1).convert(source)

        assert [p.index for p in doc.pages] == [0, 1, 2]
        assert len(wire) == 3
        for headers, raw in wire:
            assert "transfer-encoding" not in headers
            assert int(headers["content-length"]) == len(raw)
            assert len(PdfReader(BytesIO(_decoded_document(raw))).pages) == 1


# ---------------------------------------------------------------------------
# Memory: a request no longer holds copies of the whole PDF
# ---------------------------------------------------------------------------


class TestMemory:
    async def test_peak_memory_does_not_grow_with_the_pdf_size(self, tmp_path, mocker):
        size = 24 * 1024 * 1024
        source = tmp_path / "big.pdf"
        with source.open("wb") as fh:
            for _ in range(size // (1024 * 1024)):
                fh.write(os.urandom(1024 * 1024))
        # Content is random bytes: only probe_pdf would need a real PDF.
        mocker.patch.object(pdf_module, "probe_pdf", return_value=(size, 1))

        async def consume(url, *, content, headers):
            sent = 0
            async for part in content:
                sent += len(part)
            assert sent == int(headers["Content-Length"])
            return _FakeResponse(1)

        mocker.patch("httpx.AsyncClient.post", side_effect=consume)

        tracemalloc.start()
        try:
            await PdfConverter(api_key="dummy-test-key").convert(source)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # Before streaming the body, a request held ~6x the PDF (~150 MB here). Now one
        # 3 MiB block and its base64 are alive at a time.
        assert peak < 16 * 1024 * 1024, f"peak {peak / 2**20:.1f} MiB"


# ---------------------------------------------------------------------------
# mistral_ocr_concurrency: bounded PDFs in conversion at once
# ---------------------------------------------------------------------------


def _tracking_post(state: dict):
    async def post(url, *, content, headers):
        state["in_flight"] += 1
        state["max"] = max(state["max"], state["in_flight"])
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return _FakeResponse(1)

    return post


class TestOcrConcurrency:
    async def test_conversions_beyond_the_limit_wait(self, tmp_path, mocker):
        sources = [_write_pdf(tmp_path, 1, name=f"doc-{i}.pdf") for i in range(6)]
        state = {"in_flight": 0, "max": 0}
        mocker.patch("httpx.AsyncClient.post", side_effect=_tracking_post(state))

        converter = PdfConverter(api_key="dummy-test-key", concurrency=2)
        docs = await asyncio.gather(*(converter.convert(s) for s in sources))

        assert len(docs) == 6
        assert state["max"] == 2

    async def test_limit_holds_across_converter_instances(self, tmp_path, mocker):
        sources = [_write_pdf(tmp_path, 1, name=f"doc-{i}.pdf") for i in range(4)]
        state = {"in_flight": 0, "max": 0}
        mocker.patch("httpx.AsyncClient.post", side_effect=_tracking_post(state))

        await asyncio.gather(
            *(PdfConverter(api_key="dummy-test-key", concurrency=1).convert(s) for s in sources)
        )

        assert state["max"] == 1

    async def test_limit_comes_from_configure_by_default(self, tmp_path, mocker):
        sources = [_write_pdf(tmp_path, 1, name=f"doc-{i}.pdf") for i in range(5)]
        state = {"in_flight": 0, "max": 0}
        mocker.patch("httpx.AsyncClient.post", side_effect=_tracking_post(state))
        cfg = mocker.MagicMock(mistral_ocr_concurrency=3)
        mocker.patch("scinr.newton.config.get_config", return_value=cfg)

        converter = PdfConverter(
            api_key="dummy-test-key",
            safe_max_pages=900,
            safe_max_bytes=45 * 1024 * 1024,
            max_retries=3,
            retry_backoff_seconds=2.0,
            error_strategy="fail_fast",
        )
        await asyncio.gather(*(converter.convert(s) for s in sources))

        assert state["max"] == 3

    async def test_invalid_constructor_limit_raises_before_any_request(self, tmp_path, mocker):
        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)

        with pytest.raises(ConversionError, match="mistral_ocr_concurrency"):
            await PdfConverter(api_key="dummy-test-key", concurrency=0).convert(
                _write_pdf(tmp_path, 1)
            )

        assert mock_post.call_count == 0

    def test_semaphore_works_across_event_loops(self, tmp_path, mocker):
        """Celery workers call asyncio.run() once per task: the second run must not
        reuse a semaphore bound to the first, already closed, loop."""
        source = _write_pdf(tmp_path, 1)
        mocker.patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_FakeResponse(1)
        )
        converter = PdfConverter(api_key="dummy-test-key", concurrency=1)

        async def run_twice_concurrently():
            return await asyncio.gather(converter.convert(source), converter.convert(source))

        for _ in range(2):
            docs = asyncio.run(run_twice_concurrently())
            assert len(docs) == 2
