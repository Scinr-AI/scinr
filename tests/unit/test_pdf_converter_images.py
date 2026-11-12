"""WP5 (0.5.0): embedded images in PDF conversion (Mistral OCR).

With ``embedded_images_enabled()`` the OCR request asks for the images, the
chunks are capped at ``mistral_ocr_image_max_pages`` and each chunk's images
are described (and released) before the next chunk. Without it, the request
body is byte for byte the one sent before.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).parent))

from test_converters_enrichment import GREEN, RED, FakeVision  # noqa: E402
from test_pdf_converter_streaming import (  # noqa: E402
    _decoded_document,
    _make_pdf,
    _read_body,
    _write_pdf,
)

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.converters.enrichment import render_image_block  # noqa: E402
from scinr.newton.converters.pdf import (  # noqa: E402
    PdfConverter,
    _OcrRequestBody,
    _split_data_url,
)

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}

BLUE = (0, 0, 255)


def _jpeg_data_url(colour) -> str:
    out = io.BytesIO()
    Image.new("RGB", (300, 200), colour).save(out, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


@pytest.fixture
def vision():
    model = FakeVision()
    model.calls = []
    return model


@pytest.fixture
def ocr(monkeypatch):
    """MockTransport OCR server: each page carries one image of ``colour_for(abs_index)``."""
    state: dict = {"requests": [], "colour_for": lambda i: GREEN, "offset": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        raw = await request.aread()
        payload = json.loads(raw)
        state["requests"].append((request.headers, raw, payload))
        pages = len(PdfReader(BytesIO(_decoded_document(raw))).pages)
        want_images = payload.get("include_image_base64", False)
        offset = state["offset"]
        state["offset"] += pages
        body = {"pages": []}
        for i in range(pages):
            images = []
            markdown = f"page-{offset + i}"
            if want_images:
                images = [{
                    "id": f"img-{offset + i}.jpeg",
                    "top_left_x": 10, "top_left_y": 10,
                    "bottom_right_x": 310, "bottom_right_y": 210,
                    "image_base64": _jpeg_data_url(state["colour_for"](offset + i)),
                }]
                markdown += f"\n\n![img-{offset + i}.jpeg](img-{offset + i}.jpeg)"
            body["pages"].append(
                {"index": i, "markdown": markdown, "images": images, "dimensions": {}}
            )
        return httpx.Response(200, json=body)

    real_client = httpx.AsyncClient

    class _Client(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return state


class TestRequestBody:
    async def test_without_images_the_body_is_unchanged(self):
        data = _make_pdf(1)
        legacy = json.dumps({
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64," + base64.b64encode(data).decode(),
            },
        }).encode("ascii")
        assert await _read_body(_OcrRequestBody(data, None)) == legacy

    async def test_with_images_the_options_are_added_and_length_matches(self, tmp_path):
        path = _write_pdf(tmp_path, 2)
        body = _OcrRequestBody(
            path, {"include_image_base64": True, "image_limit": 20, "image_min_size": 64}
        )
        raw = await _read_body(body)
        assert len(raw) == body.content_length
        payload = json.loads(raw)
        assert payload["include_image_base64"] is True
        assert (payload["image_limit"], payload["image_min_size"]) == (20, 64)
        assert _decoded_document(raw) == path.read_bytes()


def test_split_data_url():
    assert _split_data_url("data:image/jpeg;base64,QUJD") == ("image/jpeg", "QUJD")
    assert _split_data_url("QUJD") == ("image/png", "QUJD")
    assert _split_data_url("") == ("image/png", "")


class TestConversion:
    async def test_disabled_sends_the_old_body_and_keeps_old_mapping(self, tmp_path, ocr):
        configure(**_NEO)  # no vision_llm
        source = _write_pdf(tmp_path, 2)

        doc = await PdfConverter(api_key="k").convert(source)

        [(_, raw, payload)] = ocr["requests"]
        assert "include_image_base64" not in payload
        assert raw.endswith(b'"}}')
        assert [p.images for p in doc.pages] == [[], []]

    async def test_enabled_describes_in_place_and_releases(self, tmp_path, ocr, vision):
        configure(**_NEO, vision_llm=vision)
        source = _write_pdf(tmp_path, 2)
        ocr["colour_for"] = lambda i: GREEN if i == 0 else BLUE

        doc = await PdfConverter(api_key="k").convert(source)

        [(_, _, payload)] = ocr["requests"]
        assert payload["include_image_base64"] is True
        assert payload["image_limit"] == 20 and payload["image_min_size"] == 64
        assert doc.pages[0].markdown == "page-0\n\n" + render_image_block(0, "desc 0,3,0")
        assert doc.pages[1].markdown == "page-1\n\n" + render_image_block(0, "desc 0,0,5")
        image = doc.pages[0].images[0]
        assert (image.ref, image.media_type, image.width, image.height) == (
            "img-0.jpeg", "image/jpeg", 300, 200,
        )
        assert image.base64 == "" and image.description == "desc 0,3,0"
        assert doc.missing_images is None

    async def test_image_max_pages_splits_and_state_is_shared(self, tmp_path, ocr, vision):
        configure(**_NEO, vision_llm=vision, mistral_ocr_image_max_pages=2)
        source = _write_pdf(tmp_path, 5)
        ocr["colour_for"] = lambda i: BLUE  # the same image on every page

        doc = await PdfConverter(api_key="k").convert(source)

        page_counts = [
            len(PdfReader(BytesIO(_decoded_document(raw))).pages) for _, raw, _ in ocr["requests"]
        ]
        assert page_counts == [2, 2, 1]
        assert [p.index for p in doc.pages] == [0, 1, 2, 3, 4]
        assert len(vision.calls) == 1  # described once across chunks
        assert "<image" in doc.pages[0].markdown
        assert all("<image" not in p.markdown and "![" not in p.markdown for p in doc.pages[1:])

    async def test_failed_image_is_recorded_in_best_effort(self, tmp_path, ocr, vision):
        configure(**_NEO, vision_llm=vision, mistral_ocr_image_max_pages=1)
        source = _write_pdf(tmp_path, 3)
        ocr["colour_for"] = lambda i: RED if i == 1 else (0, 40 * i, 200)

        doc = await PdfConverter(api_key="k").convert(source)

        assert doc.missing_images == [(1, 0)]
        assert doc.pages[1].markdown == "page-1\n\n"
        assert doc.missing_page_ranges is None

    async def test_failed_image_in_fail_fast_fails_the_document_not_the_chunk(
        self, tmp_path, ocr, vision
    ):
        from scinr.newton.converters.base import ConversionError

        configure(
            **_NEO, vision_llm=vision, mistral_ocr_image_max_pages=1,
            media_error_strategy="fail_fast", mistral_ocr_error_strategy="best_effort",
        )
        source = _write_pdf(tmp_path, 2)
        ocr["colour_for"] = lambda i: RED

        with pytest.raises(ConversionError, match="Embedded image"):
            await PdfConverter(api_key="k").convert(source)
        assert len(ocr["requests"]) == 1  # stopped at the first chunk
