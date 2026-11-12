"""WP4 (0.5.0): embedded-image enrichment with a fake ``vision_llm``.

The fake model answers from the colour of the image it receives, so each test
controls the outcome per image: red fails, white answers "" (no
information), any other colour answers ``"desc <r>,<g>,<b>"``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image

from scinr.newton.config import configure
from scinr.newton.converters import main as converters_main
from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
    PageImage,
)
from scinr.newton.converters.enrichment import EnrichmentState, enrich_pages, render_image_block
from scinr.newton.converters.vision import prepare_image

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}
RED, GREEN, BLUE, WHITE = (255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 255, 255)


class FakeVision(BaseChatModel):
    calls: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "fake-vision"

    def _answer(self, messages) -> str:
        human = messages[-1]
        text = next(b["text"] for b in human.content if b["type"] == "text")
        image = next(b for b in human.content if b["type"] == "image")
        raw = base64.b64decode(image["base64"])
        with Image.open(io.BytesIO(raw)) as img:
            colour = img.convert("RGB").getpixel((img.width // 2, img.height // 2))
            size = img.size
        self.calls.append(
            {"colour": colour, "size": size, "text": text, "mime": image["mime_type"]}
        )
        if all(abs(c - r) < 40 for c, r in zip(colour, RED, strict=True)):
            raise RuntimeError("vision provider exploded")
        if all(c > 215 for c in colour):
            return ""
        return f"desc {colour[0] // 50},{colour[1] // 50},{colour[2] // 50}"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self._answer(messages)))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        return self._generate(messages)


def _png(colour, size=(200, 120)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="PNG")
    return out.getvalue()


def _b64_image(index, colour, size=(200, 120), ref=None) -> PageImage:
    return PageImage(
        index=index, base64=base64.b64encode(_png(colour, size)).decode(), ref=ref
    )


@pytest.fixture
def vision():
    model = FakeVision()
    model.calls = []
    return model


@pytest.fixture
def configured(vision):
    def _configure(**kwargs):
        return configure(**_NEO, vision_llm=vision, **kwargs)

    _configure()
    return _configure


async def test_block_replaces_the_reference(configured, vision):
    page = IntermediatePage(
        index=0,
        markdown="Intro\n\n![img-0.jpeg](img-0.jpeg)\n\nOutro",
        images=[_b64_image(0, GREEN, ref="img-0.jpeg")],
    )
    missing = await enrich_pages([page], EnrichmentState())
    assert missing == []
    assert page.markdown == (
        "Intro\n\n" + render_image_block(0, "desc 0,3,0") + "\n\nOutro"
    )
    assert vision.calls[0]["mime"] == "image/jpeg"
    assert "Intro" in vision.calls[0]["text"]  # page text sent as context


async def test_blocks_without_reference_go_at_the_end_in_index_order(configured):
    page = IntermediatePage(
        index=3, markdown="Slide text\n", images=[_b64_image(1, BLUE), _b64_image(0, GREEN)]
    )
    await enrich_pages([page], EnrichmentState())
    assert page.markdown == (
        "Slide text\n\n"
        + render_image_block(0, "desc 0,3,0")
        + "\n\n"
        + render_image_block(1, "desc 0,0,5")
        + "\n"
    )


async def test_small_images_are_discarded_and_their_reference_removed(configured, vision):
    page = IntermediatePage(
        index=0,
        markdown="A ![icon](icon) B",
        images=[_b64_image(0, GREEN, size=(300, 40), ref="icon")],
    )
    await enrich_pages([page], EnrichmentState())
    assert vision.calls == []
    assert page.markdown == "A  B"
    assert page.images[0].base64 == ""


async def test_known_size_skips_decoding(configured, vision):
    image = PageImage.with_loader(lambda: b"not an image", index=0, width=10, height=10)
    page = IntermediatePage(index=0, markdown="x", images=[image])
    missing = await enrich_pages([page], EnrichmentState())
    assert missing == [] and vision.calls == [] and not image.has_content


async def test_repeated_image_is_described_once_per_document(configured, vision):
    state = EnrichmentState()
    pages = [
        IntermediatePage(index=i, markdown=f"p{i}", images=[_b64_image(0, BLUE)])
        for i in range(3)
    ]
    await enrich_pages(pages[:2], state)
    await enrich_pages(pages[2:], state)  # next PDF chunk, same state
    assert len(vision.calls) == 1
    assert "<image" in pages[0].markdown
    assert all("<image" not in p.markdown for p in pages[1:])
    assert all(not img.has_content for p in pages for img in p.images)


async def test_cap_per_page_keeps_the_largest(configured, vision):
    configured(max_images_per_page=2)
    page = IntermediatePage(
        index=0,
        markdown="x",
        images=[
            _b64_image(0, (0, 0, 200), size=(100, 100)),
            _b64_image(1, (0, 200, 0), size=(400, 300)),
            _b64_image(2, (200, 200, 0), size=(300, 300)),
        ],
    )
    await enrich_pages([page], EnrichmentState())
    assert sorted(c["size"] for c in vision.calls) == [(300, 300), (400, 300)]
    assert '<image index="0"' not in page.markdown
    assert not page.images[0].has_content


async def test_best_effort_skips_the_failed_image_and_releases_it(configured):
    page = IntermediatePage(
        index=5,
        markdown="![a](a) ![b](b)",
        images=[_b64_image(0, RED, ref="a"), _b64_image(1, GREEN, ref="b")],
    )
    missing = await enrich_pages([page], EnrichmentState())
    assert missing == [(5, 0)]
    assert page.markdown == " " + render_image_block(1, "desc 0,3,0")
    assert all(img.base64 == "" and not img.has_content for img in page.images)


async def test_fail_fast_raises_and_still_releases(configured):
    configured(media_error_strategy="fail_fast")
    page = IntermediatePage(
        index=0, markdown="x", images=[_b64_image(0, RED), _b64_image(1, GREEN)]
    )
    with pytest.raises(ConversionError, match="page 0"):
        await enrich_pages([page], EnrichmentState())
    assert all(not img.has_content for img in page.images)


async def test_empty_description_inserts_nothing(configured):
    page = IntermediatePage(
        index=0, markdown="a ![w](w) b", images=[_b64_image(0, WHITE, ref="w")]
    )
    await enrich_pages([page], EnrichmentState())
    assert page.markdown == "a  b"


async def test_already_described_images_are_left_alone(configured, vision):
    image = _b64_image(0, GREEN)
    image.description = "given"
    page = IntermediatePage(index=0, markdown="x", images=[image])
    await enrich_pages([page], EnrichmentState())
    assert vision.calls == [] and page.markdown == "x" and image.has_content


def test_prepare_image_downscales_and_flattens():
    out = io.BytesIO()
    Image.new("RGBA", (4000, 1000), (0, 0, 255, 0)).save(out, format="PNG")
    jpeg = prepare_image(out.getvalue(), 1568)
    with Image.open(io.BytesIO(jpeg)) as img:
        assert img.format == "JPEG" and max(img.size) == 1568 and img.mode == "RGB"
        assert img.getpixel((10, 10)) == (255, 255, 255)  # transparency on white


def test_prepare_image_rejects_garbage():
    with pytest.raises(ConversionError):
        prepare_image(b"definitely not an image", 512)


# ---------------------------------------------------------------------------
# Through converters/main.py
# ---------------------------------------------------------------------------


class _ImagesConverter(BaseConverter):
    supported_extensions = ("imgs",)

    def convert(self, source: Path) -> IntermediateDocument:
        return IntermediateDocument(
            pages=[
                IntermediatePage(
                    index=0,
                    markdown="Texto ![fig](fig)",
                    images=[_b64_image(0, GREEN, ref="fig")],
                )
            ]
        )


class _Pages:
    def __init__(self) -> None:
        self.markdown: list[str] = []

    async def store_page(self, raw_file_id, filename, folder_path, page_index, markdown, **kw):
        self.markdown.append(markdown)
        return "p1"


class _Raw:
    async def store_file(self, *args, **kwargs):
        return "r1"


@pytest.fixture
def images_converter(monkeypatch):
    monkeypatch.setattr(converters_main, "get_converter", lambda path: _ImagesConverter())


async def test_stored_page_and_json_carry_the_block(configured, images_converter, tmp_path):
    src = tmp_path / "doc.imgs"
    src.write_text("x", encoding="utf-8")
    pages = _Pages()

    written, failures = await converters_main.convert_one(
        src, tmp_path / "out", raw_file_repo=_Raw(), page_repo=pages
    )

    assert failures == []
    block = render_image_block(0, "desc 0,3,0")
    assert pages.markdown == ["Texto " + block]
    json_text = written[0][1].read_text(encoding="utf-8")
    assert "desc 0,3,0" in json_text
    assert '"base64": ""' in json_text  # no image bytes in the intermediate JSON


async def test_disabled_makes_no_call_and_keeps_the_images(
    configured, vision, images_converter, tmp_path
):
    configured(describe_embedded_images=False)
    src = tmp_path / "doc.imgs"
    src.write_text("x", encoding="utf-8")

    written, _ = await converters_main.convert_one(src, None)

    assert vision.calls == []
    doc = written[0][2]
    assert doc.pages[0].markdown == "Texto ![fig](fig)"
    assert doc.pages[0].images[0].has_content


async def test_without_vision_llm_nothing_happens(images_converter, tmp_path):
    configure(**_NEO)  # describe_embedded_images=True but no vision_llm
    src = tmp_path / "doc.imgs"
    src.write_text("x", encoding="utf-8")
    written, _ = await converters_main.convert_one(src, None)
    assert written[0][2].pages[0].images[0].has_content


async def test_missing_images_reach_the_document(configured, monkeypatch, tmp_path):
    class _Failing(_ImagesConverter):
        def convert(self, source):
            return IntermediateDocument(
                pages=[IntermediatePage(index=2, markdown="x", images=[_b64_image(1, RED)])]
            )

    monkeypatch.setattr(converters_main, "get_converter", lambda path: _Failing())
    src = tmp_path / "doc.imgs"
    src.write_text("x", encoding="utf-8")
    written, failures = await converters_main.convert_one(src, None)
    assert failures == []
    assert written[0][2].missing_images == [(2, 1)]
