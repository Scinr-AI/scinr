"""WP4 (0.5.0): the described embedded images are stored with their original.

Covers the storage context of ``converters/image_store.py`` wired into the
conversion paths (``converters/main.py``) and the enrichment
(``converters/enrichment.py``):

- A DOCX with a small (discarded), a repeated and a valid image: only the
  valid one is stored, with its original bytes (same SHA-256 as the package
  blob), its ``page_index`` / ``image_index`` and the description of its
  block; ``PageImage.image_id`` is filled. A PPTX picture inside a group is
  stored too. A PDF crop is stored with the global ``page_index`` of its
  chunk (D2, decision D1).
- An empty description, or an image that fails in ``best_effort``, is not
  stored (D1).
- Storage ``none``, ``store_described_images=False`` or a backend with
  ``stores_images=False``: no call at all; the last case warns once per
  process, even across two documents (D4).
- A storage failure fails the document with ``ImageStorageError``, whatever
  the ``media_error_strategy``, and the images already stored of the
  ``raw_file_id`` are cleaned up (D5); a ``_require_content`` failure after
  a stored image cleans up too.
- Memory: with ``llm_concurrency=4`` and ``image_decode_concurrency=1``,
  the storage re-reads run inside the same lane as the decoding, so no two
  original ``load_bytes`` are in flight at once (WP0 memory rule).
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from test_converters_enrichment import BLUE, GREEN, RED, FakeVision, _png  # noqa: E402
from test_pdf_converter_images import _jpeg_data_url  # noqa: E402, F401
from test_pdf_converter_images import ocr as ocr  # noqa: E402, F401, PLC0414  (fixture)
from test_pdf_converter_streaming import _write_pdf  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.converters import image_store as image_store_module  # noqa: E402
from scinr.newton.converters import main as converters_main  # noqa: E402
from scinr.newton.converters import vision as vision_module  # noqa: E402
from scinr.newton.converters.base import (  # noqa: E402
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
    PageImage,
)
from scinr.newton.converters.enrichment import EnrichmentState, enrich_pages  # noqa: E402
from scinr.newton.storage.null import NullRawFileRepository  # noqa: E402

docx = pytest.importorskip("docx")
pptx = pytest.importorskip("pptx")

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}


@pytest.fixture
def vision():
    model = FakeVision()
    model.calls = []
    return model


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class _RecordingRawRepo:
    """In-memory ``RawFileRepository`` that records its calls.

    Duck-typed, like the ``_Raw`` of ``test_converters_enrichment``: it
    implements ``stores_images`` and the image methods of the optional block
    explicitly, because that is what the ingestion calls.
    """

    def __init__(self) -> None:
        self.raw_file_id = ""
        self.images: list[dict] = []
        self.files: list[dict] = []
        self.deleted: list[str] = []
        self.fail_on_image = 0  # ordinal (1-based) that raises, 0 = never
        self.fail_delete = False
        self.calls = 0

    @property
    def stores_images(self) -> bool:
        return True

    async def store_file(self, path, filename, content_type, folder_path, **kwargs) -> str:
        self.raw_file_id = f"raw-{filename}"
        return self.raw_file_id

    async def store_image(
        self, raw_file_id, page_index, image_index, content, **kwargs
    ) -> str:
        self.calls += 1
        if self.calls == self.fail_on_image:
            raise RuntimeError("disk full")
        self.images.append(
            {
                "raw_file_id": raw_file_id,
                "page_index": page_index,
                "image_index": image_index,
                "content": content,
                **kwargs,
            }
        )
        return f"image-id-{len(self.images)}"

    async def store_image_file(self, path, raw_file_id, page_index, image_index, **kwargs) -> str:
        return await self.store_image(
            raw_file_id, page_index, image_index, path.read_bytes(), **kwargs
        )

    async def delete_images(self, raw_file_id, **kwargs) -> int:
        if self.fail_delete:
            raise RuntimeError("cannot clean up")
        self.deleted.append(raw_file_id)
        return len(self.images)


class _Fixed(BaseConverter):
    """Returns a prepared document, as any (external) converter could."""

    supported_extensions = frozenset({"x"})

    def __init__(self, doc: IntermediateDocument) -> None:
        self.doc = doc

    def convert(self, source: Path) -> IntermediateDocument:
        return self.doc


def _page(index: int, markdown: str, *images: PageImage) -> IntermediatePage:
    return IntermediatePage(index=index, markdown=markdown, images=list(images))


def _docx_with_images(tmp_path: Path) -> tuple[Path, list[bytes]]:
    """A DOCX with a small picture and, in another paragraph, the same picture
    twice (a repeated image); returns the path and the two blobs of the
    package, in document order (the second one is the repeated image)."""
    small = io.BytesIO()
    Image.new("RGB", (10, 10), RED).save(small, format="PNG")
    big = io.BytesIO()
    Image.new("RGB", (200, 120), GREEN).save(big, format="PNG")

    document = docx.Document()
    document.add_paragraph("Antes")
    document.add_picture(io.BytesIO(small.getvalue()))
    document.add_picture(io.BytesIO(big.getvalue()))
    document.add_picture(io.BytesIO(big.getvalue()))  # repeated: described once
    document.add_paragraph("Después")
    path = tmp_path / "informe.docx"
    document.save(path)
    return path, [small.getvalue(), big.getvalue()]


def _pptx_group(tmp_path: Path) -> Path:
    from pptx.util import Inches

    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    out = io.BytesIO()
    Image.new("RGB", (200, 120), GREEN).save(out, format="PNG")
    group.shapes.add_picture(io.BytesIO(out.getvalue()), Inches(1), Inches(1))
    path = tmp_path / "grupos.pptx"
    presentation.save(path)
    return path


def _sink(repo: _RecordingRawRepo, **kwargs):
    return image_store_module.image_sink(repo, "rid", filename="f.x", **kwargs)


# ---------------------------------------------------------------------------
# Stored images of DOCX, PPTX and PDF
# ---------------------------------------------------------------------------


class TestDocxStoresOnlyTheValidImage:
    async def test_docx_with_small_repeated_and_valid_images(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        path, (small, big) = _docx_with_images(tmp_path)
        repo = _RecordingRawRepo()

        written, failures = await converters_main.convert_one(path, None, raw_file_repo=repo)

        assert failures == []
        assert [i["image_index"] for i in repo.images] == [1]  # 0 is small, 2 a repeat
        assert repo.images[0]["raw_file_id"] == repo.raw_file_id
        assert repo.images[0]["page_index"] == 0
        assert hashlib.sha256(repo.images[0]["content"]).hexdigest() == hashlib.sha256(
            big
        ).hexdigest()  # the original blob of the package, not the downscale
        assert repo.images[0]["kind"] == "embedded"
        assert repo.images[0]["content_type"] == "image/png"
        assert repo.images[0]["description"] == "desc 0,3,0"  # the block's description
        assert repo.images[0]["filename"] == "informe.docx"
        assert (repo.images[0]["width"], repo.images[0]["height"]) == (None, None)
        image = written[0][2].pages[0].images[1]
        assert image.image_id == "image-id-1"
        assert not image.has_content  # released after the description
        assert not written[0][2].pages[0].images[0].has_content  # discarded, released

    async def test_pptx_picture_inside_a_group_is_stored(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        repo = _RecordingRawRepo()

        written, failures = await converters_main.convert_one(
            _pptx_group(tmp_path), None, raw_file_repo=repo
        )

        assert failures == []
        assert [(i["page_index"], i["image_index"]) for i in repo.images] == [(0, 0)]
        assert written[0][2].pages[0].images[0].image_id == "image-id-1"

    async def test_pdf_crop_is_stored_with_the_global_page_index(
        self, tmp_path, ocr, vision, caplog, monkeypatch
    ):
        from scinr.newton.converters.pdf import PdfConverter

        monkeypatch.setattr(converters_main, "get_converter", lambda path: PdfConverter(api_key="k"))
        configure(
            **_NEO, vision_llm=vision, mistral_api_key="k", mistral_ocr_image_max_pages=1
        )
        source = _write_pdf(tmp_path, 2)
        ocr["colour_for"] = lambda i: GREEN if i == 0 else BLUE
        repo = _RecordingRawRepo()

        written, failures = await converters_main.convert_one(source, None, raw_file_repo=repo)

        assert failures == [], (failures, [r.getMessage() for r in caplog.records])
        assert [(i["page_index"], i["image_index"]) for i in repo.images] == [(0, 0), (1, 0)]
        assert [i["description"] for i in repo.images] == ["desc 0,3,0", "desc 0,0,5"]
        # The stored bytes are the decoded Mistral crop of the page, as described.
        expected = _jpeg_data_url(GREEN).split(",", 1)[1]
        assert repo.images[0]["content"] == base64.b64decode(expected)
        assert written[0][2].pages[1].images[0].image_id == "image-id-2"


# ---------------------------------------------------------------------------
# What is not stored (D1) and the disabled contexts (D4)
# ---------------------------------------------------------------------------


class TestNothingIsStored:
    async def test_empty_description_and_failed_image_are_not_stored(self, vision):
        configure(**_NEO, vision_llm=vision, media_error_strategy="best_effort")
        repo = _RecordingRawRepo()
        page = _page(
            0,
            "![w](w) ![r](r) ![g](g)",
            PageImage(index=0, base64=base64.b64encode(_png((255, 255, 255))).decode(), ref="w"),
            PageImage(index=1, base64=base64.b64encode(_png(RED)).decode(), ref="r"),
            PageImage(index=2, base64=base64.b64encode(_png(GREEN)).decode(), ref="g"),
        )
        with _sink(repo) as sink:
            missing = await enrich_pages([page], EnrichmentState())
        assert missing == [(0, 1)]  # the failed image, best_effort
        assert [(i["page_index"], i["image_index"]) for i in repo.images] == [(0, 2)]
        assert repo.images[0]["description"] == "desc 0,3,0"
        assert page.images[2].image_id == "image-id-1"
        assert sink is not None and sink.stored == 1

    async def test_empty_raw_file_id_installs_no_context(self, vision):
        configure(**_NEO, vision_llm=vision)
        with image_store_module.image_sink(_RecordingRawRepo(), "", filename="f.x") as sink:
            assert sink is None
            assert image_store_module._current.get() is None

    async def test_storage_none_makes_no_call(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        path, _blobs = _docx_with_images(tmp_path)
        repo = NullRawFileRepository()

        written, failures = await converters_main.convert_one(path, None, raw_file_repo=repo)

        assert failures == []  # the null repo keeps the "" raw_file_id: no sink
        assert image_store_module._current.get() is None
        assert written[0][2].pages[0].images[1].image_id is None

    async def test_store_described_images_off_makes_no_call(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision, store_described_images=False)
        path, _blobs = _docx_with_images(tmp_path)
        repo = _RecordingRawRepo()

        written, failures = await converters_main.convert_one(path, None, raw_file_repo=repo)

        assert failures == []
        assert repo.images == []  # described, but not stored
        assert written[0][2].pages[0].images[1].description == "desc 0,3,0"

    async def test_a_backend_without_images_warns_once_for_two_documents(
        self, tmp_path, vision, caplog, monkeypatch
    ):
        configure(**_NEO, vision_llm=vision)
        image_store_module._warned_no_image_storage = False
        repo = NullRawFileRepository()
        # Force a non-empty raw_file_id so the stores_images check is the one
        # that installs no context.
        monkeypatch.setattr(repo, "store_file", _store_file_stub)

        path, _blobs = _docx_with_images(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        other, _ = _docx_with_images(sub)
        for source in (path, other):
            written, failures = await converters_main.convert_one(
                source, None, raw_file_repo=repo
            )
            assert failures == []
            assert all(img.image_id is None for img in written[0][2].pages[0].images)
        warnings = [r for r in caplog.records if "does not store images" in r.getMessage()]
        assert len(warnings) == 1


async def _store_file_stub(*args, **kwargs):
    return "rid"


# ---------------------------------------------------------------------------
# Storage failures: the document fails and the stored images are cleaned (D5)
# ---------------------------------------------------------------------------


class TestStorageFailure:
    async def test_store_failure_fails_the_document(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision, media_error_strategy="best_effort")
        path, _blobs = _docx_with_images(tmp_path)
        repo = _RecordingRawRepo()
        repo.fail_on_image = 1

        written, failures = await converters_main.convert_one(path, None, raw_file_repo=repo)

        assert written == []
        (failed_path, error), = failures
        assert failed_path == path
        assert "could not be stored" in error and "disk full" in error
        # Nothing was stored before the failure: no cleanup is needed (D5).
        assert repo.deleted == []
        # The raw_file_id of the failed conversion stays in storage, as before.

    async def test_cleanup_deletes_the_images_stored_before_the_failure(
        self, tmp_path, vision, monkeypatch
    ):
        """The second image's store fails in best_effort: the document fails
        with ``ImageStorageError`` and the image stored before it is cleaned
        up with ``delete_images`` of the ``raw_file_id`` (D5)."""
        configure(**_NEO, vision_llm=vision, media_error_strategy="best_effort")
        repo = _RecordingRawRepo()
        repo.fail_on_image = 2
        page = _page(
            0,
            "a ![g](g) b ![b](b)",
            PageImage(index=0, base64=base64.b64encode(_png(GREEN)).decode(), ref="g"),
            PageImage(index=1, base64=base64.b64encode(_png(BLUE)).decode(), ref="b"),
        )
        monkeypatch.setattr(
            converters_main, "get_converter", lambda path: _Fixed(IntermediateDocument(pages=[page]))
        )
        source = tmp_path / "f.x"
        source.write_text("x", encoding="utf-8")

        written, failures = await converters_main.convert_one(source, None, raw_file_repo=repo)

        (failed_path, error), = failures
        assert "could not be stored" in error and "disk full" in error
        assert written == []
        assert [i["image_index"] for i in repo.images] == [0]  # one was stored…
        assert repo.deleted == ["raw-f.x"]  # …and cleaned up before propagating

    async def test_a_failed_cleanup_is_logged_and_does_not_mask_the_error(
        self, vision, caplog
    ):
        configure(**_NEO, vision_llm=vision)
        repo = _RecordingRawRepo()
        repo.fail_delete = True
        with _sink(repo) as sink:
            sink.stored = 1  # as if an image had been stored
            await image_store_module.discard_stored(sink)  # does not raise
        assert repo.deleted == []  # the delete itself failed
        warnings = [r for r in caplog.records if "could not be cleaned up" in r.getMessage()]
        assert len(warnings) == 1
        assert "rid" in warnings[0].getMessage()

    async def test_require_content_failure_cleans_up_the_stored_images(
        self, tmp_path, vision, monkeypatch
    ):
        """A conversion failure *after* a stored image (here ``_require_content``)
        cleans up the images already stored of the ``raw_file_id`` (D5). The
        image is described and stored for real; ``_require_content`` is made
        to fail as it does when nothing could be converted."""
        configure(**_NEO, vision_llm=vision, media_error_strategy="best_effort")
        repo = _RecordingRawRepo()
        page = _page(
            0,
            "![g](g)",
            PageImage(index=0, base64=base64.b64encode(_png(BLUE)).decode(), ref="g"),
        )
        monkeypatch.setattr(
            converters_main, "get_converter", lambda path: _Fixed(IntermediateDocument(pages=[page]))
        )
        source = tmp_path / "f.x"
        source.write_text("x", encoding="utf-8")

        def _failing_require_content(doc, src):
            raise ConversionError(f"Nothing could be converted from {src.name}")

        monkeypatch.setattr(converters_main, "_require_content", _failing_require_content)

        written, failures = await converters_main.convert_one(source, None, raw_file_repo=repo)

        (failed_path, error), = failures
        assert failed_path == source
        assert "Nothing could be converted from f.x" in error
        assert written == []
        assert [i["image_index"] for i in repo.images] == [0]  # one was stored…
        assert repo.deleted == ["raw-f.x"]  # …and cleaned up before propagating


# ---------------------------------------------------------------------------
# Memory: the storage re-read runs in the decode lane
# ---------------------------------------------------------------------------


class TestMemoryLane:
    async def test_no_two_storage_reads_are_in_flight_at_once(self, monkeypatch, vision):
        """Same pattern as ``test_embedded_images_are_read_and_decoded_one_at_a_time``:
        with ``llm_concurrency=4`` and ``image_decode_concurrency=1``, the
        decode of one image and the storage re-read of the previous one are
        strictly serialized by the lane semaphore."""
        configure(**_NEO, vision_llm=vision, llm_concurrency=4, image_decode_concurrency=1)
        repo = _RecordingRawRepo()
        events: list[str] = []

        real_prepare = vision_module.prepare_image

        def _recording_prepare(data, max_side):
            events.append("decode")
            try:
                return real_prepare(data, max_side)
            finally:
                events.append("done")

        monkeypatch.setattr(vision_module, "prepare_image", _recording_prepare)

        def _image(index: int) -> PageImage:
            data = _png((0, 0, 200 - index), size=(300 + index, 200))

            def _load() -> bytes:
                events.append("read")
                return data

            return PageImage.with_loader(_load, index=index)

        real_store = repo.store_image

        async def _recording_store(raw_file_id, page_index, image_index, content, **kwargs):
            events.append("store")
            result = await real_store(raw_file_id, page_index, image_index, content, **kwargs)
            events.append("stored")
            return result

        repo.store_image = _recording_store  # type: ignore[method-assign]

        page = _page(0, "texto", *[_image(i) for i in range(4)])
        with _sink(repo):
            await enrich_pages([page], EnrichmentState())

        # The filter reads each image once (header and hash); then the lane
        # serializes each image completely: its decode read+decode+done, and
        # later its store read+store+stored — never two originals in memory
        # at once, even with llm_concurrency=4 model calls allowed in flight.
        assert events == (
            ["read"] * 4 + ["read", "decode", "done"] * 4 + ["read", "store", "stored"] * 4
        )
        assert [img.image_id for img in page.images] == [f"image-id-{i + 1}" for i in range(4)]
