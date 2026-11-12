"""WP2 (0.5.0): ``FileMetadata``, lazy ``PageImage`` and the page time range.

- ``PageImage`` loads its bytes lazily and never serialises the loader.
- ``start_ms`` / ``end_ms`` round-trip through the Mongo page repository, and
  records stored before 0.5.0 read as ``None``.
- The converter passes ``start_ms`` / ``end_ms`` to ``store_page`` only when
  they are set, so a custom repository with the pre-0.5.0 signature keeps
  working for every other format.
- ``PageText`` exposes the time range.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))

from test_storage_tenancy import mongo  # noqa: E402,F401  (fixture)

from scinr.newton.converters.base import (  # noqa: E402
    IntermediateDocument,
    IntermediatePage,
    PageImage,
)
from scinr.newton.converters.main import _store_page, convert_one  # noqa: E402
from scinr.newton.models.file_metadata import FILE_METADATA_KEYS, FileMetadata  # noqa: E402
from scinr.newton.models.transcript import TranscriptCue  # noqa: E402
from scinr.newton.navigation.pages import _page_text  # noqa: E402
from scinr.newton.storage.models import ConvertedPageRecord  # noqa: E402
from scinr.newton.storage.mongodb.pages import MongoDBPageRepository  # noqa: E402
from scinr.newton.storage.null import NullPageRepository  # noqa: E402

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestFileMetadata:
    def test_keys_are_the_closed_field_list(self):
        assert FILE_METADATA_KEYS == (
            "captured_at", "device_make", "device_model", "gps_latitude", "gps_longitude",
            "width", "height", "duration_ms", "container", "audio_codec", "video_codec",
            "frame_rate",
        )

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValidationError):
            FileMetadata(colour="red")

    def test_round_trip_through_intermediate_json(self):
        doc = IntermediateDocument(
            pages=[IntermediatePage(index=0, markdown="x", start_ms=0, end_ms=300_000)],
            file_metadata=FileMetadata(duration_ms=300_000, container="mp3"),
            missing_segments=[(300_000, 600_000)],
            missing_images=[(0, 1)],
        )
        back = IntermediateDocument.model_validate_json(doc.to_json())
        assert back == doc

    def test_old_intermediate_json_still_validates(self):
        old = {"pages": [{"index": 0, "markdown": "x", "images": [
            {"index": 0, "base64": "QUJD", "media_type": "image/png", "description": ""}
        ]}]}
        doc = IntermediateDocument.model_validate(old)
        assert doc.file_metadata is None and doc.missing_images is None
        assert doc.pages[0].start_ms is None
        assert doc.pages[0].images[0].load_bytes() == b"ABC"


class TestPageImage:
    def test_base64_is_optional(self):
        assert PageImage(index=0).has_content is False

    def test_lazy_loader_is_read_on_demand_and_never_serialised(self):
        calls: list[int] = []

        def _load() -> bytes:
            calls.append(1)
            return b"blob"

        image = PageImage.with_loader(_load, index=2, ref="docx-img-2", media_type="image/jpeg")
        dumped = json.loads(image.model_dump_json())
        assert calls == []
        assert dumped["base64"] == "" and "_loader" not in dumped
        assert image.has_content
        assert image.load_bytes() == b"blob"
        assert calls == [1]

    def test_release_drops_base64_and_loader(self):
        image = PageImage(index=0, base64=base64.b64encode(b"x").decode())
        image.release()
        assert image.base64 == "" and not image.has_content
        lazy = PageImage.with_loader(lambda: b"x", index=0)
        lazy.release()
        assert not lazy.has_content


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestPageTimeRange:
    async def test_mongo_round_trip(self, mongo):  # noqa: F811
        db, _ = mongo
        repo = MongoDBPageRepository()
        pid = await repo.store_page(
            "rf", "clip.mp3", None, 1, "texto", start_ms=300_000, end_ms=600_000
        )
        (stored,) = db["converted_pages"].docs
        assert (stored["start_ms"], stored["end_ms"]) == (300_000, 600_000)
        (record,) = await repo.get_pages_by_ids([pid])
        assert (record.start_ms, record.end_ms) == (300_000, 600_000)

    async def test_mongo_without_range_stores_null(self, mongo):  # noqa: F811
        db, _ = mongo
        await MongoDBPageRepository().store_page("rf", "a.pdf", None, 0, "x")
        (stored,) = db["converted_pages"].docs
        assert stored["start_ms"] is None and stored["end_ms"] is None

    async def test_mongo_transcript_round_trip(self, mongo):  # noqa: F811
        db, _ = mongo
        repo = MongoDBPageRepository()
        cues = [
            {"start_ms": 300_000, "end_ms": 305_000, "text": "hello world"},
            {"start_ms": 305_000, "end_ms": 310_000, "text": "second cue"},
        ]
        pid = await repo.store_page(
            "rf", "clip.mp3", None, 1, "texto", start_ms=300_000, end_ms=600_000,
            transcript=cues,
        )
        (stored,) = db["converted_pages"].docs
        assert stored["transcript"] == cues
        (record,) = await repo.get_pages_by_ids([pid])
        assert record.transcript is not None
        assert [(c.start_ms, c.end_ms, c.text) for c in record.transcript] == [
            (300_000, 305_000, "hello world"),
            (305_000, 310_000, "second cue"),
        ]

    async def test_mongo_without_transcript_stores_null(self, mongo):  # noqa: F811
        db, _ = mongo
        await MongoDBPageRepository().store_page("rf", "a.pdf", None, 0, "x")
        (stored,) = db["converted_pages"].docs
        assert stored["transcript"] is None

    async def test_record_stored_before_0_5_reads_none(self, mongo):  # noqa: F811
        db, _ = mongo
        await db["converted_pages"].insert_one({
            "raw_file_id": "rf", "filename": "old", "folder_path": None, "page_index": 0,
            "markdown": "legacy", "converted_at": datetime.now(UTC),
        })
        (record,) = await MongoDBPageRepository().get_pages("rf")
        assert record.start_ms is None and record.end_ms is None
        assert record.transcript is None

    async def test_null_repository_accepts_the_range(self):
        assert await NullPageRepository().store_page(
            "rf", "a.mp3", None, 0, "x", start_ms=0, end_ms=1
        ) == ""

    async def test_null_repository_accepts_the_transcript(self):
        assert await NullPageRepository().store_page(
            "rf", "a.mp3", None, 0, "x",
            start_ms=0, end_ms=1,
            transcript=[{"start_ms": 0, "end_ms": 1, "text": "hello"}],
        ) == ""


class _OldSignaturePages:
    """A custom repository written against the pre-0.5.0 ``store_page``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def store_page(
        self, raw_file_id, filename, folder_path, page_index, markdown,
        *, tenant_id=None, created_by_user_id=None, job_id=None,
    ):
        self.calls.append({"filename": filename, "page_index": page_index})
        return f"page-{len(self.calls)}"


class _Raw:
    async def store_file(self, path, filename, content_type, folder_path, **owner):
        return "raw-1"


class TestConverterPassesTheRangeOnlyWhenSet:
    async def test_old_signature_repository_still_works_for_text(self, tmp_path: Path):
        src = tmp_path / "doc.txt"
        src.write_text("hola", encoding="utf-8")
        pages = _OldSignaturePages()

        written, failures = await convert_one(src, None, raw_file_repo=_Raw(), page_repo=pages)

        assert failures == []
        assert pages.calls == [{"filename": "doc.txt", "page_index": 0}]
        assert written[0][2].pages[0].page_id == "page-1"

    async def test_range_is_forwarded_when_present(self):
        seen: dict = {}

        class _Pages:
            async def store_page(self, *args, **kwargs):
                seen.update(kwargs)
                return "p"

        page = IntermediatePage(index=3, markdown="x", start_ms=900_000, end_ms=1_200_000)
        await _store_page(
            _Pages(), page, raw_file_id="rf", filename="a.mp3", folder_path=None,
            tenant_id=None, created_by_user_id=None, job_id=None,
        )
        assert (seen["start_ms"], seen["end_ms"], seen["page_index"]) == (900_000, 1_200_000, 3)
        assert "transcript" not in seen

    async def test_transcript_is_forwarded_when_present(self):
        seen: dict = {}

        class _Pages:
            async def store_page(self, *args, **kwargs):
                seen.update(kwargs)
                return "p"

        page = IntermediatePage(
            index=3, markdown="x", start_ms=900_000, end_ms=1_200_000,
            transcript=[TranscriptCue(start_ms=900_000, end_ms=901_000, text="hello")],
        )
        await _store_page(
            _Pages(), page, raw_file_id="rf", filename="a.mp3", folder_path=None,
            tenant_id=None, created_by_user_id=None, job_id=None,
        )
        assert seen["transcript"] == [{"start_ms": 900_000, "end_ms": 901_000, "text": "hello"}]

    async def test_range_is_omitted_when_absent(self):
        seen: dict = {}

        class _Pages:
            async def store_page(self, *args, **kwargs):
                seen.update(kwargs)
                return "p"

        await _store_page(
            _Pages(), IntermediatePage(index=0, markdown="x"), raw_file_id="rf",
            filename="a.pdf", folder_path=None, tenant_id=None, created_by_user_id=None,
            job_id=None,
        )
        assert "start_ms" not in seen and "end_ms" not in seen
        assert "transcript" not in seen


def test_page_text_exposes_the_time_range():
    record = ConvertedPageRecord(
        id="p1", raw_file_id="rf", filename="clip.mp3", folder_path=None, page_index=2,
        markdown="x", converted_at=datetime.now(UTC), start_ms=600_000, end_ms=900_000,
    )
    page = _page_text(record)
    assert (page.start_ms, page.end_ms) == (600_000, 900_000)
    legacy = record.model_copy(update={"start_ms": None, "end_ms": None})
    assert _page_text(legacy).start_ms is None
