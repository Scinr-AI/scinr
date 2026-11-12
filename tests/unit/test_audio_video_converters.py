"""WP9 / WP10 (0.5.0): audio and video converters with fake ffmpeg and transcriber."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from test_converters_enrichment import GREEN, RED, FakeVision  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.converters import image_store  # noqa: E402
from scinr.newton.converters import main as converters_main  # noqa: E402
from scinr.newton.converters.audio import AudioConverter  # noqa: E402
from scinr.newton.converters.base import ConversionError  # noqa: E402
from scinr.newton.converters.docx import DocxConverter  # noqa: E402
from scinr.newton.converters.enrichment import render_image_block  # noqa: E402
from scinr.newton.converters.video import VideoConverter  # noqa: E402
from scinr.newton.media import ffmpeg  # noqa: E402
from scinr.newton.media.transcriber import Transcriber, TranscriptSegment  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)


class FakeTranscriber(Transcriber):
    """Reads the fake segment file (its start second) and answers from a script."""

    def __init__(self, script=None):
        self.calls: list[tuple[str, str | None]] = []
        self.script = script or {}

    async def transcribe(self, path: Path, language=None):
        start = int(path.read_text())
        self.calls.append((path.name, language))
        answer = self.script.get(start, f"habla desde {start}")
        if isinstance(answer, Exception):
            raise answer
        if not answer:
            return []
        return [
            TranscriptSegment(start_ms=0, end_ms=1000, text=answer),
            TranscriptSegment(start_ms=1000, end_ms=2000, text="  "),
        ]


def _all_green(start_s, k):
    return GREEN


@pytest.fixture
def fake_ffmpeg(monkeypatch, tmp_path):
    state: dict = {
        "info": ffmpeg.MediaInfo(duration_ms=650_000, has_audio=True, has_video=False,
                                 container="mp3", audio_codec="mp3"),
        "segment_files": [],
        "frame_files": [],
        "frame_colours": _all_green,
        "frame_requests": [],
    }

    monkeypatch.setattr(
        ffmpeg, "find_binaries", lambda a, b: ffmpeg.Binaries(ffmpeg="/ff", ffprobe="/fp")
    )

    async def _probe(ffprobe, path):
        return state["info"]

    async def _extract_audio(ff, src, start_s, duration_s, dst):
        dst.write_text(str(int(start_s)))
        state["segment_files"].append(dst)

    async def _extract_frames(ff, src, start_s, duration_s, count, max_side, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        state["frame_requests"].append((start_s, duration_s, count, max_side))
        files = []
        for k in range(count):
            path = out_dir / f"frame_{k + 1:04d}.jpg"
            buf = io.BytesIO()
            Image.new("RGB", (160, 90), state["frame_colours"](start_s, k)).save(buf, "JPEG")
            # Distinct bytes per frame: identical consecutive frames are skipped
            # by their digest, so a request that wants them all described needs
            # them all different. (The default is all-green: kept only k == 0.)
            if state["frame_colours"] is not _all_green:
                buf.write(b"\xff\xd9" + k.to_bytes(2, "big"))
            path.write_bytes(buf.getvalue())
            files.append(path)
            state["frame_files"].append(path)
        return files

    monkeypatch.setattr(ffmpeg, "probe", _probe)
    monkeypatch.setattr(ffmpeg, "extract_audio_segment", _extract_audio)
    monkeypatch.setattr(ffmpeg, "extract_frames", _extract_frames)
    return state


@pytest.fixture
def media_file(tmp_path):
    def _make(name):
        path = tmp_path / name
        path.write_bytes(b"not really media")
        return path

    return _make


@pytest.fixture
def vision():
    model = FakeVision()
    model.calls = []
    return model


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration_ms", "segment_s", "min_s", "expected"),
    [
        # 300.05 s in 300 s segments: the cut moves to 290 s, the tail lasts 10.05 s
        (300_050, 300, 10, [(0, 290_000), (290_000, 300_050)]),
        # a tail of exactly the minimum is kept as it is
        (310_000, 300, 10, [(0, 300_000), (300_000, 310_000)]),
        # 0 disables the adjustment
        (300_050, 300, 0, [(0, 300_000), (300_000, 300_050)]),
        # only the last cut moves
        (600_500, 300, 10, [(0, 300_000), (300_000, 590_000), (590_000, 600_500)]),
        # a medium shorter than the minimum is one segment, untouched
        (3_000, 300, 10, [(0, 3_000)]),
        # the minimum never exceeds half a segment
        (20_200, 10, 10, [(0, 10_000), (10_000, 15_000), (15_000, 20_200)]),
    ],
)
def test_segments_never_end_with_a_tiny_tail(duration_ms, segment_s, min_s, expected):
    from scinr.newton.converters.media_common import iter_segments

    segments = list(iter_segments(duration_ms, segment_s, min_s))
    assert [(s.start_ms, s.end_ms) for s in segments] == expected
    assert [s.index for s in segments] == list(range(len(expected)))
    assert all(s.end_ms - s.start_ms <= segment_s * 1000 for s in segments)


async def test_audio_tail_is_merged_through_configure(fake_ffmpeg, media_file):
    fake_ffmpeg["info"] = ffmpeg.MediaInfo(
        duration_ms=600_050, has_audio=True, has_video=False, container="mp3"
    )
    configure(**_NEO, transcriber=FakeTranscriber())
    doc = await AudioConverter().convert(media_file("a.mp3"))
    assert [(p.start_ms, p.end_ms) for p in doc.pages] == [
        (0, 300_000), (300_000, 590_000), (590_000, 600_050),
    ]
    assert doc.pages[-1].markdown.startswith("[00:09:50 – 00:10:00]")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class TestAudio:
    async def test_segments_times_and_markdown(self, fake_ffmpeg, media_file):
        transcriber = FakeTranscriber()
        configure(**_NEO, transcriber=transcriber, media_language="es")

        doc = await AudioConverter().convert(media_file("charla.mp3"))

        assert [(p.index, p.start_ms, p.end_ms) for p in doc.pages] == [
            (0, 0, 300_000), (1, 300_000, 600_000), (2, 600_000, 650_000),
        ]
        assert doc.pages[1].markdown == "[00:05:00 – 00:10:00]\n\nhabla desde 300"
        assert doc.pages[2].markdown.startswith("[00:10:00 – 00:10:50]")
        assert all(lang == "es" for _, lang in transcriber.calls)
        assert doc.file_metadata.duration_ms == 650_000
        assert doc.file_metadata.audio_codec == "mp3"
        assert doc.missing_segments is None
        assert not any(p.exists() for p in fake_ffmpeg["segment_files"])

    async def test_silence_gives_no_page_and_keeps_indexes(self, fake_ffmpeg, media_file):
        configure(**_NEO, transcriber=FakeTranscriber({300: ""}))
        doc = await AudioConverter().convert(media_file("a.wav"))
        assert [p.index for p in doc.pages] == [0, 2]

    async def test_best_effort_records_the_segment(self, fake_ffmpeg, media_file):
        configure(**_NEO, transcriber=FakeTranscriber({300: RuntimeError("timeout")}))
        doc = await AudioConverter().convert(media_file("a.mp3"))
        assert [p.index for p in doc.pages] == [0, 2]
        assert doc.missing_segments == [(300_000, 600_000)]
        assert not any(p.exists() for p in fake_ffmpeg["segment_files"])

    async def test_fail_fast_names_the_range(self, fake_ffmpeg, media_file):
        configure(
            **_NEO, transcriber=FakeTranscriber({300: RuntimeError("timeout")}),
            media_error_strategy="fail_fast",
        )
        with pytest.raises(ConversionError, match=r"00:05:00 – 00:10:00"):
            await AudioConverter().convert(media_file("a.mp3"))
        assert not any(p.exists() for p in fake_ffmpeg["segment_files"])

    async def test_duration_limit_before_any_transcription(self, fake_ffmpeg, media_file):
        transcriber = FakeTranscriber()
        configure(**_NEO, transcriber=transcriber, max_media_duration_seconds=600)
        with pytest.raises(ConversionError, match="max_media_duration_seconds"):
            await AudioConverter().convert(media_file("a.mp3"))
        assert transcriber.calls == []

    async def test_no_audio_track(self, fake_ffmpeg, media_file):
        configure(**_NEO, transcriber=FakeTranscriber())
        fake_ffmpeg["info"] = ffmpeg.MediaInfo(duration_ms=1000, has_audio=False, has_video=True)
        with pytest.raises(ConversionError, match="no audio track"):
            await AudioConverter().convert(media_file("a.m4a"))

    async def test_without_transcriber_the_message_names_it(self, media_file):
        configure(**_NEO)
        with pytest.raises(ConversionError, match="transcriber"):
            await AudioConverter().convert(media_file("a.mp3"))

    async def test_time_range_reaches_store_page(self, fake_ffmpeg, media_file, monkeypatch):
        configure(**_NEO, transcriber=FakeTranscriber())
        monkeypatch.setattr(converters_main, "get_converter", lambda p: AudioConverter())
        seen: list[dict] = []

        class _Pages:
            async def store_page(self, *args, **kwargs):
                seen.append(kwargs)
                return "p"

        class _Raw:
            async def store_file(self, *a, **k):
                return "r"

        await converters_main.convert_one(
            media_file("a.mp3"), None, raw_file_repo=_Raw(), page_repo=_Pages()
        )
        assert [(k["start_ms"], k["end_ms"]) for k in seen] == [
            (0, 300_000), (300_000, 600_000), (600_000, 650_000),
        ]


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _video(duration_ms=360_000, has_audio=True):
    return ffmpeg.MediaInfo(
        duration_ms=duration_ms, has_audio=has_audio, has_video=True, container="mov",
        audio_codec="aac" if has_audio else None, video_codec="h264", width=1920, height=1080,
        frame_rate=25.0,
    )


class TestVideo:
    async def test_transcription_and_frames_per_segment(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video()
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE

        doc = await VideoConverter().convert(media_file("clase.mp4"))

        assert [(p.index, p.start_ms, p.end_ms) for p in doc.pages] == [
            (0, 0, 300_000), (1, 300_000, 360_000),
        ]
        # 2 frames/min: 10 frames in 5 min, 2 in the last minute.
        assert [r[2] for r in fake_ffmpeg["frame_requests"]] == [10, 2]
        assert fake_ffmpeg["frame_requests"][0][3] == 1568
        first = doc.pages[0].markdown
        assert first.startswith("[00:00:00 – 00:05:00]\n\nhabla desde 0\n\n")
        assert render_image_block(1, "desc 0,0,5", time="00:00:30") in first
        assert first.count("<image") == 10
        assert doc.file_metadata.video_codec == "h264" and doc.file_metadata.frame_rate == 25.0
        assert not any(p.exists() for p in fake_ffmpeg["frame_files"])

    async def test_repeated_frames_are_skipped(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video(duration_ms=120_000)
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        doc = await VideoConverter().convert(media_file("v.mov"))  # all frames GREEN
        assert len(vision.calls) == 1
        assert doc.pages[0].markdown.count("<image") == 1

    async def test_video_without_audio_uses_frames_only(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video(duration_ms=60_000, has_audio=False)
        configure(**_NEO, vision_llm=vision)  # no transcriber needed
        doc = await VideoConverter().convert(media_file("v.webm"))
        assert doc.pages[0].markdown.startswith("[00:00:00 – 00:01:00]\n\n<image")
        assert fake_ffmpeg["segment_files"] == []

    async def test_frames_disabled_means_transcription_only(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video()
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision,
                  video_frames_per_minute=0)
        doc = await VideoConverter().convert(media_file("v.mkv"))
        assert fake_ffmpeg["frame_requests"] == [] and vision.calls == []
        assert "<image" not in doc.pages[0].markdown

    async def test_without_vision_llm_frames_are_skipped(self, fake_ffmpeg, media_file):
        fake_ffmpeg["info"] = _video()
        configure(**_NEO, transcriber=FakeTranscriber())
        doc = await VideoConverter().convert(media_file("v.mp4"))
        assert fake_ffmpeg["frame_requests"] == []
        assert doc.pages[0].markdown == "[00:00:00 – 00:05:00]\n\nhabla desde 0"

    async def test_audio_without_transcriber_fails(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video()
        configure(**_NEO, vision_llm=vision)
        with pytest.raises(ConversionError, match="transcriber"):
            await VideoConverter().convert(media_file("v.mp4"))

    async def test_nothing_to_extract_fails(self, fake_ffmpeg, media_file):
        fake_ffmpeg["info"] = _video(has_audio=False)
        configure(**_NEO)
        with pytest.raises(ConversionError, match="Nothing to extract"):
            await VideoConverter().convert(media_file("v.mp4"))

    async def test_failed_frame_best_effort(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        fake_ffmpeg["frame_colours"] = lambda start, k: RED if k == 1 else GREEN
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        doc = await VideoConverter().convert(media_file("v.mp4"))
        assert doc.missing_images == [(0, 1)]
        assert doc.pages[0].markdown.count("<image") == 1

    async def test_failed_frame_extraction_best_effort_keeps_the_video(
        self, fake_ffmpeg, media_file, vision, monkeypatch
    ):
        fake_ffmpeg["info"] = _video(duration_ms=600_000)  # two 5-minute segments
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        real = ffmpeg.extract_frames

        async def _broken_second(ff, src, start_s, *args):
            if start_s >= 300:
                raise ConversionError("ffmpeg failed: corrupt packet")
            return await real(ff, src, start_s, *args)

        monkeypatch.setattr(ffmpeg, "extract_frames", _broken_second)
        doc = await VideoConverter().convert(media_file("v.mp4"))

        # Every frame planned for the second segment (2 per minute x 5) is missing,
        # its transcription is kept, and the first segment is complete.
        assert doc.missing_images == [(1, k) for k in range(10)]
        assert doc.missing_segments is None
        assert [p.index for p in doc.pages] == [0, 1]
        assert "<image" in doc.pages[0].markdown
        assert "<image" not in doc.pages[1].markdown
        assert "habla desde 300" in doc.pages[1].markdown

    async def test_failed_frame_extraction_fail_fast(
        self, fake_ffmpeg, media_file, vision, monkeypatch
    ):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        configure(
            **_NEO, transcriber=FakeTranscriber(), vision_llm=vision,
            media_error_strategy="fail_fast",
        )

        async def _broken(*args):
            raise ConversionError("ffmpeg failed: corrupt packet")

        monkeypatch.setattr(ffmpeg, "extract_frames", _broken)
        with pytest.raises(ConversionError, match=r"Frames of segment \[00:00:00 – 00:01:00\]"):
            await VideoConverter().convert(media_file("v.mp4"))

    async def test_failed_transcription_keeps_the_frames(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        configure(**_NEO, transcriber=FakeTranscriber({0: RuntimeError("x")}), vision_llm=vision)
        doc = await VideoConverter().convert(media_file("v.mp4"))
        assert doc.missing_segments == [(0, 60_000)]
        assert "<image" in doc.pages[0].markdown

    # ------------------------------------------------------------------
    # WP5 5a: the segment's transcript is the context of its frames (D6)
    # ------------------------------------------------------------------

    async def test_frames_receive_the_segment_transcript_as_context(
        self, fake_ffmpeg, media_file, vision
    ):
        fake_ffmpeg["info"] = _video(duration_ms=360_000)
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)

        doc = await VideoConverter().convert(media_file("clase.mp4"))

        assert len(vision.calls) == 12  # 10 frames in segment 0, 2 in segment 1
        assert all(
            "<transcript>\nhabla desde 0\n</transcript>" in c["text"]
            for c in vision.calls[:10]
        )
        assert all(
            "<transcript>\nhabla desde 300\n</transcript>" in c["text"]
            for c in vision.calls[10:]
        )
        assert "habla desde 0" in doc.pages[0].markdown  # the transcript is kept as text

    async def test_video_without_audio_gives_no_context(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video(duration_ms=60_000, has_audio=False)
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        configure(**_NEO, vision_llm=vision)  # no transcriber

        await VideoConverter().convert(media_file("v.webm"))

        assert len(vision.calls) == 2
        assert all("<transcript>" not in c["text"] for c in vision.calls)

    async def test_a_failed_transcription_gives_no_context(
        self, fake_ffmpeg, media_file, vision
    ):
        fake_ffmpeg["info"] = _video(duration_ms=360_000)  # two segments
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        configure(
            **_NEO, transcriber=FakeTranscriber({0: RuntimeError("timeout")}),
            vision_llm=vision,
        )

        await VideoConverter().convert(media_file("v.mp4"))

        assert "<transcript>" not in vision.calls[0]["text"]  # failed in best_effort
        assert "<transcript>\nhabla desde 300\n</transcript>" in vision.calls[-1]["text"]


# ---------------------------------------------------------------------------
# WP5 5b: the described frames are stored with their instant (D5)
# ---------------------------------------------------------------------------


class _RecordingRawRepo:
    """In-memory ``RawFileRepository`` that records the image calls.

    Duck-typed, like the one of ``test_stored_images_ingest``: it implements
    ``stores_images`` and the image methods of the optional block explicitly,
    because that is what the ingestion calls.
    """

    def __init__(self) -> None:
        self.raw_file_id = ""
        self.images: list[dict] = []
        self.deleted: list[str] = []
        self.fail_on_image = 0  # ordinal (1-based) that raises, 0 = never

    @property
    def stores_images(self) -> bool:
        return True

    async def store_file(self, path, filename, content_type, folder_path, **kwargs) -> str:
        self.raw_file_id = f"raw-{filename}"
        return self.raw_file_id

    async def store_image(
        self, raw_file_id, page_index, image_index, content, **kwargs
    ) -> str:
        if len(self.images) + 1 == self.fail_on_image:
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
        self.deleted.append(raw_file_id)
        return len(self.images)


def _docx_with_image(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    buf = io.BytesIO()
    Image.new("RGB", (200, 120), GREEN).save(buf, format="PNG")
    document = docx.Document()
    document.add_picture(io.BytesIO(buf.getvalue()))
    path = tmp_path / "informe.docx"
    document.save(path)
    return path


class TestFramesAreStored:
    async def test_described_frames_are_stored_with_their_time(
        self, fake_ffmpeg, media_file, vision, monkeypatch
    ):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)  # 2 frames/min → k = 0, 1
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        monkeypatch.setattr(converters_main, "get_converter", lambda path: VideoConverter())
        repo = _RecordingRawRepo()
        # Track what the fake ffmpeg wrote, per frame file (they are deleted
        # after being uploaded, so read them as they are written).
        frame_bytes: list[bytes] = []
        real_extract = ffmpeg.extract_frames

        async def _tracking_extract(*args):
            files = await real_extract(*args)
            frame_bytes.extend(f.read_bytes() for f in files[len(frame_bytes):])
            return files

        monkeypatch.setattr(ffmpeg, "extract_frames", _tracking_extract)

        written, failures = await converters_main.convert_one(
            media_file("v.mp4"), None, raw_file_repo=repo
        )

        assert failures == [] and len(written) == 1
        assert [(i["page_index"], i["image_index"]) for i in repo.images] == [(0, 0), (0, 1)]
        assert all(i["kind"] == "frame" for i in repo.images)
        assert [i["time_ms"] for i in repo.images] == [0, 30_000]
        assert all(i["content_type"] == "image/jpeg" for i in repo.images)
        assert all((i["width"], i["height"]) == (160, 90) for i in repo.images)
        assert [i["description"] for i in repo.images] == ["desc 0,3,0", "desc 0,0,5"]
        # The uploaded bytes are exactly what the fake ffmpeg extracted (the
        # whole frame file, streamed from disk before it is deleted).
        assert [i["content"] for i in repo.images] == frame_bytes
        assert not any(p.exists() for p in fake_ffmpeg["frame_files"])

    async def test_an_identical_frame_is_not_stored(self, fake_ffmpeg, media_file, vision):
        fake_ffmpeg["info"] = _video(duration_ms=120_000)  # 4 frames, all GREEN
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        repo = _RecordingRawRepo()
        with image_store.image_sink(repo, "rid", filename="v.mov") as sink:
            doc = await VideoConverter().convert(media_file("v.mov"))
        assert doc.pages[0].markdown.count("<image") == 1
        assert [(i["image_index"], i["time_ms"]) for i in repo.images] == [(0, 0)]
        assert sink is not None and sink.stored == 1

    async def test_frames_switch_off_stores_nothing_but_a_docx_still_is_stored(
        self, fake_ffmpeg, media_file, vision, monkeypatch, tmp_path
    ):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        configure(
            **_NEO, transcriber=FakeTranscriber(), vision_llm=vision,
            store_video_frames=False, store_described_images=True,
        )
        repo = _RecordingRawRepo()
        monkeypatch.setattr(converters_main, "get_converter", lambda path: VideoConverter())
        written, failures = await converters_main.convert_one(
            media_file("v.mp4"), None, raw_file_repo=repo
        )
        assert failures == [] and len(written) == 1
        assert "<image" in written[0][2].pages[0].markdown  # described…
        assert repo.images == []  # …but not stored

        monkeypatch.setattr(converters_main, "get_converter", lambda path: DocxConverter())
        written, failures = await converters_main.convert_one(
            _docx_with_image(tmp_path), None, raw_file_repo=repo
        )
        assert failures == []
        assert [(i["kind"], i["image_index"]) for i in repo.images] == [("embedded", 0)]

    async def test_a_storage_failure_fails_the_video_and_cleans_up(
        self, fake_ffmpeg, media_file, vision, monkeypatch
    ):
        fake_ffmpeg["info"] = _video(duration_ms=360_000)  # two segments
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        monkeypatch.setattr(converters_main, "get_converter", lambda path: VideoConverter())
        repo = _RecordingRawRepo()
        repo.fail_on_image = 2  # the second stored frame fails

        written, failures = await converters_main.convert_one(
            media_file("v.mp4"), None, raw_file_repo=repo
        )

        assert written == []
        (failed_path, error), = failures
        assert failed_path == media_file("v.mp4")
        assert "could not be stored" in error and "disk full" in error
        assert len(repo.images) == 1  # one frame was stored…
        assert repo.deleted == ["raw-v.mp4"]  # …and cleaned up (D5)
        assert not any(p.exists() for p in fake_ffmpeg["frame_files"])

    async def test_a_storage_failure_fails_the_video_even_in_fail_fast(
        self, fake_ffmpeg, media_file, vision, monkeypatch
    ):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        # Digits make the two frames distinct: the dedup would skip the second.
        configure(
            **_NEO, transcriber=FakeTranscriber(), vision_llm=vision,
            media_error_strategy="fail_fast",
        )
        monkeypatch.setattr(converters_main, "get_converter", lambda path: VideoConverter())
        repo = _RecordingRawRepo()
        repo.fail_on_image = 1
        written, failures = await converters_main.convert_one(
            media_file("v.mp4"), None, raw_file_repo=repo
        )
        assert written == []
        (_, error), = failures
        assert "could not be stored" in error

    async def test_the_frame_file_is_deleted_after_being_uploaded(
        self, fake_ffmpeg, media_file, vision
    ):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        fake_ffmpeg["frame_colours"] = lambda start, k: GREEN if k % 2 == 0 else BLUE
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        repo = _RecordingRawRepo()
        seen: list[bool] = []  # frame file still on disk when its upload runs?
        real_store = repo.store_image_file

        async def _recording_store(path, *args, **kwargs):
            seen.append(path.exists())
            return await real_store(path, *args, **kwargs)

        repo.store_image_file = _recording_store  # type: ignore[method-assign]

        with image_store.image_sink(repo, "rid", filename="v.mp4"):
            await VideoConverter().convert(media_file("v.mp4"))

        assert seen == [True] * len(seen) and len(seen) == 2  # streamed from disk
        assert not any(p.exists() for p in fake_ffmpeg["frame_files"])  # deleted after

    async def test_a_white_frame_gets_no_description_and_is_not_stored(
        self, fake_ffmpeg, media_file, vision
    ):
        fake_ffmpeg["info"] = _video(duration_ms=60_000)
        fake_ffmpeg["frame_colours"] = lambda start, k: WHITE if k == 1 else GREEN
        configure(**_NEO, transcriber=FakeTranscriber(), vision_llm=vision)
        repo = _RecordingRawRepo()
        with image_store.image_sink(repo, "rid", filename="v.mp4") as sink:
            doc = await VideoConverter().convert(media_file("v.mp4"))
        assert [i["image_index"] for i in repo.images] == [0]
        assert sink is not None and sink.stored == 1
        assert doc.pages[0].markdown.count("<image") == 1
        assert doc.missing_images is None  # an empty description is not a failure


# ---------------------------------------------------------------------------
# WP6: transcript cues with absolute times (D7)
# ---------------------------------------------------------------------------


class _TimestampedTranscriber(Transcriber):
    """Answers with several timestamped sentences per segment."""

    async def transcribe(self, path: Path, language=None):
        start = int(path.read_text())
        return [
            TranscriptSegment(start_ms=0, end_ms=5_000, text=f"first at {start}"),
            TranscriptSegment(start_ms=5_000, end_ms=10_000, text=f"second at {start}"),
        ]


class TestTranscriptCues:
    async def test_cues_have_absolute_times_clipped_to_the_segment(
        self, fake_ffmpeg, media_file
    ):
        fake_ffmpeg["info"] = ffmpeg.MediaInfo(
            duration_ms=650_000, has_audio=True, has_video=False, container="mp3"
        )
        configure(**_NEO, transcriber=_TimestampedTranscriber())

        doc = await AudioConverter().convert(media_file("a.mp3"))

        assert [(p.start_ms, p.end_ms) for p in doc.pages] == [
            (0, 300_000), (300_000, 600_000), (600_000, 650_000),
        ]
        # Second segment: cues start at its start_ms (absolute), clipped to it.
        second = doc.pages[1]
        assert second.transcript is not None
        assert [(c.start_ms, c.end_ms) for c in second.transcript] == [
            (300_000, 305_000), (305_000, 310_000),
        ]
        assert second.transcript[0].text == "first at 300"

    async def test_a_transcriber_without_timestamps_gives_one_cue(
        self, fake_ffmpeg, media_file
    ):
        class _NoTimestamps(Transcriber):
            async def transcribe(self, path: Path, language=None):
                return [TranscriptSegment(start_ms=0, end_ms=0, text="no times here")]

        fake_ffmpeg["info"] = ffmpeg.MediaInfo(
            duration_ms=60_000, has_audio=True, has_video=False, container="mp3"
        )
        configure(**_NEO, transcriber=_NoTimestamps())

        doc = await AudioConverter().convert(media_file("a.mp3"))

        assert len(doc.pages) == 1
        (cue,) = doc.pages[0].transcript
        assert (cue.start_ms, cue.end_ms) == (0, 60_000)
        assert cue.text == "no times here"

    async def test_markdown_is_unchanged(self, fake_ffmpeg, media_file):
        fake_ffmpeg["info"] = ffmpeg.MediaInfo(
            duration_ms=650_000, has_audio=True, has_video=False, container="mp3"
        )
        configure(**_NEO, transcriber=FakeTranscriber())

        doc = await AudioConverter().convert(media_file("a.mp3"))

        assert doc.pages[1].markdown == "[00:05:00 – 00:10:00]\n\nhabla desde 300"
        assert doc.pages[2].markdown.startswith("[00:10:00 – 00:10:50]")

    async def test_page_transcript_reaches_store_page(self, fake_ffmpeg, media_file, monkeypatch):
        fake_ffmpeg["info"] = ffmpeg.MediaInfo(
            duration_ms=650_000, has_audio=True, has_video=False, container="mp3"
        )
        configure(**_NEO, transcriber=FakeTranscriber())
        monkeypatch.setattr(converters_main, "get_converter", lambda p: AudioConverter())
        seen: list[dict] = []

        class _Pages:
            async def store_page(self, *args, **kwargs):
                seen.append(kwargs)
                return "p"

        class _Raw:
            async def store_file(self, *a, **k):
                return "r"

        await converters_main.convert_one(
            media_file("a.mp3"), None, raw_file_repo=_Raw(), page_repo=_Pages()
        )
        assert [(k["start_ms"], k["end_ms"]) for k in seen] == [
            (0, 300_000), (300_000, 600_000), (600_000, 650_000),
        ]
        assert all("transcript" in k for k in seen)
        # The fake transcriber returns one real segment and one whitespace-only
        # one: only the real one is kept, with absolute times.
        assert seen[1]["transcript"] == [
            {"start_ms": 300_000, "end_ms": 301_000, "text": "habla desde 300"},
        ]
