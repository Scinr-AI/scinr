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
from scinr.newton.converters import main as converters_main  # noqa: E402
from scinr.newton.converters.audio import AudioConverter  # noqa: E402
from scinr.newton.converters.base import ConversionError  # noqa: E402
from scinr.newton.converters.enrichment import render_image_block  # noqa: E402
from scinr.newton.converters.video import VideoConverter  # noqa: E402
from scinr.newton.media import ffmpeg  # noqa: E402
from scinr.newton.media.transcriber import Transcriber, TranscriptSegment  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}
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


@pytest.fixture
def fake_ffmpeg(monkeypatch, tmp_path):
    state: dict = {
        "info": ffmpeg.MediaInfo(duration_ms=650_000, has_audio=True, has_video=False,
                                 container="mp3", audio_codec="mp3"),
        "segment_files": [],
        "frame_files": [],
        "frame_colours": lambda start_s, k: GREEN,
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
