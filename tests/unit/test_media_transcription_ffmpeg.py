"""WP8 (0.5.0): transcribers and the ffmpeg helpers.

- ``OpenAICompatibleTranscriber`` against ``httpx.MockTransport``: multipart,
  retries, with and without timestamps, no ``Authorization`` without a key,
  ``verbose_json`` → ``json`` fallback remembered per instance.
- ``ffmpeg.py`` with the subprocess faked (exact arguments, errors), plus a
  real-ffmpeg test that is skipped when ffmpeg is not installed.
- ``FasterWhisperTranscriber`` with a fake ``faster_whisper`` module: the model
  is loaded once per process.
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import httpx
import pytest

from scinr.newton.converters.base import ConversionError
from scinr.newton.media import ffmpeg
from scinr.newton.media.transcribers import faster_whisper as fw_module
from scinr.newton.media.transcribers.faster_whisper import FasterWhisperTranscriber
from scinr.newton.media.transcribers.openai_compatible import (
    OpenAICompatibleTranscriber,
    TranscriptionError,
)

# ---------------------------------------------------------------------------
# OpenAI-compatible transcriber
# ---------------------------------------------------------------------------


@pytest.fixture
def server(monkeypatch):
    """Records requests; ``state["reply"](request, fields)`` builds the response."""
    state: dict = {"requests": [], "reply": None}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        fields = _multipart_fields(request.headers["content-type"], body)
        state["requests"].append((request, fields))
        return state["reply"](request, fields)

    real_client = httpx.AsyncClient

    class _Client(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return state


def _multipart_fields(content_type: str, body: bytes) -> dict[str, list[str]]:
    boundary = content_type.split("boundary=")[1].encode()
    fields: dict[str, list[str]] = {}
    for part in body.split(b"--" + boundary):
        if b"Content-Disposition" not in part:
            continue
        head, _, value = part.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        if b"filename=" in head:
            fields.setdefault(name, []).append("<file:" + value.rstrip(b"\r\n").decode() + ">")
        else:
            fields.setdefault(name, []).append(value.rstrip(b"\r\n").decode())
    return fields


@pytest.fixture
def segment_file(tmp_path) -> Path:
    path = tmp_path / "seg.flac"
    path.write_bytes(b"FLACDATA")
    return path


def _verbose(request, fields):
    return httpx.Response(200, json={
        "text": "hola mundo",
        "segments": [
            {"start": 0.0, "end": 1.25, "text": " hola"},
            {"start": 1.25, "end": 2.5, "text": " mundo "},
            {"start": 2.5, "end": 3.0, "text": "  "},
        ],
    })


class TestOpenAICompatible:
    async def test_multipart_with_timestamps(self, server, segment_file):
        server["reply"] = _verbose
        t = OpenAICompatibleTranscriber("whisper-1", api_key="sk-test", base_url="http://x/v1/")

        segments = await t.transcribe(segment_file, "es")

        assert [(s.start_ms, s.end_ms, s.text) for s in segments] == [
            (0, 1250, "hola"), (1250, 2500, "mundo"),
        ]
        ((request, fields),) = server["requests"]
        assert str(request.url) == "http://x/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer sk-test"
        assert fields["model"] == ["whisper-1"]
        assert fields["response_format"] == ["verbose_json"]
        assert fields["timestamp_granularities[]"] == ["segment"]
        assert fields["language"] == ["es"]
        assert fields["file"] == ["<file:FLACDATA>"]

    async def test_without_key_there_is_no_authorization_header(self, server, segment_file):
        server["reply"] = _verbose
        await OpenAICompatibleTranscriber("m", base_url="http://local:8000/v1").transcribe(
            segment_file
        )
        ((request, fields),) = server["requests"]
        assert "authorization" not in request.headers
        assert "language" not in fields

    async def test_rejected_verbose_json_falls_back_and_is_remembered(self, server, segment_file):
        def reply(request, fields):
            if fields["response_format"] == ["verbose_json"]:
                return httpx.Response(400, json={"error": "response_format not supported"})
            return httpx.Response(200, json={"text": " todo el tramo "})

        server["reply"] = reply
        t = OpenAICompatibleTranscriber("gpt-4o-mini-transcribe", api_key="k")

        first = await t.transcribe(segment_file)
        second = await t.transcribe(segment_file)

        assert [(s.start_ms, s.text) for s in first] == [(0, "todo el tramo")]
        assert second == first
        formats = [f["response_format"][0] for _, f in server["requests"]]
        assert formats == ["verbose_json", "json", "json"]
        assert "timestamp_granularities[]" not in server["requests"][1][1]

    async def test_retries_on_429_then_succeeds(self, server, segment_file, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", _no_sleep)
        replies = iter([httpx.Response(429), httpx.Response(503)])

        def reply(request, fields):
            return next(replies, None) or _verbose(request, fields)

        server["reply"] = reply
        segments = await OpenAICompatibleTranscriber("m", max_retries=3).transcribe(segment_file)
        assert len(segments) == 2 and len(server["requests"]) == 3

    async def test_exhausted_retries_raise(self, server, segment_file, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", _no_sleep)
        server["reply"] = lambda r, f: httpx.Response(500, text="boom")
        with pytest.raises(TranscriptionError, match="after 2 attempt"):
            await OpenAICompatibleTranscriber("m", max_retries=2).transcribe(segment_file)

    async def test_non_retryable_error_raises_at_once(self, server, segment_file):
        server["reply"] = lambda r, f: httpx.Response(401, text="bad key")
        with pytest.raises(TranscriptionError, match="401"):
            await OpenAICompatibleTranscriber("m", api_key="x").transcribe(segment_file)
        assert len(server["requests"]) == 1

    async def test_silence_gives_no_segments(self, server, segment_file):
        server["reply"] = lambda r, f: httpx.Response(200, json={"text": "", "segments": []})
        assert await OpenAICompatibleTranscriber("m").transcribe(segment_file) == []

    @pytest.mark.parametrize(
        ("threshold", "expected"),
        [(0.6, ["hola", "musica dudosa"]), (None, ["hola", "Gracias por ver", "musica dudosa"])],
    )
    async def test_segments_marked_as_silence_are_dropped(
        self, server, segment_file, threshold, expected
    ):
        server["reply"] = lambda request, fields: httpx.Response(200, json={
            "text": "hola Gracias por ver musica dudosa",
            "segments": [
                {"start": 0, "end": 1, "text": "hola", "no_speech_prob": 0.1, "avg_logprob": -0.3},
                # Whisper's silence rule: no_speech_prob > 0.6 and avg_logprob < -1
                {"start": 1, "end": 2, "text": "Gracias por ver",
                 "no_speech_prob": 0.9, "avg_logprob": -1.4},
                # Only one of the two signals: kept
                {"start": 2, "end": 3, "text": "musica dudosa",
                 "no_speech_prob": 0.9, "avg_logprob": -0.5},
            ],
        })
        transcriber = OpenAICompatibleTranscriber("whisper-1", no_speech_threshold=threshold)
        segments = await transcriber.transcribe(segment_file)
        assert [s.text for s in segments] == expected

    def test_repr_hides_the_key(self):
        assert "sk-secret" not in repr(OpenAICompatibleTranscriber("m", api_key="sk-secret"))


async def _no_sleep(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# ffmpeg helpers (subprocess faked)
# ---------------------------------------------------------------------------


class _FakeProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self._out, self._err = returncode, stdout, stderr

    async def communicate(self):
        return self._out, self._err


@pytest.fixture
def fake_exec(monkeypatch):
    state: dict = {"calls": [], "result": _FakeProcess()}

    async def _exec(*args, **kwargs):
        state["calls"].append((list(args), kwargs))
        return state["result"]

    monkeypatch.setattr(ffmpeg.asyncio, "create_subprocess_exec", _exec)
    return state


class TestFfmpegArguments:
    async def test_extract_audio_segment(self, fake_exec, tmp_path):
        await ffmpeg.extract_audio_segment(
            "/bin/ffmpeg", tmp_path / "in.mp3", 300, 300, tmp_path / "s.flac"
        )
        ((args, kwargs),) = fake_exec["calls"]
        assert args == [
            "/bin/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", "300.000", "-t", "300.000", "-i", str((tmp_path / "in.mp3").resolve()),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac",
            "-y", str((tmp_path / "s.flac").resolve()),
        ]
        assert kwargs["stdout"] is ffmpeg.asyncio.subprocess.DEVNULL
        assert args.index("-ss") < args.index("-i")  # fast seek

    async def test_extract_frames(self, fake_exec, tmp_path):
        out = tmp_path / "frames"
        await ffmpeg.extract_frames("/bin/ffmpeg", tmp_path / "v.mp4", 0, 300, 10, 1568, out)
        ((args, _),) = fake_exec["calls"]
        vf = args[args.index("-vf") + 1]
        assert vf.startswith("fps=0.033333,scale=w='min(iw,1568)'")
        assert "force_original_aspect_ratio=decrease" in vf
        assert args[args.index("-frames:v") + 1] == "10"
        assert args[-1].endswith("frame_%04d.jpg")

    async def test_probe_parses_streams(self, fake_exec, tmp_path):
        fake_exec["result"] = _FakeProcess(stdout=b"""{
          "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
             "avg_frame_rate": "30000/1001"},
            {"codec_type": "audio", "codec_name": "aac"},
            {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}
          ],
          "format": {"format_name": "mov,mp4,m4a", "duration": "125.5"}
        }""")
        info = await ffmpeg.probe("/bin/ffprobe", tmp_path / "v.mp4")
        assert info == ffmpeg.MediaInfo(
            duration_ms=125500, has_audio=True, has_video=True, container="mov",
            audio_codec="aac", video_codec="h264", width=1920, height=1080, frame_rate=29.97,
        )

    async def test_audio_with_cover_art_has_no_video(self, fake_exec, tmp_path):
        fake_exec["result"] = _FakeProcess(stdout=b"""{"streams": [
            {"codec_type": "audio", "codec_name": "mp3"},
            {"codec_type": "video", "codec_name": "png", "disposition": {"attached_pic": 1}}],
            "format": {"format_name": "mp3", "duration": "10"}}""")
        info = await ffmpeg.probe("/bin/ffprobe", tmp_path / "a.mp3")
        assert info.has_audio and not info.has_video

    async def test_non_zero_exit_is_a_conversion_error_with_stderr_tail(self, fake_exec, tmp_path):
        fake_exec["result"] = _FakeProcess(returncode=1, stderr=b"x" * 5000 + b"Invalid data")
        with pytest.raises(ConversionError, match="Invalid data") as info:
            await ffmpeg.extract_audio_segment("/bin/ffmpeg", tmp_path / "a", 0, 1, tmp_path / "b")
        assert len(str(info.value)) < 2200

    def test_missing_binary(self, tmp_path):
        with pytest.raises(ConversionError, match="ffmpeg was not found"):
            ffmpeg.find_binaries(str(tmp_path / "nope"), None)


_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg / ffprobe not installed")
class TestRealFfmpeg:
    async def test_probe_cut_and_frames(self, tmp_path):
        import subprocess

        src = tmp_path / "clip.mp4"
        subprocess.run(
            [
                shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=12",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                "-shortest", "-c:v", "mpeg4", "-c:a", "aac", "-y", str(src),
            ],
            check=True,
        )
        binaries = ffmpeg.find_binaries(None, None)
        info = await ffmpeg.probe(binaries.ffprobe, src)
        assert info.has_audio and info.has_video and abs(info.duration_ms - 12000) < 200
        seg = tmp_path / "s.flac"
        await ffmpeg.extract_audio_segment(binaries.ffmpeg, src, 5, 5, seg)
        seg_info = await ffmpeg.probe(binaries.ffprobe, seg)
        assert seg_info.audio_codec == "flac" and abs(seg_info.duration_ms - 5000) < 200
        frames = await ffmpeg.extract_frames(binaries.ffmpeg, src, 0, 10, 3, 320, tmp_path / "f")
        assert len(frames) == 3
        from PIL import Image

        with Image.open(frames[0]) as img:
            assert max(img.size) == 320


# ---------------------------------------------------------------------------
# faster-whisper (fake module)
# ---------------------------------------------------------------------------


class TestFasterWhisper:
    async def test_model_loaded_once_and_segments_mapped(self, monkeypatch, tmp_path):
        loads: list[tuple] = []

        class WhisperModel:
            def __init__(self, size, device, compute_type):
                loads.append((size, device, compute_type))

            def transcribe(self, path, language=None, **options):
                seg = types.SimpleNamespace
                return iter([seg(start=0.0, end=1.5, text=" uno "), seg(start=1.5, end=2, text="")]), None

        monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
        monkeypatch.setattr(fw_module, "_models", {})
        monkeypatch.setattr(fw_module, "_decode_pcm", lambda path: [0.0])
        audio = tmp_path / "a.flac"
        audio.write_bytes(b"x")

        a = FasterWhisperTranscriber("tiny", device="cpu")
        b = FasterWhisperTranscriber("tiny", device="cpu")
        first = await a.transcribe(audio, "es")
        await b.transcribe(audio)

        assert [(s.start_ms, s.end_ms, s.text) for s in first] == [(0, 1500, "uno")]
        assert loads == [("tiny", "cpu", "default")]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, {"language": None, "vad_filter": True}),
            (
                {"vad_parameters": {"min_silence_duration_ms": 500}},
                {"language": None, "vad_filter": True,
                 "vad_parameters": {"min_silence_duration_ms": 500}},
            ),
            ({"vad_filter": False, "vad_parameters": {"x": 1}},
             {"language": None, "vad_filter": False}),
        ],
    )
    async def test_voice_activity_filter_is_on_by_default(
        self, monkeypatch, tmp_path, kwargs, expected
    ):
        seen: list[dict] = []

        class WhisperModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, path, **options):
                seen.append(options)
                return iter([]), None

        monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
        monkeypatch.setattr(fw_module, "_models", {})
        monkeypatch.setattr(fw_module, "_decode_pcm", lambda path: [0.0])
        audio = tmp_path / "a.flac"
        audio.write_bytes(b"x")

        assert await FasterWhisperTranscriber("tiny", **kwargs).transcribe(audio) == []
        assert seen == [expected]

    def test_audio_is_decoded_with_ffmpeg_not_pyav(self, tmp_path):
        import shutil as _shutil

        if _shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        import subprocess as _sp

        from scinr.newton.config import configure

        configure(neo4j_user="n", neo4j_password="p", neo4j_database="n")
        audio = tmp_path / "tono.m4a"
        _sp.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                 "sine=frequency=440:duration=2", str(audio)], check=True)
        samples = fw_module._decode_pcm(audio)
        assert str(samples.dtype) == "float32"
        assert abs(len(samples) - 32_000) < 1_600  # 2 s at 16 kHz, mono

    async def test_missing_package_says_how_to_install(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        monkeypatch.setattr(fw_module, "_models", {})
        monkeypatch.setattr(fw_module, "_decode_pcm", lambda path: [0.0])
        with pytest.raises(ImportError, match=r"scinr\[media\]"):
            await FasterWhisperTranscriber().transcribe(tmp_path / "a.flac")
