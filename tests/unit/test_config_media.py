"""WP3 (0.5.0): multimodal configuration (vision, transcription, media limits)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import scinr.newton.config as config_module
from scinr.newton.config import (
    configure,
    embedded_images_enabled,
    get_transcriber,
    get_vision_llm,
)
from scinr.newton.exceptions import ConfigurationError
from scinr.newton.media.transcriber import Transcriber, TranscriptSegment

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}

_ENV_VARS = (
    "DESCRIBE_EMBEDDED_IMAGES", "MEDIA_ERROR_STRATEGY", "STRIP_LOCATION",
    "IMAGE_MAX_SIDE_PX", "IMAGE_MIN_SIDE_PX", "MAX_IMAGES_PER_PAGE", "MEDIA_SEGMENT_SECONDS",
    "MEDIA_LANGUAGE", "MAX_MEDIA_DURATION_SECONDS", "VIDEO_FRAMES_PER_MINUTE",
    "MEDIA_CONCURRENCY", "MISTRAL_OCR_IMAGE_MAX_PAGES", "FFMPEG_PATH", "FFPROBE_PATH",
)


@pytest.fixture(autouse=True)
def _clean_media_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _chat_model():
    from langchain_core.language_models import BaseChatModel

    class _Vision(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "vision"

        def _generate(self, *args, **kwargs):  # pragma: no cover - never called
            raise NotImplementedError

    return _Vision()


class _Transcriber(Transcriber):
    async def transcribe(self, path: Path, language: str | None = None):
        return [TranscriptSegment(start_ms=0, end_ms=1, text="x")]


class TestDefaults:
    def test_defaults(self):
        cfg = configure(**_NEO)
        assert cfg.vision_llm is None and cfg.transcriber is None
        assert cfg.describe_embedded_images is True
        assert cfg.media_error_strategy == "best_effort"
        assert cfg.strip_location is True
        assert (cfg.image_max_side_px, cfg.image_min_side_px, cfg.max_images_per_page) == (
            1568, 64, 10,
        )
        assert cfg.media_segment_seconds == 300
        assert cfg.media_min_segment_seconds == 10
        assert cfg.image_decode_concurrency == 1
        assert cfg.media_language is None
        assert cfg.max_media_duration_seconds == 14400
        assert cfg.video_frames_per_minute == 2.0
        assert cfg.media_concurrency == 1
        assert cfg.mistral_ocr_image_max_pages == 50
        assert cfg.ffmpeg_path is None and cfg.ffprobe_path is None

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("DESCRIBE_EMBEDDED_IMAGES", "false")
        monkeypatch.setenv("MEDIA_ERROR_STRATEGY", "fail_fast")
        monkeypatch.setenv("STRIP_LOCATION", "false")
        monkeypatch.setenv("IMAGE_MAX_SIDE_PX", "1024")
        monkeypatch.setenv("IMAGE_MIN_SIDE_PX", "32")
        monkeypatch.setenv("MAX_IMAGES_PER_PAGE", "3")
        monkeypatch.setenv("MEDIA_SEGMENT_SECONDS", "60")
        monkeypatch.setenv("MEDIA_MIN_SEGMENT_SECONDS", "20")
        monkeypatch.setenv("IMAGE_DECODE_CONCURRENCY", "3")
        monkeypatch.setenv("MEDIA_LANGUAGE", "ES")
        monkeypatch.setenv("MAX_MEDIA_DURATION_SECONDS", "none")
        monkeypatch.setenv("VIDEO_FRAMES_PER_MINUTE", "0")
        monkeypatch.setenv("MEDIA_CONCURRENCY", "2")
        monkeypatch.setenv("MISTRAL_OCR_IMAGE_MAX_PAGES", "20")
        monkeypatch.setenv("FFMPEG_PATH", "/opt/ffmpeg")
        monkeypatch.setenv("FFPROBE_PATH", "/opt/ffprobe")

        cfg = configure(**_NEO)

        assert cfg.describe_embedded_images is False
        assert cfg.media_error_strategy == "fail_fast"
        assert cfg.strip_location is False
        assert (cfg.image_max_side_px, cfg.image_min_side_px, cfg.max_images_per_page) == (
            1024, 32, 3,
        )
        assert cfg.media_segment_seconds == 60
        assert cfg.media_min_segment_seconds == 20
        assert cfg.image_decode_concurrency == 3
        assert cfg.media_language == "es"
        assert cfg.max_media_duration_seconds is None
        assert cfg.video_frames_per_minute == 0.0
        assert cfg.media_concurrency == 2
        assert cfg.mistral_ocr_image_max_pages == 20
        assert (cfg.ffmpeg_path, cfg.ffprobe_path) == ("/opt/ffmpeg", "/opt/ffprobe")

    def test_explicit_none_disables_the_duration_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_MEDIA_DURATION_SECONDS", "60")
        assert configure(**_NEO, max_media_duration_seconds=None).max_media_duration_seconds is None
        assert configure(**_NEO).max_media_duration_seconds == 60


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"media_error_strategy": "retry"},
            {"media_segment_seconds": 9},
            {"image_max_side_px": 10},
            {"image_min_side_px": -1},
            {"max_images_per_page": 0},
            {"media_language": "spanish"},
            {"max_media_duration_seconds": 0},
            {"video_frames_per_minute": -1},
            {"media_concurrency": 0},
            {"image_decode_concurrency": 0},
            {"media_min_segment_seconds": -1},
            {"media_min_segment_seconds": 151},  # more than half of the default 300
            {"media_segment_seconds": 20, "media_min_segment_seconds": 11},
            {"mistral_ocr_image_max_pages": 0},
            {"vision_llm": object()},
            {"transcriber": object()},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ConfigurationError):
            configure(**_NEO, **kwargs)

    def test_default_min_segment_is_capped_to_half_a_segment(self, monkeypatch):
        # Not given: the default 10 s shrinks for short segments instead of failing.
        assert configure(**_NEO, media_segment_seconds=15).media_min_segment_seconds == 7
        assert configure(**_NEO, media_segment_seconds=20).media_min_segment_seconds == 10
        assert configure(
            **_NEO, media_segment_seconds=20, media_min_segment_seconds=0
        ).media_min_segment_seconds == 0
        monkeypatch.setenv("MEDIA_MIN_SEGMENT_SECONDS", "8")  # given through the env: checked
        with pytest.raises(ConfigurationError, match="at most half"):
            configure(**_NEO, media_segment_seconds=15)

    def test_bad_env_integer_raises(self, monkeypatch):
        monkeypatch.setenv("MEDIA_SEGMENT_SECONDS", "five")
        with pytest.raises(ConfigurationError, match="MEDIA_SEGMENT_SECONDS"):
            configure(**_NEO)

    def test_vision_llm_does_not_need_structured_output(self):
        vision = _chat_model()
        assert configure(**_NEO, vision_llm=vision).vision_llm is vision


class TestAccessors:
    def test_get_vision_llm_does_not_fall_back_to_llm(self):
        configure(**_NEO, llm=_chat_model())
        with pytest.raises(ConfigurationError, match="vision_llm"):
            get_vision_llm()

    def test_get_vision_llm_returns_it(self):
        vision = _chat_model()
        configure(**_NEO, vision_llm=vision)
        assert get_vision_llm() is vision

    def test_get_transcriber(self):
        configure(**_NEO)
        with pytest.raises(ConfigurationError, match="transcriber"):
            get_transcriber()
        transcriber = _Transcriber()
        configure(**_NEO, transcriber=transcriber)
        assert get_transcriber() is transcriber

    @pytest.mark.parametrize(
        ("describe", "with_vision", "expected"),
        [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
    )
    def test_embedded_images_enabled(self, describe, with_vision, expected):
        configure(
            **_NEO,
            describe_embedded_images=describe,
            vision_llm=_chat_model() if with_vision else None,
        )
        assert embedded_images_enabled() is expected

    def test_embedded_images_disabled_without_configure(self, monkeypatch):
        monkeypatch.setattr(config_module, "_config", None)
        assert embedded_images_enabled() is False

    def test_one_info_log_when_describe_without_vision(self, caplog):
        with caplog.at_level(logging.INFO, logger="scinr.newton.config"):
            configure(**_NEO)
        messages = [r.message for r in caplog.records if "describe_embedded_images" in r.message]
        assert len(messages) == 1
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="scinr.newton.config"):
            configure(**_NEO, vision_llm=_chat_model())
        assert not [r for r in caplog.records if "describe_embedded_images" in r.message]


def test_transcriber_is_exported_from_the_package():
    import scinr.newton as newton

    assert newton.Transcriber is Transcriber
    assert newton.TranscriptSegment is TranscriptSegment
