"""converters/audio.py — audio files to text with the configured ``transcriber``.

One page per segment of ``media_segment_seconds``, with ``start_ms`` /
``end_ms`` and a time header::

    [00:05:00 – 00:10:00]

    transcribed text, one segment per line

A silent segment produces no page (the following indexes are not shifted,
like the gaps of ``best_effort`` in PDF). A failed segment is skipped and
recorded in ``missing_segments`` with ``media_error_strategy="best_effort"``
(the default), or fails the document with ``"fail_fast"``.

Requirements (checked at conversion time, with a :class:`ConversionError`
that says how to configure them): a ``transcriber`` and the ``ffmpeg`` /
``ffprobe`` system binaries.

Memory: the audio is never read in Python; one temporary segment file on
disk at a time. At most ``media_concurrency`` audio/video conversions run at
once per event loop.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
)
from scinr.newton.converters.media_common import (
    check_duration,
    format_timestamp,
    iter_segments,
    media_semaphore,
    resolve_binaries,
    time_header,
    transcribe_segment,
)
from scinr.newton.exceptions import ConfigurationError
from scinr.newton.media import ffmpeg
from scinr.newton.models.file_metadata import FileMetadata

logger = logging.getLogger(__name__)

__all__ = ["AudioConverter"]


class AudioConverter(BaseConverter):
    """Transcribe an audio file into one page per segment."""

    supported_extensions: frozenset[str] = frozenset({"mp3", "m4a", "wav", "ogg", "flac"})
    is_async: bool = True

    async def convert(self, source: Path) -> IntermediateDocument:
        """Transcribe *source*.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist.
        ConversionError
            Missing ``transcriber`` or ``ffmpeg``, no audio track, longer than
            ``max_media_duration_seconds``, or a failed segment with
            ``media_error_strategy="fail_fast"``.
        """
        from scinr.newton.config import get_config, get_transcriber

        try:
            transcriber = get_transcriber()
        except ConfigurationError as exc:
            raise ConversionError(
                f"Cannot convert audio {source.name}: no transcriber is configured.\n{exc}"
            ) from exc
        binaries = resolve_binaries()
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")
        cfg = get_config()
        async with media_semaphore(cfg.media_concurrency):
            return await self._convert(source, transcriber, binaries, cfg)

    async def _convert(self, source, transcriber, binaries, cfg) -> IntermediateDocument:
        info = await ffmpeg.probe(binaries.ffprobe, source)
        if not info.has_audio:
            raise ConversionError(f"{source.name} has no audio track")
        check_duration(info, cfg.max_media_duration_seconds, source.name)

        pages: list[IntermediatePage] = []
        missing: list[tuple[int, int]] = []
        with tempfile.TemporaryDirectory(prefix="scinr-audio-") as workdir:
            for segment in iter_segments(
                info.duration_ms, cfg.media_segment_seconds, cfg.media_min_segment_seconds
            ):
                try:
                    text = await transcribe_segment(
                        binaries=binaries,
                        transcriber=transcriber,
                        source=source,
                        segment=segment,
                        workdir=Path(workdir),
                        language=cfg.media_language,
                    )
                except Exception as exc:
                    label = (
                        f"[{format_timestamp(segment.start_ms)} – "
                        f"{format_timestamp(segment.end_ms)}]"
                    )
                    if cfg.media_error_strategy == "fail_fast":
                        raise ConversionError(
                            f"Transcription of {source.name} failed in segment {label}: {exc}"
                        ) from exc
                    logger.warning(
                        "Segment %s of %s failed and is skipped (best_effort): %s",
                        label, source.name, exc,
                    )
                    missing.append((segment.start_ms, segment.end_ms))
                    continue
                if not text:
                    continue  # silence: no page
                pages.append(
                    IntermediatePage(
                        index=segment.index,
                        markdown=f"{time_header(segment.start_ms, segment.end_ms)}\n\n{text}",
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                    )
                )
        logger.info(
            "Audio %s: %d page(s), %d segment(s) skipped.", source.name, len(pages), len(missing)
        )
        return IntermediateDocument(
            pages=pages,
            missing_segments=missing or None,
            file_metadata=FileMetadata(
                duration_ms=info.duration_ms,
                container=info.container,
                audio_codec=info.audio_codec,
            ),
        )
