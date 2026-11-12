"""Shared pieces of the audio and video converters.

Both cut the medium into segments of ``media_segment_seconds`` with
``ffmpeg`` (one page per segment, ``start_ms`` / ``end_ms`` from the cut, not
from the model) and process the segments **in sequence**: at most one
temporary segment file (or the frames of one segment) exists on disk at a
time, and the medium is never read in Python.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from scinr.newton.converters.base import ConversionError
from scinr.newton.media import ffmpeg
from scinr.newton.media.transcriber import TranscriptSegment
from scinr.newton.utils.loop_semaphore import LoopSemaphore

logger = logging.getLogger(__name__)

#: Audio and video conversions in flight per event loop (``media_concurrency``).
_media_semaphores = LoopSemaphore()


def media_semaphore(limit: int):
    """Semaphore bounding the audio and video conversions on the running loop."""
    return _media_semaphores.get(limit)


def format_timestamp(ms: int) -> str:
    """``315000`` → ``"00:05:15"``."""
    seconds = max(0, ms) // 1000
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def time_header(start_ms: int, end_ms: int) -> str:
    """Header of a segment page: ``[00:05:00 – 00:10:00]``."""
    return f"[{format_timestamp(start_ms)} – {format_timestamp(end_ms)}]"


@dataclass(frozen=True)
class Segment:
    index: int
    start_ms: int
    end_ms: int

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000


def iter_segments(
    duration_ms: int, segment_seconds: int, min_segment_seconds: int = 0
) -> Iterator[Segment]:
    """Consecutive segments covering ``[0, duration_ms)``; the last one ends at the duration.

    When the last segment would be shorter than *min_segment_seconds*, the cut
    before it moves earlier, to a whole second, so that it lasts at least that
    long: 300.05 s in segments of 300 s gives ``[0, 290 s)`` + ``[290 s, 300.05 s)``.
    No segment is ever longer than *segment_seconds* (provider request limits).
    *min_segment_seconds* is at most half of *segment_seconds* (``configure``
    enforces it), so the shortened segment keeps at least half its length.
    """
    step = segment_seconds * 1000
    min_ms = min(min_segment_seconds * 1000, step // 2)
    starts = list(range(0, max(duration_ms, 1), step))
    if len(starts) > 1 and duration_ms - starts[-1] < min_ms:
        starts[-1] = (duration_ms - min_ms) // 1000 * 1000
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else duration_ms
        yield Segment(index=index, start_ms=start, end_ms=end)


def check_duration(info: ffmpeg.MediaInfo, max_seconds: int | None, name: str) -> None:
    """Fail before spending anything when the medium exceeds ``max_media_duration_seconds``."""
    if max_seconds is not None and info.duration_ms > max_seconds * 1000:
        raise ConversionError(
            f"{name} lasts {format_timestamp(info.duration_ms)}, longer than "
            f"max_media_duration_seconds={max_seconds} ({format_timestamp(max_seconds * 1000)}). "
            f"Raise the limit (or pass None) in configure() to convert it."
        )


def transcript_text(segments: list[TranscriptSegment]) -> str:
    """One transcribed segment per line; empty when it is all silence."""
    return "\n".join(s.text.strip() for s in segments if s.text and s.text.strip())


async def transcribe_segment(
    *,
    binaries: ffmpeg.Binaries,
    transcriber,
    source: Path,
    segment: Segment,
    workdir: Path,
    language: str | None,
) -> str:
    """Cut *segment* to a temporary FLAC, transcribe it, delete it (also on failure)."""
    tmp = workdir / f"segment_{segment.index:05d}.flac"
    try:
        await ffmpeg.extract_audio_segment(
            binaries.ffmpeg, source, segment.start_s, segment.duration_s, tmp
        )
        result = await transcriber.transcribe(tmp, language)
    finally:
        tmp.unlink(missing_ok=True)
    return transcript_text(result)


def resolve_binaries() -> ffmpeg.Binaries:
    from scinr.newton.config import get_config

    cfg = get_config()
    return ffmpeg.find_binaries(cfg.ffmpeg_path, cfg.ffprobe_path)
