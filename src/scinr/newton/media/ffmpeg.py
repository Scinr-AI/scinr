"""``ffmpeg`` / ``ffprobe`` helpers for the audio and video converters.

Every call is an ``asyncio`` subprocess with an argument list (never a
shell) and absolute paths; ``stdout`` goes to ``DEVNULL`` (``ffprobe`` reads
its JSON from ``stdout``) and only the tail of ``stderr`` is kept for the
error message. The medium is never read in Python: ``ffmpeg`` writes one
segment (or the frames of one segment) to a temporary file at a time.

``ffmpeg`` and ``ffprobe`` are system binaries, not installed with the
package: ``ffmpeg_path`` / ``ffprobe_path`` in :func:`~scinr.newton.config.configure`,
or the ``PATH``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from scinr.newton.converters.base import ConversionError

__all__ = [
    "MediaInfo",
    "find_binaries",
    "probe",
    "extract_audio_segment",
    "extract_frames",
]

_STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class MediaInfo:
    """What ``ffprobe`` reports about a medium."""

    duration_ms: int
    has_audio: bool
    has_video: bool
    container: str | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None


@dataclass(frozen=True)
class Binaries:
    ffmpeg: str
    ffprobe: str


def find_binaries(ffmpeg_path: str | None, ffprobe_path: str | None) -> Binaries:
    """Resolve ``ffmpeg`` and ``ffprobe`` (configured path, else the ``PATH``).

    Raises
    ------
    ConversionError
        Either binary is missing, with how to install or configure it.
    """
    resolved: dict[str, str] = {}
    for name, configured in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)):
        candidate = configured or shutil.which(name)
        if candidate and Path(candidate).is_file():
            resolved[name] = str(Path(candidate).resolve())
            continue
        where = f"at {configured!r}" if configured else "in the PATH"
        raise ConversionError(
            f"{name} was not found {where}. Audio and video conversion needs ffmpeg and "
            f"ffprobe installed on the system (e.g. 'brew install ffmpeg' or "
            f"'apt-get install ffmpeg'), or their paths in "
            f"configure(ffmpeg_path=..., ffprobe_path=...) / FFMPEG_PATH, FFPROBE_PATH."
        )
    return Binaries(ffmpeg=resolved["ffmpeg"], ffprobe=resolved["ffprobe"])


async def _run(args: list[str], *, capture_stdout: bool = False) -> bytes:
    """Run *args*; return ``stdout`` (when captured). Non-zero exit → ConversionError."""
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ConversionError(f"Cannot run {Path(args[0]).name}: {exc}") from exc
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace")[-_STDERR_TAIL_CHARS:].strip()
        raise ConversionError(
            f"{Path(args[0]).name} failed with exit code {process.returncode}: {tail}"
        )
    return stdout or b""


def _fps(value: str | None) -> float | None:
    if not value or value in ("0/0", "0"):
        return None
    try:
        num, _, den = value.partition("/")
        rate = float(num) / float(den) if den else float(num)
    except (ValueError, ZeroDivisionError):
        return None
    return round(rate, 3) if rate > 0 else None


async def probe(ffprobe: str, path: Path) -> MediaInfo:
    """Duration, streams and codecs of *path* (``ffprobe -show_format -show_streams``)."""
    raw = await _run(
        [
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path.resolve()),
        ],
        capture_stdout=True,
    )
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ConversionError(f"Unexpected ffprobe output for {path.name}: {exc}") from exc
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next(
        (
            s for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")  # cover art
        ),
        None,
    )
    try:
        duration_ms = round(float(fmt.get("duration")) * 1000)
    except (TypeError, ValueError):
        raise ConversionError(f"ffprobe reports no duration for {path.name}") from None
    container = fmt.get("format_name")
    if isinstance(container, str):
        container = container.split(",")[0]
    return MediaInfo(
        duration_ms=duration_ms,
        has_audio=audio is not None,
        has_video=video is not None,
        container=container,
        audio_codec=audio.get("codec_name") if audio else None,
        video_codec=video.get("codec_name") if video else None,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        frame_rate=_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else None,
    )


async def extract_audio_segment(
    ffmpeg: str, src: Path, start_s: float, duration_s: float, dst: Path
) -> None:
    """Write ``[start_s, start_s + duration_s)`` of *src* to *dst* as FLAC, mono, 16 kHz."""
    await _run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
            "-i", str(src.resolve()),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac",
            "-y", str(dst.resolve()),
        ]
    )


async def extract_frames(
    ffmpeg: str,
    src: Path,
    start_s: float,
    duration_s: float,
    count: int,
    max_side: int,
    out_dir: Path,
) -> list[Path]:
    """Write *count* frames spread uniformly over the segment to *out_dir* as JPEG.

    Frames are scaled down to fit in ``max_side`` × ``max_side`` (aspect kept).
    Returns the frame files in time order.
    """
    if count < 1:
        return []
    rate = count / max(duration_s, 0.001)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir.resolve() / "frame_%04d.jpg"
    scale = (
        f"scale=w='min(iw,{max_side})':h='min(ih,{max_side})'"
        f":force_original_aspect_ratio=decrease"
    )
    await _run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
            "-i", str(src.resolve()),
            "-an", "-vf", f"fps={rate:.6f},{scale}",
            "-frames:v", str(count), "-q:v", "3",
            "-y", str(pattern),
        ]
    )
    return sorted(out_dir.glob("frame_*.jpg"))
