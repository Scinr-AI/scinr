"""converters/video.py — video files to text: transcription plus described frames.

Same segments as the audio converter (one page per ``media_segment_seconds``,
with ``start_ms`` / ``end_ms``). Each page holds:

1. the time header and the transcription of the segment (when the video has
   an audio track), and
2. one block per sampled frame, described by the ``vision_llm``::

       <image index="0" time="00:05:15">
       …description…
       </image>

Frames are sampled uniformly at ``video_frames_per_minute`` (``0`` =
transcription only); a frame identical to the previous one is skipped.

Requirements: ``ffmpeg`` / ``ffprobe``; a ``transcriber`` when the video has
audio; a ``vision_llm`` for the frames (without it the frames are skipped
with a warning). With neither transcription nor frames available the
conversion fails.

Failures follow ``media_error_strategy``: a failed transcription is recorded
per segment in ``missing_segments``, a failed frame per image in
``missing_images`` (``(page_index, frame_index)``). When ``ffmpeg`` cannot
extract the frames of a segment, every frame planned for it is recorded in
``missing_images``.

Memory: the video is never opened in Python. The frames of one segment are on
disk at a time, already scaled to ``image_max_side_px``; they pass through
memory one by one (or as many as the LLM semaphore lets describe at once).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
)
from scinr.newton.converters.enrichment import render_image_block
from scinr.newton.converters.media_common import (
    Segment,
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

__all__ = ["VideoConverter"]


class VideoConverter(BaseConverter):
    """Transcribe a video and describe sampled frames, one page per segment."""

    supported_extensions: frozenset[str] = frozenset({"mp4", "mov", "mkv", "webm"})
    is_async: bool = True

    async def convert(self, source: Path) -> IntermediateDocument:
        """Convert *source*.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist.
        ConversionError
            Missing ``ffmpeg``; audio track without ``transcriber``; nothing
            to extract (no audio and no usable frames); longer than
            ``max_media_duration_seconds``; or a failure with
            ``media_error_strategy="fail_fast"``.
        """
        from scinr.newton.config import get_config

        binaries = resolve_binaries()
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")
        cfg = get_config()
        async with media_semaphore(cfg.media_concurrency):
            return await self._convert(source, binaries, cfg)

    async def _convert(self, source: Path, binaries, cfg) -> IntermediateDocument:
        from scinr.newton.config import get_transcriber

        info = await ffmpeg.probe(binaries.ffprobe, source)
        transcriber = None
        if info.has_audio:
            try:
                transcriber = get_transcriber()
            except ConfigurationError as exc:
                raise ConversionError(
                    f"Cannot convert video {source.name}: it has an audio track and no "
                    f"transcriber is configured.\n{exc}"
                ) from exc
        with_frames = info.has_video and cfg.video_frames_per_minute > 0
        if with_frames and cfg.vision_llm is None:
            logger.warning(
                "No vision_llm configured: the frames of %s are skipped and only the "
                "transcription is kept.", source.name,
            )
            with_frames = False
        if transcriber is None and not with_frames:
            raise ConversionError(
                f"Nothing to extract from {source.name}: "
                + ("it has no audio track" if not info.has_audio else "no transcriber")
                + " and frames are disabled or need a vision_llm "
                "(configure(vision_llm=...), video_frames_per_minute > 0)."
            )
        check_duration(info, cfg.max_media_duration_seconds, source.name)
        fail_fast = cfg.media_error_strategy == "fail_fast"

        pages: list[IntermediatePage] = []
        missing_segments: list[tuple[int, int]] = []
        missing_images: list[tuple[int, int]] = []
        with tempfile.TemporaryDirectory(prefix="scinr-video-") as workdir_str:
            workdir = Path(workdir_str)
            last_digest: str | None = None
            for segment in iter_segments(
                info.duration_ms, cfg.media_segment_seconds, cfg.media_min_segment_seconds
            ):
                parts: list[str] = []
                if transcriber is not None:
                    try:
                        text = await transcribe_segment(
                            binaries=binaries, transcriber=transcriber, source=source,
                            segment=segment, workdir=workdir, language=cfg.media_language,
                        )
                    except Exception as exc:
                        if fail_fast:
                            raise ConversionError(
                                f"Transcription of {source.name} failed in segment "
                                f"{time_header(segment.start_ms, segment.end_ms)}: {exc}"
                            ) from exc
                        logger.warning(
                            "Transcription of segment %s of %s failed (best_effort): %s",
                            time_header(segment.start_ms, segment.end_ms), source.name, exc,
                        )
                        missing_segments.append((segment.start_ms, segment.end_ms))
                        text = ""
                    if text:
                        parts.append(text)
                if with_frames:
                    blocks, failed, last_digest = await self._frames(
                        binaries, source, segment, workdir, cfg, last_digest, fail_fast
                    )
                    parts.extend(blocks)
                    missing_images.extend((segment.index, f) for f in failed)
                if not parts:
                    continue
                pages.append(
                    IntermediatePage(
                        index=segment.index,
                        markdown=time_header(segment.start_ms, segment.end_ms)
                        + "\n\n"
                        + "\n\n".join(parts),
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                    )
                )
        logger.info("Video %s: %d page(s).", source.name, len(pages))
        return IntermediateDocument(
            pages=pages,
            missing_segments=missing_segments or None,
            missing_images=missing_images or None,
            file_metadata=FileMetadata(
                duration_ms=info.duration_ms,
                container=info.container,
                audio_codec=info.audio_codec,
                video_codec=info.video_codec,
                width=info.width,
                height=info.height,
                frame_rate=info.frame_rate,
            ),
        )

    async def _frames(
        self, binaries, source: Path, segment: Segment, workdir: Path, cfg,
        last_digest: str | None, fail_fast: bool,
    ) -> tuple[list[str], list[int], str | None]:
        """Describe the sampled frames of *segment*.

        Returns ``(blocks, failed_frame_indexes, digest_of_the_last_frame)``.
        """
        from scinr.newton.converters.vision import describe_image

        count = max(1, round(cfg.video_frames_per_minute * segment.duration_s / 60))
        frames_dir = workdir / f"frames_{segment.index:05d}"
        try:
            try:
                files = await ffmpeg.extract_frames(
                    binaries.ffmpeg, source, segment.start_s, segment.duration_s,
                    count, cfg.image_max_side_px, frames_dir,
                )
            except Exception as exc:
                # Same strategy as a failed frame: every frame planned for the
                # segment is recorded as missing and the video goes on.
                label = time_header(segment.start_ms, segment.end_ms)
                if fail_fast:
                    raise ConversionError(
                        f"Frames of segment {label} of {source.name} could not be "
                        f"extracted: {exc}"
                    ) from exc
                logger.warning(
                    "Frames of segment %s of %s could not be extracted (best_effort): %s",
                    label, source.name, exc,
                )
                return [], list(range(count)), last_digest
            step_ms = (segment.end_ms - segment.start_ms) / count
            # Sequential pass: skip a frame identical to the previous one.
            kept: list[tuple[int, Path, int]] = []  # (frame index, file, time ms)
            for k, frame in enumerate(files):
                digest = hashlib.sha1(frame.read_bytes(), usedforsecurity=False).hexdigest()
                if digest == last_digest:
                    frame.unlink(missing_ok=True)
                    continue
                last_digest = digest
                kept.append((k, frame, segment.start_ms + round(k * step_ms)))

            async def _describe(path: Path) -> str:
                # Read only on its turn to be decoded (vision.prepare_for_vision).
                try:
                    return await describe_image(path.read_bytes, mode="frame")
                finally:
                    path.unlink(missing_ok=True)

            results = await asyncio.gather(
                *(_describe(path) for _, path, _ in kept), return_exceptions=True
            )
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

        blocks: list[str] = []
        failed: list[int] = []
        for (k, _, time_ms), result in zip(kept, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                if fail_fast:
                    raise ConversionError(
                        f"Frame at {format_timestamp(time_ms)} of {source.name} could not be "
                        f"described: {result}"
                    ) from result
                logger.warning(
                    "Frame at %s of %s could not be described (best_effort): %s",
                    format_timestamp(time_ms), source.name, result,
                )
                failed.append(k)
                continue
            if result:
                blocks.append(render_image_block(k, result, time=format_timestamp(time_ms)))
        return blocks, failed, last_digest
