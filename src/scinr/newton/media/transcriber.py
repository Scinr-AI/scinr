"""Speech-to-text interface used by the audio and video converters.

Implement :class:`Transcriber` to plug any provider in. The contract:

- :meth:`Transcriber.transcribe` is called **once per segment** with a
  temporary FLAC file, mono, 16 kHz, at most ``media_segment_seconds`` long.
  The converter cuts the segments with ``ffmpeg``; the transcriber never
  sees the whole medium.
- The returned times are **relative to the start of that file** (the
  converter adds the segment offset).
- It may return a single segment with all the text when the provider gives
  no timestamps.
- ``language`` is an ISO 639-1 code, or ``None`` to let the provider detect it.
- Raise any exception on failure: the converter applies
  ``media_error_strategy`` to the segment (``best_effort`` skips it and
  records it in ``missing_segments``; ``fail_fast`` fails the document).
  Retries are the implementation's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

__all__ = ["Transcriber", "TranscriptSegment"]


class TranscriptSegment(BaseModel):
    """A piece of transcribed speech.

    Attributes
    ----------
    start_ms, end_ms:
        Time range in milliseconds, relative to the start of the file passed
        to :meth:`Transcriber.transcribe`.
    text:
        Transcribed text.
    speaker:
        Speaker label. Not used in this version (no diarisation).
    """

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


class Transcriber(ABC):
    """Turns one audio segment into text. See the module docstring for the contract."""

    @abstractmethod
    async def transcribe(
        self, path: Path, language: str | None = None
    ) -> list[TranscriptSegment]:
        """Transcribe the audio file *path* (FLAC, mono, 16 kHz, one segment).

        Parameters
        ----------
        path:
            Temporary file with the segment's audio. Deleted by the caller
            right after this call returns.
        language:
            ISO 639-1 code (``"es"``, ``"en"``…) or ``None`` for automatic
            detection.

        Returns
        -------
        list[TranscriptSegment]
            Segments in time order, times relative to the start of *path*.
            An empty list (or segments with empty text) means silence.
        """
