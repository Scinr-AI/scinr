"""Transcription cues of an audio / video page (decision D7).

A :class:`TranscriptCue` is one transcribed sentence with its absolute time
range in the original medium (never relative to the segment the converter
cut). The cues of a page let a caller jump from a source-text quote to its
exact second (see ``navigation.models.PageText.find_in_transcript``); the
page markdown does not change: it is still the cues' text, one per line.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["TranscriptCue"]


class TranscriptCue(BaseModel):
    """One transcribed sentence with its time range in the source medium.

    Attributes
    ----------
    start_ms, end_ms:
        Time range ``[start_ms, end_ms)`` of the sentence, **absolute in the
        original medium** (not relative to the segment the converter cut).
        Times are never invented: a transcriber without timestamps yields a
        single cue with the whole segment range.
    text:
        The transcribed sentence, stripped (a cue without text is discarded).
    """

    model_config = ConfigDict(extra="forbid")

    start_ms: int
    end_ms: int
    text: str
