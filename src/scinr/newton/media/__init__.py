"""media — speech-to-text and ``ffmpeg`` helpers for the audio and video converters.

Nothing heavy is imported here: ``faster_whisper`` is imported only when a
:class:`~scinr.newton.media.transcribers.faster_whisper.FasterWhisperTranscriber`
is used.
"""

from scinr.newton.media.transcriber import Transcriber, TranscriptSegment

__all__ = ["Transcriber", "TranscriptSegment"]
