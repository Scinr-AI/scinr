"""Ready-made :class:`~scinr.newton.media.transcriber.Transcriber` implementations.

- :class:`OpenAICompatibleTranscriber`: any OpenAI-compatible
  ``/audio/transcriptions`` endpoint (cloud or self-hosted). No extra package.
- :class:`FasterWhisperTranscriber`: local inference with ``faster-whisper``
  (``pip install "scinr[media]"``; imported only when used).
"""

from scinr.newton.media.transcribers.faster_whisper import FasterWhisperTranscriber
from scinr.newton.media.transcribers.openai_compatible import OpenAICompatibleTranscriber

__all__ = ["OpenAICompatibleTranscriber", "FasterWhisperTranscriber"]
