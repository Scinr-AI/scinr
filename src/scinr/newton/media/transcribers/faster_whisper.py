"""Local transcription with ``faster-whisper`` (``pip install "scinr[media]"``).

The model is loaded once per process and cached by
``(model_size, device, compute_type)``. Transcriptions run in a worker thread,
one at a time per model (a ``threading.Lock``): the model is not shared by
concurrent calls, and its memory is paid once.

Silence: Whisper tends to invent text ("Thanks for watching") on silent or
music-only audio. ``vad_filter=True`` (the default here, unlike
faster-whisper's own default) runs its bundled Silero voice-activity
detector first, so the stretches without speech are not transcribed and a
silent segment produces no page.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Any

from scinr.newton.media.transcriber import Transcriber, TranscriptSegment

__all__ = ["FasterWhisperTranscriber"]

_models: dict[tuple[str, str, str], tuple[Any, threading.Lock]] = {}
_models_lock = threading.Lock()


def _load(model_size: str, device: str, compute_type: str) -> tuple[Any, threading.Lock]:
    key = (model_size, device, compute_type)
    with _models_lock:
        entry = _models.get(key)
        if entry is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is required for FasterWhisperTranscriber. "
                    'Install it with: pip install "scinr[media]"'
                ) from exc
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            entry = (model, threading.Lock())
            _models[key] = entry
        return entry


class FasterWhisperTranscriber(Transcriber):
    """Transcribe locally with a Whisper model through ``faster-whisper``.

    Parameters
    ----------
    model_size:
        Model name or path (``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
        ``"large-v3"``…).
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``.
    compute_type:
        ``"default"``, ``"int8"``, ``"float16"``…
    vad_filter:
        Skip the stretches without speech before transcribing (see the
        module docstring). ``False`` transcribes everything.
    vad_parameters:
        Options of the voice-activity detector (faster-whisper's
        ``VadOptions``, e.g. ``{"min_silence_duration_ms": 500}``).
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        *,
        vad_filter: bool = True,
        vad_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.vad_parameters = dict(vad_parameters) if vad_parameters else None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_size={self.model_size!r}, device={self.device!r}, "
            f"compute_type={self.compute_type!r}, vad_filter={self.vad_filter!r})"
        )

    def _transcribe_sync(self, path: Path, language: str | None) -> list[TranscriptSegment]:
        model, lock = _load(self.model_size, self.device, self.compute_type)
        with lock:
            options: dict[str, Any] = {"language": language, "vad_filter": self.vad_filter}
            if self.vad_filter and self.vad_parameters:
                options["vad_parameters"] = self.vad_parameters
            # Decoded here with ffmpeg instead of faster-whisper's own PyAV
            # decoder, whose calls break with some PyAV releases (e.g.
            # "open() got an unexpected keyword argument 'metadata_errors'").
            audio = _decode_pcm(path)
            segments, _info = model.transcribe(audio, **options)
            out: list[TranscriptSegment] = []
            for seg in segments:  # a generator: decoding happens here
                text = (seg.text or "").strip()
                if text:
                    out.append(
                        TranscriptSegment(
                            start_ms=round(seg.start * 1000),
                            end_ms=round(seg.end * 1000),
                            text=text,
                        )
                    )
            return out

    async def transcribe(
        self, path: Path, language: str | None = None
    ) -> list[TranscriptSegment]:
        return await asyncio.to_thread(self._transcribe_sync, path, language)


def _decode_pcm(path: Path):
    """Decode *path* to the mono 16 kHz float32 samples faster-whisper expects.

    Uses the configured ``ffmpeg`` (``ffmpeg_path`` or the ``PATH``). Memory:
    4 bytes per sample, about 19 MB for a 5-minute segment.
    """
    import numpy as np

    from scinr.newton.config import get_config
    from scinr.newton.media.ffmpeg import find_binaries

    cfg = get_config()
    ffmpeg = find_binaries(cfg.ffmpeg_path, cfg.ffprobe_path).ffmpeg
    completed = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {tail}")
    return np.frombuffer(completed.stdout, dtype=np.float32)
