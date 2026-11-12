"""Transcription through an OpenAI-compatible ``/audio/transcriptions`` endpoint.

The multipart body is streamed from disk (one block in memory at a time).

Works with OpenAI and with any server that implements the same endpoint
(self-hosted Whisper servers, vLLM, other providers): point ``base_url`` at
it. ``api_key`` is optional for servers that do not ask for one.

Timestamps: the request asks for ``response_format=verbose_json`` with
``timestamp_granularities[]=segment``. Servers and models that reject those
parameters (HTTP 400 / 422) get the same segment again with
``response_format=json``, and the instance remembers it so the next segments
go straight to ``json``. Without timestamps the whole segment is one
:class:`~scinr.newton.media.transcriber.TranscriptSegment`.

Silence: Whisper tends to invent text on silent or music-only audio. With
``verbose_json`` each segment carries ``no_speech_prob`` and
``avg_logprob``; a segment with ``no_speech_prob > no_speech_threshold`` and
``avg_logprob < -1`` is dropped (Whisper's own silence rule). Servers that
do not report those fields, or answer plain ``json``, are not filtered.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from scinr.newton.media.transcriber import Transcriber, TranscriptSegment

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatibleTranscriber"]

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_FORMAT_REJECTED_STATUS_CODES = frozenset({400, 422})


_FILE_BLOCK_BYTES = 1024 * 1024

#: Whisper's ``logprob_threshold``: together with ``no_speech_prob`` it marks silence.
_SILENCE_LOGPROB = -1.0


class _MultipartBody:
    """``multipart/form-data`` body streamed from disk, re-iterable for retries.

    Only one block of the audio file is in memory at a time; the length is
    known up front, so the request carries an explicit ``Content-Length``.
    """

    def __init__(self, fields: list[tuple[str, str]], path: Path, content_type: str) -> None:
        self.boundary = "scinr-" + secrets.token_hex(16)
        head = bytearray()
        for name, value in fields:
            head += (
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        filename = path.name.replace('"', "")
        head += (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        self._head = bytes(head)
        self._tail = f"\r\n--{self.boundary}--\r\n".encode()
        self._path = path
        self._size = path.stat().st_size
        self.content_length = len(self._head) + self._size + len(self._tail)
        self.content_type = f"multipart/form-data; boundary={self.boundary}"

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._head
        remaining = self._size
        with self._path.open("rb") as fh:
            while remaining > 0:
                block = await asyncio.to_thread(fh.read, min(_FILE_BLOCK_BYTES, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
        if remaining:
            raise TranscriptionError(f"{self._path.name} changed size while it was being sent")
        yield self._tail


class TranscriptionError(RuntimeError):
    """The transcription request failed (after retries, when retryable)."""


class OpenAICompatibleTranscriber(Transcriber):
    """``POST {base_url}/audio/transcriptions`` (multipart, file streamed from disk).

    Parameters
    ----------
    model:
        Model name on the server (``"whisper-1"``, ``"gpt-4o-mini-transcribe"``,
        the name a self-hosted server expects…).
    api_key:
        Bearer token. ``None`` sends no ``Authorization`` header.
    base_url:
        API root, without ``/audio/transcriptions``.
    timeout:
        Seconds per request.
    max_retries:
        Attempts per segment on network errors and HTTP 429 / 5xx, with
        exponential backoff.
    retry_backoff_seconds:
        Base of the backoff.
    no_speech_threshold:
        Drop the segments the model itself marks as silence (see the module
        docstring). ``None`` keeps every segment.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 300,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        no_speech_threshold: float | None = 0.6,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.no_speech_threshold = no_speech_threshold
        # Set to False the first time the server rejects verbose_json.
        self._timestamps_supported = True

    def __repr__(self) -> str:  # never print the key
        return (
            f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key={'***' if self._api_key else None})"
        )

    async def transcribe(
        self, path: Path, language: str | None = None
    ) -> list[TranscriptSegment]:
        if self._timestamps_supported:
            status, data = await self._post(path, language, verbose=True)
            if status in _FORMAT_REJECTED_STATUS_CODES:
                logger.info(
                    "Transcription server rejected verbose_json (HTTP %d); "
                    "falling back to json without timestamps.",
                    status,
                )
                self._timestamps_supported = False
            else:
                return _segments(data, self.no_speech_threshold)
        status, data = await self._post(path, language, verbose=False)
        if status >= 400:
            raise TranscriptionError(f"Transcription request failed with HTTP {status}: {data}")
        return _segments(data, self.no_speech_threshold)

    async def _post(self, path: Path, language: str | None, *, verbose: bool):
        """One request (with retries). Returns ``(status, json_or_text)``.

        A 400 / 422 with ``verbose`` is returned to the caller (format
        fallback); any other non-retryable error raises.
        """
        import httpx

        fields: list[tuple[str, str]] = [("model", self.model)]
        if verbose:
            fields += [
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
            ]
        else:
            fields.append(("response_format", "json"))
        if language:
            fields.append(("language", language))
        body = _MultipartBody(fields, path, "audio/flac")
        headers = {
            "Content-Type": body.content_type,
            "Content-Length": str(body.content_length),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self.base_url}/audio/transcriptions"

        last_error: Exception | str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, content=body)
            except httpx.RequestError as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    try:
                        return response.status_code, response.json()
                    except ValueError:
                        return response.status_code, {"text": response.text}
                if verbose and response.status_code in _FORMAT_REJECTED_STATUS_CODES:
                    return response.status_code, response.text
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise TranscriptionError(
                        f"Transcription request failed with HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if attempt < self.max_retries:
                delay = min(self.retry_backoff_seconds * (2 ** (attempt - 1)), 60.0)
                logger.warning(
                    "Transcription request failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt, self.max_retries, last_error, delay,
                )
                await asyncio.sleep(delay)
        raise TranscriptionError(
            f"Transcription request failed after {self.max_retries} attempt(s): {last_error}"
        )


def _is_silence(seg: dict, no_speech_threshold: float | None) -> bool:
    """Whisper's rule: high ``no_speech_prob`` and low ``avg_logprob``."""
    if no_speech_threshold is None:
        return False
    no_speech, logprob = seg.get("no_speech_prob"), seg.get("avg_logprob")
    if not isinstance(no_speech, int | float) or not isinstance(logprob, int | float):
        return False
    return no_speech > no_speech_threshold and logprob < _SILENCE_LOGPROB


def _segments(data, no_speech_threshold: float | None = None) -> list[TranscriptSegment]:
    """Map a ``json`` / ``verbose_json`` response to segments (times relative to the file)."""
    if not isinstance(data, dict):
        text = str(data or "").strip()
        return [TranscriptSegment(start_ms=0, end_ms=0, text=text)] if text else []
    segments = data.get("segments")
    if isinstance(segments, list) and segments:
        out: list[TranscriptSegment] = []
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            if not text or _is_silence(seg, no_speech_threshold):
                continue
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
            out.append(
                TranscriptSegment(
                    start_ms=round(start * 1000), end_ms=round(end * 1000), text=text
                )
            )
        return out
    text = str(data.get("text") or "").strip()
    if not text:
        return []
    duration = data.get("duration")
    end_ms = round(float(duration) * 1000) if isinstance(duration, int | float) else 0
    return [TranscriptSegment(start_ms=0, end_ms=end_ms, text=text)]
