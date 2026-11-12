"""Semaphores bound to the running event loop.

An :class:`asyncio.Semaphore` belongs to the loop that first waits on it, and
callers such as Celery workers run ``asyncio.run()`` once per task. A
:class:`LoopSemaphore` keeps one semaphore per event loop (weakly, so a closed
loop releases its entry) and replaces it when the limit changes between runs
(``configure(...)`` with a new value).

Used by the PDF OCR (``mistral_ocr_concurrency``) and by the audio and video
converters (``media_concurrency``).
"""

from __future__ import annotations

import asyncio
import weakref

__all__ = ["LoopSemaphore"]


class LoopSemaphore:
    """One :class:`asyncio.Semaphore` per running event loop."""

    def __init__(self) -> None:
        self._by_loop: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
        ] = weakref.WeakKeyDictionary()

    def get(self, limit: int) -> asyncio.Semaphore:
        """Semaphore of size *limit* for the running loop.

        A different *limit* than the one the loop's semaphore was created
        with replaces it for the work that starts afterwards.
        """
        loop = asyncio.get_running_loop()
        entry = self._by_loop.get(loop)
        if entry is None or entry[0] != limit:
            entry = (limit, asyncio.Semaphore(limit))
            self._by_loop[loop] = entry
        return entry[1]
