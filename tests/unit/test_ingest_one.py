"""
tests/unit/test_ingest_one.py — Unit tests for
scinr.newton.ingest.loader.ingest_one() / ingest_one_from_path().

Focus: the get_neo4j_sync_semaphore() wiring — the semaphore must bound how
many asyncio.to_thread() dispatches are in flight simultaneously, and it
must be acquired/released in the event loop (never inside the worker
thread it wraps).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from scinr.newton.config import (
    configure,
    reset_neo4j_sync_semaphore,
)
from scinr.newton.ingest.loader import ingest_one, ingest_one_from_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_URI = "bolt://localhost:7687"
_DUMMY_USER = "neo4j"
_DUMMY_PASS = "test"

_SLEEP_SECONDS = 0.15


def _make_mock_llm():
    """Create a minimal mock that passes _validate_llm (mirrors test_config.py)."""
    try:
        from langchain_core.language_models import BaseChatModel

        class _MockLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "mock"

            def _generate(self, *args, **kwargs):
                pass

        return _MockLLM()
    except Exception:
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.with_structured_output = MagicMock()
        return llm


class _ConcurrencyTracker:
    """Records the max number of simultaneously in-flight synchronous calls."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock_for_counters: list[int] = []

    def blocking_call(self, *_args, **_kwargs) -> str:
        # NOTE: this runs inside the asyncio.to_thread() worker thread.
        # It must NOT touch any asyncio primitive (no Semaphore, no
        # event-loop calls) — only plain time.sleep(), exactly like the
        # real _load_document_object()/load_file() bodies do.
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(_SLEEP_SECONDS)
        self.in_flight -= 1
        return "ok"


@pytest.fixture(autouse=True)
def _configure_sync_concurrency_1(monkeypatch):
    """Configure neo4j_sync_concurrency=1 and reset the semaphore before each test."""
    monkeypatch.delenv("NEO4J_SYNC_CONCURRENCY", raising=False)
    llm = _make_mock_llm()
    configure(
        llm=llm,
        neo4j_uri=_DUMMY_URI,
        neo4j_user=_DUMMY_USER,
        neo4j_password=_DUMMY_PASS,
        neo4j_sync_concurrency=1,
    )
    reset_neo4j_sync_semaphore()
    yield
    reset_neo4j_sync_semaphore()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestOneSemaphore:
    async def test_ingest_one_respects_sync_concurrency_limit(self):
        """Two concurrent ingest_one() calls with neo4j_sync_concurrency=1 must
        never run their blocking work simultaneously.

        This also indirectly confirms the semaphore is acquired/released in
        the event loop rather than inside the worker thread: if it were
        acquired inside the thread (or not acquired at all), both blocking
        calls would overlap and max_in_flight would be 2, not 1 — this
        assertion is deterministic (no timing flakiness), not a heuristic.
        """
        tracker = _ConcurrencyTracker()

        with patch(
            "scinr.newton.ingest.loader._load_document_object",
            side_effect=tracker.blocking_call,
        ):
            results = await asyncio.gather(
                ingest_one(doc=object(), driver=object()),
                ingest_one(doc=object(), driver=object()),
            )

        assert results == ["ok", "ok"]
        assert tracker.max_in_flight == 1

    async def test_ingest_one_from_path_respects_sync_concurrency_limit(self):
        """Same guarantee as above, for the Path-based ingest_one_from_path()."""
        tracker = _ConcurrencyTracker()

        with patch(
            "scinr.newton.ingest.loader.load_file",
            side_effect=tracker.blocking_call,
        ):
            results = await asyncio.gather(
                ingest_one_from_path(path=object(), driver=object()),
                ingest_one_from_path(path=object(), driver=object()),
            )

        assert results == ["ok", "ok"]
        assert tracker.max_in_flight == 1

    async def test_ingest_one_releases_semaphore_after_completion(self):
        """After ingest_one() returns, the semaphore slot must be free again
        (not leaked) — a third sequential call must not deadlock or block."""
        tracker = _ConcurrencyTracker()

        with patch(
            "scinr.newton.ingest.loader._load_document_object",
            side_effect=tracker.blocking_call,
        ):
            await ingest_one(doc=object(), driver=object())
            await ingest_one(doc=object(), driver=object())
            await ingest_one(doc=object(), driver=object())

        assert tracker.max_in_flight == 1
