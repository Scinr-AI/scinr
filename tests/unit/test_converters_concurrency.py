"""
tests/unit/test_converters_concurrency.py — Wall-clock concurrency tests for
Stage 0 (converters) parallelism.

These tests verify that `convert_folder(parallel_docs=K)` actually achieves
real concurrency for both:

1. Synchronous converters (BaseConverter subclasses with the default
   `is_async = False`), whose blocking `convert()` is dispatched to a worker
   thread via `asyncio.to_thread()` by `_run_convert()` — proving the
   dispatch does not silently serialise on the event loop.
2. Async converters (`is_async = True`, e.g. `PdfConverter`), whose native
   coroutine `convert()` is awaited directly on the event loop by
   `_run_convert()` — proving real overlap without any thread involved.

Wall-clock time (``time.monotonic()``) is used rather than an in-flight
counter/tracker so that genuine, real-world overlap (including actual
thread-pool scheduling) is exercised end-to-end, not just simulated
scheduling order.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import scinr.newton.converters.registry as registry_module
from scinr.newton.converters.base import (
    BaseConverter,
    IntermediateDocument,
    IntermediatePage,
)
from scinr.newton.converters.main import convert_folder
from scinr.newton.converters.registry import apply_converter_overrides

SLEEP_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Fixture: restore registry state after each test (mirrors
# test_converters_registry.py's own fixture, kept local/independent here so
# this file has no cross-file fixture dependency).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_registry():
    """Capture and restore the converter registry dict after each test."""
    original = dict(registry_module._get_registry())
    yield
    reg = registry_module._get_registry()
    reg.clear()
    reg.update(original)


# ---------------------------------------------------------------------------
# Fake converters
# ---------------------------------------------------------------------------


class _FakeSyncSleepConverter(BaseConverter):
    """Sync converter (is_async=False, the BaseConverter default) whose
    `convert()` blocks the calling thread for SLEEP_SECONDS via a real
    `time.sleep()` — NOT `asyncio.sleep()` — so that a test proving real
    parallelism must actually dispatch it to a worker thread rather than
    just interleaving coroutines on a single thread."""

    supported_extensions: frozenset[str] = frozenset({"faketest"})

    def convert(self, source: Path) -> IntermediateDocument:
        time.sleep(SLEEP_SECONDS)
        return IntermediateDocument(
            pages=[IntermediatePage(index=0, markdown=f"content of {source.name}")]
        )


class _FakeAsyncSleepConverter(BaseConverter):
    """Async converter (is_async=True) whose `convert()` is a native
    coroutine that awaits `asyncio.sleep(SLEEP_SECONDS)` — mirroring the
    PdfConverter contract of genuine async I/O, awaited directly on the
    event loop by `_run_convert()`, with no worker thread involved."""

    supported_extensions: frozenset[str] = frozenset({"fakeasync"})
    is_async: bool = True

    async def convert(self, source: Path) -> IntermediateDocument:
        await asyncio.sleep(SLEEP_SECONDS)
        return IntermediateDocument(
            pages=[IntermediatePage(index=0, markdown=f"content of {source.name}")]
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_files(tmp_path: Path, count: int, suffix: str) -> None:
    for i in range(count):
        (tmp_path / f"doc{i}.{suffix}").write_text(f"placeholder {i}", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Sync converter dispatched via asyncio.to_thread() — real parallelism
# ---------------------------------------------------------------------------


class TestSyncConverterConcurrency:
    async def test_parallel_docs_achieves_real_concurrency_for_sync_converter(self, tmp_path):
        """K=4 files, SLEEP_SECONDS=0.5, parallel_docs=4: total wall-clock
        time must be substantially less than K * SLEEP_SECONDS (which would
        be the fully-sequential time), proving the sync `convert()` calls
        actually ran concurrently in worker threads rather than blocking the
        event loop one at a time."""
        K = 4
        _make_files(tmp_path, K, "faketest")
        apply_converter_overrides({"faketest": _FakeSyncSleepConverter})

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        start = time.monotonic()
        written, failures = await convert_folder(
            tmp_path, output_dir, parallel_docs=K
        )
        elapsed = time.monotonic() - start

        assert failures == []
        assert len(written) == K
        # Fully sequential would take K * SLEEP_SECONDS == 2.0s. A generous
        # margin (< SLEEP_SECONDS * 2 == 1.0s) leaves ample room for thread
        # scheduling / CI jitter while still clearly distinguishing from the
        # sequential case.
        assert elapsed < SLEEP_SECONDS * 2, (
            f"Expected concurrent execution (~{SLEEP_SECONDS}s), took {elapsed:.2f}s"
        )

    async def test_parallel_docs_one_remains_sequential_for_sync_converter(self, tmp_path):
        """Regression guard: parallel_docs=1 (the default) must still behave
        sequentially for sync converters — this is pre-existing behaviour
        that must NOT be broken by the parallelism fix."""
        K = 4
        _make_files(tmp_path, K, "faketest")
        apply_converter_overrides({"faketest": _FakeSyncSleepConverter})

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        start = time.monotonic()
        written, failures = await convert_folder(
            tmp_path, output_dir, parallel_docs=1
        )
        elapsed = time.monotonic() - start

        assert failures == []
        assert len(written) == K
        # Sequential: K * SLEEP_SECONDS == 2.0s. Allow a 20% margin below
        # for timer/scheduling slack.
        assert elapsed >= K * SLEEP_SECONDS * 0.8, (
            f"Expected sequential execution (~{K * SLEEP_SECONDS}s), took {elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# 2. Async converter awaited directly on the event loop — real parallelism
# ---------------------------------------------------------------------------


class TestAsyncConverterConcurrency:
    async def test_parallel_docs_achieves_real_concurrency_for_async_converter(self, tmp_path):
        """Same shape as the sync-converter test above, but for an
        is_async=True converter (mirroring PdfConverter): `_run_convert()`
        must await it directly on the event loop (no thread), and the
        semaphore-bounded `asyncio.gather()` in `convert_folder()` must let
        all K coroutines' `asyncio.sleep()` calls overlap."""
        K = 4
        _make_files(tmp_path, K, "fakeasync")
        apply_converter_overrides({"fakeasync": _FakeAsyncSleepConverter})

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        start = time.monotonic()
        written, failures = await convert_folder(
            tmp_path, output_dir, parallel_docs=K
        )
        elapsed = time.monotonic() - start

        assert failures == []
        assert len(written) == K
        assert elapsed < SLEEP_SECONDS * 2, (
            f"Expected concurrent execution (~{SLEEP_SECONDS}s), took {elapsed:.2f}s"
        )
