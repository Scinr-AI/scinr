"""
tests/unit/test_entity_extraction_agent.py — Regression tests for the
event-loop-blocking fix in scinr.newton.entity_extraction.agent /
scinr.newton.entity_extraction.neo4j_ops.

Prior to the fix, `fetch_extraction_targets()` was a *synchronous* Neo4j call
executed directly inside the async `_run_entity_extraction_parallel()` /
`run_entity_extraction_agent()` coroutines — meaning it ran on the event loop
thread itself rather than yielding control, so two concurrent documents could
never make progress on their Neo4j fetch at the same time (one would fully
block the loop while the other waited). The fix made `fetch_extraction_targets()`
(and the driver/session plumbing it and its callers use) genuinely `async def`,
using `AsyncDriver` + `await session.run(...)` throughout.

These tests prove the fix DETERMINISTICALLY (no real timing/flakiness): each
mock's `side_effect` does a real `await asyncio.sleep(...)`, which only
"counts" as concurrent if the event loop is free to run the *other* call's
`await` while the first is suspended. We track this via a shared
`in_flight`/`max_in_flight` counter guarded by an `asyncio.Lock()` — if the
code under test ever regresses to blocking synchronously, `max_in_flight`
would drop to `1` instead of `2` (see `test_ingest_one.py` /
`test_pipeline_orchestration.py::TestStage0Baseline` for the same pattern
already used elsewhere in this suite).
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scinr.newton.entity_extraction.agent import (
    _run_entity_extraction_parallel,
    run_entity_extraction_agent,
)

# ---------------------------------------------------------------------------
# Helpers — fake async Neo4j driver/session (async context manager protocol)
# ---------------------------------------------------------------------------


class _FakeAsyncResult:
    """Mimics neo4j.AsyncResult for the precondition-check queries."""

    def __init__(self, record: dict) -> None:
        self._record = record

    async def single(self):
        return self._record


class _FakeAsyncSession:
    """Mimics neo4j.AsyncSession — answers every `.run(...)` with `n=1`,
    which is enough for both `run_entity_extraction_agent()` precondition
    checks (document exists, at least one annotated StructureNode exists) to
    pass regardless of the exact query text.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, query, **kwargs):
        return _FakeAsyncResult({"n": 1})


class _FakeAsyncDriver:
    """Mimics neo4j.AsyncDriver — `.session()` is a plain (non-async) method
    that returns a fresh async-context-manager session each call, exactly
    like the real driver's API.
    """

    def session(self):
        return _FakeAsyncSession()


class _ConcurrencyTracker:
    """Records the max number of simultaneously in-flight async calls, via a
    real `await asyncio.sleep(...)` inside a lock-guarded counter. This is
    only ever `> 1` if the event loop was free to interleave the two calls —
    i.e. if the mocked function is a real coroutine that actually yields
    control, matching what a real `await session.run(...)` would do.
    """

    def __init__(self, sleep_seconds: float = 0.03) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._sleep_seconds = sleep_seconds
        self._lock = asyncio.Lock()

    async def track(self) -> None:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self._sleep_seconds)
        async with self._lock:
            self.in_flight -= 1


# ---------------------------------------------------------------------------
# 1. _run_entity_extraction_parallel — root-cause fix (fetch_extraction_targets)
# ---------------------------------------------------------------------------


class TestRunEntityExtractionParallelConcurrency:
    async def test_two_concurrent_calls_overlap_during_fetch(self):
        """Two concurrent `_run_entity_extraction_parallel()` calls must
        overlap their `fetch_extraction_targets()` awaits — proving it no
        longer blocks the event loop synchronously.

        Returns an empty target list from the mock so the rest of
        `_run_entity_extraction_parallel()` (LLM semaphore, per-target
        gather) is never reached — this test is scoped exactly to the
        fetch itself.
        """
        tracker = _ConcurrencyTracker()

        async def _tracking_fetch(driver, document_name, only_unextracted=False):
            await tracker.track()
            return []

        with (
            patch(
                "scinr.newton.entity_extraction.neo4j_ops.fetch_extraction_targets",
                AsyncMock(side_effect=_tracking_fetch),
            ),
            patch(
                "scinr.newton.ingest.config.get_async_driver",
                MagicMock(return_value=MagicMock(name="async_driver")),
            ),
        ):
            results = await asyncio.gather(
                _run_entity_extraction_parallel("docA"),
                _run_entity_extraction_parallel("docB"),
            )

        assert results == [
            {"document_name": "docA", "targets": [], "errors": []},
            {"document_name": "docB", "targets": [], "errors": []},
        ]
        assert tracker.max_in_flight == 2


# ---------------------------------------------------------------------------
# 2. run_entity_extraction_agent — precondition-check + leaf resolution
# ---------------------------------------------------------------------------


class TestRunEntityExtractionAgentPreconditionConcurrency:
    async def test_two_concurrent_calls_overlap_during_leaf_resolution(self):
        """Two concurrent `run_entity_extraction_agent()` calls must overlap
        during `resolve_leaf_document_names_async()` — proving the
        precondition-check / leaf-resolution plumbing (async driver session,
        `await session.run(...)`, `await result.single()`) no longer blocks
        the event loop synchronously either.

        `_run_entity_extraction_for_single_document()` is mocked to return
        trivially so the test stays scoped to the agent's own orchestration
        code, not the downstream per-target extraction logic (already
        covered by `TestRunEntityExtractionParallelConcurrency` above).
        """
        tracker = _ConcurrencyTracker()

        async def _tracking_resolve(driver, document_name):
            await tracker.track()
            return [document_name]

        fake_driver = _FakeAsyncDriver()

        with (
            patch(
                "scinr.newton.ingest.config.get_async_driver",
                MagicMock(return_value=fake_driver),
            ),
            patch(
                "scinr.newton.utils.document_resolver.resolve_leaf_document_names_async",
                AsyncMock(side_effect=_tracking_resolve),
            ),
            patch(
                "scinr.newton.entity_extraction.agent._run_entity_extraction_for_single_document",
                AsyncMock(
                    side_effect=lambda document_name, only_unextracted=False: {
                        "document_name": document_name,
                        "targets": [],
                        "errors": [],
                    }
                ),
            ),
        ):
            results = await asyncio.gather(
                run_entity_extraction_agent("docA"),
                run_entity_extraction_agent("docB"),
            )

        assert results == [
            {"document_name": "docA", "targets": [], "errors": []},
            {"document_name": "docB", "targets": [], "errors": []},
        ]
        assert tracker.max_in_flight == 2


# ---------------------------------------------------------------------------
# 3. Guardrails against silent future regressions
# ---------------------------------------------------------------------------


class TestFetchExtractionTargetsIsAsync:
    def test_fetch_extraction_targets_is_a_coroutine_function(self):
        """Guard against a future regression that quietly turns
        `fetch_extraction_targets()` back into a synchronous function without
        anyone noticing (e.g. during a refactor) — which would silently
        reintroduce the event-loop-blocking bug this suite protects against.
        """
        from scinr.newton.entity_extraction.neo4j_ops import fetch_extraction_targets

        assert inspect.iscoroutinefunction(fetch_extraction_targets) is True


class TestMarkInfoUnitsExtractedMigration:
    def test_sync_mark_info_units_extracted_was_removed(self):
        """The dead synchronous `mark_info_units_extracted()` function must
        stay removed — it was replaced entirely by the async
        `mark_info_units_extracted_async()` below.
        """
        import scinr.newton.entity_extraction.neo4j_ops as neo4j_ops_module

        assert not hasattr(neo4j_ops_module, "mark_info_units_extracted")

    def test_async_mark_info_units_extracted_exists_and_is_a_coroutine_function(self):
        from scinr.newton.entity_extraction.neo4j_ops import mark_info_units_extracted_async

        assert inspect.iscoroutinefunction(mark_info_units_extracted_async) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
