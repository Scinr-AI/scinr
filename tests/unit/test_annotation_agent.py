"""
tests/unit/test_annotation_agent.py — Regression tests for the event-loop-
blocking fix in scinr.newton.annotation.agent /
scinr.newton.annotation.neo4j_ops, and for
scinr.newton.utils.document_resolver's new async leaf-resolution helper.

Mirrors tests/unit/test_entity_extraction_agent.py's mocking strategy and
rationale (see that file's module docstring for the full explanation of the
`_ConcurrencyTracker` pattern and why it deterministically detects a
regression to synchronous-blocking behavior).
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scinr.newton.annotation.agent import (
    _run_annotation_parallel,
    run_annotation_agent,
)

# ---------------------------------------------------------------------------
# Helpers — fake async Neo4j driver/session (async context manager protocol)
# ---------------------------------------------------------------------------


class _FakeAsyncResult:
    """Mimics neo4j.AsyncResult for the precondition-check query."""

    def __init__(self, record: dict) -> None:
        self._record = record

    async def single(self):
        return self._record


class _FakeAsyncSession:
    """Mimics neo4j.AsyncSession — answers every `.run(...)` with `n=1`,
    enough for `run_annotation_agent()`'s single precondition check
    (document exists) to pass regardless of the exact query text.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, query, **kwargs):
        return _FakeAsyncResult({"n": 1})


class _FakeAsyncDriver:
    """Mimics neo4j.AsyncDriver — `.session()` is a plain (non-async) method
    that returns a fresh async-context-manager session each call.
    """

    def session(self):
        return _FakeAsyncSession()


class _ConcurrencyTracker:
    """See test_entity_extraction_agent.py::_ConcurrencyTracker — identical
    pattern, duplicated here to keep each test file self-contained.
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
# 1. _run_annotation_parallel — root-cause fix (fetch_nodes_to_annotate)
# ---------------------------------------------------------------------------


class TestRunAnnotationParallelConcurrency:
    async def test_two_concurrent_calls_overlap_during_fetch(self):
        """Two concurrent `_run_annotation_parallel()` calls must overlap
        their `fetch_nodes_to_annotate()` awaits — proving it no longer
        blocks the event loop synchronously.

        `ensure_catalog_models_once()` / `ensure_theme_structure_once()` /
        `fetch_document_context_instructions()` are mocked as trivial
        no-ops (returning immediately) since they are not the function
        under test here; `fetch_nodes_to_annotate()` returns an empty node
        list so the rest of `_run_annotation_parallel()` (LLM semaphore,
        per-node gather) is never reached.
        """
        tracker = _ConcurrencyTracker()

        async def _tracking_fetch(driver, document_name, only_unannotated=False):
            await tracker.track()
            return []

        with (
            patch(
                "scinr.newton.annotation.neo4j_ops.ensure_catalog_models_once",
                AsyncMock(),
            ),
            patch(
                "scinr.newton.annotation.neo4j_ops.ensure_theme_structure_once",
                AsyncMock(),
            ),
            patch(
                "scinr.newton.annotation.neo4j_ops.fetch_nodes_to_annotate",
                AsyncMock(side_effect=_tracking_fetch),
            ),
            patch(
                "scinr.newton.annotation.neo4j_ops.fetch_document_context_instructions",
                AsyncMock(return_value=""),
            ),
            patch(
                "scinr.newton.ingest.config.get_async_driver",
                MagicMock(return_value=MagicMock(name="async_driver")),
            ),
            patch(
                "scinr.newton.utils.theme_registry.get_theme_registry",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            results = await asyncio.gather(
                _run_annotation_parallel("docA"),
                _run_annotation_parallel("docB"),
            )

        assert results == [
            {"document_name": "docA", "nodes_to_annotate": [], "errors": []},
            {"document_name": "docB", "nodes_to_annotate": [], "errors": []},
        ]
        assert tracker.max_in_flight == 2


# ---------------------------------------------------------------------------
# 2. run_annotation_agent — precondition-check + leaf resolution
# ---------------------------------------------------------------------------


class TestRunAnnotationAgentPreconditionConcurrency:
    async def test_two_concurrent_calls_overlap_during_leaf_resolution(self):
        """Two concurrent `run_annotation_agent()` calls must overlap during
        `resolve_leaf_document_names_async()` — proving the precondition-
        check / leaf-resolution plumbing (async driver session,
        `await session.run(...)`, `await result.single()`) no longer blocks
        the event loop synchronously either.

        `_run_annotation_for_single_document()` is mocked to return
        trivially so the test stays scoped to the agent's own orchestration
        code, not the downstream per-node annotation logic (already covered
        by `TestRunAnnotationParallelConcurrency` above).
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
                "scinr.newton.annotation.agent._run_annotation_for_single_document",
                AsyncMock(
                    side_effect=lambda document_name, only_unannotated=False, context_instructions_override=None: {
                        "document_name": document_name,
                        "nodes_to_annotate": [],
                        "errors": [],
                    }
                ),
            ),
        ):
            results = await asyncio.gather(
                run_annotation_agent("docA"),
                run_annotation_agent("docB"),
            )

        assert results == [
            {"document_name": "docA", "nodes_to_annotate": [], "errors": []},
            {"document_name": "docB", "nodes_to_annotate": [], "errors": []},
        ]
        assert tracker.max_in_flight == 2


# ---------------------------------------------------------------------------
# 3. Guardrails against silent future regressions
# ---------------------------------------------------------------------------


class TestDocumentResolverAsyncMigration:
    def test_resolve_leaf_document_names_async_is_a_coroutine_function(self):
        """Guard against a future regression that quietly turns
        `resolve_leaf_document_names_async()` back into a synchronous
        function without anyone noticing.
        """
        from scinr.newton.utils.document_resolver import resolve_leaf_document_names_async

        assert inspect.iscoroutinefunction(resolve_leaf_document_names_async) is True

    def test_sync_resolve_leaf_document_names_still_exists(self):
        """The original synchronous `resolve_leaf_document_names()` must
        remain untouched — `run_manual_annotation()`'s sibling code path and
        `pipeline_units._discover_pre_ingested_units()` (via
        `asyncio.to_thread()`) still depend on it directly.
        """
        from scinr.newton.utils.document_resolver import resolve_leaf_document_names

        assert inspect.iscoroutinefunction(resolve_leaf_document_names) is False
        assert callable(resolve_leaf_document_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
