"""
tests/unit/test_entity_extraction_agent.py — Regression tests for the
event-loop-blocking fix in scinr.newton.entity_extraction.agent /
scinr.newton.graph.neo4j.extraction.

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
from scinr.newton.utils.document_resolver import LeafDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    async def test_two_concurrent_calls_overlap_during_fetch(self, fake_graph_store):
        """Two concurrent `_run_entity_extraction_parallel()` calls must
        overlap their `fetch_extraction_targets()` awaits — proving it no
        longer blocks the event loop synchronously.

        Returns an empty target list from the mock so the rest of
        `_run_entity_extraction_parallel()` (LLM semaphore, per-target
        gather) is never reached — this test is scoped exactly to the
        fetch itself.
        """
        tracker = _ConcurrencyTracker()

        async def _tracking_fetch(*, tenant_id, doc_path, only_unextracted=False):
            await tracker.track()
            return []

        fake_graph_store.extraction.fetch_targets.side_effect = _tracking_fetch

        results = await asyncio.gather(
            _run_entity_extraction_parallel("docA", tenant_id=None, doc_path="docA"),
            _run_entity_extraction_parallel("docB", tenant_id=None, doc_path="docB"),
        )

        assert results == [
            {"document_name": "docA", "doc_path": "docA", "targets": [], "errors": []},
            {"document_name": "docB", "doc_path": "docB", "targets": [], "errors": []},
        ]
        assert tracker.max_in_flight == 2


# ---------------------------------------------------------------------------
# 2. run_entity_extraction_agent — precondition-check + leaf resolution
# ---------------------------------------------------------------------------


class TestRunEntityExtractionAgentPreconditionConcurrency:
    async def test_two_concurrent_calls_overlap_during_leaf_resolution(self, fake_graph_store):
        """Two concurrent `run_entity_extraction_agent()` calls must overlap
        during `resolve_leaf_documents_async()` — proving the
        precondition-check / leaf-resolution plumbing (async driver session,
        `await session.run(...)`, `await result.single()`) no longer blocks
        the event loop synchronously either.

        `_run_entity_extraction_for_single_document()` is mocked to return
        trivially so the test stays scoped to the agent's own orchestration
        code, not the downstream per-target extraction logic (already
        covered by `TestRunEntityExtractionParallelConcurrency` above).
        """
        fake_graph_store.documents.count_latest_documents.return_value = 1
        fake_graph_store.annotation.count_annotated_nodes.return_value = 1
        tracker = _ConcurrencyTracker()

        async def _tracking_resolve(*, tenant_id, doc_path=None, document_name=None):
            await tracker.track()
            return [LeafDocument(document_name, document_name)]

        with (
            patch(
                "scinr.newton.utils.document_resolver.resolve_leaf_documents_async",
                AsyncMock(side_effect=_tracking_resolve),
            ),
            patch(
                "scinr.newton.entity_extraction.agent._run_entity_extraction_for_single_document",
                AsyncMock(
                    side_effect=lambda document_name, only_unextracted=False, *, tenant_id, doc_path: {
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


class TestRunEntityExtractionAgentPrecondition:
    async def test_missing_document_raises_precondition_error(self, fake_graph_store):
        from scinr.newton.exceptions import PreconditionError
        from scinr.newton.graph.types import DocumentSelector

        fake_graph_store.documents.count_latest_documents.return_value = 0

        with pytest.raises(PreconditionError, match="not found in the graph database"):
            await run_entity_extraction_agent("docA", tenant_id="acme", doc_path="F/docA")

        fake_graph_store.documents.count_latest_documents.assert_awaited_once_with(
            DocumentSelector(tenant_id="acme", path="F/docA", document_name="docA", latest=True)
        )
        fake_graph_store.annotation.count_annotated_nodes.assert_not_called()

    async def test_document_without_annotated_nodes_raises_precondition_error(
        self, fake_graph_store
    ):
        from scinr.newton.exceptions import PreconditionError
        from scinr.newton.graph.types import DocumentSelector

        fake_graph_store.documents.count_latest_documents.return_value = 1
        fake_graph_store.annotation.count_annotated_nodes.return_value = 0

        with pytest.raises(PreconditionError, match="no annotated StructureNodes"):
            await run_entity_extraction_agent("docA", tenant_id=None)

        fake_graph_store.annotation.count_annotated_nodes.assert_awaited_once_with(
            DocumentSelector(tenant_id=None, path=None, document_name="docA", latest=True)
        )
        fake_graph_store.documents.resolve_leaf_documents_async.assert_not_called()


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
        from scinr.newton.graph.neo4j.extraction import Neo4jExtractionOps

        assert inspect.iscoroutinefunction(Neo4jExtractionOps.fetch_targets) is True


class TestMarkInfoUnitsExtractedMigration:
    def test_sync_mark_info_units_extracted_was_removed(self):
        """The dead synchronous `mark_info_units_extracted()` function must
        stay removed — it was replaced entirely by the async
        `store.extraction.mark_info_units_extracted()` below.
        """
        import scinr.newton.graph.neo4j.extraction as extraction_module

        assert not hasattr(extraction_module, "mark_info_units_extracted")

    def test_async_mark_info_units_extracted_exists_and_is_a_coroutine_function(self):
        from scinr.newton.graph.neo4j.extraction import Neo4jExtractionOps

        assert inspect.iscoroutinefunction(Neo4jExtractionOps.mark_info_units_extracted) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
