"""
tests/unit/test_manual_annotation.py — Regression tests for the missing
`await` bug in `scinr.newton.annotation.agent.run_manual_annotation()`.

Root cause (fixed): line ~343 called
`write_manual_annotation(driver, leaf_name, model_class)` without `await`.
Since `write_manual_annotation()` is a coroutine function, this produced a
coroutine object instead of an `int`. `total_count += count` then raised a
`TypeError`, which was silently swallowed by the broad
`except Exception as exc: logger.error(...)` wrapping each leaf's processing
— so `run_manual_annotation()` always logged an error per leaf, never wrote
anything to Neo4j, and always returned `0`, without ever raising a visible
exception to the caller.

Mirrors tests/unit/test_annotation_agent.py's mocking strategy and style
(patching symbols at the module where `run_manual_annotation()`'s *local*
imports resolve them from, since those imports happen inside the function
body at call time).
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scinr.newton.annotation.agent import run_manual_annotation

# ---------------------------------------------------------------------------
# 1. Success path, single (non-folder) document
# ---------------------------------------------------------------------------


class TestRunManualAnnotationSingleDocumentSuccess:
    async def test_awaits_write_manual_annotation_and_returns_its_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single leaf document must have `write_manual_annotation()`
        properly *awaited* (not merely called) and its returned `int` count
        propagated as the final result — not `0`, not a coroutine object,
        and not a `TypeError` silently swallowed by the per-leaf `except`.

        This is exactly the assertion that distinguishes the buggy
        (missing-`await`) behavior from the fixed one: with `AsyncMock`,
        calling without awaiting still "succeeds" (no exception at call
        time) but leaves `await_count == 0` and the coroutine unresolved, so
        `assert_awaited_once_with(...)` fails and `total_count` stays `0`
        (the `TypeError` from `total_count += <coroutine>` is caught and
        logged as an ERROR instead of propagating).
        """
        driver_dummy = MagicMock(name="async_driver")
        write_mock = AsyncMock(return_value=5)

        with (
            patch(
                "scinr.newton.entity_extraction.model_resolver.resolve_model_class",
                MagicMock(return_value=object),
            ),
            patch(
                "scinr.newton.ingest.config.get_async_driver",
                MagicMock(return_value=driver_dummy),
            ),
            patch(
                "scinr.newton.utils.document_resolver.resolve_leaf_document_names_async",
                AsyncMock(return_value=["MyDoc"]),
            ),
            patch(
                "scinr.newton.annotation.neo4j_ops.write_manual_annotation",
                write_mock,
            ),
            caplog.at_level(logging.ERROR, logger="scinr.newton.annotation.agent"),
        ):
            result = await run_manual_annotation("MyDoc", "SomeModelClass")

        assert result == 5
        write_mock.assert_awaited_once_with(driver_dummy, "MyDoc", "SomeModelClass")

        error_records = [
            r for r in caplog.records if r.name == "scinr.newton.annotation.agent"
        ]
        assert not any("Manual annotation failed" in r.getMessage() for r in error_records)


# ---------------------------------------------------------------------------
# 2. Success path, folder with multiple leaves
# ---------------------------------------------------------------------------


class TestRunManualAnnotationFolderMultipleLeaves:
    async def test_sums_counts_across_all_leaves(self) -> None:
        """When `document_name` resolves to multiple leaf documents, the
        per-leaf counts returned by `write_manual_annotation()` must be
        summed correctly, and each leaf must be called with its own name.
        """
        driver_dummy = MagicMock(name="async_driver")
        write_mock = AsyncMock(side_effect=[3, 7])

        with (
            patch(
                "scinr.newton.entity_extraction.model_resolver.resolve_model_class",
                MagicMock(return_value=object),
            ),
            patch(
                "scinr.newton.ingest.config.get_async_driver",
                MagicMock(return_value=driver_dummy),
            ),
            patch(
                "scinr.newton.utils.document_resolver.resolve_leaf_document_names_async",
                AsyncMock(return_value=["Leaf1", "Leaf2"]),
            ),
            patch(
                "scinr.newton.annotation.neo4j_ops.write_manual_annotation",
                write_mock,
            ),
        ):
            result = await run_manual_annotation("SomeFolder", "SomeModelClass")

        assert result == 10
        assert write_mock.await_count == 2
        write_mock.assert_any_await(driver_dummy, "Leaf1", "SomeModelClass")
        write_mock.assert_any_await(driver_dummy, "Leaf2", "SomeModelClass")


# ---------------------------------------------------------------------------
# 3. Per-leaf isolated failure (pre-existing behavior, must not regress)
# ---------------------------------------------------------------------------


class TestRunManualAnnotationPerLeafFailureIsolation:
    async def test_one_failing_leaf_does_not_abort_remaining_leaves(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A genuine failure in `write_manual_annotation()` for one leaf
        (e.g., a Neo4j error) must be caught and logged, while subsequent
        leaves keep processing normally and their counts still contribute
        to the final total. No exception should propagate out of
        `run_manual_annotation()`.
        """
        driver_dummy = MagicMock(name="async_driver")
        write_mock = AsyncMock(side_effect=[Exception("boom"), 4])

        with (
            patch(
                "scinr.newton.entity_extraction.model_resolver.resolve_model_class",
                MagicMock(return_value=object),
            ),
            patch(
                "scinr.newton.ingest.config.get_async_driver",
                MagicMock(return_value=driver_dummy),
            ),
            patch(
                "scinr.newton.utils.document_resolver.resolve_leaf_document_names_async",
                AsyncMock(return_value=["Leaf1", "Leaf2"]),
            ),
            patch(
                "scinr.newton.annotation.neo4j_ops.write_manual_annotation",
                write_mock,
            ),
            caplog.at_level(logging.ERROR, logger="scinr.newton.annotation.agent"),
        ):
            result = await run_manual_annotation("SomeFolder", "SomeModelClass")

        assert result == 4
        error_records = [
            r for r in caplog.records if r.name == "scinr.newton.annotation.agent"
        ]
        assert any(
            "Manual annotation failed" in r.getMessage() and "Leaf1" in r.getMessage()
            for r in error_records
        )


# ---------------------------------------------------------------------------
# 4. Guardrail against a future regression back to a synchronous function
# ---------------------------------------------------------------------------


class TestWriteManualAnnotationIsCoroutineFunction:
    def test_write_manual_annotation_is_a_coroutine_function(self) -> None:
        """Guard against a future regression that quietly turns
        `write_manual_annotation()` back into a synchronous function
        without anyone updating its (currently `await`-ing) caller in
        `run_manual_annotation()`.
        """
        from scinr.newton.annotation.neo4j_ops import write_manual_annotation

        assert inspect.iscoroutinefunction(write_manual_annotation) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
