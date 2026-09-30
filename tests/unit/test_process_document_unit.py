"""
tests/unit/test_process_document_unit.py — Unit tests for
scinr.newton.pipeline._process_document_unit().

_process_document_unit() is a new building block (Bloque B, Paso 1) not yet
wired into run_pipeline() (that is Paso 2, a separate task). These tests
exercise it in isolation with every real stage call mocked — no filesystem
I/O, no Neo4j, no LLM, no network.

Mocking strategy
-----------------
_process_document_unit() imports its collaborators via deferred imports
executed at call time:

    from scinr.newton.converters.main import convert_one
    from scinr.newton.ingest.loader import ingest_one, ingest_one_from_path
    from scinr.newton.pipeline_units import UnitResult
    from scinr.newton.stages import run_annotation, run_entity_extraction
    from scinr.newton.stages.extraction import extract_one_file, extract_one_intermediate

So monkeypatching the attributes on the *defining* modules
(``scinr.newton.converters.main``, ``scinr.newton.ingest.loader``,
``scinr.newton.stages``, ``scinr.newton.stages.extraction``) is picked up
correctly on every call, mirroring the pattern already used in
``tests/unit/test_pipeline_orchestration.py`` for ``run_pipeline()``.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scinr.newton.pipeline import _process_document_unit
from scinr.newton.pipeline_units import DocumentUnit
from scinr.newton.results import DocumentResult, StageResult


def _base_kwargs(**overrides: object) -> dict:
    """Build the keyword-only arguments of _process_document_unit() with
    sane defaults, overridable per test."""
    kwargs = dict(
        effective_stages=[
            "preprocess",
            "extraction",
            "ingestion",
            "annotation",
            "entity_extraction",
        ],
        document_semaphore=asyncio.Semaphore(1),
        sync_driver=MagicMock(),
        shared_ingest_version=1,
        converter_output_dir=None,
        extraction_output_dir=None,
        extraction_input_dir=None,
        raw_file_repo=None,
        page_repo=None,
        context_instructions=None,
        update_mode=False,
        manual=False,
        model_class=None,
        only_unannotated=False,
        only_unextracted=False,
        fast_extraction=False,
        on_partial_failure="abort",
    )
    kwargs.update(overrides)
    return kwargs


class TestFullSuccessRawFile:
    """Scenario (a): a raw_file unit traverses all five stages successfully."""

    async def test_all_five_stages_run_and_stopped_at_is_none(self, monkeypatch):
        unit = DocumentUnit(
            kind="raw_file",
            source_path=Path("/tmp/does-not-matter.pdf"),
            doc_path="doc1",
            relative_dir=Path("."),
            document_name_hint="doc1",
        )

        intermediate_doc = MagicMock(name="IntermediateDocument")
        mock_convert_one = AsyncMock(
            return_value=([(unit.source_path, Path("/tmp/out.json"), intermediate_doc)], [])
        )
        monkeypatch.setattr("scinr.newton.converters.main.convert_one", mock_convert_one)

        extracted_doc = SimpleNamespace(document_name="doc1", doc_path="doc1")
        mock_extract_one_intermediate = AsyncMock(return_value=extracted_doc)
        monkeypatch.setattr(
            "scinr.newton.stages.extraction.extract_one_intermediate",
            mock_extract_one_intermediate,
        )
        mock_extract_one_file = AsyncMock()
        monkeypatch.setattr(
            "scinr.newton.stages.extraction.extract_one_file", mock_extract_one_file
        )

        mock_ingest_one = AsyncMock(return_value="doc1")
        monkeypatch.setattr("scinr.newton.ingest.loader.ingest_one", mock_ingest_one)
        mock_ingest_one_from_path = AsyncMock()
        monkeypatch.setattr(
            "scinr.newton.ingest.loader.ingest_one_from_path", mock_ingest_one_from_path
        )

        mock_run_annotation = AsyncMock(
            return_value=StageResult(
                stage="annotation",
                success=True,
                documents=[DocumentResult("doc1", 3, 0)],
                total_processed=3,
                total_failed=0,
                duration_seconds=0.01,
            )
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)

        mock_run_entity_extraction = AsyncMock(
            return_value=StageResult(
                stage="entity_extraction",
                success=True,
                documents=[DocumentResult("doc1", 2, 0)],
                total_processed=2,
                total_failed=0,
                duration_seconds=0.01,
            )
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        kwargs = _base_kwargs()
        result = await _process_document_unit(unit, **kwargs)

        assert result.stopped_at is None
        assert result.fatal_error is None
        assert result.unit_id == "doc1"
        assert set(result.stage_results.keys()) == {
            "preprocess",
            "extraction",
            "ingestion",
            "annotation",
            "entity_extraction",
        }
        assert result.stage_results["preprocess"] == DocumentResult("doc1", 1, 0, [])
        assert result.stage_results["extraction"] == DocumentResult("doc1", 1, 0, [])
        assert result.stage_results["ingestion"] == DocumentResult("doc1", 1, 0, [])
        assert result.stage_results["annotation"] == DocumentResult("doc1", 3, 0, [])
        assert result.stage_results["entity_extraction"] == DocumentResult("doc1", 2, 0, [])

        mock_convert_one.assert_awaited_once()
        mock_extract_one_intermediate.assert_awaited_once_with(
            intermediate_doc,
            None,
            fast_extraction=False,
            tenant_id=None,
            created_by_user_id=None,
            job_id=None,
        )
        mock_extract_one_file.assert_not_called()
        mock_ingest_one.assert_awaited_once_with(
            extracted_doc,
            kwargs["sync_driver"],
            False,
            1,
            tenant_id=None,
            created_by_user_id=None,
            job_id=None,
        )
        mock_ingest_one_from_path.assert_not_called()
        mock_run_annotation.assert_awaited_once()
        mock_run_entity_extraction.assert_awaited_once()


class TestExtractionFailureStopsUnit:
    """Scenario (b): extraction returns None -> stopped_at='extraction' and
    later stages (ingestion/annotation/entity_extraction) are never called.
    """

    @pytest.mark.parametrize("on_partial_failure", ["abort", "continue", "warn"])
    async def test_extraction_none_stops_before_ingestion(self, monkeypatch, on_partial_failure):
        """Regression guard for scope creep: `extraction` returning `None`
        means there is no valid `doc_obj` for `ingestion` to operate on — a
        *total* failure of the unit, not a partial per-node one. This must
        always stop the unit's remaining stages, regardless of
        `on_partial_failure` (parametrized over all three values here to
        pin that explicitly), unlike the `annotation`/`entity_extraction`
        gate covered in `TestOnPartialFailureGatesAnnotationAndEntityExtraction`
        below.
        """
        unit = DocumentUnit(
            kind="extraction_json",
            source_path=Path("/tmp/some-file.json"),
            doc_path="doc2",
            relative_dir=Path("."),
            document_name_hint="doc2",
        )

        mock_convert_one = AsyncMock()
        monkeypatch.setattr("scinr.newton.converters.main.convert_one", mock_convert_one)

        mock_extract_one_intermediate = AsyncMock()
        monkeypatch.setattr(
            "scinr.newton.stages.extraction.extract_one_intermediate",
            mock_extract_one_intermediate,
        )
        mock_extract_one_file = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "scinr.newton.stages.extraction.extract_one_file", mock_extract_one_file
        )

        mock_ingest_one = AsyncMock()
        monkeypatch.setattr("scinr.newton.ingest.loader.ingest_one", mock_ingest_one)
        mock_ingest_one_from_path = AsyncMock()
        monkeypatch.setattr(
            "scinr.newton.ingest.loader.ingest_one_from_path", mock_ingest_one_from_path
        )

        mock_run_annotation = AsyncMock()
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock()
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        result = await _process_document_unit(
            unit,
            **_base_kwargs(
                effective_stages=[
                    "extraction",
                    "ingestion",
                    "annotation",
                    "entity_extraction",
                ],
                on_partial_failure=on_partial_failure,
            ),
        )

        assert result.stopped_at == "extraction"
        assert result.fatal_error is None
        assert result.stage_results["extraction"] == DocumentResult(
            "doc2", 0, 1, ["No pages to process"]
        )
        assert "ingestion" not in result.stage_results
        assert "annotation" not in result.stage_results
        assert "entity_extraction" not in result.stage_results

        mock_convert_one.assert_not_called()
        mock_extract_one_file.assert_awaited_once()
        mock_extract_one_intermediate.assert_not_called()
        mock_ingest_one.assert_not_called()
        mock_ingest_one_from_path.assert_not_called()
        mock_run_annotation.assert_not_called()
        mock_run_entity_extraction.assert_not_called()


class TestOnPartialFailureGatesAnnotationAndEntityExtraction:
    """Regression tests for the bug under fix: `annotation` reporting
    `nodes_failed > 0` for a document must only stop that document's
    advancement to `entity_extraction` when `on_partial_failure == "abort"`
    (the default). With `"continue"` or `"warn"`, `entity_extraction` must
    still run for that same document.
    """

    def _annotation_partial_failure_result(self, name: str) -> StageResult:
        return StageResult(
            stage="annotation",
            success=False,
            documents=[DocumentResult(name, 4, 1, ["node boom"])],
            total_processed=4,
            total_failed=1,
            duration_seconds=0.01,
        )

    def _entity_extraction_success_result(self, name: str) -> StageResult:
        return StageResult(
            stage="entity_extraction",
            success=True,
            documents=[DocumentResult(name, 2, 0)],
            total_processed=2,
            total_failed=0,
            duration_seconds=0.01,
        )

    async def test_abort_default_stops_before_entity_extraction(self, monkeypatch):
        """No `on_partial_failure` passed at all (relies on the function's
        own default) -> legacy behavior preserved: `entity_extraction` is
        never called and `stopped_at == "annotation"`.
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        result = await _process_document_unit(
            unit, **_base_kwargs(effective_stages=["annotation", "entity_extraction"])
        )

        assert result.stopped_at == "annotation"
        assert result.fatal_error is None
        assert result.stage_results["annotation"].nodes_failed == 1
        assert "entity_extraction" not in result.stage_results
        mock_run_annotation.assert_awaited_once()
        mock_run_entity_extraction.assert_not_called()

    async def test_explicit_abort_stops_before_entity_extraction(self, monkeypatch):
        """Same as above but with `on_partial_failure="abort"` passed
        explicitly, to pin the same behavior regardless of relying on the
        default.
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        result = await _process_document_unit(
            unit,
            **_base_kwargs(
                effective_stages=["annotation", "entity_extraction"],
                on_partial_failure="abort",
            ),
        )

        assert result.stopped_at == "annotation"
        assert "entity_extraction" not in result.stage_results
        mock_run_entity_extraction.assert_not_called()

    async def test_continue_advances_to_entity_extraction(self, monkeypatch):
        """`on_partial_failure="continue"` -> entity_extraction still runs
        for this document despite annotation's partial node failure, and
        `stopped_at` is `None` since entity_extraction itself succeeds.
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        result = await _process_document_unit(
            unit,
            **_base_kwargs(
                effective_stages=["annotation", "entity_extraction"],
                on_partial_failure="continue",
            ),
        )

        assert result.stopped_at is None
        assert result.fatal_error is None
        assert result.stage_results["annotation"].nodes_failed == 1
        assert result.stage_results["entity_extraction"] == DocumentResult(
            "doc-partial", 2, 0, []
        )
        mock_run_annotation.assert_awaited_once()
        mock_run_entity_extraction.assert_awaited_once()

    async def test_warn_advances_to_entity_extraction_like_continue(self, monkeypatch):
        """`on_partial_failure="warn"` behaves like `"continue"` at the
        per-unit level: entity_extraction still runs. (The stage-level
        aggregated warning log is emitted by `run_pipeline()`'s own
        aggregation loop, not by `_process_document_unit()` — out of scope
        here; the per-document warning emitted directly by
        `_process_document_unit()` itself is asserted in the dedicated
        `test_warn_logs_per_document_warning_for_annotation_failure` test
        below.)
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        result = await _process_document_unit(
            unit,
            **_base_kwargs(
                effective_stages=["annotation", "entity_extraction"],
                on_partial_failure="warn",
            ),
        )

        assert result.stopped_at is None
        assert result.stage_results["entity_extraction"] == DocumentResult(
            "doc-partial", 2, 0, []
        )
        mock_run_entity_extraction.assert_awaited_once()

    async def test_warn_logs_per_document_warning_for_annotation_failure(
        self, monkeypatch, caplog
    ):
        """`on_partial_failure="warn"` must emit an immediate per-document
        warning right at the point this unit decides to keep advancing
        despite `annotation` reporting a partial node failure — naming the
        document, the stage, the failed-node count, and the concrete error
        detail (`"node boom"`, injected via
        `_annotation_partial_failure_result`).
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        with caplog.at_level(logging.WARNING, logger="scinr.newton.pipeline"):
            result = await _process_document_unit(
                unit,
                **_base_kwargs(
                    effective_stages=["annotation", "entity_extraction"],
                    on_partial_failure="warn",
                ),
            )

        assert result.stopped_at is None
        warning_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_messages) == 1
        message = warning_messages[0]
        assert "doc-partial" in message
        assert "annotation" in message
        assert "1" in message  # failed-node count
        assert "node boom" in message
        assert "continuing" in message.lower()

    async def test_warn_logs_per_document_warning_for_entity_extraction_failure(
        self, monkeypatch, caplog
    ):
        """Same as the annotation case above, but for a partial node
        failure reported by `entity_extraction` itself.
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=StageResult(
                stage="annotation",
                success=True,
                documents=[DocumentResult("doc-partial", 4, 0)],
                total_processed=4,
                total_failed=0,
                duration_seconds=0.01,
            )
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=StageResult(
                stage="entity_extraction",
                success=False,
                documents=[DocumentResult("doc-partial", 1, 1, ["entity boom"])],
                total_processed=1,
                total_failed=1,
                duration_seconds=0.01,
            )
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        with caplog.at_level(logging.WARNING, logger="scinr.newton.pipeline"):
            result = await _process_document_unit(
                unit,
                **_base_kwargs(
                    effective_stages=["annotation", "entity_extraction"],
                    on_partial_failure="warn",
                ),
            )

        assert result.stopped_at is None
        warning_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_messages) == 1
        message = warning_messages[0]
        assert "doc-partial" in message
        assert "entity_extraction" in message
        assert "1" in message  # failed-node count
        assert "entity boom" in message
        assert "continuing" in message.lower()

    async def test_continue_does_not_log_per_document_warning(self, monkeypatch, caplog):
        """`on_partial_failure="continue"` must stay completely silent — no
        per-document warning is logged, even though the unit advances past
        the partial failure exactly like `"warn"` does.
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        with caplog.at_level(logging.WARNING, logger="scinr.newton.pipeline"):
            result = await _process_document_unit(
                unit,
                **_base_kwargs(
                    effective_stages=["annotation", "entity_extraction"],
                    on_partial_failure="continue",
                ),
            )

        assert result.stopped_at is None
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    async def test_abort_does_not_log_per_document_warning(self, monkeypatch, caplog):
        """`on_partial_failure="abort"` (default) must not emit the new
        per-document warning either — the unit aborts instead of
        advancing, so the "continuing" branch is never reached.
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=self._entity_extraction_success_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        with caplog.at_level(logging.WARNING, logger="scinr.newton.pipeline"):
            result = await _process_document_unit(
                unit,
                **_base_kwargs(
                    effective_stages=["annotation", "entity_extraction"],
                    on_partial_failure="abort",
                ),
            )

        assert result.stopped_at == "annotation"
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    async def test_continue_entity_extraction_own_failure_does_not_set_stopped_at(
        self, monkeypatch
    ):
        """`on_partial_failure="continue"` gates the soft-abort check after
        BOTH the `annotation` and `entity_extraction` blocks identically —
        so if `entity_extraction` itself also reports `nodes_failed > 0`,
        `stopped_at` stays `None` too (there's nothing left to advance to
        in this run anyway, but the point is the gate is applied
        consistently to both blocks, not just the first one).
        """
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="doc-partial",
            relative_dir=Path("."),
            document_name_hint="doc-partial",
        )

        mock_run_annotation = AsyncMock(
            return_value=self._annotation_partial_failure_result("doc-partial")
        )
        monkeypatch.setattr("scinr.newton.stages.run_annotation", mock_run_annotation)
        mock_run_entity_extraction = AsyncMock(
            return_value=StageResult(
                stage="entity_extraction",
                success=False,
                documents=[DocumentResult("doc-partial", 1, 1, ["entity boom"])],
                total_processed=1,
                total_failed=1,
                duration_seconds=0.01,
            )
        )
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", mock_run_entity_extraction)

        result = await _process_document_unit(
            unit,
            **_base_kwargs(
                effective_stages=["annotation", "entity_extraction"],
                on_partial_failure="continue",
            ),
        )

        assert result.stopped_at is None
        assert result.stage_results["entity_extraction"].nodes_failed == 1
        mock_run_entity_extraction.assert_awaited_once()


class TestPreprocessInMemoryOnly:
    """With `converter_output_dir=None` (the documented default for
    `run_pipeline(input_raw="files/")`) Stage 0 must not serialise the converted
    document to disk at all: `convert_one()` is called with `output_dir=None`
    and no temporary directory is created (the old implementation serialised
    the whole document to a temp dir that was deleted right away).
    """

    @staticmethod
    def _unit() -> DocumentUnit:
        return DocumentUnit(
            kind="raw_file",
            source_path=Path("/tmp/does-not-matter.pdf"),
            doc_path="doc1",
            relative_dir=Path("."),
            document_name_hint="doc1",
        )

    async def test_convert_one_receives_none_and_no_tempdir_entries_appear(self, monkeypatch):
        unit = self._unit()
        captured: list[object] = []

        async def _fake_convert_one(entry, output_dir, **kwargs):
            captured.append(output_dir)
            return ([(entry, None, MagicMock(name="IntermediateDocument"))], [])

        monkeypatch.setattr("scinr.newton.converters.main.convert_one", _fake_convert_one)

        before = set(Path(tempfile.gettempdir()).iterdir())
        result = await _process_document_unit(
            unit,
            **_base_kwargs(effective_stages=["preprocess"], converter_output_dir=None),
        )
        after = set(Path(tempfile.gettempdir()).iterdir())

        assert result.stage_results["preprocess"] == DocumentResult("doc1", 1, 0, [])
        assert captured == [None]
        assert after - before == set()

    async def test_explicit_converter_output_dir_is_forwarded_as_path(self, monkeypatch, tmp_path):
        unit = self._unit()
        captured: list[object] = []

        async def _fake_convert_one(entry, output_dir, **kwargs):
            captured.append(output_dir)
            return ([(entry, output_dir / "x.json", MagicMock())], [])

        monkeypatch.setattr("scinr.newton.converters.main.convert_one", _fake_convert_one)

        await _process_document_unit(
            unit,
            **_base_kwargs(effective_stages=["preprocess"], converter_output_dir=str(tmp_path)),
        )

        assert captured == [tmp_path]

    async def test_convert_one_raising_becomes_fatal_error_without_tempdir_leak(
        self, monkeypatch
    ):
        unit = self._unit()

        async def _raising_convert_one(entry, output_dir, **kwargs):
            raise RuntimeError("boom: unexpected converter crash")

        monkeypatch.setattr("scinr.newton.converters.main.convert_one", _raising_convert_one)

        before = set(Path(tempfile.gettempdir()).iterdir())
        result = await _process_document_unit(
            unit,
            **_base_kwargs(effective_stages=["preprocess"], converter_output_dir=None),
        )
        after = set(Path(tempfile.gettempdir()).iterdir())

        assert result.fatal_error is not None
        assert "boom" in result.fatal_error
        assert after - before == set()

    def test_pipeline_module_no_longer_uses_tempfile(self):
        import scinr.newton.pipeline as pipeline_mod

        assert not hasattr(pipeline_mod, "tempfile")


class TestArtifactsReleasedPerStage:
    """Each per-document artifact is dropped as soon as its stage is done, so it
    does not stay alive through the (long) later stages. The test keeps only
    weak references and asserts them from inside the fake next-stage calls.
    """

    async def test_intermediate_and_extracted_docs_are_freed_before_later_stages(
        self, monkeypatch
    ):
        import gc
        import weakref

        unit = DocumentUnit(
            kind="raw_file",
            source_path=Path("/tmp/does-not-matter.pdf"),
            doc_path="doc1",
            relative_dir=Path("."),
            document_name_hint="doc1",
        )

        class _Intermediate:
            pass

        class _Extracted:
            document_name = "doc1"
            doc_path = "doc1"

        refs: dict[str, weakref.ref] = {}
        seen: dict[str, bool] = {}

        async def fake_convert_one(entry, output_dir, **kwargs):
            doc = _Intermediate()
            refs["intermediate"] = weakref.ref(doc)
            return ([(entry, None, doc)], [])

        async def fake_extract_one_intermediate(doc, *args, **kwargs):
            extracted = _Extracted()
            refs["extracted"] = weakref.ref(extracted)
            return extracted

        async def fake_ingest_one(doc, *args, **kwargs):
            gc.collect()
            # Stage 1 is over: the intermediate document must already be gone.
            seen["intermediate_freed_at_ingest"] = refs["intermediate"]() is None
            return "doc1"

        def _stage_result(stage: str) -> StageResult:
            return StageResult(
                stage=stage,
                success=True,
                documents=[DocumentResult("doc1", 1, 0)],
                total_processed=1,
                total_failed=0,
                duration_seconds=0.0,
            )

        async def fake_run_annotation(*args, **kwargs):
            gc.collect()
            # Stage 2 is over: the extracted document must already be gone.
            seen["intermediate_freed_at_annotation"] = refs["intermediate"]() is None
            seen["extracted_freed_at_annotation"] = refs["extracted"]() is None
            return _stage_result("annotation")

        async def fake_run_entity_extraction(*args, **kwargs):
            return _stage_result("entity_extraction")

        monkeypatch.setattr("scinr.newton.converters.main.convert_one", fake_convert_one)
        monkeypatch.setattr(
            "scinr.newton.stages.extraction.extract_one_intermediate",
            fake_extract_one_intermediate,
        )
        monkeypatch.setattr("scinr.newton.ingest.loader.ingest_one", fake_ingest_one)
        monkeypatch.setattr("scinr.newton.stages.run_annotation", fake_run_annotation)
        monkeypatch.setattr(
            "scinr.newton.stages.run_entity_extraction", fake_run_entity_extraction
        )

        result = await _process_document_unit(unit, **_base_kwargs())

        assert result.stopped_at is None
        assert result.fatal_error is None
        assert seen == {
            "intermediate_freed_at_ingest": True,
            "intermediate_freed_at_annotation": True,
            "extracted_freed_at_annotation": True,
        }


class TestTenantAndPathReachStages3And4:
    """Annotation / entity extraction select the document by (tenant, path),
    never by name — see plans/multitenancy-document-identity-plan.md WP4."""

    @staticmethod
    def _ok(stage: str, name: str) -> StageResult:
        return StageResult(
            stage=stage,
            success=True,
            documents=[DocumentResult(name, 1, 0, [])],
            total_processed=1,
            total_failed=0,
            duration_seconds=0.0,
        )

    async def test_pre_ingested_unit_forwards_run_tenant_and_leaf_path(self, monkeypatch):
        unit = DocumentUnit(
            kind="pre_ingested",
            source_path=None,
            doc_path="Folder/leaf",
            relative_dir=Path("."),
            document_name_hint="leaf",
        )
        ann = AsyncMock(return_value=self._ok("annotation", "leaf"))
        ee = AsyncMock(return_value=self._ok("entity_extraction", "leaf"))
        monkeypatch.setattr("scinr.newton.stages.run_annotation", ann)
        monkeypatch.setattr("scinr.newton.stages.run_entity_extraction", ee)

        result = await _process_document_unit(
            unit,
            **_base_kwargs(effective_stages=["annotation", "entity_extraction"], tenant_id="acme"),
        )

        assert result.unit_id == "leaf"
        for mock in (ann, ee):
            args, kwargs = mock.await_args
            assert args[0] == "leaf"
            assert kwargs["tenant_id"] == "acme"
            assert kwargs["doc_path"] == "Folder/leaf"

    async def test_ingestion_json_unit_uses_the_baked_tenant_when_run_has_none(
        self, monkeypatch, tmp_path
    ):
        extract = tmp_path / "extract-doc.json"
        extract.write_text(
            '{"document_name": "doc", "doc_path": "F/doc", "tenant_id": "baked"}',
            encoding="utf-8",
        )
        unit = DocumentUnit(
            kind="ingestion_json",
            source_path=extract,
            doc_path="F/doc",
            relative_dir=Path("."),
            document_name_hint="doc",
        )
        monkeypatch.setattr(
            "scinr.newton.ingest.loader.ingest_one_from_path", AsyncMock(return_value="doc")
        )
        ann = AsyncMock(return_value=self._ok("annotation", "doc"))
        monkeypatch.setattr("scinr.newton.stages.run_annotation", ann)

        await _process_document_unit(
            unit, **_base_kwargs(effective_stages=["ingestion", "annotation"])
        )

        assert ann.await_args.kwargs["tenant_id"] == "baked"
        assert ann.await_args.kwargs["doc_path"] == "F/doc"

    async def test_extracted_document_path_wins_over_the_predicted_one(self, monkeypatch):
        unit = DocumentUnit(
            kind="extraction_json",
            source_path=Path("/tmp/x.json"),
            doc_path="predicted",
            relative_dir=Path("."),
            document_name_hint="x",
        )
        monkeypatch.setattr(
            "scinr.newton.stages.extraction.extract_one_file",
            AsyncMock(return_value=SimpleNamespace(document_name="x", doc_path="Real/x")),
        )
        monkeypatch.setattr("scinr.newton.ingest.loader.ingest_one", AsyncMock(return_value="x"))
        ann = AsyncMock(return_value=self._ok("annotation", "x"))
        monkeypatch.setattr("scinr.newton.stages.run_annotation", ann)

        await _process_document_unit(
            unit,
            **_base_kwargs(effective_stages=["extraction", "ingestion", "annotation"], tenant_id=None),
        )

        assert ann.await_args.kwargs == {**ann.await_args.kwargs, "tenant_id": None, "doc_path": "Real/x"}


class TestPreprocessStoresUnderTheUnitsOwner:
    """The raw file / pages stored by preprocess carry the unit's tenant and
    provenance (plans/multitenancy-document-storage-plan.md WP3)."""

    async def test_convert_one_receives_tenant_and_provenance(self, monkeypatch):
        unit = DocumentUnit(
            kind="raw_file",
            source_path=Path("/tmp/does-not-matter.pdf"),
            doc_path="doc1",
            relative_dir=Path("."),
            document_name_hint="doc1",
        )
        mock_convert_one = AsyncMock(return_value=([], []))
        monkeypatch.setattr("scinr.newton.converters.main.convert_one", mock_convert_one)

        await _process_document_unit(
            unit,
            **_base_kwargs(
                effective_stages=["preprocess"],
                tenant_id="acme",
                created_by_user_id="u1",
                job_id="j1",
            ),
        )

        kwargs = mock_convert_one.await_args.kwargs
        assert (kwargs["tenant_id"], kwargs["created_by_user_id"], kwargs["job_id"]) == (
            "acme", "u1", "j1",
        )
