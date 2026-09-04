"""
tests/unit/test_pipeline_orchestration.py — Pinning/regression tests for
scinr.newton.pipeline.run_pipeline().

These tests capture the CURRENT orchestration behavior of run_pipeline() —
stage sequencing, on_partial_failure semantics ("abort" / "continue" / "warn"),
and the way the per-document-unit engine (Bloque B) aggregates per-document
results produced via `asyncio.gather(..., return_exceptions=True)` into a
single synthetic StageResult per stage — using mocks only. No Neo4j, no LLM,
no real filesystem I/O beyond `tmp_path`, no network.

Purpose
-------
Act as a safety net around the per-document-unit orchestration engine
(`_process_document_unit()` / `_discover_units()`, Bloque B Pasos 1-2) that
now backs `run_pipeline()`. Run this exact suite against future refactors:
  - If a test fails and the difference is an unintended regression -> fix the
    production code.
  - If a test fails because the change is an *intentional* behavior change of
    the refactor -> update this test deliberately (and explain why in the
    diff/commit).

Mocking strategy
-----------------
`run_pipeline()` no longer calls the batch stage functions
(`run_preprocess`/`run_extraction`/`run_ingestion`) directly — it discovers
one `DocumentUnit` per document/file via `_discover_units()` and fans out to
`_process_document_unit()`, which in turn calls the per-document primitives
`convert_one()`, `extract_one_file()`/`extract_one_intermediate()`,
`ingest_one()`/`ingest_one_from_path()`, and (still) the batch-shaped
`run_annotation()`/`run_entity_extraction()` (each invoked with a single leaf
document name). All of these are imported *inside* the relevant function via
a deferred `from <module> import ...` executed at call time, so
monkeypatching the attribute on the function's ORIGIN module (not on
`scinr.newton.pipeline`) is what gets picked up on every call. See
`mock_stages`, `mock_infra`, and `mock_unit_stage_fns` below for exactly which
module each patch target lives on.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from scinr.newton.config import configure
from scinr.newton.pipeline import run_pipeline
from scinr.newton.results import DocumentResult, PipelineResult, StageResult
from scinr.newton.stages import run_preprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sr(
    stage: str,
    success: bool = True,
    documents: list[DocumentResult] | None = None,
    processed: int = 0,
    failed: int = 0,
    duration: float = 0.01,
    errors: list[str] | None = None,
) -> StageResult:
    """Build a StageResult with sane defaults, to keep tests terse."""
    return StageResult(
        stage=stage,
        success=success,
        documents=documents or [],
        total_processed=processed,
        total_failed=failed,
        duration_seconds=duration,
        errors=errors or [],
    )


@pytest.fixture
def mock_stages(monkeypatch):
    """Patch every stage function on scinr.newton.stages with an AsyncMock.

    `run_pipeline()` (for `run_tabular_pipeline`) and `_process_document_unit()`
    (for `run_annotation`/`run_entity_extraction`, each called once per leaf
    document) re-import these names from `scinr.newton.stages` on every call
    (deferred import inside the function body), so patching the attributes
    on the package module is sufficient — no need to patch
    `scinr.newton.pipeline` itself. `run_preprocess`/`run_extraction`/
    `run_ingestion` are no longer called by `run_pipeline()` (see
    `mock_unit_stage_fns` for their per-document replacements) but are kept
    patched here too since `scinr.newton.stages.run_preprocess` is still a
    public re-export used directly by `cli.py` and by
    `TestStage0Baseline.test_stage0_reports_conversion_failures_as_failed_documents`
    below (which imports the real, unpatched `run_preprocess`).
    """
    import scinr.newton.stages as stages_mod

    names = [
        "run_preprocess",
        "run_extraction",
        "run_ingestion",
        "run_annotation",
        "run_entity_extraction",
        "run_tabular_pipeline",
    ]
    mocks = {name: AsyncMock() for name in names}
    for name, mock in mocks.items():
        monkeypatch.setattr(stages_mod, name, mock)
    return mocks


@pytest.fixture
def mock_infra(monkeypatch):
    """Patch every Neo4j/driver/theme-registry touchpoint that
    `run_pipeline()` and `_discover_units()` hit directly (outside the
    per-document stage primitives), so tests never open a real Neo4j
    connection or scan the real theme/model registry.

    Patches (module-attribute patching on each function's ORIGIN module,
    matching the deferred-import pattern documented on `mock_stages` above):
      - `scinr.newton.ingest.config.get_driver` / `get_async_driver`
      - `scinr.newton.ingest.schema.setup_schema`
      - `scinr.newton.ingest.loader.resolve_batch_version_sync`
      - `scinr.newton.annotation.neo4j_ops.ensure_catalog_models_once` /
        `ensure_theme_structure_once` (no-op AsyncMocks — see
        `TestCatalogMemoization` below for a test that leaves these REAL and
        mocks one layer deeper instead)
      - `scinr.newton.utils.theme_registry.get_theme_registry`
      - `scinr.newton.utils.document_resolver.resolve_leaf_document_names`
        (identity: returns `[document_name]`, i.e. "already a leaf",
        emulating `document_names` resolution without touching Neo4j)
    """
    import scinr.newton.annotation.neo4j_ops as neo4j_ops_mod
    import scinr.newton.ingest.config as ingest_config_mod
    import scinr.newton.ingest.loader as ingest_loader_mod
    import scinr.newton.ingest.schema as ingest_schema_mod
    import scinr.newton.utils.document_resolver as document_resolver_mod
    import scinr.newton.utils.theme_registry as theme_registry_mod

    fake_sync_driver = MagicMock(name="sync_driver")
    fake_async_driver = MagicMock(name="async_driver")

    monkeypatch.setattr(ingest_config_mod, "get_driver", MagicMock(return_value=fake_sync_driver))
    monkeypatch.setattr(
        ingest_config_mod, "get_async_driver", MagicMock(return_value=fake_async_driver)
    )
    monkeypatch.setattr(ingest_schema_mod, "setup_schema", MagicMock())
    monkeypatch.setattr(ingest_loader_mod, "resolve_batch_version_sync", MagicMock(return_value=1))
    monkeypatch.setattr(neo4j_ops_mod, "ensure_catalog_models_once", AsyncMock())
    monkeypatch.setattr(neo4j_ops_mod, "ensure_theme_structure_once", AsyncMock())
    monkeypatch.setattr(
        theme_registry_mod, "get_theme_registry", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(
        document_resolver_mod,
        "resolve_leaf_document_names",
        MagicMock(side_effect=lambda driver, name: [name]),
    )
    return {"sync_driver": fake_sync_driver, "async_driver": fake_async_driver}


@pytest.fixture
def mock_unit_stage_fns(monkeypatch):
    """Patch the per-document primitives that `_process_document_unit()`
    calls directly for Stages 0-2, on their ORIGIN modules (one layer deeper
    than the batch functions `mock_stages` patches).

    Default `side_effect`s derive a deterministic `document_name`/return
    value from the input file so multi-document tests can distinguish
    documents with no extra plumbing; override `.side_effect` /
    `.return_value` on the returned mocks for scenarios needing
    per-document branching (e.g. one document failing).
    """
    import scinr.newton.converters.main as converters_main_mod
    import scinr.newton.ingest.loader as ingest_loader_mod
    import scinr.newton.stages.extraction as extraction_mod

    async def _default_extract_one_file(json_file, output_path, input_folder, fast_extraction=False):
        doc = MagicMock(name=f"Document({json_file.stem})")
        doc.document_name = json_file.stem
        return doc

    async def _default_ingest_one(doc, driver, update_mode=False, shared_version=None):
        return doc.document_name

    async def _default_ingest_one_from_path(path, driver, update_mode=False, shared_version=None):
        return path.stem

    mocks = {
        "extract_one_file": AsyncMock(side_effect=_default_extract_one_file),
        "extract_one_intermediate": AsyncMock(),
        "ingest_one": AsyncMock(side_effect=_default_ingest_one),
        "ingest_one_from_path": AsyncMock(side_effect=_default_ingest_one_from_path),
        "convert_one": AsyncMock(),
    }
    monkeypatch.setattr(extraction_mod, "extract_one_file", mocks["extract_one_file"])
    monkeypatch.setattr(
        extraction_mod, "extract_one_intermediate", mocks["extract_one_intermediate"]
    )
    monkeypatch.setattr(ingest_loader_mod, "ingest_one", mocks["ingest_one"])
    monkeypatch.setattr(ingest_loader_mod, "ingest_one_from_path", mocks["ingest_one_from_path"])
    monkeypatch.setattr(converters_main_mod, "convert_one", mocks["convert_one"])
    return mocks


def _write_extraction_jsons(tmp_path, names: list[str]) -> None:
    """Write one minimal Stage-0-shaped intermediate JSON file per name in
    *names* under *tmp_path*, matching exactly what
    `_discover_extraction_json_units()` reads (only the top-level
    `"folder_path"` field is inspected during discovery; the real per-file
    content is irrelevant here since `extract_one_file()` is mocked by
    `mock_unit_stage_fns` for every test that uses this helper).
    """
    for name in names:
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"folder_path": None, "pages": []}), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# on_partial_failure="abort" (current default)
# ---------------------------------------------------------------------------


class TestOnPartialFailureAbort:
    async def test_abort_stops_pipeline_before_later_stages(
        self,
        mock_stages,
        mock_infra,
        mock_unit_stage_fns,
        tmp_path,
    ):
        """A failed intermediate stage (annotation) for one document must
        stop THAT document's own remaining stages ("soft-abort", per-unit)
        when `on_partial_failure="abort"` (the default): entity_extraction is
        never invoked for the failing document, while its sibling — whose
        annotation succeeds — DOES reach entity_extraction normally.

        This per-unit soft-abort is gated by `on_partial_failure` for the
        `annotation`/`entity_extraction` stages specifically (see
        `TestOnPartialFailureContinue` / `TestOnPartialFailureWarn` below for
        the "continue"/"warn" counterparts, where the failing document's
        chain keeps advancing instead of stopping here).
        """
        _write_extraction_jsons(tmp_path, ["doc-good", "doc-bad"])

        async def _annotation_side_effect(document_name, **kwargs):
            if document_name == "doc-bad":
                return _sr(
                    "annotation",
                    success=False,
                    documents=[DocumentResult("doc-bad", 0, 1, ["boom"])],
                    processed=0,
                    failed=1,
                    errors=["boom"],
                )
            return _sr(
                "annotation",
                success=True,
                documents=[DocumentResult(document_name, 5, 0)],
                processed=5,
                failed=0,
            )

        mock_stages["run_annotation"].side_effect = _annotation_side_effect
        mock_stages["run_entity_extraction"].return_value = _sr(
            "entity_extraction",
            success=True,
            documents=[DocumentResult("doc-good", 5, 0)],
            processed=5,
            failed=0,
        )

        result = await run_pipeline(
            stages=["extraction", "ingestion", "annotation", "entity_extraction"],
            extraction_input_dir=str(tmp_path),
            on_partial_failure="abort",
        )

        # entity_extraction is invoked exactly once — for the surviving
        # sibling only. The failing document never reaches it.
        mock_stages["run_entity_extraction"].assert_called_once()
        ee_call_args = mock_stages["run_entity_extraction"].call_args[0]
        assert ee_call_args[0] == "doc-good"

        assert result.annotation is not None
        assert result.annotation.total_failed == 1
        assert result.entity_extraction is not None
        assert [d.document_name for d in result.entity_extraction.documents] == ["doc-good"]

    async def test_abort_is_the_default_value(self, mock_stages):
        """Pin the current default of on_partial_failure to 'abort' — if the
        refactor changes this default, this test must fail loudly.
        """
        import inspect

        sig = inspect.signature(run_pipeline)
        assert sig.parameters["on_partial_failure"].default == "abort"


# ---------------------------------------------------------------------------
# on_partial_failure="continue"
# ---------------------------------------------------------------------------


class TestOnPartialFailureContinue:
    async def test_continue_runs_all_stages_despite_failure(
        self, mock_stages, mock_infra, mock_unit_stage_fns, tmp_path
    ):
        """Same failing-annotation scenario as the abort test, but explicitly
        pinned for on_partial_failure='continue': BOTH documents advance all
        the way to entity_extraction — the surviving sibling because its own
        annotation succeeded, and doc-bad because 'continue' means a partial
        node-level failure in one stage (`nodes_failed > 0`) no longer stops
        that document from reaching its next requested stage.
        """
        _write_extraction_jsons(tmp_path, ["doc-good", "doc-bad"])

        async def _annotation_side_effect(document_name, **kwargs):
            if document_name == "doc-bad":
                return _sr(
                    "annotation",
                    success=False,
                    documents=[DocumentResult("doc-bad", 0, 1, ["boom"])],
                    processed=0,
                    failed=1,
                    errors=["boom"],
                )
            return _sr(
                "annotation",
                success=True,
                documents=[DocumentResult(document_name, 5, 0)],
                processed=5,
                failed=0,
            )

        async def _entity_extraction_side_effect(document_name, **kwargs):
            return _sr(
                "entity_extraction",
                success=True,
                documents=[DocumentResult(document_name, 5, 0)],
                processed=5,
                failed=0,
            )

        mock_stages["run_annotation"].side_effect = _annotation_side_effect
        mock_stages["run_entity_extraction"].side_effect = _entity_extraction_side_effect

        result = await run_pipeline(
            stages=["extraction", "ingestion", "annotation", "entity_extraction"],
            extraction_input_dir=str(tmp_path),
            on_partial_failure="continue",
        )

        assert mock_stages["run_entity_extraction"].call_count == 2
        ee_call_names = {
            call.args[0] for call in mock_stages["run_entity_extraction"].call_args_list
        }
        assert ee_call_names == {"doc-good", "doc-bad"}

        # Overall pipeline is still reported as failed (annotation failed for
        # doc-bad), but BOTH documents' chains ran to completion.
        assert result.success is False
        assert result.annotation.success is False
        assert result.entity_extraction is not None
        assert result.entity_extraction.success is True
        assert {d.document_name for d in result.entity_extraction.documents} == {
            "doc-good",
            "doc-bad",
        }


# ---------------------------------------------------------------------------
# on_partial_failure="warn"
# ---------------------------------------------------------------------------


class TestOnPartialFailureWarn:
    async def test_warn_logs_and_continues_like_continue(
        self, mock_stages, mock_infra, mock_unit_stage_fns, tmp_path, caplog
    ):
        """on_partial_failure='warn' logs a warning for the failed stage but
        behaves like 'continue' at the per-unit level: entity_extraction
        still runs for BOTH documents (the surviving sibling, and doc-bad
        whose earlier annotation only partially failed).

        Two distinct warnings must coexist in 'warn' mode: the immediate
        per-document warning emitted by `_process_document_unit()` itself
        the moment doc-bad decides to keep advancing despite its partial
        annotation failure, AND the pre-existing aggregated per-stage
        warning emitted at the end of the batch by `run_pipeline()`'s own
        aggregation loop. Neither replaces the other.
        """
        _write_extraction_jsons(tmp_path, ["doc-good", "doc-bad"])

        async def _annotation_side_effect(document_name, **kwargs):
            if document_name == "doc-bad":
                return _sr(
                    "annotation",
                    success=False,
                    documents=[DocumentResult("doc-bad", 0, 1, ["boom"])],
                    processed=0,
                    failed=1,
                    errors=["boom"],
                )
            return _sr(
                "annotation",
                success=True,
                documents=[DocumentResult(document_name, 5, 0)],
                processed=5,
                failed=0,
            )

        async def _entity_extraction_side_effect(document_name, **kwargs):
            return _sr(
                "entity_extraction",
                success=True,
                documents=[DocumentResult(document_name, 5, 0)],
                processed=5,
                failed=0,
            )

        mock_stages["run_annotation"].side_effect = _annotation_side_effect
        mock_stages["run_entity_extraction"].side_effect = _entity_extraction_side_effect

        with caplog.at_level(logging.WARNING, logger="scinr.newton.pipeline"):
            result = await run_pipeline(
                stages=["extraction", "ingestion", "annotation", "entity_extraction"],
                extraction_input_dir=str(tmp_path),
                on_partial_failure="warn",
            )

        assert mock_stages["run_entity_extraction"].call_count == 2
        ee_call_names = {
            call.args[0] for call in mock_stages["run_entity_extraction"].call_args_list
        }
        assert ee_call_names == {"doc-good", "doc-bad"}
        assert result.entity_extraction is not None
        assert result.entity_extraction.success is True

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        # Aggregated per-stage warning (pre-existing, emitted by run_pipeline()
        # at the end of the batch) — must remain intact.
        assert any("completed with failures" in m for m in warning_messages)
        assert any("annotation" in m for m in warning_messages)
        # New immediate per-document warning (emitted by
        # _process_document_unit() itself at the moment doc-bad decides to
        # keep advancing) — must coexist alongside the aggregated one, not
        # replace it.
        assert any(
            "doc-bad" in m and "boom" in m and "continuing" in m.lower()
            for m in warning_messages
        )
        # Exactly two warnings total: one per-document + one aggregated —
        # neither drowns out nor duplicates the other.
        assert len(warning_messages) == 2


# ---------------------------------------------------------------------------
# StageResult aggregation via asyncio.gather (Stage 3 / Stage 4)
# ---------------------------------------------------------------------------


class TestStageResultAggregation:
    async def test_sums_totals_and_concatenates_documents_across_docs(
        self, mock_stages, mock_infra
    ):
        """Multiple documents processed concurrently in Stage 3 (annotation)
        must be aggregated into ONE synthetic StageResult: total_processed and
        total_failed are the sum across per-document results, duration_seconds
        is summed, and `documents` is the concatenation (order-independent set
        equality — asyncio.gather order is deterministic for the input list,
        but we assert on the set/sum, not on list order, to stay robust).

        `document_names=[...]` now resolves through `_discover_units()` ->
        `resolve_leaf_document_names()` (mocked via `mock_infra` to return
        `[name]` for each input name — i.e. every name is already a leaf, no
        fan-out), instead of being forwarded verbatim to `run_annotation()`
        as in the old batch orchestration. The per-document call still
        happens through `run_annotation()`, one call per leaf name.

        Behavior change vs. the pre-refactor pinned test: `StageResult.success`
        is now always recomputed by `run_pipeline()`'s own aggregation loop
        from `total_failed == 0` (see the per-stage aggregation block), not
        copied from whatever `success` flag each mocked per-document
        `StageResult` happened to report. doc-b reports 1 failed node, so the
        combined `success` is correctly `False` here — the old test's
        `success is True` assertion pinned what was, in retrospect, an
        aggregation bug (a per-document `success=True` flag masking that same
        document's own `nodes_failed=1`).
        """
        per_doc_results = {
            "doc-a": _sr(
                "annotation",
                success=True,
                documents=[DocumentResult("doc-a", 5, 0)],
                processed=5,
                failed=0,
                duration=1.0,
            ),
            "doc-b": _sr(
                "annotation",
                success=True,
                documents=[DocumentResult("doc-b", 3, 1)],
                processed=3,
                failed=1,
                duration=2.0,
            ),
            "doc-c": _sr(
                "annotation",
                success=True,
                documents=[DocumentResult("doc-c", 2, 0)],
                processed=2,
                failed=0,
                duration=0.5,
            ),
        }

        async def _side_effect(document_name, **kwargs):
            return per_doc_results[document_name]

        mock_stages["run_annotation"].side_effect = _side_effect

        result = await run_pipeline(
            stages=["annotation"],
            document_names=["doc-a", "doc-b", "doc-c"],
            on_partial_failure="continue",
        )

        ann = result.annotation
        assert ann is not None
        assert ann.stage == "annotation"
        assert ann.total_processed == 5 + 3 + 2
        assert ann.total_failed == 0 + 1 + 0
        assert {d.document_name for d in ann.documents} == {"doc-a", "doc-b", "doc-c"}
        assert len(ann.documents) == 3
        # total_failed == 1 (doc-b), so the combined result is correctly
        # success=False — see the docstring note above on why this differs
        # from the old pinned (buggy) expectation.
        assert ann.success is False
        assert ann.errors == []

    async def test_exception_in_one_doc_counts_as_a_failed_document(self, mock_stages, mock_infra):
        """NEW semantics (Bloque B): if `run_annotation()` raises for one
        document inside `_process_document_unit()`'s per-unit try/except, the
        crashed unit's `stage_results` dict never gets an `"annotation"` key
        assigned for it (the exception happens at the `run_annotation()`
        call itself, before any `DocumentResult` can be built) — so the
        crashed document still does NOT appear in `documents` and
        contributes ZERO to `total_processed`/`total_failed`, exactly as in
        the old pinned (surprising) behavior.

        What DID change (this is the Paso 3 aggregation fix under test): the
        crashed unit's `UnitResult.fatal_error` is no longer silently
        dropped. The per-stage aggregation loop in `run_pipeline()` now folds
        any unit's `fatal_error` into `StageResult.errors` for the first
        stage being aggregated that the unit never reached — so the error
        message IS visible again (as a stage-level error, not a per-document
        one), and `success` correctly flips to `False` because of it.
        """

        async def _side_effect(document_name, **kwargs):
            if document_name == "doc-bad":
                raise RuntimeError("annotation agent crashed")
            return _sr(
                "annotation",
                success=True,
                documents=[DocumentResult(document_name, 4, 0)],
                processed=4,
                failed=0,
                duration=1.0,
            )

        mock_stages["run_annotation"].side_effect = _side_effect

        result = await run_pipeline(
            stages=["annotation"],
            document_names=["doc-good", "doc-bad"],
            on_partial_failure="continue",
        )

        ann = result.annotation
        assert ann is not None
        assert ann.total_processed == 4
        assert ann.total_failed == 0
        assert [d.document_name for d in ann.documents] == ["doc-good"]
        assert ann.errors == ["annotation agent crashed"]
        assert ann.success is False


# ---------------------------------------------------------------------------
# PipelineResult shape
# ---------------------------------------------------------------------------


class TestPipelineResultShape:
    async def test_successful_run_populates_expected_stage_fields(
        self, mock_stages, mock_infra, mock_unit_stage_fns, tmp_path
    ):
        """A fully successful run (extraction -> ingestion -> annotation ->
        entity_extraction, preprocess skipped) returns a PipelineResult with
        success=True, stages_executed listing exactly the stages that ran (in
        execution order), each populated StageResult field matching its stage
        name, and unexecuted stage fields (preprocess, tabular) left as None.
        """
        _write_extraction_jsons(tmp_path, ["doc1"])

        mock_stages["run_annotation"].return_value = _sr(
            "annotation",
            success=True,
            documents=[DocumentResult("doc1", 5, 0)],
            processed=5,
            failed=0,
            duration=0.1,
        )
        mock_stages["run_entity_extraction"].return_value = _sr(
            "entity_extraction",
            success=True,
            documents=[DocumentResult("doc1", 5, 0)],
            processed=5,
            failed=0,
            duration=0.1,
        )

        result = await run_pipeline(
            stages=["extraction", "ingestion", "annotation", "entity_extraction"],
            extraction_input_dir=str(tmp_path),
        )

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.stages_executed == [
            "extraction",
            "ingestion",
            "annotation",
            "entity_extraction",
        ]
        assert result.preprocess is None
        assert result.tabular is None

        assert result.extraction is not None and result.extraction.stage == "extraction"
        assert result.ingestion is not None and result.ingestion.stage == "ingestion"
        assert result.annotation is not None and result.annotation.stage == "annotation"
        assert (
            result.entity_extraction is not None
            and result.entity_extraction.stage == "entity_extraction"
        )
        assert isinstance(result.total_duration_seconds, float)
        assert result.total_duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# Stage 0 (preprocess) baseline — parallel_docs plumbing as of writing this suite
# ---------------------------------------------------------------------------


class TestStage0Baseline:
    """Stage 0 (preprocess) concurrency, as of the Bloque B per-document-unit
    engine: `run_pipeline()` no longer calls the batch `run_preprocess()` /
    `convert_folder()` functions at all for Stage 0 — it discovers one
    `DocumentUnit` per raw file (`_discover_raw_file_units()`) and calls
    `convert_one()` once per unit, each call bounded by the single shared
    `document_semaphore = asyncio.Semaphore(parallel_docs)` created once in
    `run_pipeline()` and held for that unit's entire multi-stage duration.

    This is a strictly global cap across the whole discovered set (unlike
    the old per-directory-level `convert_folder()` semaphore documented in
    prior revisions of this docstring — `convert_folder()`/`run_preprocess()`
    are simply not on this call path anymore when going through
    `run_pipeline()`; they remain available as public standalone functions
    for direct use, e.g. from `cli.py`, and are still exercised directly by
    `test_stage0_reports_conversion_failures_as_failed_documents` below).
    """

    async def test_preprocess_receives_and_forwards_parallel_docs(self, tmp_path, monkeypatch):
        """`parallel_docs` bounds the shared `document_semaphore` that every
        `_process_document_unit()` task acquires for its entire duration; for
        a `stages=["preprocess"]`-only run that means at most `parallel_docs`
        calls to `convert_one()` are ever in flight at once, even with more
        raw files discovered than the cap.
        """
        import scinr.newton.converters.main as converters_main_mod
        import scinr.newton.storage.factory as storage_factory_mod

        for i in range(5):
            (tmp_path / f"doc{i}.txt").write_text(f"hello {i}", encoding="utf-8")

        monkeypatch.setattr(
            storage_factory_mod, "get_storage", MagicMock(return_value=(None, None))
        )

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def _tracking_convert_one(entry, output_dir, **kwargs):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            async with lock:
                in_flight -= 1
            return ([], [])

        monkeypatch.setattr(
            converters_main_mod, "convert_one", AsyncMock(side_effect=_tracking_convert_one)
        )

        result = await run_pipeline(
            input_raw=str(tmp_path),
            stages=["preprocess"],
            parallel_docs=2,
        )

        assert max_in_flight == 2
        assert result.preprocess is not None

    async def test_stage0_reports_conversion_failures_as_failed_documents(self, tmp_path, mock_llm):
        """Phase 7b fix: Stage 0 now correctly surfaces per-file conversion
        failures instead of silently swallowing them.

        `convert_folder()` (converters/main.py) returns `(written, failures)`,
        where `failures` is the list of `(entry_path, error_message)` pairs
        for files that raised `ConversionError` or any unexpected exception
        during conversion (unsupported-format files are deliberately excluded
        from `failures` -- see `convert_one()` -- since skipping an
        unrecognized extension is not a conversion error, it is by design).
        `run_preprocess()` (stages/preprocess.py) now unpacks this tuple and,
        for every entry in `failures`, appends a `DocumentResult` with
        `nodes_processed=0`, `nodes_failed=1`, and `errors=[error_message]`
        to the same list used for successes. The final `StageResult` now
        derives `total_failed` from `len(failures)` and `success` from
        `len(failures) == 0`, instead of being hardcoded to `total_failed=0`
        / `success=True` regardless of actual per-file outcomes.

        This was previously a known, pinned limitation (see prior revision of
        this test, `test_stage0_swallows_unsupported_format_errors_silently`);
        this test now pins the CORRECT, desired behavior instead.

        Exercises the REAL `run_preprocess()` / `convert_folder()` code path
        (no mocking of stage internals) against a `tmp_path` containing one
        valid `.txt` file (handled by the dependency-free `TextConverter`)
        alongside one `.csv` file. `CsvConverter.convert()`
        (converters/csv.py) unconditionally raises `ConversionError` for
        every `.csv` file -- by design, since CSV files are meant to be
        routed to the tabular ingestion pipeline instead -- which makes it a
        deterministic, dependency-free way to trigger a real conversion
        failure without needing a corrupted binary file.
        """
        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
        )

        (tmp_path / "good.txt").write_text("hello world", encoding="utf-8")
        (tmp_path / "bad.csv").write_text("col1,col2\nval1,val2\n", encoding="utf-8")

        stage_result, intermediate_docs = await run_preprocess(input_raw=str(tmp_path))

        # The valid file converted fine.
        assert stage_result.total_processed == 1
        assert len(intermediate_docs) == 1

        # The .csv file failed conversion (ConversionError) and is now
        # correctly surfaced as a failed document.
        assert stage_result.success is False
        assert stage_result.total_failed == 1
        assert len(stage_result.documents) == 2

        documents_by_name = {doc.document_name: doc for doc in stage_result.documents}
        assert documents_by_name["good"].nodes_failed == 0
        assert documents_by_name["good"].errors == []
        assert documents_by_name["bad"].nodes_failed == 1
        assert documents_by_name["bad"].nodes_processed == 0
        assert len(documents_by_name["bad"].errors) == 1


# ---------------------------------------------------------------------------
# Paso 3 fix: fatal_error surfacing in per-stage aggregation
# ---------------------------------------------------------------------------


class TestFatalErrorAggregation:
    async def test_fatal_error_surfaces_in_first_aggregated_stage_errors(
        self, mock_infra, monkeypatch
    ):
        """Regression test for the Paso 3 aggregation fix in `run_pipeline()`.

        `_process_document_unit()` never lets a real exception escape — its
        own catch-all wraps ANY unexpected error (a genuine bug, not
        attributable to a specific stage) into a normal `UnitResult` with
        `fatal_error` set and `stage_results={}`. Before this fix, the
        per-stage aggregation loop only looked for `isinstance(ur,
        BaseException)` to build `StageResult.errors` — since
        `_process_document_unit()` never raises, that list was always empty,
        and the unit's `fatal_error` message vanished silently: the unit
        simply disappeared from `documents` for every stage with no trace in
        `errors` either.

        This test crafts that exact scenario directly at the `UnitResult`
        boundary (patching `_process_document_unit()` itself, per the task
        spec) to confirm the fix: the `fatal_error` message now appears in
        `StageResult.errors` for the first (only, here) requested stage that
        the crashed unit never reached, while the surviving sibling's
        `DocumentResult` is unaffected.
        """
        import scinr.newton.pipeline as pipeline_mod
        from scinr.newton.pipeline_units import UnitResult

        async def _fake_process_unit(unit, **kwargs):
            if unit.document_name_hint == "doc-crash":
                return UnitResult("doc-crash", {}, None, "boom: unexpected TypeError")
            return UnitResult(
                unit.document_name_hint,
                {"annotation": DocumentResult(unit.document_name_hint, 3, 0, [])},
                None,
                None,
            )

        monkeypatch.setattr(pipeline_mod, "_process_document_unit", _fake_process_unit)

        result = await run_pipeline(
            stages=["annotation"],
            document_names=["doc-good", "doc-crash"],
        )

        ann = result.annotation
        assert ann is not None
        assert ann.errors == ["boom: unexpected TypeError"]
        assert [d.document_name for d in ann.documents] == ["doc-good"]
        assert ann.total_processed == 3
        assert ann.total_failed == 0
        assert ann.success is False

    async def test_fatal_error_duplicated_across_all_unreached_requested_stages(
        self, mock_infra, tmp_path, monkeypatch
    ):
        """Regression test for reviewer point 3: when a unit's `fatal_error`
        is set and MULTIPLE stages were requested together
        (`["extraction", "ingestion", "annotation"]`), none of which the
        crashed unit ever reached (`stage_results={}`), the chosen behavior
        (documented in a comment at the aggregation site in `pipeline.py`)
        is to fold the SAME `fatal_error` message into `StageResult.errors`
        for EVERY one of those requested stages — not deduplicate it into
        only the first one — because each stage's own aggregate is
        independently missing this unit and must independently report
        `success=False` because of it.

        This test pins that exact multi-stage duplication explicitly, so any
        future change to deduplicate it (an equally defensible alternative
        design) must update this test deliberately rather than regress it
        silently.
        """
        import scinr.newton.pipeline as pipeline_mod
        from scinr.newton.pipeline_units import UnitResult

        _write_extraction_jsons(tmp_path, ["doc-good", "doc-crash"])

        async def _fake_process_unit(unit, **kwargs):
            if unit.document_name_hint == "doc-crash":
                # Simulates an exception not attributable to any single
                # concrete stage (e.g. raised before any stage-specific
                # try/except in _process_document_unit() could run) —
                # stage_results is empty for every requested stage.
                return UnitResult("doc-crash", {}, None, "boom: unattributable crash")
            return UnitResult(
                unit.document_name_hint,
                {
                    "extraction": DocumentResult(unit.document_name_hint, 1, 0, []),
                    "ingestion": DocumentResult(unit.document_name_hint, 1, 0, []),
                    "annotation": DocumentResult(unit.document_name_hint, 3, 0, []),
                },
                None,
                None,
            )

        monkeypatch.setattr(pipeline_mod, "_process_document_unit", _fake_process_unit)

        result = await run_pipeline(
            stages=["extraction", "ingestion", "annotation"],
            extraction_input_dir=str(tmp_path),
        )

        for stage_name in ("extraction", "ingestion", "annotation"):
            sr = getattr(result, stage_name)
            assert sr is not None, stage_name
            assert sr.errors == ["boom: unattributable crash"], stage_name
            assert [d.document_name for d in sr.documents] == ["doc-good"], stage_name
            assert sr.success is False, stage_name


# ---------------------------------------------------------------------------
# Catalog / theme-structure pre-warm memoization
# ---------------------------------------------------------------------------


class TestCatalogMemoization:
    async def test_ensure_catalog_and_theme_setup_run_at_most_once_per_pipeline_run(
        self, mock_stages, monkeypatch
    ):
        """`ensure_catalog_models_once()` / `ensure_theme_structure_once()`
        (the Stage 3/4 pre-warm step run once before the per-document
        fan-out in `run_pipeline()`) must invoke the real, NON-memoized
        `ensure_catalog_models()` / `ensure_theme_structure()` at most once
        each per `run_pipeline()` call, even with several documents
        processed concurrently — that is the entire point of the
        process-level memoization guard in `annotation/neo4j_ops.py`.

        Unlike `mock_infra` above (which replaces the `_once` wrappers
        themselves with no-op mocks, for tests that don't care about this
        guard), this test leaves the real memoized wrappers in place and
        mocks only the inner, non-memoized functions they guard — so the
        real check-lock-check memoization logic is what's under test here.
        `reset_catalog_memoization()` is called before (and after, for test
        isolation) since the guard is process-level global state, not reset
        by the `clean_config` autouse fixture.
        """
        import scinr.newton.annotation.neo4j_ops as neo4j_ops_mod
        import scinr.newton.ingest.config as ingest_config_mod
        import scinr.newton.utils.document_resolver as document_resolver_mod
        import scinr.newton.utils.theme_registry as theme_registry_mod

        neo4j_ops_mod.reset_catalog_memoization()
        try:
            fake_ensure_catalog_models = AsyncMock()
            fake_ensure_theme_structure = AsyncMock()
            monkeypatch.setattr(neo4j_ops_mod, "ensure_catalog_models", fake_ensure_catalog_models)
            monkeypatch.setattr(
                neo4j_ops_mod, "ensure_theme_structure", fake_ensure_theme_structure
            )
            monkeypatch.setattr(
                ingest_config_mod, "get_driver", MagicMock(return_value=MagicMock())
            )
            monkeypatch.setattr(
                ingest_config_mod, "get_async_driver", MagicMock(return_value=MagicMock())
            )
            monkeypatch.setattr(
                theme_registry_mod, "get_theme_registry", MagicMock(return_value=MagicMock())
            )
            monkeypatch.setattr(
                document_resolver_mod,
                "resolve_leaf_document_names",
                MagicMock(side_effect=lambda driver, name: [name]),
            )

            mock_stages["run_annotation"].return_value = _sr(
                "annotation",
                success=True,
                documents=[DocumentResult("x", 1, 0)],
                processed=1,
                failed=0,
            )

            result = await run_pipeline(
                stages=["annotation"],
                document_names=["doc-a", "doc-b", "doc-c", "doc-d"],
            )

            assert fake_ensure_catalog_models.call_count <= 1
            assert fake_ensure_theme_structure.call_count <= 1
            assert result.annotation is not None
        finally:
            neo4j_ops_mod.reset_catalog_memoization()


# ---------------------------------------------------------------------------
# Event-loop-blocking fix regression test — annotation/entity_extraction
# concurrency across documents through the full run_pipeline() call.
# ---------------------------------------------------------------------------


class _FakeAsyncResult:
    """Mimics neo4j.AsyncResult — always answers `n=1`, which is enough for
    every precondition-check query issued by `run_annotation_agent()` /
    `run_entity_extraction_agent()` (document exists / has annotated nodes)
    to pass regardless of the exact query text.
    """

    async def single(self):
        return {"n": 1}


class _FakeAsyncSession:
    """Mimics neo4j.AsyncSession as an async context manager."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, query, **kwargs):
        return _FakeAsyncResult()


class _FakeAsyncDriver:
    """Mimics neo4j.AsyncDriver — `.session()` is a plain (non-async) method
    returning a fresh session each call.
    """

    def session(self):
        return _FakeAsyncSession()


class TestAnnotationEntityExtractionConcurrency:
    """End-to-end regression test (through the real, unmocked
    `run_annotation()` / `run_entity_extraction()` from `stages.py`, and the
    real `run_annotation_agent()` / `run_entity_extraction_agent()` they
    delegate to) proving that `run_pipeline(stages=["annotation",
    "entity_extraction"], ...)` genuinely processes multiple documents
    concurrently rather than serializing them — i.e. the event-loop-blocking
    fix holds at the full pipeline-orchestration level, not just in the
    lower-level unit tests in `test_annotation_agent.py` /
    `test_entity_extraction_agent.py`.

    Deliberately does NOT reuse the `mock_infra` fixture: `mock_infra`'s
    `fake_async_driver` is a plain `MagicMock()`, which cannot be used as an
    async context manager (`async with driver.session(database=cfg.neo4j_database) as _session:` would
    raise) — that has never mattered for the *other* tests in this file
    because they all patch `run_annotation`/`run_entity_extraction` at the
    `stages.py` level via `mock_stages`, so the real agent code (and its
    real `async with driver.session(database=cfg.neo4j_database)` precondition checks) is never
    reached. This test intentionally leaves `run_annotation`/
    `run_entity_extraction` REAL (per `TestCatalogMemoization`'s mocking
    philosophy above) to exercise that exact code path, so it needs its own
    async-context-manager-capable fake driver (`_FakeAsyncDriver` above)
    instead. Also mocks `resolve_leaf_document_names_async` directly (the
    new async helper `mock_infra` does not know about — it only patches the
    original *sync* `resolve_leaf_document_names`, which is still used
    as-is by `_discover_pre_ingested_units()` for the `document_names`
    discovery branch, and is patched separately below for that purpose).
    """

    async def test_multiple_documents_process_annotation_and_entity_extraction_concurrently(
        self, monkeypatch
    ):
        """`parallel_docs` bounds how many `DocumentUnit`s run concurrently;
        for a `document_names`-driven `stages=["annotation",
        "entity_extraction"]` run, that means up to `parallel_docs` documents
        can be inside `fetch_nodes_to_annotate()` / `fetch_extraction_targets()`
        at the same time. Tracked via one shared `in_flight`/`max_in_flight`
        counter (a given document is only ever inside one of the two fetch
        calls at a time, but different documents may be inside *either* one
        concurrently — both count towards the same cross-document
        concurrency guarantee this test protects).
        """
        import scinr.newton.annotation.neo4j_ops as annotation_neo4j_ops_mod
        import scinr.newton.entity_extraction.neo4j_ops as entity_extraction_neo4j_ops_mod
        import scinr.newton.ingest.config as ingest_config_mod
        import scinr.newton.utils.document_resolver as document_resolver_mod
        import scinr.newton.utils.theme_registry as theme_registry_mod

        document_names = ["doc-a", "doc-b", "doc-c", "doc-d"]
        parallel_docs = 3

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def _tracked_sleep() -> None:
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            async with lock:
                in_flight -= 1

        async def _tracking_fetch_nodes(driver, document_name, only_unannotated=False):
            await _tracked_sleep()
            return []

        async def _tracking_fetch_targets(driver, document_name, only_unextracted=False):
            await _tracked_sleep()
            return []

        fake_async_driver = _FakeAsyncDriver()
        fake_sync_driver = MagicMock(name="sync_driver")

        # Sync resolver used by `_discover_pre_ingested_units()` (unchanged,
        # untouched code path) — identity: every name is already a leaf.
        monkeypatch.setattr(
            document_resolver_mod,
            "resolve_leaf_document_names",
            MagicMock(side_effect=lambda driver, name: [name]),
        )
        # New async resolver used by `run_annotation_agent()` /
        # `run_entity_extraction_agent()` — same identity semantics.
        monkeypatch.setattr(
            document_resolver_mod,
            "resolve_leaf_document_names_async",
            AsyncMock(side_effect=lambda driver, name: [name]),
        )
        monkeypatch.setattr(
            ingest_config_mod, "get_driver", MagicMock(return_value=fake_sync_driver)
        )
        monkeypatch.setattr(
            ingest_config_mod, "get_async_driver", MagicMock(return_value=fake_async_driver)
        )
        monkeypatch.setattr(
            annotation_neo4j_ops_mod, "ensure_catalog_models_once", AsyncMock()
        )
        monkeypatch.setattr(
            annotation_neo4j_ops_mod, "ensure_theme_structure_once", AsyncMock()
        )
        monkeypatch.setattr(
            annotation_neo4j_ops_mod,
            "fetch_nodes_to_annotate",
            AsyncMock(side_effect=_tracking_fetch_nodes),
        )
        monkeypatch.setattr(
            annotation_neo4j_ops_mod,
            "fetch_document_context_instructions",
            AsyncMock(return_value=""),
        )
        monkeypatch.setattr(
            entity_extraction_neo4j_ops_mod,
            "fetch_extraction_targets",
            AsyncMock(side_effect=_tracking_fetch_targets),
        )
        monkeypatch.setattr(
            theme_registry_mod, "get_theme_registry", MagicMock(return_value=MagicMock())
        )

        result = await run_pipeline(
            stages=["annotation", "entity_extraction"],
            document_names=document_names,
            parallel_docs=parallel_docs,
        )

        assert max_in_flight == parallel_docs
        assert result.annotation is not None
        assert result.entity_extraction is not None
        assert result.annotation.total_failed == 0
        assert result.entity_extraction.total_failed == 0


# ---------------------------------------------------------------------------
# Placeholder for Phase 3 of the refactor (batch vs. async-per-document parity)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "OBSOLETE placeholder, kept only for history. Bloque B (Pasos 1-3) has "
        "now landed: run_pipeline() was rewritten in place to use the "
        "per-document-unit engine (_discover_units() / _process_document_unit() "
        "in pipeline.py / pipeline_units.py) — it does not exist ALONGSIDE the "
        "old batch orchestration (run_extraction()/run_ingestion() called "
        "per-document-name-list via asyncio.gather()) as a separate code path; "
        "the old path was replaced, not duplicated. There is therefore nothing "
        "left to compare it against — a literal 'batch vs. per-document-task "
        "parity' test as originally envisioned is no longer possible to write. "
        "TestStageResultAggregation, TestOnPartialFailureAbort/Continue/Warn, "
        "TestFatalErrorAggregation, and TestCatalogMemoization above now cover "
        "the new engine's aggregation semantics directly instead."
    )
)
def test_batch_version_parity_placeholder():
    """OBSOLETE(Bloque B Paso 3 complete): batch orchestration no longer exists
    as a separate path to compare against; see skip reason above."""
    raise NotImplementedError
