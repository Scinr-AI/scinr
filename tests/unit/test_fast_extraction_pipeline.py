"""
tests/unit/test_fast_extraction_pipeline.py — Threading/validation tests for
``fast_extraction`` in scinr.newton.pipeline (``run_pipeline()`` /
``_process_document_unit()``).

Mocking strategy
-----------------
Mirrors tests/unit/test_pipeline_orchestration.py and
tests/unit/test_process_document_unit.py: ``_process_document_unit()``'s
per-document primitives (``extract_one_intermediate`` / ``extract_one_file``)
are imported via a deferred ``from scinr.newton.stages.extraction import ...``
executed at call time, so monkeypatching the attribute on
``scinr.newton.stages.extraction`` (the ORIGIN module) — not on
``scinr.newton.pipeline`` — is what gets picked up on every call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scinr.newton.pipeline import _process_document_unit, run_pipeline
from scinr.newton.pipeline_units import DocumentUnit


def _base_kwargs(**overrides: object) -> dict:
    """Mirrors tests/unit/test_process_document_unit.py's helper of the same
    name — keyword-only arguments of _process_document_unit() with sane
    defaults, overridable per test."""
    kwargs = dict(
        effective_stages=["extraction"],
        document_semaphore=asyncio.Semaphore(2),
        sync_driver=None,
        shared_ingest_version=None,
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


# ---------------------------------------------------------------------------
# 1. run_pipeline() validation — fast_extraction requires "extraction" stage
# ---------------------------------------------------------------------------


class TestFastExtractionValidation:
    async def test_fast_extraction_without_extraction_stage_raises_value_error(self):
        """fast_extraction=True with 'extraction' absent from stages must
        raise ValueError with a clear message, before any real work (no
        discovery, no Neo4j, no LLM) happens. 'ingestion' with an explicit
        ingestion_input_dir is chosen here specifically because it passes
        every OTHER validation gate in run_pipeline() (no document_names
        requirement, no missing-input-source error) so the ValueError
        raised is unambiguously attributable to the fast_extraction check.
        """
        with pytest.raises(ValueError, match="fast_extraction"):
            await run_pipeline(
                stages=["ingestion"],
                ingestion_input_dir="/tmp/does-not-matter-for-this-check",
                fast_extraction=True,
            )

    async def test_fast_extraction_with_extraction_stage_does_not_raise_for_this_reason(
        self, tmp_path: Path, mock_llm
    ):
        """fast_extraction=True with 'extraction' present in stages must NOT
        raise the fast_extraction ValueError. An empty input_raw directory
        with stages=["preprocess", "extraction"] discovers zero units (no
        files to process) and completes successfully — proving the run
        never even reaches the point of raising, let alone for this reason.

        configure() is required here only because "preprocess" is in stages
        (run_pipeline() calls get_storage() -> get_config() unconditionally
        for that stage) — unrelated to the fast_extraction check itself.
        """
        from scinr.newton.config import configure

        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
        )

        result = await run_pipeline(
            stages=["preprocess", "extraction"],
            input_raw=str(tmp_path),
            fast_extraction=True,
        )
        assert result.success is True
        assert result.stages_executed == ["preprocess", "extraction"]


# ---------------------------------------------------------------------------
# 2. fast_extraction reaches extract_one_intermediate/extract_one_file as an
#    explicit value, all the way from run_pipeline()
# ---------------------------------------------------------------------------


def _write_extraction_json(path: Path, name: str) -> None:
    (path / f"{name}.json").write_text(
        json.dumps({"folder_path": None, "pages": []}), encoding="utf-8"
    )


class TestFastExtractionReachesExtractOneFile:
    async def test_fast_extraction_true_reaches_extract_one_file_explicitly(
        self, tmp_path: Path, monkeypatch
    ):
        _write_extraction_json(tmp_path, "doc1")

        import scinr.newton.stages.extraction as extraction_mod

        async def _fake_extract_one_file(json_file, output_path, input_folder, fast_extraction=False):
            doc = SimpleNamespace(document_name=json_file.stem)
            return doc

        mock_extract_one_file = AsyncMock(side_effect=_fake_extract_one_file)
        monkeypatch.setattr(extraction_mod, "extract_one_file", mock_extract_one_file)

        result = await run_pipeline(
            stages=["extraction"],
            extraction_input_dir=str(tmp_path),
            fast_extraction=True,
        )

        assert result.success is True
        mock_extract_one_file.assert_called_once()
        _, call_kwargs = mock_extract_one_file.call_args
        assert call_kwargs["fast_extraction"] is True

    async def test_fast_extraction_default_false_reaches_extract_one_file_explicitly(
        self, tmp_path: Path, monkeypatch
    ):
        _write_extraction_json(tmp_path, "doc1")

        import scinr.newton.stages.extraction as extraction_mod

        async def _fake_extract_one_file(json_file, output_path, input_folder, fast_extraction=False):
            return SimpleNamespace(document_name=json_file.stem)

        mock_extract_one_file = AsyncMock(side_effect=_fake_extract_one_file)
        monkeypatch.setattr(extraction_mod, "extract_one_file", mock_extract_one_file)

        result = await run_pipeline(
            stages=["extraction"],
            extraction_input_dir=str(tmp_path),
            # fast_extraction omitted entirely — must default to False and
            # still be passed explicitly.
        )

        assert result.success is True
        mock_extract_one_file.assert_called_once()
        _, call_kwargs = mock_extract_one_file.call_args
        assert call_kwargs["fast_extraction"] is False


# ---------------------------------------------------------------------------
# 3. Concurrency-safety: two concurrent _process_document_unit() calls with
#    DIFFERENT fast_extraction values must not interfere with each other.
# ---------------------------------------------------------------------------


class _ConcurrencyTracker:
    """Mirrors tests/unit/test_annotation_agent.py's ``_ConcurrencyTracker``."""

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


class TestConcurrentProcessDocumentUnitFastExtractionIsolation:
    async def test_two_concurrent_units_each_receive_their_own_fast_extraction_value(
        self, monkeypatch
    ):
        """This is the concurrency-safety property the explicit
        fast_extraction-threading design exists to guarantee: fast_extraction
        is a plain function parameter passed all the way down (never read
        from global config), so two _process_document_unit() calls running
        genuinely concurrently (forced to overlap via a shared tracker with
        an internal await point) must each observe only their OWN value —
        never the other unit's.
        """
        import scinr.newton.stages.extraction as extraction_mod

        tracker = _ConcurrencyTracker(sleep_seconds=0.03)
        captured: list[tuple[str, bool]] = []

        async def _fake_extract_one_intermediate(doc, output_path, fast_extraction=False):
            await tracker.track()
            captured.append(("unitA", fast_extraction))
            return SimpleNamespace(document_name="unitA")

        async def _fake_extract_one_file(json_file, output_path, input_folder, fast_extraction=False):
            await tracker.track()
            captured.append(("unitB", fast_extraction))
            return SimpleNamespace(document_name="unitB")

        monkeypatch.setattr(
            extraction_mod,
            "extract_one_intermediate",
            AsyncMock(side_effect=_fake_extract_one_intermediate),
        )
        monkeypatch.setattr(
            extraction_mod,
            "extract_one_file",
            AsyncMock(side_effect=_fake_extract_one_file),
        )

        unit_a = DocumentUnit(
            kind="raw_file",
            source_path=Path("/tmp/unitA-does-not-matter.pdf"),
            doc_path="unitA",
            relative_dir=Path("."),
            document_name_hint="unitA",
        )
        unit_b = DocumentUnit(
            kind="extraction_json",
            source_path=Path("/tmp/unitB-does-not-matter.json"),
            doc_path="unitB",
            relative_dir=Path("."),
            document_name_hint="unitB",
        )

        shared_semaphore = asyncio.Semaphore(2)

        result_a, result_b = await asyncio.gather(
            _process_document_unit(
                unit_a,
                **_base_kwargs(
                    document_semaphore=shared_semaphore, fast_extraction=True
                ),
            ),
            _process_document_unit(
                unit_b,
                **_base_kwargs(
                    document_semaphore=shared_semaphore, fast_extraction=False
                ),
            ),
        )

        # Both calls actually overlapped in time (genuine concurrency, not
        # accidental sequential execution that would make this test
        # meaningless).
        assert tracker.max_in_flight == 2

        assert dict(captured) == {"unitA": True, "unitB": False}
        assert result_a.stage_results["extraction"].document_name == "unitA"
        assert result_b.stage_results["extraction"].document_name == "unitB"
        assert result_a.stopped_at is None
        assert result_b.stopped_at is None
