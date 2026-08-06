"""
tests/unit/test_extraction_page_id_gaps.py — Regression tests for the
position-vs-absolute-index bug in ``extract_one_intermediate()`` and
``extract_one_file()`` (stages/extraction.py).

Context
-------
In ``mistral_ocr_error_strategy="best_effort"`` mode, ``PdfConverter``
may skip a failed chunk of pages, leaving gaps in the absolute
``IntermediatePage.index`` sequence (e.g. ``[0, 1, 4, 5]`` if the
original pages 2 and 3 were omitted). ``doc.pages`` (or the ``pages``
list read from the Stage 0 JSON) is still a plain list with *positions*
``0..len-1``, so position no longer equals absolute index once there
are gaps.

Both ``extract_one_intermediate()`` and ``extract_one_file()`` build a
``page_ids_by_index`` dict keyed by the *absolute* index, then must
translate each chunk's *position* in the pages list to the correct
absolute index before looking up the page_id. These tests pin that
translation by capturing the ``curr_page_ids`` argument passed into
each mocked ``extract_chunk()`` call and asserting it matches the
page_id of the page that actually occupies that position (by absolute
index), not the page_id that happens to share the same numeric key as
the position.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scinr.newton.config import configure
from scinr.newton.converters.base import IntermediateDocument, IntermediatePage
from scinr.newton.stages.extraction import extract_one_file, extract_one_intermediate


@pytest.fixture(autouse=True)
def _configure_min(mock_llm):
    """Minimal configure() so get_config()/get_llm()/get_llm_semaphore() work.

    extraction_batch_size=1 forces one page per chunk, so each chunk's
    curr_start_idx equals its position in the pages list — the simplest
    setup to demonstrate the position-vs-absolute-index translation.
    """
    configure(
        llm=mock_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        extraction_batch_size=1,
    )


def _capturing_extract_chunk(calls: list):
    """Return an AsyncMock-compatible side_effect that records curr_page_ids
    and returns an empty node list (compact_extraction() tolerates that)."""

    async def _fn(*, prev_page, curr_pages, active_hierarchy, llm, curr_page_ids, user_context):
        calls.append(curr_page_ids)
        return []

    return _fn


class TestExtractOneIntermediatePageIdGaps:
    async def test_gaps_resolve_to_correct_page_id_by_absolute_index(self, mocker):
        """Pages with a deliberate gap in `index` (0, 1, 4, 5 — as if original
        pages 2 and 3 were skipped in best_effort mode) must still map each
        chunk's curr_page_ids to the page_id of the page with that ABSOLUTE
        index, not the page_id that happens to sit at the same list position.
        """
        doc = IntermediateDocument(
            pages=[
                IntermediatePage(index=0, markdown="p0", page_id="pid-0"),
                IntermediatePage(index=1, markdown="p1", page_id="pid-1"),
                IntermediatePage(index=4, markdown="p4", page_id="pid-4"),
                IntermediatePage(index=5, markdown="p5", page_id="pid-5"),
            ],
            document_name="doc-with-gaps",
        )

        calls: list = []
        mocker.patch(
            "scinr.newton.stages.extraction.extract_chunk",
            AsyncMock(side_effect=_capturing_extract_chunk(calls)),
        )

        result = await extract_one_intermediate(doc, output_path=None)

        assert result is not None
        # 4 pages, batch_size=1 -> 4 chunks, one page each.
        assert calls == [["pid-0"], ["pid-1"], ["pid-4"], ["pid-5"]]

    async def test_no_gaps_no_regression(self, mocker):
        """Continuous indices 0..N-1 (the common case) must behave exactly
        as before this fix: position == absolute index everywhere."""
        doc = IntermediateDocument(
            pages=[
                IntermediatePage(index=0, markdown="p0", page_id="pid-0"),
                IntermediatePage(index=1, markdown="p1", page_id="pid-1"),
                IntermediatePage(index=2, markdown="p2", page_id="pid-2"),
                IntermediatePage(index=3, markdown="p3", page_id="pid-3"),
            ],
            document_name="doc-continuous",
        )

        calls: list = []
        mocker.patch(
            "scinr.newton.stages.extraction.extract_chunk",
            AsyncMock(side_effect=_capturing_extract_chunk(calls)),
        )

        result = await extract_one_intermediate(doc, output_path=None)

        assert result is not None
        assert calls == [["pid-0"], ["pid-1"], ["pid-2"], ["pid-3"]]


class TestExtractOneFilePageIdGaps:
    async def test_gaps_resolve_to_correct_page_id_by_absolute_index(
        self, tmp_path: Path, mocker
    ):
        """Same scenario as the in-memory test, but reading pages (with a
        gap in `index`) from a Stage 0 JSON file on disk."""
        raw = {
            "pages": [
                {"index": 0, "markdown": "p0", "page_id": "pid-0"},
                {"index": 1, "markdown": "p1", "page_id": "pid-1"},
                {"index": 4, "markdown": "p4", "page_id": "pid-4"},
                {"index": 5, "markdown": "p5", "page_id": "pid-5"},
            ],
            "folder_path": None,
            "context_instructions": None,
            "raw_file_id": None,
        }
        json_file = tmp_path / "doc-with-gaps.json"
        json_file.write_text(json.dumps(raw), encoding="utf-8")

        calls: list = []
        mocker.patch(
            "scinr.newton.stages.extraction.extract_chunk",
            AsyncMock(side_effect=_capturing_extract_chunk(calls)),
        )

        result = await extract_one_file(json_file, output_path=None, input_folder=None)

        assert result is not None
        assert calls == [["pid-0"], ["pid-1"], ["pid-4"], ["pid-5"]]

    async def test_no_gaps_no_regression(self, tmp_path: Path, mocker):
        """Continuous indices 0..N-1 read from disk must behave exactly as
        before this fix."""
        raw = {
            "pages": [
                {"index": 0, "markdown": "p0", "page_id": "pid-0"},
                {"index": 1, "markdown": "p1", "page_id": "pid-1"},
                {"index": 2, "markdown": "p2", "page_id": "pid-2"},
                {"index": 3, "markdown": "p3", "page_id": "pid-3"},
            ],
            "folder_path": None,
            "context_instructions": None,
            "raw_file_id": None,
        }
        json_file = tmp_path / "doc-continuous.json"
        json_file.write_text(json.dumps(raw), encoding="utf-8")

        calls: list = []
        mocker.patch(
            "scinr.newton.stages.extraction.extract_chunk",
            AsyncMock(side_effect=_capturing_extract_chunk(calls)),
        )

        result = await extract_one_file(json_file, output_path=None, input_folder=None)

        assert result is not None
        assert calls == [["pid-0"], ["pid-1"], ["pid-2"], ["pid-3"]]
