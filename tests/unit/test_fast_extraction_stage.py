"""
tests/unit/test_fast_extraction_stage.py — Dual-path integration tests for
``fast_extraction`` in scinr.newton.stages.extraction
(``extract_one_intermediate()`` / ``extract_one_file()``).

Mocking strategy
-----------------
Mirrors tests/unit/test_extraction_page_id_gaps.py: ``extract_chunk`` is
patched at the boundary via
``mocker.patch("scinr.newton.stages.extraction.extract_chunk", ...)``. The
SAME fixture data (keyed by page content) is used for both
``fast_extraction=True`` and ``fast_extraction=False`` so both branches are
exercised against an identical chunk-extraction result.

To verify which code path actually ran, every collaborator specific to one
branch or the other (``compact_extraction``, ``get_active_hierarchy`` for the
legacy path; ``namespace_node_ids``, ``consolidate_structure``,
``assemble_tree``, ``write_map_checkpoint``, ``delete_map_checkpoint`` for the
fast path) is patched on ``scinr.newton.stages.extraction`` with a
``wraps=<real implementation>`` spy — so real behavior is preserved (the
document is still correctly assembled) while call counts are observable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import scinr.newton.extraction.compact_extraction as compact_extraction_mod
import scinr.newton.extraction.structure_consolidation as consolidation_mod
from scinr.newton.config import configure
from scinr.newton.converters.base import IntermediateDocument, IntermediatePage
from scinr.newton.extraction.extraction import ExtractionMaxRetriesError
from scinr.newton.models.document_structure import NodeRole, StructureNode
from scinr.newton.stages.extraction import extract_one_intermediate


@pytest.fixture(autouse=True)
def _configure_min(mock_llm):
    """extraction_batch_size=1 -> one page per chunk (2 pages -> 2 chunks)."""
    configure(
        llm=mock_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        extraction_batch_size=1,
    )


def _make_chunk_nodes(which: str) -> list[StructureNode]:
    """Fixed, deterministic per-chunk fixture data keyed by page content.

    Both chunks return only SECTION top-level nodes (no orphan roles) so
    the fast path's consolidate_structure() short-circuits without needing
    a real LLM call, keeping this test scoped to the dual-path plumbing
    rather than consolidation logic (already covered in
    tests/unit/test_structure_consolidation.py).
    """
    if which == "PAGE0":
        return [
            StructureNode(
                node_id="1_intro",
                role=NodeRole.SECTION,
                title="Intro",
                appearance_order=1,
                children=[],
            )
        ]
    if which == "PAGE1":
        child = StructureNode(
            node_id="2_body_1",
            role=NodeRole.SUBSECTION,
            title="Detail",
            appearance_order=1,
            children=[],
        )
        return [
            StructureNode(
                node_id="2_body",
                role=NodeRole.SECTION,
                title="Body",
                appearance_order=1,
                children=[child],
            )
        ]
    raise AssertionError(f"Unexpected chunk content: {which!r}")


def _make_doc() -> IntermediateDocument:
    return IntermediateDocument(
        pages=[
            IntermediatePage(index=0, markdown="PAGE0", page_id="pid-0"),
            IntermediatePage(index=1, markdown="PAGE1", page_id="pid-1"),
        ],
        document_name="dualpath-doc",
    )


async def _fake_extract_chunk(
    *,
    prev_page,
    curr_pages,
    active_hierarchy,
    llm,
    curr_page_ids=None,
    user_context="",
    defer_hierarchy=False,
):
    return [n.model_copy(deep=True) for n in _make_chunk_nodes(curr_pages[0])]


def _patch_spies(mocker):
    """Patch every branch-specific collaborator on
    scinr.newton.stages.extraction with a wraps=<real impl> spy, returning a
    dict of the resulting mocks keyed by name.
    """
    spies = {
        "compact_extraction": mocker.patch(
            "scinr.newton.stages.extraction.compact_extraction",
            MagicMock(wraps=compact_extraction_mod.compact_extraction),
        ),
        "get_active_hierarchy": mocker.patch(
            "scinr.newton.stages.extraction.get_active_hierarchy",
            MagicMock(wraps=compact_extraction_mod.get_active_hierarchy),
        ),
        "namespace_node_ids": mocker.patch(
            "scinr.newton.stages.extraction.namespace_node_ids",
            MagicMock(wraps=consolidation_mod.namespace_node_ids),
        ),
        "consolidate_structure": mocker.patch(
            "scinr.newton.stages.extraction.consolidate_structure",
            AsyncMock(wraps=consolidation_mod.consolidate_structure),
        ),
        "assemble_tree": mocker.patch(
            "scinr.newton.stages.extraction.assemble_tree",
            MagicMock(wraps=consolidation_mod.assemble_tree),
        ),
        "write_map_checkpoint": mocker.patch(
            "scinr.newton.stages.extraction.write_map_checkpoint",
            MagicMock(wraps=consolidation_mod.write_map_checkpoint),
        ),
        "delete_map_checkpoint": mocker.patch(
            "scinr.newton.stages.extraction.delete_map_checkpoint",
            MagicMock(wraps=consolidation_mod.delete_map_checkpoint),
        ),
    }
    return spies


class TestDualPathExtraction:
    @pytest.mark.parametrize("fast_extraction", [False, True])
    async def test_dual_path_produces_correct_structure_and_exercises_right_collaborators(
        self, mocker, tmp_path: Path, fast_extraction: bool
    ):
        spies = _patch_spies(mocker)
        mocker.patch(
            "scinr.newton.stages.extraction.extract_chunk",
            AsyncMock(side_effect=_fake_extract_chunk),
        )

        doc = _make_doc()
        result = await extract_one_intermediate(
            doc, output_path=tmp_path, fast_extraction=fast_extraction
        )

        assert result is not None
        # Both paths must produce the same shape given non-orphan-producing
        # fixture data: two root SECTION nodes, "Body" with local child
        # "Detail" preserved. (node_id is intentionally NOT compared here —
        # fast_extraction=True namespaces node_id by page prefix, a
        # documented, intentional divergence from the legacy path.)
        titles = [n.title for n in result.document_structure]
        assert titles == ["Intro", "Body"]
        body_node = result.document_structure[1]
        assert [c.title for c in body_node.children] == ["Detail"]

        if not fast_extraction:
            # Legacy path: get_active_hierarchy/compact_extraction exercised
            # once per chunk; none of the new fast_extraction machinery is
            # ever touched.
            assert spies["compact_extraction"].call_count == 2
            assert spies["get_active_hierarchy"].call_count == 2
            spies["namespace_node_ids"].assert_not_called()
            spies["consolidate_structure"].assert_not_called()
            spies["assemble_tree"].assert_not_called()
            spies["write_map_checkpoint"].assert_not_called()
            spies["delete_map_checkpoint"].assert_not_called()
        else:
            # Fast path: compact_extraction/get_active_hierarchy are NEVER
            # called; the new namespacing/checkpoint/consolidation machinery
            # IS called.
            spies["compact_extraction"].assert_not_called()
            spies["get_active_hierarchy"].assert_not_called()
            assert spies["namespace_node_ids"].call_count == 2  # once per chunk.
            spies["consolidate_structure"].assert_called_once()
            spies["assemble_tree"].assert_called_once()
            spies["write_map_checkpoint"].assert_called_once()
            spies["delete_map_checkpoint"].assert_called_once()


class TestFastExtractionChunkFailureIsolation:
    async def test_one_chunk_extraction_max_retries_does_not_abort_document(self, mocker):
        """Under fast_extraction=True, one chunk raising
        ExtractionMaxRetriesError must not abort the whole document — the
        other chunk's nodes still make it into the final document, exactly
        like the legacy path's per-chunk skip-and-continue behavior.
        """

        async def _fake_extract_chunk_one_fails(
            *,
            prev_page,
            curr_pages,
            active_hierarchy,
            llm,
            curr_page_ids=None,
            user_context="",
            defer_hierarchy=False,
        ):
            if curr_pages[0] == "PAGE1":
                raise ExtractionMaxRetriesError("simulated failure for PAGE1's chunk")
            return [n.model_copy(deep=True) for n in _make_chunk_nodes(curr_pages[0])]

        mocker.patch(
            "scinr.newton.stages.extraction.extract_chunk",
            AsyncMock(side_effect=_fake_extract_chunk_one_fails),
        )

        doc = _make_doc()
        result = await extract_one_intermediate(doc, output_path=None, fast_extraction=True)

        assert result is not None
        titles = [n.title for n in result.document_structure]
        assert titles == ["Intro"]  # only the surviving chunk's node made it through.
