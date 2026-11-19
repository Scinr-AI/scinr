"""
tests/unit/test_extraction_stage_verbatim.py — End-to-end Stage 1 with paragraph anchors.

``extract_one_intermediate()`` runs for the four combinations of ``fast_extraction`` and
``info_units_mode`` against a fake LLM that answers each chunk with scripted nodes citing
paragraph ranges.

Document (``extraction_batch_size=1``, one chunk per page):

    page 0 (pid-0): "# Heading A" / "Body A text"   -> paragraphs 1, 2
    page 1 (pid-1): a table                          -> paragraph 3
    page 2 (pid-2): empty                            -> no paragraphs
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from scinr.newton.config import configure
from scinr.newton.converters.base import IntermediateDocument, IntermediatePage
from scinr.newton.extraction.extraction import reset_extraction_cache
from scinr.newton.stages.extraction import extract_one_intermediate

TABLE = "| a | b |\n|---|---|\n| 1 | 2 |"
PAGES = ["# Heading A\n\nBody A text", TABLE, ""]


def _node(node_id, title, start, end, info_units):
    node = {
        "node_id": node_id,
        "title": title,
        "role": "section",
        "appearance_order": 1,
        "paragraph_ranges": [{"start": start, "end": end}],
    }
    if info_units is not None:
        node["info_units"] = info_units
    return node


def _chunk_payload(chunk: int, mode: str) -> dict:
    def units(text):
        return [{"title": "LLM unit", "order": 0, "description": text}] if mode == "summary" else None

    if chunk == 0:
        return {"nodes": [_node("1-a", "Heading A", 1, 2, units("LLM summary A"))]}
    if chunk == 1:
        return {"nodes": [_node("2-b", "Table", 3, 3, units("LLM summary table"))]}
    return {"nodes": []}


class FakeLLM:
    """Answers each chunk by the ``[[n]]`` markers found in the human message."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.human_messages: list[str] = []

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class _Runner:
            async def ainvoke(self, messages):
                human = messages[1].content
                outer.human_messages.append(human)
                if "[[1]]" in human:
                    chunk = 0
                elif "[[3]]" in human:
                    chunk = 1
                else:
                    chunk = 2
                parsed = schema.model_validate(_chunk_payload(chunk, outer.mode))
                return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}

        return _Runner()


@pytest.fixture(autouse=True)
def _configure_min(mock_llm):
    configure(
        llm=mock_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        extraction_batch_size=1,
    )
    reset_extraction_cache()
    yield
    reset_extraction_cache()


def _make_doc() -> IntermediateDocument:
    return IntermediateDocument(
        pages=[
            IntermediatePage(index=i, markdown=md, page_id=f"pid-{i}")
            for i, md in enumerate(PAGES)
        ],
        document_name="verbatim-doc",
    )


async def _run(mocker, *, fast_extraction: bool, info_units_mode: str):
    llm = FakeLLM(info_units_mode)
    mocker.patch("scinr.newton.stages.extraction.get_llm", return_value=llm)
    result = await extract_one_intermediate(
        _make_doc(),
        None,
        fast_extraction=fast_extraction,
        info_units_mode=info_units_mode,
    )
    return result, llm


@pytest.mark.parametrize("fast_extraction", [False, True])
@pytest.mark.parametrize("info_units_mode", ["summary", "verbatim"])
async def test_four_combinations(mocker, fast_extraction, info_units_mode):
    result, _ = await _run(
        mocker, fast_extraction=fast_extraction, info_units_mode=info_units_mode
    )

    assert result is not None
    heading, table = result.document_structure
    assert heading.title == "Heading A" and table.title == "Table"

    assert heading.source_page_ids == ["pid-0"]
    assert table.source_page_ids == ["pid-1"]
    assert heading.source_paragraphs == ["1:1-2"]
    assert table.source_paragraphs == ["2:1-1"]

    for node in (heading, table):
        assert node.info_units_mode == info_units_mode
        assert node.paragraph_segmenter_version == 1

    if info_units_mode == "verbatim":
        assert [u.description for u in heading.info_units] == ["# Heading A", "Body A text"]
        assert [u.order for u in heading.info_units] == [0, 1]
        assert [u.description for u in table.info_units] == [TABLE]
    else:
        assert [u.description for u in heading.info_units] == ["LLM summary A"]
        assert [u.description for u in table.info_units] == ["LLM summary table"]


@pytest.mark.parametrize("info_units_mode", ["summary", "verbatim"])
async def test_numbering_is_identical_in_both_branches(mocker, info_units_mode):
    _, normal = await _run(mocker, fast_extraction=False, info_units_mode=info_units_mode)
    _, fast = await _run(mocker, fast_extraction=True, info_units_mode=info_units_mode)

    def marks(llm):
        # The fast branch runs chunks concurrently, so compare as sets of messages.
        return sorted(
            tuple(m for m in ("[[1]]", "[[2]]", "[[3]]") if m in msg)
            for msg in llm.human_messages
        )

    assert len(normal.human_messages) == len(fast.human_messages) == 3
    assert marks(normal) == marks(fast) == [(), ("[[1]]", "[[2]]"), ("[[3]]",)]
