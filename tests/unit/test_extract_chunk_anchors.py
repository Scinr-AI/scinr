"""
tests/unit/test_extract_chunk_anchors.py — extract_chunk() with paragraph anchors.

A fake LLM answers every structured-output call with a scripted payload and records the
messages it receives.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage

import scinr.newton.extraction.extraction as extraction_mod
from scinr.newton.config import configure
from scinr.newton.extraction.extraction import (
    _build_human_message,
    extract_chunk,
    reset_extraction_cache,
)
from scinr.newton.extraction.paragraphs import ParagraphIndex, ParagraphStats
from scinr.newton.models.document_structure import ParagraphRange, StructureNode

PAGES = ["Heading one\n\nBody one", "Heading two\n\nBody two"]  # paragraphs 1-2 and 3-4
PAGE_IDS = ["pid0", "pid1"]


class FakeLLM:
    """``with_structured_output`` returns a runner that replays ``payloads`` in order."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls: list[list] = []
        self.schemas: list[type] = []

    def with_structured_output(self, schema, include_raw=True):
        self.schemas.append(schema)
        outer = self

        class _Runner:
            async def ainvoke(self, messages):
                outer.calls.append(messages)
                payload = outer.payloads.pop(0)
                parsed = None if payload is None else schema.model_validate(payload)
                return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}

        return _Runner()


def _node(node_id="1", ranges=(), **extra):
    return {
        "node_id": node_id,
        "title": extra.pop("title", "Heading"),
        "role": "section",
        "appearance_order": 1,
        "paragraph_ranges": [{"start": a, "end": b} for a, b in ranges],
        **extra,
    }


@pytest.fixture(autouse=True)
def _configure_min(mock_llm):
    configure(
        llm=mock_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
    )
    reset_extraction_cache()
    yield
    reset_extraction_cache()


@pytest.fixture
def index():
    return ParagraphIndex(PAGES, [0, 1], {0: "pid0", 1: "pid1"})


async def _run(llm, index, *, positions=(1,), mode="summary", stats=None, **kwargs):
    positions = list(positions)
    return await extract_chunk(
        "PREV",
        [PAGES[p] for p in positions],
        "(none)",
        llm=llm,
        curr_page_ids=[PAGE_IDS[p] for p in positions],
        paragraph_index=index,
        curr_positions=positions,
        info_units_mode=mode,
        paragraph_stats=stats,
        log_context="chunk-test",
        **kwargs,
    )


async def test_message_has_markers_in_current_pages_only(index):
    llm = FakeLLM([{"nodes": []}])
    await _run(llm, index)
    human = llm.calls[0][1].content
    assert "[[3]]\nHeading two" in human and "[[4]]\nBody two" in human
    prev = human.split("</previous_page>")[0]
    assert "[[" not in prev


async def test_phase2_message_also_has_markers(index, monkeypatch):
    async def no_repair(**_):
        return None

    monkeypatch.setattr(extraction_mod, "run_repair_loop", no_repair)
    llm = FakeLLM([None, {"nodes": []}])
    await _run(llm, index)
    for call in llm.calls:
        human = call[1].content
        assert "[[3]]\nHeading two" in human
        assert "[[" not in human.split("</previous_page>")[0]
    assert "<previous_page>\n\n</previous_page>" in llm.calls[1][1].content


async def test_without_index_message_is_unchanged():
    llm = FakeLLM([{"nodes": []}])
    await extract_chunk("PREV", ["page text"], "H", llm=llm)
    assert llm.calls[0][1].content == _build_human_message("PREV", ["page text"], "H")


async def test_discard_reasons_and_stats(index, caplog):
    stats = ParagraphStats()
    payload = {"nodes": [_node(ranges=[(4, 3), (1, 2), (3, 9), (3, 4)])]}
    llm = FakeLLM([payload])
    with caplog.at_level(logging.WARNING, logger=extraction_mod.__name__):
        nodes = await _run(llm, index, stats=stats)
    assert [(r.start, r.end) for r in nodes[0].paragraph_ranges] == [(3, 4)]
    assert stats.discarded_ranges == 3
    text = caplog.text
    assert "start > end" in text
    assert "outside the current pages (allowed 3-4)" in text
    assert "chunk-test" in text


async def test_discard_when_no_paragraphs_in_current_pages(caplog):
    empty_index = ParagraphIndex([""], [0], {0: "pid0"})
    stats = ParagraphStats()
    llm = FakeLLM([{"nodes": [_node(ranges=[(1, 1)])]}])
    with caplog.at_level(logging.WARNING, logger=extraction_mod.__name__):
        nodes = await extract_chunk(
            "",
            [""],
            "(none)",
            llm=llm,
            paragraph_index=empty_index,
            curr_positions=[0],
            paragraph_stats=stats,
        )
    assert nodes[0].paragraph_ranges == []
    assert stats.discarded_ranges == 1
    assert "no paragraphs in the current pages" in caplog.text


async def test_overlapping_ranges_are_merged(index):
    llm = FakeLLM([{"nodes": [_node(ranges=[(3, 3), (3, 4)])]}])
    nodes = await _run(llm, index)
    assert [(r.start, r.end) for r in nodes[0].paragraph_ranges] == [(3, 4)]


async def test_nested_children_keep_valid_ranges(index):
    child = _node("1_1", ranges=[(4, 4)])
    llm = FakeLLM([{"nodes": [_node(ranges=[(3, 3)], children=[child])]}])
    nodes = await _run(llm, index)
    assert nodes[0].paragraph_ranges == [ParagraphRange(start=3, end=3)]
    assert nodes[0].children[0].paragraph_ranges == [ParagraphRange(start=4, end=4)]


async def test_node_without_valid_ranges_falls_back_to_chunk_pages(index):
    llm = FakeLLM([{"nodes": [_node(ranges=[(99, 99)])]}])
    nodes = await _run(llm, index, positions=(0, 1))
    assert nodes[0].paragraph_ranges == []
    assert nodes[0].source_page_ids == PAGE_IDS


async def test_markers_are_stripped_from_title_and_infounits(index):
    node = _node(
        ranges=[(3, 4)],
        title="[[3]] Heading two",
        info_units=[{"title": "T [[4]]", "order": 0, "description": "[[3]] desc [[4]]"}],
    )
    llm = FakeLLM([{"nodes": [node]}])
    nodes = await _run(llm, index)
    assert nodes[0].title == "Heading two"
    assert nodes[0].info_units[0].title == "T"
    assert nodes[0].info_units[0].description == "desc"


async def test_verbatim_schema_and_prompt(index):
    llm = FakeLLM([{"nodes": [_node(ranges=[(3, 4)])]}])
    await _run(llm, index, mode="verbatim")
    node_cls = llm.schemas[0].model_fields["nodes"].annotation.__args__[0]
    assert "info_units" not in node_cls.model_fields
    system = llm.calls[0][0].content
    system_text = system if isinstance(system, str) else str(system)
    assert extraction_mod._get_extraction_prompt("verbatim") in system_text
    assert "info_unit" not in system_text.lower()


async def test_verbatim_without_index_raises():
    with pytest.raises(ValueError):
        await extract_chunk("", ["x"], "", llm=FakeLLM([]), info_units_mode="verbatim")


async def test_inconsistent_positions_raise(index):
    with pytest.raises(ValueError):
        await extract_chunk(
            "", ["x"], "", llm=FakeLLM([]), paragraph_index=index, curr_positions=None
        )
    with pytest.raises(ValueError):
        await extract_chunk(
            "", ["x"], "", llm=FakeLLM([]), paragraph_index=index, curr_positions=[0, 1]
        )


async def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        await extract_chunk("", ["x"], "", llm=FakeLLM([]), info_units_mode="other")  # type: ignore[arg-type]


async def test_returns_structure_node_instances(index):
    llm = FakeLLM([{"nodes": [_node(ranges=[(3, 4)], children=[_node("1_1", ranges=[(4, 4)])])]}])
    nodes = await _run(llm, index)
    assert type(nodes[0]) is StructureNode
    assert type(nodes[0].children[0]) is StructureNode


# ── auto mode ────────────────────────────────────────────────────────────────


async def test_auto_schema_has_decision_and_no_info_units(index):
    llm = FakeLLM([{"nodes": []}])
    await _run(llm, index, mode="auto")
    props = llm.schemas[0].model_json_schema()
    node_schema = next(v for k, v in props["$defs"].items() if k.endswith("StructureNode"))
    assert "info_units_mode" in node_schema["required"]
    assert "info_units" not in node_schema["properties"]


async def test_auto_decisions_reach_nodes_including_children(index):
    payload = {
        "nodes": [
            _node(
                ranges=[(3, 4)],
                info_units_mode="summary",
                children=[_node("1_1", ranges=[(4, 4)], info_units_mode="verbatim")],
            )
        ]
    }
    nodes = await _run(FakeLLM([payload]), index, mode="auto")
    assert nodes[0].info_units_mode == "summary"
    assert nodes[0].children[0].info_units_mode == "verbatim"


async def test_auto_requires_decision(index):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await _run(FakeLLM([{"nodes": [_node(ranges=[(3, 4)])]}]), index, mode="auto")


async def test_auto_without_index_raises():
    with pytest.raises(ValueError):
        await extract_chunk("", ["x"], "", llm=FakeLLM([]), info_units_mode="auto")
