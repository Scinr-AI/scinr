"""
tests/unit/test_auto_info_units.py — second call of info_units_mode="auto".

A fake LLM answers InfoUnitsOutput requests; the answer is chosen by the ``<title>`` found in
the human message.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

import scinr.newton.extraction.info_unit_extraction as mod
from scinr.newton.config import configure
from scinr.newton.extraction.extraction import reset_extraction_cache
from scinr.newton.extraction.info_unit_extraction import (
    AutoStats,
    build_info_units_human_message,
    group_paragraphs,
    resolve_auto_info_units,
)
from scinr.newton.extraction.paragraphs import ParagraphIndex
from scinr.newton.models.document_structure import Document, ParagraphRange, StructureNode

PAGES = ["Alpha para\n\nBeta para\n\nGamma para", "Delta para\n\nEpsilon para"]  # 1-3, 4-5


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
    return ParagraphIndex(PAGES, [0, 1], {0: "p0", 1: "p1"})


class FakeLLM:
    def __init__(self, answers=None, raise_for=(), unparsed_for=()):
        self.answers = answers or {}
        self.raise_for = set(raise_for)
        self.unparsed_for = set(unparsed_for)
        self.human: list[str] = []
        self.system: list = []
        self.active = 0
        self.max_active = 0

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class _Runner:
            async def ainvoke(self, messages):
                human = messages[1].content
                outer.human.append(human)
                outer.system.append(messages[0])
                title = human.split("<title>")[1].split("</title>")[0]
                outer.active += 1
                outer.max_active = max(outer.max_active, outer.active)
                try:
                    await asyncio.sleep(0)
                    if title in outer.raise_for:
                        raise RuntimeError("boom")
                    if title in outer.unparsed_for:
                        return {
                            "raw": AIMessage(content="{}"),
                            "parsed": None,
                            "parsing_error": "bad",
                        }
                    units = outer.answers.get(
                        title, [{"title": f"U-{title} [[7]]", "order": 9, "description": f"D {title} [[2]]"}]
                    )
                    return {
                        "raw": AIMessage(content=""),
                        "parsed": schema.model_validate({"info_units": units}),
                        "parsing_error": None,
                    }
                finally:
                    outer.active -= 1

        return _Runner()


def _node(node_id, title, ranges, mode, children=()):
    return StructureNode(
        node_id=node_id,
        title=title,
        role="section",
        appearance_order=1,
        paragraph_ranges=[ParagraphRange(start=a, end=b) for a, b in ranges],
        info_units_mode=mode,
        children=list(children),
    )


def _doc(nodes, context=None):
    return Document(
        document_name="doc", raw_file_id="r", document_structure=nodes, context_instructions=context
    )


async def _resolve(doc, index, llm, *, max_input=10000, stats=None):
    stats = stats or AutoStats()
    await resolve_auto_info_units(
        doc, index, llm=llm, max_input_chars=max_input, verbatim_max_chars=4000, stats=stats
    )
    return stats


def test_group_paragraphs(index):
    assert group_paragraphs([1, 2, 3], index, 1000) == [[1, 2, 3]]
    # each paragraph is ~10 chars
    assert group_paragraphs([1, 2, 3, 4, 5], index, 20) == [[1, 2], [3, 4], [5]]
    assert group_paragraphs([1, 2], index, 5) == [[1], [2]]
    assert group_paragraphs([], index, 5) == []


def test_human_message_format():
    node = _node("n", "Title", [(1, 1)], "summary")
    with_ctx = build_info_units_human_message(
        node, ["a", "b"], title_path="Root › Parent", user_context="CTX"
    )
    assert with_ctx == (
        "<user_context>\nCTX\n</user_context>\n\n"
        "<node>\n<title_path>Root › Parent</title_path>\n<title>Title</title>\n"
        "<role>section</role>\n</node>\n\n"
        "<paragraphs>\na\n\nb\n</paragraphs>"
    )
    without = build_info_units_human_message(node, ["a"], title_path="", user_context="")
    assert without.startswith("<node>\n<title_path></title_path>")


async def test_summary_node_gets_llm_units_verbatim_gets_source(index):
    doc = _doc(
        [_node("a", "A", [(1, 2)], "summary"), _node("b", "B", [(4, 5)], "verbatim")],
        context="CTX",
    )
    llm = FakeLLM({"A": [{"title": "t1", "order": 5, "description": "d1"}, {"title": "t2", "order": 5, "description": "d2"}]})
    stats = await _resolve(doc, index, llm)
    a, b = doc.document_structure
    assert (a.info_units_mode, [u.order for u in a.info_units]) == ("summary", [0, 1])
    assert [u.description for u in a.info_units] == ["d1", "d2"]
    assert (b.info_units_mode, [u.description for u in b.info_units]) == (
        "verbatim",
        ["Delta para", "Epsilon para"],
    )
    assert len(llm.human) == 1 and "<user_context>\nCTX" in llm.human[0]
    assert "Alpha para\n\nBeta para" in llm.human[0] and "[[" not in llm.human[0]
    assert (stats.summary_nodes, stats.verbatim_nodes, stats.calls, stats.fallbacks) == (1, 1, 1, 0)


async def test_markers_are_stripped_from_llm_output(index):
    doc = _doc([_node("a", "A", [(1, 1)], "summary")])
    await _resolve(doc, index, FakeLLM())
    unit = doc.document_structure[0].info_units[0]
    assert "[[" not in unit.title and "[[" not in unit.description
    assert unit.order == 0


async def test_node_without_ranges_is_verbatim_and_empty(index):
    doc = _doc([_node("a", "A", [], "summary")])
    llm = FakeLLM()
    stats = await _resolve(doc, index, llm)
    node = doc.document_structure[0]
    assert (node.info_units_mode, node.info_units) == ("verbatim", [])
    assert llm.human == [] and stats.verbatim_nodes == 1


async def test_split_node_makes_two_calls_in_order(index):
    doc = _doc([_node("a", "A", [(1, 5)], "summary")])
    llm = FakeLLM()
    stats = await _resolve(doc, index, llm, max_input=1000)
    assert stats.calls == 1  # fits in one group

    llm = FakeLLM()
    stats = AutoStats()
    # force 2 groups by shrinking the limit below the node's text (bypass the config minimum)
    await _resolve(doc, index, llm, max_input=25, stats=stats)
    assert stats.calls == len(llm.human) >= 2
    assert [u.order for u in doc.document_structure[0].info_units] == list(range(stats.calls))


async def test_unparsed_and_failed_repair_falls_back_to_verbatim(index, monkeypatch, caplog):
    async def no_repair(**_):
        return None

    monkeypatch.setattr(mod, "run_repair_loop", no_repair)
    doc = _doc([_node("a", "A", [(1, 1)], "summary")])
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        stats = await _resolve(doc, index, FakeLLM(unparsed_for=["A"]))
    node = doc.document_structure[0]
    assert (node.info_units_mode, [u.description for u in node.info_units]) == ("verbatim", ["Alpha para"])
    assert stats.fallbacks == 1 and stats.verbatim_nodes == 1 and stats.summary_nodes == 0
    assert "falling back to verbatim" in caplog.text and "all repair attempts failed" in caplog.text


async def test_llm_exception_falls_back_to_verbatim(index, caplog):
    doc = _doc([_node("a", "A", [(1, 1)], "summary"), _node("b", "B", [(2, 2)], "summary")])
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        stats = await _resolve(doc, index, FakeLLM(raise_for=["A"]))
    a, b = doc.document_structure
    assert a.info_units_mode == "verbatim" and a.info_units[0].description == "Alpha para"
    assert b.info_units_mode == "summary"
    assert "boom" in caplog.text
    assert (stats.fallbacks, stats.summary_nodes) == (1, 1)


async def test_grandchild_title_path(index):
    grand = _node("g", "Grand", [(5, 5)], "summary")
    parent = _node("p", None, [(2, 2)], "verbatim", children=[grand])
    doc = _doc([_node("r", "Root", [(1, 1)], "verbatim", children=[parent])])
    llm = FakeLLM()
    await _resolve(doc, index, llm)
    assert "<title_path>Root › p</title_path>" in llm.human[0]


async def test_calls_use_the_llm_semaphore(index, monkeypatch):
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(mod, "get_llm_semaphore", lambda: sem)
    doc = _doc([_node(str(i), f"N{i}", [(i, i)], "summary") for i in (1, 2, 3, 4)])
    llm = FakeLLM()
    await _resolve(doc, index, llm)
    assert llm.max_active == 1 and len(llm.human) == 4


async def test_final_log_and_no_auto_left(index, caplog):
    doc = _doc([_node("a", "A", [(1, 1)], "summary"), _node("b", "B", [(2, 2)], "verbatim")])
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        await _resolve(doc, index, FakeLLM())
    assert "1 summary nodes, 1 verbatim nodes, 1 calls, 0 fallbacks" in caplog.text
    assert all(n.info_units_mode in ("summary", "verbatim") for n in doc.document_structure)
