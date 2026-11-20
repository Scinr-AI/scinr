"""
tests/unit/test_extraction_stage_auto.py — End-to-end Stage 1 in info_units_mode="auto".

Same document as test_extraction_stage_verbatim.py: the fake LLM decides ``summary`` for the
table node and ``verbatim`` for the heading node; the second call only happens for the table.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from scinr.newton.config import StageLLMConfig, configure
from scinr.newton.converters.base import IntermediateDocument, IntermediatePage
from scinr.newton.extraction.extraction import reset_extraction_cache
from scinr.newton.stages.extraction import extract_one_intermediate

TABLE = "| a | b |\n|---|---|\n| 1 | 2 |"
PAGES = ["# Heading A\n\nBody A text", TABLE, ""]


def _node(node_id, title, start, end, mode):
    return {
        "node_id": node_id,
        "title": title,
        "role": "section",
        "appearance_order": 1,
        "paragraph_ranges": [{"start": start, "end": end}],
        "info_units_mode": mode,
    }


class AutoFakeLLM(BaseChatModel):
    """Answers structure requests by ``[[n]]`` markers and InfoUnits requests by schema."""

    structure_calls: list = Field(default_factory=list)
    info_units_calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "auto-fake"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        outer = self

        class _Runner:
            async def ainvoke(self, messages):
                human = messages[1].content
                if schema.__name__ == "InfoUnitsOutput":
                    outer.info_units_calls.append(human)
                    payload = {
                        "info_units": [
                            {"title": "Table values", "order": 0, "description": "a=1, b=2"}
                        ]
                    }
                else:
                    outer.structure_calls.append(human)
                    if "[[1]]" in human:
                        payload = {"nodes": [_node("1-a", "Heading A", 1, 2, "verbatim")]}
                    elif "[[3]]" in human:
                        payload = {"nodes": [_node("2-b", "Table", 3, 3, "summary")]}
                    else:
                        payload = {"nodes": []}
                return {
                    "raw": AIMessage(content=""),
                    "parsed": schema.model_validate(payload),
                    "parsing_error": None,
                }

        return _Runner()


@pytest.fixture(autouse=True)
def _reset():
    reset_extraction_cache()
    yield
    reset_extraction_cache()


def _make_doc() -> IntermediateDocument:
    return IntermediateDocument(
        pages=[
            IntermediatePage(index=i, markdown=md, page_id=f"pid-{i}")
            for i, md in enumerate(PAGES)
        ],
        document_name="auto-doc",
    )


def _configure(global_llm, **kwargs):
    configure(
        llm=global_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        extraction_batch_size=1,
        **kwargs,
    )


@pytest.mark.parametrize("fast_extraction", [False, True])
async def test_auto_decides_per_node(fast_extraction):
    llm = AutoFakeLLM()
    _configure(llm)
    result = await extract_one_intermediate(
        _make_doc(), None, fast_extraction=fast_extraction, info_units_mode="auto"
    )

    heading, table = result.document_structure
    assert heading.info_units_mode == "verbatim"
    assert [u.description for u in heading.info_units] == ["# Heading A", "Body A text"]
    assert table.info_units_mode == "summary"
    assert [u.description for u in table.info_units] == ["a=1, b=2"]
    assert heading.source_paragraphs == ["1:1-2"] and table.source_paragraphs == ["2:1-1"]

    # the second call happens only for the "summary" node, and sees the original text
    assert len(llm.info_units_calls) == 1
    assert TABLE in llm.info_units_calls[0] and "Heading A" not in llm.info_units_calls[0].split("<paragraphs>")[1]
    assert len(llm.structure_calls) == 3


async def test_second_call_uses_the_extraction_stage_llm():
    global_llm = AutoFakeLLM()
    stage_llm = AutoFakeLLM()
    _configure(global_llm, llm_stage_configuration={"extraction": StageLLMConfig(llm=stage_llm)})

    result = await extract_one_intermediate(_make_doc(), None, info_units_mode="auto")

    assert result.document_structure[1].info_units_mode == "summary"
    assert len(stage_llm.info_units_calls) == 1
    assert global_llm.info_units_calls == [] and global_llm.structure_calls == []
