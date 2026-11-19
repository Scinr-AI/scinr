"""
tests/unit/test_extraction_prompts_modes.py — Extraction prompts per info_units_mode.

Summary mode keeps the InfoUnit rules and adds the paragraph-anchors section; verbatim mode
asks only for structure + paragraph references and must never mention InfoUnits.
"""

from __future__ import annotations

import pytest

import scinr.newton.extraction.extraction as extraction_mod
from scinr.newton.config import PromptFamily, configure
from scinr.newton.prompts.system_prompt import (
    _VERBATIM_CONTENT_SENTENCE,
    PARAGRAPH_ANCHORS_SECTION,
    build_extraction_prompt,
)

FAMILIES = [PromptFamily.CLAUDE, PromptFamily.GENERIC, PromptFamily.GPT_REASONING]
FORBIDDEN = ("info_unit", "infounit", "info unit")


@pytest.fixture(autouse=True)
def _configure_min(mock_llm):
    configure(
        llm=mock_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
    )
    extraction_mod.reset_extraction_cache()
    yield
    extraction_mod.reset_extraction_cache()


@pytest.mark.parametrize("family", FAMILIES)
def test_summary_prompt_has_anchors_and_infounit_rules(family):
    prompt = build_extraction_prompt("THEMES", family=family, info_units_mode="summary")
    assert PARAGRAPH_ANCHORS_SECTION in prompt
    assert "InfoUnit" in prompt
    assert _VERBATIM_CONTENT_SENTENCE not in prompt
    assert "THEMES" in prompt


@pytest.mark.parametrize("family", FAMILIES)
def test_verbatim_prompt_has_anchors_and_no_infounit_mentions(family):
    prompt = build_extraction_prompt("THEMES", family=family, info_units_mode="verbatim")
    assert PARAGRAPH_ANCHORS_SECTION in prompt
    assert _VERBATIM_CONTENT_SENTENCE in prompt
    lowered = prompt.lower()
    for forbidden in FORBIDDEN:
        assert forbidden not in lowered
    assert "paragraph_ranges" in prompt


def test_prompt_cache_is_per_family_and_mode_and_resettable():
    summary = extraction_mod._get_extraction_prompt("summary")
    verbatim = extraction_mod._get_extraction_prompt("verbatim")
    assert summary != verbatim
    assert extraction_mod._get_extraction_prompt("summary") is summary
    assert extraction_mod._get_extraction_prompt("verbatim") is verbatim
    assert len(extraction_mod._EXTRACTION_PROMPTS) == 2

    extraction_mod.reset_extraction_cache()
    assert extraction_mod._EXTRACTION_PROMPTS == {}
    assert extraction_mod._EXTRACTION_OUTPUT_CLASSES == {}


def test_output_class_cache_is_per_mode():
    summary = extraction_mod._get_extraction_output_class("summary")
    verbatim = extraction_mod._get_extraction_output_class("verbatim")
    assert summary is not verbatim
    assert extraction_mod._get_extraction_output_class("summary") is summary


def _node_fields(schema: type) -> set[str]:
    node_cls = schema.model_fields["nodes"].annotation.__args__[0]
    return set(node_cls.model_fields)


def test_output_schema_fields_per_mode():
    assert "info_units" in _node_fields(extraction_mod._get_extraction_output_class("summary"))
    verbatim_fields = _node_fields(extraction_mod._get_extraction_output_class("verbatim"))
    assert "info_units" not in verbatim_fields
    assert "paragraph_ranges" in verbatim_fields
