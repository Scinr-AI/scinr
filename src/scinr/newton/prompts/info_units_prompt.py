"""System prompt of the second call of ``info_units_mode="auto"``.

The first call (structure extraction) decides, per node, ``verbatim`` or ``summary``. For
the ``summary`` nodes a second call writes the InfoUnits from the original paragraphs of
the node. The InfoUnit rules are the same ones the ``summary`` mode uses in the extraction
prompt of each prompt family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.config import PromptFamily

INFO_UNITS_PROMPT_HEADER = """You convert the content of ONE structural node of a document into InfoUnits.

You receive:
- <user_context> (optional): context supplied by the caller.
- <node>: the node's <title>, <role> and <title_path> (the titles of its ancestors, outermost first). Use them only to make descriptions self-contained; never create InfoUnits for the ancestors.
- <paragraphs>: the original paragraphs of this node, in document order. They are your only source.

Return `info_units`: an ordered list, in order of appearance in <paragraphs>. Every description must preserve all quantitative values, named entities, conditions, restrictions and qualifiers of its source passage, and contain nothing that is not in <paragraphs>. Do not translate.

In the rules below, CURRENT_PAGE means <paragraphs>. Ignore any rule about other output fields: you only return `info_units`.

Rules for InfoUnits:"""

_CACHE: dict[str, str] = {}


def build_info_units_prompt(family: PromptFamily) -> str:
    """System prompt of the auto-mode second call for *family* (cached per family)."""
    from scinr.newton.config import PromptFamily

    key = family.value
    if key not in _CACHE:
        if family == PromptFamily.CLAUDE:
            from scinr.newton.prompts.system_prompt_claude import _INFO_UNIT_RULES as rules
        elif family == PromptFamily.GPT_REASONING:
            from scinr.newton.prompts.system_prompt_gpt_reasoning import (
                _INFO_UNIT_RULES_GPT_REASONING as rules,
            )
        else:
            from scinr.newton.prompts.system_prompt_generic import (
                _INFO_UNIT_RULES_GENERIC as rules,
            )
        _CACHE[key] = INFO_UNITS_PROMPT_HEADER + "\n\n" + rules
    return _CACHE[key]


def reset_info_units_prompt_cache() -> None:
    _CACHE.clear()
