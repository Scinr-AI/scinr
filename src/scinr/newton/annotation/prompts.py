"""
Annotation agent prompts dispatcher for scinr-ingest.

Selects between Claude-optimized, generic, and OpenAI reasoning model prompt variants
based on the configured PromptFamily. Callers import from this module as before —
no changes needed in nodes.py or other call sites.

To add a new prompt family:
  1. Create annotation/prompts_<family>.py with the builder functions
  2. Add the new PromptFamily member to config.py
  3. Add an elif branch in each dispatcher function below
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeNode


def _m():
    """Return the prompt module for the currently configured PromptFamily."""
    from scinr.newton.config import PromptFamily, get_prompt_family

    family = get_prompt_family()
    if family == PromptFamily.CLAUDE:
        from scinr.newton.annotation import prompts_claude as m
    elif family == PromptFamily.GPT_REASONING:
        from scinr.newton.annotation import prompts_gpt_reasoning as m
    else:
        from scinr.newton.annotation import prompts_generic as m
    return m


def build_theme_classification_prompt(
    themes_block: str,
    document_name: str = "",
    theme_histogram: str = "",
) -> str:
    """Build the theme classification prompt for the configured prompt family."""
    return _m().build_theme_classification_prompt(themes_block, document_name, theme_histogram)


def build_annotation_decision_prompt(theme_node: "ThemeNode") -> str:
    """Build the annotation decision prompt for the configured prompt family."""
    return _m().build_annotation_decision_prompt(theme_node)
