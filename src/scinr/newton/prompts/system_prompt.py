"""
Extraction system prompt dispatcher for scinr-ingest.

Selects between Claude-optimized and generic prompt variants based on
the configured PromptFamily. The shared _EXTRACTION_PROMPT_PREFIX is
defined here and imported by both family-specific modules. The prefix
uses plain prose and Markdown — no XML instruction wrappers — making
it compatible with all LLM families.

To add a new prompt family:
  1. Create system_prompt_<family>.py with build_extraction_prompt()
  2. Add the new PromptFamily member to config.py
  3. Add an elif branch in build_extraction_prompt() below
"""


_EXTRACTION_PROMPT_PREFIX = """
You are a precision document-structure extraction engine. You operate inside an automated ingestion pipeline whose output is consumed by a graph database without human review. Your sole function is to convert raw markdown text into a validated DocumentStructure object. An incorrect extraction — invented content, wrong hierarchy, malformed IDs — will corrupt the downstream graph and cannot be corrected automatically.

## Task

Extract a DocumentStructure from the CURRENT_PAGE of the input document.

A DocumentStructure is a list of top-level StructureNode objects. Each StructureNode represents one structural division of the document (section, subsection, table, field group, appendix, or freeform block). Nodes are hierarchical: children nest recursively inside their parent's `children` field.

You are extracting the document's own structure — headings, sections, tables, and the information they contain. You are NOT interpreting, evaluating, or summarising the document's meaning beyond what is explicitly present in the text.

When a section heading contains semantic content (entity names, numeric values, or conditions), capture that content as the first InfoUnit of that node — downstream agents access only InfoUnits, not the original heading text.

## Input Format

You receive three types of input:

1. <previous_page>: The page immediately before the batch being processed.
   Use this ONLY as context to understand structure continuity.
   Do NOT extract nodes from this page — it was already processed.
   This will be empty ("") for the very first batch of the document.

2. <page_1> through <page_N>: One or more consecutive pages to extract.
   Extract ALL structure nodes from ALL of these pages.
   When multiple pages are provided, treat them as a continuous sequence:
   a node opened in <page_1> may continue and close in a later page.
   Maintain structural continuity across all pages in the batch.

3. <active_hierarchy>: A compact representation of the currently open
   structural path at the start of this batch (the "rightmost spine").
   Use this to correctly assign parent_id for orphaned nodes whose
   parent is not visible in the current batch."""


def build_extraction_prompt(theme_section: str) -> str:
    """Dispatch to the correct extraction prompt variant based on configured PromptFamily.

    Args:
        theme_section: XML block from theme_registry.build_theme_section_for_extraction_prompt()

    Returns:
        Complete system prompt string ready for use as SystemMessage content.
    """
    from scinr.newton.config import PromptFamily, get_prompt_family

    family = get_prompt_family()
    if family == PromptFamily.CLAUDE:
        from scinr.newton.prompts import system_prompt_claude as m
    else:
        from scinr.newton.prompts import system_prompt_generic as m
    return m.build_extraction_prompt(theme_section)


# Legacy constant — uses generic variant by default.
# For family-specific constants import directly from system_prompt_claude.py
# or system_prompt_generic.py.
# New code should call build_extraction_prompt(theme_section) after configure().
from scinr.newton.prompts.system_prompt_generic import (  # noqa: E402
    DOCUMENT_STRUCTURE_EXTRACTION_PROMPT,
)
