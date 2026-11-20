"""
Extraction system prompt dispatcher for scinr-ingest.

Selects between Claude-optimized, generic, and OpenAI reasoning model prompt variants
based on the configured PromptFamily. The shared _EXTRACTION_PROMPT_PREFIX is
defined here and imported by all family-specific modules. The prefix uses plain
prose and Markdown — no XML instruction wrappers — making it compatible with
all LLM families.

To add a new prompt family:
  1. Create system_prompt_<family>.py with build_extraction_prompt()
  2. Add the new PromptFamily member to config.py
  3. Add an elif branch in build_extraction_prompt() below
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.config import PromptFamily

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
   parent is not visible in the current batch.

You may also receive an optional fourth input:

4. <extraction_mode>: Present only when this batch is processed in
   isolation from the rest of the document (fast/parallel mode). Its
   content explains how to adjust the ambiguous-role fallback in that
   case. Absent entirely in the normal sequential path — do not assume
   any particular extraction mode when this tag is not present."""

PARAGRAPH_ANCHORS_SECTION = """## Paragraph Anchors

Every paragraph inside <page_1> ... <page_N> is preceded by an anchor on its own line, of the form [[n]], where n is an integer. Anchors are numbered continuously across the whole document. <previous_page> has no anchors.

For every StructureNode you output, fill `paragraph_ranges` with the anchors of the paragraphs that form THAT node's own content:
- Include the node's heading paragraph and its body paragraphs.
- Do NOT include paragraphs that belong to its children — each child cites its own.
- Use one range {"start": a, "end": b} per continuous run of paragraphs (start == end for a single paragraph). Use several ranges when the node's content is interrupted, for example by a page footer.
- Cite only anchors that appear inside <page_1> ... <page_N>. Never invent an anchor number.
- Do not cite running headers, footers, page numbers or other layout noise.
- A node that continues beyond the current pages cites only the paragraphs visible in the current pages.

Anchors are references only. NEVER copy an anchor ([[n]]) into `title` or into any other text field."""

_HEADING_INFO_UNIT_SENTENCE = (
    "When a section heading contains semantic content (entity names, numeric values, or "
    "conditions), capture that content as the first InfoUnit of that node — downstream "
    "agents access only InfoUnits, not the original heading text."
)
_VERBATIM_CONTENT_SENTENCE = (
    "Do not rewrite or summarise the content. The pipeline copies the original text of "
    "every paragraph you cite in `paragraph_ranges`: your output is the structure and the "
    "paragraph references only."
)
assert _HEADING_INFO_UNIT_SENTENCE in _EXTRACTION_PROMPT_PREFIX
_EXTRACTION_PROMPT_PREFIX_VERBATIM = _EXTRACTION_PROMPT_PREFIX.replace(
    _HEADING_INFO_UNIT_SENTENCE, _VERBATIM_CONTENT_SENTENCE
)


AUTO_DECISION_SECTION = """## InfoUnits Mode Decision

For every StructureNode, set `info_units_mode` to decide how the pipeline will build that node's InfoUnits from the paragraphs you cite in `paragraph_ranges`.

InfoUnits are what later stages use to classify each node against a catalog of domain data models, to extract into those models the values and any other information they require, and to search and answer questions over the document.

- "verbatim" keeps the original wording exactly. It is the right choice when the text is already clear enough on its own: prose, definitions, requirements, procedure steps, or lists whose items are complete sentences.
- "summary" rewrites the content into explicit, self-contained notes. It is the right choice when the information would only be understood with context the text itself does not carry: tables or forms whose values depend on their headers, fragments that depend on the parent section, content interrupted by layout elements, or text with heavy layout or OCR noise.

To decide, ask: if this text were the only thing a reader had, could they use all the information in it correctly? If yes, "verbatim"; if not, "summary". A node whose only cited paragraph is its heading uses "verbatim"."""


def build_extraction_prompt(
    theme_section: str,
    *,
    family: "PromptFamily | None" = None,
    info_units_mode: str = "summary",
) -> str:
    """Dispatch to the correct extraction prompt variant based on configured PromptFamily.

    Args:
        theme_section: XML block from theme_registry.build_theme_section_for_extraction_prompt()
        family: Prompt family to use. Defaults to the family configured for the
            extraction stage.
        info_units_mode: "summary" (LLM writes InfoUnits), "verbatim" (LLM only
            cites paragraph ranges) or "auto" (verbatim prompt plus the per-node mode
            decision, ``AUTO_DECISION_SECTION``).

    Returns:
        Complete system prompt string ready for use as SystemMessage content.
    """
    from scinr.newton.config import PromptFamily, get_prompt_family

    family = family if family is not None else get_prompt_family(stage="extraction")
    if family == PromptFamily.CLAUDE:
        from scinr.newton.prompts import system_prompt_claude as m
    elif family == PromptFamily.GPT_REASONING:
        from scinr.newton.prompts import system_prompt_gpt_reasoning as m
    else:
        from scinr.newton.prompts import system_prompt_generic as m
    if info_units_mode == "auto":
        return (
            m.build_extraction_prompt(theme_section, info_units_mode="verbatim")
            + "\n\n"
            + AUTO_DECISION_SECTION
        )
    return m.build_extraction_prompt(theme_section, info_units_mode=info_units_mode)


# Legacy constant — uses generic variant by default.
# For family-specific constants import directly from system_prompt_claude.py
# or system_prompt_generic.py.
# New code should call build_extraction_prompt(theme_section) after configure().
from scinr.newton.prompts.system_prompt_generic import (  # noqa: E402
    DOCUMENT_STRUCTURE_EXTRACTION_PROMPT,
)
