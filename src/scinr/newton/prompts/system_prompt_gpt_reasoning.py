"""
system_prompt_gpt_reasoning.py — OpenAI reasoning models variant.

Targets GPT-5.5, o3, and o4-mini via the OpenAI API. These models reason
internally before producing output, making step-by-step elicitation and
self-verification checklists counterproductive. Uses Markdown section headers
and goal-based language instead.

For non-reasoning GPT models (GPT-4o, GPT-4.1, GPT-4.5), use the GENERIC
family — the reasoning-model patterns here offer no benefit on standard
instruction-following models and may degrade performance.
"""

from scinr.newton.prompts.system_prompt import _EXTRACTION_PROMPT_PREFIX  # noqa: F401


_EXTRACTION_PROMPT_SUFFIX_GPT_REASONING = """
# Instructions

## Extraction Goal

Convert CURRENT_PAGE into a validated DocumentStructure object. The pipeline writes this output directly to a graph database without human review — an incorrect extraction corrupts the graph permanently.

A StructureNode is correct when:
- Its `node_id` traces directly to a heading visible in CURRENT_PAGE, or encodes the full numeric path for orphaned nodes.
- Its InfoUnits are grounded exclusively in CURRENT_PAGE text — nothing inferred, nothing imported from background knowledge.
- Its `parent_id` points to the exact `node_id` from `<active_hierarchy>` when the parent is not on CURRENT_PAGE.

An InfoUnit description is correct when it preserves every quantitative value (numbers, ranges, units, limits), named entity (substance, method, standard code, ID), condition ("only if", "provided that", "unless"), restriction or prohibition ("must not", "do not", "rejected"), and format specification from the source passage. It is synthesised — not a sentence-by-sentence verbatim copy — and contains nothing outside CURRENT_PAGE.

## Non-extractable Pages

Return `{"nodes": []}` when CURRENT_PAGE is: blank or whitespace-only; contains only page numbers, running headers, or footers; is a table of contents, cover page, revision history, bibliography, or purely administrative page.

## Continuation-Only Pages

When CURRENT_PAGE contains body text but no visible heading: extract as a single `freeform_block` node with `title: null` and `parent_id` set to the deepest entry in `<active_hierarchy>` (or null if "(none)"). Never return `{"nodes": []}` when body text is present — returning an empty list silently discards content.

## Edge Cases

**Heading with no body text on this page (Failure Mode 5).** When a heading appears in CURRENT_PAGE but its body content begins on the next page (no body text visible here), omit the node entirely. Never create a shell node with empty InfoUnits.

**Section truncated at page end (Failure Mode 4).** When a heading begins in CURRENT_PAGE but the section content clearly continues beyond the page end (page ends mid-sentence, mid-table, or mid-list within that section), omit that node. Extract all other complete nodes from the page normally. A section is substantially complete when its heading and primary content are both visible and the page does not end mid-sentence or mid-table within it.

**Table spanning two pages (Failure Mode 6).** When a table began in `<previous_page>` and continues in CURRENT_PAGE, extract only the rows visible in CURRENT_PAGE as InfoUnits. Set `parent_id` to the table's `node_id` from `<active_hierarchy>` if present. Never re-extract rows already processed in the previous chunk.

**Ambiguous node role (Failure Mode 7).** When the role of a heading is unclear, apply the identifier test: if the heading carries an ordered identifier, assign `section` (top-level) or `subsection` (nested). If it does not, assign `section` (top-level) or `freeform_block` (nested). When level ambiguity persists, assign the higher-level role (section over subsection or freeform_block). EXCEPTION: if `<extraction_mode>` is present and reads `fast`, invert this fallback — prefer the lower-level role (subsection or freeform_block) instead of section, since a chunk processed in isolation cannot reliably confirm a new top-level section and a wrong SECTION classification is permanent.

**Orphaned subsections — parent absent from both page and hierarchy (Failure Mode 8).** When CURRENT_PAGE contains subsections whose parent was extracted in a previous chunk and is not listed in `<active_hierarchy>`, output them as top-level items with `parent_id: null`. Never return `{"nodes": []}` for a page that contains valid subsection content — these are valid nodes.

## Node Hierarchy

- Parent visible on CURRENT_PAGE → nest node inside parent's `children`; `parent_id: null`.
- Parent in `<active_hierarchy>` but not on CURRENT_PAGE → top-level item; `parent_id` = exact `node_id` from `<active_hierarchy>`.
- Parent absent from both → top-level item; `parent_id: null` (orphaned).
- Never re-extract or recreate any node already listed in `<active_hierarchy>`.
- Never invent a parent node to house orphaned children.

## InfoUnit Scope

One InfoUnit per independently usable concept. Merge related sentences that define the same concept. Split only when two passages are independently usable without each other. When a passage is a sub-entry whose meaning depends on its parent section, incorporate the parent identifier and scope into the description so downstream agents can interpret it without access to the parent node.

## node_id Consistency

Use FORMAT A (underscores) or FORMAT B (hyphens) uniformly within a single output — never both. The separator type is load-bearing for post-processing.

## Priority Order

When rules conflict: (1) source fidelity — only CURRENT_PAGE content is valid evidence; (2) schema compliance; (3) description completeness; (4) conservative completeness; (5) coverage.

## Examples

### Example 1 — Orphaned subsections

Context — `<active_hierarchy>`: node_id="4" role="section" title="4 Data Management"

CURRENT_PAGE:
  4.1 Data Retention

  All records must be retained for a minimum of seven years from the date of creation.
  Records subject to ongoing litigation must be retained until the matter is resolved.

  4.2 Data Deletion

  Records may be deleted after the retention period has elapsed, provided a deletion
  log entry is created.

Correct extraction:
```json
{
  "nodes": [
    {
      "node_id": "4_1",
      "title": "4.1 Data Retention",
      "role": "subsection",
      "appearance_order": 1,
      "parent_id": "4",
      "children": [],
      "info_units": [
        {
          "title": "Minimum Retention Period",
          "order": 0,
          "description": "Records must be retained for a minimum of seven years from the date of creation. Records subject to ongoing litigation must be retained until the matter is resolved, regardless of the standard retention period."
        }
      ]
    },
    {
      "node_id": "4_2",
      "title": "4.2 Data Deletion",
      "role": "subsection",
      "appearance_order": 2,
      "parent_id": "4",
      "children": [],
      "info_units": [
        {
          "title": "Deletion Eligibility Condition",
          "order": 0,
          "description": "Records may be deleted only after the retention period has elapsed, and only if a deletion log entry is created at the time of deletion."
        }
      ]
    }
  ]
}
```

### Example 2 — Continuation-only page

Context — `<active_hierarchy>`: node_id="3_1" role="subsection" title="3.1 Data Collection"

CURRENT_PAGE:
  Interviews were conducted using a semi-structured protocol. Each session lasted between
  45 and 90 minutes. All sessions were recorded with participant consent and transcribed verbatim.

  Field notes were taken during each session to capture non-verbal observations.

Correct extraction:
```json
{
  "nodes": [
    {
      "node_id": "4-continuation",
      "title": null,
      "role": "freeform_block",
      "appearance_order": 4,
      "parent_id": "3_1",
      "children": [],
      "info_units": [
        {
          "title": "Interview Protocol Details",
          "order": 0,
          "description": "Interviews used a semi-structured protocol lasting 45–90 minutes per session; all sessions were recorded with participant consent and transcribed verbatim."
        },
        {
          "title": "Field Notes Practice",
          "order": 1,
          "description": "Field notes were taken during each session to capture non-verbal observations."
        }
      ]
    }
  ]
}
```

Return ONLY the JSON DocumentStructure object. Respond directly. Start your response with `{`.
"""


def build_extraction_prompt(theme_section: str) -> str:
    """Assemble the OpenAI reasoning models extraction system prompt with the dynamic theme section."""
    return _EXTRACTION_PROMPT_PREFIX + "\n\n" + theme_section + "\n\n" + _EXTRACTION_PROMPT_SUFFIX_GPT_REASONING


# Legacy constant for tests
DOCUMENT_STRUCTURE_EXTRACTION_PROMPT = build_extraction_prompt("")
