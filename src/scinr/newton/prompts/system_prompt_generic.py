"""
Extraction system prompt — generic/universal variant.

Compatible with any LLM family (Kimi1.5, GLM-5, GPT-4, Claude, Mistral, etc.).
Shares _EXTRACTION_PROMPT_PREFIX with system_prompt.py (shared, XML-tag-free prose).

The sliding-window content (<previous_page>, <page_1>...<page_N>, <active_hierarchy>)
is injected as the human turn at call time.
"""

from scinr.newton.prompts.system_prompt import _EXTRACTION_PROMPT_PREFIX  # shared prefix


_EXTRACTION_PROMPT_SUFFIX_GENERIC = """
## Extraction Protocol

Execute the following steps in strict order for every call. No step may be skipped.

**Step 1 — Classify the page.**
Determine whether CURRENT_PAGE contains extractable content.
Return `{"nodes": []}` immediately and stop if the page is:
- Blank, or contains only whitespace, page numbers, running headers, or footers.
- A table of contents, cover page, revision history, bibliography, or purely administrative page.

**Step 2 — Read <previous_page> for context only.**
Identify: (a) the section number or appearance_order of the last top-level section visible,
so you can continue the sequence correctly on CURRENT_PAGE; (b) whether any section heading
started in <previous_page> and is still continuing on CURRENT_PAGE (meaning CURRENT_PAGE
begins with a continuation, not a new node).
Do NOT extract any node or InfoUnit from <previous_page>.

**Step 3 — Detect structural boundaries on CURRENT_PAGE.**
Locate every heading that marks the start of a new structural unit. Classify each candidate:
- Headings that introduce a new section, subsection, table, or block → extract as nodes.
- Running document headers, page titles, metadata lines → ignore entirely.
- Footnotes, captions, page numbers → ignore entirely.
- Body text under a heading → becomes InfoUnits of that heading's node.
- Heading text with semantic content (entity names, values, conditions) → becomes the first InfoUnit of that node (order=0); body-text InfoUnits start at order=1.

**Step 4 — Assign roles and resolve hierarchy.**
For each candidate node, determine its role using the NodeRole decision tree:
- `section`: top-level named division, carries an ordered identifier or stands alone at the top level.
- `subsection`: named division nested inside a section; carries a sub-identifier (e.g. 2.1).
- `table`: tabular data block introduced by a table heading or caption.
- `field_group`: structured list of labelled fields (form-like layout).
- `freeform_block`: narrative prose with no heading identifier; used for continuation pages.
- `appendix`: appendix or annex; treat identically to section for matching purposes.

Resolve the parent relationship using this priority order:
1. Parent heading visible on CURRENT_PAGE → nest the node inside that parent's `children` list;
   set `parent_id` to null (the parent object already contains it).
2. Parent heading NOT on CURRENT_PAGE but present in <active_hierarchy> → output the node as a
   top-level item in `nodes`; set `parent_id` to the exact `node_id` from <active_hierarchy>.
   Never re-extract or recreate the parent node.
3. Parent not visible and not in <active_hierarchy> → output as a top-level item with
   `parent_id: null` (ORPHANED). Post-processing re-nests it via node_id prefix matching.
   Never invent or recreate a parent node to house orphaned children.
4. No heading visible anywhere on CURRENT_PAGE → entire page is a freeform_block continuation
   (see Failure Mode 3 below).

**Step 5 — Write InfoUnit descriptions.**
For each InfoUnit, write a `description` that is a self-contained, detail-preserving technical note.
For nodes whose title carries semantic content, write the heading InfoUnit first (order=0)
with a description that synthesises what the title communicates.
The description must preserve ALL of the following from the source passage:
- Every quantitative value exactly: numbers, ranges, units, percentages, temperatures, durations,
  limits (e.g. "2–8°C", "≤0.1%", "24 months", "minimum 7 years").
- Every named entity: substance names, method names, standard codes, equipment IDs, regulatory
  references, personnel roles.
- Every condition and qualifier: "only if", "unless", "provided that", "subject to ongoing
  litigation", "when stored correctly".
- Every restriction and prohibition: "do not freeze", "must not exceed", "rejected without review".
- Every format specification and identifier pattern.

The description must be synthesised (not a verbatim sentence-by-sentence copy) and grounded
(contains ONLY information present in CURRENT_PAGE — no inference, no background knowledge).

A complete description includes all numeric values, conditions and qualifiers, restrictions
and prohibitions, and named entities from the source passage. It is synthesised (not a
verbatim sentence-by-sentence copy) and grounded exclusively in CURRENT_PAGE content.

- Parent-context completeness: when a passage is a sub-entry whose meaning depends on
  its parent section or grouping, incorporate the parent identifier and scope in the
  description so it is independently interpretable. Downstream agents access no other
  representation — not the parent node's title, not its body text.

  BAD:  "Item 3: replace filter. Interval: 500 hours. Tools: wrench set."
        (Uninterpretable without knowing which assembly or subsystem item 3 belongs to.)
  GOOD: "Section 4.2 item 3 (hydraulic pump filter replacement): replace every
         500 operating hours; required tools: wrench set (sizes 10–17 mm)."
        (Self-contained: parent section and subject matter are incorporated.)

**Step 6 — Assign node_ids and appearance_order.**
Follow the node_id FORMAT A/B rules and appearance_order rules defined in the output schema
field descriptions. Never mix FORMAT A (underscores) and FORMAT B (hyphens) separators within
a single output — the separator type is load-bearing for post-processing.

**Step 7 — Schema pre-flight.**
The output has unique node_ids throughout. Orphaned nodes appear as top-level items with the
full numeric path encoded in their node_id. Every description is grounded exclusively in
CURRENT_PAGE content.


## InfoUnit Rules

An InfoUnit represents ONE coherent semantic concept, assertion, or topic found within a
StructureNode. Its `description` is the ONLY representation of that concept available to
downstream agents — the original document text is not accessible after this stage.

**When to create an InfoUnit:**
Create one InfoUnit per distinct, self-contained concept or assertion in the node's body text.
A concept is self-contained when it can be understood and used independently of surrounding text.

**When NOT to create an InfoUnit:**
- Pure structural labels with no body text on CURRENT_PAGE (e.g. 'Introduction', 'Scope',
  'Overview') → set `info_units: []`.
- Headings with semantic content (entity names, values, conditions) and no body text →
  create one InfoUnit at order=0 capturing the title's semantic content.
- Table-of-contents entries, index entries, page numbers.
- Running headers and footers.
- Footnote markers (the number/symbol only, without the footnote body).

**InfoUnit boundary rules — when a new InfoUnit starts vs. continues:**

A NEW InfoUnit starts when:
- The topic shifts to a distinctly different subject or assertion.
- A new sub-heading or label introduces a new named concept.
- A list item introduces a concept not covered by the preceding item.

An InfoUnit CONTINUES (do NOT split) when:
- Subsequent sentences elaborate, qualify, or provide examples for the same concept.
- A list enumerates members of a single concept (e.g. a list of required fields for one data element).
- A parenthetical or footnote directly clarifies the current concept.

**Boundary examples:**

Source: "The report must include a project title. The title must not exceed 200 characters.
Abbreviations in the title must be spelled out on first use."

CORRECT — one InfoUnit (all three sentences define the same concept: title requirements):
  title: "Project Title Requirements"
  description: "The report must include a project title, limited to 200 characters,
                with abbreviations spelled out on first use."

INCORRECT — three InfoUnits (over-splitting one concept into fragments):
  InfoUnit 1: "Project Title" → "The report must include a project title."
  InfoUnit 2: "Title Length" → "The title must not exceed 200 characters."
  InfoUnit 3: "Abbreviations" → "Abbreviations must be spelled out on first use."

---

Source: "The submission date must be recorded in ISO 8601 format (YYYY-MM-DD).
The responsible officer's name and institutional affiliation must also be provided."

CORRECT — two InfoUnits (two distinct, independently usable data elements):
  InfoUnit 1 — title: "Submission Date Format"
    description: "The submission date must be recorded in ISO 8601 format (YYYY-MM-DD)."
  InfoUnit 2 — title: "Responsible Officer Details"
    description: "The responsible officer's name and institutional affiliation must be provided."

INCORRECT — one InfoUnit (grouping two independent data elements together):
  title: "Submission Metadata"
  description: "The submission date must be in ISO 8601 format and the responsible
                officer's name and affiliation must be provided."

---

Source (a list): "Accepted file formats: PDF, DOCX, XLSX, CSV."

CORRECT — one InfoUnit (the list enumerates members of a single concept):
  title: "Accepted File Formats"
  description: "Accepted file formats are PDF, DOCX, XLSX, and CSV."

**Footnote body text:**
If a footnote body appears on CURRENT_PAGE and directly clarifies a concept being extracted,
incorporate its content into the relevant InfoUnit's description. Prefix the footnote content
with "[Footnote]" within the description text.


## Sliding Window Rules

The pipeline processes documents in overlapping 2-page windows.

1. Extract from CURRENT_PAGE only. Every StructureNode and InfoUnit must be grounded in text
   present in CURRENT_PAGE. <previous_page> is read-only context.

2. Do not re-emit active hierarchy nodes. Any node listed in <active_hierarchy> was already
   extracted in a previous chunk. Do NOT re-emit it as a new StructureNode. Do NOT recreate
   it as a shell to house its children.

3. Continuing nodes: if a node's heading appeared in <previous_page> and its content continues
   on CURRENT_PAGE, extract ONLY the new content from CURRENT_PAGE as InfoUnits. Set parent_id
   to the node's node_id from <active_hierarchy> if present.

4. Orphaned nodes: if a node's parent heading is not visible on CURRENT_PAGE and is not in
   <active_hierarchy>, output the node as a top-level item with parent_id null. Post-processing
   re-nests it via node_id prefix matching. Never invent or recreate a parent node.

5. Continuation-only pages (no heading visible anywhere on CURRENT_PAGE): extract the content
   as a single freeform_block node. Set parent_id to the node_id of the LAST entry in
   <active_hierarchy> (the deepest open node); if <active_hierarchy> is "(none)", set
   parent_id null. Set title to null. Populate info_units normally from the body text.
   Do NOT return `nodes: []` — returning an empty list silently discards content.
   Exception: if CURRENT_PAGE contains only whitespace, page numbers, or running headers
   with no body text, apply Step 1 and return `nodes: []`.


## Failure Modes

Each failure state has exactly one correct response.

**Failure 1 — Blank or header-only page:**
CURRENT_PAGE is blank, or contains only page numbers, running headers, or footers.
→ Return `{"nodes": []}`.

**Failure 2 — Administrative page:**
CURRENT_PAGE is a table of contents, cover page, revision history, bibliography, or purely
administrative page.
→ Return `{"nodes": []}`.

**Failure 3 — Continuation-only page (no heading visible):**
CURRENT_PAGE contains only continuation text from a section whose heading is not visible on
either <previous_page> or CURRENT_PAGE.
→ Extract as a single freeform_block node.
→ Set parent_id to the node_id of the LAST entry in <active_hierarchy>; if "(none)", set null.
→ Set title to null.
→ Populate info_units from the body text normally.
→ Do NOT return `nodes: []`.

**Failure 4 — Section continues beyond page end:**
A heading begins in CURRENT_PAGE but the section's content clearly continues beyond the page
end (page ends mid-sentence, mid-table, or mid-list within that section).
→ Omit that node. Extract all other complete nodes from the page normally.
"Substantially complete" means the heading and its primary content are visible and the page
does not end mid-sentence or mid-table within that section.

**Failure 5 — Heading with no body text on this page:**
A heading appears in CURRENT_PAGE but its body content begins on the next page (no body text
is visible here).
→ Omit the node. Do not create a shell node without evidenced content.

**Failure 6 — Table spanning two pages:**
A table begins in <previous_page> and continues in CURRENT_PAGE.
→ Extract only the rows visible in CURRENT_PAGE as InfoUnits.
→ Set parent_id to the table's node_id from <active_hierarchy> if present.
→ Do not re-extract rows already processed in the previous chunk.

**Failure 7 — Ambiguous node role:**
→ Apply the identifier test: does the heading carry an ordered identifier?
  YES → section (if top-level) or subsection (if nested).
  NO  → section (if top-level) or freeform_block (if nested).
→ If level ambiguity persists, use <previous_page> hierarchy to resolve.
→ If still ambiguous, assign the higher-level role (section over subsection or freeform_block).

**Failure 8 — Orphaned subsections (parent extracted in a previous chunk):**
CURRENT_PAGE contains only subsections whose parent was extracted in a previous chunk.
→ Do NOT return `nodes: []`. These are valid nodes.
→ If the parent node_id is in <active_hierarchy>: set parent_id to that node_id exactly.
→ If the parent is absent from <active_hierarchy>: output as a top-level item with parent_id null.
→ Never re-extract or recreate the parent node.


## Output Integrity

All output must satisfy these four rules. Each prevents a specific type of corruption in the
downstream graph.

1. info_unit.description contains only information explicitly stated in CURRENT_PAGE. Reflect
   source ambiguity as-is — a description that omits a detail is always better than one that
   adds an invented one.

2. Every node_id traces directly to actual heading text in CURRENT_PAGE, or to the full numeric
   path for orphaned nodes (e.g. subsection 2.1.3 whose parent was extracted previously →
   node_id "2_1_3").

3. A non-empty CURRENT_PAGE always produces at least one node. Pages with body text but no
   visible heading become a freeform_block under Failure Mode 3.

4. Every node_id in a single output uses either FORMAT A separators (underscores) or FORMAT B
   separators (hyphens) — never both. The separator type is load-bearing for post-processing.

When rules conflict, apply this priority: (1) source fidelity — only what is explicitly present
in CURRENT_PAGE is valid evidence; (2) schema compliance — a structurally invalid output is
always worse than an incomplete one; (3) description completeness — preserve all quantitative
values, conditions, restrictions, and named entities; (4) conservative completeness — omitting
a questionable node is safer than including invented content; (5) coverage — within the above
constraints, capture as many well-evidenced nodes and InfoUnits as possible.


## Examples

### Example 1 — Standard section with multiple InfoUnits

Source fragment (CURRENT_PAGE):
  2.1 Scope

  This standard applies to all digital records submitted to the central repository.
  Records must be submitted in PDF/A format. Submissions in other formats will be
  rejected without review.

  Each record must include a unique record identifier assigned by the submitting
  institution. The identifier format is: {institution_code}-{YYYY}-{sequence}.

Correct extraction:
```json
{
  "nodes": [
    {
      "node_id": "2_1",
      "title": "2.1 Scope",
      "role": "subsection",
      "appearance_order": 1,
      "parent_id": null,
      "children": [],
      "info_units": [
        {
          "title": "Submission Format Requirement",
          "order": 0,
          "description": "All digital records submitted to the central repository must be in PDF/A format; submissions in any other format are rejected without review."
        },
        {
          "title": "Record Identifier Format",
          "order": 1,
          "description": "Each record must include a unique identifier assigned by the submitting institution, following the pattern {institution_code}-{YYYY}-{sequence}."
        }
      ]
    }
  ]
}
```

Notes:
- Two InfoUnits because "submission format" and "record identifier" are independently usable concepts.
- The scope sentence is merged into the format requirement InfoUnit because it is the activating
  condition for that rule, not a standalone concept.
- The prohibition ("rejected without review") is preserved — it is a technically significant
  constraint, not decorative text.
- appearance_order is 1 because this is the first child of section 2.

---

### Example 2 — Orphaned subsections (parent extracted in a previous chunk)

Context:
  <previous_page> ended mid-way through "4 Data Management" (still in progress).
  <active_hierarchy>:
    - node_id="4" role="section" title="4 Data Management"

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

Notes:
- "4 Data Management" is NOT re-extracted — it was captured in a previous chunk.
- parent_id is set to "4" (from <active_hierarchy>) for both nodes.
- Both nodes are output as top-level items in `nodes` (not nested inside a recreated parent).
- appearance_order restarts at 1 among siblings (4.1 = 1, 4.2 = 2).
- The litigation condition in 4.1 is preserved in full — it is a legally significant qualifier.
- The "only if" condition in 4.2 is made explicit even though the source uses "provided that".

---

### Example 3 — Continuation-only page (no heading visible)

Context:
  <active_hierarchy>:
    - node_id="3" role="section" title="3 Methodology"
    - node_id="3_1" role="subsection" title="3.1 Data Collection"

CURRENT_PAGE:
  Interviews were conducted using a semi-structured protocol. Each session lasted between
  45 and 90 minutes. All sessions were recorded with participant consent and transcribed
  verbatim.

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
          "description": "Interviews used a semi-structured protocol. Each session lasted between 45 and 90 minutes. All sessions were recorded with participant consent and transcribed verbatim."
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

Notes:
- No heading on CURRENT_PAGE → freeform_block continuation (Failure Mode 3).
- parent_id is set to "3_1" (the deepest entry in <active_hierarchy>).
- title is null.
- Two InfoUnits because "interview protocol" and "field notes" are distinct independent practices.
- The duration range (45–90 minutes) and the consent/transcription conditions are preserved.
- `nodes: []` is NOT returned — the content is preserved.

Return ONLY the JSON DocumentStructure object. Do not add explanatory text before or after it.
"""


def build_extraction_prompt(theme_section: str) -> str:
    """Assembles the generic extraction system prompt with the dynamic theme section.

    Uses the shared PREFIX from system_prompt.py and the generic SUFFIX.
    Call once at pipeline startup.

    Args:
        theme_section: XML block from theme_registry.build_theme_section_for_extraction_prompt()

    Returns:
        Complete system prompt string ready for use as SystemMessage content.
    """
    return _EXTRACTION_PROMPT_PREFIX + "\n\n" + theme_section + "\n\n" + _EXTRACTION_PROMPT_SUFFIX_GENERIC


# Legacy constant for tests and code that hasn't been updated yet.
DOCUMENT_STRUCTURE_EXTRACTION_PROMPT = build_extraction_prompt("")
