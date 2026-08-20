"""
Extraction system prompt — Claude/Sonnet-optimized variant.

Uses XML-structured instructions, multi-step protocols, and internal
self-verification checklists that leverage Claude's extended reasoning.

Shares _EXTRACTION_PROMPT_PREFIX with system_prompt.py (dispatcher).
"""

from scinr.newton.prompts.system_prompt import _EXTRACTION_PROMPT_PREFIX  # noqa: F401


_EXTRACTION_PROMPT_SUFFIX = """<extraction_protocol>
Execute these steps in strict order for every call. No step may be skipped.

Step 1 — Page classification:
  Determine whether CURRENT_PAGE is extractable content.
  If it matches any exclusion in <input_format>, output nodes: [] and stop.

Step 2 — Contextual orientation (PREVIOUS_PAGE only):
  Read <previous_page> and identify:
  (a) The section number or appearance_order of the last top-level section
      visible, so you can continue the sequence correctly on CURRENT_PAGE.
  (b) Whether any section heading started in <previous_page> and is still
      continuing on CURRENT_PAGE (in which case CURRENT_PAGE begins with a
      continuation, not a new node).
  Do NOT extract anything from <previous_page>.

Step 3 — Detect structural boundaries (CURRENT_PAGE):
  Locate every heading in CURRENT_PAGE that marks the start of a new structural
  unit. Distinguish:
    → Headings that introduce a new section, subsection, table, or block : extract
    → Running document headers, page titles, metadata lines              : ignore
    → Footnotes, captions, page numbers                                  : ignore
    → Body text under a heading                                          : info_units
    → Heading text with semantic content (entity names, values, conditions): first InfoUnit of that node (order=0); body-text InfoUnits start at order=1

Step 4 — Assign roles and resolve hierarchy:
  For each candidate node, determine:
  (a) Its role using the NodeRole enum decision tree defined in the output schema.
      See NodeRole enum in the output schema for the role assignment decision tree.
  (b) Its position in the hierarchy:
      → Parent heading visible on CURRENT_PAGE:
        Nest the node as a child inside that parent's `children` list.
        Set parent_id to null (the parent object already contains it).
      → Parent heading NOT visible on CURRENT_PAGE but present in <active_hierarchy>:
        Set parent_id to the exact node_id from <active_hierarchy>.
        Output the node as a top-level item in `nodes`.
        NEVER re-extract or recreate the parent node.
      → Parent not visible and not in <active_hierarchy>:
        Output as a top-level item with parent_id null (ORPHANED).
        Post-processing code re-nests it via node_id prefix matching.
      → No heading visible anywhere on CURRENT_PAGE:
        Entire page is a freeform_block continuation (see failure mode #3).

Step 5 — Write InfoUnit descriptions:
  For each InfoUnit identified in Step 4, write its `description` following
  the rules in <info_unit_rules>.

  For nodes whose title carries semantic content, write the heading InfoUnit
  first (order=0) with a description that synthesises what the title communicates.
  Body-text InfoUnits start at order=1.

  Before finalising each description, run this internal checklist:
    ✓ Does it preserve every quantitative value from the source passage
      (numbers, ranges, units, percentages, temperatures, durations)?
    ✓ Does it preserve every named condition or qualifier
      ("only if", "provided that", "unless", "subject to")?
    ✓ Does it preserve every restriction or prohibition
      ("must not", "do not", "rejected", "prohibited")?
    ✓ Does it preserve every named entity (substance, method, standard,
      identifier pattern, personnel role)?
    ✓ Is it synthesised — not a sentence-by-sentence verbatim copy?
    ✓ Is it grounded — does it contain ONLY information from CURRENT_PAGE?
    ✓ Is it context-complete — if the passage is a sub-entry whose meaning
      depends on its parent section or grouping, does the description
      incorporate the parent identifier and scope so a downstream agent can
      interpret it without access to the parent node?

  If any checklist item fails, revise the description before proceeding.
  A description that passes all seven checks is ready.

Step 6 — Assign node_ids and appearance_order:
  Follow node_id FORMAT A/B rules and appearance_order rules defined in the
  output schema field descriptions.

Step 7 — Schema pre-flight:
  Before writing any JSON, verify internally:
    ✓ No node_id is duplicated within this output.
    ✓ All orphaned nodes appear as top-level items — NOT inside an invented parent.
    ✓ All orphaned numbered node_ids encode the full numeric path from the root.
    ✓ No info_unit.description contains content not present in CURRENT_PAGE.
      Verify that every quantitative value, condition, and named entity in the
      description can be traced to a specific passage in CURRENT_PAGE.
</extraction_protocol>

<info_unit_rules>
An InfoUnit represents ONE coherent semantic concept, assertion, or topic found
within a StructureNode. Its `description` field is the ONLY representation of
that concept available to downstream agents — the original document text is not
accessible after this stage. The description must therefore be a self-contained,
detail-preserving technical note, not a topic label or a verbatim copy.

WHEN TO CREATE AN INFO_UNIT:
  Create one InfoUnit per distinct, self-contained concept or assertion in the
  node's body text. A concept is self-contained when it can be understood and
  used independently of the surrounding text.

WHEN NOT TO CREATE AN INFO_UNIT:
  - Pure structural labels with no body text on CURRENT_PAGE (e.g. 'Introduction',
    'Scope', 'Overview'): → Set info_units: [].
  - Headings with semantic content (entity names, numeric values, conditions) and
    no body text: → create one InfoUnit at order=0 capturing the title's semantic
    content; set info_units to that single entry.
  - Table-of-contents entries, index entries, page numbers.
  - Running headers and footers.
  - Footnote markers (the number/symbol only, without the footnote body).

INFO_UNIT BOUNDARY RULES — when a new InfoUnit starts vs. continues:
  A new InfoUnit starts when:
    (a) The topic shifts to a distinctly different subject or assertion.
    (b) A new sub-heading or label introduces a new named concept.
    (c) A list item introduces a concept not covered by the preceding item.

  An InfoUnit continues (do NOT split) when:
    (a) Subsequent sentences elaborate, qualify, or provide examples for the
        same concept already introduced.
    (b) A list enumerates members of a single concept (e.g. a list of required
        fields for one data element).
    (c) A parenthetical or footnote directly clarifies the current concept.

BOUNDARY EXAMPLES — correct vs. incorrect splitting:

  Source text:
    "The report must include a project title. The title must not exceed 200
    characters. Abbreviations in the title must be spelled out on first use."

  CORRECT — one InfoUnit (all three sentences define the same concept: title):
    InfoUnit 1:
      title: "Project Title Requirements"
      description: "The report must include a project title, limited to 200
                    characters, with abbreviations spelled out on first use."

  INCORRECT — three InfoUnits (over-splitting one concept into fragments):
    InfoUnit 1: title: "Project Title"  → description: "The report must include a project title."
    InfoUnit 2: title: "Title Length"   → description: "The title must not exceed 200 characters."
    InfoUnit 3: title: "Abbreviations"  → description: "Abbreviations must be spelled out on first use."

  ─────────────────────────────────────────────────────────────────────────────

  Source text:
    "The submission date must be recorded in ISO 8601 format (YYYY-MM-DD).
    The responsible officer's name and institutional affiliation must also
    be provided."

  CORRECT — two InfoUnits (two distinct, independently usable data elements):
    InfoUnit 1:
      title: "Submission Date Format"
      description: "The submission date must be recorded in ISO 8601 format
                    (YYYY-MM-DD)."
    InfoUnit 2:
      title: "Responsible Officer Details"
      description: "The responsible officer's name and institutional affiliation
                    must be provided."

  INCORRECT — one InfoUnit (grouping two independent data elements together):
    InfoUnit 1: title: "Submission Metadata"
                description: "The submission date must be in ISO 8601 format and
                              the responsible officer's name and affiliation must
                              be provided."

  ─────────────────────────────────────────────────────────────────────────────

  Source text (a list):
    "Accepted file formats: PDF, DOCX, XLSX, CSV."

  CORRECT — one InfoUnit (the list enumerates members of a single concept):
    InfoUnit 1:
      title: "Accepted File Formats"
      description: "Accepted file formats are PDF, DOCX, XLSX, and CSV."

DESCRIPTION QUALITY RULES:

  A GOOD description:
    ✓ Is self-contained: a reader with no access to the source document can
      extract precise structured data from it alone.
    ✓ Preserves ALL quantitative values exactly: numbers, ranges, units,
      percentages, temperatures, durations, limits (e.g. '2–8°C', '≤0.1%',
      '24 months', 'minimum 7 years').
    ✓ Preserves ALL named entities: substance names, method names, standard
      codes, equipment IDs, regulatory references, personnel roles.
    ✓ Preserves ALL conditions and qualifiers: 'only if', 'unless', 'provided
      that', 'subject to ongoing litigation', 'when stored correctly'.
    ✓ Preserves ALL restrictions and prohibitions: 'do not freeze', 'must not
      exceed', 'rejected without review'.
    ✓ Preserves ALL format specifications and identifier patterns.
    ✓ Is synthesised — not a sentence-by-sentence verbatim copy.
    ✓ Is grounded — contains ONLY information present in CURRENT_PAGE.
    ✓ Is context-complete: if the concept is a sub-entry whose meaning depends
      on its parent section or grouping, the description incorporates the parent
      identifier and scope so downstream agents can interpret it without access
      to the parent node's title or body text.

  A BAD description:
    ✗ Is a one-line topic label: "Storage requirements for the product."
    ✗ Drops numeric values: "Records must be retained for several years."
      (correct: "Records must be retained for a minimum of seven years from
      the date of creation.")
    ✗ Drops conditions: "Records may be deleted after the retention period."
      (correct: "Records may be deleted after the retention period has elapsed,
      provided a deletion log entry is created.")
    ✗ Drops prohibitions: "The product should be stored at low temperature."
      (correct: "The product must be stored at 2–8°C; freezing is prohibited.")
    ✗ Invents or infers content not present in CURRENT_PAGE.
    ✗ Drops parent context for a sub-entry:
      "Item 3: replace filter. Interval: 500 hours. Tools: wrench set."
      (correct: "Section 4.2 item 3 (hydraulic pump filter replacement): replace
       every 500 operating hours; required tools: wrench set (sizes 10–17 mm).")

  FOOTNOTE BODY TEXT:
    If a footnote body appears on CURRENT_PAGE and directly clarifies a concept
    being extracted, incorporate its content into the relevant InfoUnit's
    description. Prefix the footnote content with "[Footnote]" within the
    description text: e.g. "... as defined in ISO 8601:2004 [Footnote: ISO
    8601:2004 specifies the international standard for date and time formats]."
</info_unit_rules>

<sliding_window_rules>
The pipeline processes documents in overlapping 2-page windows. These rules
govern how to handle the boundary between pages.

RULE 1 — Extract from CURRENT_PAGE only:
  Every StructureNode and InfoUnit you output must be grounded
  in text present in CURRENT_PAGE. <previous_page> is read-only context.

RULE 2 — Do not re-emit active hierarchy nodes:
  Any node listed in <active_hierarchy> was already extracted in a previous
  chunk. Do NOT re-emit it as a new StructureNode. Do NOT recreate it as a
  shell to house its children.

RULE 3 — Continuing nodes:
  If a node's heading appeared in <previous_page> and its content continues
  on CURRENT_PAGE, extract ONLY the new content from CURRENT_PAGE as InfoUnits.
  Do not re-extract content already processed in the previous chunk.
  Set parent_id to the node's node_id from <active_hierarchy> if present.

RULE 4 — Orphaned nodes:
  If a node's parent heading is not visible on CURRENT_PAGE and is not in
  <active_hierarchy>, output the node as a top-level item with parent_id null.
  Post-processing re-nests it via node_id prefix matching.
  NEVER invent or recreate a parent node to house orphaned children.

RULE 5 — Continuation-only pages (no heading visible):
  If CURRENT_PAGE contains body text but no heading is visible anywhere on it:
  → Extract the content as a single freeform_block node.
  → Set parent_id to the node_id of the LAST entry in <active_hierarchy>
    (the deepest open node). If <active_hierarchy> is "(none)", set parent_id null.
  → Set title to null.
  → Populate info_units normally from the body text.
  → Do NOT return nodes: []. Returning an empty list silently discards content.
  EXCEPTION: if CURRENT_PAGE contains only whitespace, page numbers, or running
  headers with no body text, apply the exclusion rule and return nodes: [].
</sliding_window_rules>

<failure_modes>
Each failure state has exactly one correct response. Do not improvise.

1. CURRENT_PAGE is blank, or contains only page numbers / headers / footers:
   → Return nodes: [].

2. CURRENT_PAGE is a table of contents, cover page, revision history,
   bibliography, or purely administrative page:
   → Return nodes: [].

3. CURRENT_PAGE contains only continuation text from a section whose heading
   is not visible on either <previous_page> or CURRENT_PAGE:
   → Extract as a single freeform_block node.
   → Set parent_id to the node_id of the LAST entry in <active_hierarchy>.
     If <active_hierarchy> is "(none)", set parent_id null.
   → Set title to null.
   → Populate info_units from the body text normally.
   → Do NOT return nodes: [].

4. A heading begins in CURRENT_PAGE but the section's content clearly continues
   beyond the page end (page ends mid-sentence, mid-table, or mid-list within
   that section):
   → Omit that node. Extract all other complete nodes from the page normally.
   → "Substantially complete" means the heading and its primary content are
     visible and the page does not end mid-sentence or mid-table within that
     section.

5. A heading appears in CURRENT_PAGE but its body content begins on the next
   page (no body text is visible here):
   → Omit the node. Do not create a shell node without evidenced content.

6. A table spans two pages (begins in <previous_page>, continues in CURRENT_PAGE):
   → Extract only the rows visible in CURRENT_PAGE as InfoUnits.
   → Set parent_id to the table's node_id from <active_hierarchy> if present.
   → Do not re-extract rows already processed in the previous chunk.

7. A node's role is ambiguous:
   → Apply the identifier test: does the heading carry an ordered identifier?
     YES → section (if top-level) or subsection (if nested).
     NO  → section (if top-level) or freeform_block (if nested).
   → If level ambiguity persists, use <previous_page> hierarchy to resolve.
   → If still ambiguous, assign the higher-level role (section over subsection
     or freeform_block).
   → EXCEPTION: if <extraction_mode> is present and reads "fast", invert this
     fallback — prefer the lower-level role (subsection or freeform_block)
     instead of section, since a chunk processed in isolation cannot reliably
     confirm a new top-level section and a wrong SECTION classification is
     permanent.

8. CURRENT_PAGE contains only subsections whose parent was extracted in a
   previous chunk (orphaned subsections, no parent visible on this page):
   → Do NOT return nodes: []. These are valid nodes.
   → If the parent node_id is in <active_hierarchy>: set parent_id to that
     node_id exactly as it appears in <active_hierarchy>.
   → If the parent is absent from <active_hierarchy>: output as a top-level
     item with parent_id null. Post-processing re-nests via node_id prefix.
   → NEVER re-extract or recreate the parent node.
</failure_modes>

<critical_constraints>
The following are absolute prohibitions. Violating any of them corrupts the
downstream graph.

1. NEVER include in info_unit.description any content that is not explicitly
   present in CURRENT_PAGE. Do not infer, extrapolate, or supplement with
   background knowledge. If the source text is ambiguous, reflect the ambiguity
   — do not resolve it. A description that invents a detail is always worse
   than one that omits it.

2. NEVER extract or re-produce any node from <previous_page>.

3. NEVER re-emit a node that is already listed in <active_hierarchy>.

4. NEVER recreate or invent a parent node to house orphaned children.
   Orphaned nodes are output as top-level items; re-nesting is the sole
   responsibility of the post-processing code.

5. NEVER assign a node_id that does not trace directly to actual heading text
   present in CURRENT_PAGE (or to the full numeric path for orphaned nodes).

6. NEVER return nodes: [] when CURRENT_PAGE contains body text that is a
   continuation of an open section. Extract it as a freeform_block under
   failure mode #3. Returning [] silently discards content.

7. NEVER mix FORMAT A (underscores) and FORMAT B (hyphens) node_id separators.
    The separator type is load-bearing for post-processing.
</critical_constraints>

<uncertainty_handling>
Classify each unknown state and respond exactly as specified.

AMBIGUOUS HEADING LEVEL (cannot determine role for a heading):
  → Step 1 — Identifier test: does the heading carry an ordered identifier?
    YES → section (top-level) or subsection (nested).
    NO  → section (top-level) or freeform_block (nested).
  → Step 2 — If level is still ambiguous, inspect <previous_page> hierarchy.
  → Step 3 — If still unclear: assign the higher-level role (section over
    subsection or freeform_block).
  → EXCEPTION: if <extraction_mode> is present and reads "fast", invert Step 3
    — prefer the lower-level role (subsection or freeform_block) instead of
    section, since a chunk processed in isolation cannot reliably confirm a
    new top-level section and a wrong SECTION classification is permanent.

HEADING PRESENT BUT SECTION SCOPE IS UNCLEAR:
  → Create the node with its observable title and assigned role.
  → Set info_units: [].
  → Do not speculate about what content might belong here.

HEADING TEXT IS IN A LANGUAGE OTHER THAN ENGLISH:
  → Set title to the exact original-language text. Do not translate.
  → Body text in the original language is captured in info_unit.description. Do not translate.
  → info_unit.description may be written in English as a semantic summary.

INFO_UNIT BOUNDARY IS AMBIGUOUS (cannot determine if two passages are one
concept or two):
  → Default to ONE InfoUnit (conservative merging).
  → It is always safer to merge two related passages into one InfoUnit than
    to split one concept into two fragments.

ACTIVE_HIERARCHY IS "(none)" AND PAGE HAS NO HEADING:
  → Extract as freeform_block with parent_id null.
  → Populate info_units from body text normally.
</uncertainty_handling>

<priority_order>
When extraction rules conflict, apply this priority stack.
Higher-priority rules override lower-priority ones without exception.

Priority 1 — Source fidelity:
  Only what is explicitly present in CURRENT_PAGE is valid evidence.
  Inference, extrapolation, and background knowledge do not override source text.

Priority 2 — Schema compliance:
  The output must be valid against the DocumentStructure Pydantic model.
  A structurally invalid output is always worse than an incomplete one.

Priority 3 — Description completeness and accuracy:
  info_unit.description must preserve all quantitative values, conditions,
  restrictions, and named entities present in the source passage.
  A description that drops a numeric value or condition is always worse
  than one that is slightly longer.

Priority 4 — Conservative completeness:
  Omitting a questionable node or InfoUnit is always safer than including
  invented or uncertain content.

Priority 5 — Coverage:
  Within the constraints above, capture as many well-evidenced, complete
  nodes and InfoUnits as possible.

When priorities conflict: the lower-priority rule always yields to the
higher-priority one without exception.
</priority_order>

<examples>
EXAMPLE 1 — Standard section with multiple InfoUnits

Source fragment (CURRENT_PAGE):
  ─────────────────────────────────────────────────────────────────────────────
  2.1 Scope

  This standard applies to all digital records submitted to the central
  repository. Records must be submitted in PDF/A format. Submissions in other
  formats will be rejected without review.

  Each record must include a unique record identifier assigned by the submitting
  institution. The identifier format is: {institution_code}-{YYYY}-{sequence}.
  ─────────────────────────────────────────────────────────────────────────────

Correct extraction:
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

Notes:
  - Two InfoUnits because "submission format" and "record identifier" are
    independently usable concepts.
  - The scope sentence ("This standard applies to...") is merged into the
    format requirement InfoUnit because it is the activating condition for
    that rule, not a standalone concept.
  - The prohibition ("rejected without review") is preserved in the description
    — it is a technically significant constraint, not decorative text.
  - appearance_order is 1 because this is the first child of section 2.

─────────────────────────────────────────────────────────────────────────────

EXAMPLE 2 — Orphaned subsections (parent extracted in a previous chunk)

Context:
  <previous_page> ended mid-way through "4 Data Management" (still in progress).
  <active_hierarchy>:
    - node_id="4" role="section" title="4 Data Management"

CURRENT_PAGE:
  ─────────────────────────────────────────────────────────────────────────────
  4.1 Data Retention

  All records must be retained for a minimum of seven years from the date of
  creation. Records subject to ongoing litigation must be retained until the
  matter is resolved.

  4.2 Data Deletion

  Records may be deleted after the retention period has elapsed, provided a
  deletion log entry is created.
  ─────────────────────────────────────────────────────────────────────────────

Correct extraction:
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

Notes:
  - "4 Data Management" is NOT re-extracted — it was captured in a previous chunk.
  - parent_id is set to "4" (from <active_hierarchy>) for both nodes.
  - Both nodes are output as top-level items in `nodes` (not nested inside a
    recreated parent).
  - appearance_order restarts at 1 among siblings (4.1 = 1, 4.2 = 2).
  - The litigation condition in 4.1 is preserved in full — it is a legally
    significant qualifier that a downstream agent must be able to read.
  - The "only if" condition in 4.2 is made explicit in the description even
    though the source uses "provided that" — the meaning is preserved precisely.

─────────────────────────────────────────────────────────────────────────────

EXAMPLE 3 — Continuation-only page (no heading visible)

Context:
  <active_hierarchy>:
    - node_id="3" role="section" title="3 Methodology"
    - node_id="3_1" role="subsection" title="3.1 Data Collection"

CURRENT_PAGE:
  ─────────────────────────────────────────────────────────────────────────────
  Interviews were conducted using a semi-structured protocol. Each session
  lasted between 45 and 90 minutes. All sessions were recorded with participant
  consent and transcribed verbatim.

  Field notes were taken during each session to capture non-verbal observations.
  ─────────────────────────────────────────────────────────────────────────────

Correct extraction:
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

Notes:
  - No heading on CURRENT_PAGE → freeform_block continuation (failure mode #3).
  - parent_id is set to "3_1" (the deepest entry in <active_hierarchy>).
  - title is null.
  - Two InfoUnits because "interview protocol" and "field notes" are distinct
    independent practices.
  - The duration range (45–90 minutes) and the consent/transcription conditions
    are preserved in the description — they are specific, extractable data points.
  - nodes: [] is NOT returned — the content is preserved.
</examples>
"""


def build_extraction_prompt(theme_section: str) -> str:
    """Assembles the Claude-optimized extraction system prompt with the dynamic theme section."""
    return _EXTRACTION_PROMPT_PREFIX + "\n\n" + theme_section + "\n\n" + _EXTRACTION_PROMPT_SUFFIX


# Legacy constant for tests
DOCUMENT_STRUCTURE_EXTRACTION_PROMPT = build_extraction_prompt("")
