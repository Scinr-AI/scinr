"""
Annotation agent prompts — generic/universal variant.

Compatible with any LLM family (Kimi1.5, GLM-5, GPT-4, Claude, Mistral, etc.).
Preserves all business logic from prompts.py (Claude variant).

LangGraph pipeline: read_theme → decide_model

Prompts:
  THEME_CLASSIFICATION_PROMPT_TEMPLATE  — determines thematic domain of a node
  DECISION_PROMPT_TEMPLATE              — selects best Pydantic model for a node

Builder functions:
  build_theme_classification_prompt(themes_block, document_name, theme_histogram) -> str
  build_annotation_decision_prompt(theme_node) -> str
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeNode


# ─── Theme Classification Prompt ─────────────────────────────────────────────

THEME_CLASSIFICATION_PROMPT_TEMPLATE = """You are a document theme classifier. Your function is to determine which thematic domain a document section belongs to, based on its content.

## Available Themes

{themes_block}

{document_context}
## Task

Analyze the document section and classify it into the MOST SPECIFIC available theme.

Rules:
1. The theme field must exactly match one of the theme paths listed above.
2. Choose the most specific (deepest) path that applies.
3. Use 'default' only if no other theme clearly fits the content.
4. The justification must reference specific content from the node (title, role, or information units) — not just restate the theme name.

Return ONLY the JSON object with fields "theme" and "justification". Do not add explanatory text.
"""


# ─── Annotation Decision Prompt ───────────────────────────────────────────────

DECISION_PROMPT_TEMPLATE = """You are a structured document content-matching specialist. Your function is to examine the actual content of a StructureNode — as represented by its node_id, title, and InfoUnits — and determine which Pydantic extraction model from a predefined catalog best fits what is actually described in that node.

You operate inside a production graph-annotation pipeline. The result of your decision is written to Neo4j as a HAS_MODEL_DECISION relationship on the evaluated StructureNode. A wrong model assignment will cause downstream extraction pipelines to apply the wrong schema to real document content, producing structurally invalid or semantically incorrect output.

## Critical Rules

Rule 1 — CONTENT-FIRST MATCHING:
You are reading what IS present in a document, not what a guideline says MUST be present.
Your task is to match the actual content to the model that best captures it — not to evaluate
compliance or completeness. Never penalise a node for missing content. Match what is there.

Rule 2 — CONSERVATIVE MATCHING:
If no catalog model covers >= 25% of the node's actual content, set matched_model_class = null
and propose a new schema. null is always safer than a wrong model. Never select a model because
its name resembles the node title. Never select a model because it is the "closest" when
coverage is below 25%.

Rule 3 — GROUND IN NODE CONTENT:
Reason from the node_id, title, and InfoUnits provided — these together are the only
representation of a node's content available at this stage; the original document text
is not accessible. The InfoUnits are the primary source; the node_id and title are
complementary signals (e.g. a CTD section code or domain-specific terminology in the
title may confirm or disambiguate what the InfoUnits describe).
Do not invent content that is absent from all three sources.
If the context is sparse, reflect that with a lower coverage score.

Rule 4 — AGGREGATE MODEL PROHIBITION:
Some catalog models are aggregate/container models — they represent an entire document or
a major top-level section. Their fields are too coarse to capture the content of any
individual StructureNode and will always produce semantically invalid extraction output.
Select only models whose individual fields map to the concepts present in the node's
InfoUnits. The catalog marks aggregate models with [list container].


## Available Models

The following Pydantic extraction models are available. Each entry shows the class name,
a summary of what it captures, and its field names. Models marked [list container] are
aggregate containers — never select them for individual StructureNodes.

{catalog_block}

Important: any model marked [list container] in the catalog above is a top-level container.
Never select it for individual StructureNodes with their own InfoUnits.


## Decision Protocol

Execute these steps in strict order. Do not skip any step.

Step 1 — Understand the node structure.
Read the node's id, title, and role carefully.
Role values and their implications:
  section / subsection: a named section — may contain mixed content.
  freeform_block: narrative prose — usually maps to a single concept.
  table: tabular data — match based on what the table contains, NOT its role label.
  field_group: structured field list — treat like a table for matching purposes.
  appendix: treat identically to section for matching purposes.
Note the depth of the node in the hierarchy. Deeper nodes tend to be more specific and
should match more specific models.

Step 2 — Analyse the node_id, title, and InfoUnits: what semantic concepts are present?
First, note any domain signals in the node_id and title (e.g. CTD section codes,
content-type keywords). Then, for each InfoUnit: (a) read its title and description;
(b) identify the domain concept it represents (e.g. an identifier, a process description,
a measurement record, a classification entry, a test result); (c) list all distinct domain
concepts present across the node_id, title, and all InfoUnits combined.
This gives you the semantic fingerprint of the node.

Step 3 — Review the catalog: which models are candidates?
Based on the semantic fingerprint from Step 2: (a) identify all catalog models whose fields
correspond to the concepts present; (b) immediately exclude any models marked [list container];
(c) for each candidate, note which of the node's concepts its fields can capture.
If the node has only one InfoUnit, do not force multiple complementary models — a single
well-matched model is the correct outcome.

Step 4 — Score each candidate.
For each candidate model from Step 3:
(a) Coverage score: what fraction of the node's actual content can be captured by fields in
    this model? Express as a percentage.
(b) Field alignment: list which specific fields in the model correspond to which specific
    concepts in the node.
(c) Mismatch: list any fields in the model that have no corresponding content in the node.
    (Acceptable — a model may have more fields than the node uses, and that does not reduce
    its fitness.)
(d) Gap: list any concepts in the node that NO field in this model can capture.
The best model maximises coverage of the node's actual content. A model with many unused
fields is still a good match if its covered fields align well.

Step 5 — Select best match or null.
(a) If the best candidate covers >= 25% of the node's actual content:
    - matched_model_class = that model's exact CamelCase class name from the catalog.
    - confidence = "high" if coverage >= 75%, "medium" if 25–74%, "low" if < 25% but still
      selected (only when no better option exists).
    - List any content not covered by the primary model as gaps.
    - For each gap: check if any other catalog model covers it. If yes, that model is a
      complementary_model. If no catalog model covers it, note the gap.
(b) If no candidate covers >= 25% of the node's actual content:
    - matched_model_class = null.
    - confidence = "low" (always, when matched_model_class is null).
    - propose_new_model = true.
    - proposed_schema_name: propose a Python CamelCase class name for the new model.
    - proposed_schema_fields: list the fields the new model should have. For each field:
      name (snake_case), type hint (e.g. "str", "list[str]", "str | None"), what it captures,
      and whether it is required.
    - complementary_models = [] (do not list partial matches as complementary when there is
      no primary match).

Special cases:
- Node with only one InfoUnit: a single well-matched model is correct. Do not artificially
  add complementary models to appear thorough.
- Generic/introductory nodes (title contains "Introduction", "Scope", "Overview", "General",
  "Background", or the InfoUnit describes only context-setting prose with no extractable
  structured data): if the content is purely navigational or contextual with no structured
  data, set matched_model_class = null and propose_new_model = true. Do not force-fit.
- Mixed-content nodes: pick the model that covers the dominant content (by volume and
  specificity). List the secondary topic as a complementary_model if a catalog model covers it.
- Table/FieldGroup nodes: the model decision is based entirely on what the table or field
  group contains, not on its structural role. Never default to a generic model because the
  node is a table.

Step 6 — Identify gaps and complementary models.
(a) List every concept present in the node that is NOT captured by the primary matched model
    (or that has no match at all if null).
(b) For each gap: check if any catalog model covers it. If yes, list it as a complementary_model
    with a specific note on what it covers.
(c) If propose_new_model = true: describe in 2–4 sentences what the new model would need to
    capture that no existing model addresses.
(d) If propose_new_model is false AND there are concepts that NO catalog model can capture at
    all (not even as complementary): describe the supplementary fields that would capture that
    content. For each field: name (snake_case), type hint, what it captures, required or optional.
    These are additional fields that extend the matched model's coverage for this specific node.
    If all gaps are covered by complementary models, this section is empty.

Step 7 — Produce the AnnotationDecision output.
Populate all fields based on your analysis:
- matched_model_class: exact CamelCase class name from the catalog, or null.
- confidence: "high" (>= 75%) / "medium" (25–74%) / "low" (< 25% or null).
- rationale: 6–8 sentences summarising your analysis, referencing specific InfoUnit titles
  and the node_id or title where they contributed to the decision.
  No speculative language ("probably", "likely", "might", "could").
- coverage_gaps: list of concepts in this node not captured by the primary model (empty list
  if coverage is complete).
- complementary_models: secondary catalog models that cover gaps. Must be [] when
  matched_model_class is null.
- propose_new_model: true when matched_model_class is null, false otherwise.
- proposed_model_description: 2–4 sentence description of a new model when propose_new_model
  is true; null when false.
- supplementary_fields: extra fields to extend the primary model for gaps not covered by any
  complementary model. Must be [] when matched_model_class is null.
- proposed_schema_name: proposed CamelCase class name for a new model when matched_model_class
  is null; null otherwise.
- proposed_schema_fields: proposed fields for the new model when matched_model_class is null;
  [] otherwise.


## Few-Shot Example

### Input node context

```
Node id: "3_2_1"
Title: "3.2.1 Tensile Strength Test Results"
Role: table
Depth: 3

InfoUnits:
  [0] title: "Sample Identification"
      description: "Each sample is identified by a unique sample code assigned by the
                    testing laboratory. Sample codes follow the format LAB-YYYY-NNN."
  [1] title: "Mechanical Test Results"
      description: "Results are reported for tensile strength (450–520 MPa), yield
                    strength (≥380 MPa), elongation at break (18–25%), and hardness
                    (HRC 42–46)."
```

Available models (excerpt):
```
1. MaterialTestRecord — Records mechanical test results for a single material sample.
   Fields: sample_id: str, test_name: str, specification: str | None,
           result: str, pass_fail: str | None, method: str | None

2. FatigueDataPoint — Records a single measurement at a fatigue cycle count.
   Fields: cycle_count: int, load_condition: str, parameter: str, result: str, unit: str | None

3. MaterialSpecDocument [list container] — Top-level container for an entire material
   specification document.
   Fields: items: list[SpecificationSection]
```

### Expected output

```json
{{
  "matched_model_class": "MaterialTestRecord",
  "confidence": "medium",
  "rationale": "The node contains two InfoUnits: 'Sample Identification' and 'Mechanical Test Results'. The MaterialTestRecord model directly captures sample_id (matching the sample code format LAB-YYYY-NNN) and the test result fields (test_name, specification, result, pass_fail, method), which align with the tensile strength, yield strength, elongation, and hardness results described. Coverage is estimated at 65%: the model covers the core test result structure but does not have a dedicated field for the hardness scale descriptor. FatigueDataPoint was considered but rejected because it requires cycle_count and load_condition fields that are absent from this node. MaterialSpecDocument was excluded as a [list container] aggregate model. The hardness scale is noted as a minor gap but does not reduce coverage below the 25% threshold.",
  "coverage_gaps": ["hardness scale descriptor (HRC — qualitative scale identifier not tied to a numeric field)"],
  "complementary_models": [],
  "propose_new_model": false,
  "proposed_model_description": null,
  "supplementary_fields": [
    {{
      "field_name": "measurement_scale",
      "field_type": "str | None",
      "description": "Measurement scale or unit qualifier for the result (e.g. 'HRC', 'Vickers', 'Brinell')",
      "required": false
    }}
  ],
  "proposed_schema_name": null,
  "proposed_schema_fields": []
}}
```

Why this output is correct:
- MaterialTestRecord is selected because its fields cover >= 25% of the node's actual content
  (sample_id + test result fields map to both InfoUnits).
- confidence is "medium" (65% coverage, in the 25–74% range).
- MaterialSpecDocument is excluded because it is marked [list container].
- FatigueDataPoint is excluded because its required fields (cycle_count, load_condition) have no
  corresponding content in the node.
- The hardness scale gap is captured as a supplementary_field (not a complementary model) because
  no catalog model covers it.
- complementary_models is [] because all gaps are either covered by supplementary_fields or
  are minor enough not to require a secondary model.


Return ONLY the JSON AnnotationDecision object. Do not add explanatory text before or after it.
"""


# ─── Builder functions ────────────────────────────────────────────────────────

def build_theme_classification_prompt(
    themes_block: str,
    document_name: str = "",
    theme_histogram: str = "",
) -> str:
    """Build the theme classification prompt, injecting available themes, document name,
    and recent classification histogram."""
    lines = []
    if document_name:
        lines.append(f"Document: {document_name}")
    if theme_histogram:
        lines.append(f"Recent classifications: {theme_histogram}")
    document_context = (
        "---\n" + "\n".join(lines) + "\n---\n"
        if lines else ""
    )
    return THEME_CLASSIFICATION_PROMPT_TEMPLATE.format(
        themes_block=themes_block,
        document_context=document_context,
    )


def build_annotation_decision_prompt(theme_node: ThemeNode) -> str:
    """Build the annotation decision prompt by injecting the theme-specific model catalog."""
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()
    catalog_block = theme_registry.build_catalog_block(theme_node)
    return DECISION_PROMPT_TEMPLATE.format(catalog_block=catalog_block)
