"""
annotation/prompts_gpt_reasoning.py — OpenAI reasoning models variant.

Targets GPT-5.5, o3, and o4-mini via the OpenAI API. These models reason
internally before producing output, making step-by-step elicitation and
self-verification checklists counterproductive. Uses Markdown section headers
and goal-based language instead.

For non-reasoning GPT models (GPT-4o, GPT-4.1, GPT-4.5), use the GENERIC
family — the reasoning-model patterns here offer no benefit on standard
instruction-following models and may degrade performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeNode


# ─── Theme Classification Prompt ─────────────────────────────────────────────

THEME_CLASSIFICATION_PROMPT_TEMPLATE = """# Instructions

You are a document theme classifier. Classify the provided document section into the most specific available theme.

## Available Themes

{themes_block}

{document_context}

## Classification Goal

Select the theme path that best matches the actual content of the section. The `theme` field must exactly match one of the paths listed above. Choose the most specific (deepest) path. Use `default` only when no other theme clearly fits. The `justification` must reference specific content from the node — not merely restate the theme name.

Return ONLY the JSON object with fields "theme" and "justification". Respond directly. Start your response with `{`.
"""


# ─── Annotation Decision Prompt ───────────────────────────────────────────────

DECISION_PROMPT_TEMPLATE = """# Instructions

You are a structured document content-matching specialist operating in a production graph-annotation pipeline. The result of your decision is written to Neo4j as a HAS_MODEL_DECISION relationship on the evaluated StructureNode. A wrong model assignment causes downstream extraction pipelines to apply the wrong schema to real document content.

## Critical Rules

**Content-first matching.** Match the actual content described in the InfoUnits to the model whose fields best capture it — not the node's title alone. Never penalise a node for missing content.

**Conservative matching.** Assign a model only when it covers >= 25% of the node's actual content. `null` is always safer than a wrong model. Never select a model because its name resembles the node title.

**Ground in node content.** Reason from the node_id, title, and InfoUnits provided — these together are the only representation of a node's content at this stage; the original document text is not accessible. The InfoUnits are the primary source; the node_id and title are complementary signals (e.g. a CTD section code or domain keyword may confirm or disambiguate what the InfoUnits describe). Do not invent content absent from all three sources.

**Aggregate model prohibition.** Never select a model marked [list container]. These are top-level containers whose fields are too coarse to capture individual StructureNode content.

## Available Models

{catalog_block}

## Decision Goal

Select the single catalog model whose fields best match the actual content described in the node_id, title, and InfoUnits, following this logic:

- If the best candidate covers >= 25% of the node's actual content: set `matched_model_class` to its exact CamelCase class name. Set `confidence` to "high" (>= 75%), "medium" (25–74%), or "low" (< 25% but no better option exists). Identify any content not covered by the primary model as gaps; check if any other catalog model covers each gap and list those as `complementary_models`.

- If no candidate covers >= 25%: set `matched_model_class = null`, `confidence = "low"`, `propose_new_model = true`, and propose a new schema (`proposed_schema_name`, `proposed_schema_fields`). Do not list partial matches as complementary when there is no primary match.

For gaps that no catalog model covers, describe `supplementary_fields` to extend the matched model's coverage.

## Boundary Conditions

- **One InfoUnit only** → a single well-matched model is correct; do not artificially add complementary models.
- **Generic/introductory nodes** (title is "Introduction", "Scope", "Overview", "General", "Background", or InfoUnit is purely context-setting with no extractable structured data) → set `matched_model_class = null` and `propose_new_model = true`.
- **Mixed-content nodes** → pick the model covering the dominant content; list the secondary topic as a `complementary_model` if a catalog model covers it.
- **Table/FieldGroup nodes** → decide based on what the table contains, not its structural role.

Return ONLY the JSON AnnotationDecision object. Respond directly. Start your response with `{`.
"""


# ─── Builder functions ────────────────────────────────────────────────────────

def build_theme_classification_prompt(
    themes_block: str,
    document_name: str = "",
    theme_histogram: str = "",
) -> str:
    """Build the theme classification prompt for OpenAI reasoning models."""
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


def build_annotation_decision_prompt(theme_node: "ThemeNode") -> str:
    """Build the annotation decision prompt for OpenAI reasoning models."""
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()
    catalog_block = theme_registry.build_catalog_block(theme_node)
    return DECISION_PROMPT_TEMPLATE.format(catalog_block=catalog_block)
