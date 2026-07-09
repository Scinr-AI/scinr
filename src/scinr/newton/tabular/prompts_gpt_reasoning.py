"""
tabular/prompts_gpt_reasoning.py — OpenAI reasoning models variant.

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


# ─── Tabular Decision Prompt ──────────────────────────────────────────────────

TABULAR_DECISION_PROMPT_TEMPLATE = """# Instructions

You are a tabular data content-matching specialist. Examine the column headers and a sample of rows from a tabular file (CSV or XLSX) and select the Pydantic extraction model that best describes what each ROW in this table represents.

This is a TABULAR file: every row represents one instance of the same entity type. The model you select will be used to extract structured data from EACH ROW individually. A wrong model assignment causes downstream extraction to apply the wrong schema to real data.

## Critical Rules

**Column-first matching.** Match based on column HEADERS and what the sample rows show, not on the filename. Check sample values to confirm header semantics.

**Conservative matching.** If no catalog model covers >= 25% of the columns with meaningful fields, set `matched_model_class = null` and propose a new model. Model selection must be based on field-to-column coverage, not on name similarity.

**Row is the unit of extraction.** Choose the model that best describes what ONE ROW represents (one transaction, one test result, one product entry, one contract clause). Do not pick a model designed to hold multiple rows.

**Aggregate model prohibition.** Never select a model marked [list container]. These are top-level containers whose fields cannot capture row-level data.

## Available Models

{catalog_block}

## Table Preview

{preview_markdown}

## Column Headers

{headers_list}

## Decision Goal

Select the single model whose fields best match the columns, following this logic:

- If the best candidate covers >= 25% of the columns: set `matched_model_class` to its exact CamelCase name. Set `confidence` to "high" (>= 75%), "medium" (25–74%), or "low" (< 25% but no better option). For each uncovered column, check if another catalog model covers it and list those as `complementary_models`.

- If no candidate covers >= 25%: set `matched_model_class = null`, `confidence = "low"`, `propose_new_model = true`, and propose a new schema (`proposed_schema_name`, `proposed_schema_fields`). Do not list partial matches as complementary when there is no primary match.

For uncovered columns that no catalog model handles, describe `supplementary_fields` to extend the primary model.

Populate all output fields: `matched_model_class`, `confidence`, `rationale` (must reference specific column headers from the preview to justify the model selection), `coverage_gaps`, `complementary_models`, `propose_new_model`, `proposed_model_description`, `supplementary_fields`, `proposed_schema_name`, `proposed_schema_fields`.

Return ONLY the JSON AnnotationDecision object. Respond directly. Start your response with `{`.
"""


# ─── Tabular Mapping Prompt ───────────────────────────────────────────────────

TABULAR_MAPPING_PROMPT_TEMPLATE = """# Instructions

You are a data mapping specialist. Map each column of a tabular file to the appropriate field of a Pydantic extraction model. This mapping is used to extract structured data from EVERY ROW without any further LLM calls — accuracy here directly determines the quality of all row-level data extraction.

## Primary Model

{matched_model_class}

## Model Fields

{model_fields_block}

## Column Headers

{headers_list}

## Table Preview

{preview_markdown}

## Mapping Goal

A mapping is correct when:
- Every column appears at least once in the output — no column is silently dropped.
- Confidence reflects semantic match quality: "high" (strong semantic match), "medium" (plausible match), "low" (no good match anywhere).
- Columns with at least one "medium" or "high" candidate emit one entry per qualifying target field (CASE A). Columns with no "medium" match emit exactly one entry with `model_field_name="__extra__"`, `confidence="low"`, `target_model="primary"` (CASE B). Never both cases for the same column; never more than one `__extra__` per column.
- `target_model` is `"primary"`, `"supplementary"`, or the exact CamelCase complementary model name.
- No two columns share the same `(target_model, model_field_name)` pair unless truly equivalent — flag duplicates in the `notes` field.
- `__extra__` is preferred over force-fitting a column to a semantically mismatched field.
- Each entry includes a brief rationale explaining the mapping decision.

Return ONLY the JSON ColumnMapping object where every column appears at least once. Respond directly. Start your response with `{`.
"""


# ─── Builder functions ────────────────────────────────────────────────────────

def build_tabular_decision_prompt(
    theme_node: "ThemeNode", preview_markdown: str, headers: list[str]
) -> str:
    """Build the model decision prompt for a tabular file (OpenAI reasoning models variant)."""
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    catalog_block = theme_registry.build_catalog_block(theme_node)
    headers_list = ", ".join(f'"{h}"' for h in headers)
    return TABULAR_DECISION_PROMPT_TEMPLATE.format(
        catalog_block=catalog_block,
        preview_markdown=preview_markdown,
        headers_list=headers_list,
    )


def build_tabular_mapping_prompt(
    matched_model_class: str,
    preview_markdown: str,
    headers: list[str],
    supplementary_fields: list[dict] | None = None,
    complementary_model_names: list[str] | None = None,
) -> str:
    """Build the column mapping prompt for a tabular file (OpenAI reasoning models variant)."""
    from scinr.newton.tabular.prompts import build_full_fields_block

    model_fields_block = build_full_fields_block(
        matched_model_class=matched_model_class,
        supplementary_fields=supplementary_fields,
        complementary_model_names=complementary_model_names,
    )
    headers_list = ", ".join(f'"{h}"' for h in headers)
    return TABULAR_MAPPING_PROMPT_TEMPLATE.format(
        matched_model_class=matched_model_class,
        model_fields_block=model_fields_block,
        preview_markdown=preview_markdown,
        headers_list=headers_list,
    )


def build_tabular_theme_prompt(
    document_name: str,
    sheet_name: str,
    headers: list[str],
    preview_markdown: str,
) -> str:
    """Build the theme classification prompt for a tabular sheet (OpenAI reasoning models variant).

    Reuses THEME_CLASSIFICATION_PROMPT_TEMPLATE from annotation/prompts_gpt_reasoning.py,
    substituting the standard node context with a tabular sheet context.
    """
    from scinr.newton.annotation.prompts_gpt_reasoning import THEME_CLASSIFICATION_PROMPT_TEMPLATE
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    themes_block = theme_registry.get_theme_list_for_prompt()
    document_context = (
        "---\n"
        f"File: {document_name}\n"
        f"Sheet: {sheet_name}\n"
        f"Columns ({len(headers)}): {', '.join(headers)}\n\n"
        f"Preview:\n{preview_markdown}\n"
        "---\n"
    )
    return THEME_CLASSIFICATION_PROMPT_TEMPLATE.format(
        themes_block=themes_block,
        document_context=document_context,
    )
