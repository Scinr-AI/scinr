"""
tabular/prompts_generic.py — LLM prompts for the tabular pipeline.

Generic/universal variant compatible with any LLM family (Kimi1.5, GLM-5, GPT-4,
Claude, Mistral, etc.). Preserves all business logic from prompts.py (Claude variant).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeNode


# ─── Tabular Decision Prompt ──────────────────────────────────────────────────

TABULAR_DECISION_PROMPT_TEMPLATE = """You are a pharmaceutical dossier content-matching specialist for tabular files. Your function is to examine the column headers and a small preview of rows from a tabular file (CSV or XLSX) and determine which Pydantic extraction model from a predefined catalog best describes what each ROW in this table represents.

This is a TABULAR file: every row represents one instance of the same entity type. The model you select will be used to extract structured data from EACH ROW individually. A wrong model assignment will cause downstream extraction to apply the wrong schema to real pharmaceutical data.

## Critical Rules

Rule 1 — COLUMN-FIRST MATCHING:
Match based on the column HEADERS and what the sample rows show, not on the filename.
A column called "batch_number" is strong evidence for a batch-related model.
Always check the sample values to confirm header semantics.

Rule 2 — CONSERVATIVE MATCHING:
If no catalog model covers >= 25% of the columns with meaningful fields, set
matched_model_class = null and propose a new model. null is always safer than a wrong model.
Model selection must be based on field-to-column coverage, not on name similarity between
the model and a column header.

Rule 3 — ROW IS THE UNIT OF EXTRACTION:
Every row in the table represents the same entity. Choose the model that best describes what
ONE ROW of data represents (e.g. one batch, one stability test, one product specification,
one component entry). Do not pick a model designed to hold multiple rows.

Rule 4 — AGGREGATE MODEL PROHIBITION:
Models marked [list container] in the catalog represent an entire document or a major section.
Their fields are too coarse to capture individual row-level data and will produce semantically
invalid extraction output. Select only models whose individual fields map to the column concepts
in this table.


## Available Models

{catalog_block}


## Table Preview

{preview_markdown}


## Column Headers

{headers_list}


## Decision Protocol

Execute these steps in strict order. Do not skip any step.

Step 1 — Understand what each COLUMN represents.
Read each column header and inspect the sample values in the preview. Determine what
pharmaceutical concept each column captures (e.g. batch identifier, test parameter name,
measured result, specification limit, storage condition, time point).

Step 2 — Build the semantic fingerprint of one row.
A single row represents one entity instance. List all the distinct pharmaceutical concepts
captured by the columns together — this is the semantic fingerprint of one row.

Step 3 — Review the catalog for candidate models.
Based on the semantic fingerprint from Step 2: (a) identify all catalog models whose fields
        correspond to the concepts present in the columns; (b) exclude all [list container] models (see Rule 4); (c) for each candidate, note which columns its fields can capture.

Step 4 — Score each candidate.
For each candidate model: (a) coverage score — what fraction of the columns can be captured
by fields in this model? Express as a percentage; (b) field alignment — list which specific
model fields correspond to which specific column headers; (c) gaps — list any columns that
no field in this model can capture.

Step 5 — Select best match or null.
(a) If the best candidate covers >= 25% of the columns:
    - matched_model_class = that model's exact CamelCase class name.
    - confidence = "high" if coverage >= 75%, "medium" if 25–74%, "low" if < 25% but still
      selected (only when no better option exists).
    - List any columns not covered by the primary model as gaps.
    - For each gap: check if any other catalog model covers it. If yes, that model is a
      complementary_model.
(b) If no candidate covers >= 25% of the columns:
    - matched_model_class = null.
    - confidence = "low".
    - propose_new_model = true.
    - proposed_schema_name: propose a Python CamelCase class name for the new model.
    - proposed_schema_fields: list the fields the new model should have. For each field:
      name (snake_case), type hint, what it captures, and whether it is required.
    - complementary_models = [].

Step 6 — Identify gaps and complementary models.
List every column concept not captured by the primary model. For each gap, check if any
catalog model covers it and list it as a complementary_model with a specific coverage note.
If propose_new_model = true, describe in 2–4 sentences what the new model would need to
capture. If propose_new_model is false and gaps exist that no catalog model covers, describe
supplementary fields to extend the primary model.

Step 7 — Produce the AnnotationDecision output.
Populate all fields: matched_model_class, confidence, rationale (6–8 sentences referencing
specific column headers), coverage_gaps, complementary_models, propose_new_model,
proposed_model_description, supplementary_fields, proposed_schema_name, proposed_schema_fields.

Remember: you are selecting a model for ONE ROW (one instance), not the whole table.

Return ONLY the JSON AnnotationDecision object. Do not add explanatory text before or after it.
"""


# ─── Tabular Mapping Prompt ───────────────────────────────────────────────────

TABULAR_MAPPING_PROMPT_TEMPLATE = """You are a data mapping specialist. Your task is to map each column of a tabular file to the appropriate field of a Pydantic extraction model. This mapping will be used to extract structured data from EVERY ROW without any further LLM calls. Accuracy here directly determines the quality of all row-level data extraction.

## Primary Model

{matched_model_class}

## Model Fields

{model_fields_block}

## Column Headers

{headers_list}

## Table Preview

{preview_markdown}

## Mapping Rules

1. Map each column header to the model field(s) that best captures its values.

2. For each column, evaluate ALL candidate fields across the primary, supplementary, and
   complementary models and assign a confidence level to each candidate:
   - "high"   → strong semantic match between column and field.
   - "medium" → plausible match (column values could populate this field).
   - "low"    → no good match anywhere in any model.

3. Apply EXACTLY ONE of the two cases below per column:

   CASE A — at least one candidate reaches "medium" or "high" confidence:
   - Emit one entry per qualifying (medium or high) target.
   - Do NOT emit any "__extra__" entry for this column.
   - Do NOT emit any "low" confidence entry for this column.

   CASE B — NO candidate reaches "medium" confidence (all candidates are "low"):
   - Emit exactly ONE entry: model_field_name="__extra__", confidence="low",
     target_model="primary".
   - Do NOT emit any other entry for this column.

4. Each column emits either: qualifying entries (medium or high confidence) under CASE A,
   or exactly one "__extra__" entry under CASE B — never both, and never more than one
   "__extra__" per column.

5. target_model values:
   - "primary"          → field belongs to the primary model.
   - "supplementary"    → field belongs to the supplementary fields listed above.
   - "<CamelCaseName>"  → exact class name of the complementary model (e.g. "BatchAnalysis").

6. If a column fits both a complementary model field at medium confidence AND a supplementary
   field at high confidence, emit BOTH entries (both qualify under CASE A). Do not suppress
   the medium-confidence entry.

7. Two different columns must NOT map to the same (target_model, model_field_name) pair unless
   they are truly equivalent. Flag such cases in the notes field of the mapping entry.

8. Every column must appear at least once in the output. Use the sample rows to infer column
   semantics when the header name is ambiguous.

9. Prefer exact field matches over approximate ones. If coverage is under 25%, use "__extra__"
   liberally rather than force-fitting columns to fields they do not match.

10. For each mapping entry, include a brief rationale explaining why this column maps to this
    field (or why it is __extra__).

Return ONLY the JSON ColumnMapping object where every column appears at least once. Do not add explanatory text before or after it.
"""


# ─── Builder functions ────────────────────────────────────────────────────────

def build_tabular_decision_prompt(
    theme_node: ThemeNode, preview_markdown: str, headers: list[str]
) -> str:
    """Build the model decision prompt for a tabular file."""
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
    """Build the column mapping prompt for a tabular file.

    Delegates field block construction to the original prompts.py helper so that
    primary, supplementary, and complementary model fields are rendered identically.
    """
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
    """Build the theme classification prompt for a tabular sheet.

    Reuses THEME_CLASSIFICATION_PROMPT_TEMPLATE from annotation/prompts_generic.py,
    substituting the standard node context with a tabular sheet context
    (file name, sheet name, column headers, preview rows).
    """
    from scinr.newton.annotation.prompts_generic import THEME_CLASSIFICATION_PROMPT_TEMPLATE
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
