"""tabular/prompts.py — LLM prompts for the tabular pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeNode

TABULAR_DECISION_PROMPT_TEMPLATE = """
<identity>
You are a pharmaceutical dossier content-matching specialist. Your function is to
examine the column headers and a small preview of rows from a tabular file (CSV or XLSX)
and determine which Pydantic extraction model from a predefined catalog best describes
what each ROW in this table represents.

This is a TABULAR file: every row represents one instance of the same entity type.
The model you select will be used to extract structured data from EACH ROW individually.
You operate inside a production graph-annotation pipeline. A wrong model assignment
will cause downstream extraction to apply the wrong schema to real pharmaceutical data.
</identity>

<critical_tabular_rules>
RULE 1 — COLUMN-FIRST MATCHING:
  Match based on the column HEADERS and what the sample rows show, not on the filename.
  A column called "batch_number" is strong evidence for a batch-related model.
  Always check the sample values to confirm header semantics.

RULE 2 — CONSERVATIVE MATCHING:
  If no catalog model covers >= 50% of the columns with meaningful fields, set
  matched_model_class = null and propose a new model.
  null is always safer than a wrong model. Never select a model only because its
  name resembles a column header.

RULE 3 — ROW IS THE UNIT OF EXTRACTION:
  Every row in the table represents the same entity. Choose the model that best
  describes what ONE ROW of data represents (e.g. one batch, one stability test,
  one product specification, one component entry).
  Do not pick a model designed to hold multiple rows (aggregate/container models).

RULE 4 — AGGREGATE MODEL PROHIBITION:
  Module3Quality, DrugSubstanceModule, DrugProductModule, and AppendicesModule are
  top-level container models. NEVER select them.
</critical_tabular_rules>

<available_models>
{catalog_block}
</available_models>

<table_preview>
{preview_markdown}
</table_preview>

<column_headers>
{headers_list}
</column_headers>

<decision_protocol>
Apply the same 7-step annotation decision protocol as normal annotation:

STEP 1 — Understand what each COLUMN represents (not rows or the file as a whole).
STEP 2 — Identify the pharmaceutical concept captured by each column header.
STEP 3 — Review the catalog for candidate models.
STEP 4 — Score each candidate: what fraction of the columns can it capture?
STEP 5 — Select best match or null (>= 50% column coverage threshold).
STEP 6 — Identify gaps and complementary models.
STEP 7 — Produce the AnnotationDecision output.

Remember: you are selecting a model for ONE ROW (one instance), not the whole table.
</decision_protocol>
"""

TABULAR_MAPPING_PROMPT_TEMPLATE = """
<identity>
You are a data mapping specialist. Your task is to map each column of a tabular file
to the appropriate field of a Pydantic extraction model. This mapping will be used
to extract structured data from EVERY ROW without any further LLM calls.

Accuracy here directly determines the quality of all row-level data extraction.
</identity>

<model_class>{matched_model_class}</model_class>

<model_fields>
{model_fields_block}
</model_fields>

<column_headers>
{headers_list}
</column_headers>

<table_preview>
{preview_markdown}
</table_preview>

<mapping_rules>
1. Map each column header to the model field(s) that best captures its values.
2. For each column, evaluate ALL candidate fields across the primary, supplementary,
   and complementary models and assign a confidence level to each candidate.
3. confidence levels:
   - 'high'   → strong semantic match
   - 'medium' → plausible match
   - 'low'    → no good match anywhere
4. Emit rules — apply EXACTLY ONE of the two cases below per column:
   CASE A — at least one candidate reaches 'medium' or 'high' confidence:
     • Emit one entry per qualifying (medium or high) target.
     • Do NOT emit any '__extra__' entry for this column.
     • Do NOT emit any 'low' confidence entry for this column.
   CASE B — NO candidate reaches 'medium' confidence (all candidates are 'low'):
     • Emit exactly ONE entry: model_field_name='__extra__', confidence='low',
       target_model='primary'.
     • Do NOT emit any other entry for this column.
5. Never emit a 'low' confidence entry alongside a 'medium' or 'high' entry for
   the same column. Never emit more than one '__extra__' entry per column.
6. target_model values:
   - 'primary'       → field belongs to the primary model
   - 'supplementary' → field belongs to the supplementary fields listed above
   - '<CamelCaseName>' → exact class name of the complementary model (e.g. 'BatchAnalysis')
7. If a column fits both a complementary model field at medium confidence AND a
   supplementary field at high confidence, emit BOTH entries (both qualify under
   CASE A). Do not suppress the medium-confidence entry.
8. Two different columns MUST NOT map to the same (target_model, model_field_name)
   pair unless they are truly equivalent. Flag such cases in notes.
9. Every column must appear at least once in the output. Use the sample rows to
   infer column semantics when the header name is ambiguous.
10. Prefer exact field matches over approximate ones. If coverage is under 50%,
    use '__extra__' liberally rather than force-fitting.
</mapping_rules>

<task>
For each column header, identify the model field(s) that best capture the values in
that column. Return a ColumnMapping where every column appears at least once. A column
may appear multiple times if it maps to fields in different models (multi-target).
</task>
"""


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


def _is_scalar_field(annotation) -> bool:
    """Return True for scalar-mappable annotations: str, int, float, bool, Literal[...],
    and Optional / Union wrappers of those.  Returns False for BaseModel subclasses
    and list/dict of BaseModel.
    """
    import typing

    from pydantic import BaseModel as _BaseModel

    if annotation is None:
        return True  # treat unknown as scalar (safe fallback)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Annotated[T, ...] — unwrap
    if origin is typing.Annotated:
        return _is_scalar_field(args[0]) if args else True

    # list — never scalar-mappable from a single CSV cell
    if origin is list:
        return False

    if origin is dict:
        return False  # dicts are not single-column mappable

    # Union / Optional — scalar if ALL non-None args are scalar
    import types as _builtin_types
    is_union = origin is typing.Union or (
        hasattr(_builtin_types, "UnionType")
        and isinstance(annotation, _builtin_types.UnionType)
    )
    if is_union:
        for arg in args:
            if arg is type(None):
                continue
            if not _is_scalar_field(arg):
                return False
        return True

    # Literal — always scalar
    if origin is typing.Literal:
        return True

    # Bare class
    if isinstance(annotation, type):
        if issubclass(annotation, _BaseModel):
            return False
        if issubclass(annotation, (list, dict)):
            return False
        return True

    return True


def build_full_fields_block(
    matched_model_class: str,
    supplementary_fields: list[dict] | None = None,
    complementary_model_names: list[str] | None = None,
) -> str:
    """Build the complete field reference block for the mapping prompt.

    Shows:
    - Primary model fields (from matched_model_class)
    - Supplementary fields proposed by decide_model (extra columns to extend the primary model)
    - Complementary model fields (from decide_model's complementary_models; shown as
      mappable targets so the LLM can route columns to them; nested BaseModel fields
      are omitted as they cannot be populated from a single CSV value)
    """
    sections: list[str] = []

    # Primary model fields
    primary_block = _build_model_fields_block(matched_model_class)
    sections.append(f"PRIMARY MODEL — {matched_model_class}:\n{primary_block}")

    # Supplementary fields (simple scalars that extend the primary model)
    if supplementary_fields:
        lines = []
        for sf in supplementary_fields:
            fname = sf.get("field_name", "")
            ftype = sf.get("field_type", "str")
            fdesc = sf.get("description", "")
            frequired = "required" if sf.get("required", False) else "optional"
            if fname:
                lines.append(f"  {fname} [{ftype}]: {fdesc} ({frequired})")
        if lines:
            sections.append(
                "SUPPLEMENTARY FIELDS (extend the primary model; use target_model='supplementary'):\n"
                + "\n".join(lines)
            )

    # Complementary model fields — shown as extractable targets (scalar fields only)
    if complementary_model_names:
        for model_name in complementary_model_names:
            try:
                comp_block = _build_model_fields_block(model_name, scalar_only=True)
                sections.append(
                    f"COMPLEMENTARY MODEL — {model_name} "
                    f"(use target_model='{model_name}' for columns that map to these fields):\n"
                    f"{comp_block}"
                )
            except Exception:
                sections.append(f"COMPLEMENTARY MODEL — {model_name}: (fields unavailable)")

    return "\n\n".join(sections)


def build_tabular_mapping_prompt(
    matched_model_class: str,
    preview_markdown: str,
    headers: list[str],
    supplementary_fields: list[dict] | None = None,
    complementary_model_names: list[str] | None = None,
) -> str:
    """Build the column mapping prompt for a tabular file.

    Shows the primary model fields, any supplementary fields from decide_model,
    and complementary model fields (scalar-only, as mappable targets) so the LLM
    can map every column to the correct extraction target.
    """
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

    Reuses THEME_CLASSIFICATION_PROMPT_TEMPLATE from annotation/prompts.py,
    substituting the standard node XML context with a tabular sheet context
    (file name, sheet name, column headers, preview rows).
    """
    from scinr.newton.annotation.prompts import THEME_CLASSIFICATION_PROMPT_TEMPLATE
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    themes_block = theme_registry.get_theme_list_for_prompt()
    document_context = (
        "<document_context>\n"
        f"File: {document_name}\n"
        f"Sheet: {sheet_name}\n"
        f"Columns ({len(headers)}): {', '.join(headers)}\n\n"
        f"Preview:\n{preview_markdown}\n"
        "</document_context>"
    )
    return THEME_CLASSIFICATION_PROMPT_TEMPLATE.format(
        themes_block=themes_block,
        document_context=document_context,
    )


def _build_model_fields_block(model_class_name: str, scalar_only: bool = False) -> str:
    """Build a human-readable field description block for the matched model.

    Format per field:
        field_name [type]: description (required/optional)

    Parameters
    ----------
    model_class_name:
        CamelCase name of the Pydantic model class to describe.
    scalar_only:
        When True, skip fields whose annotation is a BaseModel subclass or
        list/dict of BaseModel (they cannot be populated from a single CSV value).
        Such fields are annotated with '(nested — not mappable)' and excluded.
    """
    try:
        from scinr.newton.entity_extraction.model_resolver import resolve_model_class

        cls = resolve_model_class(model_class_name)
        lines = []
        for fname, finfo in cls.model_fields.items():
            ann = finfo.annotation
            if scalar_only and not _is_scalar_field(ann):
                continue  # omit nested BaseModel fields from complementary sections
            type_str = str(ann).replace("typing.", "") if ann else "Any"
            desc = finfo.description or ""
            required = "required" if finfo.is_required() else "optional"
            lines.append(f"  {fname} [{type_str}]: {desc} ({required})")
        return "\n".join(lines) if lines else f"(no mappable fields found for {model_class_name})"
    except Exception as exc:
        return f"(could not load model fields for {model_class_name}: {exc})"
