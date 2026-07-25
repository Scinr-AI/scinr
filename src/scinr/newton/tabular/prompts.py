"""
tabular/prompts.py — Tabular pipeline prompt dispatcher and shared utilities.

Shared utility functions (build_full_fields_block, _build_model_fields_block,
_is_scalar_field) live here and are imported by all family-specific modules.

build_tabular_decision_prompt(), build_tabular_mapping_prompt(), and
build_tabular_theme_prompt() dispatch to the correct family variant.

To add a new prompt family:
  1. Create tabular/prompts_<family>.py with the builder functions
  2. Add the new PromptFamily member to config.py
  3. Add an elif branch in each dispatcher function below
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeNode


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def _m():
    """Return the tabular prompt module for the currently configured PromptFamily."""
    from scinr.newton.config import PromptFamily, get_prompt_family
    family = get_prompt_family()
    if family == PromptFamily.CLAUDE:
        from scinr.newton.tabular import prompts_claude as m
    elif family == PromptFamily.GPT_REASONING:
        from scinr.newton.tabular import prompts_gpt_reasoning as m
    else:
        from scinr.newton.tabular import prompts_generic as m
    return m


def build_tabular_decision_prompt(
    theme_node: "ThemeNode", preview_markdown: str, headers: list[str]
) -> str:
    """Build the model decision prompt for a tabular file."""
    return _m().build_tabular_decision_prompt(theme_node, preview_markdown, headers)


def build_tabular_mapping_prompt(
    matched_model_class: str,
    preview_markdown: str,
    headers: list[str],
    supplementary_fields: list[dict] | None = None,
    complementary_model_names: list[str] | None = None,
) -> str:
    """Build the column mapping prompt for a tabular file."""
    return _m().build_tabular_mapping_prompt(
        matched_model_class, preview_markdown, headers, supplementary_fields, complementary_model_names
    )


def build_tabular_theme_prompt(
    document_name: str,
    sheet_name: str,
    headers: list[str],
    preview_markdown: str,
) -> str:
    """Build the theme classification prompt for a tabular sheet."""
    return _m().build_tabular_theme_prompt(document_name, sheet_name, headers, preview_markdown)


# ── Shared utilities (used by both family-specific modules) ────────────────────

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
