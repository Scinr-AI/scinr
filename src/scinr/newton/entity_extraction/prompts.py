"""
entity_extraction/prompts.py — LLM prompts for the entity extraction stage.
"""
from __future__ import annotations


def _has_complementary_fields(schema: type) -> bool:
    """Return True if the schema has any Optional[BaseModel] fields (complementary models)."""
    import typing
    for field_info in schema.model_fields.values():
        ann = field_info.annotation
        # Unwrap Annotated[X, ...]
        if typing.get_origin(ann) is typing.Annotated:
            ann = typing.get_args(ann)[0]
        args = typing.get_args(ann)
        # Only consider Optional (Union with None) fields — required bare-model
        # fields are the primary field of the composite and must be skipped.
        if type(None) not in args:
            continue  # required field — not a complementary model slot
        for arg in args:
            if arg is type(None):
                continue
            if isinstance(arg, type) and hasattr(arg, "model_fields"):
                return True
    return False


def build_extraction_system_prompt(composite_schema: type) -> str:
    """
    Build the system prompt for the entity extraction LLM call.

    Injects a description of the composite schema (field names, types, descriptions)
    so the LLM understands what to extract.

    Parameters
    ----------
    composite_schema:
        The dynamically created composite Pydantic class.
    """
    schema_description = _build_schema_description(composite_schema)

    base_rules = """## Extraction Rules
1. Extract ONLY information explicitly stated in the provided text. Do NOT infer, fabricate, or hallucinate values.
2. If a field's information is not present in the text, leave it as null or empty list (never guess).
3. Preserve exact values from the text — do not paraphrase or normalise units, names, or measurements.
4. For list fields, extract all instances mentioned in the text, not just the first one.
5. The text may be fragmented (multiple information units) — read all of it before extracting.
6. For discriminated union fields (substance_type: 'nce' | 'biotech'), choose based on explicit cues in the text."""

    complementary_rule = ""
    if _has_complementary_fields(composite_schema):
        complementary_rule = """
7. This schema contains COMPLEMENTARY sub-models (the optional nested fields).
   Treat them as strongly preferred: populate every sub-field you can find evidence
   for in the text. Only return null for a complementary sub-model if NONE of its
   fields appear anywhere in the provided text."""

    return f"""You are a precise information extraction system for pharmaceutical regulatory dossiers.

Your task is to extract structured data from the provided document content according to the schema below.

{base_rules}{complementary_rule}

## Schema to extract
{schema_description}

Respond ONLY with the structured extraction. Do not add explanatory text."""


def build_extraction_human_message(info_units: list[dict]) -> str:
    """
    Build the human message containing the ordered InfoUnit content.

    Parameters
    ----------
    info_units:
        List of InfoUnit dicts ordered by .order, each with title and description.
    """
    lines = ["<document_content>"]
    for iu in info_units:
        order = iu.get("order", "?")
        title = iu.get("title") or ""
        description = iu.get("description") or ""
        lines.append(f"  <info_unit order=\"{order}\">")
        lines.append(f"    <title>{title}</title>")
        lines.append(f"    <description>{description}</description>")
        lines.append("  </info_unit>")
    lines.append("</document_content>")
    lines.append("")
    lines.append("Extract all information from the above content according to the provided schema.")
    return "\n".join(lines)


def _build_schema_description(schema: type) -> str:
    """
    Build a human-readable description of a Pydantic model's fields for injection
    into the system prompt.

    Recurses one level into nested models regardless of how they are wrapped:
    bare class, list[T], T | None, Optional[T], or discriminated union T | U.
    """

    if not hasattr(schema, "model_fields"):
        return f"Extract data as: {schema.__name__}"

    lines = [f"Schema: {schema.__name__}"]
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        type_str = _annotation_to_str(annotation)
        description = field_info.description or ""
        required = field_info.is_required()
        req_marker = "(required)" if required else "(optional)"
        lines.append(f"  - {field_name} [{type_str}] {req_marker}: {description[:120]}")

        # Resolve the inner model(s) — handles bare class, list[T], T|None, unions
        inner = (
            annotation
            if hasattr(annotation, "model_fields")
            else _extract_inner_model(annotation)
        )
        if inner is None:
            continue

        is_list = _is_list_annotation(annotation)

        if isinstance(inner, list):
            # Discriminated union: show each variant separately
            for variant in inner:
                lines.append(f"    Fields of {variant.__name__} variant:")
                for sub_name, sub_info in variant.model_fields.items():
                    sub_type = _annotation_to_str(sub_info.annotation)
                    sub_desc = (sub_info.description or "")[:80]
                    lines.append(f"      - {sub_name} [{sub_type}]: {sub_desc}")
        else:
            # Single model: list[T] → "Fields of each T item" or T|None → "Fields of T"
            label = (
                f"Fields of each {inner.__name__} item"
                if is_list
                else f"Fields of {inner.__name__}"
            )
            lines.append(f"    {label}:")
            for sub_name, sub_info in inner.model_fields.items():
                sub_type = _annotation_to_str(sub_info.annotation)
                sub_desc = (sub_info.description or "")[:80]
                lines.append(f"      - {sub_name} [{sub_type}]: {sub_desc}")

    return "\n".join(lines)


def _extract_inner_model(annotation) -> type | list[type] | None:
    """
    Extract the inner Pydantic model class(es) from a complex annotation.

    Returns:
        - A single class  → list[T] or T | None  (T has model_fields)
        - A list of classes → discriminated union T | U (all have model_fields)
        - None → no Pydantic model extractable (primitive, list[str], etc.)
    """
    import types as builtin_types
    import typing

    if annotation is None:
        return None

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Annotated[T, ...] — unwrap the metadata wrapper and recurse
    if origin is typing.Annotated:
        return _extract_inner_model(args[0]) if args else None

    # list[T] — extract T if it is a Pydantic model
    if origin is list:
        if args and hasattr(args[0], "model_fields"):
            return args[0]
        return None

    # Union types: handles both typing.Union[X, Y] and Python 3.10+ X | Y syntax
    is_union = origin is typing.Union or (
        hasattr(builtin_types, "UnionType")
        and isinstance(annotation, builtin_types.UnionType)
    )
    if is_union:
        models: list[type] = []
        for arg in args:
            if arg is type(None):
                continue
            if hasattr(arg, "model_fields"):
                models.append(arg)
            else:
                # Handles Annotated[T | U, ...] nested inside a Union
                inner = _extract_inner_model(arg)
                if isinstance(inner, list):
                    models.extend(inner)
                elif inner is not None:
                    models.append(inner)
        if len(models) == 1:
            return models[0]
        if len(models) > 1:
            return models
        return None

    return None


def _is_list_annotation(annotation) -> bool:
    """Return True if the annotation represents a list[T] (possibly inside Annotated)."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        args = typing.get_args(annotation)
        return _is_list_annotation(args[0]) if args else False
    return origin is list


def _annotation_to_str(annotation) -> str:
    """Convert a type annotation to a compact readable string."""
    import typing

    if annotation is None:
        return "Any"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Annotated:
        return _annotation_to_str(args[0]) if args else "Any"
    if origin is list:
        inner = _annotation_to_str(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin is not None:
        # Union / Optional — render each arg by name
        rendered = [_annotation_to_str(a) for a in args]
        return " | ".join(rendered)
    if annotation is type(None):
        return "None"
    return getattr(annotation, "__name__", str(annotation))
