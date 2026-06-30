"""
entity_extraction/prompts.py — LLM prompt dispatcher and shared utilities
for the entity extraction stage.

Shared utility functions (_build_schema_description, _has_complementary_fields,
build_extraction_human_message, etc.) live here and are imported by both
family-specific prompt modules.

build_extraction_system_prompt() dispatches to the correct variant based on
the configured PromptFamily.
"""
from __future__ import annotations


# ── Shared utility: complementary-field detection ─────────────────────────────

def _has_complementary_fields(schema: type) -> bool:
    """Return True if the schema has any Optional[BaseModel] fields (complementary models)."""
    import typing
    for field_info in schema.model_fields.values():
        ann = field_info.annotation
        if typing.get_origin(ann) is typing.Annotated:
            ann = typing.get_args(ann)[0]
        args = typing.get_args(ann)
        if type(None) not in args:
            continue
        for arg in args:
            if arg is type(None):
                continue
            if isinstance(arg, type) and hasattr(arg, "model_fields"):
                return True
    return False


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def build_extraction_system_prompt(composite_schema: type) -> str:
    """Build the entity extraction system prompt for the configured PromptFamily."""
    from scinr.newton.config import PromptFamily, get_prompt_family

    family = get_prompt_family()
    if family == PromptFamily.CLAUDE:
        from scinr.newton.entity_extraction import prompts_claude as m
    else:
        from scinr.newton.entity_extraction import prompts_generic as m
    return m.build_extraction_system_prompt(composite_schema)


# ── Shared: human message builder ─────────────────────────────────────────────

def build_extraction_human_message(info_units: list[dict]) -> str:
    """Build the human message containing the ordered InfoUnit content."""
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


# ── Shared: schema description builder ────────────────────────────────────────

def _build_schema_description(schema: type) -> str:
    """Build a human-readable description of a Pydantic model's fields."""
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

        inner = (
            annotation
            if hasattr(annotation, "model_fields")
            else _extract_inner_model(annotation)
        )
        if inner is None:
            continue

        is_list = _is_list_annotation(annotation)

        if isinstance(inner, list):
            for variant in inner:
                lines.append(f"    Fields of {variant.__name__} variant:")
                for sub_name, sub_info in variant.model_fields.items():
                    sub_type = _annotation_to_str(sub_info.annotation)
                    sub_desc = (sub_info.description or "")[:80]
                    lines.append(f"      - {sub_name} [{sub_type}]: {sub_desc}")
        else:
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
    """Extract the inner Pydantic model class(es) from a complex annotation."""
    import types as builtin_types
    import typing

    if annotation is None:
        return None

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        return _extract_inner_model(args[0]) if args else None

    if origin is list:
        if args and hasattr(args[0], "model_fields"):
            return args[0]
        return None

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
    """Return True if the annotation represents a list[T]."""
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
        rendered = [_annotation_to_str(a) for a in args]
        return " | ".join(rendered)
    if annotation is type(None):
        return "None"
    return getattr(annotation, "__name__", str(annotation))
