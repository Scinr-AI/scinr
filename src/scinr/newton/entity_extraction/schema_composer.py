"""
entity_extraction/schema_composer.py

Dynamically builds a composite Pydantic model that combines:
  - The primary extraction model (matched_model_class)
  - Zero or more complementary models (from ModelDecision.complementary_models)
  - Zero or more supplementary fields (from ModelDecision.supplementary_fields)

The resulting class is passed to LLM.with_structured_output() so the LLM
extracts all relevant information in a single call.

Usage:
    from entity_extraction.schema_composer import compose_extraction_schema
    CompositeSchema = compose_extraction_schema(
        primary_class=DrugProductComposition,
        complementary_classes=[RegulatoryReference],
        supplementary_fields=[{"field_name": "reg_refs", "field_type": "list[str]", ...}],
    )
"""
from __future__ import annotations

import logging
import typing
from typing import Any

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

log = logging.getLogger(__name__)


def _parse_type_string(type_str: str) -> Any:
    """
    Convert a field_type string like 'list[str]', 'str | None', 'int' into
    a real Python type annotation.

    Supports a safe subset of common type expressions only.
    Falls back to Any on parse failure.
    """

    safe_ns: dict[str, Any] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "Any": Any,
        "None": type(None),
        "Optional": typing.Optional,
        "Union": typing.Union,
        "List": list,
        "Dict": dict,
    }

    cleaned = type_str.strip()
    # Handle "X | None" as Optional[X]
    if " | None" in cleaned or "None | " in cleaned:
        inner = cleaned.replace(" | None", "").replace("None | ", "").strip()
        try:
            inner_type = eval(inner, {"__builtins__": {}}, safe_ns)  # noqa: S307
            return typing.Optional[inner_type]
        except Exception:
            return Any

    try:
        return eval(cleaned, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception:
        log.warning("schema_composer: could not parse type string %r, using Any", type_str)
        return Any


def compose_extraction_schema(
    primary_class: type,
    complementary_classes: list[type] | None = None,
    supplementary_fields: list[dict] | None = None,
) -> type:
    """
    Build and return a dynamic Pydantic model class that wraps:
      - The primary model as a required field
      - Each complementary model as an optional field (snake_case class name)
      - Each supplementary field parsed from its type string

    Parameters
    ----------
    primary_class:
        The primary Pydantic model class for this extraction.
    complementary_classes:
        List of additional Pydantic model classes to include.
    supplementary_fields:
        List of dicts with keys: field_name, field_type, description, required.

    Returns
    -------
    type
        A dynamically created Pydantic BaseModel subclass.
    """
    complementary_classes = complementary_classes or []
    supplementary_fields = supplementary_fields or []

    # Build field definitions: {field_name: (annotation, FieldInfo)}
    field_definitions: dict[str, tuple[Any, FieldInfo]] = {}

    # Primary model — always required
    primary_field_name = _to_snake_case(primary_class.__name__)
    field_definitions[primary_field_name] = (
        primary_class,
        Field(description=f"Primary extraction using {primary_class.__name__}."),
    )
    log.debug("schema_composer: primary field '%s' → %s", primary_field_name, primary_class.__name__)

    # Complementary models — optional
    for cls in complementary_classes:
        field_name = _to_snake_case(cls.__name__)
        # Avoid collision with primary
        if field_name == primary_field_name:
            field_name = f"{field_name}_complementary"
        field_definitions[field_name] = (
            typing.Optional[cls],
            Field(
                default=None,
                description=(
                    f"Extract {cls.__name__} data if present in the text. "
                    f"Populate all sub-fields you can find: {', '.join(cls.model_fields.keys())}. "
                    f"Set to null ONLY if absolutely none of these fields appear anywhere in the text."
                ),
            ),
        )
        log.debug("schema_composer: complementary field '%s' → %s", field_name, cls.__name__)

    # Supplementary fields — parsed from type string
    for sf in supplementary_fields:
        fname = sf.get("field_name", "")
        ftype_str = sf.get("field_type", "Any")
        fdesc = sf.get("description", "")
        frequired = sf.get("required", False)

        if not fname:
            log.warning("schema_composer: skipping supplementary field with empty field_name")
            continue

        parsed_type = _parse_type_string(ftype_str)
        if frequired:
            field_definitions[fname] = (
                parsed_type,
                Field(description=fdesc),
            )
        else:
            field_definitions[fname] = (
                typing.Optional[parsed_type],
                Field(default=None, description=fdesc),
            )
        log.debug("schema_composer: supplementary field '%s' → %s", fname, ftype_str)

    # Create the dynamic model
    composite_name = f"Composite_{primary_class.__name__}"
    CompositeModel = create_model(composite_name, **field_definitions)
    log.info(
        "schema_composer: built %s with fields: %s",
        composite_name,
        list(field_definitions.keys()),
    )
    return CompositeModel


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case. E.g. DrugProductComposition → drug_product_composition."""
    import re
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
