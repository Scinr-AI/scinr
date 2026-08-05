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
import re
import typing
from typing import Any

from pydantic import Field, create_model, field_validator
from pydantic.fields import FieldInfo

log = logging.getLogger(__name__)


# Matches a bare `dict`/`Dict` word reference (case-sensitive spelling
# preserved by construction, since we never rewrite anything BUT the "dict"
# occurrence itself — see _sanitize_dict_types). The \b...\b boundaries
# prevent matching inside longer identifiers such as "Dictionary" or a
# hypothetical "SomeDict" class name.
_DICT_WORD_RE = re.compile(r"\b(?:dict|Dict)\b")


def _find_balanced_bracket_end(s: str, open_idx: int) -> int:
    """
    Given the index of an opening ``[`` character in *s*, scan forward
    counting bracket depth and return the index just past its matching
    ``]`` (i.e. the index at which the balanced ``[...]`` group ends).

    Handles arbitrary nesting (``[[[...]]]``) correctly, unlike a
    non-nesting-aware regex character class such as ``[^\\]]*``. If the
    string is malformed (unbalanced), falls back to consuming to the end
    of the string rather than raising.
    """
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n  # unbalanced input — safe fallback, consume to end of string


def _sanitize_dict_types(type_str: str) -> str:
    """
    Rewrite a field_type string so that no `dict`/`Dict` reference survives,
    since Neo4j cannot store nested Maps as node/relationship properties.

    Design decision (bracket-balancing, not regex-only):
    Any occurrence of the word ``dict``/``Dict`` is replaced by ``str``
    together with its *entire* type-parameter group, if any — e.g. for
    ``dict[str, Any]`` the whole ``dict[str, Any]`` span becomes ``str``.
    The type-parameter group is located by counting matching ``[``/``]``
    pairs (manual balancing, not a regex like ``\\[[^\\]]*\\]``, which
    cannot correctly handle nested brackets and would stop at the first
    ``]`` it sees — breaking on inputs like ``dict[str, list[dict]]``).

    Critically, this means a `dict[...]` block is sanitized to `str` in its
    entirety *regardless of what it contains internally* — including cases
    where it contains further nested containers or even other dict/Dict
    references (e.g. ``dict[str, list[dict]]`` -> ``str``, not some attempt
    to preserve/rebuild its inner structure as a type). This is intentional:
    conceptually, any `dict[...]`, no matter how deeply parameterized, is
    still "a nested object" that Neo4j cannot store as a property — so there
    is nothing useful to preserve inside it, and shrinking the whole block
    down to `str` is the only sound outcome.

    Everything *outside* a matched `dict`/`Dict` span — notably an enclosing
    ``list[...]``/``List[...]`` wrapper, ``| None`` suffix, or
    ``Optional[...]``/``Union[...]`` wrapper — is left completely untouched
    (copied through verbatim), which is what naturally turns e.g.
    ``list[dict[str, Any]]`` into ``list[str]`` and
    ``Optional[dict[str, list[dict]]]`` into ``Optional[str]`` without any
    special-cased "list-of-dict" handling: the outer wrapper was simply never
    part of the matched span in the first place.

    - ``list[dict]`` / ``List[Dict]`` / ``list[dict[str, Any]]`` (optionally
      wrapped in ``| None`` or ``Optional[...]``) become ``list[str]`` /
      ``List[str]`` — the ``list``/``List`` wrapper and any surrounding
      ``| None`` / ``Optional[...]`` are preserved.
    - A top-level ``dict`` / ``Dict`` / ``dict[str, Any]`` (optionally wrapped
      in ``| None`` or ``Optional[...]``) becomes ``str`` — again preserving
      the optional wrapper.
    - Strings that do not reference dict/Dict at all are returned unchanged.

    Parameters
    ----------
    type_str:
        The raw field_type string as supplied by supplementary_fields.

    Returns
    -------
    str
        The sanitized type string, safe to eval() against the safe_ns used
        by _parse_type_string.
    """
    parts: list[str] = []
    i = 0
    n = len(type_str)
    while i < n:
        match = _DICT_WORD_RE.match(type_str, i)
        if match is None:
            parts.append(type_str[i])
            i += 1
            continue

        # Found a "dict"/"Dict" word — check whether it is immediately
        # (modulo whitespace) followed by a "[...]" type-parameter group,
        # and if so, consume that whole balanced group as part of the span.
        j = match.end()
        while j < n and type_str[j] == " ":
            j += 1
        span_end = _find_balanced_bracket_end(type_str, j) if j < n and type_str[j] == "[" else match.end()

        parts.append("str")
        i = span_end

    sanitized = "".join(parts)

    if sanitized != type_str:
        log.info(
            "schema_composer: sanitized dict-typed field_type %r -> %r "
            "(Neo4j cannot store nested Maps as properties)",
            type_str,
            sanitized,
        )
    return sanitized


def _coerce_dict_like_to_str(value: Any) -> Any:
    """
    Defensive coercion: if *value* is a dict, or a list containing dicts,
    flatten each dict into a human-readable "key: value; key2: value2" string.
    Any non-dict item in a list is left untouched. Non-dict, non-list values
    pass through unchanged. Used as a last-resort safety net so that content
    the annotation/extraction pipeline still emits as a nested object (despite
    the declared field type being sanitized to str/list[str] in
    _sanitize_dict_types) never reaches Neo4j as an unsupported Map property.
    """

    def _stringify_dict(d: dict) -> str:
        return "; ".join(f"{k}: {v}" for k, v in d.items())

    coerced = False

    if isinstance(value, dict):
        result = _stringify_dict(value)
        coerced = True
    elif isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(_stringify_dict(item))
                coerced = True
            else:
                result.append(item)
    else:
        result = value

    if coerced:
        log.warning(
            "schema_composer: coerced unexpected dict/list[dict] value to string "
            "(Neo4j cannot store nested Maps) — original type: %s",
            type(value).__name__,
        )
    return result


def _parse_type_string(type_str: str) -> Any:
    """
    Convert a field_type string like 'list[str]', 'str | None', 'int' into
    a real Python type annotation.

    Supports a safe subset of common type expressions only.
    Falls back to Any on parse failure.
    """
    type_str = _sanitize_dict_types(type_str)

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
    supplementary_field_names: list[str] = []
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
        supplementary_field_names.append(fname)
        log.debug("schema_composer: supplementary field '%s' → %s", fname, ftype_str)

    # Attach a defensive mode="before" validator to every supplementary field
    # (never to primary_class/complementary_classes fields, which are already
    # well-typed by the catalog) so that a dict/list[dict] value that still
    # slips through despite the sanitized field type is flattened to a string
    # instead of raising a ValidationError.
    validators: dict[str, Any] = {}
    if supplementary_field_names:
        @field_validator(*supplementary_field_names, mode="before")
        @classmethod
        def _coerce_supplementary_dicts(cls, v: Any) -> Any:  # noqa: N805
            return _coerce_dict_like_to_str(v)

        validators["_coerce_supplementary_dicts"] = _coerce_supplementary_dicts

    # Create the dynamic model
    composite_name = f"Composite_{primary_class.__name__}"
    CompositeModel = create_model(
        composite_name, __validators__=validators or None, **field_definitions
    )
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
