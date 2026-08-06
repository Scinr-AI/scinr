"""
tests/unit/test_schema_composer.py — Regression tests for the Neo4j
"nested Map property" fix in scinr.newton.entity_extraction.schema_composer.

Context: the annotation agent's `supplementary_fields` mechanism can propose
`field_type: "dict"` or `field_type: "list[dict]"` for dynamically-detected
tabular content (e.g. INID code tables, translation glossaries, jurisprudence
annexes) that isn't covered by any catalog model. Neo4j cannot store a raw
dict/Map as a node property (`Neo.ClientError.Statement.TypeError`), so:

  1. `_sanitize_dict_types()` rewrites the declared field_type string so no
     `dict`/`Dict` reference survives (dict -> str, list[dict] -> list[str]),
     preserving `| None` / `Optional[...]` wrappers.
  2. `compose_extraction_schema()` additionally attaches a `mode="before"`
     field_validator to every supplementary field so that, even if a dict/
     list[dict] value slips through despite the sanitized type, it is
     flattened to a string instead of raising a ValidationError.

These tests cover both layers.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from scinr.newton.entity_extraction.schema_composer import (
    _coerce_dict_like_to_str,
    _sanitize_dict_types,
    compose_extraction_schema,
)

# ---------------------------------------------------------------------------
# _sanitize_dict_types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "type_str, expected",
    [
        ("dict", "str"),
        ("Dict", "str"),
        ("dict[str, Any]", "str"),
        ("dict | None", "str | None"),
        ("Optional[dict]", "Optional[str]"),
        ("list[dict]", "list[str]"),
        ("List[Dict]", "List[str]"),
        ("list[dict[str, Any]]", "list[str]"),
        ("list[dict] | None", "list[str] | None"),
        ("Optional[list[dict]]", "Optional[list[str]]"),
    ],
)
def test_sanitize_dict_types_rewrites_dict_references(type_str: str, expected: str) -> None:
    assert _sanitize_dict_types(type_str) == expected


@pytest.mark.parametrize(
    "type_str",
    [
        "list[str]",
        "str | None",
        "int",
        "list[str] | None",
        "float",
        "bool | None",
        "Any",
    ],
)
def test_sanitize_dict_types_leaves_non_dict_types_untouched(type_str: str) -> None:
    assert _sanitize_dict_types(type_str) == type_str


@pytest.mark.parametrize(
    "type_str, expected",
    [
        # 2-level nesting: dict[str, list[dict]] — a bare `_BARE_DICT_RE`
        # second pass (with a non-nesting-aware `\[[^\]]*\]` group) breaks
        # on this input (stops at the first `]`, which actually closes the
        # *inner* list[dict], producing a mangled "str]" that fails eval()
        # and silently falls back to Any via the generic except-Exception
        # in _parse_type_string — "correct" only by accident of the
        # catch-all, not by design). The whole dict[...] block, regardless
        # of what it contains, is not representable in Neo4j, so it must be
        # sanitized to `str` in its entirety.
        ("dict[str, list[dict]]", "str"),
        # A dict[...] nested *inside* a list[...] wrapper: the dict[...]
        # block (even though its own inner content is an unremarkable
        # list[str]) still makes the whole object non-representable in
        # Neo4j, so it is sanitized the same way; the outer list[...]
        # wrapper is preserved untouched.
        ("list[dict[str, list[str]]]", "list[str]"),
        # Same 2-level nesting again, this time wrapped in Optional[...].
        ("Optional[dict[str, list[dict]]]", "Optional[str]"),
    ],
)
def test_sanitize_dict_types_handles_nested_dict_type_params(
    type_str: str, expected: str
) -> None:
    assert _sanitize_dict_types(type_str) == expected


def test_sanitize_dict_types_does_not_touch_dict_like_class_names() -> None:
    """
    A hypothetical class name that merely *contains* the substring "dict"
    (e.g. a future "SomeDict" catalog model, or "OrderedDict"/"Dictionary")
    must not be mangled by the word-boundary-based substitution.
    """
    assert _sanitize_dict_types("SomeDict") == "SomeDict"
    assert _sanitize_dict_types("OrderedDict") == "OrderedDict"
    assert _sanitize_dict_types("Dictionary") == "Dictionary"


def test_sanitize_dict_types_logs_info_only_on_substitution(caplog) -> None:
    caplog.set_level("INFO", logger="scinr.newton.entity_extraction.schema_composer")

    caplog.clear()
    _sanitize_dict_types("list[str]")
    assert not any("sanitized dict-typed" in r.message for r in caplog.records)

    caplog.clear()
    _sanitize_dict_types("list[dict]")
    assert any("sanitized dict-typed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _coerce_dict_like_to_str
# ---------------------------------------------------------------------------


def test_coerce_dict_like_to_str_flattens_dict() -> None:
    result = _coerce_dict_like_to_str({"a": "1", "b": "2"})
    assert result == "a: 1; b: 2"


def test_coerce_dict_like_to_str_flattens_dicts_in_list() -> None:
    result = _coerce_dict_like_to_str([{"a": "1"}, "plain string", {"b": "2"}])
    assert result == ["a: 1", "plain string", "b: 2"]


def test_coerce_dict_like_to_str_passes_through_non_dict_values() -> None:
    assert _coerce_dict_like_to_str("already a string") == "already a string"
    assert _coerce_dict_like_to_str(["a", "b"]) == ["a", "b"]
    assert _coerce_dict_like_to_str(None) is None
    assert _coerce_dict_like_to_str(42) == 42


# ---------------------------------------------------------------------------
# compose_extraction_schema — integration
# ---------------------------------------------------------------------------


class PrimaryModel(BaseModel):
    name: str = Field(default="x")


def test_compose_extraction_schema_list_dict_supplementary_field() -> None:
    """
    A supplementary_field declared as field_type="list[dict]" must produce a
    composite model field typed list[str] (sanitized), which accepts both a
    genuine list[str] AND a defensive list[dict] value (coerced via the
    mode="before" validator) without raising ValidationError.
    """
    Model = compose_extraction_schema(
        primary_class=PrimaryModel,
        supplementary_fields=[
            {
                "field_name": "inid_codes",
                "field_type": "list[dict]",
                "description": "INID code table rows",
                "required": False,
            }
        ],
    )

    # Declared annotation was sanitized to list[str] | None
    assert Model.model_fields["inid_codes"].annotation == list[str] | None

    # Normal list[str] value works as expected
    inst = Model(primary_model={"name": "x"}, inid_codes=["11", "21"])
    assert inst.inid_codes == ["11", "21"]

    # A real list[dict] value (LLM still emitting nested objects) is coerced,
    # not rejected.
    inst2 = Model(
        primary_model={"name": "x"},
        inid_codes=[{"code": "11", "meaning": "Country"}, "plain"],
    )
    assert inst2.inid_codes == ["code: 11; meaning: Country", "plain"]


def test_compose_extraction_schema_dict_supplementary_field() -> None:
    """
    A supplementary_field declared as field_type="dict" must produce a
    composite model field typed str (sanitized), which accepts both a
    genuine string AND a defensive dict value (coerced via the
    mode="before" validator) without raising ValidationError.
    """
    Model = compose_extraction_schema(
        primary_class=PrimaryModel,
        supplementary_fields=[
            {
                "field_name": "glossary",
                "field_type": "dict",
                "description": "Translation glossary",
                "required": False,
            }
        ],
    )

    assert Model.model_fields["glossary"].annotation == str | None

    inst = Model(primary_model={"name": "x"}, glossary="already a string")
    assert inst.glossary == "already a string"

    inst2 = Model(primary_model={"name": "x"}, glossary={"en": "hello", "es": "hola"})
    assert inst2.glossary == "en: hello; es: hola"


def test_compose_extraction_schema_validator_not_applied_to_primary_or_complementary() -> None:
    """
    The defensive mode="before" validator must only be attached to
    supplementary_fields — primary_class / complementary_classes fields are
    already well-typed by the catalog and must keep raising ValidationError
    for genuinely malformed input (a bare int where a BaseModel is expected).
    """

    class ComplementaryModel(BaseModel):
        value: str = Field(default="y")

    Model = compose_extraction_schema(
        primary_class=PrimaryModel,
        complementary_classes=[ComplementaryModel],
    )
    with pytest.raises((ValueError, TypeError)):
        Model(primary_model={"name": "x"}, complementary_model=123)
