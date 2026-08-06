"""
tests/unit/test_graph_mapper.py — Unit tests for pure helper functions in
scinr.newton.entity_extraction.graph_mapper.

`_stringify_if_dict()` is the third and final defense layer against Neo4j's
"Property values can only be of primitive types or arrays thereof" error:
even if a raw dict slips past the schema_composer.py field-type sanitization
and mode="before" validator, `_write_model_fields()` flattens any stray dict
value into a human-readable string right before writing it as a Neo4j scalar
property, instead of letting the whole `write_extraction_subgraph()` call
fail (and losing all extracted data for that node).

These tests exercise `_stringify_if_dict()` directly — no Neo4j driver
required.

The end-to-end test below exercises the `list`-case fix in
`_write_model_fields()` (`scalar_props[field_name] = scalarValues`, which
previously read `= value` — i.e. it wrote the raw, unflattened list instead
of the dict-flattened `scalarValues` accumulator built during the loop) via
a mocked Neo4j session, without touching a real driver.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from pydantic import create_model

from scinr.newton.entity_extraction.graph_mapper import _stringify_if_dict, _write_model_fields


def test_stringify_if_dict_flattens_dict() -> None:
    result = _stringify_if_dict({"code": "11", "meaning": "Country"})
    assert result == "code: 11; meaning: Country"


def test_stringify_if_dict_leaves_string_untouched() -> None:
    assert _stringify_if_dict("already a string") == "already a string"


def test_stringify_if_dict_leaves_int_untouched() -> None:
    assert _stringify_if_dict(42) == 42


def test_stringify_if_dict_leaves_none_untouched() -> None:
    assert _stringify_if_dict(None) is None


def test_stringify_if_dict_leaves_bool_and_float_untouched() -> None:
    assert _stringify_if_dict(True) is True
    assert _stringify_if_dict(3.14) == 3.14


def test_stringify_if_dict_on_mixed_list_only_transforms_dict_items() -> None:
    """
    _stringify_if_dict() itself only handles a single value (dict or not) —
    the list-item-level iteration happens in _write_model_fields(). This test
    documents that behavior explicitly: applying it item-by-item over a
    mixed list transforms only the dict entries, leaving strings untouched.
    """
    items = [{"code": "11"}, "plain string", {"code": "21", "note": "x"}]
    result = [_stringify_if_dict(item) for item in items]
    assert result == ["code: 11", "plain string", "code: 21; note: x"]


# ---------------------------------------------------------------------------
# _write_model_fields — end-to-end regression test for the
# `scalar_props[field_name] = scalarValues` fix (previously `= value`)
# ---------------------------------------------------------------------------


async def test_write_model_fields_writes_flattened_scalar_values_for_mixed_list() -> None:
    """
    Regression test for the `_write_model_fields()` list-case bug: a `list`
    field whose real runtime value mixes a raw dict, a plain string, and a
    None (simulating something that reaches this function despite not being
    caught by the schema_composer.py layer-1/layer-2 defenses — e.g. a
    catalog model field that isn't a supplementary field at all) must be
    written to Neo4j using the *flattened* `scalarValues` accumulator (dict
    -> "k: v" via `_stringify_if_dict`, None dropped), not the raw original
    `value` list. The buggy version wrote `scalar_props[field_name] = value`,
    i.e. the untouched `[{"a": "1"}, "plain", None]`, which would have
    raised `Neo.ClientError.Statement.TypeError` against a real Neo4j
    session because Neo4j properties cannot contain dicts/None inside a
    list.

    Uses a dynamically-created Pydantic model (no dependency on any real
    catalog model) and a minimal `AsyncMock` session — no real Neo4j driver.
    """
    DynamicModel = create_model("DynamicModel", mixed_field=(list[Any], ...))
    instance = DynamicModel(mixed_field=[{"a": "1"}, "plain", None])

    mock_session = AsyncMock()

    # Should not raise.
    await _write_model_fields(
        session=mock_session,
        instance=instance,
        parent_uid="test-uid",
        parent_label="ExtractionResult",
        field_path_prefix="",
        entity_nodes={},
        depth=0,
    )

    # Locate the final batch-SET call: it is the one whose Cypher query
    # contains "SET" and whose kwargs include our field name.
    set_calls = [
        call
        for call in mock_session.run.call_args_list
        if "SET" in call.args[0] and "mixed_field" in call.kwargs
    ]
    assert len(set_calls) == 1, (
        f"expected exactly one SET call carrying 'mixed_field', "
        f"got {len(set_calls)} (all calls: {mock_session.run.call_args_list!r})"
    )
    assert set_calls[0].kwargs["mixed_field"] == ["a: 1", "plain"]
    assert set_calls[0].kwargs["parent_uid"] == "test-uid"
