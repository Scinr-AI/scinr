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
required. The ones that run the Cypher of the sink (`Neo4jExtractionSink`, in
``graph/neo4j/extraction.py``) are in ``tests/unit/graph_neo4j/test_extraction_ops.py``;
the emitted sink calls are in ``tests/unit/test_graph_mapper_traversal.py``.
"""

from __future__ import annotations

from scinr.newton.entity_extraction.graph_mapper import _stringify_if_dict


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
