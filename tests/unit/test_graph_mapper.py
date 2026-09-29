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

from pydantic import BaseModel, Field, create_model

from scinr.newton.entity_extraction.graph_mapper import (
    _merge_labeled_entity,
    _stringify_if_dict,
    _write_model_fields,
)


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


# ---------------------------------------------------------------------------
# Multi-tenant isolation — tenant_id is folded into the uid of every
# merge-deduplicated node (ModelInstance with instance_key, LabeledEntity,
# Entity), and provenance is written on every node.
# ---------------------------------------------------------------------------


class _Country(BaseModel):
    code: str = Field(json_schema_extra={"instance_key": True})
    name: str = Field(json_schema_extra={"entity_label": "CountryName"})


class _Wrapper(BaseModel):
    country: _Country
    related: list[str] = Field(json_schema_extra={"instance_relationships": [
        {"target_model": "_Country", "join_via": {"related": "code"}, "rel_type": "RELATED"}
    ]})


async def _merged_uids(tenant_id: str | None, user: str = "u1", job: str = "j1") -> tuple[dict, list]:
    """Run _write_model_fields for the same content and return
    {cypher-kind: uid} for every MERGE'd node plus the raw calls."""
    session = AsyncMock()
    await _write_model_fields(
        session=session,
        instance=_Wrapper(country=_Country(code="ES", name="Spain"), related=["FR"]),
        parent_uid="er",
        parent_label="ExtractionResult",
        field_path_prefix="",
        entity_nodes={},
        depth=0,
        tenant_id=tenant_id,
        created_by_user_id=user,
        job_id=job,
    )
    uids: dict[str, str] = {}
    for call in session.run.call_args_list:
        q, kw = call.args[0], call.kwargs
        if "MERGE (child:ModelInstance" in q:
            uids["mi_child"] = kw["child_uid"]
        elif "MERGE (tgt:ModelInstance" in q:
            uids["mi_target"] = kw["tgt_uid"]
        elif "MERGE (le:LabeledEntity" in q:
            uids["labeled_entity"] = kw["uid"]
    return uids, session.run.call_args_list


async def test_same_content_different_tenants_never_share_merged_nodes() -> None:
    a, _ = await _merged_uids("tenant_A")
    b, _ = await _merged_uids("tenant_B")
    assert set(a) == {"mi_child", "mi_target", "labeled_entity"}
    for kind in a:
        assert a[kind] != b[kind], f"{kind} collides across tenants"


async def test_same_tenant_different_job_dedups_to_same_nodes() -> None:
    first, _ = await _merged_uids("tenant_A", user="u1", job="j1")
    second, _ = await _merged_uids("tenant_A", user="u2", job="j2")
    assert first == second


async def test_merged_nodes_write_provenance_on_create_and_on_match() -> None:
    _, calls = await _merged_uids("tenant_A", user="u1", job="j1")
    merges = [c for c in calls if "MERGE (child:ModelInstance" in c.args[0]
              or "MERGE (tgt:ModelInstance" in c.args[0]
              or "MERGE (le:LabeledEntity" in c.args[0]]
    assert len(merges) == 3
    for c in merges:
        q = c.args[0]
        on_match = q.split("ON MATCH", 1)[1]
        assert ".tenant_id = $tenant_id" in on_match
        assert ".created_by_user_ids = CASE" in on_match
        assert ".job_ids = CASE" in on_match
        assert c.kwargs["tenant_id"] == "tenant_A"
        assert c.kwargs["created_by_user_id"] == "u1"
        assert c.kwargs["job_id"] == "j1"


async def test_labeled_entity_is_merged_by_tenant_scoped_uid() -> None:
    """MERGE must match on the tenant-scoped uid, not on (label, normalized_value) —
    otherwise two tenants' entities would still collapse onto one node."""
    session = AsyncMock()
    uid_a = await _merge_labeled_entity(session, "CountryName", "Spain", tenant_id="tenant_A")
    uid_b = await _merge_labeled_entity(session, "CountryName", "Spain", tenant_id="tenant_B")
    assert uid_a != uid_b
    for call in session.run.call_args_list:
        assert "MERGE (le:LabeledEntity {uid: $uid})" in call.args[0]


# ---------------------------------------------------------------------------
# Labelled lookups — every MATCH on the parent / relationship source carries
# its label, so it is a seek on the uid constraint, not a whole-graph scan.
# ---------------------------------------------------------------------------


async def test_parent_matches_carry_the_parent_label() -> None:
    _, calls = await _merged_uids("tenant_A")
    parent_matches = [c.args[0] for c in calls if "MATCH (parent" in c.args[0]]
    assert parent_matches, "expected MATCHes on the parent node"
    for q in parent_matches:
        assert "MATCH (parent:ExtractionResult {uid: $parent_uid})" in q or (
            "MATCH (parent:ModelInstance {uid: $parent_uid})" in q
        ), q
    # The nested _Country hangs from a ModelInstance parent (its REFERENCES write).
    assert any("MATCH (parent:ModelInstance {uid: $parent_uid})" in q for q in parent_matches)
    assert any("MATCH (parent:ExtractionResult {uid: $parent_uid})" in q for q in parent_matches)


async def test_instance_relationship_source_carries_its_label() -> None:
    _, calls = await _merged_uids("tenant_A")
    src_matches = [c.args[0] for c in calls if "MATCH (src" in c.args[0]]
    assert len(src_matches) == 1
    # _Wrapper is the root model: the relationship source is the ExtractionResult.
    assert "MATCH (src:ExtractionResult {uid: $src_uid})" in src_matches[0]


async def test_write_model_fields_rejects_unknown_parent_label() -> None:
    import pytest

    with pytest.raises(ValueError, match="parent label"):
        await _write_model_fields(
            session=AsyncMock(),
            instance=_Country(code="ES", name="Spain"),
            parent_uid="x",
            parent_label="Document) DETACH DELETE (n",
            field_path_prefix="",
            entity_nodes={},
            depth=0,
        )
