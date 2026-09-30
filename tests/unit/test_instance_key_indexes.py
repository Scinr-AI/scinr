"""
tests/unit/test_instance_key_indexes.py — (tenant_id, <key>) indexes on
:ModelInstance for instance_key fields, and the composite tenant indexes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from scinr.newton.annotation.neo4j_ops import (
    _ensure_instance_key_indexes,
    instance_key_index_name,
)
from scinr.newton.ingest.schema import _REGULAR_INDEXES


def test_identifier_names_map_directly():
    assert instance_key_index_name("code") == "idx_mi_key_code"
    assert instance_key_index_name("variation_code") == "idx_mi_key_variation_code"


def test_non_identifier_names_are_sanitised_with_hash_suffix():
    a = instance_key_index_name("a-b")
    b = instance_key_index_name("a b")
    assert a.startswith("idx_mi_key_a_b_") and b.startswith("idx_mi_key_a_b_")
    assert a != b
    assert all(ch.isalnum() or ch == "_" for ch in a)


async def test_one_index_per_distinct_key_property():
    session = AsyncMock()
    await _ensure_instance_key_indexes(session, {"code", "variation_code"})
    queries = [c.args[0] for c in session.run.call_args_list]
    assert queries == [
        "CREATE INDEX idx_mi_key_code IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.tenant_id, mi.`code`)",
        "CREATE INDEX idx_mi_key_variation_code IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.tenant_id, mi.`variation_code`)",
    ]


async def test_property_backticks_are_escaped():
    session = AsyncMock()
    await _ensure_instance_key_indexes(session, {"we`ird"})
    assert "mi.`we``ird`" in session.run.call_args_list[0].args[0]


async def test_index_failure_does_not_raise(caplog):
    session = AsyncMock()
    session.run.side_effect = RuntimeError("no schema privilege")
    await _ensure_instance_key_indexes(session, {"code"})
    assert "Could not create index idx_mi_key_code" in caplog.text


def test_composite_tenant_indexes_are_declared():
    names = {name for name, _ in _REGULAR_INDEXES}
    for expected in (
        "idx_model_instance_tenant_model_class",
        "idx_document_tenant_latest",
        "idx_document_tenant_name",
        "idx_structure_node_tenant_role",
        "idx_labeled_entity_tenant_label",
    ):
        assert expected in names
    # The single tenant indexes stay: tenant_id=None queries cannot use a composite.
    assert "idx_model_instance_tenant_id" in names
