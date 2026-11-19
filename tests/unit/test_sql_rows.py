"""
tests/unit/test_sql_rows.py -- ``sql.rows``: node rows <-> (labels, properties).
"""

from __future__ import annotations

import pytest

from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.sql import tables
from scinr.newton.sql.names import LABEL_TABLE
from scinr.newton.sql.rows import props_to_row, row_to_node


def test_every_node_table_is_declared_once() -> None:
    assert set(LABEL_TABLE.values()) == set(tables.NODE_TABLES)
    assert len(set(tables.NODE_TABLES)) == len(tables.NODE_TABLES) == 16


def test_props_split_into_typed_columns_and_props_json() -> None:
    row = props_to_row(
        "info_units",
        {"uid": "u", "tenant_id": "t", "title": "T", "extracted": True, "gone": None, "n": 3},
    )
    assert row == {
        "uid": "u",
        "tenant_id": "t",
        "title": "T",
        "props": {"extracted": True, "n": 3},
    }


def test_props_is_always_present_and_never_holds_typed_keys() -> None:
    row = props_to_row("entities", {"uid": "e", "tenant_id": "t", "value": "v"})
    assert row["props"] == {}
    for table, typed in tables.TYPED_COLUMNS.items():
        assert not typed & set(props_to_row(table, {c: 1 for c in typed})["props"])


def test_non_json_value_in_props_is_a_type_error() -> None:
    from datetime import datetime

    with pytest.raises(TypeError):
        props_to_row("model_instances", {"uid": "m", "when": datetime(2026, 1, 1)})


def test_round_trip_per_table() -> None:
    for table, typed in tables.TYPED_COLUMNS.items():
        props = {c: f"v-{c}" for c in typed} | {"custom": {"a": [1, 2]}, "flag": False}
        for column in tables.LIST_COLUMNS.get(table, ()):
            props[column] = [f"{column}-1", f"{column}-2"]
        if table == "structure_nodes":
            props["role"] = "section"
        _, back = row_to_node(table, props_to_row(table, props))
        assert back == props, table


def test_null_columns_are_absent_properties() -> None:
    _, props = row_to_node("documents", {"_key": "k", "tenant_id": "t", "name": None, "props": {}})
    assert props == {"tenant_id": "t"}


def test_auxiliary_columns_are_never_returned() -> None:
    row = {
        "uid": "u",
        "tenant_id": "t",
        "_document_key": "d",
        "_structure_node_id": "s",
        "props": {},
    }
    _, props = row_to_node("info_units", row)
    assert props == {"uid": "u", "tenant_id": "t"}


def test_lists_come_back_as_lists_and_empty_stays_empty() -> None:
    _, props = row_to_node(
        "model_instances",
        {"uid": "u", "created_by_user_ids": ("a", "b"), "job_ids": [], "props": {}},
    )
    assert props["created_by_user_ids"] == ["a", "b"] and isinstance(
        props["created_by_user_ids"], list
    )
    assert props["job_ids"] == []
    _, absent = row_to_node("model_instances", {"uid": "u", "job_ids": None, "props": {}})
    assert "job_ids" not in absent


@pytest.mark.parametrize("role, label", ROLE_TO_LABEL.items())
def test_structure_node_gets_its_role_label_after_the_primary_one(role: str, label: str) -> None:
    labels, _ = row_to_node("structure_nodes", {"id": "i", "role": role, "props": {}})
    assert labels == ("StructureNode", label)


def test_structure_node_without_a_known_role_has_only_the_primary_label() -> None:
    assert row_to_node("structure_nodes", {"id": "i", "props": {}})[0] == ("StructureNode",)
    assert row_to_node("structure_nodes", {"id": "i", "role": "weird", "props": {}})[0] == (
        "StructureNode",
    )


def test_primary_labels_of_the_other_tables() -> None:
    for label, table in LABEL_TABLE.items():
        if table != "structure_nodes":
            assert row_to_node(table, {"props": {}})[0] == (label,)
