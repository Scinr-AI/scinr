"""
tests/unit/test_sql_names.py -- ``sql.names``: label/table maps and relationship identity.
"""

from __future__ import annotations

import re
from pathlib import Path

import scinr.newton as _newton
from scinr.newton.ingest.freeze import CATALOG_KEY_FIELDS, node_identity
from scinr.newton.sql.names import (
    CATALOG_LABELS,
    DYNAMIC_IDENTITY_PROPS,
    EDGE_IDENTITY_PROPS,
    KEY_COLUMN,
    LABEL_TABLE,
    edge_ident,
)

GRAPH_NEO4J = Path(_newton.__file__).parent / "graph" / "neo4j"
# MERGE (a)-[:TYPE {prop: $x, ...}]->(b), with f-string doubled braces in some modules.
_MERGE_WITH_PROPS = re.compile(r"MERGE \(\w*\)-\[\w*:([A-Z_]+) *\{\{?([^}]*)\}")


def test_label_maps_cover_every_label_with_a_business_key() -> None:
    assert set(LABEL_TABLE) == set(KEY_COLUMN)
    assert CATALOG_LABELS == frozenset(CATALOG_KEY_FIELDS)
    for label in LABEL_TABLE:
        node_identity([label])  # raises for a label the snapshot cannot rebuild


def test_key_column_is_the_business_key_field_except_for_composite_keys() -> None:
    for label, column in KEY_COLUMN.items():
        _, fields = node_identity([label])
        assert column == (fields[0] if len(fields) == 1 else "_key"), label


def test_edge_ident_of_types_without_identity_properties_is_empty() -> None:
    assert edge_ident("HAS_CHILD", {"anything": 1}) == ""


def test_edge_ident_encodes_the_identity_properties_in_order() -> None:
    assert edge_ident("REFERENCES", {"field_name": "f", "list_index": 2, "x": 1}) == '["f",2]'
    assert edge_ident("REFERENCES", {"field_name": "f"}) == '["f",null]'
    assert edge_ident("HAS_ENTITY", {"role": "subject"}) == '["subject"]'
    assert edge_ident("AGGREGATES", {"field_name": "ñ"}) == '["ñ"]'


def test_edge_identity_matches_the_merge_patterns_of_the_neo4j_store() -> None:
    """Every relationship MERGEd with properties in Cypher has exactly those as identity."""
    found: dict[str, set[str]] = {}
    for path in sorted(GRAPH_NEO4J.glob("*.py")):
        for rel_type, body in _MERGE_WITH_PROPS.findall(path.read_text(encoding="utf-8")):
            found.setdefault(rel_type, set()).update(re.findall(r"(\w+)\s*:", body))
    assert found, "the pattern no longer finds any MERGE with properties"
    assert {t: set(props) for t, props in EDGE_IDENTITY_PROPS.items()} == found


# Dynamic relationship types: ``[:`{rel_type}` {{a: $a, ...}}]`` in the Neo4j store.
_DYNAMIC_WITH_PROPS = re.compile(r":`\{\w+\}` *\{\{([^}]*)\}\}")


def test_dynamic_edge_identity_matches_the_merge_patterns_of_the_neo4j_store() -> None:
    """Every dynamic-type relationship MERGEd with properties has a group with exactly those."""
    found: set[tuple[str, ...]] = set()
    for path in sorted(GRAPH_NEO4J.glob("*.py")):
        for body in _DYNAMIC_WITH_PROPS.findall(path.read_text(encoding="utf-8")):
            found.add(tuple(re.findall(r"(\w+)\s*:", body)))
    assert found, "the pattern no longer finds any dynamic MERGE with properties"
    assert found == set(DYNAMIC_IDENTITY_PROPS)


def test_edge_ident_of_dynamic_types() -> None:
    assert edge_ident("lines", {"index": 3}) == "[3]"
    assert edge_ident("lines", {}) == ""
    props = {"via_model": "M", "from_field": "a", "to_field": "b"}
    assert edge_ident("RELATED_TO", props) == '["M","a","b"]'
    assert edge_ident("JOINS", {"via_field": "f", "join_via": "[]", "composite": True}) == (
        '["f","[]"]'
    )
    # Properties set after the MERGE are not identity.
    assert edge_ident("TREATS", {"predicate_raw": "treats"}) == ""
