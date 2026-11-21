"""
tests/unit/test_navigation_browse_neo4j.py -- ``get_node`` / ``count_node_relationships`` /
``expand_node`` / ``search_nodes`` of the Neo4j navigator with a canned driver.

No Neo4j: the Cypher is not executed here (the real engine is checked by
``tests/equivalence``, which needs Neo4j). These tests pin what the navigator sends (anchor,
scope, parameters) and what it does with the rows (identity, order, paging, edge ends).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import FakeAsyncDriver, make_fake_llm  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.exceptions import NavigationError  # noqa: E402
from scinr.newton.navigation.neo4j.navigator import Neo4jGraphNavigator  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}
DOC_KEY = '["acme","a.pdf",1]'


def _mk(responder=None) -> tuple[Neo4jGraphNavigator, FakeAsyncDriver]:
    configure(**_NEO)
    drv = FakeAsyncDriver(responder or (lambda cypher, params: []))
    return Neo4jGraphNavigator(driver=drv), drv


def _ident(**props: Any) -> dict[str, Any]:
    keys = ("tenant_id", "path", "version", "id", "name", "model", "label", "uid")
    return {k: props.get(k) for k in keys}


async def test_get_node_matches_by_key_fields_and_scopes_by_the_kind_of_the_label() -> None:
    row = {"n": {"tenant_id": "acme", "path": "a.pdf", "version": 1, "_labels": ["Document"]}}
    nav, drv = _mk(lambda cypher, params: [row])
    node = await nav.get_node("Document", DOC_KEY, tenant_id="acme")
    assert (node.label, node.key) == ("Document", DOC_KEY)
    ((cypher, params),) = drv.last_queries
    assert (
        "(n:`Document` {`tenant_id`: $anchor_0, `path`: $anchor_1, `version`: $anchor_2})" in cypher
    )
    assert "n.tenant_id = $scope_tenant" in cypher
    assert [params[f"anchor_{i}"] for i in range(3)] == ["acme", "a.pdf", 1]


async def test_get_node_catalogue_label_has_no_scope_and_bad_keys_send_no_query() -> None:
    nav, drv = _mk()
    assert await nav.get_node("CatalogModel", "Drug", tenant_id="acme") is None
    ((cypher, _),) = drv.last_queries
    assert "scope_tenant" not in cypher
    drv.sessions.clear()
    for label, key in (("Document", "not-json"), ("Document", '["a","b"]'), ("StructureNode", "")):
        assert await nav.get_node(label, key) is None
    assert drv.last_queries == []
    for bad in ("Nope", "bad label"):
        with pytest.raises(NavigationError):
            await nav.get_node(bad, "x")


async def test_count_merges_role_labels_and_orders_the_rows() -> None:
    def respond(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if "(n)-[r]->(m)" in cypher:
            return [
                {"type": "HAS_CHILD", "labels": ["StructureNode", "Section"], "count": 2},
                {"type": "HAS_CHILD", "labels": ["StructureNode", "Table"], "count": 1},
                {"type": "HAS_FIELD", "labels": ["ModelField"], "count": 1},
                {"type": "X", "labels": ["Mystery"], "count": 9},
            ]
        return [{"type": "HAS_STRUCTURE", "labels": ["Document"], "count": 1}]

    nav, drv = _mk(respond)
    rows = await nav.count_node_relationships("StructureNode", "sn1", tenant_id="acme")
    assert [(r.type, r.direction, r.neighbor_label, r.count) for r in rows] == [
        ("HAS_CHILD", "out", "StructureNode", 3),
        ("HAS_FIELD", "out", "ModelField", 1),
        ("HAS_STRUCTURE", "in", "Document", 1),
    ]
    assert len(drv.last_queries) == 2
    for cypher, params in drv.last_queries:
        assert "n.tenant_id = $scope_tenant" in cypher and "m.tenant_id = $scope_tenant" in cypher
        assert "$scope_catalog_labels" in cypher and params["anchor_0"] == "sn1"


def _expansion_responder(
    neighbours: list[tuple[str, dict[str, Any], str]], edges: list[dict[str, Any]]
):
    def respond(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if "AS anchor" in cypher:
            return [
                {"anchor": "a0", "eid": eid, "label": label, "ident": ident}
                for label, ident, eid in neighbours
            ]
        if "startNode(r)" in cypher:
            return [e for e in edges if e["eid"] in params["eids"]]
        if "RETURN elementId(m) AS eid, m {" in cypher:
            return [
                {"eid": eid, "m": {**ident, "_labels": [label]}}
                for label, ident, eid in neighbours
                if eid in params["eids"]
            ]
        return []

    return respond


NEIGHBOURS = [
    ("StructureNode", _ident(id="sn2"), "e2"),
    ("Document", _ident(tenant_id="acme", path="a.pdf", version=1), "e1"),
    ("InfoUnit", _ident(uid="iu1"), "e3"),
]
EDGES = [
    {"eid": "e2", "type": "HAS_CHILD", "props": {}, "outgoing": True},
    {"eid": "e1", "type": "HAS_STRUCTURE", "props": {"k": 1}, "outgoing": False},
    {"eid": "e3", "type": "HAS_INFO_UNIT", "props": None, "outgoing": True},
]


async def test_expand_orders_by_label_and_key_pages_and_names_the_edge_ends() -> None:
    nav, drv = _mk(_expansion_responder(NEIGHBOURS, EDGES))
    page = await nav.expand_node("StructureNode", "sn1", limit=2, skip=1, tenant_id="acme")
    assert page.total == 3
    assert [(n.label, n.key) for n in page.nodes] == [("InfoUnit", "iu1"), ("StructureNode", "sn2")]
    assert [
        (e.type, e.source_label, e.source_key, e.target_label, e.target_key) for e in page.edges
    ] == [
        ("HAS_INFO_UNIT", "StructureNode", "sn1", "InfoUnit", "iu1"),
        ("HAS_CHILD", "StructureNode", "sn1", "StructureNode", "sn2"),
    ]
    first, *_ = drv.last_queries
    cypher, params = first
    assert "m <> n" in cypher and "n.tenant_id = $scope_tenant" in cypher
    assert "m.tenant_id = $scope_tenant" in cypher  # the neighbours are scoped too
    assert params["edge_types"] is None and params["neighbor_labels"] is None


async def test_expand_without_limit_returns_every_neighbour_and_edge_direction_follows_the_relationship() -> (
    None
):
    nav, _ = _mk(_expansion_responder(NEIGHBOURS, EDGES))
    out = await nav.expand_node("StructureNode", "sn1", limit=None)
    assert [(n.label, n.key) for n in out.nodes] == [
        ("Document", '["acme","a.pdf",1]'),
        ("InfoUnit", "iu1"),
        ("StructureNode", "sn2"),
    ]
    incoming = out.edges[0]
    assert (incoming.type, incoming.source_label, incoming.target_label, incoming.target_key) == (
        "HAS_STRUCTURE",
        "Document",
        "StructureNode",
        "sn1",
    )
    assert incoming.properties == {"k": 1}
    assert out.edges[2].properties == {}  # null properties


async def test_expand_sends_the_filters_and_checks_the_arguments() -> None:
    nav, drv = _mk(_expansion_responder(NEIGHBOURS, EDGES))
    await nav.expand_node(
        "StructureNode",
        "sn1",
        direction="out",
        edge_types=["HAS_CHILD"],
        neighbor_labels=["Section"],
    )
    cypher, params = drv.last_queries[0]
    assert "(n)-[r]->(m)" in cypher
    assert params["edge_types"] == ["HAS_CHILD"] and params["neighbor_labels"] == ["Section"]
    empty, _ = _mk()
    out = await empty.expand_node("StructureNode", "sn1")
    assert (out.nodes, out.edges, out.total) == ([], [], 0)
    assert (await empty.expand_node("Document", "not-json")).total == 0
    for bad in ({"skip": -1}, {"limit": 0}, {"direction": "up"}, {"edge_types": ["bad type"]}):
        with pytest.raises(NavigationError):
            await empty.expand_node("StructureNode", "sn1", **bad)


async def test_search_nodes_merges_labels_key_matches_first() -> None:
    def respond(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if "MATCH (n:`StructureNode`)" in cypher:
            return [
                {"eid": "s2", "field": "title", "ident": _ident(id="sn2")},
                {"eid": "s1", "field": "key", "ident": _ident(id="sn-1")},
            ]
        if "MATCH (n:`Document`)" in cypher:
            return [
                {
                    "eid": "d1",
                    "field": "name",
                    "ident": _ident(tenant_id="acme", path="a.pdf", version=1),
                }
            ]
        if "elementId(m) IN $eids" in cypher:
            return [
                {"eid": eid, "m": {"id": ident, "_labels": ["StructureNode"]}}
                for eid, ident in (("s1", "sn-1"), ("s2", "sn2"))
                if eid in params["eids"]
            ] + [
                {
                    "eid": "d1",
                    "m": {
                        "tenant_id": "acme",
                        "path": "a.pdf",
                        "version": 1,
                        "_labels": ["Document"],
                    },
                }
            ]
        return []

    nav, drv = _mk(respond)
    hits = await nav.search_nodes("Sn", labels=["StructureNode", "Document"], tenant_id="acme")
    assert [(h.node.label, h.node.key, h.field) for h in hits] == [
        ("StructureNode", "sn-1", "key"),
        ("Document", DOC_KEY, "name"),
        ("StructureNode", "sn2", "title"),
    ]
    page = await nav.search_nodes("Sn", labels=["StructureNode", "Document"], limit=1, skip=2)
    assert [(h.node.key, h.field) for h in page] == [("sn2", "title")]
    cypher, params = next(q for q in drv.last_queries if "MATCH (n:`StructureNode`)" in q[0])
    assert params["needle"] == "sn"  # the text, lower case
    assert (
        "toLower(n.`title`) CONTAINS $needle" in cypher
        and "toLower(n.`id`) CONTAINS $needle" in cypher
    )
    assert "n.`id` = $key_0" in cypher and "n.tenant_id = $scope_tenant" in cypher


async def test_search_nodes_validates_before_querying() -> None:
    nav, drv = _mk()
    assert await nav.search_nodes("a") == []
    assert await nav.search_nodes("") == []
    assert drv.last_queries == []
    for bad in ({"limit": 0}, {"skip": -1}, {"labels": ["Nope"]}, {"labels": "Document"}):
        with pytest.raises(NavigationError):
            await nav.search_nodes("ab", **bad)
