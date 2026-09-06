"""
Unit tests for the Neo4j navigation backend — query construction & row mapping
against a fake async driver (no real Neo4j).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import FakeAsyncDriver, make_fake_llm  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.exceptions import NavigationError  # noqa: E402
from scinr.newton.navigation.filters import Gte, In  # noqa: E402
from scinr.newton.navigation.neo4j.navigator import Neo4jGraphNavigator  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}


def _mk(responder) -> tuple[Neo4jGraphNavigator, FakeAsyncDriver]:
    configure(**_NEO)
    drv = FakeAsyncDriver(responder)
    return Neo4jGraphNavigator(driver=drv), drv


def _resp(mapping):
    """Return rows for the first mapping key that is a substring of the query."""

    def responder(cypher: str, params: dict):
        for needle, rows in mapping.items():
            if needle in cypher:
                return rows(params) if callable(rows) else rows
        return []

    return responder


def _has_query(drv, needle: str) -> bool:
    return any(needle in q for q, _ in drv.last_queries)


def _query_with(drv, needle: str) -> tuple[str, dict]:
    for q, p in drv.last_queries:
        if needle in q:
            return q, p
    raise AssertionError(f"no query contained {needle!r}")


async def test_ping_ok() -> None:
    nav, _ = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    assert await nav.ping() is True


async def test_list_root_documents_query_and_mapping() -> None:
    doc = {"path": "p", "name": "n", "version": 2, "latest": True, "is_folder": False}
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "NOT ( ()-[:IS_COMPOSED_OF]->(d) )": [{"d": doc}]}))
    out = await nav.list_root_documents()
    assert [d.path for d in out] == ["p"]
    q = drv.last_queries[-1][0]
    assert "d.latest = true" in q  # latest_only default
    assert "is_folder" not in q  # no only_* filter


async def test_list_root_documents_only_folders_and_leaves_conflict() -> None:
    nav, _ = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    with pytest.raises(NavigationError):
        await nav.list_root_documents(only_folders=True, only_leaves=True)


async def test_get_one_document_uses_composite_key() -> None:
    doc = {"path": "p", "name": "n", "version": 5, "latest": False, "is_folder": False}
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "{path: $path, version: $version}": [{"d": doc}]}))
    out = await nav.get_one_document("p", 5)
    assert out is not None and out.version == 5
    _, params = drv.last_queries[-1]
    assert params == {"path": "p", "version": 5}


async def test_get_documents_dynamic_where_only_supplied_filters() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "MATCH (d:Document)": []}))
    await nav.get_documents(name_contains="foo", where={"version": Gte(value=3)})
    q, params = drv.last_queries[-1]
    assert "toLower(d.name) CONTAINS toLower($name_contains)" in q
    assert "d.`version` >= $w_version" in q
    assert "d.path = $path" not in q  # path not supplied → not in query
    assert params["name_contains"] == "foo"
    assert params["w_version"] == 3


async def test_get_child_documents_depth_interpolated() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "IS_COMPOSED_OF*1..": []}))
    await nav.get_child_documents("root", depth=3)
    q = drv.last_queries[-1][0]
    assert "IS_COMPOSED_OF*1..3]->(c:Document)" in q
    assert "{path: $path, latest: true}" in q  # version omitted -> latest


async def test_get_child_documents_version_pins_match() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "IS_COMPOSED_OF*1..": []}))
    await nav.get_child_documents("root", version=4)
    q, params = drv.last_queries[-1]
    assert "{path: $path, version: $version}" in q
    assert params["version"] == 4


async def test_get_structure_nodes_where_and_title_filter() -> None:
    node = {"id": "i", "node_id": "n", "role": "table", "_labels": ["StructureNode", "Table"]}
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "HAS_STRUCTURE|HAS_CHILD": [
        {"n": node, "dp": "d", "dv": 1}
    ]}))
    out = await nav.get_structure_nodes(
        "d", roles=["table"], title_contains="cap", where={"appearance_order": Gte(value=2)}
    )
    assert out[0].types == ["StructureNode", "Table"]
    assert out[0].document_path == "d"
    q = drv.last_queries[-1][0]
    assert "n.role IN $roles" in q
    assert "toLower(n.title) CONTAINS toLower($title_contains)" in q
    assert "n.`appearance_order` >= $w_appearance_order" in q


async def test_get_model_instances_by_class_where_verbatim_and_order() -> None:
    mi = {"uid": "u1", "model_class": "ProcedureTypeModel", "procedure_type": "ib"}
    nav, drv = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "MATCH (:CatalogModel {name: $mc})": [{"n": 1}],
        "MATCH (mi:ModelInstance {model_class: $model_class})": [{"mi": mi}],
    }))
    out = await nav.get_model_instances_by_class(
        "ProcedureTypeModel", where={"procedure_type": In(values=["ia", "ib"])}, order_by="procedure_type"
    )
    assert out[0].properties == {"procedure_type": "ib"}
    q, params = _query_with(drv, "MATCH (mi:ModelInstance {model_class: $model_class})")
    assert "mi.`procedure_type` IN $w_procedure_type" in q
    assert "ORDER BY mi.`procedure_type`" in q
    assert params["w_procedure_type"] == ["ia", "ib"]  # values passed through verbatim


async def test_is_shell_flag_computed() -> None:
    shell = {"uid": "u", "model_class": "M", "k1": "x"}  # 2 real props (+k1) -> <= 1 key + 2
    nav, _ = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "MATCH (cm:CatalogModel {name: $mc})": [{"n": 1}],
        "MATCH (mi:ModelInstance {uid: $uid}) RETURN mi": [{"mi": shell}],
    }))
    out = await nav.get_model_instance("u")
    assert out is not None and out.is_shell is True


async def test_incoming_outgoing_no_has_filter() -> None:
    other = {"uid": "o", "model_class": "M"}
    nav, drv = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "-[r*1..": [{"mi": other, "via_rel": "QA_MENTIONS_PROCEDURE_TYPE", "direction": "out"}],
    }))
    out = await nav.get_outgoing_model_instances("u", depth=2)
    assert out[0].via_rel == "QA_MENTIONS_PROCEDURE_TYPE"
    assert out[0].direction == "out"
    q = drv.last_queries[-1][0]
    assert "STARTS WITH 'HAS_'" not in q  # unfiltered


async def test_get_triples_optional_predicate() -> None:
    nav, drv = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "model_class: 'Triple'": [
            {"subject": "A", "predicate": "COVERS", "predicate_raw": "covers", "object": "B"},
            {"subject": "C", "predicate": None, "predicate_raw": None, "object": None},
        ],
    }))
    out = await nav.get_triples("nid")
    assert out[0].predicate == "COVERS"
    assert out[1].predicate is None and out[1].object is None
    assert "OPTIONAL MATCH (s)-[p]->(o:Entity)" in drv.last_queries[-1][0]


async def test_nodes_referencing_entity_navigates_via_model_instance() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "REFERENCES]-(mi:ModelInstance)": []}))
    await nav.get_nodes_referencing_entity("le-uid")
    q = drv.last_queries[-1][0]
    assert "<-[:REFERENCES]-(mi:ModelInstance)" in q
    assert "(sn:StructureNode)-[:HAS_EXTRACTION]->(er)" in q


async def test_execute_raw_rejects_write() -> None:
    nav, _ = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    with pytest.raises(NavigationError):
        await nav.execute_raw("MATCH (n) SET n.x = 1")


async def test_execute_raw_dialect_mismatch() -> None:
    nav, _ = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    with pytest.raises(NavigationError):
        await nav.execute_raw("RETURN 1", dialect="gremlin")


async def test_execute_raw_passes_through_read() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "RETURN 42 AS a": [{"a": 42}]}))
    rows = await nav.execute_raw("RETURN 42 AS a")
    assert rows == [{"a": 42}]


async def test_list_relationship_types_structural_only_query() -> None:
    nav, drv = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "collect(DISTINCT (labels(a)[0]": [{"types": ["HAS_CHILD", "QA_MENTIONS_PROCEDURE_TYPE"]}],
    }))
    out = await nav.list_relationship_types()
    assert out == ["HAS_CHILD", "QA_MENTIONS_PROCEDURE_TYPE"]
    assert "NOT all(p IN pairs WHERE p = 'Entity->Entity')" in drv.last_queries[-1][0]


async def test_close_never_closes_a_driver_it_did_not_create() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    await nav.close()
    assert drv.closed is False  # external driver — caller owns its lifecycle
