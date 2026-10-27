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


async def test_get_structure_nodes_by_ids_one_query_in_request_order() -> None:
    def node(i: str) -> dict:
        return {"id": i, "node_id": i, "role": "table", "source_page_ids": [f"p-{i}"]}

    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}], "n.id IN $ids": [
        {"n": node("b")}, {"n": node("a")}
    ]}))
    out = await nav.get_structure_nodes_by_ids(
        ["a", "missing", "b", "a"], tenant_id="acme", job_id="j1"
    )
    assert [(n.id, n.source_page_ids) for n in out] == [("a", ["p-a"]), ("b", ["p-b"])]
    q, params = _query_with(drv, "n.id IN $ids")
    assert q.startswith("MATCH (n:StructureNode) WHERE n.id IN $ids AND ")
    assert "$scope_tenant" in q and "$scope_job_ids" in q
    assert params["ids"] == ["a", "missing", "b"]
    assert params["scope_tenant"] == "acme"


async def test_get_structure_nodes_by_ids_empty_list_does_not_query() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    assert await nav.get_structure_nodes_by_ids([], tenant_id="acme") == []
    assert not _has_query(drv, "StructureNode")


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


# -- G. Catalogue in use by a scope -----------------------------------------

_MI_ACME = "EXISTS { MATCH (mi:ModelInstance {model_class: cm.name}) WHERE mi.tenant_id = $scope_tenant }"
_ER_ACME = "EXISTS { MATCH (er:ExtractionResult {model_class: cm.name}) WHERE er.tenant_id = $scope_tenant }"
_CATALOG_ALL_FIELDS = (
    "MATCH (cm:CatalogModel) "
    "OPTIONAL MATCH (cm)-[hf:HAS_FIELD]->(f:ModelField) "
    "OPTIONAL MATCH (cm)-[:BELONGS_TO_THEME]->(t:Theme) "
    "RETURN cm, "
    "collect(DISTINCT {name: f.name, type: f.type, entity_label: f.entity_label, "
    "is_instance_key: f.is_instance_key, required: hf.required, description: hf.description}) AS fields, "
    "collect(DISTINCT t.name) AS themes ORDER BY cm.name"
)
_CATALOG_RELATIONS = (
    "MATCH (a)-[r]->(b) "
    "WHERE (a:CatalogModel OR a:EntityLabel) AND (b:CatalogModel OR b:EntityLabel) "
    "AND NOT type(r) IN ['HAS_FIELD', 'BELONGS_TO_THEME'] "
    "RETURN labels(a)[0] AS source_kind, coalesce(a.name, a.label) AS source, "
    "type(r) AS rel_type, properties(r) AS props, "
    "labels(b)[0] AS target_kind, coalesce(b.name, b.label) AS target"
)


async def test_list_model_classes_present_tenant_query() -> None:
    nav, drv = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "RETURN cm.name AS name": [{"name": "A"}, {"name": "Triple"}],
    }))
    out = await nav.list_model_classes_present(tenant_id="acme")
    assert out == ["A", "Triple"]
    q, p = _query_with(drv, "RETURN cm.name AS name ORDER BY name")
    # The OR sits inside the WHERE: a model present only through the
    # ExtractionResult branch is not dropped.
    assert f"MATCH (cm:CatalogModel) WHERE ({_MI_ACME} OR {_ER_ACME}) RETURN" in q
    assert p["scope_tenant"] == "acme"
    assert "model_classes" not in p


async def test_list_model_classes_present_include_public_and_provenance() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    await nav.list_model_classes_present(
        tenant_id="acme", include_public=True, created_by_user_id="u1", job_id="j1"
    )
    q, p = _query_with(drv, "RETURN cm.name AS name")
    assert "mi.tenant_id IN $scope_tenants" in q and "er.tenant_id IN $scope_tenants" in q
    assert p["scope_tenants"] == ["acme", "__public__"]
    assert "coalesce(mi.created_by_user_ids, [])" in q
    assert "coalesce(mi.job_ids, [])" in q
    assert "er.created_by_user_id IN $scope_user_ids" in q
    assert "er.job_id IN $scope_job_ids" in q
    assert p["scope_user_ids"] == ["u1"] and p["scope_job_ids"] == ["j1"]


async def test_list_model_classes_present_model_classes_filter() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    await nav.list_model_classes_present(model_classes=["A"], tenant_id="acme")
    q, p = _query_with(drv, "RETURN cm.name AS name")
    assert "WHERE cm.name IN $model_classes AND (EXISTS" in q
    assert p["model_classes"] == ["A"]


async def test_list_model_classes_present_empty_list_runs_no_query() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    assert await nav.list_model_classes_present(model_classes=[], tenant_id="acme") == []
    assert not _has_query(drv, "CatalogModel")


def _in_use_responder(**overrides):
    mapping = {
        "RETURN 1 AS ok": [{"ok": 1}],
        "MATCH (el:EntityLabel)": [{"label": "Manufacturer"}, {"label": None}],
        "MATCH (a)-[r]->(b)": [
            {"source_kind": "CatalogModel", "source": "A", "rel_type": "AGGREGATES",
             "props": {}, "target_kind": "CatalogModel", "target": "B"},
            {"source_kind": "CatalogModel", "source": "A", "rel_type": "AGGREGATES",
             "props": {}, "target_kind": "CatalogModel", "target": "Absent"},
            {"source_kind": "CatalogModel", "source": "A", "rel_type": "SPECIFIED_IN",
             "props": {"via_field": "x"}, "target_kind": "EntityLabel", "target": "AbsentLabel"},
            {"source_kind": "EntityLabel", "source": "Manufacturer", "rel_type": "MAKES",
             "props": {}, "target_kind": "CatalogModel", "target": "B"},
        ],
        "MATCH (cm:CatalogModel)": [
            {"cm": {"name": "A"}, "fields": [], "themes": []},
            {"cm": {"name": "B"}, "fields": [], "themes": []},
        ],
    }
    mapping.update(overrides)
    return _resp(mapping)


async def test_get_catalog_graph_in_use_filters_relationships() -> None:
    nav, drv = _mk(_in_use_responder())
    g = await nav.get_catalog_graph_in_use(tenant_id="acme")
    assert [m.name for m in g.models] == ["A", "B"]
    assert g.entity_labels == ["Manufacturer"]
    assert [(r.source, r.target) for r in g.relationships] == [("A", "B"), ("Manufacturer", "B")]
    assert g.raw == {"models": 2, "relationships": 2}
    q, p = _query_with(drv, "MATCH (cm:CatalogModel)")
    assert f"MATCH (cm:CatalogModel) WHERE ({_MI_ACME} OR {_ER_ACME}) OPTIONAL MATCH" in q
    assert "HAS_FIELD" in q and p["scope_tenant"] == "acme"
    q, p = _query_with(drv, "MATCH (el:EntityLabel)")
    assert (
        "EXISTS { MATCH (le:LabeledEntity {label: el.label}) WHERE le.tenant_id = $scope_tenant }"
        in q
    )
    assert p["scope_tenant"] == "acme"


async def test_get_catalog_graph_in_use_without_fields_or_relationships() -> None:
    nav, drv = _mk(_in_use_responder())
    g = await nav.get_catalog_graph_in_use(
        tenant_id="acme", include_fields=False, include_relationships=False
    )
    assert g.relationships == []
    q, _ = _query_with(drv, "MATCH (cm:CatalogModel)")
    assert "HAS_FIELD" not in q
    assert not _has_query(drv, "MATCH (a)-[r]->(b)")


async def test_catalog_queries_unchanged() -> None:
    nav, drv = _mk(_in_use_responder())
    await nav.list_catalog_models()
    await nav.get_catalog_graph()
    queries = [q for q, _ in drv.last_queries]
    assert "MATCH (cm:CatalogModel) RETURN cm ORDER BY cm.name" in queries
    assert _CATALOG_ALL_FIELDS in queries
    assert "MATCH (el:EntityLabel) RETURN el.label AS label ORDER BY label" in queries
    assert _CATALOG_RELATIONS in queries


async def test_list_model_classes_in_use_counts_both_labels() -> None:
    nav, drv = _mk(_resp({
        "RETURN 1 AS ok": [{"ok": 1}],
        "UNION ALL": [
            {"model_class": "Triple", "model_instances": 0, "extraction_results": 7, "count": 7},
            {"model_class": "A", "model_instances": 3, "extraction_results": 2, "count": 5},
        ],
    }))
    out = await nav.list_model_classes_in_use(
        tenant_id="acme", created_by_user_id="u1", job_id="j1"
    )
    assert [(s.model_class, s.count, s.model_instances, s.extraction_results, s.kind) for s in out] == [
        ("Triple", 7, 0, 7, "in_use"),
        ("A", 5, 3, 2, "in_use"),
    ]
    q, p = _query_with(drv, "UNION ALL")
    mi_branch, er_branch = q.split("UNION ALL")
    assert "MATCH (mi:ModelInstance) WHERE mi.tenant_id = $scope_tenant" in mi_branch
    assert "coalesce(mi.created_by_user_ids, [])" in mi_branch
    assert "coalesce(mi.job_ids, [])" in mi_branch
    assert "MATCH (er:ExtractionResult) WHERE er.tenant_id = $scope_tenant" in er_branch
    assert "er.created_by_user_id IN $scope_user_ids" in er_branch
    assert "er.job_id IN $scope_job_ids" in er_branch
    assert "WHERE model_class IS NOT NULL" in q
    assert "ORDER BY count DESC, model_class" in q
    assert p["scope_tenant"] == "acme"


async def test_list_model_classes_in_use_by_document() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    await nav.list_model_classes_in_use(document="p", tenant_id="acme")
    q, p = _query_with(drv, "UNION ALL")
    mi_branch, er_branch = q.split("UNION ALL")
    assert "(:Document {path: $doc_path})" in mi_branch
    assert "EXISTS { MATCH (er)<-[:HAS_EXTRACTION]-(:StructureNode)" in er_branch
    assert "(:Document {path: $doc_path})" in er_branch
    assert p["doc_path"] == "p"


async def test_list_model_classes_in_use_unscoped_has_no_where() -> None:
    nav, drv = _mk(_resp({"RETURN 1 AS ok": [{"ok": 1}]}))
    await nav.list_model_classes_in_use()
    q, _ = _query_with(drv, "UNION ALL")
    assert "MATCH (mi:ModelInstance) RETURN" in q
    assert "MATCH (er:ExtractionResult) RETURN" in q


def test_model_class_stat_without_breakdown_leaves_none() -> None:
    from scinr.newton.navigation.neo4j import _map

    s = _map.model_class_stat({"model_class": "A", "count": 2}, kind="matched")
    assert (s.count, s.model_instances, s.extraction_results) == (2, None, None)
