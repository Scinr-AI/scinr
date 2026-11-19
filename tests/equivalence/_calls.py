"""
tests/equivalence/_calls.py -- The navigation calls of the read-equivalence suite.

:func:`build_calls` receives the neutral dump of a scenario (``tests/golden/_dump.py``) and
returns, for **every** method of ``GraphNavigator``, the calls to make: with real anchors
taken from the dump (the first document, a node with children, an instance with
relationships, an entity...) or, when the scenario has none, with an anchor that does not
exist (the answer is then "nothing", which both engines must agree on). Every anchor comes
with its variants of filters: ``tenant_id``, ``include_public``, ``created_by_user_id``,
``job_id``, ``where=``, ``limit`` / ``skip`` and ``depth``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from scinr.newton.navigation.filters import (
    Contains,
    EndsWith,
    Eq,
    Gte,
    In,
    IsNotNull,
    IsNull,
    Lte,
    Ne,
    NotIn,
    Regex,
    StartsWith,
)
from scinr.newton.navigation.models import NodeSelector

#: Methods of ``GraphNavigator`` that are not navigation calls (lifecycle).
LIFECYCLE = frozenset({"connect", "close", "ping"})

#: Methods that take a free-form query: not compared across engines (each has its own tests).
NOT_COMPARED = frozenset({"execute_raw", "execute_raw_one"})

MISSING = "does-not-exist"


@dataclass(frozen=True)
class Call:
    """One call: the method, its keyword arguments and a short label for the failure report."""

    method: str
    kwargs: dict[str, Any]
    label: str = ""


@dataclass
class _Graph:
    """The dump of a scenario, indexed for the anchors."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    out: dict[str, list[tuple[str, str, dict[str, Any]]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    inc: dict[str, list[tuple[str, str, dict[str, Any]]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @classmethod
    def of(cls, dump: dict[str, Any]) -> _Graph:
        graph = cls()
        for node in dump["nodes"]:
            graph.nodes[node["key"]] = node
        for rel in dump["relationships"]:
            graph.out[rel["from"]].append((rel["type"], rel["to"], rel["props"]))
            graph.inc[rel["to"]].append((rel["type"], rel["from"], rel["props"]))
        return graph

    def of_label(self, label: str) -> list[dict[str, Any]]:
        return sorted(
            (n for n in self.nodes.values() if n["key"].startswith(f"{label}|")),
            key=lambda n: n["key"],
        )

    def targets(self, key: str, rel: str) -> list[str]:
        return [to for t, to, _ in self.out.get(key, ()) if t == rel]


def _first(items: Iterable[Any], default: Any = None) -> Any:
    for item in items:
        return item
    return default


def _scopes(tenants: list[str], users: list[str], jobs: list[str]) -> list[dict[str, Any]]:
    """The scope variants every call is repeated with."""
    scopes: list[dict[str, Any]] = [{}]
    if tenants:
        scopes.append({"tenant_id": tenants[0]})
        scopes.append({"tenant_id": tenants[0], "include_public": True})
    scopes.append({"tenant_id": "__public__"})
    if users:
        scopes.append({"created_by_user_id": users[0]})
    if jobs:
        scopes.append({"job_id": jobs[0]})
    if tenants and users:
        scopes.append({"tenant_id": tenants[0], "created_by_user_id": users})
    return scopes


def _one_tenant(scope: dict[str, Any]) -> bool:
    """Scopes that name one tenant: a ``limit`` over rows that tie across tenants is only
    comparable inside one tenant (Neo4j breaks such ties in an arbitrary way)."""
    return "tenant_id" in scope and not scope.get("include_public")


def _scoped(scope: dict[str, Any]) -> bool:
    return bool(scope)


def build_calls(dump: dict[str, Any]) -> list[Call]:
    """Every call of the suite for a scenario (``dump`` is its neutral graph dump)."""
    graph = _Graph.of(dump)
    calls: list[Call] = []

    def add(
        method: str,
        kwargs: dict[str, Any],
        *,
        scoped: bool = True,
        label: str = "",
        only: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        variants = scopes if scoped else [{}]
        if only is not None:
            variants = [v for v in variants if only(v)]
        for scope in variants:
            calls.append(Call(method, {**kwargs, **scope}, label))

    # ── anchors ──────────────────────────────────────────────────────────────
    docs = [d["props"] for d in graph.of_label("Document")]
    tenants = sorted(
        {d["tenant_id"] for d in docs if d.get("tenant_id") not in (None, "__public__")}
    )
    users = sorted(
        {
            n["props"]["created_by_user_id"]
            for n in graph.nodes.values()
            if n["props"].get("created_by_user_id")
        }
    )
    jobs = sorted({n["props"]["job_id"] for n in graph.nodes.values() if n["props"].get("job_id")})
    scopes = _scopes(tenants, users, jobs)

    def doc_key(d: dict[str, Any]) -> str:
        return next(
            k for k, n in graph.nodes.items() if k.startswith("Document|") and n["props"] is d
        )

    leaf_docs = [d for d in docs if not d.get("is_folder")]
    folder_docs = [d for d in docs if d.get("is_folder")]
    structured = [d for d in docs if graph.targets(doc_key(d), "HAS_STRUCTURE")]
    doc = _first(structured, _first(leaf_docs, _first(docs, {"path": MISSING, "version": 1})))
    folder = _first(folder_docs, doc)
    old = _first((d for d in docs if not d.get("latest")), doc)
    paths = [doc["path"], MISSING]

    structure_nodes = [n["props"] for n in graph.of_label("StructureNode")]
    node_keys = {
        n["props"]["id"]: k for k, n in graph.nodes.items() if k.startswith("StructureNode|")
    }
    parents = [n for n in structure_nodes if graph.targets(node_keys[n["id"]], "HAS_CHILD")]
    with_parent = [
        n
        for n in structure_nodes
        if any(t == "HAS_CHILD" for t, _, _ in graph.inc.get(node_keys[n["id"]], ()))
    ]
    top = _first(
        (
            n
            for n in structure_nodes
            if any(t == "HAS_STRUCTURE" for t, _, _ in graph.inc.get(node_keys[n["id"]], ()))
        ),
        None,
    )
    parent = _first(parents, top or {"id": MISSING})
    child = _first(with_parent, parent)
    deep = _first(
        (
            n
            for n in with_parent
            if any(
                p in {node_keys[x["id"]] for x in with_parent}
                for _, p, _ in graph.inc.get(node_keys[n["id"]], ())
            )
        ),
        child,
    )
    node_ids = [parent["id"], child["id"], deep["id"], MISSING]
    with_units = _first(
        (n for n in structure_nodes if graph.targets(node_keys[n["id"]], "HAS_INFO_UNIT")), parent
    )
    with_decision = _first(
        (n for n in structure_nodes if graph.targets(node_keys[n["id"]], "HAS_MODEL_DECISION")),
        parent,
    )
    with_extraction = _first(
        (n for n in structure_nodes if graph.targets(node_keys[n["id"]], "HAS_EXTRACTION")), parent
    )
    with_triples = _first(
        (
            n
            for n in structure_nodes
            if any(
                graph.nodes[er]["props"].get("model_class") == "Triple"
                for er in graph.targets(node_keys[n["id"]], "HAS_EXTRACTION")
            )
        ),
        with_extraction,
    )
    roles = sorted({n["role"] for n in structure_nodes if n.get("role")})
    themes = sorted({n["theme"] for n in structure_nodes if n.get("theme")})
    titles = [n["title"] for n in structure_nodes if n.get("title")]

    units = [n["props"] for n in graph.of_label("InfoUnit")]
    unit = _first(units, {"uid": MISSING, "title": "x"})

    instances = [n["props"] for n in graph.of_label("ModelInstance")]
    instance_keys = {
        n["props"]["uid"]: k for k, n in graph.nodes.items() if k.startswith("ModelInstance|")
    }
    classes = sorted({i["model_class"] for i in instances if i.get("model_class")})
    cls = _first(classes, "GoldenProduct")
    linked = _first(
        (
            i
            for i in instances
            if any(
                t[:4] != "HAS_" and to.startswith("ModelInstance|")
                for t, to, _ in graph.out.get(instance_keys[i["uid"]], ())
            )
        ),
        None,
    )
    with_entities = _first(
        (i for i in instances if graph.targets(instance_keys[i["uid"]], "REFERENCES")),
        _first(instances, {"uid": MISSING, "model_class": cls}),
    )
    instance = linked or with_entities
    rel_types = sorted(
        {
            t
            for i in instances
            for t, to, _ in graph.out.get(instance_keys[i["uid"]], ())
            if to.startswith("ModelInstance|")
        }
    )
    rel_type = _first(rel_types, "HAS_PART")
    of_class = [i for i in instances if i.get("model_class") == cls]
    inst_prop = _first(
        (
            k
            for k, v in (of_class[0] if of_class else {}).items()
            if isinstance(v, str)
            and k not in ("uid", "model_class", "tenant_id")
            and len({x.get(k) for x in of_class}) == len(of_class)
        ),
        "uid",
    )

    entities = [n["props"] for n in graph.of_label("LabeledEntity")]
    entity = _first(entities, {"uid": MISSING, "label": "x", "value": "x", "normalized_value": "x"})
    entity_keys = {
        n["props"]["uid"]: k for k, n in graph.nodes.items() if k.startswith("LabeledEntity|")
    }
    linked_entity = _first((e for e in entities if graph.out.get(entity_keys[e["uid"]])), entity)
    labels = sorted({e["label"] for e in entities if e.get("label")})
    plain = [n["props"] for n in graph.of_label("Entity")]
    triple_entity = _first(plain, {"uid": MISSING, "value": "x"})

    catalog = [n["props"]["name"] for n in graph.of_label("CatalogModel")]
    decision_classes = sorted(
        {
            n["props"]["matched_model_class"]
            for n in graph.of_label("ModelDecision")
            if n["props"].get("matched_model_class")
        }
    )

    # ── documents ────────────────────────────────────────────────────────────
    add("list_root_documents", {})
    add("list_root_documents", {"latest_only": False})
    add("list_root_documents", {"only_folders": True})
    add("list_root_documents", {"only_leaves": True, "limit": 1, "skip": 1}, only=_one_tenant)
    add("list_root_documents", {"only_folders": True, "only_leaves": True}, scoped=False)
    add("count_root_documents", {})
    add("count_root_documents", {"latest_only": False, "only_leaves": True})
    add("count_root_documents", {"only_folders": True, "only_leaves": True}, scoped=False)
    for d in (doc, old):
        add("get_one_document", {"path": d["path"], "version": d["version"]})
    add("get_one_document", {"path": MISSING, "version": 1}, scoped=False)
    add("get_documents", {})
    add("get_documents", {"path": doc["path"]})
    add(
        "get_documents",
        {"name_contains": str(doc.get("name", "a"))[:3].upper(), "latest_only": False},
    )
    add("get_documents", {"version": 1})
    add("get_documents", {"is_folder": True, "latest_only": False, "limit": 2}, only=_one_tenant)
    add("get_documents", {"path_prefix": doc["path"][:3], "limit": 2, "skip": 1}, only=_one_tenant)
    add(
        "get_documents",
        {
            "where": {
                "version": Gte(value=1),
                "name": StartsWith(value=str(doc.get("name", "a"))[:2]),
            }
        },
    )
    add("get_documents", {"where": {"latest": True}})
    add("get_documents", {"where": {"version": Eq(value="1")}}, scoped=False)  # type clash
    add(
        "get_documents",
        {"where": {"name": Regex(pattern=".*" + str(doc.get("name", "a"))[:2] + ".*")}},
    )
    add(
        "get_documents",
        {"where": {"name": In(values=[doc.get("name"), "zz"]), "path": NotIn(values=["zz"])}},
    )
    add("get_documents", {"where": {"load_date": IsNotNull(), "context_instructions": IsNull()}})
    add("get_documents", {"where": {"name": Contains(value=str(doc.get("name", "a"))[1:3])}})
    add(
        "get_documents",
        {
            "where": {
                "name": EndsWith(value=str(doc.get("name", "a"))[-3:]),
                "version": Lte(value=9),
            }
        },
    )
    add("get_documents", {"where": {"version": Ne(value=99)}})
    add("get_documents", {"where": {"bad key": 1}}, scoped=False)
    for path in paths:
        add("document_exists", {"path": path})
    add("document_exists", {"path": doc["path"], "version": doc["version"]})
    add("document_exists", {"path": doc["path"], "version": 99})
    for d in (folder, doc):
        add("get_child_documents", {"path": d["path"]})
        add("get_child_documents", {"path": d["path"], "depth": None, "is_folder": False})
        add(
            "get_child_documents",
            {"path": d["path"], "depth": 2, "version": d["version"], "limit": 2},
            only=_one_tenant,
        )
        add("get_document_tree", {"path": d["path"]})
        add("get_document_tree", {"path": d["path"], "depth": 1, "version": d["version"]})
        add("get_document_parent", {"path": d["path"]})
        add("get_document_ancestors", {"path": d["path"]})
        add("get_document_ancestors", {"path": d["path"], "depth": 1, "version": d["version"]})
        add("get_document_leaves", {"path": d["path"]})
        add("get_document_leaves", {"path": d["path"], "depth": 1})
    for path in paths:
        add("get_document_tree", {"path": path})
        add("get_document_parent", {"path": path, "version": 1})
        add("get_document_ancestors", {"path": path})
        add("get_document_leaves", {"path": path})
        add("get_child_documents", {"path": path})
        add("list_document_versions", {"path": path})
        add("get_version_chain", {"path": path})
        add("get_latest_version", {"path": path})
        add("get_document_stats", {"path": path})
    add("get_document_stats", {"path": doc["path"], "version": doc["version"]})
    add("get_document_stats", {"path": folder["path"]})
    add("get_document_tree", {"path": doc["path"], "depth": 0}, scoped=False)

    # ── structure ────────────────────────────────────────────────────────────
    for d in (doc, folder):
        selector = d["path"]
        add("get_structure_nodes", {"document": selector})
        add("get_structure_nodes", {"document": selector, "version": d["version"], "depth": 1})
        add("get_root_structure_nodes", {"document": selector})
        add("count_structure_nodes", {"document": selector})
        add("count_structure_nodes", {"document": selector, "roles": roles[:1], "depth": 2})
    add("get_structure_nodes", {"document": doc["path"], "roles": roles[:2], "limit": 3, "skip": 1})
    add(
        "get_structure_nodes",
        {"document": doc["path"], "title_contains": (titles or ["a"])[0][:3].upper()},
    )
    add("get_structure_nodes", {"document": doc["path"], "theme": (themes or ["x"])[0]})
    add(
        "get_structure_nodes",
        {
            "document": doc["path"],
            "where": {"appearance_order": Gte(value=1), "role": Ne(value="zzz")},
        },
    )
    add("get_structure_nodes", {"document": doc["path"], "where": {"row_index": IsNotNull()}})
    add("get_structure_nodes", {"document": MISSING})
    for node_id in node_ids:
        add("get_structure_node", {"node_id": node_id})
        add("get_child_nodes", {"node_id": node_id})
        add("get_child_nodes", {"node_id": node_id, "depth": None, "roles": roles[:1], "limit": 2})
        add("get_structure_subtree", {"node_id": node_id})
        add("get_structure_subtree", {"node_id": node_id, "depth": 1, "include_info_units": True})
        add("get_parent_node", {"node_id": node_id})
        add("get_node_ancestors", {"node_id": node_id})
        add("get_node_ancestors", {"node_id": node_id, "depth": 1})
        add("get_node_path", {"node_id": node_id})
        add("get_document_of_node", {"node_id": node_id})
        add("get_sibling_nodes", {"node_id": node_id})
        add("get_sibling_nodes", {"node_id": node_id, "include_self": True})
        add("describe_node", {"node_id": node_id})
    add("describe_node", {"node_id": with_extraction["id"], "include_source_text": True})
    add("get_structure_nodes_by_ids", {"node_ids": [deep["id"], MISSING, parent["id"], deep["id"]]})
    add("get_structure_nodes_by_ids", {"node_ids": []}, scoped=False)
    add("find_structure_nodes", {})
    add("find_structure_nodes", {"title_contains": (titles or ["a"])[0][:3].upper(), "limit": 2})
    add("find_structure_nodes", {"role": (roles or ["x"])[0], "skip": 1})
    add("find_structure_nodes", {"theme": (themes or ["x"])[0]})
    add("find_structure_nodes", {"node_id": parent.get("node_id", "x"), "document": doc["path"]})
    add("find_structure_nodes", {"document": doc["path"], "where": {"title": Contains(value="e")}})
    add("get_nodes_by_theme", {"theme": (themes or ["x"])[0]})
    add("get_nodes_by_theme", {"theme": (themes or ["x"])[0], "document": doc["path"], "limit": 1})

    # ── info units ───────────────────────────────────────────────────────────
    for node_id in (with_units["id"], MISSING):
        add("get_info_units", {"node_id": node_id})
        add("get_info_units", {"node_id": node_id, "order_by": "title"})
    add("count_info_units", {"document": doc["path"]})
    add("count_info_units", {"document": doc["path"], "version": doc["version"], "depth": 1})
    word = (str(unit.get("title", "x")).split() or ["x"])[0]
    add("search_info_units", {"text": word})
    add("search_info_units", {"text": word, "field": "title", "limit": 50})
    add("search_info_units", {"text": word, "field": "description", "document": doc["path"]})
    add("search_info_units", {"text": "zzzz-nothing"})
    add("get_info_unit", {"uid": unit["uid"]})
    add("get_info_unit", {"uid": MISSING})
    add("get_node_for_info_unit", {"uid": unit["uid"]})
    add("get_node_for_info_unit", {"uid": MISSING})

    # ── annotation ───────────────────────────────────────────────────────────
    for node_id in (with_decision["id"], MISSING):
        add("get_model_decision", {"node_id": node_id})
    add("get_document_model_decisions", {"document": doc["path"]})
    add("get_document_model_decisions", {"document": doc["path"], "matched_only": True, "depth": 2})
    add("get_document_model_decisions", {"document": doc["path"], "matched_only": False})
    add("get_document_model_profile", {"document": doc["path"]})
    add("get_document_model_profile", {"document": MISSING})
    add("get_nodes_by_annotated_model", {"model_class": _first(decision_classes, cls)})
    add(
        "get_nodes_by_annotated_model",
        {"model_class": _first(decision_classes, cls), "document": doc["path"]},
    )
    add("get_unannotated_nodes", {"document": doc["path"]})
    add("get_unannotated_nodes", {"document": doc["path"], "depth": 1})
    add("get_proposed_models", {})
    add("get_proposed_models", {"document": doc["path"]})
    add("get_annotation_coverage", {"document": doc["path"]})
    add("get_annotation_coverage", {"document": doc["path"], "version": doc["version"], "depth": 2})
    add("get_annotation_coverage", {"document": MISSING})

    # ── extraction results and model instances ───────────────────────────────
    for node_id in (with_extraction["id"], MISSING):
        add("get_extraction_result", {"node_id": node_id})
    add("get_document_extraction_results", {"document": doc["path"]})
    add(
        "get_document_extraction_results",
        {"document": doc["path"], "model_class": cls, "limit": 1000},
    )
    add("get_node_model_instances", {"node_id": with_extraction["id"]})
    add(
        "get_node_model_instances",
        {"node_id": with_extraction["id"], "model_class": cls, "depth": 2},
    )
    add("get_node_model_instances", {"node_id": with_extraction["id"], "direct_only": True})
    add(
        "get_node_model_instances",
        {"node_id": with_extraction["id"], "where": {inst_prop: IsNotNull()}},
    )
    add("get_document_model_instances", {"document": doc["path"]})
    add(
        "get_document_model_instances",
        {"document": doc["path"], "model_class": cls, "limit": 2, "skip": 1},
    )
    add(
        "get_document_model_instances",
        {"document": doc["path"], "where": {inst_prop: Regex(pattern=".*")}, "depth": 3},
    )
    add("count_document_model_instances", {"document": doc["path"]})
    add("count_document_model_instances", {"document": doc["path"], "model_class": cls, "depth": 2})
    add("get_model_instances_by_class", {"model_class": cls})
    add(
        "get_model_instances_by_class",
        {"model_class": cls, "order_by": inst_prop, "limit": 2, "skip": 1},
    )
    add("get_model_instances_by_class", {"model_class": cls, "document": doc["path"]})
    add("get_model_instances_by_class", {"model_class": cls, "where": {inst_prop: Ne(value="zz")}})
    add("get_model_instances_by_class", {"model_class": cls, "order_by": "bad key"}, scoped=False)
    add("count_model_instances_by_class", {"model_class": cls})
    add("count_model_instances_by_class", {"model_class": cls, "document": doc["path"]})
    add("count_model_instances_by_class", {"model_class": MISSING})
    for uid in (instance["uid"], MISSING):
        add("get_model_instance", {"uid": uid})
        add("get_structure_nodes_for_model_instance", {"uid": uid})
        add("get_documents_for_model_instance", {"uid": uid})
        add("get_extraction_results_for_model_instance", {"uid": uid})
        add("get_incoming_model_instances", {"uid": uid})
        add("get_incoming_model_instances", {"uid": uid, "depth": 3, "limit": 2})
        add("get_outgoing_model_instances", {"uid": uid})
        add("get_outgoing_model_instances", {"uid": uid, "rel_type": rel_type, "depth": 2})
        add("get_model_instance_subtree", {"uid": uid})
        add("get_model_instance_subtree", {"uid": uid, "depth": 1})
        add("get_model_instance_relationships", {"uid": uid})
        add(
            "get_model_instance_relationships",
            {"uid": uid, "direction": "in", "rel_type": rel_type},
        )
        add("get_model_instance_relationships", {"uid": uid, "direction": "out"})
        add("get_related_model_instances", {"uid": uid, "rel_type": rel_type})
        add("get_related_model_instances", {"uid": uid, "rel_type": rel_type, "direction": "in"})
        add("get_model_instance_entities", {"uid": uid})
        add("get_model_instance_entities", {"uid": uid, "label": _first(labels, "x")})
    add(
        "get_model_instance_by_key",
        {
            "model_class": cls,
            "key_fields": {"product_name": "Widget A"},
            "tenant_id": _first(tenants, "__public__"),
        },
    )
    add(
        "get_model_instance_by_key",
        {
            "model_class": cls,
            "key_fields": {"product_name": "Widget A"},
            "tenant_id": _first(tenants, "__public__"),
            "include_public": True,
        },
    )
    add(
        "get_model_instance_by_key",
        {"model_class": cls, "key_fields": {"product_name": "nope"}},
        scoped=False,
    )
    add("find_shell_model_instances", {})
    add("find_shell_model_instances", {"model_class": cls, "limit": 1})
    add("list_model_instance_relationship_types", {})
    add("list_model_instance_relationship_types", {"document": doc["path"]})

    # ── entities ─────────────────────────────────────────────────────────────
    add("get_node_entities", {"node_id": with_extraction["id"]})
    add(
        "get_node_entities",
        {"node_id": with_extraction["id"], "label": _first(labels, "x"), "depth": 3},
    )
    add("get_document_entities", {"document": doc["path"]})
    add(
        "get_document_entities",
        {
            "document": doc["path"],
            "label": _first(labels, "x"),
            "limit": 2,
            "version": doc["version"],
        },
    )
    add("list_entity_labels", {})
    add("get_labeled_entities", {})
    add(
        "get_labeled_entities",
        {"label": entity["label"], "limit": 2, "skip": 1},
        only=_one_tenant,
    )
    add("get_labeled_entities", {"value": entity["value"]})
    add("get_labeled_entities", {"normalized_value": entity["normalized_value"]})
    add("get_labeled_entities", {"where": {"value": Contains(value=str(entity["value"])[:2])}})
    for uid in (entity["uid"], linked_entity["uid"], MISSING):
        add("get_labeled_entity", {"uid": uid})
        add("get_model_instances_referencing_entity", {"uid": uid})
        add("get_model_instances_referencing_entity", {"uid": uid, "model_class": cls, "limit": 1})
        add("get_nodes_referencing_entity", {"uid": uid})
        add("get_nodes_referencing_entity", {"uid": uid, "depth": 2, "limit": 1})
        add("get_entity_relationships", {"uid": uid})
        add("get_entity_relationships", {"uid": uid, "direction": "out", "rel_type": "R"})
        add("get_related_entities", {"uid": uid, "rel_type": "R"})
        add("get_related_entities", {"uid": uid, "rel_type": "R", "direction": "in"})
    for node_id in (with_triples["id"], MISSING):
        add("get_triples", {"node_id": node_id})
    for key in (triple_entity.get("value", "x"), triple_entity["uid"], "nope"):
        add("get_entity_triples", {"value_or_uid": key})
        add("get_entity_triples", {"value_or_uid": key, "direction": "out"})
        add("get_entity_triples", {"value_or_uid": key, "direction": "in"})

    # ── introspection ────────────────────────────────────────────────────────
    add("list_catalog_models", {}, scoped=False)
    add("list_catalog_models", {"include_fields": True}, scoped=False)
    add("get_catalog_graph", {}, scoped=False)
    add(
        "get_catalog_graph", {"include_fields": False, "include_relationships": False}, scoped=False
    )
    for presence in ("both", "model_instances", "annotated"):
        add("list_model_classes_present", {"presence": presence})
        add("get_catalog_graph_in_use", {"presence": presence})
    add("list_model_classes_present", {"model_classes": catalog[:2] + ["zz"]})
    add("list_model_classes_present", {"model_classes": []}, scoped=False)
    add("list_model_classes_present", {"presence": "bogus"}, scoped=False)
    add("get_catalog_graph_in_use", {"include_fields": False, "include_relationships": False})
    add("list_model_classes_in_use", {})
    add("list_model_classes_in_use", {"document": doc["path"]})
    add("get_model_properties", {"model_class": cls})
    add("get_model_properties", {"model_class": cls, "document": doc["path"]})
    add("get_model_properties", {"model_class": MISSING})
    add("list_node_roles", {})
    add("list_node_roles", {"document": doc["path"]})
    add("list_themes", {}, scoped=False)
    add("list_relationship_types", {}, scoped=False)
    add("list_relationship_types", {"structural_only": False}, scoped=False)
    add("list_node_labels", {}, scoped=False)
    add("get_graph_summary", {})

    # ── power tools ──────────────────────────────────────────────────────────
    node_selector = NodeSelector(type="StructureNode", key="id", value=parent["id"])
    child_selector = NodeSelector(type="StructureNode", key="id", value=deep["id"])
    mi_selector = NodeSelector(type="ModelInstance", key="uid", value=instance["uid"])
    le_selector = NodeSelector(type="LabeledEntity", key="uid", value=linked_entity["uid"])
    cm_selector = NodeSelector(type="CatalogModel", key="name", value=_first(catalog, cls))
    ghost = NodeSelector(type="StructureNode", key="id", value=MISSING)
    # Neo4j picks an arbitrary pair when a selector names several nodes (the versions of a
    # document share their path): the shortest paths start from a path held by one document.
    single = [d for d in docs if sum(1 for x in docs if x["path"] == d["path"]) == 1]
    path_selector = (
        NodeSelector(type="Document", key="path", value=single[0]["path"])
        if single
        else NodeSelector(type="StructureNode", key="id", value=parent["id"])
    )
    doc_selector = path_selector
    for selector in (doc_selector, node_selector, mi_selector, le_selector, cm_selector, ghost):
        add("neighbors", {"selector": selector})
        add("neighbors", {"selector": selector, "direction": "out", "depth": 2, "limit": 500})
        add(
            "neighbors",
            {
                "selector": selector,
                "direction": "in",
                "edge_types": ["HAS_CHILD", "HAS_STRUCTURE", "HAS_EXTRACTION", "HAS_PART"],
            },
        )
        add(
            "neighbors",
            {"selector": selector, "depth": 2, "target_types": ["StructureNode", "ModelInstance"]},
        )
        add("subgraph", {"selector": selector})
        add("subgraph", {"selector": selector, "max_nodes": 1})
        add(
            "subgraph",
            {
                "selector": selector,
                "depth": 1,
                "edge_types": ["HAS_CHILD", "HAS_STRUCTURE"],
                "max_nodes": 500,
            },
            only=_scoped,  # without a scope Neo4j (no APOC) ignores edge_types
        )
    add("neighbors", {"selector": node_selector, "target_types": ["Section"], "depth": 3})
    add("neighbors", {"selector": NodeSelector(type="Nope", key="x", value=1)}, scoped=False)
    add("neighbors", {"selector": NodeSelector(type="bad type", key="x", value=1)}, scoped=False)
    add("subgraph", {"selector": node_selector, "depth": 0}, scoped=False)
    add("shortest_path", {"from_selector": path_selector, "to_selector": child_selector})
    add(
        "shortest_path",
        {"from_selector": path_selector, "to_selector": child_selector, "max_hops": 1},
    )
    add(
        "shortest_path",
        {
            "from_selector": node_selector,
            "to_selector": mi_selector,
            "edge_types": ["HAS_CHILD", "HAS_EXTRACTION", "HAS_PART", "HAS_PRIMARY"],
        },
    )
    add("shortest_path", {"from_selector": mi_selector, "to_selector": le_selector})
    add("shortest_path", {"from_selector": ghost, "to_selector": path_selector})
    add(
        "shortest_path",
        {"from_selector": path_selector, "to_selector": child_selector, "max_hops": 0},
        scoped=False,
    )
    return calls


def covered_methods(calls: Iterable[Call]) -> set[str]:
    """The methods ``build_calls`` covers, to compare with the abstract ones."""
    return {c.method for c in calls}
