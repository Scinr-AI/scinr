"""
tests/sql/test_browse_sql.py -- ``get_node``, ``count_node_relationships``, ``expand_node`` and
``search_nodes`` of ``SqlGraphNavigator`` (SQLite and Postgres) over a small hand-built graph
with expected values written by hand.

Graph (tenant ``t``, a second tenant ``other`` and the global catalogue)::

    dir(folder) -IS_COMPOSED_OF-> a.pdf -HAS_STRUCTURE-> sn1 -HAS_CHILD,REFERS-> sn2
    sn2 -SELF-> sn2 ; sn2 -HAS_INFO_UNIT-> iu1 ; sn2 -HAS_EXTRACTION-> er1 -EXTRACTED-> d1
    er1 -USES_PRIMARY_MODEL-> Drug -HAS_FIELD-> ModelField(dose, Drug) ; Drug -BELONGS_TO_THEME-> Theme
    (other tenant) a.pdf, erx -USES_PRIMARY_MODEL-> Drug, x1 (an "Aspirin" record)
"""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import NavigationError
from scinr.newton.graph.identity import encode_key
from scinr.newton.navigation.sql.navigator import SqlGraphNavigator
from scinr.newton.sql import tables as T
from scinr.newton.sql.engine import get_sync_engine

TENANT = "t"
OTHER = "other"
KW = {"tenant_id": TENANT}
A_KEY = encode_key("Document", {"tenant_id": TENANT, "path": "a.pdf", "version": 1})
DIR_KEY = encode_key("Document", {"tenant_id": TENANT, "path": "dir", "version": 1})
OTHER_A_KEY = encode_key("Document", {"tenant_id": OTHER, "path": "a.pdf", "version": 1})
FIELD_KEY = encode_key("ModelField", {"name": "dose", "model": "Drug"})


def _edge(rel, src, dst, tenant=TENANT, props=None):
    return T.edges.insert().values(
        tenant_id=tenant,
        type=rel,
        src_label=src[0],
        src_key=src[1],
        dst_label=dst[0],
        dst_key=dst[1],
        _ident="" if props is None else str(sorted(props.items())),
        props=props or {},
    )


def _catalog_edge(rel, src, dst):
    return T.catalog_edges.insert().values(
        type=rel,
        src_label=src[0],
        src_key=src[1],
        dst_label=dst[0],
        dst_key=dst[1],
        _ident="",
        props={},
    )


def _doc(key, tenant, path, folder=False):
    return T.documents.insert().values(
        _key=key,
        tenant_id=tenant,
        path=path,
        version=1,
        name=path,
        latest=True,
        is_folder=folder,
        created_by_user_id="u1",
        job_id="j1",
        props={},
    )


def _sn(sn_id, title):
    return T.structure_nodes.insert().values(
        id=sn_id,
        tenant_id=TENANT,
        role="section",
        title=title,
        created_by_user_id="u1",
        job_id="j1",
        props={},
        _document_key=A_KEY,
    )


def _mi(uid, model_class, tenant, props):
    return T.model_instances.insert().values(
        uid=uid,
        tenant_id=tenant,
        model_class=model_class,
        created_by_user_ids=["u1"],
        job_ids=["j1"],
        props=props,
    )


def _er(uid, tenant):
    return T.extraction_results.insert().values(
        uid=uid,
        tenant_id=tenant,
        model_class="Drug",
        created_by_user_id="u1",
        job_id="j1",
        props={},
    )


@pytest.fixture
async def nav(sql_backend):
    navigator = SqlGraphNavigator()
    await navigator.connect()
    with get_sync_engine().begin() as c:
        for stmt in (
            T.catalog_models.insert().values(name="Drug", props={"description": "A medicine"}),
            T.model_fields.insert().values(_key=FIELD_KEY, name="dose", model="Drug", props={}),
            T.themes.insert().values(path="t/default", props={"name": "Default"}),
            _catalog_edge("HAS_FIELD", ("CatalogModel", "Drug"), ("ModelField", FIELD_KEY)),
            _catalog_edge("BELONGS_TO_THEME", ("CatalogModel", "Drug"), ("Theme", "t/default")),
            _doc(DIR_KEY, TENANT, "dir", folder=True),
            _doc(A_KEY, TENANT, "a.pdf"),
            _doc(OTHER_A_KEY, OTHER, "a.pdf"),
            _sn("sn1", "Intro"),
            _sn("sn2", "Results"),
            T.info_units.insert().values(
                uid="iu1",
                tenant_id=TENANT,
                title="Aspirin dosage",
                description="How much",
                created_by_user_id="u1",
                job_id="j1",
                props={},
                _document_key=A_KEY,
                _structure_node_id="sn2",
            ),
            _er("er1", TENANT),
            _er("erx", OTHER),
            _mi("d1", "Drug", TENANT, {"name": "Aspirin"}),
            _mi("x1", "Drug", OTHER, {"name": "Aspirin other"}),
            _edge("IS_COMPOSED_OF", ("Document", DIR_KEY), ("Document", A_KEY)),
            _edge("HAS_STRUCTURE", ("Document", A_KEY), ("StructureNode", "sn1")),
            _edge("HAS_CHILD", ("StructureNode", "sn1"), ("StructureNode", "sn2")),
            _edge("REFERS", ("StructureNode", "sn1"), ("StructureNode", "sn2"), props={"why": "x"}),
            _edge("SELF", ("StructureNode", "sn2"), ("StructureNode", "sn2")),
            _edge("HAS_INFO_UNIT", ("StructureNode", "sn2"), ("InfoUnit", "iu1")),
            _edge("HAS_EXTRACTION", ("StructureNode", "sn2"), ("ExtractionResult", "er1")),
            _edge("EXTRACTED", ("ExtractionResult", "er1"), ("ModelInstance", "d1")),
            _edge("USES_PRIMARY_MODEL", ("ExtractionResult", "er1"), ("CatalogModel", "Drug")),
            _edge(
                "USES_PRIMARY_MODEL",
                ("ExtractionResult", "erx"),
                ("CatalogModel", "Drug"),
                tenant=OTHER,
            ),
        ):
            c.execute(stmt)
    try:
        yield navigator
    finally:
        await navigator.close()


def _pairs(nodes):
    return [(n.label, n.key) for n in nodes]


# -- get_node ------------------------------------------------------------------


async def test_get_node_returns_the_identity_and_respects_the_scope(nav) -> None:
    node = await nav.get_node("Document", A_KEY, **KW)
    assert (node.label, node.key, node.types) == ("Document", A_KEY, ["Document"])
    assert node.properties["path"] == "a.pdf"
    assert (await nav.get_node("Document", A_KEY)).key == A_KEY  # unscoped
    assert await nav.get_node("Document", OTHER_A_KEY, **KW) is None  # another tenant
    assert (await nav.get_node("Document", OTHER_A_KEY, tenant_id=OTHER)).key == OTHER_A_KEY
    sn = await nav.get_node("StructureNode", "sn1", **KW)
    assert (sn.label, sn.key) == ("StructureNode", "sn1") and "section" not in sn.types
    assert await nav.get_node("StructureNode", "sn1", tenant_id=OTHER) is None
    assert await nav.get_node("StructureNode", "sn1", job_id="nope", **KW) is None


async def test_get_node_catalogue_is_visible_to_every_scope(nav) -> None:
    for scope in ({}, KW, {"tenant_id": "nobody"}):
        node = await nav.get_node("CatalogModel", "Drug", **scope)
        assert (node.label, node.key) == ("CatalogModel", "Drug")
    field = await nav.get_node("ModelField", FIELD_KEY, tenant_id="nobody")
    assert field.key == FIELD_KEY and field.properties["name"] == "dose"


async def test_get_node_malformed_or_missing_key_is_none_and_unknown_label_fails(nav) -> None:
    assert await nav.get_node("Document", "not-json") is None
    assert await nav.get_node("Document", '["t","a.pdf"]') is None
    assert await nav.get_node("Document", '["t","a.pdf","1"]') is None  # version as text
    assert await nav.get_node("StructureNode", "missing") is None
    assert await nav.get_node("StructureNode", "") is None
    for bad in ("Nope", "bad label", "Section"):
        with pytest.raises(NavigationError):
            await nav.get_node(bad, "x")


# -- count_node_relationships -------------------------------------------------


async def test_count_node_relationships(nav) -> None:
    rows = await nav.count_node_relationships("StructureNode", "sn1", **KW)
    assert [(r.type, r.direction, r.neighbor_label, r.count) for r in rows] == [
        ("HAS_CHILD", "out", "StructureNode", 1),
        ("HAS_STRUCTURE", "in", "Document", 1),
        ("REFERS", "out", "StructureNode", 1),
    ]
    # a relationship from the node to itself counts once out and once in
    rows = await nav.count_node_relationships("StructureNode", "sn2", **KW)
    assert [(r.type, r.direction, r.neighbor_label, r.count) for r in rows] == [
        ("HAS_CHILD", "in", "StructureNode", 1),
        ("HAS_EXTRACTION", "out", "ExtractionResult", 1),
        ("HAS_INFO_UNIT", "out", "InfoUnit", 1),
        ("REFERS", "in", "StructureNode", 1),
        ("SELF", "in", "StructureNode", 1),
        ("SELF", "out", "StructureNode", 1),
    ]


async def test_count_node_relationships_of_a_catalogue_node_only_sees_the_tenant(nav) -> None:
    def shape(rows):
        return [(r.type, r.direction, r.neighbor_label, r.count) for r in rows]

    assert shape(await nav.count_node_relationships("CatalogModel", "Drug", **KW)) == [
        ("BELONGS_TO_THEME", "out", "Theme", 1),
        ("HAS_FIELD", "out", "ModelField", 1),
        ("USES_PRIMARY_MODEL", "in", "ExtractionResult", 1),
    ]
    assert shape(await nav.count_node_relationships("CatalogModel", "Drug")) == [
        ("USES_PRIMARY_MODEL", "in", "ExtractionResult", 2),
        ("BELONGS_TO_THEME", "out", "Theme", 1),
        ("HAS_FIELD", "out", "ModelField", 1),
    ]
    assert shape(
        await nav.count_node_relationships("CatalogModel", "Drug", tenant_id="nobody")
    ) == [
        ("BELONGS_TO_THEME", "out", "Theme", 1),
        ("HAS_FIELD", "out", "ModelField", 1),
    ]


async def test_count_node_relationships_missing_or_outside_the_scope_is_empty(nav) -> None:
    assert await nav.count_node_relationships("StructureNode", "missing", **KW) == []
    assert await nav.count_node_relationships("Document", OTHER_A_KEY, **KW) == []
    assert await nav.count_node_relationships("Document", "not-json", **KW) == []
    with pytest.raises(NavigationError):
        await nav.count_node_relationships("Nope", "x")


# -- expand_node -----------------------------------------------------------------


async def test_expand_node_neighbours_edges_and_total(nav) -> None:
    out = await nav.expand_node("StructureNode", "sn1", **KW)
    assert _pairs(out.nodes) == [("Document", A_KEY), ("StructureNode", "sn2")]
    assert out.total == 2
    assert [
        (e.type, e.source_label, e.source_key, e.target_label, e.target_key) for e in out.edges
    ] == [
        ("HAS_STRUCTURE", "Document", A_KEY, "StructureNode", "sn1"),
        ("HAS_CHILD", "StructureNode", "sn1", "StructureNode", "sn2"),
        ("REFERS", "StructureNode", "sn1", "StructureNode", "sn2"),
    ]
    assert out.edges[2].properties == {"why": "x"}


async def test_expand_node_pages_in_a_stable_order(nav) -> None:
    first = await nav.expand_node("StructureNode", "sn1", limit=1, **KW)
    second = await nav.expand_node("StructureNode", "sn1", limit=1, skip=1, **KW)
    assert _pairs(first.nodes) == [("Document", A_KEY)] and first.total == 2
    assert [e.type for e in first.edges] == ["HAS_STRUCTURE"]  # only the neighbours of the page
    assert _pairs(second.nodes) == [("StructureNode", "sn2")] and second.total == 2
    assert [e.type for e in second.edges] == ["HAS_CHILD", "REFERS"]
    beyond = await nav.expand_node("StructureNode", "sn1", skip=10, **KW)
    assert beyond.nodes == [] and beyond.edges == [] and beyond.total == 2
    everything = await nav.expand_node("StructureNode", "sn2", limit=None, **KW)
    assert _pairs(everything.nodes) == [
        ("ExtractionResult", "er1"),
        ("InfoUnit", "iu1"),
        ("StructureNode", "sn1"),
    ]


async def test_expand_node_filters_and_excludes_the_node_itself(nav) -> None:
    only_out = await nav.expand_node("StructureNode", "sn1", direction="out", **KW)
    assert _pairs(only_out.nodes) == [("StructureNode", "sn2")] and only_out.total == 1
    only_in = await nav.expand_node("StructureNode", "sn1", direction="in", **KW)
    assert _pairs(only_in.nodes) == [("Document", A_KEY)]
    refers = await nav.expand_node("StructureNode", "sn1", edge_types=["REFERS"], **KW)
    assert _pairs(refers.nodes) == [("StructureNode", "sn2")]
    assert [e.type for e in refers.edges] == ["REFERS"]
    by_label = await nav.expand_node("StructureNode", "sn1", neighbor_labels=["Document"], **KW)
    assert _pairs(by_label.nodes) == [("Document", A_KEY)] and by_label.total == 1
    by_role = await nav.expand_node("StructureNode", "sn1", neighbor_labels=["Section"], **KW)
    assert _pairs(by_role.nodes) == [("StructureNode", "sn2")]
    # sn2 has a relationship with itself: it is not its own neighbour
    assert ("StructureNode", "sn2") not in _pairs(
        (await nav.expand_node("StructureNode", "sn2", **KW)).nodes
    )


async def test_expand_node_catalogue_node_only_shows_the_tenant_neighbours(nav) -> None:
    mine = await nav.expand_node("CatalogModel", "Drug", **KW)
    assert _pairs(mine.nodes) == [
        ("ExtractionResult", "er1"),
        ("ModelField", FIELD_KEY),
        ("Theme", "t/default"),
    ]
    assert [(e.type, e.source_label, e.target_label) for e in mine.edges] == [
        ("HAS_FIELD", "CatalogModel", "ModelField"),  # by source, then target, then type
        ("BELONGS_TO_THEME", "CatalogModel", "Theme"),
        ("USES_PRIMARY_MODEL", "ExtractionResult", "CatalogModel"),
    ]
    other = await nav.expand_node("CatalogModel", "Drug", tenant_id=OTHER)
    assert ("ExtractionResult", "erx") in _pairs(other.nodes)
    assert ("ExtractionResult", "er1") not in _pairs(other.nodes)
    nobody = await nav.expand_node("CatalogModel", "Drug", tenant_id="nobody")
    assert _pairs(nobody.nodes) == [("ModelField", FIELD_KEY), ("Theme", "t/default")]
    everyone = await nav.expand_node("CatalogModel", "Drug")
    assert [k for _, k in _pairs(everyone.nodes)][:2] == ["er1", "erx"] and everyone.total == 4


async def test_expand_node_missing_or_outside_the_scope_is_empty_and_arguments_are_checked(
    nav,
) -> None:
    for label, key in (
        ("StructureNode", "missing"),
        ("Document", OTHER_A_KEY),
        ("Document", "not-json"),
    ):
        out = await nav.expand_node(label, key, **KW)
        assert (out.nodes, out.edges, out.total) == ([], [], 0)
    for bad in (
        {"skip": -1},
        {"limit": 0},
        {"limit": True},
        {"direction": "up"},
        {"edge_types": ["bad type"]},
        {"edge_types": "HAS_CHILD"},
        {"neighbor_labels": ["bad label"]},
    ):
        with pytest.raises(NavigationError):
            await nav.expand_node("StructureNode", "sn1", **bad, **KW)
    with pytest.raises(NavigationError):
        await nav.expand_node("Nope", "x")


# -- search_nodes -------------------------------------------------------------------


def _hits(matches):
    return [(m.node.label, m.node.key, m.field) for m in matches]


async def test_search_nodes_by_exact_key_and_by_field(nav) -> None:
    assert _hits(await nav.search_nodes("sn1", **KW)) == [("StructureNode", "sn1", "key")]
    assert _hits(await nav.search_nodes(A_KEY, **KW)) == [("Document", A_KEY, "key")]
    assert _hits(await nav.search_nodes("Intro", labels=["StructureNode"], **KW)) == [
        ("StructureNode", "sn1", "title")
    ]
    # case insensitive, and the first field that matches names the hit
    assert _hits(await nav.search_nodes("SN2", labels=["StructureNode"], **KW)) == [
        ("StructureNode", "sn2", "id")
    ]
    assert _hits(await nav.search_nodes("a.pdf", labels=["Document"], **KW)) == [
        ("Document", A_KEY, "name")
    ]
    assert _hits(await nav.search_nodes("DEFAULT", labels=["Theme"], **KW)) == [
        ("Theme", "t/default", "path")
    ]
    assert _hits(await nav.search_nodes("medicine", labels=["CatalogModel"], **KW)) == [
        ("CatalogModel", "Drug", "description")
    ]


async def test_search_nodes_uses_the_info_unit_and_record_searches(nav) -> None:
    hits = _hits(await nav.search_nodes("aspirin", **KW))
    assert hits == [("InfoUnit", "iu1", "text"), ("ModelInstance", "d1", "name")]
    node = (await nav.search_nodes("aspirin", labels=["ModelInstance"], **KW))[0].node
    assert node.properties["name"] == "Aspirin" and node.types == ["ModelInstance"]
    assert _hits(await nav.search_nodes("d1", labels=["ModelInstance"], **KW)) == [
        ("ModelInstance", "d1", "key")
    ]


async def test_search_nodes_orders_pages_and_respects_the_scope(nav) -> None:
    drug = ["CatalogModel", "ExtractionResult", "ModelField"]
    assert _hits(await nav.search_nodes("drug", labels=drug, **KW)) == [
        ("CatalogModel", "Drug", "name"),
        ("ExtractionResult", "er1", "model_class"),
        ("ModelField", FIELD_KEY, "model"),
    ]
    page = await nav.search_nodes("drug", labels=drug, limit=1, skip=1, **KW)
    assert _hits(page) == [("ExtractionResult", "er1", "model_class")]
    other = _hits(await nav.search_nodes("drug", labels=drug, tenant_id=OTHER))
    assert ("ExtractionResult", "erx", "model_class") in other and all(
        k != "er1" for _, k, _ in other
    )
    # the exact key comes first
    exact = _hits(await nav.search_nodes("Drug", labels=drug, **KW))
    assert exact[0] == ("CatalogModel", "Drug", "key")
    assert await nav.search_nodes("aspirin other", **KW) == []
    assert _hits(await nav.search_nodes("aspirin other", tenant_id=OTHER))[0][:2] == (
        "ModelInstance",
        "x1",
    )


async def test_search_nodes_edge_cases(nav) -> None:
    assert await nav.search_nodes("a", **KW) == []
    assert await nav.search_nodes("", **KW) == []
    assert await nav.search_nodes("zz-no-match", **KW) == []
    assert await nav.search_nodes("100%_", **KW) == []  # not a pattern
    for bad in ({"limit": 0}, {"skip": -1}, {"labels": ["Nope"]}, {"labels": "Document"}):
        with pytest.raises(NavigationError):
            await nav.search_nodes("intro", **bad, **KW)


# -- the scoped navigator fills the scope ------------------------------------------


async def test_a_scoped_navigator_fills_and_guards_the_scope_of_the_new_methods(nav) -> None:
    scoped = nav.scoped(tenant_id=TENANT)
    assert await scoped.get_node("Document", OTHER_A_KEY) is None
    assert (await scoped.get_node("Document", A_KEY)).key == A_KEY
    assert await scoped.count_node_relationships("Document", OTHER_A_KEY) == []
    assert (await scoped.expand_node("Document", OTHER_A_KEY)).total == 0
    assert _pairs((await scoped.expand_node("CatalogModel", "Drug")).nodes)[0] == (
        "ExtractionResult",
        "er1",
    )
    assert _hits(await scoped.search_nodes("drug", labels=["ExtractionResult"])) == [
        ("ExtractionResult", "er1", "model_class")
    ]
    for call in (
        scoped.get_node("Document", A_KEY, tenant_id=OTHER),
        scoped.count_node_relationships("Document", A_KEY, tenant_id=OTHER),
        scoped.expand_node("Document", A_KEY, tenant_id=OTHER),
        scoped.search_nodes("drug", tenant_id=OTHER),
    ):
        with pytest.raises(NavigationError):
            await call
