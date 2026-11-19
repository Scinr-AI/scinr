"""
tests/sql/test_navigator_sql.py -- Behaviour of ``SqlGraphNavigator`` that the equivalence suite
cannot compare with Neo4j: the lifecycle, an empty database, ``execute_raw`` (each engine has
its own language and its read-only guarantee), the walk helper and the tenant isolation of
the free graph walks.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from scinr.newton import configure
from scinr.newton.exceptions import GraphConnectionError, NavigationError
from scinr.newton.navigation.models import NodeSelector
from scinr.newton.navigation.sql._walk import PATH_SEP, reached, walk
from scinr.newton.navigation.sql.navigator import SqlGraphNavigator
from scinr.newton.sql import tables as T
from scinr.newton.sql.engine import get_sync_engine, reset_engines
from scinr.newton.sql.schema import ensure_schema


@pytest.fixture
async def nav(sql_backend):
    navigator = SqlGraphNavigator()
    await navigator.connect()
    try:
        yield navigator
    finally:
        await navigator.close()


def _edge(rel_type: str, src: tuple[str, str], dst: tuple[str, str], tenant: str = "t") -> dict:
    return {
        "tenant_id": tenant,
        "type": rel_type,
        "src_label": src[0],
        "src_key": src[1],
        "dst_label": dst[0],
        "dst_key": dst[1],
        "_ident": "",
        "props": {},
    }


def _insert(*statements) -> None:
    with get_sync_engine().begin() as connection:
        for statement in statements:
            connection.execute(statement)


# ── lifecycle ────────────────────────────────────────────────────────────────


async def test_connect_creates_the_schema_and_reports_the_dialect(sql_backend) -> None:
    navigator = SqlGraphNavigator()
    assert navigator.dialect == "sql"
    await navigator.connect()
    assert navigator.dialect == sql_backend.backend
    assert await navigator.ping() is True
    await navigator.close()
    await navigator.close()  # closing twice (or never connecting) is harmless


async def test_an_empty_database_navigates_to_nothing(nav) -> None:
    assert await nav.list_root_documents() == []
    assert await nav.count_root_documents() == 0
    assert await nav.get_structure_node("x") is None
    assert await nav.get_document_tree("nope") is None
    assert await nav.list_node_labels() == []
    assert await nav.list_themes() == []
    assert await nav.list_relationship_types(structural_only=False) == []
    summary = await nav.get_graph_summary()
    assert (summary.total_nodes, summary.total_relationships, summary.documents) == (0, 0, 0)
    assert (await nav.get_catalog_graph()).models == []
    assert await nav.search_info_units("anything") == []
    assert await nav.neighbors(NodeSelector(type="Document", key="path", value="x")) == []


async def test_connect_to_an_unreachable_database_is_a_connection_error(tmp_path) -> None:
    configure(
        graph_backend="sqlite",
        storage_backend="sqlite",
        freeze_backend="sqlite",
        sql_url=f"sqlite:///{tmp_path}/missing-folder/newton.db",
    )
    try:
        with pytest.raises(GraphConnectionError):
            await SqlGraphNavigator().connect()
    finally:
        reset_engines()


# ── execute_raw ──────────────────────────────────────────────────────────────


async def test_execute_raw_reads_with_parameters(nav, sql_backend) -> None:
    _insert(
        sa.insert(T.documents).values(
            _key='["t","a.txt",1]', tenant_id="t", path="a.txt", version=1, latest=True
        )
    )
    # Table names need no schema: Postgres puts the configured one first in the search_path.
    rows = await nav.execute_raw(
        "SELECT path, version FROM documents WHERE tenant_id = :tenant", {"tenant": "t"}
    )
    assert rows == [{"path": "a.txt", "version": 1}]
    assert await nav.execute_raw_one("SELECT count(*) AS n FROM documents") == {"n": 1}
    assert (
        await nav.execute_raw_one("SELECT path FROM documents WHERE path = :p", {"p": "x"}) is None
    )
    assert await nav.execute_raw("SELECT 1 AS one", dialect=sql_backend.backend) == [{"one": 1}]


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (_key, tenant_id, path, version) VALUES ('k', 't', 'p', 1)",
        "UPDATE documents SET path = 'z'",
        "DELETE FROM documents",
        "DROP TABLE documents",
    ],
)
async def test_execute_raw_cannot_write(nav, statement) -> None:
    _insert(
        sa.insert(T.documents).values(
            _key='["t","a.txt",1]', tenant_id="t", path="a.txt", version=1, latest=True
        )
    )
    with pytest.raises(NavigationError, match="read-only"):
        await nav.execute_raw(statement)
    # the engine refused it: the data is intact and the connection is usable again
    assert await nav.execute_raw_one("SELECT count(*) AS n FROM documents") == {"n": 1}
    _insert(sa.insert(T.entity_labels).values(label="after", props={}))  # the engine is writable


async def test_execute_raw_checks_the_dialect_and_the_query(nav) -> None:
    with pytest.raises(NavigationError, match="dialect"):
        await nav.execute_raw("SELECT 1", dialect="cypher")
    with pytest.raises(NavigationError):
        await nav.execute_raw("   ")


# ── walks ────────────────────────────────────────────────────────────────────


def _chain(nav) -> None:
    """a -R-> b -R-> c -R-> a (a cycle), b -HAS_X-> d."""
    ensure_schema()
    n = lambda k: ("StructureNode", k)  # noqa: E731
    _insert(
        sa.insert(T.edges).values(
            [
                _edge("R", n("a"), n("b")),
                _edge("R", n("b"), n("c")),
                _edge("R", n("c"), n("a")),
                _edge("HAS_X", n("b"), n("d")),
            ]
        )
    )


async def _walk_keys(nav, **options) -> list[tuple[str, int]]:
    start = sa.select(sa.literal("StructureNode").label("label"), sa.literal("a").label("key"))
    cte = walk(start, **options)
    rows = await nav._read(sa.select(cte.c.key, cte.c.depth).order_by(cte.c.depth, cte.c.key))
    return [(r["key"], r["depth"]) for r in rows]


async def test_walk_is_bounded_and_survives_cycles(nav) -> None:
    _chain(nav)
    assert await _walk_keys(nav, max_hops=1) == [("a", 0), ("b", 1)]
    assert await _walk_keys(nav, max_hops=3, types=["R"]) == [
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("a", 3),
    ]
    assert await _walk_keys(nav, max_hops=2, prefix="HAS_") == [("a", 0)]
    assert await _walk_keys(nav, max_hops=2, direction="in") == [("a", 0), ("c", 1), ("b", 2)]
    with pytest.raises(ValueError, match="max_hops"):
        walk(sa.select(sa.literal("x"), sa.literal("y")), max_hops=0)


async def test_walk_tracks_paths_and_does_not_go_back_on_the_same_edge(nav) -> None:
    _chain(nav)
    start = sa.select(sa.literal("StructureNode").label("label"), sa.literal("a").label("key"))
    cte = walk(start, max_hops=2, types=["R"], track_path=True)
    rows = await nav._read(sa.select(cte.c.path).where(cte.c.depth == 2))
    assert [r["path"].split(PATH_SEP) for r in rows] == [["a", "b", "c"]]
    both = walk(start, max_hops=2, direction="both", no_backtrack=True)
    two = await nav._read(sa.select(both.c.key).where(both.c.depth == 2).order_by(both.c.key))
    # a-b-c, a-b-d and a-c-b (never a-b-a through the edge just used)
    assert [r["key"] for r in two] == ["b", "c", "d"]
    free = walk(start, max_hops=2, direction="both", name="free")
    back = await nav._read(sa.select(free.c.key).where(free.c.depth == 2, free.c.key == "a"))
    assert back  # without the guard a hop can undo the previous one


async def test_reached_keeps_each_node_once(nav) -> None:
    _chain(nav)
    start = sa.select(sa.literal("StructureNode").label("label"), sa.literal("a").label("key"))
    cte = walk(start, max_hops=4, types=["R"])
    nodes = reached(cte, "StructureNode")
    rows = await nav._read(sa.select(nodes.c.key).order_by(nodes.c.key))
    assert [r["key"] for r in rows] == ["a", "b", "c"]


# ── tenant isolation of the free walks ───────────────────────────────────────


def _two_tenants_sharing_a_model() -> None:
    ensure_schema()
    _insert(
        sa.insert(T.catalog_models).values(name="M", props={}),
        sa.insert(T.model_decisions).values(
            [
                {"uid": "d-a", "tenant_id": "a", "job_id": "ja", "props": {}, "_document_key": "x"},
                {"uid": "d-b", "tenant_id": "b", "job_id": "jb", "props": {}, "_document_key": "y"},
            ]
        ),
        sa.insert(T.edges).values(
            [
                _edge("MATCHED_MODEL", ("ModelDecision", "d-a"), ("CatalogModel", "M"), "a"),
                _edge("MATCHED_MODEL", ("ModelDecision", "d-b"), ("CatalogModel", "M"), "b"),
            ]
        ),
    )


async def test_a_scoped_walk_never_crosses_a_catalogue_node_into_another_tenant(nav) -> None:
    _two_tenants_sharing_a_model()
    decision = NodeSelector(type="ModelDecision", key="uid", value="d-a")
    everything = await nav.neighbors(decision, depth=2)
    assert {n.properties.get("uid") or n.properties.get("name") for n in everything} == {
        "M",
        "d-b",
    }
    scoped = await nav.neighbors(decision, depth=2, tenant_id="a")
    assert [n.properties.get("name") for n in scoped] == ["M"]  # the catalogue node is the far end
    assert await nav.neighbors(decision, depth=2, tenant_id="b") == []  # the anchor is not b's
    path = await nav.shortest_path(
        decision, NodeSelector(type="ModelDecision", key="uid", value="d-b"), tenant_id="a"
    )
    assert path is None
    unscoped = await nav.shortest_path(
        decision, NodeSelector(type="ModelDecision", key="uid", value="d-b")
    )
    assert unscoped is not None and unscoped.length == 2
    assert [r["type"] for r in unscoped.relationships] == ["MATCHED_MODEL", "MATCHED_MODEL"]
    graph = await nav.subgraph(decision, depth=2, tenant_id="a")
    assert {n.properties.get("uid") or n.properties.get("name") for n in graph.nodes} == {
        "d-a",
        "M",
    }


async def test_a_shortest_path_to_the_same_node_is_refused(nav) -> None:
    _two_tenants_sharing_a_model()
    decision = NodeSelector(type="ModelDecision", key="uid", value="d-a")
    with pytest.raises(NavigationError, match="same node"):
        await nav.shortest_path(decision, decision)
    with pytest.raises(NavigationError, match="max_hops"):
        await nav.shortest_path(decision, decision, max_hops=0)
    with pytest.raises(NavigationError, match="depth"):
        await nav.subgraph(decision, depth=0)
    with pytest.raises(NavigationError):
        await nav.neighbors(NodeSelector(type="bad type", key="uid", value=1))
