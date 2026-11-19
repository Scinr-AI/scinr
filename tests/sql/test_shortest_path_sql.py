"""
tests/sql/test_shortest_path_sql.py -- ``SqlGraphNavigator.shortest_path`` (a bidirectional
breadth-first search, ``navigation/sql/_bfs.py``) against the former recursive CTE of every
walk (``_shortest_oracle.py``), on both engines.

The random graphs mix two tenants, public nodes, catalogue nodes, cross-tenant edges, parallel
edges, self loops and selectors that name several nodes (or the same nodes on both sides), so
the scope guards, the ``key != origin`` rule and the tie-break by path string are all exercised.
"""

from __future__ import annotations

import random

import pytest
import sqlalchemy as sa

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import NodeSelector
from scinr.newton.navigation.sql.navigator import SqlGraphNavigator
from scinr.newton.sql import tables as T
from scinr.newton.sql.engine import get_sync_engine
from scinr.newton.sql.schema import ensure_schema
from scinr.newton.utils.tenancy import PUBLIC_TENANT
from tests.sql._shortest_oracle import old_shortest_path


@pytest.fixture
async def nav(sql_backend):
    navigator = SqlGraphNavigator()
    await navigator.connect()
    try:
        yield navigator
    finally:
        await navigator.close()


def _edge(rel_type: str, src: tuple[str, str], dst: tuple[str, str], tenant: str | None) -> dict:
    row = {
        "type": rel_type,
        "src_label": src[0],
        "src_key": src[1],
        "dst_label": dst[0],
        "dst_key": dst[1],
        "_ident": "",
        "props": {"w": f"{rel_type}:{src[1]}>{dst[1]}"},
    }
    if tenant is not None:
        row["tenant_id"] = tenant
    return row


def _write(decisions: list[dict], models: list[str], edges: list[dict], catalog: list[dict]):
    ensure_schema()
    with get_sync_engine().begin() as connection:
        for table in (T.edges, T.catalog_edges, T.model_decisions, T.catalog_models):
            connection.execute(sa.delete(table))
        connection.execute(sa.insert(T.model_decisions), decisions)
        if models:
            connection.execute(
                sa.insert(T.catalog_models), [{"name": m, "props": {}} for m in models]
            )
        if edges:
            connection.execute(sa.insert(T.edges), edges)
        if catalog:
            connection.execute(sa.insert(T.catalog_edges), catalog)


def _decision(uid: str, tenant: str, job: str, group: str) -> dict:
    return {
        "uid": uid,
        "tenant_id": tenant,
        "job_id": job,
        "source": group,
        "props": {},
        "_document_key": "doc",
    }


def _random_graph(rng: random.Random) -> tuple[list[str], list[str]]:
    # Keys share prefixes ("d1" / "d10") so the tie-break by path string is not trivial.
    uids = [f"d{i}" for i in rng.sample(range(12), rng.randint(6, 10))]
    tenants = ["a", "b", PUBLIC_TENANT]
    owner = {u: rng.choice(tenants) for u in uids}
    decisions = [_decision(u, owner[u], rng.choice(("j1", "j2")), rng.choice("gh")) for u in uids]
    models = ["M", "M1"][: rng.randint(0, 2)]
    edges: list[dict] = []
    for _ in range(rng.randint(len(uids), 3 * len(uids))):
        a, b = rng.choice(uids), rng.choice(uids)
        edges.append(
            _edge(rng.choice(("R", "S")), ("ModelDecision", a), ("ModelDecision", b), owner[a])
        )
    for model in models:
        for uid in rng.sample(uids, rng.randint(1, 3)):
            edges.append(
                _edge("MATCHED_MODEL", ("ModelDecision", uid), ("CatalogModel", model), owner[uid])
            )
    catalog = (
        [_edge("HAS_FIELD", ("CatalogModel", "M"), ("CatalogModel", "M1"), None)]
        if len(models) == 2
        else []
    )
    # One edge per identity: the unique constraint (src, type, dst, _ident).
    unique = {(e["type"], e["src_key"], e["dst_key"]): e for e in edges}
    _write(decisions, models, list(unique.values()), catalog)
    return uids, models


def _selector(rng: random.Random, uids: list[str], models: list[str]) -> NodeSelector:
    roll = rng.random()
    if roll < 0.55:
        return NodeSelector(type="ModelDecision", key="uid", value=rng.choice(uids))
    if roll < 0.8:
        return NodeSelector(type="ModelDecision", key="source", value=rng.choice("gh"))
    if roll < 0.95 and models:
        return NodeSelector(type="CatalogModel", key="name", value=rng.choice(models))
    return NodeSelector(type="ModelDecision", key="uid", value="missing")


_SCOPES = [
    {},
    {"tenant_id": "a"},
    {"tenant_id": "a", "include_public": True},
    {"job_id": "j1"},
    {"tenant_id": "b", "job_id": ["j1", "j2"]},
]


async def _outcome(call):
    try:
        result = await call
    except NavigationError as exc:
        return ("error", str(exc))
    return ("ok", None if result is None else result.model_dump())


@pytest.mark.parametrize("seed", range(10))
async def test_shortest_path_matches_the_former_cte(nav, seed) -> None:
    rng = random.Random(seed)
    uids, models = _random_graph(rng)
    found = 0
    for _ in range(40):
        source = _selector(rng, uids, models)
        target = source if rng.random() < 0.1 else _selector(rng, uids, models)
        kwargs = {
            "max_hops": rng.choice((1, 2, 3, 4, 6)),
            "edge_types": rng.choice((None, None, ["R"], ["R", "MATCHED_MODEL", "HAS_FIELD"])),
            **rng.choice(_SCOPES),
        }
        want = await _outcome(old_shortest_path(nav, source, target, **kwargs))
        got = await _outcome(nav.shortest_path(source, target, **kwargs))
        assert got == want, (source, target, kwargs)
        found += want[0] == "ok" and want[1] is not None
    assert found  # the comparison is not only between two Nones


async def test_ties_go_to_the_smallest_path_string(nav) -> None:
    # s - d10 - g and s - d1 - g tie; "d1" + PATH_SEP sorts before "d10".
    decisions = [_decision(u, "a", "j1", "g") for u in ("s", "d1", "d10", "g")]
    edges = [
        _edge("R", ("ModelDecision", "s"), ("ModelDecision", "d10"), "a"),
        _edge("R", ("ModelDecision", "d10"), ("ModelDecision", "g"), "a"),
        _edge("S", ("ModelDecision", "g"), ("ModelDecision", "d1"), "a"),
        _edge("S", ("ModelDecision", "d1"), ("ModelDecision", "s"), "a"),
    ]
    _write(decisions, [], edges, [])
    s = NodeSelector(type="ModelDecision", key="uid", value="s")
    g = NodeSelector(type="ModelDecision", key="uid", value="g")
    path = await nav.shortest_path(s, g, tenant_id="a")
    assert path is not None and path.length == 2
    assert [n.properties["uid"] for n in path.nodes] == ["s", "d1", "g"]
    assert [r["type"] for r in path.relationships] == ["S", "S"]
    only_r = await nav.shortest_path(s, g, edge_types=["R"])
    assert [n.properties["uid"] for n in only_r.nodes] == ["s", "d10", "g"]
    assert await nav.shortest_path(s, g, max_hops=1) is None
    assert await nav.shortest_path(s, g, edge_types=["NOPE"]) is None


async def test_inner_nodes_stay_inside_the_tenant(nav) -> None:
    # s -(b's node)- g is short; the long way stays in tenant a. A catalogue node may be the
    # goal but never a bridge.
    decisions = [_decision("s", "a", "j1", "g"), _decision("x", "b", "j1", "g")]
    decisions += [_decision(u, "a", "j1", "g") for u in ("m1", "m2", "g")]
    path = [("s", "m1"), ("m1", "m2"), ("m2", "g")]
    edges = [_edge("R", ("ModelDecision", u), ("ModelDecision", v), "a") for u, v in path]
    edges += [
        _edge("R", ("ModelDecision", "s"), ("ModelDecision", "x"), "a"),
        _edge("R", ("ModelDecision", "x"), ("ModelDecision", "g"), "b"),
        _edge("MATCHED_MODEL", ("ModelDecision", "s"), ("CatalogModel", "M"), "a"),
        _edge("MATCHED_MODEL", ("ModelDecision", "x"), ("CatalogModel", "M"), "b"),
    ]
    _write(decisions, ["M"], edges, [])
    s = NodeSelector(type="ModelDecision", key="uid", value="s")
    g = NodeSelector(type="ModelDecision", key="uid", value="g")
    x = NodeSelector(type="ModelDecision", key="uid", value="x")
    assert (await nav.shortest_path(s, g)).length == 2
    scoped = await nav.shortest_path(s, g, tenant_id="a")
    assert [n.properties["uid"] for n in scoped.nodes] == ["s", "m1", "m2", "g"]
    assert await nav.shortest_path(s, g, tenant_id="a", max_hops=2) is None
    assert await nav.shortest_path(s, x, tenant_id="a") is None  # the goal is b's
    assert await nav.shortest_path(x, g, tenant_id="a") is None  # the start is b's
    model = NodeSelector(type="CatalogModel", key="name", value="M")
    assert (await nav.shortest_path(s, model, tenant_id="a")).length == 1
    assert await nav.shortest_path(x, s, tenant_id="b", edge_types=["MATCHED_MODEL"]) is None
    assert (await nav.shortest_path(x, s, edge_types=["MATCHED_MODEL"])).length == 2


async def test_selectors_naming_several_nodes(nav) -> None:
    decisions = [_decision(u, "a", "j1", grp) for u, grp in (("a1", "p"), ("a2", "p"), ("z", "q"))]
    chain = [("a1", "z"), ("z", "a2")]
    _write(
        decisions,
        [],
        [_edge("R", ("ModelDecision", u), ("ModelDecision", v), "a") for u, v in chain],
        [],
    )
    group = NodeSelector(type="ModelDecision", key="source", value="p")
    one = NodeSelector(type="ModelDecision", key="uid", value="a1")
    # The same nodes on both sides: a path between two of them, never from a node to itself.
    both = await nav.shortest_path(group, group)
    assert [n.properties["uid"] for n in both.nodes] == ["a1", "z", "a2"]
    assert [n.properties["uid"] for n in (await nav.shortest_path(one, group)).nodes] == [
        "a1",
        "z",
        "a2",
    ]
    with pytest.raises(NavigationError, match="same node"):
        await nav.shortest_path(one, one)
