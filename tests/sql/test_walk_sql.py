"""
tests/sql/test_walk_sql.py -- ``navigation/sql/_walk.walk`` (one indexed branch per edge table
and direction) against the former single join on ``src = node OR dst = node``
(``_walk_oracle.py``), on both engines.

The random graphs have parallel edges (other type, or same type and another ``_ident``),
self loops, edges both ways, catalogue edges, an end with no node row and the same key under
two labels. Every option combination the navigators use is compared row for row (the CTE
rows themselves, not only what a caller keeps), and ``neighbors`` / ``subgraph`` are compared
end to end with the former walk patched in. On SQLite the Postgres shape of the step (one
``SELECT`` over a ``UNION ALL`` of the branches) is checked too, since SQLite can run it.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from scinr.newton.navigation.models import NodeSelector
from scinr.newton.navigation.scope import make_scope
from scinr.newton.navigation.sql import _power
from scinr.newton.navigation.sql._walk import walk
from scinr.newton.navigation.sql.navigator import SqlGraphNavigator
from scinr.newton.sql import tables as T
from scinr.newton.sql.dialect import PostgresDialect, SqliteDialect
from scinr.newton.sql.engine import get_dialect_for_config, get_sync_engine
from scinr.newton.sql.schema import ensure_schema
from scinr.newton.utils.tenancy import PUBLIC_TENANT
from tests.sql._walk_oracle import old_walk

_TYPES = ("HAS_STRUCTURE", "HAS_CHILD", "HAS_FIELD", "R", "S")
_TENANTS = ("a", "b", PUBLIC_TENANT)


@pytest.fixture
async def nav(sql_backend):
    navigator = SqlGraphNavigator()
    await navigator.connect()
    try:
        yield navigator
    finally:
        await navigator.close()


def _edge(rel_type: str, src: tuple[str, str], dst: tuple[str, str], ident: str = "") -> dict:
    return {
        "type": rel_type,
        "src_label": src[0],
        "src_key": src[1],
        "dst_label": dst[0],
        "dst_key": dst[1],
        "_ident": ident,
        "props": {"w": f"{rel_type}:{src[1]}>{dst[1]}{ident}"},
    }


def _random_graph(rng: random.Random) -> tuple[list[str], list[str]]:
    """Write a random graph; the uids of its decisions and the names of its models."""
    uids = [f"d{i}" for i in rng.sample(range(14), rng.randint(5, 10))]
    owner = {u: rng.choice(_TENANTS) for u in uids}
    models = ["M", "M1", "d1"][: rng.randint(0, 3)]  # "d1": a key shared with a decision
    nodes = [("ModelDecision", u) for u in uids] + [("CatalogModel", m) for m in models]
    nodes.append(("StructureNode", "ghost"))  # an end without a node row
    edges: list[dict] = []
    for _ in range(rng.randint(len(nodes), 3 * len(nodes))):
        a, b = rng.choice(nodes), rng.choice(nodes)
        roll = rng.random()
        if roll < 0.08:
            b = a  # a self loop
        edges.append(_edge(rng.choice(_TYPES), a, b, rng.choice(("", "", "", "x"))))
        if roll > 0.9:  # the same ends the other way, or with another type
            edges.append(_edge(rng.choice(_TYPES), b, a) if roll > 0.95 else _edge("S", a, b))
    plain, catalog = [], []
    for edge in edges:
        if rng.random() < 0.15:
            catalog.append(edge)
        else:
            tenant_of = owner.get(edge["src_key"], rng.choice(_TENANTS))
            plain.append({**edge, "tenant_id": tenant_of})
    # The unique constraint is (src, type, dst, _ident) per table.
    plain = list({_identity(e): e for e in plain}.values())
    catalog = list({_identity(e): e for e in catalog}.values())
    decisions = [
        {
            "uid": u,
            "tenant_id": owner[u],
            "job_id": rng.choice(("j1", "j2")),
            "source": rng.choice("gh"),
            "props": {},
            "_document_key": "doc",
        }
        for u in uids
    ]
    ensure_schema()
    with get_sync_engine().begin() as connection:
        for table in (T.edges, T.catalog_edges, T.model_decisions, T.catalog_models):
            connection.execute(sa.delete(table))
        connection.execute(sa.insert(T.model_decisions), decisions)
        if models:
            connection.execute(
                sa.insert(T.catalog_models), [{"name": m, "props": {}} for m in models]
            )
        if plain:
            connection.execute(sa.insert(T.edges), plain)
        if catalog:
            connection.execute(sa.insert(T.catalog_edges), catalog)
    return uids, models


def _identity(edge: dict) -> tuple:
    return tuple(
        edge[k] for k in ("src_label", "src_key", "type", "dst_label", "dst_key", "_ident")
    )


def _start(rng: random.Random, uids: list[str], models: list[str]) -> sa.Select | sa.CompoundSelect:
    md, cm = T.model_decisions, T.catalog_models
    chosen = rng.sample(uids, rng.randint(1, min(3, len(uids))))
    parts: list[sa.Select] = [
        sa.select(sa.literal("ModelDecision", sa.Text).label("label"), md.c.uid.label("key")).where(
            md.c.uid.in_(chosen)
        )
    ]
    if models and rng.random() < 0.4:
        parts.append(
            sa.select(
                sa.literal("CatalogModel", sa.Text).label("label"), cm.c.name.label("key")
            ).where(cm.c.name == rng.choice(models))
        )
    if rng.random() < 0.15:
        parts.append(
            sa.select(
                sa.literal("StructureNode", sa.Text).label("label"),
                sa.literal("ghost", sa.Text).label("key"),
            )
        )
    return parts[0] if len(parts) == 1 else sa.union_all(*parts)


def _random_options(rng: random.Random, nav: SqlGraphNavigator) -> dict:
    """Any combination of the options of :func:`walk`."""
    options: dict = {
        "max_hops": rng.choice((1, 2, 3, 4)),
        "direction": rng.choice(("out", "in", "both")),
        "edge_table": rng.choice((None, T.all_edges)),
    }
    roll = rng.random()
    if roll < 0.3:
        options["types"] = rng.sample(_TYPES, rng.randint(1, 3))
    elif roll < 0.5:
        options["prefix"] = "HAS_"
    elif roll < 0.6:
        options["types"] = ["HAS_CHILD", "R"]
        options["prefix"] = "HAS_"
    for flag in ("track_path", "track_types", "no_backtrack"):
        options[flag] = rng.random() < 0.5
    options["path_labels"] = options["track_path"] and rng.random() < 0.5
    if rng.random() < 0.2:
        options["only_labels"] = rng.choice((("ModelDecision",), ("ModelDecision", "CatalogModel")))
    if rng.random() < 0.4:
        sc = make_scope(rng.choice(("a", "b")), rng.random() < 0.5, None, None)
        options["guard"] = nav._tenant_guard(sc, nav._node_index())
    elif rng.random() < 0.1:
        options["guard"] = lambda cte: cte.c.label != "CatalogModel"
    return options


#: The combinations the navigators use (direction, types / prefix and flags).
_CALLER_OPTIONS = [
    {"types": ("HAS_STRUCTURE", "HAS_CHILD")},
    {"types": ("HAS_STRUCTURE", "HAS_CHILD"), "track_path": True},
    {"direction": "in", "types": ("HAS_STRUCTURE", "HAS_CHILD"), "track_path": True},
    {"direction": "in", "types": ("HAS_STRUCTURE", "HAS_CHILD")},
    {"types": ("HAS_STRUCTURE",)},
    {"prefix": "HAS_"},
    {"direction": "in", "prefix": "HAS_"},
    {"direction": "out", "types": ("R",), "track_types": True},
    {"direction": "in", "types": None, "track_types": True},
    {"track_path": True, "track_types": True, "only_labels": ("ModelDecision",)},
    *[
        {
            "direction": direction,
            "no_backtrack": True,
            "types": types,
            "edge_table": T.all_edges,
            "scoped": scoped,
        }
        for direction in ("out", "in", "both")
        for types in (None, ["R", "S"])
        for scoped in (False, True)
    ],
    {
        "direction": "both",
        "no_backtrack": True,
        "track_path": True,
        "path_labels": True,
        "edge_table": T.all_edges,
        "scoped": True,
    },
]


def _rows(statement) -> Counter:
    with get_sync_engine().connect() as connection:
        return Counter(tuple(row) for row in connection.execute(statement))


def _shapes() -> list:
    """The dialects whose step shape this engine can run (SQLite runs both)."""
    engine = get_dialect_for_config()
    return [SqliteDialect(), PostgresDialect()] if engine.recursive_branches else [engine]


def _compare(start, options: dict, nav: SqlGraphNavigator) -> int:
    options = dict(options)
    if options.pop("scoped", False):
        options["guard"] = nav._tenant_guard(make_scope("a", True, None, None), nav._node_index())
    want = _rows(sa.select(old_walk(start, **options)))
    for dialect in _shapes():
        got = _rows(sa.select(walk(start, dialect=dialect, **options)))
        assert got == want, (type(dialect).__name__, options)
    return sum(want.values())


@pytest.mark.parametrize("seed", range(12))
async def test_walk_rows_match_the_former_join(nav, seed) -> None:
    rng = random.Random(seed)
    uids, models = _random_graph(rng)
    total = 0
    for options in _CALLER_OPTIONS:
        start = _start(rng, uids, models)
        total += _compare(start, {"max_hops": rng.choice((1, 2, 3, 4)), **options}, nav)
    for _ in range(40):
        total += _compare(_start(rng, uids, models), _random_options(rng, nav), nav)
    assert total > 0


async def test_loops_and_parallel_edges(nav) -> None:
    # a -R-> a (loop), a -R-> b, a -S-> b, a -R[x]-> b (parallel, same type), b -R-> a.
    md = [
        {
            "uid": u,
            "tenant_id": "a",
            "job_id": "j",
            "source": "g",
            "props": {},
            "_document_key": "d",
        }
        for u in ("a", "b")
    ]
    pairs = [("R", "a", "a", ""), ("R", "a", "b", ""), ("S", "a", "b", ""), ("R", "a", "b", "x")]
    pairs.append(("R", "b", "a", ""))
    edges = [
        {**_edge(t, ("ModelDecision", s), ("ModelDecision", d), i), "tenant_id": "a"}
        for t, s, d, i in pairs
    ]
    ensure_schema()
    with get_sync_engine().begin() as connection:
        for table in (T.edges, T.catalog_edges, T.model_decisions, T.catalog_models):
            connection.execute(sa.delete(table))
        connection.execute(sa.insert(T.model_decisions), md)
        connection.execute(sa.insert(T.edges), edges)
    start = sa.select(
        sa.literal("ModelDecision", sa.Text).label("label"), sa.literal("a", sa.Text).label("key")
    )
    for direction in ("out", "in", "both"):
        for no_backtrack in (False, True):
            for edge_table in (None, T.all_edges):
                options = {
                    "max_hops": 3,
                    "direction": direction,
                    "no_backtrack": no_backtrack,
                    "track_path": True,
                    "track_types": True,
                    "edge_table": edge_table,
                }
                _compare(start, options, nav)
    # The loop reaches the start at depth 1, once.
    rows = _rows(
        sa.select(walk(start, max_hops=1, direction="both", types=["R"], no_backtrack=True))
    )
    loops = [row for row in rows if row[1] == "a" and row[2] == 1]
    assert len(loops) == 1 and rows[loops[0]] == 1


def _selector(rng: random.Random, uids: list[str], models: list[str]) -> NodeSelector:
    roll = rng.random()
    if roll < 0.6:
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


def _former_walk(start, *, dialect=None, **options):
    return old_walk(start, **options)


@pytest.mark.parametrize("seed", range(8))
async def test_neighbors_and_subgraph_match_the_former_walk(nav, seed, monkeypatch) -> None:
    rng = random.Random(100 + seed)
    uids, models = _random_graph(rng)
    calls = []
    for _ in range(25):
        selector = _selector(rng, uids, models)
        scope = rng.choice(_SCOPES)
        types = rng.choice((None, None, ["R"], ["R", "S", "HAS_FIELD"]))
        calls.append(
            (
                "neighbors",
                selector,
                {
                    "edge_types": types,
                    "direction": rng.choice(("out", "in", "both")),
                    "target_types": rng.choice((None, None, ["ModelDecision"], ["CatalogModel"])),
                    "depth": rng.choice((1, 2, 3, 4)),
                    "limit": rng.choice((None, None, 3)),
                    **scope,
                },
            )
        )
        calls.append(
            (
                "subgraph",
                selector,
                {
                    "edge_types": types,
                    "depth": rng.choice((1, 2, 3)),
                    "max_nodes": rng.choice((500, 500, 3, 1)),
                    **scope,
                },
            )
        )
    got = [(await getattr(nav, m)(s, **kw)) for m, s, kw in calls]
    monkeypatch.setattr(_power, "walk", _former_walk)
    want = [(await getattr(nav, m)(s, **kw)) for m, s, kw in calls]
    for (method, selector, kwargs), g, w in zip(calls, got, want, strict=True):
        dump = (lambda r: [n.model_dump() for n in r]) if method == "neighbors" else None
        assert (dump(g) if dump else g.model_dump()) == (dump(w) if dump else w.model_dump()), (
            method,
            selector,
            kwargs,
        )
    assert any(r if isinstance(r, list) else r.nodes for r in got)  # not only empty results


@pytest.mark.parametrize(
    "options",
    [
        {"types": ("HAS_STRUCTURE", "HAS_CHILD"), "track_path": True},
        {"direction": "in", "prefix": "HAS_", "track_types": True},
        {"track_path": True, "track_types": True, "only_labels": ("ModelInstance",)},
        {"direction": "in", "no_backtrack": True, "path_labels": True, "track_path": True},
    ],
)
def test_one_branch_steps_compile_as_before(options) -> None:
    # A hop along one table in one direction is the former statement, character for character.
    start = sa.select(
        sa.literal("Document", sa.Text).label("label"), sa.literal("k", sa.Text).label("key")
    )
    for engine in (sqlite.dialect(), postgresql.dialect()):
        for dialect in (SqliteDialect(), PostgresDialect()):
            new = sa.select(walk(start, max_hops=3, dialect=dialect, **options))
            old = sa.select(old_walk(start, max_hops=3, **options))
            assert str(new.compile(dialect=engine)) == str(old.compile(dialect=engine))
