"""
tests/unit/test_sql_traversal.py -- The bounded recursive CTEs of graph/sql/_traversal.py (SQLite).
"""

from __future__ import annotations

import pytest

sa = pytest.importorskip("sqlalchemy")

from scinr.newton.graph.sql._traversal import reachable, reaches_from  # noqa: E402
from scinr.newton.sql import tables  # noqa: E402


@pytest.fixture
def connection():
    engine = sa.create_engine("sqlite://")
    meta = sa.MetaData()
    edges = tables.edges.to_metadata(meta)
    nodes = sa.Table("nodes", meta, sa.Column("key", sa.Text, primary_key=True))
    meta.create_all(engine)
    with engine.begin() as conn:
        yield conn, edges, nodes
    engine.dispose()


def _edge(src: str, dst: str, rel_type: str = "NEXT", label: str = "N") -> dict:
    return {
        "tenant_id": "t",
        "type": rel_type,
        "src_label": label,
        "src_key": src,
        "dst_label": label,
        "dst_key": dst,
        "_ident": "",
        "props": {},
    }


def _start(*keys: str):
    return sa.union_all(
        *(sa.select(sa.literal("N").label("label"), sa.literal(k).label("key")) for k in keys)
    )


def _reached(conn, cte) -> set[str]:
    return {row[0] for row in conn.execute(sa.select(cte.c.key).distinct())}


@pytest.fixture
def chain(connection):
    conn, edges, _ = connection
    # n0 -> n1 -> ... -> n10: ten hops.
    conn.execute(edges.insert(), [_edge(f"n{i}", f"n{i + 1}") for i in range(10)])
    return conn


@pytest.mark.parametrize("hops", [1, 7, 10])
def test_chain_reaches_exactly_max_hops(chain, hops) -> None:
    reached = _reached(chain, reachable(_start("n0"), max_hops=hops))
    assert reached == {f"n{i}" for i in range(1, hops + 1)}


def test_chain_beyond_its_length_stops_at_the_end(chain) -> None:
    assert len(_reached(chain, reachable(_start("n0"), max_hops=50))) == 10


def test_backwards(chain) -> None:
    cte = reachable(_start("n10"), max_hops=3, direction="in")
    assert _reached(chain, cte) == {"n9", "n8", "n7"}


def test_depth_column(chain) -> None:
    cte = reachable(_start("n0"), max_hops=4)
    rows = chain.execute(sa.select(cte.c.key, cte.c.depth)).all()
    assert sorted(rows) == [("n1", 1), ("n2", 2), ("n3", 3), ("n4", 4)]


def test_a_cycle_terminates(connection) -> None:
    conn, edges, _ = connection
    conn.execute(edges.insert(), [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")])
    cte = reachable(_start("a"), max_hops=100)
    # The start node comes back through the cycle; each depth at most once per node.
    assert _reached(conn, cte) == {"a", "b", "c"}
    assert conn.execute(sa.select(sa.func.count()).select_from(cte)).scalar_one() == 100


def test_type_filter(connection) -> None:
    conn, edges, _ = connection
    conn.execute(
        edges.insert(),
        [_edge("a", "b", "KEEP"), _edge("b", "c", "OTHER"), _edge("a", "d", "KEEP")],
    )
    cte = reachable(_start("a"), max_hops=5, rel_types={"KEEP"})
    assert _reached(conn, cte) == {"b", "d"}


def test_several_start_nodes(chain) -> None:
    cte = reachable(_start("n0", "n5"), max_hops=1)
    assert _reached(chain, cte) == {"n1", "n6"}


def test_max_hops_must_be_positive() -> None:
    with pytest.raises(ValueError):
        reachable(_start("a"), max_hops=0)
    with pytest.raises(ValueError):
        reaches_from(sa.literal("N"), sa.literal("a"), source_label="S", max_hops=0)


def test_reaches_from_is_correlated_and_bounded(connection) -> None:
    conn, edges, nodes = connection
    conn.execute(nodes.insert(), [{"key": k} for k in ("a", "b", "c", "z")])
    rows = [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")]
    root = _edge("er", "a")
    root["src_label"] = "Source"
    conn.execute(edges.insert(), [root, *rows])

    def orphans(hops: int) -> list[str]:
        exists = reaches_from(sa.literal("N"), nodes.c.key, source_label="Source", max_hops=hops)
        query = sa.select(nodes.c.key).where(~exists).order_by(nodes.c.key)
        return [row[0] for row in conn.execute(query)]

    assert orphans(1) == ["b", "c", "z"]
    assert orphans(2) == ["c", "z"]
    assert orphans(7) == ["z"]
