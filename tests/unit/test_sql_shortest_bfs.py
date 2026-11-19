"""
tests/unit/test_sql_shortest_bfs.py -- The bidirectional shortest-path search of
navigation/sql/_bfs.py, against a brute-force enumeration of the walks (the former CTE).

The graphs live in memory: ``neighbours`` / ``passable`` are plain functions over an edge list,
so the search logic is tested without a database. ``tests/sql/test_shortest_path_sql.py``
compares the whole ``shortest_path`` (selectors, scope, the SQL of each level) with the former
recursive CTE on both engines.
"""

from __future__ import annotations

import itertools
import random

import pytest

pytest.importorskip("sqlalchemy")

from scinr.newton.navigation.sql import _bfs as bfs  # noqa: E402
from scinr.newton.navigation.sql._walk import PATH_SEP  # noqa: E402

Node = tuple[str, str]
Edge = tuple[str, Node, Node]  # (type, src, dst)


class Graph:
    """An edge list with the two callbacks of ``bfs.shortest_path``."""

    def __init__(self, edges: list[Edge], impassable: set[Node] | None = None) -> None:
        self.edges = edges
        self.impassable = impassable
        self.queries = 0
        self.batch_sizes: list[int] = []

    async def neighbours(self, batch):
        self.queries += 1
        self.batch_sizes.append(len(batch))
        wanted = set(batch)
        out = []
        for _, src, dst in self.edges:
            if src in wanted:
                out.append((src, dst))
            if dst in wanted:
                out.append((dst, src))
        return out

    async def passable(self, batch):
        return {n for n in batch if n not in self.impassable}

    async def search(self, starts, goals, max_hops):
        return await bfs.shortest_path(
            set(starts),
            set(goals),
            max_hops=max_hops,
            neighbours=self.neighbours,
            passable=self.passable if self.impassable is not None else None,
        )


def brute_force(graph: Graph, starts, goals, max_hops):
    """The former CTE, literally: every walk without an immediate backtrack, guard on the node
    a hop leaves (the start exempt), smallest ``(depth, path)`` ending at a goal."""
    best = None
    impassable = graph.impassable or set()
    rows = [(s, 0, s, bfs.path_token(s), None) for s in starts]
    while rows:
        following = []
        for node, depth, origin, path, last in rows:
            if depth >= max_hops or (depth > 0 and node in impassable):
                continue
            for edge in graph.edges:
                _, src, dst = edge
                for near, far in ((src, dst), (dst, src)):
                    if near != node or edge == last:
                        continue
                    row = (far, depth + 1, origin, path + PATH_SEP + bfs.path_token(far), edge)
                    following.append(row)
                    if far in goals and far[1] != origin[1]:
                        candidate = (depth + 1, row[3])
                        if best is None or candidate < best:
                            best = candidate
                    if src == dst:
                        break  # a self loop is one edge, walked once
        rows = following
    return best


def _random_graph(rng: random.Random) -> tuple[Graph, list[Node]]:
    keys = ["x", "xy", "x0", "a", "b", "c", "ab", "z"]
    nodes = [(rng.choice("AB"), k) for k in rng.sample(keys, rng.randint(4, len(keys)))]
    edges: list[Edge] = []
    for _ in range(rng.randint(len(nodes) - 1, 2 * len(nodes) + 2)):
        a, b = rng.choice(nodes), rng.choice(nodes)
        edges.append((rng.choice(("R", "S")), a, b))  # self loops and parallel edges included
    impassable = {n for n in nodes if rng.random() < 0.25} if rng.random() < 0.6 else None
    return Graph(edges, impassable), nodes


@pytest.mark.parametrize("seed", range(300))
async def test_matches_the_enumeration_of_every_walk(seed) -> None:
    rng = random.Random(seed)
    graph, nodes = _random_graph(rng)
    starts = rng.sample(nodes, rng.randint(1, 2))
    rest = [n for n in nodes if n[1] not in {s[1] for s in starts}]
    goals = rng.sample(rest, min(len(rest), rng.randint(1, 3)))
    max_hops = rng.randint(1, 5)
    assert await graph.search(starts, goals, max_hops) == brute_force(
        graph, starts, goals, max_hops
    )


def _grid(width: int, height: int) -> list[Edge]:
    """A grid: the number of shortest paths between opposite corners is binomial."""
    edges = []
    for x, y in itertools.product(range(width), range(height)):
        if x + 1 < width:
            edges.append(("R", ("N", f"{x}-{y}"), ("N", f"{x + 1}-{y}")))
        if y + 1 < height:
            edges.append(("R", ("N", f"{x}-{y}"), ("N", f"{x}-{y + 1}")))
    return edges


async def test_many_tied_shortest_paths_cost_one_query_per_level() -> None:
    graph = Graph(_grid(4, 4))
    found = await graph.search([("N", "0-0")], [("N", "3-3")], 6)
    assert found is not None
    length, path = found
    assert length == 6  # 20 shortest paths tie
    # the smallest path string: "0-0" "0-1" ... (the ``y`` steps first)
    assert path.split(PATH_SEP) == [
        bfs.path_token(("N", k)) for k in ("0-0", "0-1", "0-2", "0-3", "1-3", "2-3", "3-3")
    ]
    assert graph.queries <= 6  # one query per level (the frontiers fit in one batch)
    assert found == brute_force(graph, [("N", "0-0")], [("N", "3-3")], 6)


async def test_the_tie_break_compares_whole_path_strings() -> None:
    # Both shortest paths go s - ? - g; "x" + PATH_SEP sorts before "xy" (PATH_SEP is \x1f).
    s, g = ("A", "s"), ("A", "g")
    edges = [
        ("R", s, ("A", "xy")),
        ("R", ("A", "xy"), g),
        ("R", s, ("A", "x")),
        ("R", ("A", "x"), g),
    ]
    graph = Graph(edges)
    found = await graph.search([s], [g], 3)
    assert found == (2, PATH_SEP.join(bfs.path_token(n) for n in (s, ("A", "x"), g)))
    assert found == brute_force(graph, [s], [g], 3)


async def test_the_depth_limit_and_no_path() -> None:
    chain = [("R", ("N", str(i)), ("N", str(i + 1))) for i in range(5)]
    graph = Graph(chain + [("R", ("N", "x"), ("N", "y"))])
    assert await graph.search([("N", "0")], [("N", "5")], 4) is None
    assert (await graph.search([("N", "0")], [("N", "5")], 5))[0] == 5
    assert await graph.search([("N", "0")], [("N", "y")], 10) is None  # another component
    assert await graph.search([], [("N", "y")], 10) is None
    assert await graph.search([("N", "0")], [], 10) is None


async def test_inner_nodes_must_be_passable_but_the_ends_need_not() -> None:
    s, m, g, detour = ("A", "s"), ("A", "m"), ("C", "g"), ("A", "d")
    edges = [
        ("R", s, m),
        ("R", m, g),
        ("R", s, detour),
        ("R", detour, ("A", "e")),
        ("R", ("A", "e"), g),
    ]
    blocked = Graph(edges, impassable={m, s, g})
    found = await blocked.search([s], [g], 6)
    assert found is not None and found[0] == 3  # around m; s and g are the ends
    assert await Graph(edges, impassable={m, detour}).search([s], [g], 6) is None


async def test_several_starts_and_goals_take_the_closest_pair() -> None:
    chain = [("R", ("N", str(i)), ("N", str(i + 1))) for i in range(8)]
    graph = Graph(chain)
    found = await graph.search([("N", "0"), ("N", "3")], [("N", "8"), ("N", "5")], 6)
    assert found == (2, PATH_SEP.join(bfs.path_token(("N", k)) for k in ("3", "4", "5")))


async def test_frontiers_are_batched() -> None:
    start, goal = ("H", "start"), ("H", "goal")
    count = bfs.FRONTIER_BATCH * 2 + 5
    near = [("L", f"a{i:04d}") for i in range(count)]
    far = [("M", f"b{i:04d}") for i in range(count)]
    edges = [("R", start, n) for n in near] + [("R", f, goal) for f in far]
    edges += [("R", n, f) for n, f in zip(near, far, strict=True)]
    graph = Graph(edges)
    found = await graph.search([start], [goal], 4)
    assert found == (3, PATH_SEP.join(bfs.path_token(n) for n in (start, near[0], far[0], goal)))
    assert max(graph.batch_sizes) == bfs.FRONTIER_BATCH
    assert graph.queries == 2 + 3  # each root once, then one frontier in three batches
