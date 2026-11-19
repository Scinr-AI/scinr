"""
navigation/sql/_bfs.py — Shortest undirected path by a bidirectional, level-synchronous search.

``shortest_path`` used to enumerate every walk of up to ``max_hops`` hops in one recursive CTE
and keep the first by ``ORDER BY depth, path``: the number of walks grows exponentially with
the hops. This module finds the same path with a breadth-first search driven from Python, one
batched query per level and side (the caller supplies the queries, see :func:`shortest_path`),
expanding from the start nodes and from the goal nodes alternately (the side with the smaller
frontier first) and keeping a visited set per side.

Semantics kept from the CTE (the proofs are in the comments of :func:`shortest_path`):

* The length is that of the shortest walk without an immediate backtrack, i.e. of the shortest
  *simple* path: a shortest walk never repeats a node.
* A node may be walked *through* (be an inner node of the path) only if it is *passable* (the
  tenant guard). The start node is always left, the goal node is only entered.
* Among the shortest paths, the one whose path string (the ``label + LABEL_SEP + key`` tokens
  joined with ``PATH_SEP``, start first) is the smallest by code point wins. Keeping only the
  smallest prefix per node is exact because two prefixes that end at the same node have the
  same number of tokens and end with the same token, so neither is a proper prefix of the
  other (keys never contain ``PATH_SEP``: the path string could not be split otherwise).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence

from scinr.newton.navigation.sql._walk import LABEL_SEP, PATH_SEP

__all__ = ["FRONTIER_BATCH", "Node", "path_token", "shortest_path"]

#: A node as ``(primary label, key)``.
Node = tuple[str, str]

#: Nodes per frontier query. A query has two ``IN`` lists of at most this size (one per edge
#: direction), far below the bound-parameter limits (999 on old SQLite builds, 65,535 Postgres).
FRONTIER_BATCH = 400

#: ``neighbours(batch)`` -> ``(near, far)`` for every relationship with an end in *batch*
#: (``near`` is that end, ``far`` the other), in both directions, with the type filter applied.
Neighbours = Callable[[Sequence[Node]], Awaitable[Iterable[tuple[Node, Node]]]]

#: ``passable(batch)`` -> the nodes of *batch* that may be walked through (``None``: all).
Passable = Callable[[Sequence[Node]], Awaitable[Collection[Node]]]


def path_token(node: Node) -> str:
    """The token of *node* in a path string (as the recursive CTE of ``walk`` builds it)."""
    return node[0] + LABEL_SEP + node[1]


def _batches(nodes: Sequence[Node]) -> Iterable[Sequence[Node]]:
    for i in range(0, len(nodes), FRONTIER_BATCH):
        yield nodes[i : i + FRONTIER_BATCH]


class _Side:
    """One direction of the search: the nodes it reached, by distance from its roots."""

    def __init__(self, roots: Collection[Node]) -> None:
        self.dist: dict[Node, int] = dict.fromkeys(roots, 0)
        #: node -> the expanded nodes one level closer to the roots it was reached from.
        self.links: dict[Node, set[Node]] = {}
        #: The nodes of the outer level that may be expanded (roots are always expanded).
        self.frontier: list[Node] = sorted(roots)
        self.radius = 0


async def _expand(
    side: _Side, other: _Side, neighbours: Neighbours, passable: Passable | None
) -> list[tuple[Node, Node]]:
    """Expand the frontier of *side* by one level; the meetings with *other*.

    A meeting is ``(near, far)``: ``near`` an expanded node of *side*, ``far`` a node *other*
    reached. It is kept only if ``far`` is a root of *other* or passable (otherwise ``far``
    would be an inner node that may not be walked through).
    """
    level = side.radius + 1
    fresh: list[Node] = []
    meetings: list[tuple[Node, Node]] = []
    for batch in _batches(side.frontier):
        for near, far in await neighbours(batch):
            seen = side.dist.get(far)
            if seen is None:
                side.dist[far] = seen = level
                fresh.append(far)
            if seen != level:
                continue  # reached earlier (or a self loop / an edge inside the level)
            side.links.setdefault(far, set()).add(near)
            if far in other.dist:
                meetings.append((near, far))
    if passable is None:
        allowed: Collection[Node] = set(fresh)
    else:
        allowed = set()
        for batch in _batches(fresh):
            allowed.update(await passable(batch))
    side.radius = level
    side.frontier = sorted(n for n in fresh if n in allowed)
    return [(near, far) for near, far in meetings if other.dist[far] == 0 or far in allowed]


def _prefixes(side: _Side, targets: Collection[Node]) -> dict[Node, str]:
    """The smallest path string from a root of *side* to each target (all at one distance)."""
    levels = [set(targets)]
    while side.dist[next(iter(levels[-1]))] > 0:
        levels.append({u for n in levels[-1] for u in side.links[n]})
    best = {n: path_token(n) for n in levels[-1]}
    for level in reversed(levels[:-1]):
        best = {
            n: min(best[u] + PATH_SEP + path_token(n) for u in side.links[n] if u in best)
            for n in level
        }
    return best


def _best_path(
    starts: _Side, goals: _Side, meetings: Iterable[tuple[Node, Node]]
) -> tuple[int, str]:
    """``(length, path string)`` of the smallest shortest path through the meeting edges.

    *meetings* are oriented start side first: ``(x, y)`` with ``x`` reached from the starts and
    ``y`` from the goals.
    """
    pairs = list(meetings)
    length = min(starts.dist[x] + 1 + goals.dist[y] for x, y in pairs)
    pairs = [(x, y) for x, y in pairs if starts.dist[x] + 1 + goals.dist[y] == length]
    prefix = _prefixes(starts, {x for x, _ in pairs})
    current: dict[Node, str] = {}
    for x, y in pairs:
        candidate = prefix[x] + PATH_SEP + path_token(y)
        if y not in current or candidate < current[y]:
            current[y] = candidate
    for _ in range(goals.dist[pairs[0][1]]):
        following: dict[Node, str] = {}
        for node, text in current.items():
            for nxt in goals.links[node]:
                candidate = text + PATH_SEP + path_token(nxt)
                if nxt not in following or candidate < following[nxt]:
                    following[nxt] = candidate
        current = following
    return length, min(current.values())


async def shortest_path(
    starts: Collection[Node],
    goals: Collection[Node],
    *,
    max_hops: int,
    neighbours: Neighbours,
    passable: Passable | None,
) -> tuple[int, str] | None:
    """``(length, path string)`` of the smallest shortest path from a start to a goal.

    Args:
        starts: The nodes a path may start from (walked out of whether passable or not).
        goals: The nodes a path may end at (entered whether passable or not). Disjoint from
            *starts*: a path has at least one hop.
        max_hops: Longest path considered (``>= 1``).
        neighbours: The query of one level (see :data:`Neighbours`).
        passable: The tenant guard of the inner nodes (``None``: no guard).

    Returns:
        ``None`` if no goal is within *max_hops* hops.
    """
    # Invariant: when neither side has met the other with radii (a, b), no valid path of
    # length <= a + b exists. Expanding one side by a level then finds every shortest path
    # of length a + b + 1 as a meeting edge between the two frontiers, and nothing shorter
    # (it would contradict the invariant): the first level with a meeting gives the length.
    # A side whose frontier is empty means that no node at its outer distance may be walked
    # through, so no longer path exists either.
    assert not set(starts) & set(goals), "shortest_path: starts and goals overlap"
    if not starts or not goals:
        return None
    forward, backward = _Side(starts), _Side(goals)
    while forward.radius + backward.radius < max_hops:
        if not forward.frontier or not backward.frontier:
            return None
        if len(forward.frontier) <= len(backward.frontier):
            met = await _expand(forward, backward, neighbours, passable)
            if met:
                return _best_path(forward, backward, met)
        else:
            met = await _expand(backward, forward, neighbours, passable)
            if met:
                return _best_path(forward, backward, [(x, y) for y, x in met])
    return None
