"""
tests/sql/_shortest_oracle.py -- The former ``SqlNavigator.shortest_path``, kept as a test oracle.

It enumerates every walk (no immediate backtrack) up to ``max_hops`` in one recursive CTE and
keeps the first by ``ORDER BY depth, path``: exponential in ``max_hops``, but obviously right,
so the tests compare the level-synchronous search of ``navigation/sql/_bfs.py`` with it on
small graphs. The only change from the original is the ``ORDER BY path`` collation: it goes
through ``Dialect.order_text`` (code points on both engines; the original used the database
collation on Postgres, ``BINARY`` = code points on SQLite).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation import _map
from scinr.newton.navigation.models import NodeSelector, PathResult
from scinr.newton.navigation.scope import make_scope
from scinr.newton.navigation.sql._walk import LABEL_SEP, PATH_SEP, walk
from scinr.newton.sql import tables as T


async def old_shortest_path(
    nav: Any,
    from_selector: NodeSelector,
    to_selector: NodeSelector,
    *,
    max_hops: int = 6,
    edge_types: Sequence[str] | None = None,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> PathResult | None:
    """The shortest undirected path between two selected nodes, up to *max_hops* hops."""
    sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
    h = int(max_hops)
    if h < 1:
        raise NavigationError("max_hops must be >= 1")
    start = nav._selector_start(from_selector)
    goal_select = nav._selector_start(to_selector)
    starts = {(r["label"], r["key"]) for r in await nav._read(start)}
    goals = {(r["label"], r["key"]) for r in await nav._read(goal_select)}
    if len(starts) == 1 and starts == goals:
        raise NavigationError("shortest_path: from_selector and to_selector name the same node")
    goal = goal_select.subquery("goal")
    index = nav._node_index()
    cte = walk(
        start,
        max_hops=h,
        direction="both",
        no_backtrack=True,
        types=list(edge_types) if edge_types else None,
        track_path=True,
        path_labels=True,
        guard=nav._tenant_guard(sc, index),
        edge_table=T.all_edges,
        name="shortest",
    )
    conditions: list[ColumnElement[bool]] = [cte.c.depth >= 1, cte.c.key != cte.c.origin]
    anchors = [nav._anchor_allowed(start, sc)] if nav._allowed(index, sc) is not None else []
    end_allowed = nav._allowed(index, sc)
    statement = (
        sa.select(cte.c.path, cte.c.depth)
        .select_from(
            cte.join(goal, sa.and_(goal.c.label == cte.c.label, goal.c.key == cte.c.key)).join(
                index, sa.and_(index.c.label == cte.c.label, index.c.key == cte.c.key)
            )
        )
        .where(*conditions, *anchors, *([end_allowed] if end_allowed is not None else []))
        .order_by(cte.c.depth, nav._dl.order_text(cte.c.path))
        .limit(1)
    )
    rec = await nav._read_one(statement)
    if not rec:
        return None
    tokens = [tuple(t.split(LABEL_SEP, 1)) for t in rec["path"].split(PATH_SEP)]
    pairs = [(label, key) for label, key in tokens]
    nodes = await nav._fetch_nodes(pairs)
    relationships = await nav._path_relationships(pairs, edge_types)
    return PathResult(
        raw={"length": int(rec["depth"])},
        length=int(rec["depth"]),
        nodes=[_map.graph_node(nodes[p]) for p in pairs if p in nodes],
        relationships=relationships,
    )
