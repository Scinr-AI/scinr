"""
navigation/sql/_walk.py — Variable-length walks over the relationship tables (recursive CTEs).

The navigation twin of ``graph/sql/_traversal.py``: it mirrors the Cypher patterns
``(a)-[:T1|T2*0..N]->(b)`` the Neo4j navigator uses, but it also carries the *origin* of every
walk (the start node it came from) and, on request, the *path* (the keys of the nodes visited,
joined with :data:`PATH_SEP`) because several methods return lineages.

Every row of the CTE is ``(label, key, depth, origin[, path])``. The start nodes are in the
CTE at ``depth = 0``; consumers filter ``depth >= 1`` for the ``*1..N`` patterns and keep the
zero for ``*0..N``. ``UNION`` (not ``UNION ALL``) plus the ``depth < max_hops`` bound make a
cycle end after ``max_hops`` steps. A node reached at several depths or from several origins
appears once per combination: use :func:`reached` to select the distinct ones.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.expression import CTE, CompoundSelect, Select

from scinr.newton.sql import tables
from scinr.newton.sql.dialect import Dialect
from scinr.newton.sql.engine import get_dialect_for_config

__all__ = ["LABEL_SEP", "PATH_SEP", "UNBOUNDED_HOPS", "reached", "split_path", "walk"]

Direction = Literal["out", "in", "both"]

#: Separator of the keys of :func:`walk` paths (a control character no key contains).
PATH_SEP = "\x1f"

#: Separator of label and key inside a path token when ``path_labels`` is on.
LABEL_SEP = "\x1e"

#: Stand-in for the ``*1..`` of Cypher (a path this long does not occur in the graph).
UNBOUNDED_HOPS = 64

_ZERO = sa.literal_column("0")
_ONE = sa.literal_column("1")


def _ends(alias: Any, direction: Direction) -> tuple[Any, Any, Any, Any]:
    """``(near_label, near_key, far_label, far_key)`` of an edge walked in *direction*."""
    if direction == "out":
        return alias.c.src_label, alias.c.src_key, alias.c.dst_label, alias.c.dst_key
    return alias.c.dst_label, alias.c.dst_key, alias.c.src_label, alias.c.src_key


def _type_filter(
    alias: Any, types: Collection[str] | None, prefix: str | None
) -> list[ColumnElement[bool]]:
    out: list[ColumnElement[bool]] = []
    if types is not None:
        out.append(alias.c.type.in_(sorted(types)))
    if prefix is not None:
        out.append(sa.func.substr(alias.c.type, 1, len(prefix)) == prefix)
    return out


def _edge_identity(step: Any) -> Any:
    """Text identifying an edge by its type and both ends."""
    sep = sa.literal(LABEL_SEP, sa.Text)
    return (
        step.c.type
        + sep
        + step.c.src_label
        + sep
        + step.c.src_key
        + sep
        + step.c.dst_label
        + sep
        + step.c.dst_key
    )


def split_path(path: str) -> list[str]:
    """The node keys of a ``path`` column value, start node first."""
    return path.split(PATH_SEP)


def _branches(
    direction: Direction, edge_table: sa.Table | None
) -> list[tuple[sa.Table, Direction]]:
    """The ``(table, direction)`` pairs a hop may follow: one per edge table and direction.

    The ``all_edges`` view is replaced by its two tables so that every branch can search the
    table's own index on its near end.
    """
    if edge_table is None:
        sources: tuple[sa.Table, ...] = (tables.edges,)
    elif edge_table is tables.all_edges:
        sources = (tables.edges, tables.catalog_edges)
    else:
        sources = (edge_table,)
    ways: tuple[Direction, ...] = ("out", "in") if direction == "both" else (direction,)
    return [(table, way) for table in sources for way in ways]


def _oriented_edges(branches: list[tuple[sa.Table, Direction]], name: str) -> sa.Subquery:
    """The edges of every branch as one ``UNION ALL`` with ``near_*`` / ``far_*`` columns."""
    arms = []
    for table, way in branches:
        near_label, near_key, far_label, far_key = _ends(table, way)
        arms.append(
            sa.select(
                near_label.label("near_label"),
                near_key.label("near_key"),
                far_label.label("far_label"),
                far_key.label("far_key"),
                table.c.type,
                table.c.src_label,
                table.c.src_key,
                table.c.dst_label,
                table.c.dst_key,
            )
        )
    return sa.union_all(*arms).subquery(name)


def walk(
    start: Select | CompoundSelect,
    *,
    max_hops: int,
    direction: Direction = "out",
    types: Collection[str] | None = None,
    prefix: str | None = None,
    track_path: bool = False,
    track_types: bool = False,
    path_labels: bool = False,
    no_backtrack: bool = False,
    only_labels: Collection[str] | None = None,
    guard: Callable[[CTE], ColumnElement[bool]] | None = None,
    name: str = "walk",
    edge_table: sa.Table | None = None,
    dialect: Dialect | None = None,
) -> CTE:
    """The nodes reachable from the start nodes in 0..max_hops hops.

    A hop follows one edge table in one direction per branch, each joined on the indexed
    columns of its near end (``(src_label, src_key, ...)`` or ``edges_dst``): ``"both"``
    and the ``all_edges`` view give several branches. SQLite gets one recursive ``SELECT`` per
    branch; Postgres one ``SELECT`` over a ``UNION ALL`` of the branches
    (:attr:`Dialect.recursive_branches`). An edge that is a loop on the node is found by
    both directions with the same row, which ``UNION`` keeps once, as the former single join
    on ``src = node OR dst = node`` did.

    Args:
        start: A ``SELECT`` whose first two columns are the label and the key of the start
            nodes.
        max_hops: Upper bound of the path length (``>= 1``).
        direction: ``"out"`` follows edges from source to destination, ``"in"`` backwards,
            ``"both"`` either way.
        types: Relationship types allowed on every hop (``None``: any).
        prefix: Allowed relationship types must also start with this text (``HAS_``).
        track_path: Add a ``path`` column with the keys from the start node to this one.
        track_types: Add ``first_type`` / ``last_type`` columns: the type of the hop next to
            the start node and of the last hop (``NULL`` at depth 0).
        path_labels: Path tokens are ``label + LABEL_SEP + key`` instead of the key alone.
        no_backtrack: A hop may not be the edge the previous hop used (a Cypher path never
            repeats a relationship, so ``(a)-[*2]-(b)`` cannot go along an edge and back).
        only_labels: Every node reached must have one of these primary labels.
        guard: Extra condition on the node a hop leaves from (it receives the CTE); the hop
            from a start node (depth 0) is not guarded by the caller's condition if it
            returns ``True`` for ``depth = 0``.
        name: Name of the CTE (distinct names when one statement has several).
        edge_table: ``edges`` (default) or the ``all_edges`` view.
        dialect: The engine's dialect, which shapes a step of several branches (default:
            the dialect of the configured ``sql_url``).

    Raises:
        ValueError: If *max_hops* is lower than 1.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    origin = start.subquery(f"{name}_start")
    origin_label, origin_key = list(origin.c)[:2]

    columns: list[Any] = [
        origin_label.label("label"),
        origin_key.label("key"),
        _ZERO.label("depth"),
        origin_key.label("origin"),
    ]
    if track_path:
        first = (
            origin_label + sa.literal(LABEL_SEP, sa.Text) + origin_key
            if path_labels
            else origin_key
        )
        columns.append(sa.cast(first, sa.Text).label("path"))
    if track_types:
        columns.append(sa.cast(sa.null(), sa.Text).label("first_type"))
        columns.append(sa.cast(sa.null(), sa.Text).label("last_type"))
    if no_backtrack:
        columns.append(sa.cast(sa.null(), sa.Text).label("last_edge"))
    cte = sa.select(*columns).select_from(origin).cte(name, recursive=True)

    def hop(step: Any, near_label: Any, near_key: Any, far_label: Any, far_key: Any) -> Select:
        """One recursive ``SELECT``: the hop along *step* from its near end to its far end."""
        joined = step.join(cte, sa.and_(near_label == cte.c.label, near_key == cte.c.key))
        recursive_columns: list[Any] = [far_label, far_key, cte.c.depth + _ONE, cte.c.origin]
        if track_path:
            token = far_label + sa.literal(LABEL_SEP, sa.Text) + far_key if path_labels else far_key
            recursive_columns.append(cte.c.path + sa.literal(PATH_SEP, sa.Text) + token)
        if track_types:
            recursive_columns.append(sa.func.coalesce(cte.c.first_type, step.c.type))
            recursive_columns.append(step.c.type)
        conditions = [cte.c.depth < max_hops, *_type_filter(step, types, prefix)]
        if no_backtrack:
            identity = _edge_identity(step)
            recursive_columns.append(identity)
            conditions.append(sa.or_(cte.c.last_edge.is_(None), identity != cte.c.last_edge))
        if only_labels is not None:
            conditions.append(far_label.in_(sorted(only_labels)))
        if guard is not None:
            conditions.append(guard(cte))
        return sa.select(*recursive_columns).select_from(joined).where(*conditions)

    branches = _branches(direction, edge_table)
    if len(branches) == 1:
        table, way = branches[0]
        step = table.alias(f"{name}_e")
        return cte.union(hop(step, *_ends(step, way)))
    if (dialect if dialect is not None else get_dialect_for_config()).recursive_branches:
        hops = []
        for table, way in branches:
            step = table.alias(f"{name}_{table.name}_{way}")
            hops.append(hop(step, *_ends(step, way)))
        return cte.union(*hops)
    oriented = _oriented_edges(branches, f"{name}_e")
    c = oriented.c
    return cte.union(hop(oriented, c.near_label, c.near_key, c.far_label, c.far_key))


def reached(
    cte: CTE,
    label: str | None = None,
    *,
    min_depth: int = 1,
    name: str = "reached",
) -> sa.Subquery:
    """The distinct ``(key, origin)`` pairs of a walk at depth ``>= min_depth``.

    Args:
        cte: A :func:`walk` result.
        label: Keep only the nodes with this primary label (any if ``None``).
        min_depth: ``1`` for ``*1..N`` patterns, ``0`` to keep the start nodes.
        name: Name of the subquery.
    """
    conditions: list[ColumnElement[bool]] = [cte.c.depth >= min_depth]
    if label is not None:
        conditions.append(cte.c.label == label)
    return (
        sa.select(cte.c.key.label("key"), cte.c.origin.label("origin"))
        .where(*conditions)
        .distinct()
        .subquery(name)
    )
