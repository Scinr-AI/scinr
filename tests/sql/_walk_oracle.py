"""
tests/sql/_walk_oracle.py -- The former ``navigation/sql/_walk.walk``, kept as a test oracle.

Its recursive step joined the CTE with the relationship table (or the ``all_edges`` view) on
an ``OR`` of both directions: correct, but SQLite scanned every relationship for each queued
row. ``tests/sql/test_walk_sql.py`` compares the rows of the current ``walk`` (one indexed
branch per direction and table) with these on random graphs. Copied verbatim.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.expression import CTE, CompoundSelect, Select

from scinr.newton.navigation.sql._walk import LABEL_SEP, PATH_SEP
from scinr.newton.sql import tables

Direction = Literal["out", "in", "both"]

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


def old_walk(
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
) -> CTE:
    """The nodes reachable from the start nodes in 0..max_hops hops.

    Args:
        start: A ``SELECT`` whose first two columns are the label and the key of the start
            nodes.
        max_hops: Upper bound of the path length (``>= 1``).
        direction: ``"out"`` follows edges from source to destination, ``"in"`` backwards.
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

    Raises:
        ValueError: If *max_hops* is lower than 1.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    relationships = edge_table if edge_table is not None else tables.edges
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

    step = relationships.alias(f"{name}_e")
    if direction == "both":
        near_is_src = sa.and_(step.c.src_label == cte.c.label, step.c.src_key == cte.c.key)
        near_is_dst = sa.and_(step.c.dst_label == cte.c.label, step.c.dst_key == cte.c.key)
        joined = step.join(cte, sa.or_(near_is_src, near_is_dst))
        far_label = sa.case((near_is_src, step.c.dst_label), else_=step.c.src_label)
        far_key = sa.case((near_is_src, step.c.dst_key), else_=step.c.src_key)
    else:
        near_label, near_key, far_label, far_key = _ends(step, direction)
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
    recursive = sa.select(*recursive_columns).select_from(joined).where(*conditions)
    return cte.union(recursive)
