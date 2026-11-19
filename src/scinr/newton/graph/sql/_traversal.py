"""
graph/sql/_traversal.py — Bounded variable-length paths over ``edges`` (recursive CTEs).

The only place that writes a recursive CTE. :func:`reachable` mirrors the Cypher pattern
``(a)-[*1..N]->(b)`` from a set of start nodes, :func:`reaches_from` the predicate
``EXISTS { MATCH (:Label)-[*1..N]->(x) }`` of the orphan collection.

Both walk ``edges`` only, never ``catalog_edges``: by invariant I1 no relationship leads
from a catalog node to a data node, so no path that leaves a data node can come back to
data through the catalog. A path may *end* on a catalog node (``MATCHED_MODEL``,
``USES_PRIMARY_MODEL``...), and those edges are in ``edges`` (they have a data end).

Both use ``UNION`` (not ``UNION ALL``) with a ``depth`` column and ``depth < max_hops`` in the
recursive step, so a cycle ends after ``max_hops`` steps. A node reached at several depths
appears once per depth: select ``DISTINCT`` on ``(label, key)``.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.expression import CTE, CompoundSelect, Select

from scinr.newton.sql.tables import edges

__all__ = ["reachable", "reaches_from"]

Direction = Literal["out", "in"]

_ONE = sa.literal_column("1")


def _ends(alias: Any, direction: Direction) -> tuple[Any, Any, Any, Any]:
    """``(near_label, near_key, far_label, far_key)`` of an edge walked in *direction*."""
    if direction == "out":
        return alias.c.src_label, alias.c.src_key, alias.c.dst_label, alias.c.dst_key
    return alias.c.dst_label, alias.c.dst_key, alias.c.src_label, alias.c.src_key


def _type_filter(alias: Any, rel_types: Collection[str] | None) -> list[ColumnElement[bool]]:
    return [] if rel_types is None else [alias.c.type.in_(sorted(rel_types))]


def reachable(
    start: Select | CompoundSelect,
    *,
    max_hops: int,
    direction: Direction = "out",
    rel_types: Collection[str] | None = None,
    name: str = "reachable",
) -> CTE:
    """Nodes reachable from the (label, key) rows of *start* in 1..max_hops hops over edges.

    Columns: label, key, depth. Mirrors (a)-[*1..max_hops]->(b) of Cypher (any type unless
    rel_types is given). Duplicates by depth are possible; use DISTINCT on (label, key).

    Args:
        start: A ``SELECT`` whose first two columns are the label and the key of the start
            nodes (the start nodes themselves are only in the result if a path leads back).
        max_hops: Upper bound of the path length (``>= 1``).
        direction: ``"out"`` follows edges from source to destination, ``"in"`` backwards.
        rel_types: Relationship types allowed on every hop; ``None`` allows any.
        name: Name of the CTE (distinct names when one statement uses several).

    Raises:
        ValueError: If *max_hops* is lower than 1.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    origin = start.subquery(f"{name}_start")
    origin_label, origin_key = list(origin.c)[:2]

    first = edges.alias(f"{name}_e0")
    near_label, near_key, far_label, far_key = _ends(first, direction)
    anchor = sa.select(
        far_label.label("label"), far_key.label("key"), _ONE.label("depth")
    ).select_from(first.join(origin, sa.and_(near_label == origin_label, near_key == origin_key)))
    for condition in _type_filter(first, rel_types):
        anchor = anchor.where(condition)
    cte = anchor.cte(name, recursive=True)

    step = edges.alias(f"{name}_e1")
    near_label, near_key, far_label, far_key = _ends(step, direction)
    recursive = (
        sa.select(far_label, far_key, cte.c.depth + _ONE)
        .select_from(step.join(cte, sa.and_(near_label == cte.c.label, near_key == cte.c.key)))
        .where(cte.c.depth < max_hops, *_type_filter(step, rel_types))
    )
    return cte.union(recursive)


def reaches_from(
    target_label_col: Any,
    target_key_col: Any,
    *,
    source_label: str,
    max_hops: int,
    name: str = "reaches",
) -> sa.Exists:
    """EXISTS: some node with *source_label* reaches the target in 1..max_hops hops
    (the NOT EXISTS { MATCH (:ExtractionResult)-[*1..7]->(mi) } of the GC), walking backwards.

    The CTE is correlated with the enclosing query through *target_label_col* /
    *target_key_col* (a column or a literal), and rendered inside the ``EXISTS``.

    Raises:
        ValueError: If *max_hops* is lower than 1.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    first = edges.alias(f"{name}_e0")
    anchor = (
        sa.select(
            first.c.src_label.label("label"), first.c.src_key.label("key"), _ONE.label("depth")
        )
        .where(first.c.dst_label == target_label_col, first.c.dst_key == target_key_col)
        .correlate_except(first)
    )
    cte = anchor.cte(name, recursive=True, nesting=True)
    step = edges.alias(f"{name}_e1")
    recursive = (
        sa.select(step.c.src_label, step.c.src_key, cte.c.depth + _ONE)
        .select_from(
            step.join(cte, sa.and_(step.c.dst_label == cte.c.label, step.c.dst_key == cte.c.key))
        )
        .where(cte.c.depth < max_hops)
    )
    cte = cte.union(recursive)
    return sa.exists(sa.select(_ONE).select_from(cte).where(cte.c.label == source_label))
