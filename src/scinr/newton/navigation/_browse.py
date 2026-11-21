"""
navigation/_browse.py — Engine-neutral pieces of ``get_node`` / ``count_node_relationships`` /
``expand_node`` / ``search_nodes``.

Both backends validate and order with these helpers so a page of neighbours or of search
results is the same on Neo4j and SQL: the order is decided here, in Python, by code point.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.graph.identity import PRIMARY_LABELS, decode_key, encode_key, node_identity
from scinr.newton.navigation import _map
from scinr.newton.navigation._safe import safe_ident
from scinr.newton.navigation._search_fields import SPECIAL_SEARCH_LABELS
from scinr.newton.navigation.models import GraphEdge, GraphNode, NodeMatch, RelationshipCount

__all__ = [
    "MIN_SEARCH_LENGTH",
    "Hit",
    "check_edge_types",
    "check_labels",
    "check_page",
    "count_relationships",
    "encode_ref",
    "ident_fields",
    "ident_key",
    "node_ref",
    "primary_label",
    "search_labels",
    "sort_edges",
    "special_hits",
]

#: Shortest ``text`` ``search_nodes`` accepts; a shorter one finds nothing.
MIN_SEARCH_LENGTH = 2

_KEY_TYPES = (str, int)


def node_ref(label: str, key: str) -> tuple[str, dict[str, Any]] | None:
    """``(primary label, key fields)`` of ``(label, key)``.

    Raises:
        NavigationError: *label* is not an identifier or not a primary label of the graph.

    Returns ``None`` when *key* is not a well-formed key of that label (a lookup by it finds
    nothing).
    """
    safe_ident(label, kind="node label")
    if label not in PRIMARY_LABELS:
        raise NavigationError(f"Unknown node label: {label!r}. Known: {sorted(PRIMARY_LABELS)}.")
    if not isinstance(key, str) or not key:
        return None
    try:
        fields = decode_key(label, key)
    except ValueError:
        return None
    for value in fields.values():
        if isinstance(value, bool) or not isinstance(value, _KEY_TYPES):
            return None
    return label, fields


def encode_ref(ref: tuple[str, Mapping[str, Any]]) -> str:
    """The normalised key text of a :func:`node_ref` result."""
    label, fields = ref
    return encode_key(label, fields)


def ident_fields() -> list[str]:
    """Every property that is part of some label's key, sorted (for an identity projection)."""
    return sorted({f for label in PRIMARY_LABELS for f in node_identity([label])[1]})


def ident_key(label: str, ident: Mapping[str, Any]) -> str | None:
    """The encoded key of a node from its key properties; ``None`` if one is missing."""
    fields = node_identity([label])[1]
    if any(ident.get(f) is None for f in fields):
        return None
    return encode_key(label, ident)


def primary_label(labels: Iterable[str]) -> str | None:
    """The primary label of a node's labels, ``None`` when it has no known identity."""
    try:
        return node_identity(list(labels))[0]
    except Exception:  # noqa: BLE001 — FreezeError: not a node the identity knows
        return None


def check_page(skip: int, limit: int | None, *, allow_unlimited: bool = False) -> None:
    """Validate ``skip`` (``>= 0``) and ``limit`` (``>= 1``, or ``None`` when allowed)."""
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise NavigationError(f"skip must be an integer >= 0, got {skip!r}.")
    if limit is None and allow_unlimited:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise NavigationError(
            f"limit must be an integer >= 1{' or None' if allow_unlimited else ''}, got {limit!r}."
        )


def check_edge_types(edge_types: Sequence[str] | None) -> list[str] | None:
    """Validated relationship types (``None`` or empty: any)."""
    if not edge_types:
        return None
    if isinstance(edge_types, (str, bytes)):
        raise NavigationError("edge_types must be a list of strings.")
    return [safe_ident(t, kind="edge type") for t in edge_types]


def check_labels(labels: Sequence[str] | None, *, kind: str = "neighbor label") -> list[str] | None:
    """Validated labels (``None`` or empty: any). A label may also be a role label."""
    if not labels:
        return None
    if isinstance(labels, (str, bytes)):
        raise NavigationError(f"{kind}s must be a list of strings.")
    return [safe_ident(label, kind=kind) for label in labels]


def search_labels(labels: Sequence[str] | None) -> list[str]:
    """The primary labels ``search_nodes`` looks into (all by default), without repeats."""
    wanted = check_labels(labels, kind="label")
    if wanted is None:
        return list(PRIMARY_LABELS)
    unknown = sorted(set(wanted) - set(PRIMARY_LABELS))
    if unknown:
        raise NavigationError(f"Unknown node label(s): {unknown}. Known: {sorted(PRIMARY_LABELS)}.")
    return [label for label in PRIMARY_LABELS if label in wanted]


def count_relationships(rows: Iterable[tuple[str, str, str, int]]) -> list[RelationshipCount]:
    """Merge ``(type, direction, neighbour label, count)`` rows and sort them.

    Order: ``count`` descending, ``type``, ``direction``, ``neighbor_label``.
    """
    merged: dict[tuple[str, str, str], int] = {}
    for rel_type, direction, neighbour, count in rows:
        merged[(rel_type, direction, neighbour)] = (
            merged.get((rel_type, direction, neighbour), 0) + count
        )
    out = [
        RelationshipCount(type=t, direction=d, neighbor_label=n, count=c)  # type: ignore[arg-type]
        for (t, d, n), c in merged.items()
    ]
    out.sort(key=lambda r: (-r.count, r.type, r.direction, r.neighbor_label))
    return out


def sort_edges(edges: Iterable[GraphEdge]) -> list[GraphEdge]:
    """Edges in a stable order (several relationships may join the same pair)."""
    return sorted(
        edges,
        key=lambda e: (
            e.source_label,
            e.source_key,
            e.target_label,
            e.target_key,
            e.type,
            json.dumps(e.properties, sort_keys=True, default=str),
        ),
    )


@dataclass(frozen=True)
class Hit:
    """One candidate of ``search_nodes`` before paging."""

    label: str
    key: str
    field: str
    node: GraphNode | None = None  # already built (info units, records); else fetched later

    @property
    def order(self) -> tuple[int, str, str]:
        """Key matches first, then by label and key (code point order)."""
        return (0 if self.field == "key" else 1, self.label, self.key)

    def match(self, node: GraphNode) -> NodeMatch:
        return NodeMatch(node=node, field=self.field)


async def special_hits(
    nav: Any, label: str, text: str, window: int, scope: Mapping[str, Any]
) -> list[Hit]:
    """Candidates of the labels searched through another method (``SPECIAL_SEARCH_LABELS``).

    *window* is ``skip + limit``: the most candidates a page can need from one label.
    """
    if label not in SPECIAL_SEARCH_LABELS:
        raise NavigationError(f"{label!r} is not searched through another method.")
    hits: dict[str, Hit] = {}
    exact = await nav.get_node(label, text, **scope)
    if exact is not None and exact.key is not None:
        hits[exact.key] = Hit(label, exact.key, "key", exact)
    if label == "InfoUnit":
        query = getattr(nav, "_fulltext_text", lambda t: t)(text)  # Neo4j: Lucene syntax escaped
        for unit in await nav.search_info_units(query, limit=window, **scope):
            node = _map.graph_node({**unit.raw, "_labels": ["InfoUnit"]})
            if node.key is not None and node.key not in hits:
                hits[node.key] = Hit(label, node.key, "text", node)
    else:
        for found in await nav.search_model_instances(text, limit=window, **scope):
            node = _map.graph_node({**found.instance.raw, "_labels": ["ModelInstance"]})
            if node.key is not None and node.key not in hits:
                hits[node.key] = Hit(
                    label, node.key, "key" if found.field == "uid" else found.field, node
                )
    return list(hits.values())
