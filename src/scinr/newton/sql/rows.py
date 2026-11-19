"""
sql/rows.py — From a table row to a graph node and back.

:func:`row_to_node` is the only way a row becomes a node: navigation, the freeze export and
the test dump all use it. Auxiliary columns (names that start with ``_``) are never part of
a node (invariant I4) and a ``NULL`` column or an absent ``props`` key means "the property
does not exist" (invariant I5).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from scinr.newton.ingest.freeze import _ordered_labels
from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.sql.names import TABLE_LABEL
from scinr.newton.sql.tables import LIST_COLUMNS, TYPED_COLUMNS

__all__ = ["props_to_row", "row_to_node"]

_ROLE_COLUMN = "role"


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value)


def row_to_node(table: str, row: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """``(labels, properties)`` of a node row, exactly as Neo4j would return them.

    Args:
        table: Name of the node table the row comes from.
        row: The row, including its ``props`` column when it has one.

    Returns:
        The labels (primary label first, then the role label of a ``StructureNode``) and
        the properties: typed columns that are not ``NULL`` plus every key of ``props``.
        List properties are ``list`` objects.
    """
    primary = TABLE_LABEL[table]
    lists = LIST_COLUMNS.get(table, frozenset())
    properties: dict[str, Any] = {}
    for name in TYPED_COLUMNS[table]:
        value = row.get(name)
        if value is None:
            continue
        if name in lists:
            value = list(json.loads(value) if isinstance(value, str) else value)
        properties[name] = value
    properties.update(_as_dict(row.get("props")))

    labels = [primary]
    if table == "structure_nodes":
        role_label = ROLE_TO_LABEL.get(properties.get(_ROLE_COLUMN, ""))
        if role_label:
            labels.append(role_label)
    return tuple(_ordered_labels(labels, primary)), properties


def props_to_row(table: str, props: Mapping[str, Any]) -> dict[str, Any]:
    """Split node properties into typed columns and the ``props`` JSON.

    ``None`` values are dropped (a null property does not exist). The result has one entry
    per typed column that has a value, plus ``"props"`` (always present, maybe empty).

    Raises:
        TypeError: If a value that goes into ``props`` is not JSON-serialisable.
    """
    typed = TYPED_COLUMNS[table]
    row: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for name, value in props.items():
        if value is None:
            continue
        if name in typed:
            row[name] = value
        else:
            extra[name] = value
    json.dumps(extra)  # TypeError for anything that is not JSON
    row["props"] = extra
    return row
