"""
graph/sql/_common.py — Engines, tables and statement builders shared by the SQL graph store.

Every SQL module of the graph store builds its statements with these helpers, so the
translation recipes of the plan (``MERGE`` of a node or a relationship, ``SET x = null``,
``DETACH DELETE``) are written once:

* :func:`merge_node` — ``MERGE (n:L {key}) ON CREATE SET … ON MATCH SET …`` as an
  ``INSERT … ON CONFLICT``. A ``None`` value removes the property (invariant I5).
* :func:`merge_edge` — ``MERGE (a)-[:R {identity}]->(b) [SET r.x = …]``. The table
  (``edges`` or ``catalog_edges``) follows from the labels of the ends; ``_ident`` from
  :func:`~scinr.newton.sql.names.edge_ident`.
* :func:`detach_delete` — ``DETACH DELETE`` of nodes of one label: their relationships,
  then the rows.

Execution: synchronous methods use :func:`sync_write` / :func:`sync_read`, asynchronous
ones the async engine (:func:`async_write_engine` / :func:`async_read_engine`). Reads go
through ``sqlite_deferred`` (a plain ``BEGIN`` in SQLite, so they do not queue behind the
single writer).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import ColumnElement

from scinr.newton.graph.types import DELETE_BATCH_ROWS
from scinr.newton.ingest.freeze import validate_identifier
from scinr.newton.sql import tables as _tables
from scinr.newton.sql.dialect import SQLITE_DEFERRED_OPTION, Dialect
from scinr.newton.sql.engine import get_async_engine, get_dialect_for_config, get_sync_engine
from scinr.newton.sql.keys import encode_key
from scinr.newton.sql.names import CATALOG_LABELS, KEY_COLUMN, LABEL_TABLE, edge_ident
from scinr.newton.sql.retry import with_sql_retry_sync
from scinr.newton.sql.rows import props_to_row, row_to_node
from scinr.newton.sql.tables import LIST_COLUMNS, TYPED_COLUMNS

__all__ = [
    "EDGE_IDENTITY_COLUMNS",
    "create_node",
    "merge_edges_many",
    "upsert_many",
    "IN_BATCH",
    "async_read_engine",
    "async_write_engine",
    "batched",
    "detach_delete",
    "dialect",
    "document_key",
    "edge_table",
    "edges",
    "key_column",
    "merge_edge",
    "merge_node",
    "node_from_row",
    "patch_props",
    "set_columns",
    "sync_read",
    "sync_write",
    "table",
]

T = TypeVar("T")

#: Keys per ``IN (...)`` list (the batch size of the deletes; far below the bind limits).
IN_BATCH = DELETE_BATCH_ROWS

#: The unique key of ``edges`` / ``catalog_edges`` (``ON CONFLICT`` target).
EDGE_IDENTITY_COLUMNS = ("src_label", "src_key", "type", "dst_label", "dst_key", "_ident")

edges = _tables.edges
catalog_edges = _tables.catalog_edges


# ── Engines ──────────────────────────────────────────────────────────────────


def dialect() -> Dialect:
    """The dialect of the configured ``sql_url``."""
    return get_dialect_for_config()


def sync_write(fn: Callable[[Connection], T], *, retry: bool = True) -> T:
    """Run *fn* inside one write transaction of the sync engine (retried whole if *retry*)."""

    def attempt() -> T:
        with get_sync_engine().begin() as connection:
            return fn(connection)

    return with_sql_retry_sync(attempt) if retry else attempt()


def sync_read(fn: Callable[[Connection], T], *, retry: bool = True) -> T:
    """Run *fn* on a read connection of the sync engine (retried whole if *retry*)."""

    def attempt() -> T:
        engine: Engine = get_sync_engine().execution_options(**{SQLITE_DEFERRED_OPTION: True})
        with engine.connect() as connection:
            return fn(connection)

    return with_sql_retry_sync(attempt) if retry else attempt()


def async_write_engine() -> AsyncEngine:
    """The async engine for write transactions (SQLite: ``BEGIN IMMEDIATE``)."""
    return get_async_engine()


def async_read_engine() -> AsyncEngine:
    """The async engine for reads (SQLite: a plain ``BEGIN``)."""
    return get_async_engine().execution_options(**{SQLITE_DEFERRED_OPTION: True})


# ── Tables and keys ──────────────────────────────────────────────────────────


def table(label: str) -> sa.Table:
    """The node table of a primary label."""
    return _tables.metadata.tables[LABEL_TABLE[label]]


def key_column(label: str) -> sa.Column:
    """The column that holds the ``_key`` of the nodes of *label*."""
    return table(label).c[KEY_COLUMN[label]]


def document_key(tenant: str, path: str, version: int) -> str:
    """The ``_key`` of a :Document (also the ``_document_key`` of what hangs from it)."""
    return encode_key("Document", {"tenant_id": tenant, "path": path, "version": version})


def node_from_row(label: str, row: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """``(labels, properties)`` of a row of the table of *label*."""
    return row_to_node(LABEL_TABLE[label], row)


def batched(items: Sequence[T], size: int) -> Iterable[list[T]]:
    """Consecutive slices of *items* of at most *size* elements."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# ── Properties ───────────────────────────────────────────────────────────────


def patch_props(column: Any, values: Mapping[str, Any]) -> Any:
    """``props`` with the keys of *values* set, or removed when the value is ``None``."""
    d = dialect()
    to_set = {k: v for k, v in values.items() if v is not None}
    to_remove = [k for k, v in values.items() if v is None]
    expression = column
    if to_set:
        expression = d.json_set_keys(expression, to_set)
    if to_remove:
        expression = d.json_remove_keys(expression, to_remove)
    return expression


def set_columns(target: sa.Table, values: Mapping[str, Any]) -> dict[str, Any]:
    """The ``SET`` of an ``UPDATE`` / ``ON CONFLICT DO UPDATE`` for ``SET n.k = v, …``.

    Typed columns take the value (``NULL`` for ``None``); the other properties are patched
    into ``props``. A value may be a SQL expression (it then goes to a typed column only).
    Later keys win, as consecutive ``SET`` items of Cypher do.
    """
    typed = TYPED_COLUMNS[target.name]
    result: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for name, value in values.items():
        if name in typed:
            result[name] = value
        else:
            extra[name] = value
    if extra:
        result["props"] = patch_props(target.c.props, extra)
    return result


def _insert_row(target: sa.Table, props: Mapping[str, Any], aux: Mapping[str, Any] | None):
    row = props_to_row(target.name, props)
    lists = LIST_COLUMNS.get(target.name, frozenset())
    for name in lists:
        if name in row and not isinstance(row[name], list):
            row[name] = list(row[name])
    row.update(aux or {})
    return row


# ── Nodes ────────────────────────────────────────────────────────────────────


def merge_node(
    label: str,
    key: Mapping[str, Any],
    *,
    on_create: Mapping[str, Any] | None = None,
    on_match: Mapping[str, Any] | None = None,
    aux: Mapping[str, Any] | None = None,
    aux_on_match: Mapping[str, Any] | None = None,
) -> sa.Insert:
    """``MERGE (n:<label> {key}) ON CREATE SET on_create ON MATCH SET on_match``.

    For a plain ``SET`` (both branches) pass the same mapping in both. *aux* are auxiliary
    columns (``_document_key``…) written on create, *aux_on_match* on match.
    """
    target = table(label)
    props = {**key, **(on_create or {})}
    if label in ("Document", "ModelField"):
        aux = {"_key": encode_key(label, key), **(aux or {})}
    statement = dialect().insert(target).values(**_insert_row(target, props, aux))
    update = set_columns(target, on_match or {})
    update.update(aux_on_match or {})
    if not update:
        return statement.on_conflict_do_nothing(index_elements=[KEY_COLUMN[label]])
    return statement.on_conflict_do_update(index_elements=[KEY_COLUMN[label]], set_=update)


def create_node(
    label: str, props: Mapping[str, Any], aux: Mapping[str, Any] | None = None
) -> sa.Insert:
    """``CREATE (n:<label> {props})``: a plain insert (a duplicate key violates the primary
    key, as it violates the uniqueness constraint in Neo4j)."""
    target = table(label)
    return sa.insert(target).values(**_insert_row(target, props, aux))


def upsert_many(label: str, update: Sequence[str]) -> sa.Insert:
    """``UNWIND $rows MERGE (n:<label> {key}) SET n.<col> = row.<col>, …`` for executemany.

    The statement takes a list of complete rows (the same keys in each, typed columns plus
    ``props`` and auxiliary columns); an existing row gets the *update* columns from the new
    one (typed or auxiliary columns only; ``props`` is left as it is).
    """
    statement = dialect().insert(table(label))
    return statement.on_conflict_do_update(
        index_elements=[KEY_COLUMN[label]],
        set_={name: statement.excluded[name] for name in update},
    )


def merge_edges_many(src_label: str, dst_label: str) -> sa.Insert:
    """``MERGE (a)-[r]->(b)`` for executemany over rows built by :func:`edge_row`."""
    return (
        dialect()
        .insert(edge_table(src_label, dst_label))
        .on_conflict_do_nothing(index_elements=list(EDGE_IDENTITY_COLUMNS))
    )


def detach_delete(label: str, keys: Sequence[str]) -> list[sa.Delete]:
    """``DETACH DELETE`` of the nodes of *label* with the given keys (two statements)."""
    keys = list(keys)
    if not keys:
        return []
    return [
        sa.delete(edges).where(
            sa.or_(
                sa.and_(edges.c.src_label == label, edges.c.src_key.in_(keys)),
                sa.and_(edges.c.dst_label == label, edges.c.dst_key.in_(keys)),
            )
        ),
        sa.delete(table(label)).where(key_column(label).in_(keys)),
    ]


# ── Relationships ────────────────────────────────────────────────────────────


def edge_table(src_label: str, dst_label: str) -> sa.Table:
    """``catalog_edges`` between two catalog nodes, ``edges`` otherwise."""
    if src_label in CATALOG_LABELS and dst_label in CATALOG_LABELS:
        return catalog_edges
    return edges


def edge_row(
    rel_type: str,
    src_label: str,
    src_key: str,
    dst_label: str,
    dst_key: str,
    *,
    tenant: str | None,
    props: Mapping[str, Any] | None = None,
) -> tuple[sa.Table, dict[str, Any]]:
    """The table and the row of one relationship (``None`` properties dropped).

    Raises:
        ValueError: If *rel_type* is not a valid relationship type, or a relationship with a
            data end has no tenant (``edges.tenant_id`` is the tenant of its data end).
    """
    validate_identifier(rel_type)
    clean = {k: v for k, v in (props or {}).items() if v is not None}
    target = edge_table(src_label, dst_label)
    row: dict[str, Any] = {
        "type": rel_type,
        "src_label": src_label,
        "src_key": src_key,
        "dst_label": dst_label,
        "dst_key": dst_key,
        "_ident": edge_ident(rel_type, clean),
        "props": clean,
    }
    if target is edges:
        if tenant is None:
            raise ValueError(f"A {rel_type} relationship with a data end needs a tenant.")
        row["tenant_id"] = tenant
    return target, row


def merge_edge(
    rel_type: str,
    src_label: str,
    src_key: str,
    dst_label: str,
    dst_key: str,
    *,
    tenant: str | None,
    props: Mapping[str, Any] | None = None,
    on_match: Mapping[str, Any] | None = None,
) -> sa.Insert:
    """``MERGE (a)-[r:<rel_type> {identity}]->(b) [SET r.k = v]``.

    *props* are written when the relationship is created (its identity properties and any
    ``ON CREATE SET`` / ``SET``); *on_match* are patched into an existing one (``SET``).
    """
    target, row = edge_row(
        rel_type, src_label, src_key, dst_label, dst_key, tenant=tenant, props=props
    )
    statement = dialect().insert(target).values(**row)
    if on_match:
        return statement.on_conflict_do_update(
            index_elements=list(EDGE_IDENTITY_COLUMNS),
            set_={"props": patch_props(target.c.props, on_match)},
        )
    return statement.on_conflict_do_nothing(index_elements=list(EDGE_IDENTITY_COLUMNS))


Step = tuple[str | Sequence[str] | None, str | None]
"""One hop of :func:`follow`: relationship type(s) (any if ``None``) and the label of the node
reached (any if ``None``)."""


def follow(start: Any, steps: Sequence[Step]) -> sa.Select:
    """``(label, key)`` of the nodes at the end of a fixed-length path, each once.

    ``follow(start, [("HAS_MODEL_DECISION", "ModelDecision"), ("HAS_PROPOSED_MODEL", None)])``
    is the Cypher ``(s)-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(n)``
    from every ``(label, key)`` row of *start* (a ``SELECT`` or a subquery whose first two
    columns are label and key). Only ``edges`` is walked: every path here leaves a data node.
    """
    origin = start.subquery() if isinstance(start, (sa.Select, sa.CompoundSelect)) else start
    label, key = list(origin.c)[:2]
    source: Any = origin
    for index, (rel_types, dst_label) in enumerate(steps):
        hop = edges.alias(f"hop{index}")
        condition = sa.and_(hop.c.src_label == label, hop.c.src_key == key)
        if rel_types is not None:
            types = [rel_types] if isinstance(rel_types, str) else list(rel_types)
            condition = sa.and_(condition, hop.c.type.in_(types))
        if dst_label is not None:
            condition = sa.and_(condition, hop.c.dst_label == dst_label)
        source = source.join(hop, condition)
        label, key = hop.c.dst_label, hop.c.dst_key
    return sa.select(label.label("label"), key.label("key")).select_from(source).distinct()


def nodes_with_keys(label: str, keys: Sequence[str]) -> sa.Select:
    """``(label, key)`` rows of the given nodes (the *start* of :func:`follow`)."""
    column = key_column(label)
    return sa.select(sa.literal(label, sa.Text).label("label"), column.label("key")).where(
        column.in_(list(keys))
    )


def group_by_label(rows: Iterable[Any]) -> dict[str, list[str]]:
    """``{label: [keys]}`` of ``(label, key)`` rows, in order of first appearance."""
    grouped: dict[str, dict[str, None]] = {}
    for row in rows:
        grouped.setdefault(row[0], {})[row[1]] = None
    return {label: list(keys) for label, keys in grouped.items()}


def column_values(rows: Iterable[Any]) -> list[Any]:
    """The first column of every row."""
    return [row[0] for row in rows]


def json_text(column: Any, key: str) -> ColumnElement:
    """The text value of a top-level key of a ``props`` column (``NULL`` when absent)."""
    return dialect().json_text(column, key)
