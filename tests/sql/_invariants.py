"""
tests/sql/_invariants.py -- Invariants I1, I2, I3 and I5 (plus ``_document_key`` consistency) of the SQL data model.

Each function takes the URL (and the Postgres schema) of a database, reads it with an engine of
its own and raises ``AssertionError`` with a message that names the offenders. The contract
and equivalence suites of the SQL backends call :func:`check_all` at the end of every test.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import sqlalchemy as sa

from scinr.newton.sql import tables
from scinr.newton.sql.names import (
    CATALOG_LABELS,
    KEY_COLUMN,
    LABEL_TABLE,
    PER_DOCUMENT_TABLES,
    TABLE_LABEL,
)
from tests.sql._dump import reader

#: Relationship types that lead from a document down to its structure nodes.
_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")
_MAX_LISTED = 10


def _fail(what: str, offenders: list[Any]) -> None:
    shown = ", ".join(repr(o) for o in offenders[:_MAX_LISTED])
    more = f" (+{len(offenders) - _MAX_LISTED} more)" if len(offenders) > _MAX_LISTED else ""
    raise AssertionError(f"{what}: {shown}{more}")


def _edge_rows(connection: sa.Connection) -> list[Any]:
    return list(connection.execute(sa.select(tables.all_edges)).mappings())


def _node_keys(connection: sa.Connection, label: str) -> set[str]:
    table = tables.metadata.tables[LABEL_TABLE[label]]
    return {r[0] for r in connection.execute(sa.select(table.c[KEY_COLUMN[label]]))}


def no_dangling_edges(url: str, schema: str | None = None) -> None:
    """I2: both ends of every relationship exist in their table."""
    with reader(url, schema) as connection:
        wanted: dict[str, set[str]] = defaultdict(set)
        for row in _edge_rows(connection):
            wanted[row["src_label"]].add(row["src_key"])
            wanted[row["dst_label"]].add(row["dst_key"])
        dangling: list[tuple[str, str]] = []
        for label, keys in wanted.items():
            if label not in LABEL_TABLE:
                dangling.extend((label, k) for k in sorted(keys))
                continue
            dangling.extend((label, k) for k in sorted(keys - _node_keys(connection, label)))
    if dangling:
        _fail("dangling relationship ends (label, key)", dangling)


def no_catalog_to_data_edges(url: str, schema: str | None = None) -> None:
    """I1: no relationship goes from a catalog node to a data node."""
    with reader(url, schema) as connection:
        offenders = [
            (r["type"], r["src_label"], r["dst_label"])
            for r in _edge_rows(connection)
            if r["src_label"] in CATALOG_LABELS and r["dst_label"] not in CATALOG_LABELS
        ]
    if offenders:
        _fail("catalog -> data relationships (type, src, dst)", offenders)


def edge_tenants_consistent(url: str, schema: str | None = None) -> None:
    """I3: ``edges.tenant_id`` is the tenant of the data ends; two data ends share a tenant."""
    with reader(url, schema) as connection:
        tenant_of: dict[tuple[str, str], str | None] = {}
        for label, table_name in LABEL_TABLE.items():
            if label in CATALOG_LABELS:
                continue
            table = tables.metadata.tables[table_name]
            for key, tenant in connection.execute(
                sa.select(table.c[KEY_COLUMN[label]], table.c.tenant_id)
            ):
                tenant_of[(label, key)] = tenant
        offenders: list[Any] = []
        for row in connection.execute(sa.select(tables.edges)).mappings():
            ends = [
                tenant_of[end]
                for end in ((row["src_label"], row["src_key"]), (row["dst_label"], row["dst_key"]))
                if end in tenant_of
            ]
            if not ends or any(t != row["tenant_id"] for t in ends):
                offenders.append((row["type"], row["tenant_id"], ends))
    if offenders:
        _fail("relationships with inconsistent tenant (type, edge tenant, end tenants)", offenders)


def _null_paths(value: Any, path: str = "") -> list[str]:
    if value is None:
        return [path or "<root>"]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _null_paths(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _null_paths(v, f"{path}[{i}]")]
    return []


def no_null_props(url: str, schema: str | None = None) -> None:
    """I5: no ``null`` inside any ``props`` document (nodes and relationships)."""
    offenders: list[Any] = []
    with reader(url, schema) as connection:
        for name in (*tables.NODE_TABLES, "edges", "catalog_edges"):
            table = tables.metadata.tables[name]
            for (props,) in connection.execute(sa.select(table.c.props)):
                if not isinstance(props, dict):
                    offenders.append((name, "props is not an object", props))
                    continue
                offenders.extend((name, p) for p in _null_paths(props))
    if offenders:
        _fail("null values stored in props (table, path)", offenders)


def document_keys_consistent(url: str, schema: str | None = None) -> None:
    """Every ``_document_key`` names an existing document; structure reached from a document
    through ``HAS_STRUCTURE`` / ``HAS_CHILD`` carries that document's ``_document_key``."""
    offenders: list[Any] = []
    with reader(url, schema) as connection:
        documents = _node_keys(connection, "Document")
        for name in PER_DOCUMENT_TABLES:
            if name == "documents":
                continue
            table = tables.metadata.tables[name]
            for key, doc_key in connection.execute(
                sa.select(table.c[KEY_COLUMN[TABLE_LABEL[name]]], table.c._document_key)
            ):
                if doc_key not in documents:
                    offenders.append((name, key, doc_key))

        children: dict[str, list[str]] = defaultdict(list)
        for row in connection.execute(
            sa.select(tables.edges.c.src_key, tables.edges.c.dst_key)
            .where(tables.edges.c.type.in_(_STRUCTURE_EDGES))
            .where(tables.edges.c.src_label.in_(("Document", "StructureNode")))
            .where(tables.edges.c.dst_label == "StructureNode")
        ):
            children[row[0]].append(row[1])
        owner = dict(
            connection.execute(
                sa.select(tables.structure_nodes.c.id, tables.structure_nodes.c._document_key)
            ).all()
        )
        for doc_key in documents:
            seen: set[str] = set()
            stack = list(children.get(doc_key, []))
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                if owner.get(node) != doc_key:
                    offenders.append(("structure_nodes", node, owner.get(node), doc_key))
                stack.extend(children.get(node, []))
    if offenders:
        _fail("inconsistent _document_key", offenders)


def check_all(url: str, schema: str | None = None) -> None:
    """Run every invariant check."""
    no_dangling_edges(url, schema)
    no_catalog_to_data_edges(url, schema)
    edge_tenants_consistent(url, schema)
    no_null_props(url, schema)
    document_keys_consistent(url, schema)
