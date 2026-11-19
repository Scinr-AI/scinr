"""
tests/sql/_dump.py -- Neutral dump of a SQL backend, comparable with the Neo4j one.

Reads the database with an engine of its own (never the store's) and returns the structure
of ``tests/golden/_dump.py``: nodes keyed by business key with their labels and properties,
and relationships between those keys.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import sqlalchemy as sa

from scinr.newton.sql import tables
from scinr.newton.sql.dialect import get_dialect
from scinr.newton.sql.keys import decode_key
from scinr.newton.sql.rows import row_to_node
from tests.golden._dump import build_dump, node_key


@contextmanager
def reader(url: str, schema: str | None) -> Iterator[sa.Connection]:
    """A connection of a throwaway engine that reads *schema* (Postgres) or the file (SQLite)."""
    dialect = get_dialect(url)
    engine = sa.create_engine(
        dialect.sync_url(url),
        execution_options={
            "schema_translate_map": dialect.schema_translate_map(schema or "newton")
        },
    )
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


def read_nodes(connection: sa.Connection) -> Iterator[tuple[str, tuple[str, ...], dict[str, Any]]]:
    """``(table, labels, properties)`` of every node row of every node table."""
    for name in tables.NODE_TABLES:
        for row in connection.execute(sa.select(tables.metadata.tables[name])).mappings():
            labels, props = row_to_node(name, row)
            yield name, labels, props


def dump_sql(url: str, schema: str | None) -> dict[str, Any]:
    """The neutral dump of a SQL backend, read with its own engine (not the store's)."""
    with reader(url, schema) as connection:
        nodes = [(labels, props) for _, labels, props in read_nodes(connection)]
        rels = []
        for row in connection.execute(sa.select(tables.all_edges)).mappings():
            ends = []
            for label, key in (
                (row["src_label"], row["src_key"]),
                (row["dst_label"], row["dst_key"]),
            ):
                ends.append(node_key([label], decode_key(label, key)))
            rels.append((row["type"], ends[0], ends[1], dict(row["props"] or {})))
    return build_dump(nodes, rels)
