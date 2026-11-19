"""
tests/sql/test_schema_matches_tables.py -- The 0001 migration and sql/tables.py agree.

Applies the packaged migration of each engine and compares the database, through
``sqlalchemy.inspect``, with the ``Table`` objects (columns, primary keys and the
uniqueness the plan relies on).
"""

from __future__ import annotations

import re

import pytest
import sqlalchemy as sa

from scinr.newton.sql import tables
from scinr.newton.sql.dialect import get_dialect
from scinr.newton.sql.migrations import runner
from scinr.newton.sql.schema import ensure_schema

#: Objects that exist in the database but are not query tables.
_IGNORED_TABLES = re.compile(r"^(newton_schema|info_units_fts(_.*)?)$")
#: Postgres generated columns: addressed by name by the dialect, not declared as columns.
_UNDECLARED_COLUMNS = re.compile(r"^_fts_")


@pytest.fixture
def inspected(sql_backend):
    ensure_schema()
    dialect = get_dialect(sql_backend.url)
    engine = sa.create_engine(dialect.sync_url(sql_backend.url))
    schema = sql_backend.schema if sql_backend.backend == "postgres" else None
    try:
        yield sa.inspect(engine), schema, sql_backend.backend
    finally:
        engine.dispose()


def _db_columns(inspector, schema, name) -> set[str]:
    return {
        c["name"]
        for c in inspector.get_columns(name, schema=schema)
        if not _UNDECLARED_COLUMNS.match(c["name"])
    }


def test_migration_is_the_latest_version(sql_backend) -> None:
    ensure_schema()
    dialect = get_dialect(sql_backend.url)
    # Postgres has one more migration than SQLite: the version of the RLS track.
    assert runner.latest_version(dialect.name) == {"postgres": 2, "sqlite": 1}[dialect.name]


def test_every_table_exists_with_the_same_columns_and_the_other_way_round(inspected) -> None:
    inspector, schema, backend = inspected
    declared = tables.tables_for(backend)
    present = set(inspector.get_table_names(schema=schema)) | set(
        inspector.get_view_names(schema=schema)
    )
    present = {name for name in present if not _IGNORED_TABLES.match(name)}
    assert present == set(declared)
    for name, table in declared.items():
        assert _db_columns(inspector, schema, name) == {c.name for c in table.columns}, name


def test_primary_keys_match(inspected) -> None:
    inspector, schema, backend = inspected
    for name, table in tables.tables_for(backend).items():
        if table.info.get("is_view"):
            continue
        in_db = inspector.get_pk_constraint(name, schema=schema)["constrained_columns"]
        assert set(in_db) == {c.name for c in table.primary_key.columns}, name


def test_no_chunk_tables_in_sqlite_and_chunk_tables_in_postgres(inspected) -> None:
    inspector, schema, backend = inspected
    names = set(inspector.get_table_names(schema=schema))
    chunks = {"raw_file_chunks", "page_image_chunks", "frozen_snapshot_chunks"}
    assert (chunks <= names) if backend == "postgres" else not (chunks & names)


def test_document_and_edge_uniqueness(inspected) -> None:
    inspector, schema, _ = inspected

    def unique_sets(name: str) -> list[tuple[str, ...]]:
        found = [
            tuple(u["column_names"]) for u in inspector.get_unique_constraints(name, schema=schema)
        ]
        found += [
            tuple(i["column_names"])
            for i in inspector.get_indexes(name, schema=schema)
            if i.get("unique")
        ]
        return found

    assert ("tenant_id", "path", "version") in unique_sets("documents")
    edge_identity = ("src_label", "src_key", "type", "dst_label", "dst_key", "_ident")
    assert edge_identity in unique_sets("edges")
    assert edge_identity in unique_sets("catalog_edges")
    assert ("raw_file_id", "page_index") in unique_sets("pages")
    assert ("tenant_id", "raw_file_id", "page_index", "image_index") in unique_sets("page_images")


def test_declared_unique_constraints_exist_in_the_database(inspected) -> None:
    inspector, schema, backend = inspected
    for name, table in tables.tables_for(backend).items():
        if table.info.get("is_view"):
            continue
        in_db = {
            frozenset(u["column_names"])
            for u in inspector.get_unique_constraints(name, schema=schema)
        } | {
            frozenset(i["column_names"])
            for i in inspector.get_indexes(name, schema=schema)
            if i.get("unique")
        }
        for constraint in table.constraints:
            if isinstance(constraint, sa.UniqueConstraint):
                assert frozenset(c.name for c in constraint.columns) in in_db, name


def test_typed_columns_exclude_props_and_auxiliary_columns() -> None:
    for name, columns in tables.TYPED_COLUMNS.items():
        assert "props" not in columns
        assert not any(c.startswith("_") for c in columns), name
    assert "_key" not in tables.TYPED_COLUMNS["documents"]
    assert {"name", "model"} == tables.TYPED_COLUMNS["model_fields"]
