"""
storage/sql/_common.py — Helpers shared by the SQL storage and freeze repositories.

Every database access of the repositories goes through :func:`fetch_all` / :func:`execute`:
they retry transient errors (``with_sql_retry``) and translate whatever is left to the
``StorageError`` family (``async_sql_errors(kind="storage")``), in that order, like the
other SQL stores. Reads use :func:`read_engine`, so SQLite does not queue them behind its
single writer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from scinr.newton.sql.engine import get_async_engine, get_dialect_for_config
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.retry import with_sql_retry
from scinr.newton.sql.scope import scope_conditions
from scinr.newton.sql.tables import StorageTables, storage_tables
from scinr.newton.utils.scope import make_scope

__all__ = [
    "IN_BATCH",
    "execute",
    "fetch_all",
    "fetch_one",
    "read_engine",
    "scope_where",
    "tables",
]

#: Rows per ``IN (...)`` statement (SQLite and Postgres cap the number of parameters).
IN_BATCH = 500


def tables() -> StorageTables:
    """The storage tables of the configured engine."""
    return storage_tables(get_dialect_for_config().name)


def read_engine() -> AsyncEngine:
    """The async engine for reads (SQLite: a plain ``BEGIN`` instead of ``BEGIN IMMEDIATE``)."""
    return get_async_engine().execution_options(sqlite_deferred=True)


def scope_where(
    table: sa.Table,
    tenant_id: str | None,
    include_public: bool,
    created_by_user_id: str | Sequence[str] | None,
    job_id: str | Sequence[str] | None,
) -> list[sa.ColumnElement[bool]]:
    """Validate the four public filters and render them as conditions on *table*.

    Raises:
        ScopeError: *tenant_id* is empty, or a list filter is empty.
    """
    scope = make_scope(tenant_id, include_public, created_by_user_id, job_id)
    return scope_conditions(scope, table, dialect=get_dialect_for_config())


async def fetch_all(statement: Any) -> list[sa.RowMapping]:
    """Run a read *statement* and return its rows as mappings."""

    async def run() -> list[sa.RowMapping]:
        async with read_engine().connect() as connection:
            return list((await connection.execute(statement)).mappings().all())

    async with async_sql_errors("storage"):
        return await with_sql_retry(run)


async def fetch_one(statement: Any) -> sa.RowMapping | None:
    """Run a read *statement* and return its first row, or ``None``."""
    rows = await fetch_all(statement)
    return rows[0] if rows else None


async def execute(statement: Any) -> list[sa.RowMapping]:
    """Run a write *statement* in its own transaction; return its ``RETURNING`` rows (if any)."""

    async def run() -> list[sa.RowMapping]:
        async with get_async_engine().begin() as connection:
            result = await connection.execute(statement)
            return list(result.mappings().all()) if result.returns_rows else []

    async with async_sql_errors("storage"):
        return await with_sql_retry(run)
