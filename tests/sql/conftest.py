"""
tests/sql/conftest.py -- Fixtures shared by every suite that runs against the SQL backends.

Import them from other suites' conftest files::

    from tests.sql.conftest import pg_schema, pg_url, sql_backend, sqlite_url  # noqa: F401

Engines
-------
* SQLite: a file in ``tmp_path``. Always available.
* Postgres: a disposable container, never a shared database::

      docker run -d --name postgres-test -p 55432:5432 \
        -e POSTGRES_USER=newton_admin -e POSTGRES_PASSWORD=newton_admin \
        -e POSTGRES_DB=newton_test postgres:16

  Variables: ``SCINR_SQL_PG_URL`` (default
  ``postgresql://newton_admin:newton_admin@127.0.0.1:55432/newton_test``) and
  ``SCINR_SQL_ALLOW_WIPE=1``. Without the second one, or when the container does not
  answer, every Postgres test is skipped with the reason. The fixtures only ever drop
  schemas named ``newton_test_*`` in a local database whose name contains ``test``.

See ``tests/sql/README.md``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlparse

import pytest

DEFAULT_PG_URL = "postgresql://newton_admin:newton_admin@127.0.0.1:55432/newton_test"
WIPE_SCHEMA_PREFIX = "newton_test_"
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class SqlTarget:
    """A configured SQL engine for a test."""

    backend: str  # "sqlite" | "postgres"
    url: str
    schema: str


def _pg_base_url() -> str:
    return os.environ.get("SCINR_SQL_PG_URL", DEFAULT_PG_URL)


def _skip_unless_pg_wipeable() -> str:
    if os.environ.get("SCINR_SQL_ALLOW_WIPE") != "1":
        pytest.skip(
            "Postgres tests need SCINR_SQL_ALLOW_WIPE=1 (disposable postgres-test container)"
        )
    url = _pg_base_url()
    parsed = urlparse(url)
    if parsed.hostname not in _LOCAL_HOSTS or "test" not in (parsed.path or "").lower():
        pytest.skip(f"refusing to wipe a non-disposable database: {parsed.hostname}{parsed.path}")
    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg is not installed (pip install scinr[postgres])")
    try:
        with psycopg.connect(url, connect_timeout=2):
            pass
    except psycopg.Error as exc:
        pytest.skip(
            f"Postgres test container not reachable at {parsed.hostname}:{parsed.port}: {exc}"
        )
    return url


def drop_pg_schema(url: str, schema: str) -> None:
    """Drop one ``newton_test_*`` schema (nothing else is ever dropped)."""
    import psycopg
    from psycopg import sql

    assert schema.startswith(WIPE_SCHEMA_PREFIX), f"refusing to drop schema {schema!r}"
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
        )


@pytest.fixture
def sqlite_url(tmp_path) -> str:
    """URL of an empty SQLite database in ``tmp_path``."""
    return f"sqlite:///{tmp_path}/newton.db"


@pytest.fixture
def pg_schema() -> str:
    """Schema name owned by this test process."""
    return f"{WIPE_SCHEMA_PREFIX}{os.getpid()}"


@pytest.fixture
def pg_url(pg_schema) -> Iterator[str]:
    """URL of the Postgres test container, with this process's test schema dropped."""
    url = _skip_unless_pg_wipeable()
    drop_pg_schema(url, pg_schema)
    yield url
    drop_pg_schema(url, pg_schema)


@pytest.fixture(params=["sqlite", "postgres"])
def sql_backend(request, tmp_path) -> Iterator[SqlTarget]:
    """Configure the library for each SQL engine in turn; reset the engines afterwards."""
    from scinr.newton import configure
    from scinr.newton.sql.engine import reset_engines

    backend = request.param
    if backend == "postgres":
        url = request.getfixturevalue("pg_url")
        schema = request.getfixturevalue("pg_schema")
    else:
        url = f"sqlite:///{tmp_path}/newton.db"
        schema = "newton"
    configure(
        graph_backend=backend,
        storage_backend=backend,
        freeze_backend=backend,
        sql_url=url,
        sql_schema=schema,
    )
    try:
        yield SqlTarget(backend=backend, url=url, schema=schema)
    finally:
        reset_engines()
