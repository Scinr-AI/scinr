"""
tests/equivalence/conftest.py -- Fixtures of the Neo4j <-> SQL equivalence suite.

The suite runs the same scenario on Neo4j + MongoDB (the disposable containers of the golden
harness) and on a SQL engine (graph, storage and freeze in SQL) and compares the neutral
dumps. It needs **both** at once:

* Neo4j on 7698 and MongoDB on 27028 with ``SCINR_GOLDEN_ALLOW_WIPE=1`` and
  ``PYTHONHASHSEED=0`` (see ``tests/integration/test_graph_store_golden.py``);
* the SQL engine: SQLite (a file in ``tmp_path``, always) or the disposable ``postgres-test``
  container with ``SCINR_SQL_ALLOW_WIPE=1`` (see ``tests/sql/conftest.py``).

When one is missing every test is skipped with the reason. Never run this suite together
with the golden or contract suites: they share the Neo4j / MongoDB databases.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.golden import _runner


@dataclass(frozen=True)
class SqlEngine:
    """A SQL engine ready for one scenario (empty database)."""

    name: str
    settings: dict[str, Any]
    dump: Callable[[], dict[str, Any]]


@pytest.fixture
def neo4j_available() -> None:
    """Skip unless the disposable Neo4j + MongoDB containers of the golden harness are usable."""
    try:
        _runner.guard()
        _runner.wipe_databases()
    except _runner.GoldenUnavailable as exc:
        pytest.skip(f"Neo4j + MongoDB golden containers unavailable: {exc}")


def _sqlite(tmp_path: Path) -> tuple[str, str]:
    folder = tmp_path / "sqlite"
    folder.mkdir()
    return f"sqlite:///{folder}/newton.db", "newton"


def _postgres() -> tuple[str, str]:
    from tests.sql.conftest import WIPE_SCHEMA_PREFIX, _skip_unless_pg_wipeable, drop_pg_schema

    url = _skip_unless_pg_wipeable()
    schema = f"{WIPE_SCHEMA_PREFIX}{os.getpid()}"
    drop_pg_schema(url, schema)
    return url, schema


@pytest.fixture(params=["sqlite", "postgres"])
def sql_engine(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SqlEngine]:
    """An empty SQL database of each engine in turn (Postgres: this process's test schema)."""
    from scinr.newton.sql.engine import reset_engines
    from tests.sql._dump import dump_sql

    name = request.param
    url, schema = _sqlite(tmp_path) if name == "sqlite" else _postgres()
    settings = {
        "graph_backend": name,
        "storage_backend": name,
        "freeze_backend": name,
        "sql_url": url,
        "sql_schema": schema,
    }
    dump_schema = schema if name == "postgres" else None
    try:
        yield SqlEngine(name=name, settings=settings, dump=lambda: dump_sql(url, dump_schema))
    finally:
        reset_engines()
        if name == "postgres":
            from tests.sql.conftest import drop_pg_schema

            drop_pg_schema(url, schema)
