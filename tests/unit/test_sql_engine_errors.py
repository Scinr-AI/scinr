"""
tests/unit/test_sql_engine_errors.py -- Engines, error translation and retries of sql/.

Runs on SQLite always and on Postgres when the disposable container is up.
"""

from __future__ import annotations

import sqlite3
import sys

import pytest
import sqlalchemy as sa

from scinr.newton import configure
from scinr.newton.exceptions import (
    ConfigurationError,
    DocumentVersionConflict,
    GraphAuthenticationError,
    GraphConfigurationError,
    GraphConnectionError,
    GraphConstraintError,
    GraphError,
    GraphQueryError,
    GraphResourceError,
    ScinrError,
    StorageError,
    TenantAccessError,
    TransientGraphError,
)
from scinr.newton.sql import retry
from scinr.newton.sql.dialect import PostgresDialect, SqliteDialect
from scinr.newton.sql.engine import (
    aclose_engines,
    get_async_engine,
    get_dialect_for_config,
    get_sync_engine,
    reset_engines,
)
from scinr.newton.sql.errors import async_sql_errors, sql_errors, translate

# ── Engines ──────────────────────────────────────────────────────────────────


def test_sql_url_must_be_configured() -> None:
    configure(neo4j_user="u", neo4j_password="p")
    with pytest.raises(GraphConfigurationError, match="sql_url is not configured"):
        get_sync_engine()
    with pytest.raises(GraphConfigurationError, match="sql_url is not configured"):
        get_async_engine()


def test_missing_driver_packages_give_an_actionable_error(sqlite_url, monkeypatch) -> None:
    configure(graph_backend="sqlite", sql_url=sqlite_url)
    monkeypatch.setitem(sys.modules, "scinr.newton.sql.dialect", None)
    with pytest.raises(ConfigurationError, match=r"pip install scinr\[postgres\]"):
        get_sync_engine()


def test_engines_are_singletons_until_reset(sql_backend) -> None:
    first = get_sync_engine()
    assert get_sync_engine() is first
    assert get_dialect_for_config().name == sql_backend.backend
    reset_engines()
    assert get_sync_engine() is not first


def test_configure_resets_the_engines(sql_backend) -> None:
    first = get_sync_engine()
    configure(
        graph_backend=sql_backend.backend,
        storage_backend=sql_backend.backend,
        freeze_backend=sql_backend.backend,
        sql_url=sql_backend.url,
        sql_schema=sql_backend.schema,
    )
    assert get_sync_engine() is not first


def test_sync_engine_applies_the_schema_translate_map(sql_backend) -> None:
    table = sa.Table("probe", sa.MetaData(), sa.Column("id", sa.Integer, primary_key=True))
    engine = get_sync_engine()
    with engine.begin() as connection:
        if sql_backend.backend == "postgres":
            connection.execute(sa.schema.CreateSchema(sql_backend.schema, if_not_exists=True))
        table.create(connection)
        connection.execute(table.insert().values(id=1))
    with engine.connect() as connection:
        assert connection.execute(sa.select(table.c.id)).scalar() == 1
        if sql_backend.backend == "postgres":
            where = connection.exec_driver_sql(
                "SELECT table_schema FROM information_schema.tables WHERE table_name = 'probe'"
            ).scalar()
            assert where == sql_backend.schema


async def test_async_engine_works_and_shares_the_setup(sql_backend) -> None:
    table = sa.Table("probe_async", sa.MetaData(), sa.Column("id", sa.Integer, primary_key=True))
    engine = get_async_engine()
    assert get_async_engine() is engine
    try:
        async with engine.begin() as connection:
            if sql_backend.backend == "postgres":
                await connection.execute(
                    sa.schema.CreateSchema(sql_backend.schema, if_not_exists=True)
                )
            await connection.run_sync(table.create)
            await connection.execute(table.insert().values(id=7))
        async with engine.connect() as connection:
            assert (await connection.execute(sa.select(table.c.id))).scalar() == 7
            if sql_backend.backend == "sqlite":
                lower = await connection.exec_driver_sql("SELECT newton_lower('ÁRBOL')")
                assert lower.scalar() == "árbol"
    finally:
        await aclose_engines()
    assert get_sync_engine() is not None  # still usable after the clean shutdown


# ── Error translation ────────────────────────────────────────────────────────


class _PgError(Exception):
    def __init__(self, sqlstate: str | None, message: str = "boom") -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


PG = PostgresDialect()
LITE = SqliteDialect()


@pytest.mark.parametrize(
    "state, graph_type",
    [
        ("40001", TransientGraphError),
        ("23505", GraphConstraintError),
        ("42501", TenantAccessError),
        ("28P01", GraphAuthenticationError),
        ("53200", GraphResourceError),
        ("08006", GraphConnectionError),
    ],
)
def test_translate_graph_kinds(state: str, graph_type: type) -> None:
    translated = translate(_PgError(state, "the message"), dialect=PG)
    assert type(translated) is graph_type
    assert str(translated) == "the message"
    assert isinstance(translated, GraphError)


@pytest.mark.parametrize("state", ["40001", "23505", "28P01", "53200", "08006"])
def test_translate_storage_kinds_are_storage_errors(state: str) -> None:
    translated = translate(_PgError(state), kind="storage", dialect=PG)
    assert type(translated) is StorageError


def test_access_denied_is_a_tenant_error_on_both_surfaces() -> None:
    for kind in ("graph", "storage"):
        translated = translate(_PgError("42501"), kind=kind, dialect=PG)
        assert isinstance(translated, TenantAccessError)
        assert isinstance(translated, GraphError) and isinstance(translated, StorageError)


def test_unclassified_engine_errors_are_query_errors() -> None:
    wrapped = sa.exc.ProgrammingError("SELECT x", {}, _PgError("42P01", "no such table"))
    assert type(translate(wrapped, dialect=PG)) is GraphQueryError
    assert type(translate(wrapped, kind="storage", dialect=PG)) is StorageError


def test_errors_that_are_not_from_the_engine_pass_through() -> None:
    assert translate(ValueError("x"), dialect=PG) is None
    assert translate(KeyError("x"), kind="storage", dialect=LITE) is None
    assert translate(GraphQueryError("already translated"), dialect=PG) is None


def test_version_conflict_translation() -> None:
    message = "duplicate key value violates unique constraint documents_tenant_path_version_key"
    conflict = translate(_PgError("23505", message), dialect=PG, version_conflict="v2 exists")
    assert type(conflict) is DocumentVersionConflict
    assert str(conflict) == "v2 exists"
    other = translate(
        _PgError("23505", "duplicate key ... documents_pkey"), dialect=PG, version_conflict="v2"
    )
    assert type(other) is GraphConstraintError
    plain = translate(_PgError("23505", message), dialect=PG)
    assert type(plain) is GraphConstraintError


def test_messages_hide_the_statement_parameters_and_secrets() -> None:
    configure(graph_backend="postgres", sql_url="postgresql://u:hunter2secret@h/db")
    driver_error = _PgError("08006", "could not connect with hunter2secret")
    wrapped = sa.exc.OperationalError(
        "INSERT INTO t VALUES (%(x)s)", {"x": "private-data"}, driver_error
    )
    translated = translate(wrapped, dialect=PG)
    assert "hunter2secret" not in str(translated)
    assert "private-data" not in str(translated)
    assert "INSERT" not in str(translated)


def test_context_manager_decorator_and_chaining() -> None:
    error = _PgError("40001", "serialization failure")

    with pytest.raises(TransientGraphError) as info, sql_errors(dialect=PG):
        raise error
    assert info.value.__cause__ is error

    @sql_errors("storage", dialect=PG)
    def read() -> None:
        raise _PgError("08006")

    with pytest.raises(StorageError):
        read()

    @sql_errors(dialect=PG)
    def untouched() -> None:
        raise ScinrError("mine")

    with pytest.raises(ScinrError, match="mine"):
        untouched()

    with pytest.raises(ValueError), sql_errors(dialect=PG):
        raise ValueError("not ours")


async def test_async_variants() -> None:
    with pytest.raises(GraphConnectionError):
        async with async_sql_errors(dialect=PG):
            raise _PgError("08006")

    @async_sql_errors("storage", dialect=PG)
    async def read() -> None:
        raise _PgError("40001")

    with pytest.raises(StorageError):
        await read()

    @sql_errors(dialect=PG)
    async def write() -> None:
        raise _PgError("23505")

    with pytest.raises(GraphConstraintError):
        await write()


def test_real_unique_violation_is_translated(sql_backend) -> None:
    table = sa.Table("uq", sa.MetaData(), sa.Column("id", sa.Integer, primary_key=True))
    engine = get_sync_engine()
    with engine.begin() as connection:
        if sql_backend.backend == "postgres":
            connection.execute(sa.schema.CreateSchema(sql_backend.schema, if_not_exists=True))
        table.create(connection)
        connection.execute(table.insert().values(id=1))
    with pytest.raises(GraphConstraintError) as info, sql_errors(), engine.begin() as connection:
        connection.execute(table.insert().values(id=1))
    assert "INSERT" not in str(info.value)
    with pytest.raises(StorageError), sql_errors("storage"), engine.begin() as connection:
        connection.execute(table.insert().values(id=1))


# ── Retries ──────────────────────────────────────────────────────────────────

FAST = {"exp_retries": 2, "plateau_retries": 1, "base_delay": 0.0, "max_delay_exp": 0.0}


@pytest.fixture
def sqlite_configured(sqlite_url) -> None:
    configure(graph_backend="sqlite", sql_url=sqlite_url)


def test_retry_sync_recovers_from_transient_errors(sqlite_configured) -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise sqlite3.OperationalError("database is locked")
        return "done"

    assert retry.with_sql_retry_sync(flaky, **FAST) == "done"
    assert len(attempts) == 3


def test_retry_sync_gives_up_after_the_total(sqlite_configured) -> None:
    attempts: list[int] = []

    def always_busy() -> None:
        attempts.append(1)
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        retry.with_sql_retry_sync(always_busy, **FAST)
    assert len(attempts) == 1 + FAST["exp_retries"] + FAST["plateau_retries"]


def test_retry_sync_does_not_retry_other_errors(sqlite_configured) -> None:
    attempts: list[int] = []

    def broken() -> None:
        attempts.append(1)
        raise sqlite3.OperationalError("no such table: x")

    with pytest.raises(sqlite3.OperationalError):
        retry.with_sql_retry_sync(broken, **FAST)
    assert attempts == [1]


def test_retry_wrapped_in_sqlalchemy_errors_is_still_transient(sqlite_configured) -> None:
    attempts: list[int] = []

    def flaky() -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise sa.exc.OperationalError(
                "UPDATE", {}, sqlite3.OperationalError("database is locked")
            )
        return 1

    assert retry.with_sql_retry_sync(flaky, **FAST) == 1


async def test_retry_async(sqlite_configured) -> None:
    attempts: list[int] = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    assert await retry.with_sql_retry(flaky, **FAST) == "ok"

    async def broken() -> None:
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        await retry.with_sql_retry(broken, **FAST)


def test_retry_without_sql_configuration_never_retries() -> None:
    configure(neo4j_user="u", neo4j_password="p")
    attempts: list[int] = []

    def fails() -> None:
        attempts.append(1)
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        retry.with_sql_retry_sync(fails, **FAST)
    assert attempts == [1]


def test_retry_logs_do_not_leak_the_statement(sqlite_configured, caplog) -> None:
    def flaky(calls=[]) -> int:  # noqa: B006
        calls.append(1)
        if len(calls) == 1:
            raise sa.exc.OperationalError(
                "INSERT INTO secret_table VALUES (?)",
                ("private",),
                sqlite3.OperationalError("database is locked"),
            )
        return 1

    with caplog.at_level("WARNING", logger=retry.logger.name):
        retry.with_sql_retry_sync(flaky, **FAST)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "database is locked" in text
    assert "secret_table" not in text and "private" not in text
