"""
tests/unit/test_sql_migrations.py -- Migration runner and schema lifecycle.

Runs on SQLite always and on Postgres when the disposable container is up. The packaged
migration folders are empty until phase 2, so the tests use two migrations of their own.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import sqlalchemy as sa

from scinr.newton.exceptions import ConfigurationError
from scinr.newton.sql import schema as schema_module
from scinr.newton.sql.engine import get_dialect_for_config, get_sync_engine
from scinr.newton.sql.migrations import runner

STATEMENT = runner.STATEMENT_SEPARATOR


def _write(directory: Path, name: str, *statements: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(f"\n{STATEMENT}\n".join(statements), encoding="utf-8")


def _qualified(dialect_name: str, table: str) -> str:
    return f"{{schema}}.{table}" if dialect_name == "postgres" else table


@pytest.fixture
def migrations(sql_backend, tmp_path, monkeypatch) -> Path:
    """Two valid test migrations for the active dialect, wired into the runner."""
    root = tmp_path / "migrations"
    folder = root / sql_backend.backend
    alpha = _qualified(sql_backend.backend, "alpha")
    _write(
        folder,
        "0001_alpha.sql",
        f"CREATE TABLE {alpha} (id INTEGER PRIMARY KEY, note TEXT)",
        # Braces, a semicolon and a percent sign must reach the engine untouched.
        f"INSERT INTO {alpha} VALUES (1, '100% {{x}}; ok; :not_a_bind')",
    )
    _write(folder, "0002_extra.sql", f"ALTER TABLE {alpha} ADD COLUMN extra TEXT")
    monkeypatch.setattr(runner, "migrations_dir", lambda name: root / name)
    return folder


def _state(engine: sa.Engine, target) -> dict | None:
    dialect = get_dialect_for_config()
    table = runner.control_table(dialect, target.schema)
    with engine.connect() as connection:
        row = connection.execute(sa.select(table)).mappings().first()
    return dict(row) if row else None


def _has_table(engine: sa.Engine, target, name: str) -> bool:
    schema = target.schema if target.backend == "postgres" else None
    with engine.connect() as connection:
        return sa.inspect(connection).has_table(name, schema=schema)


def _apply(target, *, directory: Path | None = None, **kwargs) -> int:
    dialect = get_dialect_for_config()
    with get_sync_engine().begin() as connection:
        return runner.apply_migrations(
            connection,
            dialect,
            schema=target.schema,
            fulltext_config="simple",
            directory=directory,
            **kwargs,
        )


# ── Helpers of the runner ────────────────────────────────────────────────────


def test_split_statements_on_the_separator_only() -> None:
    text = f"CREATE FUNCTION f() AS $$ select 1; select 2; $$\n{STATEMENT}\nSELECT 1;\n  {STATEMENT}  \n\n"
    assert runner.split_statements(text) == [
        "CREATE FUNCTION f() AS $$ select 1; select 2; $$",
        "SELECT 1;",
    ]


def test_list_migrations_requires_consecutive_numbers(tmp_path) -> None:
    _write(tmp_path, "0001_a.sql", "SELECT 1")
    _write(tmp_path, "0003_c.sql", "SELECT 1")
    with pytest.raises(ConfigurationError, match="consecutive"):
        runner.list_migrations(tmp_path)


def test_list_migrations_rejects_misnamed_files(tmp_path) -> None:
    _write(tmp_path, "1_a.sql", "SELECT 1")
    with pytest.raises(ConfigurationError, match="Invalid migration file name"):
        runner.list_migrations(tmp_path)


def test_latest_version(tmp_path) -> None:
    assert runner.latest_version("sqlite", tmp_path / "nothing-yet") == 0
    _write(tmp_path, "0001_a.sql", "SELECT 1")
    _write(tmp_path, "0002_b.sql", "SELECT 1")
    assert runner.latest_version("sqlite", tmp_path) == 2


def test_packaged_migration_folders_follow_the_naming_rules() -> None:
    for dialect in ("postgres", "sqlite"):
        runner.list_migrations(runner.migrations_dir(dialect))  # must not raise


# ── apply_migrations ─────────────────────────────────────────────────────────


def test_apply_from_scratch(sql_backend, migrations) -> None:
    assert _apply(sql_backend) == 2
    engine = get_sync_engine()
    state = _state(engine, sql_backend)
    assert state is not None
    assert state["version"] == 2
    assert state["rls_enabled"] is False or state["rls_enabled"] == 0
    assert state["fulltext_config"] == "simple"
    assert state["applied_at"].endswith("+00:00")
    qualified = sa.table(
        "alpha",
        sa.column("note"),
        sa.column("extra"),
        schema=sql_backend.schema if sql_backend.backend == "postgres" else None,
    )
    with engine.connect() as connection:
        assert connection.execute(sa.select(qualified.c.note, qualified.c.extra)).one() == (
            "100% {x}; ok; :not_a_bind",
            None,
        )


def test_second_call_does_nothing(sql_backend, migrations) -> None:
    _apply(sql_backend)
    before = _state(get_sync_engine(), sql_backend)
    assert _apply(sql_backend) == 2
    assert _state(get_sync_engine(), sql_backend) == before


def test_only_pending_migrations_are_applied(sql_backend, migrations) -> None:
    (migrations / "0002_extra.sql").rename(migrations / "0002_extra.sql.off")
    assert _apply(sql_backend) == 1
    (migrations / "0002_extra.sql.off").rename(migrations / "0002_extra.sql")
    assert _apply(sql_backend) == 2
    assert _state(get_sync_engine(), sql_backend)["version"] == 2


def test_a_broken_migration_leaves_nothing_behind(sql_backend, migrations) -> None:
    _write(
        migrations,
        "0003_broken.sql",
        "CREATE TABLE beta (id INTEGER)",
        "SELECT * FROM missing_table_xyz",
    )
    with pytest.raises(sa.exc.DBAPIError):
        _apply(sql_backend)
    engine = get_sync_engine()
    assert _state_or_none(engine, sql_backend) is None
    assert not _has_table(engine, sql_backend, "alpha")
    assert not _has_table(engine, sql_backend, "beta")


def test_a_broken_migration_keeps_the_previous_version(sql_backend, migrations) -> None:
    assert _apply(sql_backend) == 2
    _write(
        migrations,
        "0003_broken.sql",
        "CREATE TABLE beta (id INTEGER)",
        "SELECT * FROM missing_table_xyz",
    )
    with pytest.raises(sa.exc.DBAPIError):
        _apply(sql_backend)
    engine = get_sync_engine()
    assert _state(engine, sql_backend)["version"] == 2
    assert not _has_table(engine, sql_backend, "beta")


def _state_or_none(engine: sa.Engine, target) -> dict | None:
    schema = target.schema if target.backend == "postgres" else None
    with engine.connect() as connection:
        if not sa.inspect(connection).has_table(runner.CONTROL_TABLE, schema=schema):
            return None
    return _state(engine, target)


def test_the_fulltext_marker_is_substituted_and_recorded(
    sql_backend, tmp_path, monkeypatch
) -> None:
    root = tmp_path / "m"
    note = "'{fulltext_config}'"
    table = _qualified(sql_backend.backend, "cfg")
    _write(
        root / sql_backend.backend,
        "0001_cfg.sql",
        f"CREATE TABLE {table} (v TEXT)",
        f"INSERT INTO {table} VALUES ({note})",
    )
    monkeypatch.setattr(runner, "migrations_dir", lambda name: root / name)
    dialect = get_dialect_for_config()
    with get_sync_engine().begin() as connection:
        runner.apply_migrations(
            connection, dialect, schema=sql_backend.schema, fulltext_config="spanish"
        )
    cfg = sa.table(
        "cfg",
        sa.column("v"),
        schema=sql_backend.schema if sql_backend.backend == "postgres" else None,
    )
    with get_sync_engine().connect() as connection:
        assert connection.execute(sa.select(cfg.c.v)).scalar() == "spanish"
    assert _state(get_sync_engine(), sql_backend)["fulltext_config"] == "spanish"


def test_an_existing_schema_keeps_its_fulltext_config(sql_backend, migrations) -> None:
    _apply(sql_backend)
    dialect = get_dialect_for_config()
    with get_sync_engine().begin() as connection:
        runner.apply_migrations(
            connection, dialect, schema=sql_backend.schema, fulltext_config="english"
        )
    assert _state(get_sync_engine(), sql_backend)["fulltext_config"] == "simple"


def test_invalid_identifiers_are_rejected(sql_backend, migrations) -> None:
    dialect = get_dialect_for_config()
    with get_sync_engine().connect() as connection:
        with pytest.raises(ValueError):
            runner.apply_migrations(connection, dialect, schema="a;b", fulltext_config="simple")
        with pytest.raises(ValueError):
            runner.apply_migrations(connection, dialect, schema="ok", fulltext_config="x y")


# ── ensure_schema ────────────────────────────────────────────────────────────


def test_ensure_schema_creates_in_simple_mode(sql_backend, migrations) -> None:
    schema_module.ensure_schema()
    assert _state(get_sync_engine(), sql_backend)["version"] == 2


def test_ensure_schema_runs_once_per_target(sql_backend, migrations, monkeypatch) -> None:
    schema_module.ensure_schema()
    calls: list[int] = []
    monkeypatch.setattr(runner, "apply_migrations", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(runner, "latest_version", lambda *a, **k: calls.append(1) or 99)
    schema_module.ensure_schema()
    assert calls == []


def test_reset_engines_forgets_the_verified_targets(sql_backend, migrations) -> None:
    from scinr.newton.sql.engine import reset_engines

    schema_module.ensure_schema()
    assert schema_module._verified
    reset_engines()
    assert not schema_module._verified


def test_ensure_schema_migrates_an_old_schema_automatically(sql_backend, migrations) -> None:
    (migrations / "0002_extra.sql").rename(migrations / "0002_extra.sql.off")
    schema_module.ensure_schema()
    assert _state(get_sync_engine(), sql_backend)["version"] == 1
    (migrations / "0002_extra.sql.off").rename(migrations / "0002_extra.sql")
    schema_module.reset_verified()
    schema_module.ensure_schema()
    assert _state(get_sync_engine(), sql_backend)["version"] == 2


def _set_state(target, **values) -> None:
    table = runner.control_table(get_dialect_for_config(), target.schema)
    with get_sync_engine().begin() as connection:
        connection.execute(sa.update(table).values(**values))
    schema_module.reset_verified()


def test_rls_schema_with_another_version_is_never_migrated(sql_backend, migrations) -> None:
    (migrations / "0002_extra.sql").rename(migrations / "0002_extra.sql.off")
    schema_module.ensure_schema()
    (migrations / "0002_extra.sql.off").rename(migrations / "0002_extra.sql")
    _set_state(sql_backend, rls_enabled=True)
    with pytest.raises(
        ConfigurationError, match=r"version 1 and this newton needs 2.*setup_sql_schema"
    ):
        schema_module.ensure_schema()
    assert _state(get_sync_engine(), sql_backend)["version"] == 1  # no DDL ran


def test_rls_schema_at_the_right_version_is_accepted(sql_backend, migrations) -> None:
    schema_module.ensure_schema()
    _set_state(sql_backend, rls_enabled=True)
    schema_module.ensure_schema()


def test_schema_newer_than_this_newton_is_rejected(sql_backend, migrations) -> None:
    schema_module.ensure_schema()
    _set_state(sql_backend, version=3)
    with pytest.raises(
        ConfigurationError, match=r"newer than this newton \(version 2\); upgrade scinr"
    ):
        schema_module.ensure_schema()


def test_fulltext_config_mismatch_warns_once(sql_backend, migrations, caplog) -> None:
    from scinr.newton import configure

    schema_module.ensure_schema()
    schema_module.reset_verified()
    configure(
        graph_backend=sql_backend.backend,
        storage_backend=sql_backend.backend,
        freeze_backend=sql_backend.backend,
        sql_url=sql_backend.url,
        sql_schema=sql_backend.schema,
        sql_fulltext_config="english",
    )
    with caplog.at_level(logging.WARNING, logger=schema_module.__name__):
        schema_module.ensure_schema()
        schema_module.ensure_schema()
    warnings = [r for r in caplog.records if "fulltext config" in r.getMessage()]
    assert len(warnings) == 1
    assert "'simple'" in warnings[0].getMessage() and "'english'" in warnings[0].getMessage()


def test_access_denied_while_creating_becomes_a_configuration_error(
    sql_backend, migrations, monkeypatch
) -> None:
    class Denied(Exception):
        sqlstate = "42501"

    def deny(*args, **kwargs):
        raise sa.exc.ProgrammingError("CREATE", {}, Denied("permission denied"))

    monkeypatch.setattr(runner, "apply_migrations", deny)
    dialect = get_dialect_for_config()
    monkeypatch.setattr(type(dialect), "classify_error", lambda self, exc: "access_denied")
    with pytest.raises(ConfigurationError, match="cannot create the schema.*setup_sql_schema"):
        schema_module.ensure_schema()
