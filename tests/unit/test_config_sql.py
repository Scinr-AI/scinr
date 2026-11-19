"""
tests/unit/test_config_sql.py -- configure() with the SQL backends (Postgres / SQLite).
"""

from __future__ import annotations

import pytest

from scinr.newton.config import configure, get_config
from scinr.newton.exceptions import ConfigurationError

PG_URL = "postgresql://newton:s3cr3tpass@db.example:5432/newton"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_AUTH",
        "GRAPH_BACKEND",
        "STORAGE_BACKEND",
        "FREEZE_BACKEND",
        "SQL_URL",
        "SQL_SCHEMA",
        "SQL_POOL_SIZE",
        "SQL_FULLTEXT_CONFIG",
        "SQLITE_FILES_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sqlite_graph_needs_no_neo4j_credentials() -> None:
    cfg = configure(graph_backend="sqlite", sql_url="sqlite:///x.db")
    assert cfg.graph_backend == "sqlite"
    assert cfg.neo4j_user == "" and cfg.neo4j_password == ""
    assert cfg.sql_url == "sqlite:///x.db"
    assert get_config() is cfg


def test_neo4j_without_credentials_still_fails_with_the_same_message(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    with pytest.raises(ConfigurationError, match="Neo4j username is not configured"):
        configure(graph_backend="neo4j")
    with pytest.raises(ConfigurationError, match="Neo4j password is not configured"):
        configure(neo4j_user="neo4j")
    with pytest.raises(ConfigurationError, match="Neo4j database is not configured"):
        configure(neo4j_user="neo4j", neo4j_password="pw")


def test_unknown_graph_backend_lists_the_valid_values() -> None:
    with pytest.raises(ConfigurationError, match="Unknown graph_backend.*'postgres', 'sqlite'"):
        configure(graph_backend="duckdb")  # type: ignore[arg-type]


def test_sql_defaults() -> None:
    cfg = configure(storage_backend="sqlite", graph_backend="sqlite", sql_url="sqlite:///x.db")
    assert cfg.sql_schema == "newton"
    assert cfg.sql_pool_size == 10
    assert cfg.sql_fulltext_config == "simple"
    assert cfg.sqlite_files_dir == ""


def test_sql_settings_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "postgres")
    monkeypatch.setenv("SQL_URL", PG_URL)
    monkeypatch.setenv("SQL_SCHEMA", "custom")
    monkeypatch.setenv("SQL_POOL_SIZE", "3")
    monkeypatch.setenv("SQL_FULLTEXT_CONFIG", "spanish")
    monkeypatch.setenv("SQLITE_FILES_DIR", "/tmp/files")
    cfg = configure()
    assert cfg.graph_backend == "postgres"
    assert (cfg.sql_url, cfg.sql_schema, cfg.sql_pool_size) == (PG_URL, "custom", 3)
    assert cfg.sql_fulltext_config == "spanish"
    assert cfg.sqlite_files_dir == "/tmp/files"


def test_freeze_backend_inherits_sqlite_from_storage_backend() -> None:
    cfg = configure(graph_backend="sqlite", storage_backend="sqlite", sql_url="sqlite:///x.db")
    assert cfg.freeze_backend == "sqlite"


@pytest.mark.parametrize("axis", ["graph_backend", "storage_backend", "freeze_backend"])
def test_a_sql_axis_requires_sql_url(axis: str) -> None:
    kwargs = {"neo4j_user": "u", "neo4j_password": "p", axis: "sqlite"}
    with pytest.raises(ConfigurationError, match=rf"{axis} 'sqlite' requires sql_url"):
        configure(**kwargs)


def test_the_url_scheme_must_match_every_sql_axis() -> None:
    with pytest.raises(ConfigurationError) as info:
        configure(graph_backend="postgres", sql_url="sqlite:///x.db")
    assert "graph_backend" in str(info.value) and "sqlite:///x.db" in str(info.value)

    with pytest.raises(ConfigurationError, match="storage_backend 'sqlite' does not match"):
        configure(neo4j_user="u", neo4j_password="p", storage_backend="sqlite", sql_url=PG_URL)


def test_the_mismatch_error_redacts_the_url_password() -> None:
    with pytest.raises(ConfigurationError) as info:
        configure(graph_backend="sqlite", sql_url=PG_URL)
    assert "s3cr3tpass" not in str(info.value)
    assert "newton:***@" in str(info.value)


def test_postgres_and_sqlite_cannot_be_mixed_across_axes() -> None:
    with pytest.raises(ConfigurationError, match="cannot mix Postgres and SQLite"):
        configure(
            graph_backend="postgres",
            storage_backend="sqlite",
            freeze_backend="none",
            sql_url=PG_URL,
        )


def test_a_driver_in_the_url_is_accepted() -> None:
    cfg = configure(graph_backend="postgres", sql_url="postgresql+psycopg://u:p@h/db")
    assert cfg.sql_url == "postgresql+psycopg://u:p@h/db"


def test_a_sql_url_is_not_needed_when_no_axis_uses_sql() -> None:
    cfg = configure(neo4j_user="u", neo4j_password="p", storage_backend="mongodb")
    assert cfg.sql_url == ""


@pytest.mark.parametrize("schema", ["1abc", "a-b", "a b", "x" * 64, "a;drop"])
def test_invalid_sql_schema(schema: str) -> None:
    with pytest.raises(ConfigurationError, match="sql_schema"):
        configure(graph_backend="sqlite", sql_url="sqlite:///x.db", sql_schema=schema)


@pytest.mark.parametrize("name", ["1abc", "a-b", "english;"])
def test_invalid_sql_fulltext_config(name: str) -> None:
    with pytest.raises(ConfigurationError, match="sql_fulltext_config"):
        configure(graph_backend="sqlite", sql_url="sqlite:///x.db", sql_fulltext_config=name)


def test_sql_pool_size_minimum() -> None:
    with pytest.raises(ConfigurationError, match="sql_pool_size must be >= 1"):
        configure(graph_backend="sqlite", sql_url="sqlite:///x.db", sql_pool_size=0)


def test_repr_redacts_the_sql_url_password() -> None:
    cfg = configure(graph_backend="postgres", sql_url=PG_URL)
    text = repr(cfg)
    assert "s3cr3tpass" not in text
    assert "newton:***@db.example" in text


def test_redact_secrets_masks_the_sql_url_password_in_free_text() -> None:
    from scinr.newton.utils.redaction import redact_secrets

    configure(graph_backend="postgres", sql_url=PG_URL)
    assert "s3cr3tpass" not in redact_secrets("connection failed with password s3cr3tpass")
    assert "s3cr3tpass" not in redact_secrets(f"could not connect to {PG_URL}")
