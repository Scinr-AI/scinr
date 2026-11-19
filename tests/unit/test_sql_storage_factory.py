"""tests/unit/test_sql_storage_factory.py -- The storage and freeze factories with the SQL backends."""

from __future__ import annotations

import pytest

from scinr.newton import configure
from scinr.newton.exceptions import ConfigurationError, StorageError
from scinr.newton.freeze.factory import get_freeze_storage
from scinr.newton.freeze.sql.repository import SqlFreezeRepository
from scinr.newton.sql.engine import reset_engines
from scinr.newton.storage.factory import get_storage
from scinr.newton.storage.sql.pages import SqlPageRepository
from scinr.newton.storage.sql.raw_files import SqlRawFileRepository


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("STORAGE_BACKEND", "FREEZE_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    yield
    reset_engines()


def test_sqlite_storage_and_freeze_factories(sqlite_url) -> None:
    configure(graph_backend="sqlite", storage_backend="sqlite", sql_url=sqlite_url)
    raw, pages = get_storage()
    assert isinstance(raw, SqlRawFileRepository)
    assert isinstance(pages, SqlPageRepository)
    assert raw.stores_images is True
    # freeze_backend inherits the storage backend
    assert isinstance(get_freeze_storage(), SqlFreezeRepository)


def test_storage_and_freeze_can_use_different_axes(sqlite_url) -> None:
    configure(
        graph_backend="sqlite", storage_backend="none", freeze_backend="sqlite", sql_url=sqlite_url
    )
    assert isinstance(get_freeze_storage(), SqlFreezeRepository)


def test_factory_creates_the_schema_once(sqlite_url) -> None:
    import sqlalchemy as sa

    configure(graph_backend="sqlite", storage_backend="sqlite", sql_url=sqlite_url)
    get_storage()
    get_storage()
    engine = sa.create_engine(sqlite_url.replace("sqlite:///", "sqlite+pysqlite:///"))
    try:
        with engine.connect() as connection:
            assert "raw_files" in sa.inspect(connection).get_table_names()
    finally:
        engine.dispose()


def test_unreachable_database_is_a_storage_error(tmp_path) -> None:
    configure(
        graph_backend="sqlite",
        storage_backend="sqlite",
        sql_url=f"sqlite:///{tmp_path}/missing-dir/newton.db",
    )
    with pytest.raises(StorageError):
        get_storage()


def test_freeze_none_message_mentions_the_sql_backends() -> None:
    configure(graph_backend="sqlite", storage_backend="none", sql_url="sqlite:///x.db")
    with pytest.raises(ConfigurationError, match="postgres"):
        get_freeze_storage()
