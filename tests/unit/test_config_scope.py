"""
tests/unit/test_config_scope.py -- build_config() / use_config() and the per-configuration
caches of SQL engines.
"""

from __future__ import annotations

import asyncio

import pytest

import scinr.newton.config as cfg_module
from scinr.newton.config import build_config, configure, get_config, use_config
from scinr.newton.exceptions import ConfigurationError
from scinr.newton.navigation import graph_navigator
from scinr.newton.sql import engine as sql_engine


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    """Clean env, and restore the global configuration and the engine caches afterwards."""
    for name in (
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_AUTH",
        "NEO4J_DATABASE",
        "GRAPH_BACKEND",
        "STORAGE_BACKEND",
        "FREEZE_BACKEND",
        "SQL_URL",
        "SQL_SCHEMA",
        "SQL_POOL_SIZE",
        "SQL_MAX_ENGINES",
        "SQL_FULLTEXT_CONFIG",
        "SQLITE_FILES_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    previous = cfg_module._config
    sql_engine.reset_engines()
    yield
    sql_engine.reset_engines()
    cfg_module._config = previous


def _sqlite_cfg(url: str, **kwargs):
    return build_config(graph_backend="sqlite", sql_url=url, **kwargs)


def test_build_config_has_no_side_effects() -> None:
    configure(graph_backend="sqlite", sql_url="sqlite:///global.db")
    before = get_config()
    cfg = _sqlite_cfg("sqlite:///other.db")
    assert cfg.sql_url == "sqlite:///other.db"
    assert get_config() is before


def test_build_config_validates_like_configure() -> None:
    with pytest.raises(ConfigurationError, match="Neo4j username is not configured"):
        build_config(graph_backend="neo4j")


def test_sql_max_engines_resolution(monkeypatch) -> None:
    assert _sqlite_cfg("sqlite:///x.db").sql_max_engines == 64
    monkeypatch.setenv("SQL_MAX_ENGINES", "3")
    assert _sqlite_cfg("sqlite:///x.db").sql_max_engines == 3
    assert _sqlite_cfg("sqlite:///x.db", sql_max_engines=5).sql_max_engines == 5
    with pytest.raises(ConfigurationError):
        _sqlite_cfg("sqlite:///x.db", sql_max_engines=0)


def test_use_config_overrides_nests_and_restores() -> None:
    glob = configure(graph_backend="sqlite", sql_url="sqlite:///global.db")
    cfg_a, cfg_b = _sqlite_cfg("sqlite:///a.db"), _sqlite_cfg("sqlite:///b.db")
    with use_config(cfg_a) as yielded:
        assert yielded is cfg_a
        assert get_config() is cfg_a
        with use_config(cfg_b):
            assert get_config() is cfg_b
        assert get_config() is cfg_a
    assert get_config() is glob


def test_use_config_restores_after_exception() -> None:
    glob = configure(graph_backend="sqlite", sql_url="sqlite:///global.db")
    with pytest.raises(RuntimeError):
        with use_config(_sqlite_cfg("sqlite:///a.db")):
            raise RuntimeError("boom")
    assert get_config() is glob


def test_use_config_works_without_global_configuration() -> None:
    cfg_module._config = None
    cfg = _sqlite_cfg("sqlite:///a.db")
    with use_config(cfg):
        assert get_config() is cfg
    with pytest.raises(ConfigurationError):
        get_config()


async def test_concurrent_tasks_each_see_their_own_config() -> None:
    configure(graph_backend="sqlite", sql_url="sqlite:///global.db")

    async def worker(url: str) -> str:
        with use_config(_sqlite_cfg(url)):
            await asyncio.sleep(0.01)
            return get_config().sql_url

    urls = [f"sqlite:///{name}.db" for name in "abc"]
    assert await asyncio.gather(*(worker(u) for u in urls)) == urls
    assert get_config().sql_url == "sqlite:///global.db"


async def test_to_thread_sees_the_overridden_config() -> None:
    configure(graph_backend="sqlite", sql_url="sqlite:///global.db")
    with use_config(_sqlite_cfg("sqlite:///a.db")):
        url = await asyncio.to_thread(lambda: get_config().sql_url)
    assert url == "sqlite:///a.db"


async def test_two_sqlite_files_use_their_own_engine(tmp_path) -> None:
    configure(graph_backend="sqlite", sql_url=f"sqlite:///{tmp_path}/global.db")
    urls = [f"sqlite:///{tmp_path}/{name}.db" for name in ("a", "b")]

    async def count(url: str):
        async with graph_navigator(_sqlite_cfg(url)) as nav:
            await asyncio.sleep(0.01)
            return await nav.execute_raw("SELECT count(*) AS n FROM documents")

    results = await asyncio.gather(*(count(u) for u in urls))
    assert all(r[0]["n"] == 0 for r in results)
    assert len(sql_engine._async_engines) == 2
    assert {key[0] for key in sql_engine._async_engines} == set(urls)


def test_consecutive_event_loops_do_not_share_an_async_engine(tmp_path) -> None:
    cfg = _sqlite_cfg(f"sqlite:///{tmp_path}/a.db")

    async def count():
        async with graph_navigator(cfg) as nav:
            return await nav.execute_raw("SELECT count(*) AS n FROM documents")

    for _ in range(2):
        assert asyncio.run(count())[0]["n"] == 0
    assert len(sql_engine._async_engines) == 2


def test_sql_max_engines_evicts_the_least_recently_used(tmp_path) -> None:
    cfg_a = _sqlite_cfg(f"sqlite:///{tmp_path}/a.db", sql_max_engines=1)
    cfg_b = _sqlite_cfg(f"sqlite:///{tmp_path}/b.db", sql_max_engines=1)
    with use_config(cfg_a):
        engine_a = sql_engine.get_sync_engine()
    with use_config(cfg_b):
        engine_b = sql_engine.get_sync_engine()
    assert list(sql_engine._sync_engines.values()) == [engine_b]
    assert engine_a is not engine_b


def test_sync_engine_is_reused_per_configuration(tmp_path) -> None:
    cfg_a = _sqlite_cfg(f"sqlite:///{tmp_path}/a.db")
    cfg_b = _sqlite_cfg(f"sqlite:///{tmp_path}/b.db")
    with use_config(cfg_a):
        first = sql_engine.get_sync_engine()
    with use_config(cfg_b):
        other = sql_engine.get_sync_engine()
    with use_config(cfg_a):
        assert sql_engine.get_sync_engine() is first
    assert other is not first
    sql_engine.reset_engines()
    assert not sql_engine._sync_engines
