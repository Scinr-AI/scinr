"""
sql/engine.py — Process-wide SQLAlchemy engines for the SQL backends.

Engines are cached per connection parameters (``sql_url``, ``sql_schema``, ``sql_pool_size``,
``sql_fulltext_config``), created on first use from the active configuration
(``get_config()``, so ``use_config()`` is honoured): two configurations with different
parameters never share an engine. At most ``sql_max_engines`` are kept per kind
(synchronous / asynchronous); the least recently used one is disposed when the limit is
exceeded. SQLAlchemy and the database drivers are imported inside the functions, so
``import scinr.newton`` works without the ``postgres`` / ``sqlite`` extras.

An asynchronous engine is bound to the event loop it is first used in (like the
asynchronous Neo4j driver), so the loop is part of its cache key. The engine of a loop that
no longer exists stays cached until the LRU evicts it or :func:`reset_engines` runs. Call
:func:`aclose_engines` before the loop closes, or :func:`reset_engines` (which
``configure()`` does) to start over.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from scinr.newton.exceptions import ConfigurationError, GraphConfigurationError

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

    from scinr.newton.sql.dialect import Dialect

__all__ = [
    "aclose_engines",
    "dispose_sync_engine",
    "get_async_engine",
    "get_dialect_for_config",
    "get_sync_engine",
    "reset_engines",
]

_MISSING_EXTRA = (
    "The SQL backends need the 'postgres' or 'sqlite' extra: pip install scinr[postgres]"
)

_lock = threading.Lock()
#: LRU caches (least recently used first), keyed by :func:`_engine_key`; the asynchronous
#: key also carries the event loop (``None`` outside a loop).
_sync_engines: OrderedDict[tuple, Engine] = OrderedDict()
_async_engines: OrderedDict[tuple, AsyncEngine] = OrderedDict()


def _config() -> Any:
    from scinr.newton.config import get_config

    cfg = get_config()
    if not cfg.sql_url:
        raise GraphConfigurationError("sql_url is not configured.")
    return cfg


def get_dialect_for_config() -> Dialect:
    """The :class:`~scinr.newton.sql.dialect.Dialect` of the configured ``sql_url``.

    Raises:
        GraphConfigurationError: If ``sql_url`` is empty.
        ConfigurationError: If the SQL extras are not installed.
    """
    cfg = _config()
    try:
        from scinr.newton.sql.dialect import get_dialect
    except ImportError as exc:
        raise ConfigurationError(_MISSING_EXTRA) from exc
    return get_dialect(cfg.sql_url, fulltext_config=cfg.sql_fulltext_config)


def _engine_options(cfg: Any, dialect: Dialect) -> dict[str, Any]:
    options = dialect.engine_kwargs(cfg.sql_pool_size)
    options["execution_options"] = {
        "schema_translate_map": dialect.schema_translate_map(cfg.sql_schema)
    }
    return options


def _register_events(engine: Engine, dialect: Dialect) -> None:
    from sqlalchemy import event

    event.listen(engine, "connect", lambda dbapi_conn, _record: dialect.on_connect(dbapi_conn))
    event.listen(engine, "begin", dialect.on_begin)


def _engine_key(cfg: Any) -> tuple:
    return (cfg.sql_url, cfg.sql_schema, cfg.sql_pool_size, cfg.sql_fulltext_config)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """The running event loop, or ``None``.

    The loop object itself (not its ``id()``) goes in the cache key: a closed loop's id can
    be reused by a new loop, which would hand the new loop an engine bound to the old one.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _evict(cache: OrderedDict, limit: int) -> list[Any]:
    """Remove the least recently used entries beyond *limit* (caller holds ``_lock``)."""
    evicted = []
    while len(cache) > limit:
        _, engine = cache.popitem(last=False)
        evicted.append(engine)
    return evicted


def get_sync_engine() -> Engine:
    """The synchronous engine of the active configuration, created on first use."""
    cfg = _config()
    key = _engine_key(cfg)
    evicted: list[Engine] = []
    with _lock:
        engine = _sync_engines.get(key)
        if engine is None:
            dialect = get_dialect_for_config()
            try:
                from sqlalchemy import create_engine

                engine = create_engine(
                    dialect.sync_url(cfg.sql_url), **_engine_options(cfg, dialect)
                )
            except ImportError as exc:
                raise ConfigurationError(_MISSING_EXTRA) from exc
            _register_events(engine, dialect)
            _sync_engines[key] = engine
            evicted = _evict(_sync_engines, cfg.sql_max_engines)
        _sync_engines.move_to_end(key)
    for old in evicted:
        old.dispose(close=False)
    return engine


def get_async_engine() -> AsyncEngine:
    """The asynchronous engine of the active configuration and running loop, created on first use.

    It is bound to the event loop of its first use; see the module docstring.
    """
    cfg = _config()
    key = (*_engine_key(cfg), _running_loop())
    evicted: list[AsyncEngine] = []
    with _lock:
        engine = _async_engines.get(key)
        if engine is None:
            dialect = get_dialect_for_config()
            try:
                from sqlalchemy.ext.asyncio import create_async_engine

                engine = create_async_engine(
                    dialect.async_url(cfg.sql_url), **_engine_options(cfg, dialect)
                )
            except ImportError as exc:
                raise ConfigurationError(_MISSING_EXTRA) from exc
            _register_events(engine.sync_engine, dialect)
            _async_engines[key] = engine
            evicted = _evict(_async_engines, cfg.sql_max_engines)
        _async_engines.move_to_end(key)
    for old in evicted:
        old.sync_engine.dispose(close=False)
    return engine


def dispose_sync_engine() -> None:
    """Dispose the pool of the synchronous engine of the active configuration, if one exists.

    The engine stays cached and rebuilds its pool on next use.
    """
    from scinr.newton.config import get_config

    with _lock:
        engine = _sync_engines.get(_engine_key(get_config()))
    if engine is not None:
        engine.dispose()


def _take_all() -> tuple[list[Engine], list[tuple[asyncio.AbstractEventLoop | None, AsyncEngine]]]:
    with _lock:
        sync_engines = list(_sync_engines.values())
        async_engines = [(key[-1], engine) for key, engine in _async_engines.items()]
        _sync_engines.clear()
        _async_engines.clear()
    return sync_engines, async_engines


def reset_engines() -> None:
    """Dispose every engine and forget them, and forget the verified schemas.

    The next ``get_*_engine()`` call builds new ones from the current configuration.
    Synchronous: the pool of each asynchronous engine is discarded without awaiting the
    close of its connections (use :func:`aclose_engines` for a clean shutdown).
    """
    sync_engines, async_engines = _take_all()
    for engine in sync_engines:
        engine.dispose()
    for _, engine in async_engines:
        engine.sync_engine.dispose(close=False)
    from scinr.newton.sql.schema import reset_verified

    reset_verified()


async def aclose_engines() -> None:
    """Dispose every engine cleanly and forget them. Await it on the engines' own loop.

    Asynchronous engines of another loop cannot be awaited from this one: their pool is
    discarded without awaiting the close of its connections, as :func:`reset_engines` does.
    """
    sync_engines, async_engines = _take_all()
    current_loop = _running_loop()
    for loop, engine in async_engines:
        if loop is current_loop:
            await engine.dispose()
        else:
            engine.sync_engine.dispose(close=False)
    for engine in sync_engines:
        engine.dispose()
    from scinr.newton.sql.schema import reset_verified

    reset_verified()
