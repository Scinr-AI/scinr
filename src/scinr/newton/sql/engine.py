"""
sql/engine.py — Process-wide SQLAlchemy engines for the SQL backends.

One synchronous and one asynchronous engine per process, created on first use from the
configured ``sql_url``. SQLAlchemy and the database drivers are imported inside the
functions, so ``import scinr.newton`` works without the ``postgres`` / ``sqlite`` extras.

The asynchronous engine is bound to the event loop it is first used in (like the
asynchronous Neo4j driver): call :func:`aclose_engines` before that loop closes, or
:func:`reset_engines` (which ``configure()`` does) to start over.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from scinr.newton.exceptions import ConfigurationError, GraphConfigurationError

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

    from scinr.newton.sql.dialect import Dialect

__all__ = [
    "aclose_engines",
    "get_async_engine",
    "get_dialect_for_config",
    "get_sync_engine",
    "reset_engines",
]

_MISSING_EXTRA = (
    "The SQL backends need the 'postgres' or 'sqlite' extra: pip install scinr[postgres]"
)

_lock = threading.Lock()
_sync_engine: Engine | None = None
_async_engine: AsyncEngine | None = None


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


def get_sync_engine() -> Engine:
    """The process-wide synchronous engine, created on first use."""
    global _sync_engine
    if _sync_engine is None:
        with _lock:
            if _sync_engine is None:
                cfg = _config()
                dialect = get_dialect_for_config()
                try:
                    from sqlalchemy import create_engine

                    engine = create_engine(
                        dialect.sync_url(cfg.sql_url), **_engine_options(cfg, dialect)
                    )
                except ImportError as exc:
                    raise ConfigurationError(_MISSING_EXTRA) from exc
                _register_events(engine, dialect)
                _sync_engine = engine
    return _sync_engine


def get_async_engine() -> AsyncEngine:
    """The process-wide asynchronous engine, created on first use.

    It is bound to the event loop of its first use; see the module docstring.
    """
    global _async_engine
    if _async_engine is None:
        with _lock:
            if _async_engine is None:
                cfg = _config()
                dialect = get_dialect_for_config()
                try:
                    from sqlalchemy.ext.asyncio import create_async_engine

                    engine = create_async_engine(
                        dialect.async_url(cfg.sql_url), **_engine_options(cfg, dialect)
                    )
                except ImportError as exc:
                    raise ConfigurationError(_MISSING_EXTRA) from exc
                _register_events(engine.sync_engine, dialect)
                _async_engine = engine
    return _async_engine


def reset_engines() -> None:
    """Dispose both engines and forget them, and forget the verified schemas.

    The next ``get_*_engine()`` call builds new ones from the current configuration.
    Synchronous: the pool of the asynchronous engine is discarded without awaiting the
    close of its connections (use :func:`aclose_engines` for a clean shutdown).
    """
    global _sync_engine, _async_engine
    with _lock:
        sync_engine, async_engine = _sync_engine, _async_engine
        _sync_engine = _async_engine = None
    if sync_engine is not None:
        sync_engine.dispose()
    if async_engine is not None:
        async_engine.sync_engine.dispose(close=False)
    from scinr.newton.sql.schema import reset_verified

    reset_verified()


async def aclose_engines() -> None:
    """Dispose both engines cleanly and forget them. Await it on the engine's own loop."""
    global _sync_engine, _async_engine
    with _lock:
        sync_engine, async_engine = _sync_engine, _async_engine
        _sync_engine = _async_engine = None
    if async_engine is not None:
        await async_engine.dispose()
    if sync_engine is not None:
        sync_engine.dispose()
    from scinr.newton.sql.schema import reset_verified

    reset_verified()
