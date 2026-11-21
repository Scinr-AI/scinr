"""
graph/neo4j/driver.py — Neo4j connection factories.

Credentials are read from the ScinrConfig singleton (via get_config()) rather
than from module-level constants resolved at import time. This ensures that
a call to configure(neo4j_uri=..., neo4j_user=..., neo4j_password=...) is
always respected, regardless of import order.

Usage:
    from scinr.newton.graph.neo4j.driver import get_driver, get_async_driver

    driver = get_driver()          # new sync driver each call (close it yourself)
    async_driver = get_async_driver()  # module-level singleton — do NOT close it
"""

from __future__ import annotations

import asyncio
import threading

from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase

from scinr.newton.exceptions import GraphConfigurationError

# ---------------------------------------------------------------------------
# Async driver cache
# ---------------------------------------------------------------------------
# One driver per (uri, user, password, event loop): an async driver is bound to the loop it
# is first used in, and two configurations never share a driver. The loop object itself (not
# its id(), which a new loop can reuse) is part of the key; the entries of closed loops are
# dropped when a new driver is created.
_async_drivers: dict[tuple, AsyncDriver] = {}
_lock = threading.Lock()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _driver_key(cfg) -> tuple:
    return (cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password, _running_loop())


def _reset_async_driver_singleton() -> None:
    """Forget every cached async driver (without closing them).

    Called automatically by configure() so the next call to get_async_driver() creates
    a fresh driver with the new credentials.
    """
    with _lock:
        _async_drivers.clear()


def get_async_driver() -> AsyncDriver:
    """Return the cached :class:`AsyncDriver` of the active configuration and running loop.

    Credentials are read from the active ScinrConfig (``configure()`` or ``use_config()``).
    One driver is kept per (uri, user, password, event loop) and reused across calls; do NOT
    close it from individual coroutines.

    Connection pool settings
    ------------------------
    max_connection_pool_size=64, connection_acquisition_timeout=120,
    liveness_check_timeout=30, max_connection_lifetime=600, keep_alive=True.

    Raises
    ------
    GraphConfigurationError
        If Neo4j credentials have not been configured.
    """
    from scinr.newton.config import get_config  # deferred — avoids circular import

    cfg = get_config()
    if not cfg.neo4j_user or not cfg.neo4j_password:
        raise GraphConfigurationError(
            "Neo4j credentials not configured. "
            "Call configure(neo4j_user=..., neo4j_password=...) first, "
            "or set NEO4J_USER + NEO4J_PASSWORD in your .env file."
        )
    key = _driver_key(cfg)
    with _lock:
        driver = _async_drivers.get(key)
        if driver is None:
            for stale in [k for k in _async_drivers if k[-1] is not None and k[-1].is_closed()]:
                del _async_drivers[stale]
            driver = AsyncGraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_user, cfg.neo4j_password),
                max_connection_pool_size=64,
                connection_acquisition_timeout=120,
                liveness_check_timeout=30,
                max_connection_lifetime=600,
                keep_alive=True,
            )
            _async_drivers[key] = driver
    return driver


def get_driver() -> Driver:
    """Return a new authenticated sync :class:`Driver` instance.

    A new driver is created on every call. The caller is responsible for
    closing it (use a try/finally block).

    Credentials are read from ScinrConfig (set via configure()).

    Connection pool settings
    ------------------------
    max_connection_pool_size=64, connection_acquisition_timeout=120,
    liveness_check_timeout=30, max_connection_lifetime=600, keep_alive=True.

    Raises
    ------
    GraphConfigurationError
        If Neo4j credentials have not been configured.
    """
    from scinr.newton.config import get_config  # deferred — avoids circular import

    cfg = get_config()
    if not cfg.neo4j_user or not cfg.neo4j_password:
        raise GraphConfigurationError(
            "Neo4j credentials not configured. "
            "Call configure(neo4j_user=..., neo4j_password=...) first, "
            "or set NEO4J_USER + NEO4J_PASSWORD in your .env file."
        )
    return GraphDatabase.driver(
        cfg.neo4j_uri,
        auth=(cfg.neo4j_user, cfg.neo4j_password),
        max_connection_pool_size=64,
        connection_acquisition_timeout=120,
        liveness_check_timeout=30,
        max_connection_lifetime=600,
        keep_alive=True,
    )


async def close_async_driver() -> None:
    """Close the cached async driver of the active configuration and running loop, and forget it.

    Not called by the pipeline (the driver lives as long as the process);
    :meth:`Neo4jGraphStore.aclose` uses it to release the pool explicitly.
    """
    from scinr.newton.config import get_config  # deferred — avoids circular import

    with _lock:
        driver = _async_drivers.pop(_driver_key(get_config()), None)
    if driver is not None:
        await driver.close()
