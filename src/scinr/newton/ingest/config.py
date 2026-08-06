"""
ingest/config.py — Neo4j connection factories.

Credentials are read from the ScinrConfig singleton (via get_config()) rather
than from module-level constants resolved at import time. This ensures that
a call to configure(neo4j_uri=..., neo4j_user=..., neo4j_password=...) is
always respected, regardless of import order.

Usage:
    from scinr.newton.ingest.config import get_driver, get_async_driver

    driver = get_driver()          # new sync driver each call (close it yourself)
    async_driver = get_async_driver()  # module-level singleton — do NOT close it
"""

from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase

# ---------------------------------------------------------------------------
# Async driver singleton
# ---------------------------------------------------------------------------
_async_driver: AsyncDriver | None = None


def _reset_async_driver_singleton() -> None:
    """Invalidate the async driver singleton.

    Called automatically by configure() when credentials change so the next
    call to get_async_driver() creates a fresh driver with the new credentials.
    """
    global _async_driver
    _async_driver = None


def get_async_driver() -> AsyncDriver:
    """Return a module-level singleton :class:`AsyncDriver` instance.

    Credentials are read from ScinrConfig (set via configure()) on each
    creation. The driver is reused across calls; do NOT close it from
    individual coroutines.

    Connection pool settings
    ------------------------
    max_connection_pool_size=64, connection_acquisition_timeout=120,
    liveness_check_timeout=30, max_connection_lifetime=600, keep_alive=True.

    Raises
    ------
    ValueError
        If Neo4j credentials have not been configured.
    """
    from scinr.newton.config import get_config  # deferred — avoids circular import

    global _async_driver
    cfg = get_config()
    if not cfg.neo4j_user or not cfg.neo4j_password:
        raise ValueError(
            "Neo4j credentials not configured. "
            "Call configure(neo4j_user=..., neo4j_password=...) first, "
            "or set NEO4J_USER + NEO4J_PASSWORD in your .env file."
        )
    if _async_driver is None:
        _async_driver = AsyncGraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_pool_size=64,
            connection_acquisition_timeout=120,
            liveness_check_timeout=30,
            max_connection_lifetime=600,
            keep_alive=True,
        )
    return _async_driver


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
    ValueError
        If Neo4j credentials have not been configured.
    """
    from scinr.newton.config import get_config  # deferred — avoids circular import

    cfg = get_config()
    if not cfg.neo4j_user or not cfg.neo4j_password:
        raise ValueError(
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
