"""
navigation/factory.py — Graph-navigator factory.

Mirrors ``storage/factory.py``: the concrete backend is chosen from
``ScinrConfig.graph_backend`` (env ``GRAPH_BACKEND``, default ``"neo4j"``), so
call sites never name an engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from scinr.newton.navigation.base import GraphNavigator


async def get_graph_navigator() -> GraphNavigator:
    """Return a connected :class:`GraphNavigator` for the configured backend.

    The connection is verified eagerly (``connect()`` → ``ping()``).

    Returns:
        A ready-to-use navigator. The caller is responsible for
        :meth:`GraphNavigator.close` — or use :func:`graph_navigator` instead.

    Raises:
        ConfigurationError: If ``graph_backend`` is unknown, or the library is
            not configured.
        GraphConnectionError: If the engine is unreachable.
    """
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError

    cfg = get_config()
    backend = cfg.graph_backend

    if backend == "neo4j":
        from scinr.newton.navigation.neo4j.navigator import Neo4jGraphNavigator

        nav = Neo4jGraphNavigator()
        await nav.connect()
        return nav

    raise ConfigurationError(
        f"Unknown graph_backend: {backend!r}. Valid values: 'neo4j'."
    )


@asynccontextmanager
async def graph_navigator() -> AsyncIterator[GraphNavigator]:
    """Async context manager that yields a connected navigator and closes it.

    Example:
        >>> async with graph_navigator() as nav:
        ...     roots = await nav.list_root_documents()
    """
    nav = await get_graph_navigator()
    try:
        yield nav
    finally:
        await nav.close()


__all__ = ["get_graph_navigator", "graph_navigator"]
