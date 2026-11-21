"""
navigation/factory.py — Graph-navigator factory.

Mirrors ``storage/factory.py``: the concrete backend is chosen from
``ScinrConfig.graph_backend`` (env ``GRAPH_BACKEND``, default ``"neo4j"``), so
call sites never name an engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext

from scinr.newton.config import ScinrConfig, use_config
from scinr.newton.navigation.base import GraphNavigator


async def get_graph_navigator(config: ScinrConfig | None = None) -> GraphNavigator:
    """Return a connected :class:`GraphNavigator` for the configured backend.

    The connection is verified eagerly (``connect()`` → ``ping()``).

    Args:
        config: Configuration to use instead of the active one (see
            :func:`~scinr.newton.config.build_config`). It applies only while the navigator
            is created and connected: whoever passes it must call the navigator's methods
            (and ``close()``) inside ``use_config(config)``, or use :func:`graph_navigator`,
            which does it for the whole block.

    Returns:
        A ready-to-use navigator. The caller is responsible for
        :meth:`GraphNavigator.close` — or use :func:`graph_navigator` instead.

    Raises:
        ConfigurationError: If ``graph_backend`` is unknown, or the library is
            not configured.
        GraphConnectionError: If the engine is unreachable (a
            :class:`~scinr.newton.exceptions.GraphError`, not a
            :class:`~scinr.newton.exceptions.NavigationError`).
    """
    with use_config(config) if config is not None else nullcontext():
        return await _create_navigator()


async def _create_navigator() -> GraphNavigator:
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError

    cfg = get_config()
    backend = cfg.graph_backend

    if backend == "neo4j":
        from scinr.newton.exceptions import GraphConnectionError

        # The navigator imports the driver lazily, in connect(): guard both steps.
        try:
            from scinr.newton.navigation.neo4j.navigator import Neo4jGraphNavigator

            nav = Neo4jGraphNavigator()
            await nav.connect()
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise GraphConnectionError(
                "graph_backend='neo4j' requires the optional dependency: "
                "pip install 'scinr[neo4j]'"
            ) from exc
        return nav

    if backend in ("postgres", "sqlite"):
        from scinr.newton.navigation.sql.navigator import SqlGraphNavigator

        nav = SqlGraphNavigator()
        await nav.connect()
        return nav

    raise ConfigurationError(
        f"Unknown graph_backend: {backend!r}. Valid values: 'neo4j', 'postgres', 'sqlite'."
    )


@asynccontextmanager
async def graph_navigator(config: ScinrConfig | None = None) -> AsyncIterator[GraphNavigator]:
    """Async context manager that yields a connected navigator and closes it.

    With *config*, the navigator and every call made inside the block use it (the whole
    block, including ``connect`` and ``close``, runs inside ``use_config(config)``).

    Example:
        >>> async with graph_navigator() as nav:
        ...     roots = await nav.list_root_documents()
    """
    with use_config(config) if config is not None else nullcontext():
        nav = await get_graph_navigator()
        try:
            yield nav
        finally:
            await nav.close()


__all__ = ["get_graph_navigator", "graph_navigator"]
