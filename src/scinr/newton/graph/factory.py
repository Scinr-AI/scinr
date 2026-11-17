"""
graph/factory.py — Choose the :class:`GraphStore` implementation for ``graph_backend``.
"""

from __future__ import annotations

from scinr.newton.graph.base import GraphStore

__all__ = ["create_graph_store"]


def create_graph_store() -> GraphStore:
    """Build the store for ``ScinrConfig.graph_backend`` (today only ``"neo4j"``).

    Raises:
        ConfigurationError: If ``graph_backend`` is unknown.
    """
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError

    backend = get_config().graph_backend

    if backend == "neo4j":
        from scinr.newton.graph.neo4j.store import Neo4jGraphStore

        return Neo4jGraphStore()

    raise ConfigurationError(f"Unknown graph_backend: {backend!r}. Valid values: 'neo4j'.")
