"""
graph/factory.py — Choose the :class:`GraphStore` implementation for ``graph_backend``.
"""

from __future__ import annotations

from scinr.newton.graph.base import GraphStore

__all__ = ["create_graph_store"]


def create_graph_store() -> GraphStore:
    """Build the store for ``ScinrConfig.graph_backend``.

    ``"neo4j"`` builds the Neo4j store; ``"postgres"`` and ``"sqlite"`` the SQL one (the
    engine follows from ``sql_url``).

    Raises:
        ConfigurationError: If ``graph_backend`` is unknown, or the SQL extras are missing.
        GraphConnectionError: If ``graph_backend='neo4j'`` and the ``neo4j`` extra is not
            installed.
    """
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError

    backend = get_config().graph_backend

    if backend == "neo4j":
        from scinr.newton.exceptions import GraphConnectionError

        try:
            from scinr.newton.graph.neo4j.store import Neo4jGraphStore
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise GraphConnectionError(
                "graph_backend='neo4j' requires the optional dependency: "
                "pip install 'scinr[neo4j]'"
            ) from exc

        return Neo4jGraphStore()

    if backend in ("postgres", "sqlite"):
        try:
            from scinr.newton.graph.sql.store import SqlGraphStore
        except ImportError as exc:
            raise ConfigurationError(
                "graph_backend 'postgres' / 'sqlite' needs the SQL extras: "
                "pip install scinr[postgres] (or scinr[sqlite])"
            ) from exc
        return SqlGraphStore()

    raise ConfigurationError(
        f"Unknown graph_backend: {backend!r}. Valid values: 'neo4j', 'postgres', 'sqlite'."
    )
