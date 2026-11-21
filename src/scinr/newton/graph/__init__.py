"""
graph — Write-path interface to the graph (``GraphStore``) and its backends.

Public surface: :func:`get_graph_store`, :func:`set_graph_store` and
:func:`reset_graph_store`. The interface itself lives in
:mod:`scinr.newton.graph.base`, its domain types in
:mod:`scinr.newton.graph.types` and the Neo4j implementation in
:mod:`scinr.newton.graph.neo4j`.

Outside ``graph/neo4j/`` and ``navigation/neo4j/`` the library imports only
this package, never the engine.
"""

from __future__ import annotations

import threading

from scinr.newton.graph.base import GraphStore

__all__ = ["GraphStore", "get_graph_store", "reset_graph_store", "set_graph_store"]

#: One store per connection parameters of the active configuration (see :func:`_store_key`).
_stores: dict[tuple, GraphStore] = {}
#: Store installed with :func:`set_graph_store`; wins over the cache.
_override: GraphStore | None = None
_lock = threading.Lock()


def _store_key() -> tuple:
    from scinr.newton.config import get_config

    cfg = get_config()
    return (
        cfg.graph_backend,
        cfg.neo4j_uri,
        cfg.neo4j_user,
        cfg.neo4j_database,
        cfg.sql_url,
        cfg.sql_schema,
    )


def get_graph_store() -> GraphStore:
    """Return the :class:`GraphStore` for the active configuration's ``graph_backend``.

    One store is cached per set of connection parameters (backend, Neo4j uri / user /
    database, SQL url / schema), created on first use. It is synchronous to obtain and
    opens no connection by itself. :func:`~scinr.newton.configure` discards them all, so a
    new configuration is picked up by the next call.

    Raises:
        ConfigurationError: If ``graph_backend`` is unknown.
    """
    if _override is not None:
        return _override
    key = _store_key()
    with _lock:
        store = _stores.get(key)
        if store is None:
            from scinr.newton.graph.factory import create_graph_store

            store = _stores[key] = create_graph_store()
    return store


def set_graph_store(store: GraphStore | None) -> None:
    """Replace the process-wide store (for tests and custom backends).

    ``None`` is equivalent to :func:`reset_graph_store`.
    """
    global _override
    with _lock:
        _stores.clear()
        _override = store


def reset_graph_store() -> None:
    """Discard every cached store and any installed one; the next :func:`get_graph_store` builds a new one."""
    set_graph_store(None)
