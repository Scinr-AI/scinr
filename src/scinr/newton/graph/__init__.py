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

_store: GraphStore | None = None
_lock = threading.Lock()


def get_graph_store() -> GraphStore:
    """Return the process-wide :class:`GraphStore` for the configured ``graph_backend``.

    The store is a singleton, created on first use. It is synchronous to
    obtain and opens no connection by itself. :func:`~scinr.newton.configure`
    discards it, so a new configuration is picked up by the next call.

    Raises:
        ConfigurationError: If ``graph_backend`` is unknown.
    """
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                from scinr.newton.graph.factory import create_graph_store

                _store = create_graph_store()
    return _store


def set_graph_store(store: GraphStore | None) -> None:
    """Replace the process-wide store (for tests and custom backends).

    ``None`` is equivalent to :func:`reset_graph_store`.
    """
    global _store
    with _lock:
        _store = store


def reset_graph_store() -> None:
    """Discard the process-wide store; the next :func:`get_graph_store` builds a new one."""
    set_graph_store(None)
