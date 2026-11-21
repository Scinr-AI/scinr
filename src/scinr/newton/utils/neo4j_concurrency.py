"""
utils/neo4j_concurrency.py — Deprecated compatibility shim (removed in 0.7.0).

Use ``scinr.newton.utils.graph_concurrency`` (``get_graph_semaphore`` /
``reset_graph_semaphore``) instead. The functions re-exported here are the
deprecated aliases: calling them emits a ``DeprecationWarning`` and delegates to
the graph semaphore.

To configure concurrency, call configure(graph_concurrency=N) followed by
reset_graph_semaphore() before running annotation or entity extraction stages.
"""
from __future__ import annotations

from scinr.newton.config import get_neo4j_semaphore, reset_neo4j_semaphore

__all__ = ["get_neo4j_semaphore", "reset_neo4j_semaphore"]
