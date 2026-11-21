"""
utils/graph_concurrency.py — Compatibility shim for the graph-store concurrency semaphore.

The semaphore logic lives in config.py alongside get_llm_semaphore() and
reset_llm_semaphore(). This module re-exports those functions so that callers
(annotation/nodes.py, entity_extraction/nodes.py) can import them from a stable path.

To configure concurrency, call configure(graph_concurrency=N) followed by
reset_graph_semaphore() before running annotation or entity extraction stages.
"""

from __future__ import annotations

from scinr.newton.config import get_graph_semaphore, reset_graph_semaphore

__all__ = ["get_graph_semaphore", "reset_graph_semaphore"]
