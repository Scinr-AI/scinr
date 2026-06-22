"""
utils/neo4j_concurrency.py — Compatibility shim for Neo4j concurrency semaphore.

The semaphore logic has moved to config.py alongside get_llm_semaphore() and
reset_llm_semaphore(). This module re-exports those functions for backward
compatibility so that existing callers (annotation/nodes.py,
entity_extraction/nodes.py) do not need to change their imports.

To configure concurrency, call configure(neo4j_concurrency=N) followed by
reset_neo4j_semaphore() before running annotation or entity extraction stages.
"""
from __future__ import annotations

from scinr.newton.config import get_neo4j_semaphore, reset_neo4j_semaphore

__all__ = ["get_neo4j_semaphore", "reset_neo4j_semaphore"]
