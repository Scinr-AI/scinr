"""navigation/neo4j/_safe.py — Re-export of :mod:`scinr.newton.navigation._safe` (moved there)."""

from __future__ import annotations

from scinr.newton.navigation._safe import (
    assert_read_only,
    resolve_depth,
    safe_ident,
    safe_idents,
)

__all__ = ["safe_ident", "safe_idents", "resolve_depth", "assert_read_only"]
