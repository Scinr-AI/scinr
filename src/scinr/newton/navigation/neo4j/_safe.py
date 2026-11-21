"""navigation/neo4j/_safe.py — Re-export of :mod:`scinr.newton.navigation._safe` (moved there)."""

from __future__ import annotations

from scinr.newton.navigation._safe import (
    assert_read_only,
    check_raw_limits,
    resolve_depth,
    safe_ident,
    safe_idents,
    safe_keys,
)

__all__ = ["safe_ident", "safe_idents", "safe_keys", "resolve_depth", "assert_read_only", "check_raw_limits"]
