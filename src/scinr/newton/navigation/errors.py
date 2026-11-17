"""
navigation/errors.py — Re-export of the navigation exception classes.

The classes themselves live in :mod:`scinr.newton.exceptions` alongside the rest
of the :class:`~scinr.newton.exceptions.ScinrError` hierarchy. This module is a
convenience import surface for navigation code and callers.

``GraphConnectionError`` is a :class:`~scinr.newton.exceptions.GraphError` (the
same class the graph store raises when the engine is down), not a
:class:`~scinr.newton.exceptions.NavigationError`.
"""

from __future__ import annotations

from scinr.newton.exceptions import (
    GraphConnectionError,
    NavigationError,
    UnsupportedOperationError,
)

__all__ = ["NavigationError", "GraphConnectionError", "UnsupportedOperationError"]
