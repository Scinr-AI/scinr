"""
navigation/errors.py — Re-export of the navigation exception classes.

The classes themselves live in :mod:`scinr.newton.exceptions` alongside the rest
of the :class:`~scinr.newton.exceptions.ScinrError` hierarchy. This module is a
convenience import surface for navigation code and callers.
"""

from __future__ import annotations

from scinr.newton.exceptions import (
    GraphConnectionError,
    NavigationError,
    UnsupportedOperationError,
)

__all__ = ["NavigationError", "GraphConnectionError", "UnsupportedOperationError"]
