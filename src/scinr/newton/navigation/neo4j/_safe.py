"""
navigation/neo4j/_safe.py — Identifier validation, depth resolution, read-only guard.

Everything in the Neo4j backend that could otherwise interpolate untrusted text
into Cypher funnels through here. Values are always passed as query parameters;
only validated identifiers and a resolved integer depth are ever formatted into a
query string.
"""

from __future__ import annotations

import re

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.base import DEFAULT_MAX_DEPTH

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Write / side-effecting Cypher clauses that must never appear in execute_raw().
_WRITE_RE = re.compile(
    r"(?is)\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|FOREACH|CALL\s+\{[^}]*}\s*IN\s+TRANSACTIONS"
    r"|LOAD\s+CSV|CREATE\s+OR\s+REPLACE)\b"
)
_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def safe_ident(value: str, *, kind: str = "identifier") -> str:
    """Return *value* unchanged if it is a safe Cypher identifier, else raise.

    Args:
        value: A candidate label, relationship type, or property name.
        kind: What the identifier is, for the error message.

    Returns:
        The same string.

    Raises:
        NavigationError: If *value* is not ``^[A-Za-z_][A-Za-z0-9_]*$``.
    """
    if not isinstance(value, str) or not _IDENT_RE.match(value):
        raise NavigationError(
            f"Invalid {kind}: {value!r}. Must match ^[A-Za-z_][A-Za-z0-9_]*$."
        )
    return value


def safe_idents(values, *, kind: str = "identifier") -> list[str]:
    """Validate an iterable of identifiers, returning them as a list."""
    return [safe_ident(v, kind=kind) for v in values]


def resolve_depth(depth: int | None) -> int:
    """Resolve a user-supplied ``depth`` to a concrete positive integer.

    ``None`` becomes :data:`~scinr.newton.navigation.base.DEFAULT_MAX_DEPTH`
    (a guard against runaway traversals, *not* a hard ceiling). An explicit
    value is used verbatim and must be ``>= 1``.

    Args:
        depth: The requested depth, or ``None``.

    Returns:
        A positive integer.

    Raises:
        NavigationError: If *depth* is not ``None`` and not an ``int >= 1``.
    """
    if depth is None:
        return DEFAULT_MAX_DEPTH
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise NavigationError(f"depth must be a positive integer or None, got {depth!r}.")
    return depth


def assert_read_only(query: str) -> None:
    """Raise if *query* contains a write or side-effecting clause.

    Comments are stripped first so a ``// MERGE`` in a comment does not trip the
    guard and a real ``MERGE`` hidden after ``/* ... */`` still does.

    Raises:
        NavigationError: If a write clause is detected.
    """
    if not isinstance(query, str) or not query.strip():
        raise NavigationError("execute_raw: query must be a non-empty string.")
    stripped = _COMMENT_RE.sub(" ", query)
    m = _WRITE_RE.search(stripped)
    if m:
        raise NavigationError(
            f"execute_raw is read-only; refusing query containing {m.group(1).split()[0].upper()!r}."
        )


__all__ = ["safe_ident", "safe_idents", "resolve_depth", "assert_read_only"]
