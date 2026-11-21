"""
navigation/_safe.py — Identifier validation, depth resolution, read-only guard.

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
        raise NavigationError(f"Invalid {kind}: {value!r}. Must match ^[A-Za-z_][A-Za-z0-9_]*$.")
    return value


def safe_idents(values, *, kind: str = "identifier") -> list[str]:
    """Validate an iterable of identifiers, returning them as a list."""
    return [safe_ident(v, kind=kind) for v in values]


def safe_keys(values: object, *, kind: str = "property key") -> list[str]:
    """Validate a collection of property keys (not a bare string), returning them as a list.

    Raises:
        NavigationError: *values* is a string or not a collection, or a key is not a valid
            identifier.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set, frozenset)):
        raise NavigationError(f"{kind}s must be a list of strings, got {type(values).__name__}.")
    return safe_idents(values, kind=kind)


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


def check_raw_limits(timeout_seconds: float | None, max_rows: int | None) -> None:
    """Validate the ``timeout_seconds`` / ``max_rows`` arguments of ``execute_raw``.

    Raises:
        NavigationError: *timeout_seconds* is not ``None`` or a number ``> 0``, or
            *max_rows* is not ``None`` or an ``int >= 0``.
    """
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not timeout_seconds > 0
    ):
        raise NavigationError(
            f"timeout_seconds must be None or a number > 0, got {timeout_seconds!r}."
        )
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0
    ):
        raise NavigationError(f"max_rows must be None or an integer >= 0, got {max_rows!r}.")


_SQL_WORD_RE = re.compile(r"[A-Za-z_]+")
_SQL_SPACE = frozenset(" \t\n\r\f\v;")


def leading_sql_keyword(query: str, *, nested_comments: bool) -> str | None:
    """The first word of the SQL *query*, upper-cased, or ``None`` if it does not start with one.

    Skips whitespace, empty statements (``;``), ``--`` comments (ended by ``\\n`` or ``\\r``)
    and ``/* ... */`` comments (nested when *nested_comments*). Where an engine would read a
    comment more widely, this reads it more narrowly, so it may find a keyword the engine
    treats as a comment (a false refusal), never the opposite. An unterminated comment gives
    ``None``: the engine reports it.
    """
    i, n = 0, len(query)
    while i < n:
        if query[i] in _SQL_SPACE:
            i += 1
        elif query.startswith("--", i):
            ends = [p for p in (query.find("\n", i), query.find("\r", i)) if p >= 0]
            if not ends:
                return None
            i = min(ends) + 1
        elif query.startswith("/*", i):
            depth, i = 1, i + 2
            while depth and i < n:
                if nested_comments and query.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif query.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                return None
        else:
            match = _SQL_WORD_RE.match(query, i)
            return match.group(0).upper() if match else None
    return None


def check_raw_sql(query: str, *, refused: frozenset[str], nested_comments: bool) -> None:
    """Refuse a raw SQL statement whose first keyword is in *refused* (see
    ``Dialect.raw_refused_keywords``): transaction control, session state and the like.

    Raises:
        NavigationError: The statement starts with a refused keyword (``code`` is ``None``).
    """
    keyword = leading_sql_keyword(query, nested_comments=nested_comments)
    if keyword is not None and keyword in refused:
        raise NavigationError(
            f"execute_raw runs one read-only query; {keyword} statements are refused."
        )


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


__all__ = [
    "safe_ident",
    "safe_idents",
    "safe_keys",
    "resolve_depth",
    "assert_read_only",
    "check_raw_limits",
    "check_raw_sql",
    "leading_sql_keyword",
]
