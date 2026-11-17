"""
graph/neo4j/errors.py — Translate ``neo4j.exceptions`` to the graph exception hierarchy.

:func:`translate` maps one driver exception to the matching
:class:`~scinr.newton.exceptions.GraphError` (or ``None`` when the exception
does not come from the engine). :func:`neo4j_errors` applies it as a decorator
or a context manager around a whole store method.

The translation goes **outside** the retry: ``retry.py`` still sees the raw
driver exceptions and decides what is transient; only the exception that
finally leaves the method (after the retries) is translated. The message is
kept (``str(new) == str(original)``) and the original is chained in
``__cause__``.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar

from neo4j.exceptions import (
    AuthError,
    ConstraintError,
    DriverError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)

from scinr.newton.exceptions import (
    DocumentVersionConflict,
    GraphAuthenticationError,
    GraphConnectionError,
    GraphConstraintError,
    GraphQueryError,
    GraphResourceError,
    ScinrError,
    TransientGraphError,
)
from scinr.newton.graph.neo4j.retry import _is_transient_error

F = TypeVar("F", bound=Callable[..., Any])

#: Fragments of the Neo4j status code that mean the server ran out of memory.
_RESOURCE_CODE_MARKERS = ("MemoryPoolOutOfMemoryError", "OutOfMemoryError")


def _is_version_conflict(exc: ConstraintError) -> bool:
    """The two conditions the loader has always used to spot a version conflict."""
    from scinr.newton.graph.neo4j.schema import DOCUMENT_KEY_CONSTRAINT

    msg = str(exc).lower()
    return DOCUMENT_KEY_CONSTRAINT in msg or "version" in msg


def translate(exc: BaseException, *, version_conflict: str | None = None) -> ScinrError | None:
    """Return the ``scinr`` exception that stands for the driver exception *exc*.

    Args:
        exc: The exception caught at a store boundary.
        version_conflict: Only for ``insert_document_graph``: the message of
            the :class:`DocumentVersionConflict` to raise when *exc* is a
            ``ConstraintError`` that meets the version-conflict conditions.

    Returns:
        The translated exception (message copied from *exc*, except for the
        version conflict, which carries *version_conflict*), or ``None`` when
        *exc* is not a graph-engine error and must pass through untouched.
        The caller is responsible for chaining ``__cause__``.
    """
    if isinstance(exc, ScinrError):
        # Already translated by an inner boundary (a sink method inside a
        # sink's context manager): translating twice would re-classify it.
        return None
    message = str(exc)
    if isinstance(exc, ConstraintError):
        if version_conflict is not None and _is_version_conflict(exc):
            return DocumentVersionConflict(version_conflict)
        return GraphConstraintError(message)
    if isinstance(exc, AuthError):
        return GraphAuthenticationError(message)
    if isinstance(exc, (ServiceUnavailable, SessionExpired)):
        return GraphConnectionError(message)
    if isinstance(exc, Neo4jError):
        code = exc.code or ""
        if any(marker in code for marker in _RESOURCE_CODE_MARKERS):
            return GraphResourceError(message)
    if isinstance(exc, Exception) and _is_transient_error(exc):
        return TransientGraphError(message)
    if isinstance(exc, (Neo4jError, DriverError)):
        return GraphQueryError(message)
    return None


class neo4j_errors:  # noqa: N801 — used like a function: ``@neo4j_errors()``
    """Translate driver exceptions raised inside a block or a function.

    As a context manager::

        with neo4j_errors():
            ...

    As a decorator (plain or ``async`` functions)::

        @neo4j_errors()
        def next_version(self, path, tenant): ...

    Exceptions that are not from the engine pass through untouched.

    Args:
        version_conflict: See :func:`translate`.
    """

    def __init__(self, *, version_conflict: str | None = None) -> None:
        self._version_conflict = version_conflict

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is None:
            return False
        translated = translate(exc, version_conflict=self._version_conflict)
        if translated is None:
            return False
        raise translated from exc

    def __call__(self, fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self:
                    return await fn(*args, **kwargs)

            return _async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)

        return _wrapper  # type: ignore[return-value]
