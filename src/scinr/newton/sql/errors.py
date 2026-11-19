"""
sql/errors.py — Translate SQL driver exceptions to the scinr exception hierarchy.

:func:`translate` maps one driver exception (raw, or wrapped by SQLAlchemy) to the matching
domain exception, or ``None`` when it does not come from the engine. :class:`sql_errors` and
:class:`async_sql_errors` apply it as a context manager or decorator around a whole store
method; the original exception is chained in ``__cause__``.

The translation goes **outside** the retry (``sql/retry.py``): the retry sees the raw driver
exceptions; only the one that finally leaves the method is translated. Messages are scrubbed
with ``redact_secrets`` and never include the SQL statement or its parameters.

====================  ===========================  ==================
Classified as         ``kind="graph"``             ``kind="storage"``
====================  ===========================  ==================
transient             ``TransientGraphError``      ``StorageError``
unique_violation      ``GraphConstraintError`` (1) ``StorageError``
access_denied         ``TenantAccessError``        ``TenantAccessError``
connection            ``GraphConnectionError``     ``StorageError``
authentication        ``GraphAuthenticationError`` ``StorageError``
resource              ``GraphResourceError``       ``StorageError``
unclassified          ``GraphQueryError``          ``StorageError``
====================  ===========================  ==================

(1) ``DocumentVersionConflict`` when the caller passes ``version_conflict`` and the violated
constraint is about the document version (same conditions as the Neo4j store).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from scinr.newton.exceptions import (
    DocumentVersionConflict,
    GraphAuthenticationError,
    GraphConnectionError,
    GraphConstraintError,
    GraphQueryError,
    GraphResourceError,
    ScinrError,
    StorageError,
    TenantAccessError,
    TransientGraphError,
)
from scinr.newton.utils.redaction import redact_secrets

if TYPE_CHECKING:
    from scinr.newton.sql.dialect import Dialect

F = TypeVar("F", bound=Callable[..., Any])

ErrorSurface = Literal["graph", "storage"]

_GRAPH_ERRORS: dict[str, type[ScinrError]] = {
    "transient": TransientGraphError,
    "unique_violation": GraphConstraintError,
    "connection": GraphConnectionError,
    "authentication": GraphAuthenticationError,
    "resource": GraphResourceError,
}


def _message(exc: BaseException) -> str:
    """The driver's own message: no SQL text, no parameters, no secrets."""
    orig = getattr(exc, "orig", None)
    source = orig if isinstance(orig, BaseException) else exc
    return redact_secrets(str(source)) or ""


def _is_dbapi_error(exc: BaseException) -> bool:
    try:
        from sqlalchemy.exc import DBAPIError
    except ImportError:
        return False
    return isinstance(exc, DBAPIError)


def _is_version_conflict(message: str) -> bool:
    """Same fallback the Neo4j store uses: the violated constraint is about the version."""
    return "version" in message.lower()


def translate(
    exc: BaseException,
    *,
    kind: ErrorSurface = "graph",
    dialect: Dialect | None = None,
    version_conflict: str | None = None,
) -> ScinrError | None:
    """Return the ``scinr`` exception that stands for the driver exception *exc*.

    Args:
        exc: The exception caught at a store boundary.
        kind: ``"graph"`` or ``"storage"``: which exception family to produce.
        dialect: Classifier of driver errors. Default: the dialect of the configured
            ``sql_url``.
        version_conflict: Only for ``insert_document_graph``: the message of the
            :class:`DocumentVersionConflict` to raise when *exc* is a unique violation
            about the document version.

    Returns:
        The translated exception, or ``None`` when *exc* is not an engine error and must
        pass through untouched. The caller chains ``__cause__``.
    """
    if isinstance(exc, ScinrError):
        # Already translated by an inner boundary: translating twice would re-classify it.
        return None
    if dialect is None:
        from scinr.newton.sql.engine import get_dialect_for_config

        try:
            dialect = get_dialect_for_config()
        except Exception:  # noqa: BLE001 — without a SQL configuration nothing is ours
            return None
    error_kind = dialect.classify_error(exc)
    if error_kind is None and not _is_dbapi_error(exc):
        return None
    message = _message(exc)
    if error_kind == "access_denied":
        return TenantAccessError(message)
    if kind == "storage":
        return StorageError(message)
    if error_kind == "unique_violation":
        if version_conflict is not None and _is_version_conflict(message):
            return DocumentVersionConflict(version_conflict)
        return GraphConstraintError(message)
    if error_kind is None:
        return GraphQueryError(message)
    return _GRAPH_ERRORS[error_kind](message)


class sql_errors:  # noqa: N801 — used like a function: ``@sql_errors()``
    """Translate driver exceptions raised inside a block or a function.

    As a context manager::

        with sql_errors():
            ...

    As a decorator (plain or ``async`` functions)::

        @sql_errors("storage")
        def get_file(self, file_id): ...

    Exceptions that are not from the engine pass through untouched.

    Args:
        kind: ``"graph"`` (default) or ``"storage"``; see the module docstring.
        version_conflict: See :func:`translate`.
        dialect: See :func:`translate`.
    """

    def __init__(
        self,
        kind: ErrorSurface = "graph",
        *,
        version_conflict: str | None = None,
        dialect: Dialect | None = None,
    ) -> None:
        self._kind = kind
        self._version_conflict = version_conflict
        self._dialect = dialect

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
        translated = translate(
            exc,
            kind=self._kind,
            dialect=self._dialect,
            version_conflict=self._version_conflict,
        )
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


class async_sql_errors(sql_errors):  # noqa: N801
    """:class:`sql_errors` for ``async with`` blocks (and as a decorator, like its parent)."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return self.__exit__(exc_type, exc, tb)
