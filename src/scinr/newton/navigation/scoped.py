"""
navigation/scoped.py — A navigator view with the tenant / user / job scope fixed.

Reads default to "all tenants" (see :mod:`scinr.newton.navigation.scope`), so an
API layer that forgets to pass ``tenant_id`` would expose everything.
:meth:`GraphNavigator.scoped` returns a :class:`ScopedNavigator` that fills the
scope into every call once, and refuses any call that would widen it::

    async with graph_navigator() as nav:
        acme = nav.scoped(tenant_id="acme", include_public=True)
        docs = await acme.get_documents()                       # acme + public
        await acme.get_documents(tenant_id="globex")            # NavigationError

A call may *narrow* the scope (a ``job_id`` subset, ``tenant_id="__public__"``
inside an ``include_public`` view) but never widen it.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Sequence
from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.scope import make_scope
from scinr.newton.utils.tenancy import PUBLIC_TENANT

_SCOPE_KWARGS = ("tenant_id", "include_public", "created_by_user_id", "job_id")


def _as_set(value: str | Sequence[str]) -> set[str]:
    return {value} if isinstance(value, str) else set(value)


class ScopedNavigator:
    """Duck-typed :class:`~scinr.newton.navigation.GraphNavigator` with a fixed scope.

    Scope-aware methods get the view's filters filled in; catalogue methods
    (which take no scope) are passed through. ``execute_raw`` /
    ``execute_raw_one`` are refused: raw queries cannot be scoped.
    """

    def __init__(
        self,
        inner: Any,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> None:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)  # validate now
        self._inner = inner
        self._tenant_id = tenant_id
        self._include_public = bool(include_public)
        self._user_ids = created_by_user_id
        self._job_ids = job_id

    @property
    def dialect(self) -> str:
        return self._inner.dialect

    # -- lifecycle (shared with the wrapped navigator) ----------------------

    async def connect(self) -> None:
        await self._inner.connect()

    async def close(self) -> None:
        await self._inner.close()

    async def ping(self) -> bool:
        return await self._inner.ping()

    def scoped(self, **kwargs: Any) -> ScopedNavigator:
        """Narrow this view further (never widen it)."""
        return ScopedNavigator(self, **kwargs)

    # -- raw escape hatch: refused -------------------------------------------

    async def execute_raw(self, *args: Any, **kwargs: Any) -> Any:
        raise NavigationError(
            "execute_raw is not available on a scoped navigator: a raw query cannot be "
            "confined to the scope. Use the unscoped navigator (an administrative tool)."
        )

    execute_raw_one = execute_raw

    # -- scope merging ----------------------------------------------------------

    def _merge(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)

        tenant = out.get("tenant_id")
        if tenant is not None and tenant != self._tenant_id:
            allowed_public = self._include_public and tenant == PUBLIC_TENANT
            if self._tenant_id is not None and not allowed_public:
                raise NavigationError(
                    f"tenant_id={tenant!r} is outside this navigator's scope "
                    f"(tenant_id={self._tenant_id!r})"
                )
        out["tenant_id"] = tenant if tenant is not None else self._tenant_id

        wants_public = out.get("include_public")
        if wants_public and self._tenant_id is not None and not self._include_public:
            raise NavigationError(
                "include_public=True widens this navigator's scope "
                f"(tenant_id={self._tenant_id!r}, include_public=False)"
            )
        out["include_public"] = (
            self._include_public if wants_public is None else bool(wants_public)
        ) and out["tenant_id"] not in (None, PUBLIC_TENANT)

        for kw, fixed in (("created_by_user_id", self._user_ids), ("job_id", self._job_ids)):
            asked = out.get(kw)
            if fixed is None:
                out[kw] = asked
            elif asked is None:
                out[kw] = fixed
            else:
                extra = _as_set(asked) - _as_set(fixed)
                if extra:
                    raise NavigationError(
                        f"{kw}={sorted(extra)} is outside this navigator's scope "
                        f"({sorted(_as_set(fixed))})"
                    )
                out[kw] = asked
        return out

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        try:
            params = inspect.signature(attr).parameters
        except (TypeError, ValueError):
            return attr
        if not all(k in params for k in _SCOPE_KWARGS):
            return attr

        @functools.wraps(attr)
        async def _scoped_call(*args: Any, **kwargs: Any) -> Any:
            return await attr(*args, **self._merge(kwargs))

        return _scoped_call


__all__ = ["ScopedNavigator"]
