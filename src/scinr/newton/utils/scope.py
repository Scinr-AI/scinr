"""
utils/scope.py — The read-side scope: tenant, user and job filters.

Shared by the graph navigation API (``navigation.scope``) and the storage
repositories (``storage.base``), so both layers accept the same four
keyword-only arguments with the same semantics and the same errors::

    tenant_id: str | None = None                            # None = all tenants
    include_public: bool = False                            # add the public documents
    created_by_user_id: str | Sequence[str] | None = None   # IN
    job_id: str | Sequence[str] | None = None               # IN

They are turned into a :class:`Scope` by :func:`make_scope`; each backend then
renders it as plain, parametrized predicates on the denormalized properties
written at ingestion (see ``utils/tenancy.py``) — Cypher via
:meth:`Scope.clauses`, MongoDB via ``storage.filters.scope_to_mongo``.

===============  ===============  ===========================================
``tenant_id``    ``include_public``  Predicate on the node / record
===============  ===============  ===========================================
``None``         any              none (all tenants, legacy records included)
``"__public__"`` any              ``tenant_id = '__public__'``
``"acme"``       ``False``        ``tenant_id = 'acme'``
``"acme"``       ``True``         ``tenant_id IN ['acme', '__public__']``
``""``           —                :class:`~scinr.newton.exceptions.ScopeError`
===============  ===============  ===========================================

``created_by_user_id`` / ``job_id`` accept one value or several (``IN``): OR
inside each filter, AND between filters and with the tenant. An empty list is
an error (it would be ambiguous between "no filter" and "match nothing").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from scinr.newton.exceptions import ScopeError
from scinr.newton.utils.tenancy import PUBLIC_TENANT

#: How the provenance of a node is stored, which decides the predicate:
#:  * ``"scalar"``      – ``job_id`` / ``created_by_user_id`` (one value per node)
#:  * ``"array"``       – ``job_ids`` / ``created_by_user_ids`` (merged nodes)
#:  * ``"tenant_only"`` – only the tenant predicate (anchors)
ScopeKind = Literal["scalar", "array", "tenant_only"]


def _values(name: str, value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = (value,) if isinstance(value, str) else tuple(value)
    if not items:
        raise ScopeError(f"{name} must not be an empty list (omit it for no filter)")
    for item in items:
        if not isinstance(item, str) or not item:
            raise ScopeError(f"{name} values must be non-empty strings, got {item!r}")
    return items


@dataclass(frozen=True)
class Scope:
    """Resolved read scope. ``None`` in any field means "no filter"."""

    tenants: tuple[str, ...] | None = None
    user_ids: tuple[str, ...] | None = None
    job_ids: tuple[str, ...] | None = None
    #: Kept only for the single-anchor shadowing rule (a tenant's document
    #: shadows the public one at the same path).
    include_public: bool = False

    @property
    def is_unfiltered(self) -> bool:
        return self.tenants is None and self.user_ids is None and self.job_ids is None

    def params(self, *, tenant: bool = True, provenance: bool = True) -> dict[str, Any]:
        """Query parameters for :meth:`clauses` (only the ones in use)."""
        out: dict[str, Any] = {}
        if tenant and self.tenants is not None:
            if len(self.tenants) == 1:
                out["scope_tenant"] = self.tenants[0]
            else:
                out["scope_tenants"] = list(self.tenants)
        if provenance:
            if self.user_ids is not None:
                out["scope_user_ids"] = list(self.user_ids)
            if self.job_ids is not None:
                out["scope_job_ids"] = list(self.job_ids)
        return out

    def tenant_clause(self, alias: str) -> str | None:
        if self.tenants is None:
            return None
        if len(self.tenants) == 1:
            return f"{alias}.tenant_id = $scope_tenant"
        return f"{alias}.tenant_id IN $scope_tenants"

    def provenance_clauses(self, alias: str, kind: ScopeKind = "scalar") -> list[str]:
        out: list[str] = []
        if kind == "tenant_only":
            return out
        if kind == "array":
            if self.user_ids is not None:
                out.append(
                    f"any(x IN $scope_user_ids WHERE x IN coalesce({alias}.created_by_user_ids, []))"
                )
            if self.job_ids is not None:
                out.append(f"any(x IN $scope_job_ids WHERE x IN coalesce({alias}.job_ids, []))")
        else:
            if self.user_ids is not None:
                out.append(f"{alias}.created_by_user_id IN $scope_user_ids")
            if self.job_ids is not None:
                out.append(f"{alias}.job_id IN $scope_job_ids")
        return out

    def clauses(
        self, alias: str, kind: ScopeKind = "scalar", *, tenant: bool = True
    ) -> list[str]:
        """``WHERE`` fragments for *alias*; parameters come from :meth:`params`."""
        out: list[str] = []
        if tenant:
            t = self.tenant_clause(alias)
            if t:
                out.append(t)
        out.extend(self.provenance_clauses(alias, kind))
        return out

    def with_tenant(self, tenant: str) -> Scope:
        """Same scope pinned to a single (already stored) tenant."""
        return Scope(
            tenants=(tenant,),
            user_ids=self.user_ids,
            job_ids=self.job_ids,
            include_public=False,
        )


def make_scope(
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> Scope:
    """Validate the four public filters and resolve them into a :class:`Scope`.

    Raises:
        ScopeError: *tenant_id* is empty / not a string, or a list filter
            is empty.
    """
    tenants: tuple[str, ...] | None
    if tenant_id is None:
        tenants = None
    elif not isinstance(tenant_id, str) or tenant_id == "":
        raise ScopeError(
            "tenant_id must be a non-empty string, '__public__' or None (None = all tenants)"
        )
    elif tenant_id == PUBLIC_TENANT:
        tenants = (PUBLIC_TENANT,)
    elif include_public:
        tenants = (tenant_id, PUBLIC_TENANT)
    else:
        tenants = (tenant_id,)
    return Scope(
        tenants=tenants,
        user_ids=_values("created_by_user_id", created_by_user_id),
        job_ids=_values("job_id", job_id),
        include_public=bool(include_public),
    )


__all__ = ["Scope", "ScopeKind", "make_scope"]
