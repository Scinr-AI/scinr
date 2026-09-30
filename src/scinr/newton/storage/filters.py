"""
storage/filters.py — Render the shared read scope as a MongoDB filter.

The scope (``tenant_id`` / ``include_public`` / ``created_by_user_id`` /
``job_id``) is validated and resolved by :func:`scinr.newton.utils.scope.make_scope`
— the very same function the graph navigation API uses — so storage reads and
deletes filter exactly like the navigation (``None`` = no filter, a list = ``IN``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scinr.newton.utils.scope import Scope, make_scope


def scope_to_mongo(scope: Scope) -> dict[str, Any]:
    """Return the MongoDB filter fragment for *scope* (``{}`` when unfiltered).

    Records written before multi-tenancy have no ``tenant_id``: any tenant
    filter excludes them, so they are only reachable with ``tenant_id=None``.
    """
    out: dict[str, Any] = {}
    if scope.tenants is not None:
        out["tenant_id"] = {"$in": list(scope.tenants)}
    if scope.user_ids is not None:
        out["created_by_user_id"] = {"$in": list(scope.user_ids)}
    if scope.job_ids is not None:
        out["job_id"] = {"$in": list(scope.job_ids)}
    return out


def mongo_scope_filter(
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the four public filters and render them for MongoDB.

    Raises:
        ScopeError: *tenant_id* is empty, or a list filter is empty.
    """
    return scope_to_mongo(make_scope(tenant_id, include_public, created_by_user_id, job_id))


__all__ = ["mongo_scope_filter", "scope_to_mongo"]
