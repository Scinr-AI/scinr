"""
navigation/scope.py — The read-side scope of the navigation API.

Every non-catalog navigation method takes four keyword-only arguments::

    tenant_id: str | None = None                            # None = all tenants
    include_public: bool = False                            # add the public documents
    created_by_user_id: str | Sequence[str] | None = None   # IN
    job_id: str | Sequence[str] | None = None               # IN

The scope itself (:class:`Scope`, :func:`make_scope`, the semantics table and
the errors) lives in :mod:`scinr.newton.utils.scope`, shared with the storage
repositories so both layers filter identically; this module re-exports it and
adds :func:`resolve_selector` for document selectors.
"""

from __future__ import annotations

from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.utils.scope import Scope, ScopeKind, make_scope


def resolve_selector(document: Any, scope: Scope) -> tuple[str, Scope]:
    """Return ``(path, scope)`` for a document selector (a path or a ``DocumentRef``).

    A ``DocumentRef`` carries its own tenant, which is authoritative: the scope
    is pinned to it, and an explicit *tenant_id* that excludes it is an error.
    """
    from scinr.newton.navigation.models import DocumentRef

    if isinstance(document, DocumentRef):
        ref_tenant = document.tenant_id
        if ref_tenant is not None:
            if scope.tenants is not None and ref_tenant not in scope.tenants:
                raise NavigationError(
                    f"document {document.path!r} belongs to tenant {ref_tenant!r}, "
                    f"which is outside the requested scope {list(scope.tenants)}"
                )
            scope = scope.with_tenant(ref_tenant)
        return document.path, scope
    if isinstance(document, str):
        return document, scope
    raise TypeError(
        f"document selector must be a path str or DocumentRef, got {type(document).__name__}"
    )


__all__ = ["Scope", "ScopeKind", "make_scope", "resolve_selector"]
