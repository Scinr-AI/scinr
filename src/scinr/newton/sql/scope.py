"""
sql/scope.py — Render a :class:`~scinr.newton.utils.scope.Scope` as SQLAlchemy conditions.

The SQL twin of ``Scope.clauses()`` (Cypher) and ``storage.filters.scope_to_mongo``: the same
four public filters (tenant, ``include_public``, user, job) become parametrised predicates on
the provenance columns written at ingestion. Used by the storage repositories and, later, by
the SQL navigator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scinr.newton.utils.scope import Scope, ScopeKind

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.sql.elements import ColumnElement

    from scinr.newton.sql.dialect import Dialect

__all__ = ["scope_conditions"]


def scope_conditions(
    scope: Scope,
    table: Table,
    *,
    kind: ScopeKind = "scalar",
    tenant: bool = True,
    dialect: Dialect,
) -> list[ColumnElement[bool]]:
    """Conditions of *scope* on *table* (an empty list when nothing is filtered).

    Args:
        scope: The resolved scope; a ``None`` field adds no condition.
        table: The table (or alias) to filter.
        kind: ``"scalar"``: ``created_by_user_id`` / ``job_id`` hold one value (``IN``);
            ``"array"``: ``created_by_user_ids`` / ``job_ids`` hold lists (overlap);
            ``"tenant_only"``: only the tenant condition.
        tenant: ``False`` to leave the tenant condition out.
        dialect: Used for the list-overlap predicate of ``kind="array"``.
    """
    conditions: list[ColumnElement[bool]] = []
    if tenant and scope.tenants is not None:
        if len(scope.tenants) == 1:
            conditions.append(table.c.tenant_id == scope.tenants[0])
        else:
            conditions.append(table.c.tenant_id.in_(scope.tenants))
    if kind == "tenant_only":
        return conditions
    if kind == "array":
        if scope.user_ids is not None:
            conditions.append(dialect.list_overlaps(table.c.created_by_user_ids, scope.user_ids))
        if scope.job_ids is not None:
            conditions.append(dialect.list_overlaps(table.c.job_ids, scope.job_ids))
    else:
        if scope.user_ids is not None:
            conditions.append(table.c.created_by_user_id.in_(scope.user_ids))
        if scope.job_ids is not None:
            conditions.append(table.c.job_id.in_(scope.job_ids))
    return conditions
