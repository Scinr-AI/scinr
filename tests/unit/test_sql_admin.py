"""
tests/unit/test_sql_admin.py -- Argument checks of the administration API (no server needed).
"""

from __future__ import annotations

import pytest

from scinr.newton import (
    grant_tenant_access,
    list_tenant_access,
    revoke_tenant_access,
    setup_sql_schema,
)
from scinr.newton.sql import admin
from scinr.newton.sql.schema import ensure_schema, rls_enabled

SQLITE = "sqlite:////nonexistent/newton.db"
POSTGRES = "postgresql://nobody:never@127.0.0.1:1/none"


@pytest.mark.parametrize(
    "call",
    [
        lambda: setup_sql_schema(SQLITE),
        lambda: grant_tenant_access(SQLITE, "role_a", "t1"),
        lambda: revoke_tenant_access(SQLITE, "role_a", "t1"),
        lambda: list_tenant_access(SQLITE),
    ],
)
def test_sqlite_is_refused(call) -> None:
    with pytest.raises(ValueError, match="only applies to PostgreSQL"):
        call()


@pytest.mark.parametrize("role", ["", "Role", "a b", "x;drop", "1abc", "a" * 64])
def test_invalid_role_names_are_refused_before_connecting(role) -> None:
    with pytest.raises(ValueError, match="Invalid role name"):
        grant_tenant_access(POSTGRES, role, "t1")
    with pytest.raises(ValueError, match="Invalid role name"):
        setup_sql_schema(POSTGRES, enable_rls=True, runtime_role=role)


def test_tenant_values_are_checked_before_connecting() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        grant_tenant_access(POSTGRES, "role_a", "")
    with pytest.raises(ValueError, match="reserved"):
        grant_tenant_access(POSTGRES, "role_a", "*")


def test_enable_rls_needs_a_runtime_role() -> None:
    with pytest.raises(ValueError, match="requires runtime_role"):
        setup_sql_schema(POSTGRES, enable_rls=True)


def test_the_public_tenant_is_stored_as_a_star() -> None:
    assert admin._tenant_value(None) == "*"
    assert admin._tenant_value("t1") == "t1"


def test_sqlite_never_has_rls(sqlite_url, tmp_path) -> None:
    from scinr.newton import configure
    from scinr.newton.sql.engine import reset_engines

    configure(
        graph_backend="sqlite",
        storage_backend="sqlite",
        freeze_backend="sqlite",
        sql_url=sqlite_url,
    )
    try:
        ensure_schema()
        assert rls_enabled() is False
    finally:
        reset_engines()
