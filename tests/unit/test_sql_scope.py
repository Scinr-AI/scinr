"""tests/unit/test_sql_scope.py -- ``sql/scope.py``: every scope combination, both dialects."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from scinr.newton.sql.dialect import PostgresDialect, SqliteDialect
from scinr.newton.sql.scope import scope_conditions
from scinr.newton.sql.tables import model_instances, structure_nodes
from scinr.newton.utils.scope import Scope, make_scope

DIALECTS = [
    pytest.param(PostgresDialect(), postgresql.dialect(), id="postgres"),
    pytest.param(SqliteDialect(), sqlite.dialect(), id="sqlite"),
]


def _sql(conditions, sa_dialect) -> str:
    statement = sa.select(sa.literal(1)).where(*conditions)
    return str(statement.compile(dialect=sa_dialect, compile_kwargs={"render_postcompile": False}))


def _params(conditions, sa_dialect) -> dict:
    return dict(sa.select(sa.literal(1)).where(*conditions).compile(dialect=sa_dialect).params)


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_no_filter_adds_nothing(dialect, sa_dialect) -> None:
    assert scope_conditions(Scope(), structure_nodes, dialect=dialect) == []


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_single_tenant_is_an_equality(dialect, sa_dialect) -> None:
    conditions = scope_conditions(make_scope("acme"), structure_nodes, dialect=dialect)
    assert len(conditions) == 1
    assert "structure_nodes.tenant_id = " in _sql(conditions, sa_dialect)
    assert "acme" in _params(conditions, sa_dialect).values()


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_tenant_with_public_is_an_in(dialect, sa_dialect) -> None:
    conditions = scope_conditions(make_scope("acme", True), structure_nodes, dialect=dialect)
    assert "structure_nodes.tenant_id IN" in _sql(conditions, sa_dialect)


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_scalar_provenance_is_an_in(dialect, sa_dialect) -> None:
    scope = make_scope(None, False, ["u1", "u2"], "j1")
    sql = _sql(scope_conditions(scope, structure_nodes, dialect=dialect), sa_dialect)
    assert "structure_nodes.created_by_user_id IN" in sql
    assert "structure_nodes.job_id IN" in sql


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_array_provenance_overlaps_the_lists(dialect, sa_dialect) -> None:
    scope = make_scope("acme", False, "u1", ["j1", "j2"])
    conditions = scope_conditions(scope, model_instances, kind="array", dialect=dialect)
    sql = _sql(conditions, sa_dialect)
    assert "model_instances.tenant_id = " in sql
    assert "created_by_user_ids" in sql
    assert "job_ids" in sql
    assert "created_by_user_id IN" not in sql


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_tenant_only_ignores_provenance(dialect, sa_dialect) -> None:
    scope = make_scope("acme", False, "u1", "j1")
    sql = _sql(
        scope_conditions(scope, structure_nodes, kind="tenant_only", dialect=dialect), sa_dialect
    )
    assert "tenant_id" in sql
    assert "created_by_user_id" not in sql
    assert "job_id" not in sql


@pytest.mark.parametrize("dialect, sa_dialect", DIALECTS)
def test_tenant_false_leaves_the_tenant_out(dialect, sa_dialect) -> None:
    scope = make_scope("acme", False, None, "j1")
    sql = _sql(scope_conditions(scope, structure_nodes, tenant=False, dialect=dialect), sa_dialect)
    assert "tenant_id" not in sql
    assert "structure_nodes.job_id IN" in sql
