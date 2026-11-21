"""
tests/sql_rls/test_execute_raw_single_statement.py -- ``execute_raw`` runs ONE statement and
cannot leave its ``READ ONLY`` transaction, with real organization roles (PostgreSQL only).

The finding: with no bound parameter psycopg sends the text with the simple query protocol,
which runs every statement of a script. ``COMMIT; BEGIN READ WRITE; INSERT ...`` then ended the
``READ ONLY`` transaction opened by ``execute_raw`` and wrote the role's own tenant (RLS only
stops the other tenants: the role has ``write=True`` on its own).

Each test creates its own schema (``newton_test_single_*``) in RLS mode and two organization
roles (``test_org_*``), each with write access to its tenant; everything is dropped afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql

from scinr.newton import configure
from scinr.newton.exceptions import NavigationError
from scinr.newton.sql.admin import grant_tenant_access, setup_sql_schema
from scinr.newton.sql.engine import reset_engines
from tests.sql.conftest import WIPE_SCHEMA_PREFIX, _skip_unless_pg_wipeable, drop_pg_schema

from ._support import RlsRole, create_role, with_credentials

pytestmark = pytest.mark.sql_rls

_ROLE_PREFIX = "test_org_"
TENANT_A, TENANT_B = "org_a", "org_b"


@dataclass(frozen=True)
class OrgDb:
    admin_url: str
    schema: str
    role_a: RlsRole
    role_b: RlsRole


def _drop_role(admin_url: str, name: str) -> None:
    assert name.startswith(_ROLE_PREFIX), f"refusing to drop role {name!r}"
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(name)))


@pytest.fixture
def org_db() -> Iterator[OrgDb]:
    """An RLS schema and two organization roles, each one able to write its own tenant."""
    from scinr.newton.graph import reset_graph_store

    admin_url = _skip_unless_pg_wipeable()
    token = uuid.uuid4().hex[:8]
    schema = f"{WIPE_SCHEMA_PREFIX}single_{token}"
    created: list[RlsRole] = []
    try:
        for suffix in ("a", "b"):
            created.append(create_role(admin_url, f"{_ROLE_PREFIX}{token}_{suffix}"))
        role_a, role_b = created
        for role, tenant in ((role_a, TENANT_A), (role_b, TENANT_B)):
            setup_sql_schema(admin_url, enable_rls=True, runtime_role=role.name, schema=schema)
            grant_tenant_access(admin_url, role.name, tenant, write=True, schema=schema)
        yield OrgDb(admin_url=admin_url, schema=schema, role_a=role_a, role_b=role_b)
    finally:
        reset_graph_store()
        reset_engines()
        drop_pg_schema(admin_url, schema)
        for role in created:
            _drop_role(admin_url, role.name)


@pytest.fixture
async def nav_a(org_db: OrgDb) -> AsyncIterator[object]:
    """A connected ``SqlGraphNavigator`` running as the role of organization A."""
    from scinr.newton.graph import reset_graph_store
    from scinr.newton.navigation.sql.navigator import SqlGraphNavigator

    reset_graph_store()
    reset_engines()
    configure(
        graph_backend="postgres",
        storage_backend="postgres",
        freeze_backend="postgres",
        sql_url=with_credentials(org_db.admin_url, org_db.role_a.name, org_db.role_a.password),
        sql_schema=org_db.schema,
    )
    navigator = SqlGraphNavigator()
    await navigator.connect()
    try:
        yield navigator
    finally:
        await navigator.close()


def _rows_named(db: OrgDb, uid: str) -> int:
    with psycopg.connect(db.admin_url) as connection:
        row = connection.execute(
            sql.SQL("SELECT count(*) FROM {}.entities WHERE uid = %s").format(
                sql.Identifier(db.schema)
            ),
            (uid,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _script(schema: str, tenant: str, *, commit: bool = True) -> str:
    insert = f"INSERT INTO {schema}.entities (uid, tenant_id) VALUES ('script', '{tenant}')"
    return f"COMMIT; BEGIN READ WRITE; {insert}" + ("; COMMIT" if commit else "")


async def test_the_role_can_write_its_own_tenant_outside_execute_raw(org_db) -> None:
    """Precondition of the finding: the organization role is not read-only by itself."""
    url = with_credentials(org_db.admin_url, org_db.role_a.name, org_db.role_a.password)
    with psycopg.connect(url) as connection:
        connection.execute(
            sql.SQL("INSERT INTO {}.entities (uid, tenant_id) VALUES ('direct', %s)").format(
                sql.Identifier(org_db.schema)
            ),
            (TENANT_A,),
        )
        connection.rollback()  # nothing is kept: only the permission is shown
    assert _rows_named(org_db, "direct") == 0


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda s: _script(s, TENANT_A), id="commit-begin-insert-commit"),
        pytest.param(lambda s: _script(s, TENANT_A, commit=False), id="no-final-commit"),
        pytest.param(lambda s: "SELECT 1; " + _script(s, TENANT_A), id="after-a-select"),
        pytest.param(lambda s: "/* x */ " + _script(s, TENANT_A), id="leading-comment"),
        pytest.param(lambda s: _script(s, TENANT_A) + " -- tail", id="trailing-comment"),
        pytest.param(lambda s: _script(s, TENANT_B), id="other-tenant"),
    ],
)
async def test_a_script_cannot_leave_the_read_only_transaction(org_db, nav_a, build) -> None:
    with pytest.raises(NavigationError):
        await nav_a.execute_raw(build(org_db.schema))
    assert _rows_named(org_db, "script") == 0
    # the connection pool is healthy and still read-only
    assert await nav_a.execute_raw_one("SELECT current_setting('transaction_read_only') AS ro") == {
        "ro": "on"
    }


async def test_several_statements_are_refused_by_the_server(org_db, nav_a) -> None:
    """Statements that pass the keyword check are still refused: the extended protocol only
    takes one statement (SQLSTATE 42601), whatever comments or quoting the text uses."""
    script = _script(org_db.schema, TENANT_A)
    for query in (
        "SELECT 1; SELECT 2",
        f"SELECT 1;\n{script}",
        f"SELECT $$;$$; {script}",
        f"WITH x AS (SELECT 1) SELECT * FROM x; {script}",
    ):
        with pytest.raises(NavigationError) as caught:
            await nav_a.execute_raw(query)
        assert caught.value.code == "42601", query
    assert _rows_named(org_db, "script") == 0


@pytest.mark.parametrize(
    "statement",
    [
        "COMMIT",
        "commit",
        "END",
        "ROLLBACK",
        "ABORT",
        "BEGIN",
        "START TRANSACTION READ WRITE",
        "SAVEPOINT s",
        "SET TRANSACTION READ WRITE",
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE",
        "SET ROLE NONE",
        "RESET ROLE",
        "RESET ALL",
        "SET search_path TO public",
        "DISCARD PLANS",
        "DEALLOCATE ALL",
        "PREPARE p AS SELECT 1",
        "LISTEN channel",
        "DO $$ BEGIN PERFORM 1; END $$",
        "/* x */ COMMIT",
        "-- a comment\nCOMMIT",
        "/* /* nested */ still a comment */ COMMIT",
    ],
)
async def test_transaction_and_session_statements_are_refused(nav_a, statement) -> None:
    with pytest.raises(NavigationError, match="execute_raw"):
        await nav_a.execute_raw(statement)
    assert await nav_a.execute_raw_one("SELECT 1 AS one") == {"one": 1}


async def test_reads_keep_working(org_db, nav_a) -> None:
    assert await nav_a.execute_raw("SELECT 1 AS one;") == [{"one": 1}]
    assert await nav_a.execute_raw("  -- why\nSELECT 1 AS one -- trailing") == [{"one": 1}]
    assert await nav_a.execute_raw("/* a */ (SELECT 1 AS one)") == [{"one": 1}]
    assert await nav_a.execute_raw("VALUES (1)") == [{"column1": 1}]
    assert await nav_a.execute_raw("SELECT ';' AS s, 'a:b'::text AS t") == [{"s": ";", "t": "a:b"}]
    assert await nav_a.execute_raw("SELECT CAST(:n AS integer) + 1 AS n", {"n": 41}) == [{"n": 42}]
    assert await nav_a.execute_raw("SHOW transaction_read_only") == [
        {"transaction_read_only": "on"}
    ]
    rows = await nav_a.execute_raw("SELECT count(*) AS n FROM entities")
    assert rows == [{"n": 0}]
    # graphApi's own marker (a bound parameter in a trailing comment) is still accepted
    marked = await nav_a.execute_raw(
        "SELECT 1 AS one\n-- :graphapi_single_statement", {"graphapi_single_statement": 1}
    )
    assert marked == [{"one": 1}]
