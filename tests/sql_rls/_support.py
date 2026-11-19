"""
tests/sql_rls/_support.py -- Provisioning of a throw-away RLS schema with restricted roles.

The roles are created here, by the test: newton itself never creates roles (decision P8).
Everything is created inside the disposable ``postgres-test`` container (see
``tests/sql/conftest.py``) and removed afterwards: the schema is named ``newton_test_rls_*``
(so the shared helper may drop it) and the roles ``newton_rt_*``.
"""

from __future__ import annotations

import atexit
import os
import secrets
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

from scinr.newton.sql.admin import grant_tenant_access, setup_sql_schema
from tests.sql.conftest import WIPE_SCHEMA_PREFIX, drop_pg_schema


@dataclass(frozen=True)
class RlsRole:
    """A restricted role created for a test."""

    name: str
    password: str


@dataclass(frozen=True)
class RlsDb:
    """A schema in RLS mode and the two restricted roles that can use it."""

    admin_url: str
    schema: str
    role_a: RlsRole
    role_b: RlsRole

    def url_for(self, role: RlsRole, password: str | None = None) -> str:
        """The ``sql_url`` of *role* (same host and database as the admin URL)."""
        return with_credentials(self.admin_url, role.name, password or role.password)


def with_credentials(url: str, user: str, password: str) -> str:
    """*url* with another user and password."""
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    netloc = f"{user}:{password}@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


def create_role(admin_url: str, name: str) -> RlsRole:
    """``CREATE ROLE <name> LOGIN PASSWORD '<random>'`` (what an operator does by hand)."""
    role = RlsRole(name=name, password=secrets.token_hex(12))
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(name), sql.Literal(role.password)
            )
        )
    return role


def drop_role(admin_url: str, name: str) -> None:
    """Drop a ``newton_rt_*`` role (nothing else is ever dropped)."""
    assert name.startswith("newton_rt_"), f"refusing to drop role {name!r}"
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(name)))


def provision(admin_url: str) -> RlsDb:
    """Create the schema in RLS mode and the two roles (no tenant is granted yet)."""
    token = uuid.uuid4().hex[:8]
    schema = f"{WIPE_SCHEMA_PREFIX}rls_{token}"
    role_a = create_role(admin_url, f"newton_rt_{token}_a")
    role_b = create_role(admin_url, f"newton_rt_{token}_b")
    db = RlsDb(admin_url=admin_url, schema=schema, role_a=role_a, role_b=role_b)
    setup_sql_schema(admin_url, enable_rls=True, runtime_role=role_a.name, schema=schema)
    setup_sql_schema(admin_url, enable_rls=True, runtime_role=role_b.name, schema=schema)
    return db


def teardown(db: RlsDb) -> None:
    """Drop the schema (its grants go with it) and the roles."""
    drop_pg_schema(db.admin_url, db.schema)
    drop_role(db.admin_url, db.role_a.name)
    drop_role(db.admin_url, db.role_b.name)


def provision_runtime_role(admin_url: str, schema: str, *, tenants: tuple[str | None, ...]) -> str:
    """Put *schema* in RLS mode for the contract suite and return the restricted role's URL.

    One role per process (``newton_rt_contract_<pid>``), with write access to *tenants*
    (``None``: public data). The schema is recreated by the caller before every test, so the
    role is created once and its password renewed each time; both are dropped at exit.
    """
    name = f"newton_rt_contract_{os.getpid()}"
    password = secrets.token_hex(12)
    with psycopg.connect(admin_url, autocommit=True) as connection:
        exists = connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (name,)).fetchone()
        verb = "ALTER" if exists else "CREATE"
        connection.execute(
            sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(name), sql.Literal(password)
            )
        )
    if not exists:
        atexit.register(_drop_contract_role, admin_url, schema, name)
    setup_sql_schema(admin_url, enable_rls=True, runtime_role=name, schema=schema)
    for tenant in tenants:
        grant_tenant_access(admin_url, name, tenant, write=True, schema=schema)
    return with_credentials(admin_url, name, password)


def _drop_contract_role(admin_url: str, schema: str, name: str) -> None:
    drop_pg_schema(admin_url, schema)
    drop_role(admin_url, name)
