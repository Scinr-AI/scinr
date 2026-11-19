"""
sql/admin.py — Administration of a Postgres schema: setup, RLS and per-tenant grants.

An administrator runs :func:`setup_sql_schema` once (optionally with ``enable_rls=True``) and
:func:`grant_tenant_access` for every role/tenant pair, with a connection that can create
objects and grants (``admin_url``). The runtime process then connects with a restricted role
and newton never needs DDL privileges again.

Newton never creates roles, changes passwords or stores them (decision P8): the runtime role
is created by hand (``CREATE ROLE <r> LOGIN PASSWORD '...'``). Every function here receives
``admin_url``, builds its own short-lived engine, closes it before returning and forgets the
URL: it is never stored in ``Config``, never logged and never part of an error message.

These are one-off administration calls, so there are no ``async`` versions (documented
exception to decision 17 of the functional plan).

RLS semantics (see also :func:`scinr.newton.sql.schema.rls_enabled`): a tenant is granted to a
role in ``newton_tenant_access``; ``tenant_id=None`` (public data, stored as ``__public__``) is
stored as ``'*'`` there.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from scinr.newton.exceptions import ConfigurationError

if TYPE_CHECKING:
    from sqlalchemy import Connection, Engine

    from scinr.newton.sql.dialect import Dialect

__all__ = [
    "SchemaStatus",
    "TenantAccess",
    "grant_tenant_access",
    "list_tenant_access",
    "revoke_tenant_access",
    "setup_sql_schema",
]

logger = logging.getLogger(__name__)

#: ``newton_tenant_access.tenant_id`` that stands for public data.
PUBLIC_GRANT = "*"
ACCESS_TABLE = "newton_tenant_access"
_ROLE_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_NOT_POSTGRES = (
    "setup_sql_schema only applies to PostgreSQL; SQLite schemas are created automatically"
)
_NO_RLS = "RLS is not enabled; run setup_sql_schema(enable_rls=True)"
_URL_SECRET = re.compile(r"://[^:/@\s]*:([^@\s]+)@")


@dataclass(frozen=True)
class SchemaStatus:
    """State of a schema after :func:`setup_sql_schema`."""

    version: int
    rls_enabled: bool
    rls_version: int
    fulltext_config: str


@dataclass(frozen=True)
class TenantAccess:
    """One grant of a role on a tenant (``tenant_id`` is ``None`` for public data)."""

    role: str
    tenant_id: str | None
    can_write: bool
    granted_at: datetime


def _validate_role(role: str) -> str:
    if not isinstance(role, str) or not _ROLE_RE.match(role):
        raise ValueError(f"Invalid role name {role!r}: expected ^[a-z_][a-z0-9_]{{0,62}}$.")
    return role


def _configured(name: str, default: str) -> str:
    """A ``Config`` value; the library default when ``configure()`` was never called.

    An administrator may run these functions from a script that never configures newton.
    """
    from scinr.newton.config import get_config

    try:
        return getattr(get_config(), name)
    except ConfigurationError:
        return default


def _require_postgres(admin_url: str) -> None:
    """Refuse a non-PostgreSQL URL before anything else is looked at."""
    from scinr.newton.sql.dialect import get_dialect

    if get_dialect(admin_url).name != "postgres":
        raise ValueError(_NOT_POSTGRES)


def _resolve_schema(schema: str | None, admin_url: str) -> str:
    _require_postgres(admin_url)
    return schema or _configured("sql_schema", "newton")


def _scrub(message: str, admin_url: str) -> str:
    """Remove the password of *admin_url* from *message*."""
    match = _URL_SECRET.search(admin_url)
    if match and match.group(1):
        message = message.replace(match.group(1), "***")
    return message


@contextmanager
def _admin_engine(
    admin_url: str, schema: str, fulltext_config: str
) -> Iterator[tuple[Engine, Dialect]]:
    """A short-lived engine for *admin_url*; errors leave as ``scinr`` exceptions, scrubbed.

    Raises:
        ValueError: If *admin_url* is not a PostgreSQL URL or *schema* is not an identifier.
        ConfigurationError: If the SQL extras are not installed.
    """
    from scinr.newton.sql.dialect import get_dialect
    from scinr.newton.sql.errors import translate

    dialect = get_dialect(admin_url, fulltext_config=fulltext_config)
    if dialect.name != "postgres":
        raise ValueError(_NOT_POSTGRES)
    dialect.validate_identifier(schema)
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool
    except ImportError as exc:
        raise ConfigurationError(
            "The SQL backends need the 'postgres' or 'sqlite' extra: pip install scinr[postgres]"
        ) from exc
    engine = create_engine(dialect.sync_url(admin_url), poolclass=NullPool)
    try:
        yield engine, dialect
    except Exception as exc:
        translated = translate(exc, dialect=dialect)
        if translated is None:
            raise
        raise type(translated)(_scrub(str(translated), admin_url)) from None
    finally:
        engine.dispose()


def _role_grants(schema: str, role: str) -> list[str]:
    """The statements that give the runtime *role* what it needs (idempotent)."""
    s, r = schema, role  # both validated as plain identifiers
    return [
        f"GRANT USAGE ON SCHEMA {s} TO {r}",
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {s} TO {r}",
        f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {s} TO {r}",
        f"REVOKE INSERT, UPDATE, DELETE ON {s}.{ACCESS_TABLE} FROM {r}",
        f"REVOKE INSERT, UPDATE, DELETE ON {s}.newton_schema FROM {r}",
        f"GRANT EXECUTE ON FUNCTION {s}.newton_ensure_instance_key_index(TEXT) TO {r}",
        f"GRANT EXECUTE ON FUNCTION {s}.newton_can_read(TEXT), {s}.newton_can_write(TEXT) TO {r}",
    ]


def _status(connection: Connection, schema: str) -> SchemaStatus:
    import sqlalchemy as sa

    table = sa.table(
        "newton_schema",
        sa.column("version", sa.Integer),
        sa.column("rls_enabled", sa.Boolean),
        sa.column("rls_version", sa.Integer),
        sa.column("fulltext_config", sa.Text),
        schema=schema,
    )
    row = connection.execute(sa.select(table)).mappings().one()
    return SchemaStatus(
        version=row["version"],
        rls_enabled=bool(row["rls_enabled"]),
        rls_version=row["rls_version"],
        fulltext_config=row["fulltext_config"],
    )


def setup_sql_schema(
    admin_url: str,
    *,
    enable_rls: bool = False,
    runtime_role: str | None = None,
    schema: str | None = None,
    fulltext_config: str | None = None,
) -> SchemaStatus:
    """Create or migrate the schema as an administrator, optionally in RLS mode. Idempotent.

    With ``enable_rls=True`` it also applies the Row-Level-Security policies and gives
    *runtime_role* the privileges it needs (a second role: call again with the other role;
    only its grants are added). Grants on tenants are separate:
    :func:`grant_tenant_access`. A schema that is already in RLS mode stays so (turning RLS
    off is not supported).

    Args:
        admin_url: PostgreSQL URL of a user that can create objects. Used and forgotten.
        enable_rls: Turn on RLS mode.
        runtime_role: The restricted role newton connects with. Required with ``enable_rls``;
            it must already exist (``CREATE ROLE <role> LOGIN PASSWORD '...'``).
        schema: Postgres schema. Default: ``Config.sql_schema``.
        fulltext_config: Text-search configuration used when the schema is created.
            Default: ``Config.sql_fulltext_config``.

    Returns:
        The state of the schema.

    Raises:
        ValueError: If *admin_url* is not PostgreSQL, ``enable_rls`` lacks a valid
            *runtime_role*, or the role does not exist.
        GraphError: If the database fails (the message never contains the URL).
    """
    from scinr.newton.sql.migrations import runner

    schema = _resolve_schema(schema, admin_url)
    fulltext = fulltext_config or _configured("sql_fulltext_config", "simple")
    if enable_rls:
        if runtime_role is None:
            raise ValueError("enable_rls=True requires runtime_role.")
        _validate_role(runtime_role)
    with _admin_engine(admin_url, schema, fulltext) as (engine, dialect):
        import sqlalchemy as sa

        with engine.begin() as connection:
            if enable_rls:
                roles = sa.table("pg_roles", sa.column("rolname", sa.Text))
                exists = connection.execute(
                    sa.select(sa.literal(1))
                    .select_from(roles)
                    .where(roles.c.rolname == runtime_role)
                ).first()
                if exists is None:
                    raise ValueError(
                        f"create the role first: CREATE ROLE {runtime_role} LOGIN PASSWORD '...'"
                    )
            runner.apply_migrations(connection, dialect, schema=schema, fulltext_config=fulltext)
            if enable_rls:
                runner.apply_rls_migrations(connection, dialect, schema=schema)
                assert runtime_role is not None
                for statement in _role_grants(schema, runtime_role):
                    connection.exec_driver_sql(dialect.verbatim(statement))
            status = _status(connection, schema)
        if status.rls_enabled and not enable_rls:
            logger.warning(
                "Schema %r is in RLS mode; setup_sql_schema(enable_rls=False) does not turn "
                "it off (not supported).",
                schema,
            )
    return status


def _tenant_value(tenant_id: str | None) -> str:
    if tenant_id is None:
        return PUBLIC_GRANT
    if tenant_id == "":
        raise ValueError("tenant_id must be a non-empty string, or None for public data.")
    if tenant_id == PUBLIC_GRANT:
        raise ValueError("'*' is reserved for public data: pass tenant_id=None.")
    return tenant_id


def _access_table(schema: str) -> Any:
    import sqlalchemy as sa

    return sa.table(
        ACCESS_TABLE,
        sa.column("role_name", sa.Text),
        sa.column("tenant_id", sa.Text),
        sa.column("can_write", sa.Boolean),
        sa.column("granted_at", sa.DateTime(timezone=True)),
        schema=schema,
    )


def _require_rls(connection: Connection, schema: str) -> None:
    if not _status(connection, schema).rls_enabled:
        raise ValueError(_NO_RLS)


def grant_tenant_access(
    admin_url: str,
    role: str,
    tenant_id: str | None,
    *,
    write: bool = False,
    schema: str | None = None,
) -> None:
    """Let *role* read (and, with ``write=True``, write) the data of *tenant_id*.

    ``tenant_id=None`` grants the public data. Granting again changes the write flag.

    Raises:
        ValueError: If the role or tenant is invalid, *admin_url* is not PostgreSQL, or the
            schema is not in RLS mode (``RLS is not enabled; run setup_sql_schema(enable_rls=True)``).
        GraphError: If the database fails.
    """
    _validate_role(role)
    tenant = _tenant_value(tenant_id)
    schema = _resolve_schema(schema, admin_url)
    with _admin_engine(admin_url, schema, "simple") as (engine, _dialect):
        from sqlalchemy.dialects import postgresql

        table = _access_table(schema)
        with engine.begin() as connection:
            _require_rls(connection, schema)
            insert = postgresql.insert(table).values(
                role_name=role, tenant_id=tenant, can_write=write
            )
            connection.execute(
                insert.on_conflict_do_update(
                    index_elements=["role_name", "tenant_id"],
                    set_={"can_write": insert.excluded.can_write},
                )
            )


def revoke_tenant_access(
    admin_url: str,
    role: str,
    tenant_id: str | None,
    *,
    schema: str | None = None,
) -> None:
    """Remove the grant of *role* on *tenant_id* (``None``: public). Revoking nothing is fine.

    Raises:
        ValueError: If the role or tenant is invalid, *admin_url* is not PostgreSQL, or the
            schema is not in RLS mode.
        GraphError: If the database fails.
    """
    _validate_role(role)
    tenant = _tenant_value(tenant_id)
    schema = _resolve_schema(schema, admin_url)
    with _admin_engine(admin_url, schema, "simple") as (engine, _dialect):
        import sqlalchemy as sa

        table = _access_table(schema)
        with engine.begin() as connection:
            _require_rls(connection, schema)
            connection.execute(
                sa.delete(table).where(table.c.role_name == role, table.c.tenant_id == tenant)
            )


def list_tenant_access(
    admin_url: str,
    *,
    role: str | None = None,
    schema: str | None = None,
) -> list[TenantAccess]:
    """The grants of *role* (or of every role), ordered by role and tenant.

    Raises:
        ValueError: If *admin_url* is not PostgreSQL, *role* is invalid, or the schema is not
            in RLS mode.
        GraphError: If the database fails.
    """
    if role is not None:
        _validate_role(role)
    schema = _resolve_schema(schema, admin_url)
    with _admin_engine(admin_url, schema, "simple") as (engine, _dialect):
        import sqlalchemy as sa

        table = _access_table(schema)
        statement = sa.select(table).order_by(table.c.role_name, table.c.tenant_id)
        if role is not None:
            statement = statement.where(table.c.role_name == role)
        with engine.connect() as connection:
            _require_rls(connection, schema)
            rows = connection.execute(statement).mappings().all()
    return [
        TenantAccess(
            role=row["role_name"],
            tenant_id=None if row["tenant_id"] == PUBLIC_GRANT else row["tenant_id"],
            can_write=bool(row["can_write"]),
            granted_at=row["granted_at"],
        )
        for row in rows
    ]
