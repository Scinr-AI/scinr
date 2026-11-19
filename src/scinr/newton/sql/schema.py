"""
sql/schema.py — Create, migrate or verify the SQL schema (simple mode and RLS mode).

:func:`ensure_schema` runs once per process and ``(sql_url, sql_schema)`` target. The mode
is detected from the schema itself (decision P6), with no extra setting:

* no schema yet: newton creates it (simple mode);
* schema present, ``rls_enabled`` false: newton migrates it if it is behind;
* schema present, ``rls_enabled`` true: newton only verifies the version and never runs DDL
  (the user of a RLS deployment cannot, and must not, create anything).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from scinr.newton.exceptions import ConfigurationError, GraphConnectionError

__all__ = ["SchemaVersionError", "ensure_schema", "reset_verified", "rls_enabled"]

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_verified: set[tuple[str, str]] = set()
#: Mode of every verified target: ``True`` when ``setup_sql_schema`` marked it as RLS.
_rls_modes: dict[tuple[str, str], bool] = {}

_CANNOT_CREATE = (
    "The SQL user cannot create the schema. Run setup_sql_schema(admin_url, ...) "
    "with an administrator connection, or grant it CREATE on the database."
)


class SchemaVersionError(GraphConnectionError, ConfigurationError):
    """A RLS schema is not at the version this newton needs and the user cannot migrate it."""


def reset_verified() -> None:
    """Forget the schemas already verified in this process (and their mode)."""
    with _lock:
        _verified.clear()
        _rls_modes.clear()


def rls_enabled() -> bool:
    """Whether the verified schema of the configured target is in RLS mode.

    The mode is read from the schema by :func:`ensure_schema`, so it is ``False`` until a
    store, navigator or repository has verified the schema (they all do it on first use).
    Always ``False`` for SQLite.

    In RLS mode Postgres enforces the grants of ``newton_tenant_access`` on top of the
    tenant filter newton always adds. This is **not an error**, it is how RLS works:

    * a read with a ``tenant_id`` that was not granted returns nothing (no exception);
    * a read without ``tenant_id`` sees only the granted tenants;
    * a write into a tenant that was not granted raises
      :class:`~scinr.newton.exceptions.TenantAccessError`;
    * ``delete_document`` of a visible but read-only document raises
      :class:`~scinr.newton.exceptions.TenantAccessError` before deleting anything.
    """
    from scinr.newton.config import get_config

    cfg = get_config()
    with _lock:
        return _rls_modes.get((cfg.sql_url, cfg.sql_schema), False)


def _read_state(connection: Any, dialect: Any, schema: str) -> dict[str, Any] | None:
    """The row of ``newton_schema``, or ``None`` when the table or the row is missing."""
    import sqlalchemy as sa

    from scinr.newton.sql.migrations.runner import CONTROL_TABLE, control_table

    qualified = schema if dialect.name == "postgres" else None
    if not sa.inspect(connection).has_table(CONTROL_TABLE, schema=qualified):
        return None
    row = connection.execute(sa.select(control_table(dialect, schema))).mappings().first()
    return dict(row) if row else None


def ensure_schema() -> None:
    """Create, migrate or verify the SQL schema, once per process and target.

    Uses the synchronous engine.

    Raises:
        ConfigurationError: If the user cannot create the schema or if the schema is newer
            than this newton.
        SchemaVersionError: If a RLS schema is not at the version this newton needs (both a
            :class:`ConfigurationError` and a :class:`GraphConnectionError`).
    """
    from scinr.newton.config import get_config
    from scinr.newton.sql.engine import get_dialect_for_config, get_sync_engine
    from scinr.newton.sql.migrations import runner

    cfg = get_config()
    target = (cfg.sql_url, cfg.sql_schema)
    with _lock:
        if target in _verified:
            return
        dialect = get_dialect_for_config()
        engine = get_sync_engine()
        latest = runner.latest_version(dialect.name)

        with engine.connect() as connection:
            if dialect.name == "sqlite":
                connection = connection.execution_options(sqlite_deferred=True)
            state = _read_state(connection, dialect, cfg.sql_schema)

        def migrate() -> None:
            with engine.begin() as connection:
                runner.apply_migrations(
                    connection,
                    dialect,
                    schema=cfg.sql_schema,
                    fulltext_config=cfg.sql_fulltext_config,
                )

        if state is None:
            try:
                migrate()
            except Exception as exc:
                if dialect.classify_error(exc) == "access_denied":
                    raise ConfigurationError(_CANNOT_CREATE) from exc
                raise
        else:
            version = state["version"]
            if version > latest:
                raise ConfigurationError(
                    f"The SQL schema is at version {version}, newer than this newton "
                    f"(version {latest}); upgrade scinr."
                )
            if state["rls_enabled"]:
                if version != latest:
                    raise SchemaVersionError(
                        f"The SQL schema is at version {version} and this newton needs "
                        f"{latest}: schema version {version} < {latest}; an administrator "
                        "must run setup_sql_schema(admin_url, enable_rls=True, ...) to "
                        "migrate it."
                    )
            elif version < latest:
                migrate()
            if state["fulltext_config"] != cfg.sql_fulltext_config:
                logger.warning(
                    "The SQL schema was created with fulltext config %r; %r is ignored "
                    "until the schema is recreated.",
                    state["fulltext_config"],
                    cfg.sql_fulltext_config,
                )
        _rls_modes[target] = bool(state and state["rls_enabled"])
        _verified.add(target)
