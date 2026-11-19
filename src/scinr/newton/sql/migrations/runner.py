"""
sql/migrations/runner.py — Own migration runner (no Alembic).

Migrations are files ``NNNN_description.sql`` (four digits, consecutive from ``0001``) in
``migrations/<dialect>/``. Each file holds statements separated by a line that contains only
``-- statement`` (never split on ``;``: Postgres functions contain it). The markers
``{schema}`` (Postgres schema) and ``{fulltext_config}`` are substituted before execution;
both values are validated as identifiers.

Statements run through ``exec_driver_sql`` without parameters, so they reach the driver
verbatim (a literal ``%`` or ``:`` needs no escaping).

The applied version lives in the ``newton_schema`` control table, created by the runner
itself (not by a migration). The Postgres Row-Level-Security files form a second track
(``migrations/postgres/rls/``, same naming) whose version lives in ``newton_schema.rls_version``;
:func:`apply_rls_migrations` applies it, after the normal track.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.schema import CreateSchema

from scinr.newton.exceptions import ConfigurationError

if TYPE_CHECKING:
    from scinr.newton.sql.dialect import Dialect

__all__ = [
    "CONTROL_TABLE",
    "STATEMENT_SEPARATOR",
    "RLS_FOLDER",
    "apply_migrations",
    "apply_rls_migrations",
    "control_table",
    "latest_version",
    "list_migrations",
    "migrations_dir",
    "split_statements",
]

CONTROL_TABLE = "newton_schema"
STATEMENT_SEPARATOR = "-- statement"
#: Sub-folder of ``migrations/postgres/`` with the Row-Level-Security track.
RLS_FOLDER = "rls"

_ROOT = Path(__file__).parent
_FILE_RE = re.compile(r"^(\d{4})_[A-Za-z0-9_]+\.sql$")


def migrations_dir(dialect_name: str) -> Path:
    """The folder with the migrations of one dialect."""
    return _ROOT / dialect_name


def list_migrations(directory: Path) -> list[tuple[int, Path]]:
    """Return ``(version, path)`` for every migration in *directory*, in order.

    Raises:
        ConfigurationError: If a file is misnamed or the numbers are not consecutive from 1.
    """
    found: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILE_RE.match(path.name)
        if match is None:
            raise ConfigurationError(f"Invalid migration file name: {path.name!r}.")
        found.append((int(match.group(1)), path))
    for expected, (number, path) in enumerate(found, start=1):
        if number != expected:
            raise ConfigurationError(
                f"Migrations must be consecutive from 0001; found {path.name!r} at position "
                f"{expected}."
            )
    return found


def latest_version(dialect_name: str, directory: Path | None = None) -> int:
    """The number of the last migration of a dialect (0 when there are none)."""
    migrations = list_migrations(directory or migrations_dir(dialect_name))
    return migrations[-1][0] if migrations else 0


def split_statements(sql: str) -> list[str]:
    """Split a migration file into statements at the ``-- statement`` lines."""
    chunks: list[list[str]] = [[]]
    for line in sql.splitlines():
        if line.strip() == STATEMENT_SEPARATOR:
            chunks.append([])
        else:
            chunks[-1].append(line)
    return [text for chunk in chunks if (text := "\n".join(chunk).strip())]


def control_table(dialect: Dialect, schema: str) -> sa.Table:
    """The ``newton_schema`` table (schema-qualified in Postgres, independent of the engine map)."""
    return sa.Table(
        CONTROL_TABLE,
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("rls_enabled", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("fulltext_config", sa.Text, nullable=False),
        sa.Column("applied_at", sa.Text, nullable=False),
        sa.CheckConstraint("id = 1", name="newton_schema_single_row"),
        schema=schema if dialect.name == "postgres" else None,
    )


def apply_migrations(
    connection: Any,
    dialect: Dialect,
    *,
    schema: str,
    fulltext_config: str,
    rls: bool = False,
    directory: Path | None = None,
) -> int:
    """Apply the pending migrations in the caller's transaction; return the new version.

    The caller opens the transaction (``with engine.begin() as connection``): if anything
    fails it rolls back and neither the objects nor the version change.

    Args:
        connection: A SQLAlchemy connection with an open transaction.
        dialect: The dialect of the connection.
        schema: Postgres schema (ignored by SQLite).
        fulltext_config: Postgres text-search configuration, used when the schema is
            created; an existing schema keeps the one it was created with.
        rls: Record the schema as Row-Level-Security managed (never reset to false here).
        directory: Folder of migrations. Default: the packaged folder of the dialect.

    Returns:
        The schema version after the call.

    Raises:
        ConfigurationError: If the migration files are inconsistent.
        ValueError: If *schema* or *fulltext_config* is not a plain identifier.
    """
    dialect.validate_identifier(schema)
    dialect.validate_identifier(fulltext_config)
    migrations = list_migrations(directory or migrations_dir(dialect.name))
    table = control_table(dialect, schema)

    with dialect.migration_lock(connection, schema=schema):
        if dialect.name == "postgres" and not sa.inspect(connection).has_schema(schema):
            connection.execute(CreateSchema(schema, if_not_exists=True))
        table.create(connection, checkfirst=True)
        row = connection.execute(sa.select(table)).mappings().first()
        version = row["version"] if row else 0
        stored_config = row["fulltext_config"] if row else fulltext_config
        stored_rls = bool(row["rls_enabled"]) if row else False

        applied = False
        for number, path in migrations:
            if number <= version:
                continue
            script = path.read_text(encoding="utf-8")
            script = script.replace("{schema}", schema).replace("{fulltext_config}", stored_config)
            for statement in split_statements(script):
                connection.exec_driver_sql(dialect.verbatim(statement))
            version = number
            applied = True

        if row and not (applied or rls and not stored_rls):
            return version
        values = {
            "version": version,
            "rls_enabled": stored_rls or rls,
            "fulltext_config": stored_config,
            "applied_at": datetime.now(UTC).isoformat(),
        }
        upsert = dialect.insert(table).values(id=1, **values)
        connection.execute(upsert.on_conflict_do_update(index_elements=["id"], set_=values))
    return version


def apply_rls_migrations(
    connection: Any,
    dialect: Dialect,
    *,
    schema: str,
    directory: Path | None = None,
) -> int:
    """Apply the pending Row-Level-Security migrations in the caller's transaction.

    The normal track must have been applied first (``newton_schema`` and its
    ``rls_version`` column exist). Records ``rls_enabled = true`` and the new
    ``rls_version``.

    Args:
        connection: A SQLAlchemy connection with an open transaction.
        dialect: The (Postgres) dialect of the connection.
        schema: Postgres schema.
        directory: Folder of RLS migrations. Default: the packaged ``postgres/rls`` folder.

    Returns:
        The RLS version after the call.

    Raises:
        ConfigurationError: If the schema has not been migrated yet or the files are
            inconsistent.
        ValueError: If *schema* is not a plain identifier.
    """
    dialect.validate_identifier(schema)
    migrations = list_migrations(directory or migrations_dir(dialect.name) / RLS_FOLDER)
    table = sa.table(
        CONTROL_TABLE,
        sa.column("id", sa.Integer),
        sa.column("rls_enabled", sa.Boolean),
        sa.column("rls_version", sa.Integer),
        schema=schema,
    )
    with dialect.migration_lock(connection, schema=schema):
        row = connection.execute(sa.select(table.c.rls_version)).first()
        if row is None:
            raise ConfigurationError("The schema has no migrations applied; migrate it first.")
        version = row[0]
        for number, path in migrations:
            if number <= version:
                continue
            script = path.read_text(encoding="utf-8").replace("{schema}", schema)
            for statement in split_statements(script):
                connection.exec_driver_sql(dialect.verbatim(statement))
            version = number
        connection.execute(sa.update(table).values(rls_enabled=True, rls_version=version))
    return version
