"""
sql/dialect.py — The only place that knows how Postgres and SQLite differ.

:class:`Dialect` groups every engine-specific decision of the SQL backends: driver URLs,
connection setup, ``INSERT ... ON CONFLICT``, JSON and list columns, text matching and
ordering, full-text search, read-only transactions, error classification and the
migration lock. The rest of the SQL code is written once, with SQLAlchemy Core, and asks
the dialect whenever the engines disagree. Asking ``dialect.name`` outside this module is
a design error: add a method here instead.

Use :func:`get_dialect` to obtain the instance for a ``sql_url``.

SQL engine differences that are deliberately **not** hidden (documented in the plan):
regular expressions (Python ``re`` in SQLite, POSIX in Postgres, Java in Neo4j) and the
score of the full-text search.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, ClassVar, Literal

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REGCONFIG
from sqlalchemy.sql.elements import ClauseElement, ColumnElement
from sqlalchemy.sql.expression import Insert, Select
from sqlalchemy.sql.schema import Table
from sqlalchemy.types import TypeEngine

from scinr.newton.exceptions import ConfigurationError

__all__ = [
    "Dialect",
    "ErrorKind",
    "FulltextField",
    "PostgresDialect",
    "SqliteDialect",
    "get_dialect",
]

ErrorKind = Literal[
    "transient", "unique_violation", "access_denied", "connection", "authentication", "resource"
]
FulltextField = Literal["title", "description"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Oldest SQLite that has everything the backend needs (``json_*`` functions, FTS5,
#: ``RETURNING``, ``NULLS FIRST/LAST``).
_MIN_SQLITE_VERSION = (3, 35)

#: Name of the FTS5 virtual table that indexes ``info_units`` in SQLite.
_SQLITE_FTS_TABLE = "info_units_fts"

#: Execution option that makes a SQLite transaction start with a plain ``BEGIN`` instead of
#: ``BEGIN IMMEDIATE`` (read-only work that must not queue behind the single writer).
SQLITE_DEFERRED_OPTION = "sqlite_deferred"

_PG_TRANSIENT_STATES = frozenset({"40001", "40P01", "55P03"})
_PG_AUTH_STATES = frozenset({"28P01", "28000"})
_PG_CONNECTION_STATES = frozenset({"57P01", "57P02", "57P03"})
#: Messages of libpq for a connect rejected by authentication (no SQLSTATE is available).
_PG_AUTH_MESSAGES = (
    "password authentication failed",
    "authentication failed for user",
    "no pg_hba.conf entry",
)

_FTS5_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


def _is_clause(value: Any) -> bool:
    return isinstance(value, ClauseElement) or hasattr(value, "__clause_element__")


def _expr(value: Any, type_: TypeEngine | type[TypeEngine] = sa.Text) -> Any:
    """Return *value* if it is already a SQL expression, else a typed bound parameter."""
    return value if _is_clause(value) else sa.literal(value, type_)


def _unwrap(exc: BaseException) -> BaseException:
    """The driver exception behind a SQLAlchemy ``DBAPIError`` (or *exc* itself)."""
    orig = getattr(exc, "orig", None)
    return orig if isinstance(orig, BaseException) else exc


def _mro_names(exc: BaseException) -> set[str]:
    return {c.__name__ for c in type(exc).__mro__}


def _validated_json_keys(keys: Sequence[str]) -> list[str]:
    return [Dialect.validate_identifier(k) for k in keys]


def _fts5_query(query: str) -> str | None:
    """Translate a web-search style query into a safe FTS5 expression.

    Words and ``"quoted phrases"`` become quoted FTS5 phrases (so punctuation never reaches
    the FTS5 syntax parser) and are AND-ed; a bare ``or`` between two terms becomes ``OR``.
    Negated terms (``-word``) are dropped: FTS5 has no unary NOT. Returns ``None`` when no
    term is left.
    """
    parts: list[str] = []
    pending_or = False
    for phrase, word in _FTS5_TOKEN_RE.findall(query):
        if word:
            if word.lower() == "or":
                pending_or = bool(parts)
                continue
            if word.startswith("-"):
                continue
            text = word
        else:
            text = phrase
        if not text.strip():
            continue
        quoted = '"' + text.replace('"', '""') + '"'
        if pending_or:
            parts.append("OR")
            pending_or = False
        parts.append(quoted)
    return " ".join(parts) if parts else None


class Dialect(ABC):
    """Everything that changes between SQL engines. See the module docstring."""

    name: ClassVar[str]
    """``"postgres"`` or ``"sqlite"``."""

    json_type: TypeEngine
    """Column type of the ``props`` documents."""

    list_type: TypeEngine
    """Column type of a list of texts."""

    recursive_branches: ClassVar[bool]
    """Whether the recursive part of a ``WITH RECURSIVE`` may be several ``UNION``-ed
    ``SELECT``s that each read the CTE.

    SQLite (3.34+) allows it, and needs it to search an index per edge direction: it cannot
    push the join of the CTE into a ``UNION ALL`` subquery, so it scans the subquery for every
    queued row. Postgres allows the CTE only once in the recursive part, but it does push the
    join into a ``UNION ALL`` subquery (an index scan per branch). See
    ``navigation/sql/_walk.py``.
    """

    # ── URLs and connections ──────────────────────────────────────────────────

    @staticmethod
    def _with_driver(url: str, driver_scheme: str) -> str:
        _, sep, rest = url.partition("://")
        if not sep:
            raise ConfigurationError(f"Invalid sql_url: missing '://' ({url.split(':')[0]!r}).")
        return f"{driver_scheme}://{rest}"

    @abstractmethod
    def sync_url(self, url: str) -> str:
        """URL for the synchronous driver (any driver named in *url* is replaced)."""

    @abstractmethod
    def async_url(self, url: str) -> str:
        """URL for the asynchronous driver (any driver named in *url* is replaced)."""

    @abstractmethod
    def engine_kwargs(self, pool_size: int) -> dict[str, Any]:
        """Keyword arguments of ``create_engine`` / ``create_async_engine``."""

    @abstractmethod
    def on_connect(self, dbapi_connection: Any) -> None:
        """Set up a new DBAPI connection (SQLite: PRAGMAs and registered functions)."""

    @abstractmethod
    def on_begin(self, connection: Any) -> None:
        """React to the start of a transaction (SQLite: ``BEGIN IMMEDIATE``)."""

    @abstractmethod
    def schema_translate_map(self, schema: str) -> dict[str | None, str | None]:
        """Value of the ``schema_translate_map`` execution option of the engines."""

    # ── Writes ────────────────────────────────────────────────────────────────

    @abstractmethod
    def insert(self, table: Table) -> Insert:
        """An ``INSERT`` that supports this engine's ``on_conflict_do_*``."""

    @abstractmethod
    def append_unique(self, column: Any, value: Any) -> ColumnElement:
        """List expression: *column* with *value* appended unless already present.

        A ``NULL`` column counts as an empty list and a ``NULL`` *value* appends nothing.
        *value* may be a Python string or a SQL expression (``stmt.excluded.col``).
        """

    @abstractmethod
    def json_set_keys(self, column: Any, values: dict[str, Any]) -> ColumnElement:
        """JSON expression: *column* with the given top-level keys set.

        Raises:
            ValueError: If a key is not a plain identifier or a value is ``None`` (a null
                property does not exist: use :meth:`json_remove_keys`).
        """

    @abstractmethod
    def json_remove_keys(self, column: Any, keys: Sequence[str]) -> ColumnElement:
        """JSON expression: *column* without the given top-level keys."""

    # ── Reads ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def json_get(self, column: Any, key: str) -> ColumnElement:
        """The value of a top-level JSON key (Postgres: ``jsonb``; SQLite: native value).

        To compare in Postgres use ``json_get(col, k) == func.to_jsonb(value)``.
        """

    @abstractmethod
    def json_text(self, column: Any, key: str) -> ColumnElement:
        """The value of a top-level JSON key as text; ``NULL`` when the key is absent.

        Comparing it with a string behaves like Cypher: an absent key is ``NULL``, so both
        ``=`` and ``<>`` filter the row out.
        """

    @abstractmethod
    def instance_key_index_sql(self, prop: str, *, schema: str) -> str:
        """DDL of the ``(tenant_id, <prop>)`` index of ``model_instances`` (decision P3).

        ``CREATE INDEX IF NOT EXISTS model_instances_key_<prop>``; the only DDL outside
        the migrations. *prop* and *schema* are validated identifiers.

        Raises:
            ValueError: If *prop* or *schema* is not a plain identifier.
        """

    @abstractmethod
    def json_kind(self, column: Any, key: str) -> ColumnElement:
        """JSON type of a top-level key as text: ``string``, ``number``, ``boolean``,
        ``array``, ``object`` or ``null``; SQL ``NULL`` when the key is absent."""

    @abstractmethod
    def json_number(self, column: Any, key: str) -> ColumnElement:
        """The value of a top-level key when it is a JSON number, else ``NULL``."""

    @abstractmethod
    def json_bool(self, column: Any, key: str) -> ColumnElement:
        """The value of a top-level key when it is a JSON boolean, else ``NULL``."""

    @abstractmethod
    def list_contains(self, column: Any, value: Any) -> ColumnElement:
        """Predicate: the list column contains *value*."""

    @abstractmethod
    def list_overlaps(self, column: Any, values: Sequence[str]) -> ColumnElement:
        """Predicate: the list column shares at least one element with *values*."""

    @abstractmethod
    def contains_ci(self, column: Any, needle: Any) -> ColumnElement:
        """Predicate: *column* contains *needle*, ignoring case (Unicode aware)."""

    @abstractmethod
    def regex_match(self, column: Any, pattern: Any) -> ColumnElement:
        """Predicate: *column* matches the regular expression (search, not full match)."""

    @abstractmethod
    def text_cmp(self, column: Any) -> ColumnElement:
        """*column* as text compared by code point (``<``, ``>``…), like Cypher strings."""

    @abstractmethod
    def order_text(self, column: Any, *, desc: bool = False) -> ColumnElement:
        """ORDER BY element for text: by code point, nulls last (asc) / first (desc)."""

    def order_value(self, column: Any, *, desc: bool = False) -> ColumnElement:
        """ORDER BY element for non-text values: nulls last (asc) / first (desc), like Neo4j."""
        return column.desc().nulls_first() if desc else column.asc().nulls_last()

    @abstractmethod
    def fulltext_match(self, field: FulltextField, query: str) -> ColumnElement:
        """Predicate: ``info_units.<field>`` matches the full-text *query*."""

    @abstractmethod
    def fulltext_score(self, field: FulltextField, query: str) -> ColumnElement:
        """Relevance of ``info_units.<field>`` for *query* (higher is better; engine-specific)."""

    @abstractmethod
    def fulltext_units(self, field: FulltextField, query: str) -> Select:
        """``SELECT uid, score`` of the ``info_units`` whose *field* matches *query*."""

    # ── Utilities ─────────────────────────────────────────────────────────────

    @abstractmethod
    def read_only(self, connection: Any) -> AbstractContextManager[None]:
        """Make the current transaction read-only for the duration of the block.

        Postgres: ``SET TRANSACTION READ ONLY``, which must be the first statement of the
        transaction. SQLite: ``PRAGMA query_only`` on, back to off on exit.
        """

    @abstractmethod
    def use_schema(self, connection: Any, schema: str) -> None:
        """Make unqualified table names of raw SQL resolve in *schema* for the transaction.

        Postgres: ``SET LOCAL search_path``; SQLite: nothing (one schema per file).

        Raises:
            ValueError: If *schema* is not a plain identifier.
        """

    @abstractmethod
    def classify_error(self, exc: BaseException) -> ErrorKind | None:
        """Classify a driver (or SQLAlchemy-wrapped) exception; ``None`` if unrecognised."""

    @abstractmethod
    def migration_lock(self, connection: Any, *, schema: str = "") -> AbstractContextManager[None]:
        """Serialise concurrent migrations inside the caller's open transaction."""

    def max_concurrent_writes(self, requested: int) -> int:
        """How many write transactions of one operation may run at once (restore batches).

        Postgres runs what is asked; SQLite has a single writer, so one.
        """
        return max(1, requested)

    @abstractmethod
    def verbatim(self, statement: str) -> str:
        """Escape *statement* so ``exec_driver_sql`` sends it to the driver unchanged.

        Used by the migration runner: migration files are plain SQL, and some drivers
        (psycopg) interpret ``%`` in a statement even when no parameters are passed.
        """

    @staticmethod
    def validate_identifier(name: str) -> str:
        """Return *name* if it is a plain SQL identifier.

        Raises:
            ValueError: If it does not match ``^[A-Za-z_][A-Za-z0-9_]*$``.
        """
        if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        return name


class PostgresDialect(Dialect):
    """Postgres 14+ through psycopg 3. No extension is required."""

    name = "postgres"
    json_type = JSONB(none_as_null=True)
    list_type = ARRAY(sa.Text())
    recursive_branches = False

    def __init__(self, fulltext_config: str = "simple") -> None:
        self.fulltext_config = self.validate_identifier(fulltext_config)

    def sync_url(self, url: str) -> str:
        return self._with_driver(url, "postgresql+psycopg")

    def async_url(self, url: str) -> str:
        return self._with_driver(url, "postgresql+psycopg")

    def engine_kwargs(self, pool_size: int) -> dict[str, Any]:
        return {"pool_size": pool_size, "max_overflow": pool_size, "pool_pre_ping": True}

    def on_connect(self, dbapi_connection: Any) -> None:
        return None

    def on_begin(self, connection: Any) -> None:
        return None

    def schema_translate_map(self, schema: str) -> dict[str | None, str | None]:
        return {None: schema}

    def insert(self, table: Table) -> Insert:
        return postgresql.insert(table)

    def append_unique(self, column: Any, value: Any) -> ColumnElement:
        empty = sa.cast(sa.literal("{}"), self.list_type)
        current = sa.func.coalesce(column, empty)
        item = _expr(value, sa.Text)
        return sa.case(
            (item.is_(None), current),
            (item == sa.any_(current), column),
            else_=sa.func.array_append(current, item),
        )

    def json_set_keys(self, column: Any, values: dict[str, Any]) -> ColumnElement:
        _validated_json_keys(list(values))
        if any(v is None for v in values.values()):
            raise ValueError("json_set_keys cannot store null: use json_remove_keys")
        empty = sa.cast(sa.literal("{}"), self.json_type)
        patch = sa.cast(sa.literal(json.dumps(values)), self.json_type)
        return sa.func.coalesce(column, empty).op("||", return_type=self.json_type)(patch)

    def json_remove_keys(self, column: Any, keys: Sequence[str]) -> ColumnElement:
        if not keys:
            return column
        names = sa.literal(_validated_json_keys(keys), ARRAY(sa.Text()))
        return column.op("-", return_type=self.json_type)(names)

    def json_get(self, column: Any, key: str) -> ColumnElement:
        key = self.validate_identifier(key)
        return column.op("->", return_type=self.json_type)(sa.literal(key, sa.Text))

    def json_text(self, column: Any, key: str) -> ColumnElement:
        key = self.validate_identifier(key)
        return column.op("->>", return_type=sa.Text)(sa.literal(key, sa.Text))

    def json_kind(self, column: Any, key: str) -> ColumnElement:
        return sa.func.jsonb_typeof(self.json_get(column, key), type_=sa.Text)

    def json_number(self, column: Any, key: str) -> ColumnElement:
        kind = self.json_kind(column, key)
        return sa.case((kind == "number", sa.cast(self.json_text(column, key), sa.Float)))

    def json_bool(self, column: Any, key: str) -> ColumnElement:
        kind = self.json_kind(column, key)
        return sa.case((kind == "boolean", sa.cast(self.json_text(column, key), sa.Boolean)))

    def instance_key_index_sql(self, prop: str, *, schema: str) -> str:
        prop = self.validate_identifier(prop)
        schema = self.validate_identifier(schema)
        return (
            f'CREATE INDEX IF NOT EXISTS "model_instances_key_{prop}" '
            f"ON \"{schema}\".model_instances (tenant_id, (props->>'{prop}'))"
        )

    def list_contains(self, column: Any, value: Any) -> ColumnElement:
        return _expr(value, sa.Text) == sa.any_(column)

    def list_overlaps(self, column: Any, values: Sequence[str]) -> ColumnElement:
        return column.op("&&", return_type=sa.Boolean)(sa.literal(list(values), self.list_type))

    def contains_ci(self, column: Any, needle: Any) -> ColumnElement:
        return sa.func.strpos(sa.func.lower(column), sa.func.lower(_expr(needle, sa.Text))) > 0

    def regex_match(self, column: Any, pattern: Any) -> ColumnElement:
        return column.op("~", return_type=sa.Boolean)(_expr(pattern, sa.Text))

    def text_cmp(self, column: Any) -> ColumnElement:
        return sa.collate(column, "C")

    def order_text(self, column: Any, *, desc: bool = False) -> ColumnElement:
        return self.order_value(sa.collate(column, "C"), desc=desc)

    def _tsquery(self, query: str) -> ColumnElement:
        config = sa.cast(sa.literal(self.fulltext_config), REGCONFIG)
        return sa.func.websearch_to_tsquery(config, sa.literal(query, sa.Text))

    @staticmethod
    def _tsvector(field: FulltextField) -> ColumnElement:
        return sa.column(f"_fts_{Dialect.validate_identifier(field)}", postgresql.TSVECTOR)

    def fulltext_match(self, field: FulltextField, query: str) -> ColumnElement:
        return self._tsvector(field).op("@@", return_type=sa.Boolean)(self._tsquery(query))

    def fulltext_score(self, field: FulltextField, query: str) -> ColumnElement:
        return sa.func.ts_rank(self._tsvector(field), self._tsquery(query), type_=sa.Float)

    def fulltext_units(self, field: FulltextField, query: str) -> Select:
        from scinr.newton.sql.tables import info_units

        return sa.select(
            info_units.c.uid.label("uid"), self.fulltext_score(field, query).label("score")
        ).where(self.fulltext_match(field, query))

    def verbatim(self, statement: str) -> str:
        return statement.replace("%", "%%")

    @contextmanager
    def read_only(self, connection: Any) -> Iterator[None]:
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        yield

    def use_schema(self, connection: Any, schema: str) -> None:
        connection.exec_driver_sql(f'SET LOCAL search_path TO "{self.validate_identifier(schema)}"')

    def classify_error(self, exc: BaseException) -> ErrorKind | None:
        orig = _unwrap(exc)
        state = getattr(orig, "sqlstate", None)
        if isinstance(state, str) and state:
            if state in _PG_TRANSIENT_STATES:
                return "transient"
            if state == "23505":
                return "unique_violation"
            if state == "42501":
                return "access_denied"
            if state in _PG_AUTH_STATES:
                return "authentication"
            if state.startswith("53"):
                return "resource"
            if state.startswith("08") or state in _PG_CONNECTION_STATES:
                return "connection"
            return None
        is_psycopg = type(orig).__module__.startswith("psycopg")
        if is_psycopg and "OperationalError" in _mro_names(orig):
            # A failed connect has no SQLSTATE: libpq only reports the server's message.
            message = str(orig).lower()
            if any(marker in message for marker in _PG_AUTH_MESSAGES):
                return "authentication"
            return "connection"
        return None

    @contextmanager
    def migration_lock(self, connection: Any, *, schema: str = "") -> Iterator[None]:
        key = sa.literal("newton:", sa.Text) + sa.literal(schema, sa.Text)
        connection.execute(sa.select(sa.func.pg_advisory_xact_lock(sa.func.hashtext(key))))
        yield  # transaction-level lock: released by the caller's commit or rollback


def _regexp(pattern: Any, value: Any) -> int | None:
    if pattern is None or value is None:
        return None
    return 1 if re.search(pattern, str(value)) else 0


def _newton_lower(value: Any) -> Any:
    return value.lower() if isinstance(value, str) else value


class SqliteDialect(Dialect):
    """SQLite 3.35+ through ``sqlite3`` (sync) and ``aiosqlite`` (async)."""

    name = "sqlite"
    json_type = sa.JSON(none_as_null=True)
    list_type = sa.JSON(none_as_null=True)
    recursive_branches = True

    def sync_url(self, url: str) -> str:
        return self._with_driver(url, "sqlite+pysqlite")

    def async_url(self, url: str) -> str:
        return self._with_driver(url, "sqlite+aiosqlite")

    def engine_kwargs(self, pool_size: int) -> dict[str, Any]:
        # isolation_level=None: the driver never opens transactions on its own, so the
        # ``begin`` event is the only place that starts one (SQLAlchemy's documented
        # recipe for ``BEGIN IMMEDIATE`` with pysqlite and aiosqlite).
        return {"connect_args": {"isolation_level": None}}

    def on_connect(self, dbapi_connection: Any) -> None:
        import sqlite3

        if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
            raise ConfigurationError(
                f"SQLite {sqlite3.sqlite_version} is too old: the SQL backend needs "
                f"{'.'.join(map(str, _MIN_SQLITE_VERSION))} or newer."
            )
        cursor = dbapi_connection.cursor()
        try:
            for pragma in (
                "PRAGMA journal_mode=WAL",
                "PRAGMA busy_timeout=30000",
                "PRAGMA synchronous=NORMAL",
                "PRAGMA foreign_keys=OFF",
            ):
                cursor.execute(pragma)
        finally:
            cursor.close()
        dbapi_connection.create_function("regexp", 2, _regexp, deterministic=True)
        dbapi_connection.create_function("newton_lower", 1, _newton_lower, deterministic=True)

    def on_begin(self, connection: Any) -> None:
        deferred = connection.get_execution_options().get(SQLITE_DEFERRED_OPTION, False)
        connection.exec_driver_sql("BEGIN" if deferred else "BEGIN IMMEDIATE")

    def schema_translate_map(self, schema: str) -> dict[str | None, str | None]:
        return {None: None}

    def insert(self, table: Table) -> Insert:
        return sqlite.insert(table)

    @staticmethod
    def _json_items(column: Any) -> Any:
        return sa.func.json_each(column).table_valued("value")

    def append_unique(self, column: Any, value: Any) -> ColumnElement:
        current = sa.func.coalesce(column, sa.literal("[]", sa.String))
        item = _expr(value, sa.Text)
        present = sa.exists().where(self._json_items(current).c.value == item)
        return sa.case(
            (item.is_(None), current),
            (present, column),
            else_=sa.func.json_insert(current, "$[#]", item),
        )

    def json_set_keys(self, column: Any, values: dict[str, Any]) -> ColumnElement:
        keys = _validated_json_keys(list(values))
        if any(v is None for v in values.values()):
            raise ValueError("json_set_keys cannot store null: use json_remove_keys")
        args: list[Any] = []
        for key in keys:
            args.append(f'$."{key}"')
            args.append(sa.func.json(json.dumps(values[key])))
        return sa.func.json_set(sa.func.coalesce(column, sa.literal("{}", sa.String)), *args)

    def json_remove_keys(self, column: Any, keys: Sequence[str]) -> ColumnElement:
        if not keys:
            return column
        paths = [f'$."{k}"' for k in _validated_json_keys(keys)]
        return sa.func.json_remove(column, *paths)

    def json_get(self, column: Any, key: str) -> ColumnElement:
        key = self.validate_identifier(key)
        return sa.func.json_extract(column, f'$."{key}"')

    def json_text(self, column: Any, key: str) -> ColumnElement:
        key = self.validate_identifier(key)
        return sa.func.json_extract(column, f'$."{key}"', type_=sa.Text)

    def _json_type(self, column: Any, key: str) -> ColumnElement:
        key = self.validate_identifier(key)
        return sa.func.json_type(column, f'$."{key}"', type_=sa.Text)

    def json_kind(self, column: Any, key: str) -> ColumnElement:
        native = self._json_type(column, key)
        return sa.case(
            (native.in_(("integer", "real")), "number"),
            (native.in_(("true", "false")), "boolean"),
            (native == "text", "string"),
            else_=native,
        )

    def json_number(self, column: Any, key: str) -> ColumnElement:
        native = self._json_type(column, key)
        return sa.case((native.in_(("integer", "real")), self.json_get(column, key)))

    def json_bool(self, column: Any, key: str) -> ColumnElement:
        native = self._json_type(column, key)
        return sa.case((native == "true", sa.true()), (native == "false", sa.false()))

    def instance_key_index_sql(self, prop: str, *, schema: str) -> str:
        prop = self.validate_identifier(prop)
        self.validate_identifier(schema)  # one schema per file: nothing to qualify
        return (
            f'CREATE INDEX IF NOT EXISTS "model_instances_key_{prop}" '
            f"ON model_instances (tenant_id, json_extract(props, '$.\"{prop}\"'))"
        )

    def list_contains(self, column: Any, value: Any) -> ColumnElement:
        return sa.exists().where(self._json_items(column).c.value == _expr(value, sa.Text))

    def list_overlaps(self, column: Any, values: Sequence[str]) -> ColumnElement:
        if not values:
            return sa.false()
        return sa.exists().where(self._json_items(column).c.value.in_(list(values)))

    def contains_ci(self, column: Any, needle: Any) -> ColumnElement:
        return sa.func.instr(sa.func.newton_lower(column), sa.func.newton_lower(_expr(needle))) > 0

    def regex_match(self, column: Any, pattern: Any) -> ColumnElement:
        return sa.func.regexp(_expr(pattern), column, type_=sa.Boolean)

    def text_cmp(self, column: Any) -> ColumnElement:
        return column  # SQLite compares text by bytes (BINARY collation)

    def order_text(self, column: Any, *, desc: bool = False) -> ColumnElement:
        return self.order_value(column, desc=desc)

    @staticmethod
    def _fts_table() -> ColumnElement:
        return sa.column(_SQLITE_FTS_TABLE)

    def fulltext_match(self, field: FulltextField, query: str) -> ColumnElement:
        expression = _fts5_query(query)
        if expression is None:
            return sa.false()
        column = self.validate_identifier(field)
        match = f"{{{column}}}: ({expression})"
        return self._fts_table().match(sa.literal(match, sa.Text))

    def fulltext_score(self, field: FulltextField, query: str) -> ColumnElement:
        self.validate_identifier(field)
        return -sa.func.bm25(self._fts_table(), type_=sa.Float)

    def fulltext_units(self, field: FulltextField, query: str) -> Select:
        from scinr.newton.sql.tables import info_units

        fts = sa.table(_SQLITE_FTS_TABLE, sa.column("rowid"))
        return (
            sa.select(
                info_units.c.uid.label("uid"), self.fulltext_score(field, query).label("score")
            )
            .select_from(info_units.join(fts, fts.c.rowid == sa.literal_column("info_units.rowid")))
            .where(self.fulltext_match(field, query))
        )

    def verbatim(self, statement: str) -> str:
        return statement

    def max_concurrent_writes(self, requested: int) -> int:
        # One writer: concurrent write transactions would only queue on the database lock.
        return 1

    @contextmanager
    def read_only(self, connection: Any) -> Iterator[None]:
        connection.exec_driver_sql("PRAGMA query_only = ON")
        try:
            yield
        finally:
            connection.exec_driver_sql("PRAGMA query_only = OFF")

    def use_schema(self, connection: Any, schema: str) -> None:
        self.validate_identifier(schema)  # one schema per file: nothing to select

    def classify_error(self, exc: BaseException) -> ErrorKind | None:
        import sqlite3

        orig = _unwrap(exc)
        if not isinstance(orig, sqlite3.Error):
            return None
        message = str(orig).lower()
        if isinstance(orig, sqlite3.IntegrityError) and "unique constraint failed" in message:
            return "unique_violation"
        if isinstance(orig, sqlite3.OperationalError):
            if "database is locked" in message or "table is locked" in message or "busy" in message:
                return "transient"
            if "unable to open database" in message:
                return "connection"
        return None

    def migration_lock(self, connection: Any, *, schema: str = "") -> AbstractContextManager[None]:
        # The migration transaction is already ``BEGIN IMMEDIATE``: the single writer lock.
        return nullcontext()


def get_dialect(url: str, *, fulltext_config: str = "simple") -> Dialect:
    """Return the dialect for the scheme of *url*.

    Args:
        url: A ``sql_url`` (``postgresql://…``, ``postgres://…`` or ``sqlite:///…``; a
            ``+driver`` suffix is ignored).
        fulltext_config: Postgres text-search configuration (ignored by SQLite).

    Raises:
        ConfigurationError: If the scheme is not supported.
    """
    scheme = url.partition("://")[0].partition("+")[0].lower()
    if scheme in ("postgresql", "postgres"):
        return PostgresDialect(fulltext_config)
    if scheme == "sqlite":
        return SqliteDialect()
    raise ConfigurationError(
        f"Unsupported sql_url scheme {scheme!r}: use 'postgresql://' or 'sqlite:///'."
    )
