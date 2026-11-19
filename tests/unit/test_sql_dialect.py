"""
tests/unit/test_sql_dialect.py -- The Postgres and SQLite dialects.

Two layers: the SQL each expression compiles to (no database), and its behaviour on a real
engine (SQLite in ``tmp_path``; Postgres when the disposable container is up).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite as sqlite_dialect

from scinr.newton.exceptions import ConfigurationError
from scinr.newton.sql.dialect import (
    PostgresDialect,
    SqliteDialect,
    _fts5_query,
    get_dialect,
)

PG = PostgresDialect()
LITE = SqliteDialect()
_PG_SA = postgresql.dialect()
_LITE_SA = sqlite_dialect.dialect()

_COL = sa.column("c")


def sql_of(expr: Any, dialect: Any) -> str:
    return " ".join(str(expr.compile(dialect=dialect)).split())


# ── get_dialect and URLs ─────────────────────────────────────────────────────


def test_get_dialect_by_scheme() -> None:
    assert isinstance(get_dialect("postgresql://u:p@h/db"), PostgresDialect)
    assert isinstance(get_dialect("postgres://u:p@h/db"), PostgresDialect)
    assert isinstance(get_dialect("postgresql+psycopg://u:p@h/db"), PostgresDialect)
    assert isinstance(get_dialect("sqlite:///x.db"), SqliteDialect)
    with pytest.raises(ConfigurationError, match="Unsupported sql_url scheme"):
        get_dialect("mysql://u@h/db")


def test_driver_urls() -> None:
    url = "postgresql://u:p@h:5432/db"
    assert PG.sync_url(url) == "postgresql+psycopg://u:p@h:5432/db"
    assert PG.async_url("postgresql+asyncpg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert LITE.sync_url("sqlite:///a/b.db") == "sqlite+pysqlite:///a/b.db"
    assert LITE.async_url("sqlite+pysqlite:///a/b.db") == "sqlite+aiosqlite:///a/b.db"


def test_engine_kwargs_and_schema_map() -> None:
    assert PG.engine_kwargs(4) == {"pool_size": 4, "max_overflow": 4, "pool_pre_ping": True}
    assert LITE.engine_kwargs(4) == {"connect_args": {"isolation_level": None}}
    assert PG.schema_translate_map("newton") == {None: "newton"}
    assert LITE.schema_translate_map("newton") == {None: None}


def test_verbatim_escapes_what_the_driver_would_interpret() -> None:
    assert PG.verbatim("SELECT '100%'") == "SELECT '100%%'"
    assert LITE.verbatim("SELECT '100%'") == "SELECT '100%'"


def test_validate_identifier() -> None:
    assert PG.validate_identifier("ok_name1") == "ok_name1"
    for bad in ("1a", "a-b", "a b", "", "a;b", "a'b"):
        with pytest.raises(ValueError):
            PG.validate_identifier(bad)
    with pytest.raises(ValueError):
        PostgresDialect("bad config")


# ── Compiled SQL ─────────────────────────────────────────────────────────────


def test_compiled_postgres() -> None:
    assert "array_append(" in sql_of(PG.append_unique(_COL, "x"), _PG_SA)
    assert "= ANY (" in sql_of(PG.append_unique(_COL, "x"), _PG_SA)
    assert "|| CAST(" in sql_of(PG.json_set_keys(_COL, {"a": 1}), _PG_SA)
    assert sql_of(PG.json_remove_keys(_COL, ["a", "b"]), _PG_SA).startswith("c - ")
    assert sql_of(PG.json_get(_COL, "k"), _PG_SA).startswith("c -> ")
    assert "= ANY (c)" in sql_of(PG.list_contains(_COL, "v"), _PG_SA)
    assert " && " in sql_of(PG.list_overlaps(_COL, ["a"]), _PG_SA)
    assert "strpos(lower(c), lower(" in sql_of(PG.contains_ci(_COL, "n"), _PG_SA)
    assert " ~ " in sql_of(PG.regex_match(_COL, "p"), _PG_SA)
    assert sql_of(PG.order_text(_COL), _PG_SA) == 'c COLLATE "C" ASC NULLS LAST'
    assert sql_of(PG.order_text(_COL, desc=True), _PG_SA) == 'c COLLATE "C" DESC NULLS FIRST'
    assert sql_of(PG.order_value(_COL), _PG_SA) == "c ASC NULLS LAST"
    assert sql_of(PG.order_value(_COL, desc=True), _PG_SA) == "c DESC NULLS FIRST"
    match = sql_of(PG.fulltext_match("title", "q"), _PG_SA)
    assert match.startswith("_fts_title @@ websearch_to_tsquery(CAST(")
    assert sql_of(PG.fulltext_score("description", "q"), _PG_SA).startswith(
        "ts_rank(_fts_description"
    )


def test_compiled_sqlite() -> None:
    sql = sql_of(LITE.append_unique(_COL, "x"), _LITE_SA)
    assert "FROM json_each(coalesce(c, ?))" in sql
    assert "ELSE json_insert(coalesce(c, ?), ?, ?)" in sql
    assert sql_of(LITE.json_set_keys(_COL, {"a": 1}), _LITE_SA).startswith("json_set(coalesce(")
    assert sql_of(LITE.json_remove_keys(_COL, ["a"]), _LITE_SA).startswith("json_remove(c")
    assert sql_of(LITE.json_get(_COL, "k"), _LITE_SA) == "json_extract(c, ?)"
    assert "json_each(c)" in sql_of(LITE.list_contains(_COL, "v"), _LITE_SA)
    assert "IN (" in sql_of(LITE.list_overlaps(_COL, ["a", "b"]), _LITE_SA)
    assert "instr(newton_lower(c), newton_lower(" in sql_of(LITE.contains_ci(_COL, "n"), _LITE_SA)
    assert sql_of(LITE.regex_match(_COL, "p"), _LITE_SA).startswith("regexp(")
    assert sql_of(LITE.order_text(_COL), _LITE_SA) == "c ASC NULLS LAST"
    assert sql_of(LITE.order_text(_COL, desc=True), _LITE_SA) == "c DESC NULLS FIRST"
    assert sql_of(LITE.fulltext_match("title", "q"), _LITE_SA) == "info_units_fts MATCH ?"
    assert sql_of(LITE.fulltext_score("title", "q"), _LITE_SA) == "-bm25(info_units_fts)"


def test_json_key_and_value_rules() -> None:
    for dialect in (PG, LITE):
        with pytest.raises(ValueError):
            dialect.json_set_keys(_COL, {"bad-key": 1})
        with pytest.raises(ValueError, match="null"):
            dialect.json_set_keys(_COL, {"a": None})
        with pytest.raises(ValueError):
            dialect.json_remove_keys(_COL, ["a'b"])
        with pytest.raises(ValueError):
            dialect.json_get(_COL, "a b")
        assert dialect.json_remove_keys(_COL, []) is _COL


def test_fts5_query_translation() -> None:
    assert _fts5_query("foo bar") == '"foo" "bar"'
    assert _fts5_query('say "hello world"') == '"say" "hello world"'
    assert _fts5_query("a or b") == '"a" OR "b"'
    assert _fts5_query("or a OR") == '"a"'
    assert _fts5_query("a -b") == '"a"'
    assert _fts5_query('he"llo') == '"he""llo"'
    assert _fts5_query("  ") is None
    assert _fts5_query("-only") is None
    assert "info_units_fts" not in sql_of(LITE.fulltext_match("title", ""), _LITE_SA)


# ── Behaviour on a real engine ───────────────────────────────────────────────


@dataclass
class Db:
    dialect: Any
    engine: sa.Engine
    table: sa.Table


def _make_db(dialect: Any, url: str, schema: str | None) -> Iterator[Db]:
    engine = sa.create_engine(dialect.sync_url(url), **dialect.engine_kwargs(3))
    sa.event.listen(engine, "connect", lambda conn, _rec: dialect.on_connect(conn))
    sa.event.listen(engine, "begin", dialect.on_begin)
    table = sa.Table(
        "things",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text),
        sa.Column("tags", dialect.list_type),
        sa.Column("props", dialect.json_type),
        schema=schema,
    )
    try:
        with engine.begin() as connection:
            if schema:
                connection.execute(sa.schema.CreateSchema(schema, if_not_exists=True))
            table.create(connection)
        yield Db(dialect, engine, table)
    finally:
        engine.dispose()


@pytest.fixture(params=["sqlite", "postgres"])
def db(request) -> Iterator[Db]:
    if request.param == "sqlite":
        url = request.getfixturevalue("sqlite_url")
        yield from _make_db(SqliteDialect(), url, None)
    else:
        url = request.getfixturevalue("pg_url")
        schema = request.getfixturevalue("pg_schema")
        yield from _make_db(PostgresDialect(), url, schema)


def _rows(db: Db, *columns: Any, where: Any = None, order: Any = None) -> list[tuple]:
    stmt = sa.select(*columns)
    if where is not None:
        stmt = stmt.where(where)
    stmt = stmt.order_by(order if order is not None else db.table.c.id)
    with db.engine.connect() as connection:
        return [tuple(r) for r in connection.execute(stmt)]


def _put(db: Db, **values: Any) -> None:
    with db.engine.begin() as connection:
        connection.execute(db.dialect.insert(db.table).values(**values))


def test_append_unique(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="a", tags=["x"])
    _put(db, id=2, name="b", tags=None)
    with db.engine.begin() as connection:
        for value in ("y", "x", "y", None):
            connection.execute(
                sa.update(t).where(t.c.id == 1).values(tags=d.append_unique(t.c.tags, value))
            )
        connection.execute(
            sa.update(t).where(t.c.id == 2).values(tags=d.append_unique(t.c.tags, "z"))
        )
    assert _rows(db, t.c.id, t.c.tags) == [(1, ["x", "y"]), (2, ["z"])]


def test_append_unique_in_an_upsert(db: Db) -> None:
    t, d = db.table, db.dialect
    for value in ("a", "b", "a"):
        stmt = d.insert(t).values(id=1, name="n", tags=[value])
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"], set_={"tags": d.append_unique(t.c.tags, value)}
        )
        with db.engine.begin() as connection:
            connection.execute(stmt)
    assert _rows(db, t.c.tags) == [(["a", "b"],)]


def test_list_contains_and_overlaps(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="a", tags=["x", "y"])
    _put(db, id=2, name="b", tags=["z"])
    _put(db, id=3, name="c", tags=None)
    assert _rows(db, t.c.id, where=d.list_contains(t.c.tags, "y")) == [(1,)]
    assert _rows(db, t.c.id, where=d.list_overlaps(t.c.tags, ["z", "q"])) == [(2,)]
    assert _rows(db, t.c.id, where=d.list_overlaps(t.c.tags, ["x", "z"])) == [(1,), (2,)]
    assert _rows(db, t.c.id, where=d.list_overlaps(t.c.tags, [])) == []


def test_json_set_remove_get(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="a", props={"x": 1})
    _put(db, id=2, name="b", props=None)
    with db.engine.begin() as connection:
        connection.execute(
            sa.update(t).values(
                props=d.json_set_keys(t.c.props, {"s": "text", "n": 2, "l": [1, 2], "b": True})
            )
        )
    rows = dict(_rows(db, t.c.id, t.c.props))
    assert rows[1] == {"x": 1, "s": "text", "n": 2, "l": [1, 2], "b": True}
    assert rows[2] == {"s": "text", "n": 2, "l": [1, 2], "b": True}
    with db.engine.begin() as connection:
        connection.execute(
            sa.update(t).values(props=d.json_remove_keys(t.c.props, ["x", "l", "b", "gone"]))
        )
    assert dict(_rows(db, t.c.id, t.c.props)) == {
        1: {"s": "text", "n": 2},
        2: {"s": "text", "n": 2},
    }
    assert _rows(db, t.c.id, where=d.json_get(t.c.props, "n") == _json_value(d, 2)) == [(1,), (2,)]


def _json_value(dialect: Any, value: Any) -> Any:
    return sa.func.to_jsonb(value) if dialect.name == "postgres" else value


def test_none_is_stored_as_sql_null(db: Db) -> None:
    t = db.table
    _put(db, id=1, name=None, tags=None, props=None)
    with db.engine.connect() as connection:
        raw = connection.execute(
            sa.select(t.c.tags.is_(None), t.c.props.is_(None), t.c.name.is_(None))
        ).one()
    assert tuple(raw) == (True, True, True)


def test_contains_ci_is_unicode_aware(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="El ÁRBOL grande")
    _put(db, id=2, name="el bosque")
    assert _rows(db, t.c.id, where=d.contains_ci(t.c.name, "árbol")) == [(1,)]
    assert _rows(db, t.c.id, where=d.contains_ci(t.c.name, "ÁRB")) == [(1,)]
    assert _rows(db, t.c.id, where=d.contains_ci(t.c.name, "EL ")) == [(1,), (2,)]
    assert _rows(db, t.c.id, where=d.contains_ci(t.c.name, "zzz")) == []


def test_regex_match(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="Contrato 2024-01")
    _put(db, id=2, name="Anexo")
    _put(db, id=3, name=None)
    assert _rows(db, t.c.id, where=d.regex_match(t.c.name, r"\d{4}-\d{2}")) == [(1,)]
    assert _rows(db, t.c.id, where=d.regex_match(t.c.name, "^An")) == [(2,)]


def test_order_nulls_and_code_point_order(db: Db) -> None:
    t, d = db.table, db.dialect
    for i, name in enumerate(["b", "B", "a", None, "Á", "_"], start=1):
        _put(db, id=i, name=name)
    asc = [r[0] for r in _rows(db, t.c.name, order=d.order_text(t.c.name))]
    # Code-point order (what Neo4j does): "B" < "_" < "a" < "b" < "Á"; nulls last.
    assert asc == ["B", "_", "a", "b", "Á", None]
    desc = [r[0] for r in _rows(db, t.c.name, order=d.order_text(t.c.name, desc=True))]
    assert desc == [None, "Á", "b", "a", "_", "B"]


def test_order_value_nulls(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="x")
    _put(db, id=2, name=None)
    assert [r[0] for r in _rows(db, t.c.id, order=d.order_value(t.c.name))] == [1, 2]
    assert [r[0] for r in _rows(db, t.c.id, order=d.order_value(t.c.name, desc=True))] == [2, 1]


def test_read_only_blocks_writes(db: Db) -> None:
    t, d = db.table, db.dialect
    _put(db, id=1, name="a")
    kwargs = {"sqlite_deferred": True} if d.name == "sqlite" else {}
    with db.engine.connect() as connection:
        connection = connection.execution_options(**kwargs)
        with d.read_only(connection):
            assert connection.execute(sa.select(t.c.id)).all() == [(1,)]
            with pytest.raises(sa.exc.DBAPIError):
                connection.execute(sa.delete(t))
        connection.rollback()
    assert _rows(db, t.c.id) == [(1,)]
    # The engine is writable again afterwards.
    with db.engine.begin() as connection:
        connection.execute(sa.update(t).values(name="b"))
    assert _rows(db, t.c.name) == [("b",)]


def test_classify_unique_violation(db: Db) -> None:
    _put(db, id=1, name="a")
    with pytest.raises(sa.exc.IntegrityError) as info:
        _put(db, id=1, name="again")
    assert db.dialect.classify_error(info.value) == "unique_violation"
    assert db.dialect.classify_error(info.value.orig) == "unique_violation"


def test_classify_unrelated_errors_is_none(db: Db) -> None:
    assert db.dialect.classify_error(ValueError("x")) is None
    assert db.dialect.classify_error(RuntimeError("database is locked")) is None


def test_classify_bad_sql_is_not_transient(db: Db) -> None:
    with pytest.raises(sa.exc.DBAPIError) as info:
        with db.engine.connect() as connection:
            connection.exec_driver_sql("SELECT * FROM table_that_does_not_exist")
    assert db.dialect.classify_error(info.value) is None


# ── SQLite specifics ─────────────────────────────────────────────────────────


@pytest.fixture
def lite(sqlite_url) -> Iterator[sa.Engine]:
    yield from (d.engine for d in _make_db(SqliteDialect(), sqlite_url, None))


def test_sqlite_connection_setup(lite: sa.Engine) -> None:
    with lite.connect() as connection:
        pragma = lambda name: connection.exec_driver_sql(f"PRAGMA {name}").scalar()  # noqa: E731
        assert pragma("journal_mode") == "wal"
        assert pragma("busy_timeout") == 30000
        assert pragma("synchronous") == 1  # NORMAL
        assert pragma("foreign_keys") == 0
        assert connection.exec_driver_sql("SELECT newton_lower('ÁRBOL')").scalar() == "árbol"
        assert connection.exec_driver_sql("SELECT regexp('^a', 'abc')").scalar() == 1
        assert connection.exec_driver_sql("SELECT regexp('^b', 'abc')").scalar() == 0
        assert connection.exec_driver_sql("SELECT regexp(NULL, 'abc')").scalar() is None


def test_sqlite_write_transactions_begin_immediate(lite: sa.Engine, sqlite_url) -> None:
    path = sqlite_url.removeprefix("sqlite:///")
    with lite.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE t (id INTEGER)")
        # A second connection cannot take the write lock while this one holds it.
        other = sqlite3.connect(path, timeout=0.2)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()


def test_sqlite_deferred_option_does_not_take_the_write_lock(lite: sa.Engine, sqlite_url) -> None:
    path = sqlite_url.removeprefix("sqlite:///")
    with lite.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE t (id INTEGER)")
    with lite.connect().execution_options(sqlite_deferred=True) as connection:
        connection.exec_driver_sql("SELECT * FROM t").all()
        other = sqlite3.connect(path, timeout=0.2)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
        finally:
            other.close()


def test_sqlite_concurrent_writers_are_serialised(lite: sa.Engine) -> None:
    with lite.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE counter (n INTEGER)")
        connection.exec_driver_sql("INSERT INTO counter VALUES (0)")
    errors: list[BaseException] = []

    def bump() -> None:
        try:
            for _ in range(20):
                with lite.begin() as connection:
                    n = connection.exec_driver_sql("SELECT n FROM counter").scalar()
                    connection.exec_driver_sql("UPDATE counter SET n = ?", (n + 1,))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    with lite.connect() as connection:
        assert connection.exec_driver_sql("SELECT n FROM counter").scalar() == 80


def test_sqlite_classify_locked_and_busy() -> None:
    assert LITE.classify_error(sqlite3.OperationalError("database is locked")) == "transient"
    assert LITE.classify_error(sqlite3.OperationalError("database table is locked")) == "transient"
    assert LITE.classify_error(sqlite3.OperationalError("SQLITE_BUSY: busy")) == "transient"
    assert (
        LITE.classify_error(sqlite3.OperationalError("unable to open database file"))
        == "connection"
    )
    assert LITE.classify_error(sqlite3.OperationalError("syntax error")) is None
    unique = sqlite3.IntegrityError("UNIQUE constraint failed: t.id")
    assert LITE.classify_error(unique) == "unique_violation"
    assert LITE.classify_error(sqlite3.IntegrityError("NOT NULL constraint failed: t.x")) is None


def test_sqlite_fulltext_round_trip(lite: sa.Engine) -> None:
    with lite.begin() as connection:
        try:
            connection.exec_driver_sql(
                "CREATE VIRTUAL TABLE info_units_fts USING fts5(title, description)"
            )
        except sa.exc.OperationalError:
            pytest.skip("SQLite was built without FTS5")
        connection.exec_driver_sql(
            "INSERT INTO info_units_fts(rowid, title, description) VALUES "
            "(1, 'Marketing authorisation', 'holder of the licence'), "
            "(2, 'Licence renewal', 'marketing plan'), "
            "(3, 'Pharmacovigilance', 'safety: reports')"
        )
    fts = sa.table("info_units_fts", sa.column("rowid"))

    def ids(field: str, query: str) -> list[int]:
        stmt = (
            sa.select(fts.c.rowid)
            .where(LITE.fulltext_match(field, query))
            .order_by(LITE.fulltext_score(field, query).desc(), fts.c.rowid)
        )
        with lite.connect() as connection:
            return sorted(r[0] for r in connection.execute(stmt))

    assert ids("title", "licence") == [2]
    assert ids("description", "licence") == [1]
    assert ids("title", "marketing licence") == []
    assert ids("title", "marketing or licence") == [1, 2]
    assert ids("title", '"marketing authorisation"') == [1]
    assert ids("description", "safety: reports") == [3]  # punctuation never breaks the parser
    assert ids("title", "") == []


# ── Postgres specifics ───────────────────────────────────────────────────────


def test_postgres_classify_sqlstates() -> None:
    class Err(Exception):
        def __init__(self, sqlstate: str | None) -> None:
            super().__init__("boom")
            self.sqlstate = sqlstate

    expected = {
        "40001": "transient",
        "40P01": "transient",
        "55P03": "transient",
        "23505": "unique_violation",
        "42501": "access_denied",
        "28P01": "authentication",
        "28000": "authentication",
        "53200": "resource",
        "53300": "resource",
        "08006": "connection",
        "57P03": "connection",
        "42P01": None,
    }
    for state, kind in expected.items():
        assert PG.classify_error(Err(state)) == kind, state
    assert PG.classify_error(Err(None)) is None


def test_postgres_classify_operational_error_without_sqlstate() -> None:
    psycopg = pytest.importorskip("psycopg")
    assert PG.classify_error(psycopg.OperationalError("connection refused")) == "connection"
    assert PG.classify_error(psycopg.errors.UniqueViolation("dup")) == "unique_violation"
    assert PG.classify_error(psycopg.errors.InsufficientPrivilege("no")) == "access_denied"
    assert PG.classify_error(psycopg.errors.DeadlockDetected("dl")) == "transient"
    wrapped = sa.exc.OperationalError("SELECT 1", {}, psycopg.OperationalError("down"))
    assert PG.classify_error(wrapped) == "connection"


def test_postgres_fulltext_round_trip(pg_url, pg_schema) -> None:
    dialect = PostgresDialect("simple")
    engine = sa.create_engine(dialect.sync_url(pg_url))
    try:
        with engine.begin() as connection:
            connection.execute(sa.schema.CreateSchema(pg_schema, if_not_exists=True))
            connection.exec_driver_sql(
                f"CREATE TABLE {pg_schema}.units (id int, title text, "
                "_fts_title tsvector GENERATED ALWAYS AS (to_tsvector('simple', title)) STORED)"
            )
            connection.exec_driver_sql(
                f"INSERT INTO {pg_schema}.units (id, title) VALUES "
                "(1, 'Marketing authorisation'), (2, 'Licence renewal')"
            )
        units = sa.table("units", sa.column("id"), schema=pg_schema)
        stmt = sa.select(units.c.id).where(dialect.fulltext_match("title", "licence OR marketing"))
        stmt = stmt.order_by(dialect.fulltext_score("title", "licence").desc(), units.c.id)
        with engine.connect() as connection:
            assert [r[0] for r in connection.execute(stmt)] == [2, 1]
    finally:
        engine.dispose()


def test_postgres_migration_lock_runs_in_the_transaction(pg_url, pg_schema) -> None:
    dialect = PostgresDialect()
    engine = sa.create_engine(dialect.sync_url(pg_url))
    try:
        with engine.begin() as connection, dialect.migration_lock(connection, schema=pg_schema):
            held = connection.exec_driver_sql(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
            ).scalar()
        assert held == 1
    finally:
        engine.dispose()
