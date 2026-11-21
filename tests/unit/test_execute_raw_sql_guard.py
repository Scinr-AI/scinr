"""
tests/unit/test_execute_raw_sql_guard.py -- The single-statement guard of the SQL ``execute_raw``
(no database): the first-keyword check and what each dialect does to the text.

The behaviour on real engines is in ``tests/sql/test_navigator_sql.py`` (SQLite and Postgres)
and ``tests/sql_rls/test_execute_raw_single_statement.py`` (organization roles).
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import psycopg as pg_psycopg

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation._safe import check_raw_sql, leading_sql_keyword
from scinr.newton.sql.dialect import PostgresDialect, SqliteDialect

PG = PostgresDialect()
LITE = SqliteDialect()


@pytest.mark.parametrize(
    ("query", "keyword"),
    [
        ("SELECT 1", "SELECT"),
        ("  select 1", "SELECT"),
        ("commit", "COMMIT"),
        ("; ;COMMIT", "COMMIT"),
        ("-- a\nCOMMIT", "COMMIT"),
        ("-- a\rCOMMIT", "COMMIT"),  # Postgres ends a line comment at \r too
        ("/* a */COMMIT", "COMMIT"),
        ("/* a */ /* b */\n\tSET ROLE x", "SET"),
        ("/**/BEGIN", "BEGIN"),
        ("/* ** */END", "END"),
        ("(SELECT 1)", None),
        ('"COMMIT"', None),
        ("", None),
        ("   ", None),
        ("-- only a comment", None),
        ("/* unterminated COMMIT", None),
    ],
)
def test_leading_sql_keyword(query: str, keyword: str | None) -> None:
    assert leading_sql_keyword(query, nested_comments=True) == keyword
    assert leading_sql_keyword(query, nested_comments=False) == keyword


def test_nested_comments_follow_the_engine() -> None:
    query = "/* /* inner */ COMMIT */ SELECT 1"
    # Postgres nests comments: all of "/* /* inner */ COMMIT */" is a comment
    assert leading_sql_keyword(query, nested_comments=True) == "SELECT"
    # SQLite does not: the comment ends at the first "*/" and COMMIT is the statement
    assert leading_sql_keyword(query, nested_comments=False) == "COMMIT"
    # the narrower reading never misses what the engine runs
    assert leading_sql_keyword("/* /* */ SELECT 1 */ COMMIT", nested_comments=True) == "COMMIT"


@pytest.mark.parametrize("dialect", [PG, LITE], ids=["postgres", "sqlite"])
@pytest.mark.parametrize(
    "query", ["COMMIT", "rollback", "BEGIN", "END", "SAVEPOINT s", "RELEASE s", "/* x */ commit"]
)
def test_transaction_control_is_refused_on_both_engines(dialect, query) -> None:
    with pytest.raises(NavigationError, match="refused") as caught:
        check_raw_sql(
            query, refused=dialect.raw_refused_keywords, nested_comments=dialect.nested_comments
        )
    assert caught.value.code is None


@pytest.mark.parametrize(
    "query",
    ["SET ROLE x", "RESET ROLE", "SET TRANSACTION READ WRITE", "DISCARD ALL", "DEALLOCATE ALL",
     "PREPARE p AS SELECT 1", "EXECUTE p", "DO $$ BEGIN END $$", "CALL p()", "LISTEN c",
     "COPY documents TO STDOUT", "LOCK documents", "START TRANSACTION", "ABORT"],
)  # fmt: skip
def test_postgres_session_statements_are_refused(query) -> None:
    with pytest.raises(NavigationError):
        check_raw_sql(query, refused=PG.raw_refused_keywords, nested_comments=True)


@pytest.mark.parametrize(
    "query", ["PRAGMA query_only = OFF", "pragma journal_mode", "ATTACH 'x.db' AS x", "DETACH x",
              "VACUUM INTO 'copy.db'"],
)  # fmt: skip
def test_sqlite_pragma_attach_and_vacuum_are_refused(query) -> None:
    with pytest.raises(NavigationError):
        check_raw_sql(query, refused=LITE.raw_refused_keywords, nested_comments=False)


@pytest.mark.parametrize("dialect", [PG, LITE], ids=["postgres", "sqlite"])
@pytest.mark.parametrize(
    "query",
    ["SELECT 1", "WITH x AS (SELECT 1) SELECT * FROM x", "VALUES (1)", "EXPLAIN SELECT 1",
     "(SELECT 1)", "-- why\nSELECT 1", "SELECT 'COMMIT'", "SELECT 1 AS settings"],
)  # fmt: skip
def test_queries_pass(dialect, query) -> None:
    check_raw_sql(
        query, refused=dialect.raw_refused_keywords, nested_comments=dialect.nested_comments
    )


def test_writes_are_left_to_the_engine() -> None:
    """INSERT / UPDATE / DELETE / DDL reach the engine, which refuses them with its SQLSTATE."""
    for query in (
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
    ):
        check_raw_sql(query, refused=PG.raw_refused_keywords, nested_comments=True)
        check_raw_sql(query, refused=LITE.raw_refused_keywords, nested_comments=False)


def test_postgres_text_always_carries_one_bound_parameter() -> None:
    params: dict = {"tenant": "t"}
    text = PG.single_statement("SELECT 1 -- trailing comment", params)
    assert text.startswith("SELECT 1 -- trailing comment\n-- :")
    marker = text.rsplit(":", 1)[1]
    assert params == {"tenant": "t", marker: 1}  # an int: sent with a type
    compiled = str(sa.text(text).compile(dialect=pg_psycopg.dialect()))
    # what psycopg receives: a placeholder, hence the extended protocol (one statement only)
    assert f"%({marker})s" in compiled


def test_sqlite_text_is_unchanged() -> None:
    params: dict = {}
    assert LITE.single_statement("SELECT 1", params) == "SELECT 1"
    assert params == {}
