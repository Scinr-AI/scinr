"""
tests/unit/test_sql_translate.py -- ``navigation/sql/_translate.py`` on real engines.

Every operator of ``where=`` is run against rows of ``model_instances`` (typed columns and
``props``) in SQLite and, when the test container is up, Postgres (the ``sql_backend``
fixture), and must answer like Cypher: comparisons across types never match, ``Regex`` matches
the whole string, and the string operators are case sensitive and treat ``%`` and ``_`` as
plain characters.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.filters import (
    Contains,
    EndsWith,
    Eq,
    Gt,
    Gte,
    In,
    IsNotNull,
    IsNull,
    Lt,
    Lte,
    Ne,
    NotIn,
    Regex,
    StartsWith,
)
from scinr.newton.navigation.sql._translate import prop_equals, translate_where
from scinr.newton.sql import tables as T
from scinr.newton.sql.engine import get_dialect_for_config, get_sync_engine
from scinr.newton.sql.schema import ensure_schema

ROWS = {
    "a": ("Alpha", {"name": "Alpha", "score": 5, "ratio": 2.5, "flag": True, "tag": "x%y_z"}),
    "b": ("Alpha", {"name": "alpha", "score": 10, "flag": False, "tag": "xAyBz"}),
    "c": ("Beta", {"name": "Beta Gamma", "score": "7", "tag": "x"}),
    "d": ("Beta", {"label": "no name here"}),
    "e": (None, {"name": "", "score": 0, "list": ["p", "q"]}),
}


@pytest.fixture
def instances(sql_backend):
    """``model_instances`` filled with ``ROWS`` in the engine under test."""
    ensure_schema()
    engine = get_sync_engine()
    with engine.begin() as connection:
        for uid, (model_class, props) in ROWS.items():
            connection.execute(
                sa.insert(T.model_instances).values(
                    uid=uid, tenant_id="t", model_class=model_class, props=props
                )
            )
    return get_dialect_for_config()


def _match(dialect, where) -> list[str]:
    table = T.model_instances
    statement = (
        sa.select(table.c.uid).where(*translate_where(where, table, dialect)).order_by(table.c.uid)
    )
    with get_sync_engine().connect() as connection:
        return [row[0] for row in connection.execute(statement)]


def test_no_where_adds_nothing(instances) -> None:
    assert translate_where(None, T.model_instances, instances) == []
    assert translate_where({}, T.model_instances, instances) == []


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"model_class": "Alpha"}, ["a", "b"]),  # typed column
        ({"model_class": Ne(value="Alpha")}, ["c", "d"]),  # NULL class is not "different"
        ({"name": "Alpha"}, ["a"]),  # case sensitive
        ({"score": 5}, ["a"]),
        ({"score": "7"}, ["c"]),  # the string "7" is not the number 7
        ({"score": 7}, []),
        ({"ratio": 2.5}, ["a"]),
        ({"flag": True}, ["a"]),
        ({"flag": False}, ["b"]),
        ({"score": Ne(value=5)}, ["b", "c", "e"]),  # "7" <> 5 is true; an absent score is null
        ({"score": Gt(value=5)}, ["b"]),
        ({"score": Gte(value=5)}, ["a", "b"]),
        ({"score": Lt(value=10)}, ["a", "e"]),
        ({"score": Lte(value=0)}, ["e"]),
        ({"name": Gt(value="Alpha")}, ["b", "c"]),  # by code point: "alpha" > "Alpha"
        ({"score": Gt(value="1")}, ["c"]),  # string comparison only on strings
        ({"name": Eq(value=None)}, []),  # = null never matches
        ({"score": Eq(value=[1])}, []),
    ],
)
def test_comparisons(instances, where, expected) -> None:
    assert _match(instances, where) == expected


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"model_class": In(values=["Beta", "zz"])}, ["c", "d"]),
        ({"score": In(values=[5, "7", 0])}, ["a", "c", "e"]),
        ({"score": In(values=[5, None])}, ["a"]),
        ({"score": In(values=[])}, []),
        ({"score": NotIn(values=[5])}, ["b", "c", "e"]),  # present values that are not 5
        ({"score": NotIn(values=["7", 10])}, ["a", "e"]),
        ({"score": NotIn(values=[5, None])}, []),  # NOT (x IN [.., null]) is never true
        ({"model_class": NotIn(values=["Alpha"])}, ["c", "d"]),
    ],
)
def test_in_and_not_in(instances, where, expected) -> None:
    assert _match(instances, where) == expected


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"name": Contains(value="lph")}, ["a", "b"]),
        ({"name": Contains(value="LPH")}, []),  # case sensitive
        ({"name": Contains(value="")}, ["a", "b", "c", "e"]),  # every string contains ""
        ({"name": StartsWith(value="Al")}, ["a"]),
        ({"name": StartsWith(value="")}, ["a", "b", "c", "e"]),
        ({"name": EndsWith(value="ma")}, ["c"]),
        ({"name": EndsWith(value="Alpha")}, ["a"]),
        ({"name": EndsWith(value="longer than the value")}, []),
        ({"score": Contains(value="7")}, ["c"]),  # only the string "7" is a string
        ({"tag": Contains(value="%")}, ["a"]),  # % is not a wildcard
        ({"tag": Contains(value="_")}, ["a"]),
        ({"tag": StartsWith(value="x%")}, ["a"]),
        ({"tag": EndsWith(value="_z")}, ["a"]),
        ({"model_class": StartsWith(value="Be")}, ["c", "d"]),
    ],
)
def test_string_operators(instances, where, expected) -> None:
    assert _match(instances, where) == expected


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"name": Regex(pattern="Alp.*")}, ["a"]),
        ({"name": Regex(pattern="lph")}, []),  # must match the whole value
        ({"name": Regex(pattern=".*lph.*")}, ["a", "b"]),
        ({"name": Regex(pattern="Alpha|Beta Gamma")}, ["a", "c"]),  # the alternation is anchored
        ({"name": Regex(pattern="")}, ["e"]),
        ({"score": Regex(pattern=".*")}, ["c"]),  # numbers are not strings
        ({"model_class": Regex(pattern="Be.a")}, ["c", "d"]),
    ],
)
def test_regex_matches_the_whole_value(instances, where, expected) -> None:
    assert _match(instances, where) == expected


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"name": IsNull()}, ["d"]),
        ({"name": IsNotNull()}, ["a", "b", "c", "e"]),
        ({"model_class": IsNull()}, ["e"]),
        ({"model_class": IsNotNull()}, ["a", "b", "c", "d"]),
        ({"nothing": IsNull()}, ["a", "b", "c", "d", "e"]),
        ({"created_by_user_ids": IsNotNull()}, []),  # no row has provenance lists
    ],
)
def test_null_checks(instances, where, expected) -> None:
    assert _match(instances, where) == expected


def test_several_keys_are_anded(instances) -> None:
    where = {"model_class": "Alpha", "score": Gte(value=5), "flag": False}
    assert _match(instances, where) == ["b"]


@pytest.mark.parametrize("key", ["bad key", "a-b", "1abc", "x`; DROP", ""])
def test_a_key_that_is_not_an_identifier_is_rejected(instances, key) -> None:
    with pytest.raises(NavigationError):
        translate_where({key: 1}, T.model_instances, instances)


def test_prop_equals_is_typed(instances) -> None:
    table = T.model_instances
    with get_sync_engine().connect() as connection:
        found = [
            row[0]
            for row in connection.execute(
                sa.select(table.c.uid).where(prop_equals(table, "flag", True, instances))
            )
        ]
        typed = [
            row[0]
            for row in connection.execute(
                sa.select(table.c.uid).where(prop_equals(table, "model_class", "Beta", instances))
            )
        ]
    assert found == ["a"]
    assert sorted(typed) == ["c", "d"]
    assert str(prop_equals(table, "x", None, instances)) in ("false", "0 = 1")


def test_dialect_json_helpers(instances) -> None:
    table = T.model_instances
    with get_sync_engine().connect() as connection:
        rows = connection.execute(
            sa.select(
                table.c.uid,
                instances.json_kind(table.c.props, "score").label("kind"),
                instances.json_number(table.c.props, "score").label("number"),
                instances.json_bool(table.c.props, "flag").label("boolean"),
            ).order_by(table.c.uid)
        ).all()
    kinds = {r.uid: r.kind for r in rows}
    assert kinds == {"a": "number", "b": "number", "c": "string", "d": None, "e": "number"}
    assert {r.uid: r.number for r in rows if r.number is not None} == {"a": 5, "b": 10, "e": 0}
    assert {r.uid: r.boolean for r in rows if r.boolean is not None} == {"a": True, "b": False}
