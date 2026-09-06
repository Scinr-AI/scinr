"""Unit tests for scinr.newton.navigation.neo4j._translate."""

from __future__ import annotations

import pytest

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
from scinr.newton.navigation.neo4j._translate import translate_where


def test_empty() -> None:
    assert translate_where(None) == ("", {})
    assert translate_where({}) == ("", {})


def test_simple_operators() -> None:
    frag, params = translate_where(
        {"a": Eq(value=1), "b": Ne(value=2), "c": Gt(value=3), "d": Gte(value=4),
         "e": Lt(value=5), "f": Lte(value=6)},
        alias="mi",
    )
    assert "mi.`a` = $w_a" in frag
    assert "mi.`b` <> $w_b" in frag
    assert "mi.`c` > $w_c" in frag
    assert "mi.`d` >= $w_d" in frag
    assert "mi.`e` < $w_e" in frag
    assert "mi.`f` <= $w_f" in frag
    assert params == {"w_a": 1, "w_b": 2, "w_c": 3, "w_d": 4, "w_e": 5, "w_f": 6}


def test_in_notin() -> None:
    frag, params = translate_where({"x": In(values=["a", "b"]), "y": NotIn(values=[1])}, alias="n")
    assert "n.`x` IN $w_x" in frag
    assert "NOT (n.`y` IN $w_y)" in frag
    assert params == {"w_x": ["a", "b"], "w_y": [1]}


def test_string_operators() -> None:
    frag, params = translate_where(
        {"a": Contains(value="x"), "b": StartsWith(value="y"), "c": EndsWith(value="z"),
         "d": Regex(pattern="^a.*")},
        alias="n",
    )
    assert "n.`a` CONTAINS $w_a" in frag
    assert "n.`b` STARTS WITH $w_b" in frag
    assert "n.`c` ENDS WITH $w_c" in frag
    assert "n.`d` =~ $w_d" in frag
    assert params["w_d"] == "^a.*"


def test_null_operators_take_no_param() -> None:
    frag, params = translate_where({"a": IsNull(), "b": IsNotNull()}, alias="n")
    assert "n.`a` IS NULL" in frag
    assert "n.`b` IS NOT NULL" in frag
    assert params == {}


def test_bare_value_is_eq() -> None:
    frag, params = translate_where({"status": "active"}, alias="d")
    assert frag == "d.`status` = $w_status"
    assert params == {"w_status": "active"}


def test_and_joined() -> None:
    frag, _ = translate_where({"a": 1, "b": 2}, alias="n")
    assert " AND " in frag


def test_bad_key_rejected() -> None:
    with pytest.raises(NavigationError):
        translate_where({"a b": 1})
