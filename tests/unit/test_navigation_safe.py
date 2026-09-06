"""Unit tests for scinr.newton.navigation.neo4j._safe."""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.base import DEFAULT_MAX_DEPTH
from scinr.newton.navigation.neo4j._safe import (
    assert_read_only,
    resolve_depth,
    safe_ident,
)


@pytest.mark.parametrize("good", ["n", "_x", "procedure_type", "HAS_CHILD", "a1_b2"])
def test_safe_ident_accepts(good: str) -> None:
    assert safe_ident(good) == good


@pytest.mark.parametrize("bad", ["1a", "a-b", "a b", "", "a.b", "a`b", "n)"])
def test_safe_ident_rejects(bad: str) -> None:
    with pytest.raises(NavigationError):
        safe_ident(bad)


def test_resolve_depth_none_is_default() -> None:
    assert resolve_depth(None) == DEFAULT_MAX_DEPTH


def test_resolve_depth_explicit_verbatim_no_cap() -> None:
    assert resolve_depth(1) == 1
    assert resolve_depth(3) == 3
    assert resolve_depth(100) == 100  # explicit value bypasses the guard


@pytest.mark.parametrize("bad", [0, -1, -5, True, 1.5, "3"])
def test_resolve_depth_rejects_bad(bad: object) -> None:
    with pytest.raises(NavigationError):
        resolve_depth(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "write_query",
    [
        "MATCH (n) SET n.x = 1",
        "CREATE (n:Foo)",
        "MATCH (n) DETACH DELETE n",
        "MERGE (n:Foo {id: 1})",
        "MATCH (n) REMOVE n.x",
        "DROP INDEX foo",
        "MATCH (n) FOREACH (x IN [1] | SET n.y = x)",
        "CALL { CREATE (n) } IN TRANSACTIONS",
        "LOAD CSV FROM 'x' AS row CREATE (n)",
    ],
)
def test_assert_read_only_rejects_writes(write_query: str) -> None:
    with pytest.raises(NavigationError):
        assert_read_only(write_query)


def test_assert_read_only_ignores_keyword_in_comment() -> None:
    assert_read_only("// this does not CREATE anything\nMATCH (n) RETURN n")
    assert_read_only("/* no MERGE here */ MATCH (n) RETURN n")


def test_assert_read_only_still_catches_write_after_comment() -> None:
    with pytest.raises(NavigationError):
        assert_read_only("/* comment */ MATCH (n) SET n.x = 1")


def test_assert_read_only_allows_plain_read() -> None:
    assert_read_only("MATCH (d:Document) RETURN d LIMIT 10")


def test_assert_read_only_rejects_empty() -> None:
    with pytest.raises(NavigationError):
        assert_read_only("   ")
