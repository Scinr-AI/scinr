"""Unit tests for scinr.newton.navigation.filters."""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.filters import (
    Eq,
    Gte,
    In,
    IsNull,
    Op,
    normalize_where,
    validate_key,
)


def test_bare_value_becomes_eq() -> None:
    out = normalize_where({"status": "active"})
    assert out == {"status": Eq(value="active")}


def test_operator_passes_through() -> None:
    out = normalize_where({"confidence": Gte(value=0.8), "code": In(values=["a", "b"])})
    assert out["confidence"] == Gte(value=0.8)
    assert out["code"] == In(values=["a", "b"])


def test_none_and_empty() -> None:
    assert normalize_where(None) == {}
    assert normalize_where({}) == {}


def test_invalid_key_rejected() -> None:
    for bad in ["1abc", "a-b", "a b", "", "n.title", "$x"]:
        with pytest.raises(NavigationError):
            normalize_where({bad: 1})


def test_validate_key_returns_value() -> None:
    assert validate_key("procedure_type") == "procedure_type"


def test_non_mapping_where_rejected() -> None:
    with pytest.raises(NavigationError):
        normalize_where([("a", 1)])  # type: ignore[arg-type]


def test_ops_are_frozen() -> None:
    from pydantic import ValidationError

    op = Eq(value=1)
    with pytest.raises(ValidationError):
        op.value = 2  # type: ignore[misc]


def test_isnull_has_no_operand() -> None:
    assert isinstance(IsNull(), Op)
    assert IsNull().model_dump() == {}
