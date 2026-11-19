"""
navigation/sql/_translate.py — ``where=`` operator objects → SQLAlchemy predicates.

The SQL twin of ``navigation/neo4j/_translate.py``. A property lives either in a typed column
of the table (``model_class``, ``label``…) or in the ``props`` JSON; both are compared with
the semantics of Cypher, which the plan fixes:

* a comparison between values of different types is ``null`` (the row is filtered out), not
  true or false: JSON values are type-checked with ``dialect.json_kind``;
* ``Regex`` matches the **whole** string (``=~``): the pattern is anchored with ``^(?:…)$``;
* ``Contains`` / ``StartsWith`` / ``EndsWith`` are case sensitive and only apply to strings.

Property names are validated as identifiers; every operand is a bound parameter.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation._safe import safe_ident
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
    Op,
    Regex,
    StartsWith,
    normalize_where,
)
from scinr.newton.sql.dialect import Dialect
from scinr.newton.sql.tables import LIST_COLUMNS, TYPED_COLUMNS

__all__ = ["Property", "contains", "ends_with", "prop_equals", "starts_with", "translate_where"]

_COMPARE = {
    Eq: lambda a, b: a == b,
    Ne: lambda a, b: a != b,
    Gt: lambda a, b: a > b,
    Gte: lambda a, b: a >= b,
    Lt: lambda a, b: a < b,
    Lte: lambda a, b: a <= b,
}

_ORDERING = (Gt, Gte, Lt, Lte)

_STRING, _NUMBER, _BOOLEAN, _OTHER = "string", "number", "boolean", "other"


def _value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return _BOOLEAN
    if isinstance(value, (int, float)):
        return _NUMBER
    if isinstance(value, str):
        return _STRING
    return _OTHER


def _column_kind(column: Any) -> str:
    """The comparison kind of a typed column (what Cypher would see in the property)."""
    type_ = column.type
    if isinstance(type_, sa.Boolean):
        return _BOOLEAN
    if isinstance(type_, (sa.Integer, sa.BigInteger, sa.Float, sa.Numeric)):
        return _NUMBER
    if isinstance(type_, sa.Text):
        return _STRING
    return _OTHER


def starts_with(text: Any, value: str) -> ColumnElement[bool]:
    """Case-sensitive ``text STARTS WITH value`` (``NULL`` text never matches)."""
    if not value:
        return text.is_not(None)
    return sa.func.substr(text, 1, len(value)) == value


def ends_with(text: Any, value: str) -> ColumnElement[bool]:
    """Case-sensitive ``text ENDS WITH value`` (``NULL`` text never matches)."""
    size = len(value)
    if not size:
        return text.is_not(None)
    return sa.and_(
        sa.func.length(text) >= size,
        sa.func.substr(text, sa.func.length(text) - (size - 1)) == value,
    )


def contains(text: Any, value: str) -> ColumnElement[bool]:
    """Case-sensitive ``text CONTAINS value`` (``NULL`` text never matches)."""
    if not value:
        return text.is_not(None)
    return sa.func.replace(text, value, "") != text


class Property:
    """One property of a node table, as a typed column or as a key of ``props``."""

    def __init__(self, table: Any, key: str, dialect: Dialect) -> None:
        base = getattr(table, "element", table).name
        self._dialect = dialect
        self._key = key
        self._props = table.c.props
        self._column: Any = None
        self._list_column: Any = None
        self._is_list = False
        if key in TYPED_COLUMNS.get(base, frozenset()):
            if key in LIST_COLUMNS.get(base, frozenset()):
                self._is_list = True
                self._list_column = table.c[key]
            else:
                self._column = table.c[key]

    @property
    def column(self) -> Any:
        """The typed column of the property, ``None`` when it lives in ``props``."""
        return self._column

    def values(self, kind: str) -> ColumnElement | None:
        """The property as an expression of *kind*; ``None`` when it can never be of it."""
        if self._is_list:
            return None
        if self._column is not None:
            return self._column if _column_kind(self._column) == kind else None
        if kind == _STRING:
            text = self._dialect.json_text(self._props, self._key)
            return sa.case((self._dialect.json_kind(self._props, self._key) == "string", text))
        if kind == _NUMBER:
            return self._dialect.json_number(self._props, self._key)
        if kind == _BOOLEAN:
            return self._dialect.json_bool(self._props, self._key)
        return None

    def present(self) -> ColumnElement[bool]:
        if self._is_list:
            return self._list_column.is_not(None)
        if self._column is not None:
            return self._column.is_not(None)
        kind = self._dialect.json_kind(self._props, self._key)
        return sa.and_(kind.is_not(None), kind != "null")


def prop_equals(table: sa.Table, key: str, value: Any, dialect: Dialect) -> ColumnElement[bool]:
    """Predicate ``n.<key> = <value>`` for a scalar *value* (``false`` for ``None``)."""
    safe_ident(key, kind="property name")
    kind = _value_kind(value)
    expression = Property(table, key, dialect).values(kind)
    if expression is None or value is None:
        return sa.false()
    return expression == value


def _typed(prop: Property, value: Any) -> ColumnElement | None:
    kind = _value_kind(value)
    if kind == _OTHER:
        return None
    return prop.values(kind)


def _one(op: Op, prop: Property, dialect: Dialect) -> ColumnElement[bool]:
    kind = type(op)
    if kind in _COMPARE:
        value = op.value  # type: ignore[attr-defined]
        if value is None:
            return sa.false()
        expression = _typed(prop, value)
        if kind is Ne:
            # ``<>`` between values of different types is true (only ``null`` is not).
            if expression is None:
                return prop.present()
            return sa.and_(
                prop.present(), sa.func.coalesce(expression == value, sa.false()) == sa.false()
            )
        if expression is None:
            return sa.false()
        if kind in _ORDERING and isinstance(value, str):
            expression = dialect.text_cmp(expression)
        return _COMPARE[kind](expression, value)
    if kind is In or kind is NotIn:
        values = list(op.values)  # type: ignore[attr-defined]
        if kind is NotIn and any(v is None for v in values):
            return sa.false()
        groups: dict[str, list[Any]] = {}
        for value in values:
            if value is not None and _value_kind(value) != _OTHER:
                groups.setdefault(_value_kind(value), []).append(value)
        matches: list[ColumnElement[bool]] = []
        for group_kind, items in groups.items():
            expression = prop.values(group_kind)
            if expression is not None:
                matches.append(expression.in_(items))
        found = sa.or_(*matches) if matches else sa.false()
        if kind is In:
            return found
        return sa.and_(prop.present(), sa.func.coalesce(found, sa.false()) == sa.false())
    if kind in (Contains, StartsWith, EndsWith, Regex):
        text = prop.values(_STRING)
        if text is None:
            return sa.false()
        if kind is Regex:
            return dialect.regex_match(text, "^(?:" + op.pattern + ")$")  # type: ignore[attr-defined]
        value = op.value  # type: ignore[attr-defined]
        if kind is StartsWith:
            return starts_with(text, value)
        if kind is EndsWith:
            return ends_with(text, value)
        return contains(text, value)
    if kind is IsNull:
        return sa.not_(prop.present())
    if kind is IsNotNull:
        return prop.present()
    raise NavigationError(f"Unsupported where= operator: {kind.__name__}")


def translate_where(where: Any, table: sa.Table, dialect: Dialect) -> list[ColumnElement[bool]]:
    """Translate a ``where=`` mapping to SQLAlchemy conditions on *table*.

    Args:
        where: A ``dict[str, Any | Op]`` (or ``None``). Bare values mean ``Eq``.
        table: The node table the predicates apply to (or an alias of it).
        dialect: Dialect of the engine.

    Returns:
        One condition per key (an empty list when there is nothing to filter).

    Raises:
        NavigationError: On an invalid property name or an unknown operator.
    """
    ops = normalize_where(where)
    conditions: list[ColumnElement[bool]] = []
    for key, op in ops.items():
        safe_ident(key, kind="property name")
        conditions.append(_one(op, Property(table, key, dialect), dialect))
    return conditions
