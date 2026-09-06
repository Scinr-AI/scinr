"""
navigation/filters.py — Engine-neutral property filter operators for ``where=``.

Every list-returning navigation method that filters on node properties accepts a
``where`` argument shaped as ``dict[str, Any | Op]``:

    where={"status": "active", "confidence": Gte(0.8), "code": In(["A", "B"])}

A bare value is sugar for :class:`Eq`. Each :class:`Op` is a frozen Pydantic
model carrying only its operands — it holds no engine syntax. A concrete backend
(``navigation/neo4j/_translate.py`` for Neo4j) turns each operator into its
native predicate with fully parameterised values.

Property keys are validated against ``^[A-Za-z_][A-Za-z0-9_]*$`` so they can be
safely interpolated as an identifier; values are never interpolated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from scinr.newton.exceptions import NavigationError

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Op(BaseModel):
    """Base class for all ``where=`` filter operators. Frozen and immutable."""

    model_config = ConfigDict(frozen=True)

    def describe(self) -> str:
        """Return a short human-readable form, e.g. ``">= 0.8"``. For logs/errors."""
        return type(self).__name__


class Eq(Op):
    """``field == value``."""

    value: Any


class Ne(Op):
    """``field != value``."""

    value: Any


class Gt(Op):
    """``field > value``."""

    value: Any


class Gte(Op):
    """``field >= value``."""

    value: Any


class Lt(Op):
    """``field < value``."""

    value: Any


class Lte(Op):
    """``field <= value``."""

    value: Any


class In(Op):
    """``field IN values``."""

    values: Sequence[Any]


class NotIn(Op):
    """``NOT (field IN values)``."""

    values: Sequence[Any]


class Contains(Op):
    """Substring match: ``value`` appears somewhere in the (string) ``field``."""

    value: str


class StartsWith(Op):
    """The (string) ``field`` starts with ``value``."""

    value: str


class EndsWith(Op):
    """The (string) ``field`` ends with ``value``."""

    value: str


class Regex(Op):
    """The (string) ``field`` fully matches the regular expression ``pattern``."""

    pattern: str


class IsNull(Op):
    """``field IS NULL`` — the property is absent or explicitly null."""


class IsNotNull(Op):
    """``field IS NOT NULL`` — the property is present and non-null."""


# Public operator names, for re-export and validation messages.
OPERATORS: tuple[type[Op], ...] = (
    Eq, Ne, Gt, Gte, Lt, Lte, In, NotIn,
    Contains, StartsWith, EndsWith, Regex, IsNull, IsNotNull,
)


def validate_key(key: str) -> str:
    """Return *key* unchanged if it is a safe property identifier, else raise.

    Args:
        key: A candidate node-property name.

    Returns:
        The same string.

    Raises:
        NavigationError: If *key* does not match ``^[A-Za-z_][A-Za-z0-9_]*$``.
    """
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise NavigationError(
            f"Invalid property name in where=: {key!r}. "
            r"Property names must match ^[A-Za-z_][A-Za-z0-9_]*$."
        )
    return key


def normalize_where(where: Mapping[str, Any] | None) -> dict[str, Op]:
    """Normalise a raw ``where=`` mapping into ``{validated_key: Op}``.

    A value that is not already an :class:`Op` is wrapped in :class:`Eq`. Keys
    are validated with :func:`validate_key`.

    Args:
        where: The user-supplied mapping, or ``None``.

    Returns:
        A new dict mapping each validated key to an :class:`Op`. Empty when
        *where* is ``None`` or empty.

    Raises:
        NavigationError: On an invalid key or a non-mapping *where*.
    """
    if where is None:
        return {}
    if not isinstance(where, Mapping):
        raise NavigationError(
            f"where= must be a mapping of property name to value/operator, got {type(where).__name__}."
        )
    out: dict[str, Op] = {}
    for key, val in where.items():
        vkey = validate_key(key)
        out[vkey] = val if isinstance(val, Op) else Eq(value=val)
    return out


__all__ = [
    "Op",
    "Eq", "Ne", "Gt", "Gte", "Lt", "Lte", "In", "NotIn",
    "Contains", "StartsWith", "EndsWith", "Regex", "IsNull", "IsNotNull",
    "OPERATORS", "normalize_where", "validate_key",
]
