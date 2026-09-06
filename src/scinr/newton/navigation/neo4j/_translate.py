"""
navigation/neo4j/_translate.py — ``where=`` operator objects → Cypher predicates.

Turns the engine-neutral :mod:`scinr.newton.navigation.filters` operators into a
Cypher ``WHERE`` fragment plus a parameter dict. Property names are validated as
identifiers; all operands are parameterised.
"""

from __future__ import annotations

from typing import Any

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
    Op,
    Regex,
    StartsWith,
    normalize_where,
)
from scinr.newton.navigation.neo4j._safe import safe_ident

_SIMPLE = {Eq: "=", Ne: "<>", Gt: ">", Gte: ">=", Lt: "<", Lte: "<="}


def translate_where(
    where: Any,
    *,
    alias: str = "n",
    param_prefix: str = "w_",
) -> tuple[str, dict[str, Any]]:
    """Translate a ``where=`` mapping to ``(cypher_fragment, params)``.

    Args:
        where: A ``dict[str, Any | Op]`` (or ``None``). Bare values mean ``Eq``.
        alias: The Cypher node alias the predicates apply to (e.g. ``"mi"``).
        param_prefix: Prefix for generated parameter names.

    Returns:
        ``("", {})`` when there is nothing to filter, otherwise a fragment like
        ``"mi.`confidence` >= $w_confidence AND mi.`code` IN $w_code"`` and the
        matching params.

    Raises:
        NavigationError: On an invalid property name or an unknown operator.
    """
    ops = normalize_where(where)
    if not ops:
        return "", {}

    frags: list[str] = []
    params: dict[str, Any] = {}
    for key, op in ops.items():
        safe_ident(key, kind="property name")
        pname = f"{param_prefix}{key}"
        col = f"{alias}.`{key}`"
        frag, extra = _one(op, col, pname)
        frags.append(frag)
        params.update(extra)
    return " AND ".join(frags), params


def _one(op: Op, col: str, pname: str) -> tuple[str, dict[str, Any]]:
    kind = type(op)
    if kind in _SIMPLE:
        return f"{col} {_SIMPLE[kind]} ${pname}", {pname: op.value}
    if kind is In:
        return f"{col} IN ${pname}", {pname: list(op.values)}
    if kind is NotIn:
        return f"NOT ({col} IN ${pname})", {pname: list(op.values)}
    if kind is Contains:
        return f"{col} CONTAINS ${pname}", {pname: op.value}
    if kind is StartsWith:
        return f"{col} STARTS WITH ${pname}", {pname: op.value}
    if kind is EndsWith:
        return f"{col} ENDS WITH ${pname}", {pname: op.value}
    if kind is Regex:
        return f"{col} =~ ${pname}", {pname: op.pattern}
    if kind is IsNull:
        return f"{col} IS NULL", {}
    if kind is IsNotNull:
        return f"{col} IS NOT NULL", {}
    raise NavigationError(f"Unsupported where= operator: {kind.__name__}")


__all__ = ["translate_where"]
