# utils/uid.py
"""Shared deterministic UID generation for all pipeline modules."""
from __future__ import annotations

import hashlib


def make_uid(*parts: str) -> str:
    """
    Return the first 16 hex characters of SHA-256 of the length-prefix-encoded parts.

    Each part is encoded as '<len>:<value>' and parts are joined with '||' before
    hashing. This ensures that no two distinct combinations of string values can
    produce the same raw input to SHA-256, regardless of their content (including
    values containing '||', ':', or any other separator characters).

    ⚠️  Breaking change vs. the previous implementation: UIDs produced by this
    function differ from those produced by the old `"||".join(parts)` formula
    for any input. Existing Neo4j nodes whose UIDs were generated with the old
    formula will have stale UIDs after this update.

    Examples
    --------
    >>> make_uid("a||b", "c") != make_uid("a", "b||c")  # True — no collision
    True
    >>> make_uid("hello", "world")  # deterministic
    '<some 16-char hex string>'
    """
    encoded = "||".join(f"{len(p)}:{p}" for p in parts)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def make_instance_uid(model_class: str, key_fields: dict[str, str]) -> str:
    """
    Return a deterministic, collision-free UID for a ModelInstance node identified
    by a composite key.

    The UID is stable across extractions: any two ModelInstance nodes of the same
    model_class with the same key_fields values will always receive the same UID,
    allowing Neo4j MERGE to deduplicate them globally (analogous to how
    LabeledEntity nodes are deduplicated by label + normalized_value).

    Parameters
    ----------
    model_class:
        The Pydantic model class name (e.g. 'ConditionModel').
    key_fields:
        Dict mapping field_name → already-normalized value for all fields
        marked with ``json_schema_extra={"instance_key": True}``.
        Values must be pre-normalized by the caller (lowercase, accent-stripped,
        whitespace-collapsed). The dict is sorted by key name internally so that
        field insertion order does not affect the UID.

    Returns
    -------
    str
        16-character hex UID.

    Examples
    --------
    >>> make_instance_uid("ConditionModel", {"condition_id": "1", "variation_code": "q.i.a.1(a)"})
    '<some 16-char hex string>'
    >>> # Order of keys does not matter:
    >>> make_instance_uid("ConditionModel", {"variation_code": "q.i.a.1(a)", "condition_id": "1"})
    '<same 16-char hex string>'
    """
    parts = ["mi", model_class]
    for field_name in sorted(key_fields.keys()):
        parts.append(field_name)
        parts.append(key_fields[field_name])
    return make_uid(*parts)
