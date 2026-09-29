# utils/uid.py
"""Shared deterministic UID generation for all pipeline modules."""
from __future__ import annotations

import hashlib
import re
import unicodedata

from scinr.newton.utils.tenancy import tenant_key


def normalize_key(value: str) -> str:
    """Normalise a string the way the ingestion pipeline does before hashing.

    Steps: Unicode NFKD, drop combining marks (accents), collapse whitespace to
    single spaces, strip, lower-case. This is the exact transform
    ``entity_extraction/graph_mapper._normalize`` applies to ``instance_key``
    field values and ``LabeledEntity`` values before they are hashed into a
    deterministic UID, so callers that want to rebuild a ``ModelInstance`` UID
    from raw key values must run their inputs through this first.

    Args:
        value: Any string (a raw instance-key field value, an entity value…).

    Returns:
        The normalised string.

    Examples:
        >>> normalize_key("  Q.I.A.1(a) ")
        'q.i.a.1(a)'
        >>> normalize_key("Ib(A)") == normalize_key("ÍB(a)".replace("Í", "I"))
        True
    """
    nfkd = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().lower()


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

    Examples:
        >>> make_uid("a||b", "c") != make_uid("a", "b||c")  # True — no collision
        True
        >>> make_uid("hello", "world")  # deterministic
        '<some 16-char hex string>'
    """
    encoded = "||".join(f"{len(p)}:{p}" for p in parts)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def make_instance_uid(
    model_class: str, key_fields: dict[str, str], tenant_id: str | None = None
) -> str:
    """
    Return a deterministic, collision-free UID for a ModelInstance node identified
    by a composite key.

    The UID is stable across extractions: any two ModelInstance nodes of the same
    model_class with the same key_fields values *and the same tenant_id* will
    always receive the same UID, allowing Neo4j MERGE to deduplicate them
    (analogous to how LabeledEntity nodes are deduplicated by label +
    normalized_value).

    ``tenant_id`` is folded into the hash so that two tenants extracting the same
    content (same instance_key values) never collide onto the same node — content
    dedup only ever happens *within* a tenant. This is deliberate: MERGE's
    ``ON MATCH SET`` would otherwise let one tenant's re-extraction silently
    overwrite another tenant's field values on a node they'd end up sharing.

    The hashed value is the one stored on the nodes (:func:`tenant_key`), so a
    public instance (``tenant_id=None``) hashes ``"__public__"`` — exactly what
    ingestion writes for a public document.

    ⚠️  Breaking change: folding ``tenant_id`` in changes every UID this function
    produces versus the pre-multi-tenancy formula (even when ``tenant_id`` is
    ``None`` — the reserved "tenant" part is always present). Existing Neo4j
    nodes keep their old UIDs until re-ingested.

    Args:
        model_class: The Pydantic model class name (e.g. 'ConditionModel').
        key_fields: Dict mapping field_name → already-normalized value for all fields
            marked with ``json_schema_extra={"instance_key": True}``.
            Values must be pre-normalized by the caller (lowercase, accent-stripped,
            whitespace-collapsed). The dict is sorted by key name internally so that
            field insertion order does not affect the UID.
        tenant_id: Multi-tenant owner id, or ``None`` / ``"__public__"`` for
            public content (the stored value can be passed back as is). Two calls with the same *model_class* / *key_fields*
            but different *tenant_id* always produce different UIDs.

    Raises:
        ValueError: If *tenant_id* is empty (see
            :func:`~scinr.newton.utils.tenancy.tenant_key`).

    Returns:
        16-character hex UID.

    Examples:
        >>> make_instance_uid("ConditionModel", {"condition_id": "1", "variation_code": "q.i.a.1(a)"})
        '<some 16-char hex string>'
        >>> # Order of keys does not matter:
        >>> make_instance_uid("ConditionModel", {"variation_code": "q.i.a.1(a)", "condition_id": "1"})
        '<same 16-char hex string>'
        >>> # Different tenants never collide, even with identical key_fields:
        >>> make_instance_uid("ConditionModel", {"condition_id": "1"}, tenant_id="acme") != \\
        ...     make_instance_uid("ConditionModel", {"condition_id": "1"}, tenant_id="globex")
        True
    """
    parts = ["mi", model_class, "tenant", tenant_key(tenant_id)]
    for field_name in sorted(key_fields.keys()):
        parts.append(field_name)
        parts.append(key_fields[field_name])
    return make_uid(*parts)
