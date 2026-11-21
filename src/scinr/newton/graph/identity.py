"""
graph/identity.py — Business-key identity of graph nodes, shared by every backend.

``node_identity(labels)`` names the primary label of a node and the properties that make it
unique; ``encode_key`` / ``decode_key`` turn those properties into the text ``key`` that
identifies a node in the navigation API (and in the ``_key`` column of the SQL tables).

This module has no engine dependency (no SQLAlchemy, no Neo4j driver): ``ingest.freeze`` and
``sql.keys`` re-export these names, and ``navigation._map`` imports them from here so a
Neo4j-only install does not need the ``sql`` package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from scinr.newton.exceptions import FreezeError

__all__ = [
    "CATALOG_KEY_FIELDS",
    "PRIMARY_LABELS",
    "UID_LABELS",
    "decode_key",
    "encode_key",
    "node_identity",
]

_DOCUMENT_KEY_FIELDS = ("tenant_id", "path", "version")

# Global catalog nodes: never exported as nodes nor deleted, only referenced
# as relationship targets by their own key.
CATALOG_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "CatalogModel": ("name",),
    "ModelField": ("name", "model"),
    "EntityLabel": ("label",),
    "Theme": ("path",),
}

# Per-document and merged node labels keyed by ``uid``.
UID_LABELS = (
    "InfoUnit",
    "ModelDecision",
    "ProposedModel",
    "ProposedField",
    "ComplementaryMatch",
    "SupplementaryField",
    "ExtractionResult",
    "ModelInstance",
    "Entity",
    "LabeledEntity",
)

#: Every primary label, in the order ``node_identity`` tries them.
PRIMARY_LABELS: tuple[str, ...] = ("Document", "StructureNode", *CATALOG_KEY_FIELDS, *UID_LABELS)


def node_identity(labels: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Return ``(primary label, business-key fields)`` for a node's labels.

    The primary label is the one whose constraint/index backs the key
    (``StructureNode`` for the multi-label structure nodes). Raises
    :class:`FreezeError` for a node type the snapshot does not know how to
    rebuild.
    """
    if "Document" in labels:
        return "Document", _DOCUMENT_KEY_FIELDS
    if "StructureNode" in labels:
        return "StructureNode", ("id",)
    for label, fields in CATALOG_KEY_FIELDS.items():
        if label in labels:
            return label, fields
    for label in UID_LABELS:
        if label in labels:
            return label, ("uid",)
    raise FreezeError(f"Node with labels {list(labels)!r} has no known business key.")


def encode_key(primary_label: str, props: Mapping[str, Any]) -> str:
    """The ``_key`` of a node.

    Single-field keys (``uid``, ``id``, ``name``, ``label``, ``path``) are the value itself;
    composite keys (``Document``, ``ModelField``) are a compact JSON array in the field order
    of :func:`node_identity`.

    Raises:
        KeyError: If *props* lacks a key field (a caller error).
        FreezeError: If *primary_label* is not a known node label.
    """
    _, fields = node_identity([primary_label])
    values = [props[f] for f in fields]
    if len(values) == 1:
        return str(values[0])
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def decode_key(primary_label: str, key: str) -> dict[str, Any]:
    """The key properties of a node, from its ``_key`` (inverse of :func:`encode_key`).

    Single-field keys come back as text, composite keys with the JSON types they had.
    """
    _, fields = node_identity([primary_label])
    if len(fields) == 1:
        return {fields[0]: key}
    values = json.loads(key)
    if not isinstance(values, list) or len(values) != len(fields):
        raise ValueError(f"Malformed {primary_label} key: {key!r}")
    return dict(zip(fields, values, strict=True))
