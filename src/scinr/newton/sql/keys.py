"""
sql/keys.py — The ``_key`` of a node: its business key encoded as text.

Single-field keys (``uid``, ``id``, ``name``, ``label``, ``path``) are the value itself;
composite keys (``Document``, ``ModelField``) are a compact JSON array in the field order of
``ingest.freeze.node_identity``. This module is the only place that computes keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from scinr.newton.ingest.freeze import node_identity

__all__ = ["decode_key", "encode_key"]


def encode_key(primary_label: str, props: Mapping[str, Any]) -> str:
    """The ``_key`` of a node.

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
