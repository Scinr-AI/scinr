"""
sql/keys.py — The ``_key`` of a node: its business key encoded as text.

Single-field keys (``uid``, ``id``, ``name``, ``label``, ``path``) are the value itself;
composite keys (``Document``, ``ModelField``) are a compact JSON array in the field order of
``ingest.freeze.node_identity``. The encoding lives in :mod:`scinr.newton.graph.identity`
(engine neutral); this module re-exports it for the SQL backend.
"""

from __future__ import annotations

from scinr.newton.graph.identity import decode_key, encode_key

__all__ = ["decode_key", "encode_key"]
