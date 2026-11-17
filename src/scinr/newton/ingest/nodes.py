"""
ingest/nodes.py — Engine-independent helpers for the :Document / :StructureNode graph.

The Cypher that inserts documents and structure lives in
:mod:`scinr.newton.graph.neo4j.documents`; what stays here is what does not
depend on the graph engine: the composite StructureNode id, the deterministic
uid and the role → label map.

StructureNode composite id:  ``{tenant_key}::{doc_path}::{version}::{ancestor_path/node_id}``
                             (see :func:`make_structure_node_id`)
InfoUnit uid:                ``info_unit_id`` field (16-char SHA-256 hex), derived
                             from the parent StructureNode id — so it is tenant-scoped too
"""

from __future__ import annotations

import hashlib

from scinr.newton.models.document_structure import NodeRole


def make_structure_node_id(tenant_id: str, doc_path: str, version: int, node_path: str) -> str:
    """Composite ``id`` of a :StructureNode: ``{tenant}::{doc_path}::{version}::{node_path}``.

    The tenant prefix keeps two tenants' structure apart when they ingest the
    same path, and every id derived from this one (InfoUnit, ModelDecision,
    ExtractionResult uids, tabular Row ids) inherits it.

    Args:
        tenant_id: Stored tenant key (``utils.tenancy.tenant_key()`` applied).
        doc_path: Relative path of the owning document.
        version: Integer version of the owning document.
        node_path: Slash-separated ``node_id`` chain from the root node down to
            (and including) this node, e.g. ``"5_3/5_3_1"``.
    """
    return f"{tenant_id}::{doc_path}::{version}::{node_path}"


def _make_uid(*parts: str) -> str:
    """Compute a deterministic 16-char SHA-256 hex digest from one or more string parts."""
    payload = "||".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Role → Neo4j extra label mapping
# The f-string approach is safe because the value always comes from a
# validated NodeRole enum — never from user-supplied strings.
# ---------------------------------------------------------------------------
ROLE_TO_LABEL: dict[str, str] = {
    "section": "Section",
    "subsection": "Subsection",
    "table": "Table",
    "freeform_block": "FreeformBlock",
    "field_group": "FieldGroup",
    "appendix": "Appendix",
    "row": "Row",
}

assert set(ROLE_TO_LABEL.keys()) == {e.value for e in NodeRole}, (
    f"ROLE_TO_LABEL is out of sync with NodeRole enum. "
    f"Missing: {set(e.value for e in NodeRole) - set(ROLE_TO_LABEL.keys())}"
)
