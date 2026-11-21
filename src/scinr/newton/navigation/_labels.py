"""
navigation/_labels.py — The node labels the graph knows and how a node of each is found.

Engine-neutral (no SQL or Neo4j imports): ``get_node_property_keys`` uses it in both backends to
reject an unknown label and to pick the node whose keys it returns.
"""

from __future__ import annotations

from typing import NamedTuple

from scinr.newton.exceptions import NavigationError
from scinr.newton.ingest.freeze import CATALOG_KEY_FIELDS, UID_LABELS
from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.utils.scope import ScopeKind

__all__ = ["LabelInfo", "resolve_label"]

#: Primary labels whose nodes keep one ``created_by_user_id`` / ``job_id`` (merged ones are
#: ``_ARRAY_LABELS``). Catalog labels carry no tenant nor provenance.
_ARRAY_LABELS = frozenset({"ModelInstance", "LabeledEntity", "Entity"})


class LabelInfo(NamedTuple):
    """How to find one node of a label."""

    label: str  # the label as asked (a role label such as ``Section`` or a primary one)
    primary: str  # the primary label (``StructureNode`` for a role label)
    key: str  # natural key property: ``path``, ``id``, ``uid``, ``name``...
    scope_kind: ScopeKind | None  # provenance form of its nodes; ``None``: no scope (catalog)


def _infos() -> dict[str, LabelInfo]:
    found: dict[str, LabelInfo] = {
        "Document": LabelInfo("Document", "Document", "path", "scalar"),
        "StructureNode": LabelInfo("StructureNode", "StructureNode", "id", "scalar"),
    }
    for label in UID_LABELS:
        kind: ScopeKind = "array" if label in _ARRAY_LABELS else "scalar"
        found[label] = LabelInfo(label, label, "uid", kind)
    for label, fields in CATALOG_KEY_FIELDS.items():
        found[label] = LabelInfo(label, label, fields[0], None)
    for role_label in ROLE_TO_LABEL.values():
        found[role_label] = LabelInfo(role_label, "StructureNode", "id", "scalar")
    return found


_LABELS = _infos()


def resolve_label(label: str) -> LabelInfo:
    """The :class:`LabelInfo` of a node label.

    Raises:
        NavigationError: *label* is not a label of the graph (a primary label or the role
            label of a structure node).
    """
    info = _LABELS.get(label) if isinstance(label, str) else None
    if info is None:
        raise NavigationError(f"Unknown node label: {label!r}. Known: {sorted(_LABELS)}.")
    return info
