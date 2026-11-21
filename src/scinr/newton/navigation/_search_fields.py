"""
navigation/_search_fields.py — The text fields ``search_nodes`` looks into, per node label.

Engine-neutral. Each name was checked against the writers of the graph
(``graph/neo4j/*.py``) and the SQL schema (``sql/migrations/postgres/0001_init.sql``); the
first field of a label whose value contains the text is the ``field`` of the match.

``InfoUnit`` and ``ModelInstance`` are not listed: they are searched with
``search_info_units`` (full text) and ``search_model_instances`` (any field of the record).
"""

from __future__ import annotations

__all__ = ["SEARCH_FIELDS", "SPECIAL_SEARCH_LABELS"]

#: Primary label -> text properties, in priority order.
SEARCH_FIELDS: dict[str, tuple[str, ...]] = {
    "Document": ("name", "path"),
    "StructureNode": ("title", "id"),
    "ModelDecision": ("matched_model_class",),
    "ProposedModel": ("schema_name",),
    "ProposedField": ("field_name",),
    "SupplementaryField": ("field_name",),
    "ComplementaryMatch": ("model_class",),
    "ExtractionResult": ("model_class",),
    "LabeledEntity": ("value", "label"),
    "Entity": ("value",),
    "CatalogModel": ("name", "description"),
    "ModelField": ("name", "model"),
    "EntityLabel": ("label",),
    "Theme": ("path", "name"),
}

#: Labels searched through another navigator method instead of ``SEARCH_FIELDS``.
SPECIAL_SEARCH_LABELS: tuple[str, ...] = ("InfoUnit", "ModelInstance")
