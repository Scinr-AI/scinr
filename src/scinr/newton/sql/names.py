"""
sql/names.py — Names of the SQL tables and the identity of relationships.

The graph has one table per primary label (``ingest.freeze.node_identity``). This module is
the single place that maps a label to its table and key column, and that decides which
properties of a relationship are part of its identity (the ``_ident`` column of ``edges``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from scinr.newton.ingest.freeze import CATALOG_KEY_FIELDS

__all__ = [
    "CATALOG_LABELS",
    "CATALOG_TABLES",
    "DEDUP_TABLES",
    "DYNAMIC_IDENTITY_PROPS",
    "EDGE_IDENTITY_PROPS",
    "KEY_COLUMN",
    "LABEL_TABLE",
    "PER_DOCUMENT_TABLES",
    "TABLE_LABEL",
    "edge_ident",
]

#: Primary label -> table.
LABEL_TABLE: dict[str, str] = {
    "Document": "documents",
    "StructureNode": "structure_nodes",
    "InfoUnit": "info_units",
    "ModelDecision": "model_decisions",
    "ProposedModel": "proposed_models",
    "ProposedField": "proposed_fields",
    "ComplementaryMatch": "complementary_matches",
    "SupplementaryField": "supplementary_fields",
    "ExtractionResult": "extraction_results",
    "ModelInstance": "model_instances",
    "LabeledEntity": "labeled_entities",
    "Entity": "entities",
    "CatalogModel": "catalog_models",
    "ModelField": "model_fields",
    "EntityLabel": "entity_labels",
    "Theme": "themes",
}

#: Table -> primary label.
TABLE_LABEL: dict[str, str] = {table: label for label, table in LABEL_TABLE.items()}

#: Primary label -> the column that holds the ``_key`` of its nodes.
KEY_COLUMN: dict[str, str] = {
    "Document": "_key",
    "StructureNode": "id",
    "InfoUnit": "uid",
    "ModelDecision": "uid",
    "ProposedModel": "uid",
    "ProposedField": "uid",
    "ComplementaryMatch": "uid",
    "SupplementaryField": "uid",
    "ExtractionResult": "uid",
    "ModelInstance": "uid",
    "LabeledEntity": "uid",
    "Entity": "uid",
    "CatalogModel": "name",
    "ModelField": "_key",
    "EntityLabel": "label",
    "Theme": "path",
}

#: Labels of the global catalog (no tenant, relationships live in ``catalog_edges``).
CATALOG_LABELS: frozenset[str] = frozenset(CATALOG_KEY_FIELDS)

#: Node tables owned by one document (they carry ``_document_key``), ``documents`` included.
PER_DOCUMENT_TABLES: tuple[str, ...] = (
    "documents",
    "structure_nodes",
    "info_units",
    "model_decisions",
    "proposed_models",
    "proposed_fields",
    "complementary_matches",
    "supplementary_fields",
    "extraction_results",
)

#: Node tables deduplicated by content inside a tenant (list provenance columns).
DEDUP_TABLES: tuple[str, ...] = ("model_instances", "labeled_entities", "entities")

#: Global catalog node tables.
CATALOG_TABLES: tuple[str, ...] = tuple(LABEL_TABLE[label] for label in CATALOG_KEY_FIELDS)

#: Relationship type -> properties that are part of the ``MERGE`` pattern of the relationship
#: in the Neo4j implementation, i.e. of its identity. Properties set afterwards are not.
EDGE_IDENTITY_PROPS: dict[str, tuple[str, ...]] = {
    "REFERENCES": ("field_name", "list_index"),
    "HAS_ENTITY": ("role",),
    "AGGREGATES": ("field_name",),
    "PRODUCES_ENTITY": ("field_name",),
}


#: Identity of the relationships whose type is dynamic (named after a model field or a
#: declared relationship), recognised by their properties. Each group is the property map
#: of one ``MERGE`` pattern of the Neo4j implementation: ``[:<field> {index}]`` of a list of
#: sub-models, ``[:<REL> {via_model, from_field, to_field}]`` between ``EntityLabel`` nodes
#: and ``[:<REL> {via_field, join_via}]`` between ``CatalogModel`` nodes. A group applies
#: when its first property is present; the other properties of those relationships
#: (``composite``, ``predicate_raw``) are set afterwards and are not identity.
DYNAMIC_IDENTITY_PROPS: tuple[tuple[str, ...], ...] = (
    ("index",),
    ("via_model", "from_field", "to_field"),
    ("via_field", "join_via"),
)


def _encode(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def edge_ident(rel_type: str, props: Mapping[str, Any]) -> str:
    """The ``_ident`` of a relationship: its identity properties encoded as JSON text.

    ``""`` for the types whose identity is just (source, type, destination). Types not in
    :data:`EDGE_IDENTITY_PROPS` are dynamic: their identity is the first group of
    :data:`DYNAMIC_IDENTITY_PROPS` whose first property is in *props*, if any.
    """
    fields = EDGE_IDENTITY_PROPS.get(rel_type)
    if fields:
        return _encode([props.get(f) for f in fields])
    for group in DYNAMIC_IDENTITY_PROPS:
        if group[0] in props:
            return _encode([props.get(f) for f in group])
    return ""
