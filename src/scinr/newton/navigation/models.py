"""
navigation/models.py — Engine-neutral return types for the navigation API.

Every navigator method returns one of these light Pydantic models (or a list of
them). They carry **no** engine-specific types. Each model exposes an opaque
``raw`` dict holding the backend-native record it was built from — useful for
debugging, **not** something to depend on across engines.

All models are frozen: navigation is read-only and its results are snapshots.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """Common config for all navigation result models."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    raw: dict[str, Any] = Field(
        default_factory=dict,
        repr=False,
        description="Opaque engine-native record. Do not depend on its shape across engines.",
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentRef(_Base):
    """A ``:Document`` node — a leaf document or a folder-parent."""

    path: str
    name: str
    version: int
    latest: bool
    is_folder: bool
    raw_file_id: str | None = None
    load_date: str | None = None
    tenant_id: str | None = None
    created_by_user_id: str | None = None
    job_id: str | None = None
    context_instructions: str | None = None


class DocumentTree(DocumentRef):
    """A ``DocumentRef`` plus nested ``IS_COMPOSED_OF`` children.

    Also used as a single-spine *lineage* (root → … → target) by
    ``get_document_ancestors``: there every node has exactly zero or one child.
    """

    depth: int = 0
    children: list[DocumentTree] = Field(default_factory=list)


class DocumentStats(_Base):
    """Aggregate counts for one document version."""

    path: str
    version: int
    structure_nodes: int = 0
    structure_nodes_by_role: dict[str, int] = Field(default_factory=dict)
    info_units: int = 0
    model_decisions: int = 0
    model_decisions_matched: int = 0
    model_decisions_proposed: int = 0
    extraction_results: int = 0
    model_instances: int = 0
    model_instances_by_class: dict[str, int] = Field(default_factory=dict)
    labeled_entities: int = 0
    labeled_entities_by_label: dict[str, int] = Field(default_factory=dict)
    triples: int = 0


# ---------------------------------------------------------------------------
# Structure nodes
# ---------------------------------------------------------------------------


class StructureNodeRef(_Base):
    """A ``:StructureNode`` (section, subsection, table, row, …).

    ``document_path`` / ``document_version`` are populated only when the query
    that produced this ref already carried the owning ``:Document`` — they are
    never derived by parsing the composite ``id`` (which is not safely
    parseable).
    """

    id: str
    node_id: str
    title: str | None = None
    role: str
    types: list[str] = Field(default_factory=list)
    appearance_order: int = 0
    theme: str | None = None
    source_page_ids: list[str] = Field(default_factory=list)
    row_index: int | None = None
    document_path: str | None = None
    document_version: int | None = None


class StructureTree(StructureNodeRef):
    """A ``StructureNodeRef`` plus its nested ``HAS_CHILD`` children."""

    depth: int = 0
    info_units: list[InfoUnitRef] | None = None
    children: list[StructureTree] = Field(default_factory=list)


class NodePath(_Base):
    """The chain of structure nodes from a document root down to one node."""

    document: DocumentRef | None = None
    nodes: list[StructureNodeRef] = Field(default_factory=list)


class NodeDescription(_Base):
    """An aggregate, human-oriented view of one structure node."""

    node: StructureNodeRef
    ancestors: list[StructureNodeRef] = Field(default_factory=list)
    info_units: list[InfoUnitRef] = Field(default_factory=list)
    model_decision: ModelDecisionRef | None = None
    extraction: ExtractionResultRef | None = None
    model_instance_count: int = 0
    child_count: int = 0
    source_page_ids: list[str] = Field(default_factory=list)
    source_text: list[PageText] | None = None


# ---------------------------------------------------------------------------
# InfoUnits
# ---------------------------------------------------------------------------


class InfoUnitRef(_Base):
    """A ``:InfoUnit`` — the smallest citable summary unit."""

    uid: str
    info_unit_id: str | None = None
    title: str
    description: str
    order: int = 0


class InfoUnitWithNode(InfoUnitRef):
    """An ``InfoUnitRef`` annotated with its owning structure node."""

    node_id: str
    node_title: str | None = None


class ScoredInfoUnit(InfoUnitWithNode):
    """An ``InfoUnitWithNode`` plus a relevance ``score`` (search results)."""

    score: float = 1.0


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


class ModelDecisionRef(_Base):
    """A ``:ModelDecision`` — Stage 3 annotation outcome for a node.

    ``confidence`` is a free word the annotation LLM emits (``"high"`` /
    ``"medium"`` / ``"low"`` in practice), **not** a number. ``coverage_gaps``
    is a list of strings (often empty).
    """

    uid: str
    matched_model_class: str | None = None
    confidence: str | None = None
    rationale: str | None = None
    coverage_gaps: list[str] = Field(default_factory=list)
    propose_new_model: bool | None = None
    proposed_model_description: str | None = None
    document_name: str | None = None
    timestamp: str | None = None
    source: str | None = None
    matched_model: str | None = None
    complementary_models: list[str] = Field(default_factory=list)
    supplementary_fields: list[str] = Field(default_factory=list)


class ModelDecisionWithNode(ModelDecisionRef):
    """A ``ModelDecisionRef`` annotated with its owning structure node."""

    node_id: str
    node_title: str | None = None


class DocumentModelProfile(_Base):
    """How a whole document was semantically catalogued by annotation.

    A compact roll-up of the ``matched`` and ``complementary`` model classes
    across every ``:ModelDecision`` of the document, with per-class node counts —
    without listing the individual decisions.
    """

    path: str
    version: int
    matched: list[ModelClassStat] = Field(default_factory=list)
    complementary: list[ModelClassStat] = Field(default_factory=list)
    proposed: list[str] = Field(default_factory=list)
    unannotated_nodes: int = 0


class ProposedFieldRef(_Base):
    """A single field of a ``:ProposedModel`` (or a ``:SupplementaryField``)."""

    field_name: str
    field_type: str | None = None
    description: str | None = None
    required: bool | None = None


class ProposedModelRef(_Base):
    """A ``:ProposedModel`` suggested during annotation when nothing matched."""

    uid: str
    schema_name: str | None = None
    description: str | None = None
    fields: list[ProposedFieldRef] = Field(default_factory=list)
    node_id: str | None = None

    @property
    def name(self) -> str | None:
        """Alias for :attr:`schema_name`."""
        return self.schema_name


class AnnotationCoverage(_Base):
    """How much of a document was annotated."""

    path: str
    version: int
    total_nodes: int = 0
    annotated: int = 0
    unannotated: int = 0
    matched: int = 0
    proposed: int = 0
    ratio: float = 0.0


# ---------------------------------------------------------------------------
# Extraction & model instances
# ---------------------------------------------------------------------------


class ExtractionResultRef(_Base):
    """A ``:ExtractionResult`` — Stage 4 output for a node.

    ``is_triple`` is derived (``model_class == "Triple"``), not a stored prop.
    """

    uid: str
    node_full_id: str | None = None
    model_class: str
    document_name: str | None = None
    timestamp: str | None = None
    is_triple: bool = False
    primary_model: str | None = None
    complementary_models: list[str] = Field(default_factory=list)


class ExtractionResultWithNode(ExtractionResultRef):
    """An ``ExtractionResultRef`` annotated with its owning structure node."""

    node_id: str
    node_title: str | None = None


class ModelInstanceRef(_Base):
    """A ``:ModelInstance`` — one extracted record; field values in ``properties``."""

    uid: str
    model_class: str
    properties: dict[str, Any] = Field(default_factory=dict)
    is_shell: bool | None = None
    via_rel: str | None = None
    direction: Literal["out", "in"] | None = None
    index: int | None = None


class ModelInstanceTree(ModelInstanceRef):
    """A ``ModelInstanceRef`` plus its nested outgoing-edge children."""

    depth: int = 0
    children: list[ModelInstanceTree] = Field(default_factory=list)


class ModelInstanceRelation(_Base):
    """A relationship between two ``:ModelInstance`` nodes, with direction."""

    rel_type: str
    direction: Literal["out", "in"]
    other: ModelInstanceRef


# ---------------------------------------------------------------------------
# Entities & triples
# ---------------------------------------------------------------------------


class LabeledEntityRef(_Base):
    """A ``:LabeledEntity`` — a deduplicated, labelled entity value."""

    uid: str
    label: str
    value: str
    normalized_value: str
    field_name: str | None = None
    list_index: int | None = None


class EntityRelation(_Base):
    """A Level-2 ``field_relationships`` edge between labeled entities."""

    rel_type: str
    direction: Literal["out", "in"]
    other: LabeledEntityRef


class Triple(_Base):
    """A subject–predicate–object statement (``Triple`` fallback extraction).

    ``predicate`` / ``predicate_raw`` / ``object`` are ``None`` when the subject
    entity has no predicate edge to an object entity of the same extraction
    result (a partial / dangling statement).
    """

    subject: str
    predicate: str | None = None
    object: str | None = None
    predicate_raw: str | None = None
    node_id: str | None = None


# ---------------------------------------------------------------------------
# Catalogue / schema introspection
# ---------------------------------------------------------------------------


class CatalogFieldRef(_Base):
    """One ``:ModelField`` of a ``:CatalogModel``."""

    name: str
    type: str | None = None
    entity_label: str | None = None
    is_instance_key: bool | None = None
    required: bool | None = None
    description: str | None = None


class CatalogModelRef(_Base):
    """A registered ``:CatalogModel``, optionally with its fields."""

    name: str
    description: str | None = None
    selectable: bool | None = None
    fields: list[CatalogFieldRef] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)


class CatalogRelation(_Base):
    """A declared relationship between two catalog entries.

    Covers ``AGGREGATES`` (model → model containment) and the domain
    relationship declarations between ``:CatalogModel`` / ``:EntityLabel`` nodes
    (``SPECIFIED_IN``, ``REQUIREMENT_APPLIES_TO``, …) carrying ``join_via`` /
    ``via_field`` / ``from_field`` / ``to_field`` metadata.
    """

    source: str
    target: str
    rel_type: str
    source_kind: Literal["CatalogModel", "EntityLabel"] = "CatalogModel"
    target_kind: Literal["CatalogModel", "EntityLabel"] = "CatalogModel"
    properties: dict[str, Any] = Field(default_factory=dict)


class CatalogGraph(_Base):
    """The whole model catalogue: nodes plus the relationships between them."""

    models: list[CatalogModelRef] = Field(default_factory=list)
    entity_labels: list[str] = Field(default_factory=list)
    relationships: list[CatalogRelation] = Field(default_factory=list)


class ThemeRef(_Base):
    """A ``:Theme`` node, or a distinct ``theme`` value in use."""

    name: str
    path: str | None = None


class ModelClassStat(_Base):
    """Count of ``:ModelInstance`` / decision nodes for one ``model_class``."""

    model_class: str
    count: int
    kind: Literal["in_use", "matched", "complementary"] | None = None


class RoleStat(_Base):
    """Count of ``:StructureNode`` nodes for one role."""

    role: str
    count: int


class EntityLabelStat(_Base):
    """Count of ``:LabeledEntity`` nodes for one label."""

    label: str
    count: int


class RelTypeStat(_Base):
    """Count of a distinct ``(source model, rel_type, target model)`` triple."""

    source_model: str | None = None
    rel_type: str
    target_model: str | None = None
    count: int


class GraphSummary(_Base):
    """Whole-graph counts by node type and (structural) relationship type."""

    node_counts: dict[str, int] = Field(default_factory=dict)
    relationship_counts: dict[str, int] = Field(default_factory=dict)
    total_nodes: int = 0
    total_relationships: int = 0
    total_relationship_types: int = 0
    documents: int = 0
    latest_documents: int = 0


# ---------------------------------------------------------------------------
# Generic / power tools
# ---------------------------------------------------------------------------


class NodeSelector(BaseModel):
    """Identifies a node by ``type`` + a unique ``key``/``value`` pair.

    Example: ``NodeSelector(type="ModelInstance", key="uid", value="abc123")``.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    key: str
    value: Any


class GraphNode(_Base):
    """A type-tagged node returned by the generic power tools."""

    types: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class PathResult(_Base):
    """A path between two nodes (``shortest_path``)."""

    length: int
    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)


class Subgraph(_Base):
    """A bounded neighbourhood of nodes and edges (``subgraph``)."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Source-text bridge
# ---------------------------------------------------------------------------


class PageText(_Base):
    """One converted source page, verbatim markdown."""

    page_id: str
    index: int | None = None
    markdown: str


# Resolve forward references for the self/cross-referential trees.
DocumentTree.model_rebuild()
StructureTree.model_rebuild()
ModelInstanceTree.model_rebuild()
NodeDescription.model_rebuild()
DocumentModelProfile.model_rebuild()


__all__ = [
    "DocumentRef", "DocumentTree", "DocumentStats",
    "StructureNodeRef", "StructureTree", "NodePath", "NodeDescription",
    "InfoUnitRef", "InfoUnitWithNode", "ScoredInfoUnit",
    "ModelDecisionRef", "ModelDecisionWithNode", "DocumentModelProfile",
    "ProposedFieldRef", "ProposedModelRef", "AnnotationCoverage",
    "ExtractionResultRef", "ExtractionResultWithNode",
    "ModelInstanceRef", "ModelInstanceTree", "ModelInstanceRelation",
    "LabeledEntityRef", "EntityRelation", "Triple",
    "CatalogFieldRef", "CatalogModelRef", "CatalogRelation", "CatalogGraph", "ThemeRef",
    "ModelClassStat", "RoleStat", "EntityLabelStat", "RelTypeStat", "GraphSummary",
    "NodeSelector", "GraphNode", "PathResult", "Subgraph",
    "PageText",
]
