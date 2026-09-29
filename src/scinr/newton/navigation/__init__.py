"""
navigation — Read-only, engine-abstracted traversal of the scinr knowledge graph.

Configure the library once, then obtain a navigator and call typed ``async``
methods — no Cypher required::

    from scinr.newton import configure
    from scinr.newton.navigation import graph_navigator, In, Gte

    configure(neo4j_user="neo4j", neo4j_password="…", neo4j_database="neo4j")  # graph_backend defaults to "neo4j"

    async with graph_navigator() as nav:
        roots = await nav.list_root_documents()
        tree  = await nav.get_document_tree(roots[0].path)
        rows  = await nav.get_model_instances_by_class(
            "VariationModel",
            where={"procedure_type": In(["IA", "IB"]), "confidence": Gte(0.8)},
        )

Every method takes ``tenant_id`` / ``include_public`` / ``created_by_user_id`` /
``job_id`` filters (``tenant_id=None`` = all tenants); fix them once with
``nav.scoped(tenant_id="acme")``. See :mod:`scinr.newton.navigation.scope`.

Nothing here mutates the graph. The backend is selected by
``ScinrConfig.graph_backend`` (env ``GRAPH_BACKEND``, default ``"neo4j"``).
"""

from __future__ import annotations

from scinr.newton.exceptions import (
    GraphConnectionError,
    NavigationError,
    UnsupportedOperationError,
)
from scinr.newton.navigation.base import DEFAULT_MAX_DEPTH, GraphNavigator
from scinr.newton.navigation.factory import get_graph_navigator, graph_navigator
from scinr.newton.navigation.filters import (
    Contains,
    EndsWith,
    Eq,
    Gt,
    Gte,
    In,
    IsNotNull,
    IsNull,
    Lt,
    Lte,
    Ne,
    NotIn,
    Op,
    Regex,
    StartsWith,
)
from scinr.newton.navigation.models import (
    AnnotationCoverage,
    CatalogFieldRef,
    CatalogGraph,
    CatalogModelRef,
    CatalogRelation,
    DocumentModelProfile,
    DocumentRef,
    DocumentStats,
    DocumentTree,
    EntityLabelStat,
    EntityRelation,
    ExtractionResultRef,
    ExtractionResultWithNode,
    GraphNode,
    GraphSummary,
    InfoUnitRef,
    InfoUnitWithNode,
    LabeledEntityRef,
    ModelClassStat,
    ModelDecisionRef,
    ModelDecisionWithNode,
    ModelInstanceRef,
    ModelInstanceRelation,
    ModelInstanceTree,
    NodeDescription,
    NodePath,
    NodeSelector,
    OriginalFile,
    PageText,
    PathResult,
    ProposedFieldRef,
    ProposedModelRef,
    RelTypeStat,
    RoleStat,
    ScoredInfoUnit,
    StructureNodeRef,
    StructureNodeSourcePages,
    StructureNodesSourcePages,
    StructureTree,
    Subgraph,
    ThemeRef,
    Triple,
)
from scinr.newton.navigation.scope import Scope, make_scope
from scinr.newton.navigation.scoped import ScopedNavigator

__all__ = [
    # entry points
    "get_graph_navigator",
    "graph_navigator",
    "GraphNavigator",
    "DEFAULT_MAX_DEPTH",
    # scope (tenant / user / job)
    "Scope",
    "ScopedNavigator",
    "make_scope",
    # exceptions
    "NavigationError",
    "GraphConnectionError",
    "UnsupportedOperationError",
    # filter operators
    "Op",
    "Eq", "Ne", "Gt", "Gte", "Lt", "Lte", "In", "NotIn",
    "Contains", "StartsWith", "EndsWith", "Regex", "IsNull", "IsNotNull",
    # return models
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
    "PageText", "StructureNodeSourcePages", "StructureNodesSourcePages", "OriginalFile",
]
