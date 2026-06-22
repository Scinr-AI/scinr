"""
annotation/state.py — TypedDicts for the annotation module.
"""
from __future__ import annotations

from typing import Any, TypedDict


class NodeContext(TypedDict):
    """Context for a single StructureNode fetched from Neo4j."""

    node_id: str           # e.g. "5_3_1"
    full_id: str           # e.g. "Amox 500 mg/0000/m3::2::5_3/5_3_1" (Neo4j :StructureNode.id)
    title: str
    role: str
    info_units: list[dict[str, Any]]    # list of InfoUnit dicts (info_unit_id, title, description, order)


class AnnotationState(TypedDict):
    """Result state returned by _run_annotation_parallel and run_annotation_agent."""

    document_name: str
    nodes_to_annotate: list[NodeContext]
    errors: list[str]
    context_instructions: str | None

