"""
models/consolidation.py

Pydantic models for the Stage 1 ``fast_extraction=True`` post-extraction
consolidation step (see ``extraction/structure_consolidation.py``).

These models are used as the structured-output schema for the consolidation
LLM call, which decides the parent for every "orphan" node produced by the
parallel Map phase (one independent ``extract_chunk()`` call per chunk, with
``defer_hierarchy=True``).

Public API
----------
    class NodeCandidate(StrictModel)
    class ParentDecision(StrictModel)
    class ConsolidationOutput(StrictModel)
"""

from __future__ import annotations

import json

from pydantic import Field, model_validator

from scinr.newton.models.document_structure import NodeRole, StrictModel


class NodeCandidate(StrictModel):
    """
    One node from the pool rendered in the consolidation prompt — either an
    orphan awaiting a parent decision or a reference-only node already placed
    by the Map phase (a valid decision target for other nodes).

    Used as the typed intermediate representation when rendering the node
    pool for the consolidation prompt (see
    ``structure_consolidation._render_nodes()``) — not part of the LLM
    structured-output schema itself (the pool is still rendered as prompt
    text, not sent as structured input).
    """

    node_id: str = Field(
        description="Already-namespaced node_id (see namespace_node_ids())."
    )
    role: NodeRole = Field(description="Structural role of this node.")
    title: str | None = Field(default=None, description="Heading text, if any.")


class ParentDecision(StrictModel):
    """One consolidation LLM decision: which node (if any) is the parent of a given orphan."""

    node_id: str = Field(
        description=(
            "The orphan's node_id, echoed back verbatim exactly as it appeared "
            "in the prompt's node pool. Never invented or reformatted."
        ),
    )
    decided_parent_id: str | None = Field(
        default=None,
        description=(
            "node_id of the chosen parent, which may be any node_id present in the "
            "full pool (any chunk, any depth). Set to null when this node belongs "
            "at the document root."
        ),
    )


class ConsolidationOutput(StrictModel):
    """
    LLM structured-output schema for the consolidation call — a flat list of
    parent decisions, one per orphan node the prompt asked it to resolve.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_decisions_from_string(cls, values):
        """Coerce decisions from JSON string to list if the LLM serialized it incorrectly.

        Mirrors DocumentStructure._coerce_nodes_from_string() — LLMs occasionally
        serialize a list field as a JSON string instead of an actual JSON array.
        """
        if isinstance(values, dict) and isinstance(values.get("decisions"), str):
            values = dict(values)
            values["decisions"] = json.loads(values["decisions"])
        return values

    decisions: list[ParentDecision] = Field(
        default_factory=list,
        description=(
            "One ParentDecision per orphan node this call was asked to resolve. "
            "Every orphan listed in the prompt's 'must decide' section must have "
            "exactly one corresponding entry here."
        ),
    )
