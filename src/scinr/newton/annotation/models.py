from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ThemeClassification(BaseModel):
    """Structured result of the theme classification step for a StructureNode."""

    justification: str = Field(
        description=(
            "1-3 sentences explaining why this theme was selected, referencing "
            "specific content from the node's title, role, or information units."
        )
    )
    theme: str = Field(
        description=(
            "The path of the most specific matching theme from the available themes list. "
            "Must exactly match one of the theme paths shown in the prompt "
            "(e.g. 'pharmaceutical', 'structural_specs', 'default'). "
            "Use 'default' if no specific theme clearly applies."
        )
    )


class ProposedField(BaseModel):
    field_name: str = Field(description="Python-style field name, e.g. 'sequence_analysis_methods'")
    field_type: str = Field(description="Python type hint string, e.g. 'list[str]', 'str | None'")
    description: str = Field(description="What this field captures in the document")
    required: bool = Field(description="Whether this field should be required (not Optional)")


class ComplementaryModel(BaseModel):
    """A secondary extraction model that partially covers this node's content."""

    model_class: str = Field(
        description="Exact CamelCase class name of the complementary model from the catalog."
    )
    coverage_note: str = Field(
        description=(
            "What specific content this complementary model covers "
            "that the primary model does not."
        )
    )


class AnnotationDecision(BaseModel):
    """
    The result of the annotation agent's decision for a single StructureNode.
    Written to Neo4j as: (:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_complementary_models_from_string(cls, values):
        """Coerce complementary_models from JSON string to list if the LLM serialized it incorrectly."""
        if isinstance(values, dict) and isinstance(values.get("complementary_models"), str):
            values = dict(values)
            try:
                values["complementary_models"] = json.loads(values["complementary_models"])
            except (json.JSONDecodeError, ValueError):
                values["complementary_models"] = []
        return values

    @model_validator(mode="after")
    def _enforce_cross_field_invariants(self) -> AnnotationDecision:
        """Silently coerce mutually exclusive fields based on matched_model_class."""
        if self.matched_model_class is None:
            self.supplementary_fields = []
        else:
            self.proposed_schema_name = None
            self.proposed_schema_fields = []
        return self

    matched_model_class: str | None = Field(
        default=None,
        description=(
            "Exact CamelCase class name of the best-matching extraction model from the catalog, "
            "or null if no model has >= 50% of its own fields satisfied by this node's InfoUnits."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description=(
            "Confidence in the match: high (>= 75% of the model's fields satisfied by the node), "
            "medium (50-74% of the model's fields satisfied), "
            "low (< 50% of the model's fields satisfied but still selected as best available option). "
            "Always 'low' when matched_model_class is null."
        ),
    )
    rationale: str = Field(
        description=(
            "2-10 sentences explaining why this model was selected (or why no model fits). "
            "Reference the specific requirements and the node title."
        )
    )
    coverage_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "List of requirements or topics present in the node that are NOT well covered "
            "by the matched model. Empty list if the match is complete."
        ),
    )
    supplementary_fields: list[ProposedField] = Field(
        default_factory=list,
        description=(
            "Fields that would need to be ADDED to the matched model to cover requirements "
            "present in coverage_gaps that are NOT addressed by any catalog model "
            "(neither the matched model nor any complementary model). "
            "Only populated when matched_model_class is NOT null AND coverage_gaps is non-empty "
            "AND no complementary_model covers those gaps. "
            "MUST be empty list when matched_model_class is null — use proposed_schema_fields instead."
        ),
    )
    complementary_models: list[ComplementaryModel] = Field(
        default_factory=list,
        description=(
            "Secondary models that cover content not addressed by the primary match. "
            "Must be empty when matched_model_class is null."
        ),
    )
    propose_new_model: bool = Field(
        default=False,
        description=(
            "True when no existing catalog model has >= 50% of its own fields satisfied "
            "by this node's InfoUnits. Must be True when matched_model_class is null."
        ),
    )
    proposed_model_description: str | None = Field(
        default=None,
        description=(
            "2-4 sentence description of what a new model would need to capture "
            "if propose_new_model is True. Must be null when propose_new_model is False."
        ),
    )

    proposed_schema_name: str | None = Field(
        default=None,
        description=(
            "If matched_model_class is null, the proposed Python class name for a new model "
            "(e.g. 'BioinformaticsPipelineSection'). Must be null if matched_model_class is not null."
        ),
    )
    proposed_schema_fields: list[ProposedField] = Field(
        default_factory=list,
        description=(
            "If proposing a new schema (matched_model_class is null), the list of fields "
            "the new model should have. Must be empty list if matched_model_class is not null."
        ),
    )