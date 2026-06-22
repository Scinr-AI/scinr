"""tabular/models.py — Pydantic models for LLM structured outputs in the tabular pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ColumnFieldMapping(BaseModel):
    """Mapping of one CSV/XLSX column to one field of a Pydantic model.

    A single column can appear in multiple ColumnFieldMapping entries — one entry
    per target field it maps to (across the primary model, supplementary fields,
    or any complementary model).
    """

    column_name: str = Field(description="Exact column header from the file.")
    model_field_name: str = Field(
        description=(
            "Snake_case field name of the matched Pydantic model that best captures "
            "the values in this column. Use '__extra__' if no field fits."
        )
    )
    target_model: str = Field(
        default="primary",
        description=(
            "Which model this field belongs to. Use 'primary' for the primary model, "
            "'supplementary' for supplementary fields, or the exact CamelCase class name "
            "of the complementary model (e.g. 'BatchAnalysis') for complementary model fields."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description=(
            "Confidence that this column maps to this field. "
            "Use 'high' for strong matches, 'medium' for plausible matches, "
            "'low' only when there is no good match (use __extra__ as model_field_name in that case)."
        ),
    )
    notes: str | None = Field(
        default=None,
        description="Optional note on why this mapping was chosen or any caveat.",
    )


class ColumnMapping(BaseModel):
    """Full column-to-field mapping for a single sheet/table.

    A column may appear more than once in 'mappings' when it maps to multiple
    targets (e.g. a primary field at high confidence AND a complementary model
    field at medium confidence).  Columns that cannot be mapped to any model
    field use model_field_name='__extra__' (exactly one such entry per column).
    """

    mappings: list[ColumnFieldMapping] = Field(
        description=(
            "One or more entries per column. Every column must appear at least once. "
            "A column can appear multiple times if it maps to fields in different models."
        )
    )
    unmapped_columns: list[str] = Field(
        default_factory=list,
        description="Column names mapped to '__extra__' (for quick reference).",
    )
