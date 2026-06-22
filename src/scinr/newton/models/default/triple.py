"""
Default theme models — generic RDF-style triple for content that does not fit a specific domain.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TripleItem(BaseModel):
    """A single subject-predicate-object statement."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str
    predicate: str
    object: str


class Triple(BaseModel):
    """Generic RDF-style extraction for content that does not fit a specific domain model. Extracts all subject-predicate-object statements found in the content."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    triples: list[TripleItem] = Field(
        ..., description="All subject-predicate-object statements extracted from the content."
    )
