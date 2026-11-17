"""Extraction models of bench_graph.py (never used by real documents)."""
from __future__ import annotations

from pydantic import Field

from scinr.newton.models.base import ExtractionModel


class BenchItem(ExtractionModel):
    """Item with one entity, one plain field and a list."""

    name: str | None = Field(
        default=None, description="Name.", json_schema_extra={"entity_label": "BenchName"}
    )
    note: str | None = Field(default=None, description="Note.")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags.",
        json_schema_extra={"entity_label": "BenchTag"},
    )


class BenchWide(ExtractionModel):
    """Twelve plain columns (tabular rows)."""

    c00: str | None = Field(default=None, description="c00")
    c01: str | None = Field(default=None, description="c01")
    c02: str | None = Field(default=None, description="c02")
    c03: str | None = Field(default=None, description="c03")
    c04: str | None = Field(default=None, description="c04")
    c05: str | None = Field(default=None, description="c05")
    c06: str | None = Field(default=None, description="c06")
    c07: str | None = Field(default=None, description="c07")
    c08: str | None = Field(default=None, description="c08")
    c09: str | None = Field(default=None, description="c09")
    c10: str | None = Field(default=None, description="c10")
    c11: str | None = Field(default=None, description="c11")
