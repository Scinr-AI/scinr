"""Extraction models of the golden harness: one of every shape the graph writers handle."""
from __future__ import annotations

from pydantic import Field

from scinr.newton.models.base import ExtractionModel


class GoldenSpec(ExtractionModel):
    """Unkeyed nested model (positional uid, CREATE path)."""

    material: str | None = Field(default=None, description="Material.")
    weight: str | None = Field(default=None, description="Weight with units.")


class GoldenPart(ExtractionModel):
    """Keyed model used in a list (MERGE path with list index)."""

    part_code: str | None = Field(
        default=None,
        description="Part code.",
        json_schema_extra={"entity_label": "GoldenPartCode", "instance_key": True},
    )
    description: str | None = Field(default=None, description="Part description.")


class GoldenSupplier(ExtractionModel):
    """Keyed model, target of instance relationships."""

    supplier_code: str | None = Field(
        default=None,
        description="Supplier code.",
        json_schema_extra={"instance_key": True},
    )
    supplier_name: str | None = Field(
        default=None,
        description="Supplier name.",
        json_schema_extra={"entity_label": "GoldenSupplierName"},
    )


class GoldenProduct(ExtractionModel):
    """Product with entities, nested models, lists and both kinds of relationships."""

    product_name: str | None = Field(
        default=None,
        description="Product name.",
        json_schema_extra={
            "entity_label": "GoldenProductName",
            "field_relationships": [{"to_field": "manufacturer", "rel_type": "MADE_BY"}],
        },
    )
    manufacturer: str | None = Field(
        default=None,
        description="Manufacturer.",
        json_schema_extra={"entity_label": "GoldenManufacturer"},
    )
    category: str | None = Field(default=None, description="Plain scalar.")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags.",
        json_schema_extra={"entity_label": "GoldenTag"},
    )
    supplier_codes: list[str] = Field(
        default_factory=list,
        description="Supplier codes.",
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "GoldenSupplier",
                    "join_via": {"supplier_codes": "supplier_code"},
                    "rel_type": "SUPPLIED_BY",
                }
            ]
        },
    )
    spec: GoldenSpec | None = Field(default=None, description="Specification.")
    primary_part: GoldenPart | None = Field(default=None, description="Main part.")
    parts: list[GoldenPart] = Field(default_factory=list, description="Parts.")


class GoldenRow(ExtractionModel):
    """Plain tabular row model."""

    sku: str | None = Field(
        default=None,
        description="SKU.",
        json_schema_extra={"entity_label": "GoldenSku"},
    )
    item: str | None = Field(default=None, description="Item.")
    qty: str | None = Field(default=None, description="Quantity.")


class GoldenPlace(ExtractionModel):
    """Normalized place."""

    canonical: str | None = Field(default=None, description="Canonical place name.")
    country: str | None = Field(default=None, description="Country.")


class GoldenNormRow(ExtractionModel):
    """Tabular row model with a normalized field."""

    name: str | None = Field(default=None, description="Name.")
    city: str | None = Field(default=None, description="City.")
    place: GoldenPlace | None = Field(
        default=None,
        description="Normalized place.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["name", "city"],
        },
    )
