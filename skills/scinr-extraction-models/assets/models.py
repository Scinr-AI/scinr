"""<Theme name> extraction models."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from scinr.newton.models.base import ExtractionModel

# ── KEY HELPERS ───────────────────────────────────────────────────────────────


def canonical_code(v: Any) -> Any:
    """Canonical form for item codes. Apply it on EVERY model that stores or references one."""
    return re.sub(r"\s+", "", v).upper() if isinstance(v, str) else v


# ── ENUMS ─────────────────────────────────────────────────────────────────────


class StatusEnum(str, Enum):  # noqa: UP042 — (str, Enum) is the scinr convention
    """
    Lifecycle status of an item.

    Values:
      active   — Item is currently in use.
      inactive — Item has been retired or replaced.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"


# ── TARGET MODELS (referenced via instance_relationships) ────────────────────


class ConditionModel(ExtractionModel):
    """A single numbered condition that applies to one catalogue item."""

    item_code: str = Field(
        default="",
        description=(
            "Code of the item this condition belongs to, exactly as written in the document "
            "(e.g. 'A-001'). Empty string if the parent item is not stated."
        ),
        json_schema_extra={"instance_key": True},
    )
    condition_id: str = Field(
        default="",
        description=(
            "Identifier of the condition as numbered in the document, such as '1', '2' or 'A'. "
            "Empty string when the condition is not numbered."
        ),
        json_schema_extra={"instance_key": True},
    )
    text: str | None = Field(
        default=None,
        description=(
            "Verbatim wording of the condition as stated in the document, without its number. "
            "None if only the identifier is given."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _canonical_keys(cls, data: Any) -> Any:
        # Same helper as ItemModel, so shells created from ItemModel.condition_ids merge here.
        if isinstance(data, dict):
            data = {**data, "item_code": canonical_code(data.get("item_code") or "")}
        return data


# ── MAIN MODELS ───────────────────────────────────────────────────────────────


class ItemModel(ExtractionModel):
    """A single item entry from a regulatory catalogue document."""

    item_code: str = Field(
        ...,
        description=(
            "Unique alphanumeric code as written in the document (e.g. 'A-001', 'Q.I.a.1'). "
            "Never omit the prefix. This field is required."
        ),
        json_schema_extra={"entity_label": "ItemCode", "instance_key": True},
    )
    status: StatusEnum | None = Field(
        default=None,
        description=(
            "Lifecycle status of the item per StatusEnum: 'active' if currently valid, "
            "'inactive' if retired. None if the document does not state it."
        ),
    )
    condition_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers of the conditions listed for this item, e.g. ['1', '2']. "
            "Each value is one identifier. Empty list when the item has no conditions."
        ),
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "ConditionModel",
                    "join_via": {"item_code": "item_code", "condition_ids": "condition_id"},
                    "rel_type": "HAS_CONDITION",
                }
            ]
        },
    )

    @model_validator(mode="before")
    @classmethod
    def _canonical_keys(cls, data: Any) -> Any:
        # mode="before" edits the input once; "after" + self.x = ... recurses (validate_assignment).
        if isinstance(data, dict) and "item_code" in data:
            data = {**data, "item_code": canonical_code(data["item_code"])}
        return data


# ── LIST WRAPPERS ─────────────────────────────────────────────────────────────


class ItemModelList(ExtractionModel):
    """Use when the section defines TWO OR MORE item entries in a list or table."""

    items: list[ItemModel] = Field(
        default_factory=list,
        description=(
            "List of item entries; each element is one distinct item. Use this model instead "
            "of ItemModel when the section is a table or list covering two or more items."
        ),
    )
