"""Catalog for the <theme name> theme."""
from __future__ import annotations

from .models import ConditionModel, ItemModel, ItemModelList

THEME_DESCRIPTION: str = (
    "Regulatory catalogue documents listing item entries with codes (e.g. 'A-001'), "
    "lifecycle statuses and numbered conditions. Covers item cross-references and "
    "status tracking. Distinct from procedural guidelines and Q&A documents."
)

SELECTABLE_MODELS: list[type] = [
    ItemModelList,   # multi-item sections — most specific first
    ItemModel,       # single-item sections
    ConditionModel,  # sections that define conditions separately
]
