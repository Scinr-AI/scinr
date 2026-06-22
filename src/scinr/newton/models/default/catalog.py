"""
Default theme catalog — fallback for documents with no specific thematic match.
"""
from __future__ import annotations

from .triple import Triple

THEME_DESCRIPTION: str = (
    "Generic fallback for content that does not fit a specific thematic domain; "
    "represents information as subject-predicate-object triples"
)

SELECTABLE_MODELS: list[type] = [Triple]
