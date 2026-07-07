"""
entity_extraction/state.py — TypedDicts for the entity extraction module.
"""
from __future__ import annotations

from typing import TypedDict


class ExtractionTarget(TypedDict):
    """All data needed to run entity extraction for one StructureNode."""
    node_full_id: str            # e.g. "Amox 500 mg/0000/m3::2::5_3/5_3_1"
    node_id: str | None          # short identifier, e.g. "5_3_1"
    node_title: str | None
    model_class: str | None      # primary matched_model_class name; None → fallback Triple extraction
    complementary_models: list[dict]   # list of {model_class, coverage_note}
    supplementary_fields: list[dict]   # list of {field_name, field_type, description, required}
    info_units: list[dict]       # ordered by iu.order; each has {uid, title, description, order}
