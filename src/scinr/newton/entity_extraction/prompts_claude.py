"""
entity_extraction/prompts_claude.py — Claude/Sonnet-optimized extraction prompt.
"""
from __future__ import annotations

from scinr.newton.entity_extraction.prompts import (
    _build_schema_description,
    _has_complementary_fields,
)


def build_extraction_system_prompt(composite_schema: type) -> str:
    """Build the Claude-optimized system prompt for entity extraction."""
    schema_description = _build_schema_description(composite_schema)

    base_rules = """## Extraction Rules
1. Extract ONLY information explicitly stated in the provided text. Do NOT infer, fabricate, or hallucinate values.
2. If a field's information is not present in the text, leave it as null or empty list (never guess).
3. Preserve exact values from the text — do not paraphrase or normalise units, names, or measurements.
4. For list fields, extract all instances mentioned in the text, not just the first one.
5. The text may be fragmented (multiple information units) — read all of it before extracting.
6. For discriminated union fields (e.g. record_type: 'primary' | 'supplementary'), choose based on explicit cues in the text."""

    complementary_rule = ""
    if _has_complementary_fields(composite_schema):
        complementary_rule = """
7. This schema contains COMPLEMENTARY sub-models (the optional nested fields).
   Treat them as strongly preferred: populate every sub-field you can find evidence
   for in the text. Only return null for a complementary sub-model if NONE of its
   fields appear anywhere in the provided text."""

    return f"""You are a precise information extraction system for structured documents.

Your task is to extract structured data from the provided document content according to the schema below.

{base_rules}{complementary_rule}

## Schema to extract
{schema_description}

Respond ONLY with the structured extraction. Do not add explanatory text."""
