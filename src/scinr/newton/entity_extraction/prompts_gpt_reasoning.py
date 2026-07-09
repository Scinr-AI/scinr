"""
entity_extraction/prompts_gpt_reasoning.py — OpenAI reasoning models variant.

Targets GPT-5.5, o3, and o4-mini via the OpenAI API. These models reason
internally before producing output, making step-by-step elicitation and
self-verification checklists counterproductive. Uses Markdown section headers
and goal-based language instead.

For non-reasoning GPT models (GPT-4o, GPT-4.1, GPT-4.5), use the GENERIC
family — the reasoning-model patterns here offer no benefit on standard
instruction-following models and may degrade performance.
"""

from __future__ import annotations


_COMPLEMENTARY_GOAL_ADDENDUM = """

This schema contains COMPLEMENTARY sub-models (the optional nested fields). Treat them as strongly preferred: populate every sub-field you can find evidence for in the provided text. Return null for a complementary sub-model only when none of its fields appear anywhere in the provided text."""


def build_extraction_system_prompt(composite_schema: type) -> str:
    """Build the OpenAI reasoning models entity extraction system prompt.

    Places the schema description in a '## Schema' section within '# Instructions'
    using Markdown headers and goal-based language suitable for
    GPT-5.5, o3, and o4-mini.
    """
    from scinr.newton.entity_extraction.prompts import (
        _build_schema_description,
        _has_complementary_fields,
    )

    schema_desc = _build_schema_description(composite_schema)
    complementary_addendum = (
        _COMPLEMENTARY_GOAL_ADDENDUM if _has_complementary_fields(composite_schema) else ""
    )

    return f"""# Instructions

You are a precise information extraction system for structured documents.

## Extraction Goal

Extract structured data from the provided document content that exactly matches the schema in the Schema section below.{complementary_addendum}

An extraction is correct when:
- Every field contains only values explicitly stated in the provided text.
- Fields whose information is absent are set to null or empty list — never guessed.
- Exact values from the text are preserved without paraphrase or normalisation of units, names, or measurements.
- List fields contain all instances mentioned in the text, not just the first.
- List fields are complete across all provided information units — content fragmented across multiple units is consolidated, not truncated to the first unit.
- Discriminated union fields (e.g. record_type: 'primary' | 'supplementary') are assigned based on explicit cues in the text.

## Schema

{schema_desc}

Return ONLY the structured JSON extraction. Respond directly. Start your response with `{{`."""
