"""
llm_repair.py — Generic LLM structured-output repair loop for scinr-ingest.

Provides run_repair_loop(), a reusable async function that:
  - Drives the repair LLM with with_structured_output
  - Iterates up to MAX_REPAIR_RETRIES times with escalating temperatures
  - Returns the validated Pydantic model instance on success, or None on failure

This is the single authoritative location for:
  - MAX_REPAIR_RETRIES and REPAIR_TEMPERATURES constants
  - extract_raw_payload() helper

Public API
----------
  MAX_REPAIR_RETRIES: int
  REPAIR_TEMPERATURES: list[float]
  extract_raw_payload(raw_message) -> str | dict
  async run_repair_loop(schema, initial_raw, initial_error, context_label) -> BaseModel | None
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from scinr.newton.config import get_repair_llm, make_system_message
from scinr.newton.utils.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

load_dotenv()
# ── Constants ─────────────────────────────────────────────────────────────────

MAX_REPAIR_RETRIES: int = 3
REPAIR_TEMPERATURES: list[float] = [0.0, 0.3, 0.6]

# ── Unified repair system prompt ──────────────────────────────────────────────

REPAIR_SYSTEM_PROMPT = """\
You are a JSON repair agent. You receive a JSON object that failed Pydantic
validation and the exact validation error. Your task is to return a corrected
JSON object that passes validation.

The target schema is already communicated to you via the tool definition.
Do not re-describe or re-interpret the schema — use the tool definition as
the single source of truth for field names, types, and required fields.

## Repair protocol

Execute these steps in order. Do not skip steps.

STEP 1 — Fix JSON syntax errors.
  Parse the malformed output. If it is not valid JSON, fix syntax errors
  (unclosed brackets, trailing commas, unquoted keys, etc.) before proceeding.

STEP 2 — Apply the minimum fix identified by the error message.
  Read the validation error carefully. Fix only what the error identifies.
  Do not alter fields that are not mentioned in the error.

STEP 3 — Apply structural repair rules (in priority order).
  a. Missing required field with no default and no recoverable value
     → remove the entire containing object. Never invent a value.
  b. List field that is null or missing → set to [].
     Exception: if a list field is semantically required to be non-empty
     (the error will say "at least one item required") and no items can be
     recovered → remove the entire containing object.
  c. Enum field with an unrecognised value → map to the closest valid enum
     member listed in the validation error. If no close match exists, use
     the default or most permissive valid value.
  d. Wrong type (e.g. string where int is required) → convert if lossless
     (e.g. "3" → 3, 2.0 → 2). If conversion is not possible → remove the
     containing object.
  e. Unknown keys not present in the schema → remove them.

STEP 4 — Verify cross-field invariants.
  If the error message identifies a cross-field constraint violation (e.g.
  field A must be null when field B is false), apply the minimum change to
  satisfy the constraint. Prefer nulling/clearing the dependent field over
  changing the controlling field.

STEP 5 — Output.
  Return ONLY the corrected JSON object.
  No markdown fences. No commentary. No explanation. No preamble.
  The output must be parseable by json.loads() with no preprocessing.

## Hard constraints

- Fix structural and typing issues only. Do not re-evaluate, re-interpret,
  or modify the semantic content of any field value.
- Do not add new objects (array items, nested objects) that were not present
  in the original output.
- Do not change field values that are not identified as invalid by the error.
- If a required field is missing and its value cannot be recovered from the
  malformed output or inferred unambiguously from surrounding fields in the
  same object → remove the containing object. Never fabricate content.
- If after all repairs the top-level object is structurally unrecoverable
  → return the closest valid empty skeleton (e.g. {"nodes": []} for a
  list-rooted schema, or a null-filled object for a flat schema).\
"""

# Human message template — identical for all repair call sites.
_REPAIR_HUMAN_TEMPLATE = (
    "Validation error:\n{validation_error}\n\nMalformed output:\n{malformed_output}"
)

# ── Public helpers ────────────────────────────────────────────────────────────


def extract_raw_payload(raw_message: Any) -> str | dict:
    """Unwrap tool_use payloads from Bedrock structured output responses.

    Iterates all content blocks to find the first tool_use block. This is the
    most robust approach for Bedrock Converse responses, which may include
    multiple blocks of different types before the tool_use payload.

    Returns the tool_use input dict if found, or the raw content otherwise.
    """
    payload = raw_message.content
    if isinstance(payload, list) and payload:
        for block in payload:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return block["input"]
    return payload


# ── Public API ────────────────────────────────────────────────────────────────


async def run_repair_loop(
    schema: type[BaseModel],
    initial_raw: str | dict,
    initial_error: str,
    context_label: str = "unknown",
) -> BaseModel | None:
    """Attempt to repair a failed structured LLM output using a repair LLM.

    Uses with_structured_output(schema, include_raw=True) with the
    REPAIR_MODEL_ID model. Iterates up to MAX_REPAIR_RETRIES times with
    escalating temperatures. Applies prompt caching on the system message
    when PROMPT_CACHING_ENABLED=true.

    Parameters
    ----------
    schema:
        The Pydantic model class to validate against.
    initial_raw:
        The raw output (str or dict) that failed validation on the primary call.
    initial_error:
        The validation error message from the first failed parse.
    context_label:
        Label used in log messages to identify the call site (e.g. node_id).

    Returns
    -------
    BaseModel | None
        The validated Pydantic model instance on first successful repair,
        or None if all attempts are exhausted.
    """
    system_message = make_system_message(REPAIR_SYSTEM_PROMPT)
    current_raw: str | dict = initial_raw
    current_error: str = initial_error

    for attempt in range(MAX_REPAIR_RETRIES):
        temp = REPAIR_TEMPERATURES[attempt]
        logger.warning(
            "run_repair_loop: attempt %d/%d (temperature=%.1f) for %r",
            attempt + 1,
            MAX_REPAIR_RETRIES,
            temp,
            context_label,
        )

        repair_llm = get_repair_llm(temperature=temp)
        repair_structured_llm = repair_llm.with_structured_output(
            schema, include_raw=True
        )

        human_content = _REPAIR_HUMAN_TEMPLATE.format(
            validation_error=current_error,
            malformed_output=(
                current_raw if isinstance(current_raw, str) else json.dumps(current_raw)
            ),
        )
        messages = [system_message, HumanMessage(content=human_content)]

        try:
            result: dict[str, Any] = await with_llm_retry(
                lambda: repair_structured_llm.ainvoke(messages)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "run_repair_loop: attempt %d/%d raised exception for %r: %s",
                attempt + 1,
                MAX_REPAIR_RETRIES,
                context_label,
                exc,
            )
            current_error = str(exc)
            continue

        if result["parsed"] is not None:
            logger.info(
                "run_repair_loop: repair successful on attempt %d/%d for %r",
                attempt + 1,
                MAX_REPAIR_RETRIES,
                context_label,
            )
            return result["parsed"]

        current_raw = extract_raw_payload(result["raw"])
        current_error = str(result.get("parsing_error") or "Unknown parsing error")
        logger.warning(
            "run_repair_loop: attempt %d/%d failed for %r",
            attempt + 1,
            MAX_REPAIR_RETRIES,
            context_label,
        )

    logger.error(
        "run_repair_loop: all %d attempts exhausted for %r",
        MAX_REPAIR_RETRIES,
        context_label,
    )
    return None
