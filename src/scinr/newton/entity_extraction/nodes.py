"""
entity_extraction/nodes.py — Stage 4 entity extraction helpers and orchestrator.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from scinr.newton.config import get_llm
from scinr.newton.entity_extraction.graph_mapper import (
    write_extraction_subgraph,
    write_triple_subgraph,
)
from scinr.newton.entity_extraction.model_resolver import resolve_model_class
from scinr.newton.entity_extraction.neo4j_ops import mark_info_units_extracted_async
from scinr.newton.entity_extraction.prompts import (
    build_extraction_human_message,
    build_extraction_system_prompt,
)
from scinr.newton.entity_extraction.schema_composer import compose_extraction_schema
from scinr.newton.ingest.config import get_async_driver
from scinr.newton.utils.llm_repair import extract_raw_payload, run_repair_loop
from scinr.newton.utils.llm_retry import with_llm_retry
from scinr.newton.utils.neo4j_concurrency import get_neo4j_semaphore
from scinr.newton.utils.uid import make_uid as _make_uid

load_dotenv()
logger = logging.getLogger(__name__)


# ── Private helper: compose schema ───────────────────────────────────────────

def _compose_schema(target: dict) -> tuple[type | None, str | None]:
    """Resolve model classes and build the composite Pydantic extraction schema.

    If model_class is None (no domain model matched), uses the default Triple
    model directly without going through compose_extraction_schema.

    Parameters
    ----------
    target:
        ExtractionTarget dict with keys: model_class, complementary_models,
        supplementary_fields, node_full_id.

    Returns
    -------
    tuple[type | None, str | None]
        (schema, error) — schema is None on failure, error is None on success.
    """
    # ── Triple (fallback) path ─────────────────────────────────────────────
    if target["model_class"] is None:
        from scinr.newton.models.default.triple import Triple
        logger.info(
            "_compose_schema: no model_class — using default Triple schema for node %r",
            target["node_full_id"],
        )
        return Triple, None

    # ── Normal path ────────────────────────────────────────────────────────
    try:
        primary_class = resolve_model_class(target["model_class"])
    except KeyError as exc:
        logger.error("_compose_schema: cannot resolve primary model: %s", exc)
        return None, f"_compose_schema: unknown model {target['model_class']!r}"

    complementary_classes: list[type] = []
    for cm in target.get("complementary_models") or []:
        class_name = cm.get("model_class", "")
        if not class_name:
            continue
        try:
            complementary_classes.append(resolve_model_class(class_name))
        except KeyError:
            logger.warning(
                "_compose_schema: complementary model %r not found — skipping", class_name
            )

    schema = compose_extraction_schema(
        primary_class=primary_class,
        complementary_classes=complementary_classes,
        supplementary_fields=target.get("supplementary_fields") or [],
    )
    logger.info(
        "_compose_schema: built composite schema %s for node %r",
        schema.__name__,
        target["node_full_id"],
    )
    return schema, None


# ── Private helper: extract entities ─────────────────────────────────────────

async def _extract_entities(
    composite_schema: type,
    info_units: list[dict],
    node_full_id: str,
    semaphore: asyncio.Semaphore,
    node_id: str | None = None,
    node_title: str | None = None,
) -> tuple[Any | None, str | None]:
    """Run the LLM extraction call for a single node.

    The semaphore wraps the entire LLM + repair block.

    Parameters
    ----------
    composite_schema:
        Dynamically built Pydantic class to use as structured output schema.
    info_units:
        List of InfoUnit dicts for this node.
    node_full_id:
        Full StructureNode.id (for logging and error messages).
    semaphore:
        asyncio.Semaphore controlling Bedrock concurrency.

    Returns
    -------
    tuple[Any | None, str | None]
        (extraction, error) — extraction is None on failure, error is None on success.
    """
    system_prompt = build_extraction_system_prompt(composite_schema)
    human_content = build_extraction_human_message(
        info_units, node_id=node_id, node_title=node_title
    )

    llm_structured = get_llm(temperature=0.0).with_structured_output(
        composite_schema, include_raw=True
    )
    _msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    async with semaphore:
        result = await with_llm_retry(lambda: llm_structured.ainvoke(_msgs))
        parsed = result.get("parsed")

        if parsed is not None:
            logger.info("_extract_entities: parsed successfully for node %r", node_full_id)
            return parsed, None

        # Parse failed — enter repair loop
        current_raw = extract_raw_payload(result["raw"])
        current_error = (
            str(result.get("parsing_error")) if result.get("parsing_error") else "Unknown parsing error"
        )
        logger.warning(
            "_extract_entities: parse failed for node %r, starting repair loop", node_full_id
        )

        repaired = await run_repair_loop(
            schema=composite_schema,
            initial_raw=current_raw,
            initial_error=current_error,
            context_label=node_full_id,
        )
        if repaired is not None:
            logger.info("_extract_entities: repair successful for node %r", node_full_id)
            return repaired, None

        logger.error(
            "_extract_entities: all repair attempts exhausted for node %r", node_full_id
        )
        return None, f"extract_entities failed for node {node_full_id}"


# ── Private helper: write entities ───────────────────────────────────────────

async def _write_entities(
    driver,
    target: dict,
    extraction: Any,
    document_name: str,
) -> str | None:
    """Write the extracted entities subgraph to Neo4j.

    Dispatches to write_triple_subgraph when model_class is None (Triple fallback),
    or write_extraction_subgraph for all domain-specific extractions.

    Parameters
    ----------
    driver:
        Async Neo4j driver instance.
    target:
        ExtractionTarget dict with keys: node_full_id, model_class,
        complementary_models.
    extraction:
        Populated instance of the composite schema.
    document_name:
        Neo4j Document.name.

    Returns
    -------
    str | None
        Error message on failure, None on success.
    """
    node_full_id = target["node_full_id"]
    primary_model_class = target["model_class"]  # str | None

    try:
        # ── Triple (fallback) path ─────────────────────────────────────────
        if primary_model_class is None:
            extraction_uid = _make_uid("extraction_result", node_full_id, "triple")
            await write_triple_subgraph(
                driver=driver,
                node_full_id=node_full_id,
                triple_instance=extraction,
                document_name=document_name,
                extraction_uid=extraction_uid,
            )
            logger.info(
                "_write_entities: triple subgraph written for node %r (uid=%s)",
                node_full_id, extraction_uid,
            )

        # ── Normal domain-specific path ────────────────────────────────────
        else:
            extraction_uid = _make_uid("extraction_result", node_full_id, primary_model_class)
            complementary_model_classes = [
                c["model_class"] for c in (target.get("complementary_models") or [])
                if c.get("model_class")
            ]
            await write_extraction_subgraph(
                driver=driver,
                node_full_id=node_full_id,
                composite_instance=extraction,
                primary_model_class=primary_model_class,
                complementary_model_classes=complementary_model_classes,
                document_name=document_name,
                extraction_uid=extraction_uid,
            )
            logger.info(
                "_write_entities: subgraph written for node %r (uid=%s)",
                node_full_id, extraction_uid,
            )

        return None

    except Exception as exc:
        logger.error("_write_entities: Neo4j write failed for %r: %s", node_full_id, exc)
        return f"write_entities failed for {node_full_id}: {exc}"


# ── Private helper: mark extracted ───────────────────────────────────────────

async def _mark_extracted(driver, node_full_id: str) -> str | None:
    """Mark all InfoUnits of the node as extracted.

    Parameters
    ----------
    driver:
        Async Neo4j driver instance.
    node_full_id:
        Full StructureNode.id as stored in Neo4j.

    Returns
    -------
    str | None
        Error message on failure, None on success.
    """
    try:
        await mark_info_units_extracted_async(driver, node_full_id)
        return None
    except Exception as exc:
        logger.error(
            "_mark_extracted: failed for node %r: %s", node_full_id, exc
        )
        return f"mark_extracted failed for {node_full_id}: {exc}"


# ── Slim orchestrator ─────────────────────────────────────────────────────────

async def process_single_extraction_target(
    target: dict,
    document_name: str,
    bedrock_semaphore: asyncio.Semaphore,
) -> dict:
    """Process a single ExtractionTarget through the full extraction cycle.

    Encapsulates: compose schema → extract entities (with semaphore) →
    write entities → mark extracted.
    Intended for use with asyncio.gather() for intra-document parallelism.

    The Bedrock call (extract_entities + repair loop) is executed while holding
    ``bedrock_semaphore``, bounding total concurrent LLM calls.
    Neo4j writes (write_entities, mark_extracted) are each wrapped in
    ``get_neo4j_semaphore()`` to prevent connection pool saturation under
    parallel load. The semaphore is released between the two writes so other
    coroutines can interleave their Neo4j work.

    Parameters
    ----------
    target:
        ExtractionTarget dict as returned by fetch_extraction_targets.
    document_name:
        Neo4j Document.name — passed through to write functions.
    bedrock_semaphore:
        Process-wide asyncio.Semaphore controlling Bedrock concurrency.

    Returns
    -------
    dict
        ``{"node_full_id": str, "extraction": Any | None, "error": str | None}``
    """
    node_full_id = target["node_full_id"]
    driver = get_async_driver()
    neo4j_semaphore = get_neo4j_semaphore()

    composite_schema, schema_error = _compose_schema(target)
    if composite_schema is None:
        return {"node_full_id": node_full_id, "extraction": None, "error": schema_error}

    extraction, extract_error = await _extract_entities(
        composite_schema,
        target.get("info_units") or [],
        node_full_id,
        bedrock_semaphore,
        node_id=target.get("node_id"),
        node_title=target.get("node_title"),
    )
    if extraction is None:
        return {"node_full_id": node_full_id, "extraction": None, "error": extract_error}

    # ── Write entities (bounded by Neo4j semaphore) ───────────────────────────
    async with neo4j_semaphore:
        write_error = await _write_entities(driver, target, extraction, document_name)
    # ── Mark extracted (bounded by Neo4j semaphore) ───────────────────────────
    async with neo4j_semaphore:
        mark_error = await _mark_extracted(driver, node_full_id)

    error = write_error or mark_error
    return {"node_full_id": node_full_id, "extraction": extraction, "error": error}
