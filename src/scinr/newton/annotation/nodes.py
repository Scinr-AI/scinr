from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from scinr.newton.annotation.models import AnnotationDecision
from scinr.newton.annotation.neo4j_ops import fetch_node_context, write_annotation
from scinr.newton.annotation.prompts import build_annotation_decision_prompt
from scinr.newton.config import get_llm, make_system_message
from scinr.newton.ingest.config import get_async_driver
from scinr.newton.utils.llm_repair import extract_raw_payload, run_repair_loop
from scinr.newton.utils.llm_retry import with_llm_retry
from scinr.newton.utils.neo4j_concurrency import get_neo4j_semaphore

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────
load_dotenv()


# ── Helper: build node context XML ───────────────────────────────────────────

def _build_node_context_xml(ctx: dict, depth: int = 0) -> str:
    """
    Render a NodeContext dict as an XML string for the LLM, up to depth 3.

    Each <info_unit> block contains only <title> and <description>.

    Args:
        ctx: NodeContext-shaped dict as returned by fetch_node_context.
        depth: Current recursion depth (0 = root node passed to decide_model).
    """
    node_id = ctx.get("node_id") or ""
    title = ctx.get("title") or ""
    role = ctx.get("role") or ""
    info_units: list[dict] = ctx.get("info_units") or []
    children: list[dict] = ctx.get("children") or []

    lines: list[str] = []

    lines.append("<node>")
    lines.append(f"  <id>{node_id}</id>")
    lines.append(f"  <title>{title}</title>")
    lines.append(f"  <role>{role}</role>")

    # ── InfoUnits ─────────────────────────────────────────────────────────────
    lines.append("  <info_units>")
    for iu in info_units:
        lines.append("    <info_unit>")
        lines.append(f"      <title>{iu.get('title') or ''}</title>")
        lines.append(f"      <description>{iu.get('description') or ''}</description>")
        lines.append("    </info_unit>")
    lines.append("  </info_units>")

    # ── Children (recursive, max depth 3) ────────────────────────────────────
    if children and depth < 3:
        lines.append("  <children>")
        for child in children:
            child_xml = _build_node_context_xml(child, depth=depth + 1)
            # Indent child XML by 4 spaces
            for child_line in child_xml.splitlines():
                lines.append(f"    {child_line}")
        lines.append("  </children>")

    lines.append("</node>")
    return "\n".join(lines)


def _build_annotation_human_message(ctx: dict, user_context: str = "") -> str:
    """Builds the HumanMessage content for the annotation LLM call.

    Prepends a <user_context> block when user_context is non-empty.
    """
    node_context_xml = _build_node_context_xml(ctx)
    prefix = f"<user_context>\n{user_context}\n</user_context>\n\n" if user_context else ""
    return f"{prefix}<node_context>\n{node_context_xml}\n</node_context>"


# ── Private helper: fetch node context ───────────────────────────────────────

async def _fetch_node_context(node_data: dict, driver) -> dict | None:
    """Fetch full context (InfoUnits + qualifying children) for a node.

    Args:
        node_data: Node dict with keys: full_id, node_id, title, role.
        driver: Async Neo4j driver instance.

    Returns:
        NodeContext-shaped dict on success, None on error.
    """
    node_id = node_data.get("node_id", "unknown")
    try:
        context = await fetch_node_context(
            driver=driver,
            full_node_id=node_data["full_id"],
            node_id=node_data["node_id"],
            title=node_data.get("title"),
            role=node_data.get("role"),
        )
        logger.info(
            "_fetch_node_context: loaded context for node '%s' (%d info_units, %d children)",
            node_id,
            len(context.get("info_units", [])),
            len(context.get("children", [])),
        )
        return context
    except Exception as exc:
        logger.error(
            "_fetch_node_context: failed to fetch context for '%s': %s",
            node_id,
            exc,
        )
        return None


# ── Private helper: read theme ────────────────────────────────────────────────

def _read_theme(node_data: dict) -> str:
    """Read the theme already classified during extraction from the node data.

    Returns the theme string stored on the node, defaulting to 'default'.
    """
    return node_data.get("theme", "default")


# ── Private helper: decide model ──────────────────────────────────────────────

async def _decide_model(
    ctx: dict,
    node_id: str,
    theme: str,
    user_context: str,
    semaphore: asyncio.Semaphore,
) -> tuple[AnnotationDecision | None, str | None]:
    """Run the LLM annotation decision for a single node.

    Resolves the theme-specific catalog, builds the prompt, calls the LLM,
    and runs the repair loop on parse failure. The semaphore wraps the entire
    LLM + repair block.

    Args:
        ctx: NodeContext-shaped dict as returned by _fetch_node_context.
        node_id: Node identifier string (for logging).
        theme: Theme path string (e.g. "pharmaceutical").
        user_context: Optional freeform context string to prepend to the human message.
        semaphore: asyncio.Semaphore controlling Bedrock concurrency.

    Returns:
        (decision, error) — exactly one of the two will be non-None on failure.
    """
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    theme_node = theme_registry.find_best_theme(theme)
    decision_prompt = build_annotation_decision_prompt(theme_node)
    human_content = _build_annotation_human_message(ctx, user_context=user_context)

    _msgs = [
        make_system_message(decision_prompt),
        HumanMessage(content=human_content),
    ]
    llm_structured = get_llm(temperature=0).with_structured_output(
        AnnotationDecision, include_raw=True
    )

    async with semaphore:
        result = await with_llm_retry(lambda: llm_structured.ainvoke(_msgs))
        parsed: AnnotationDecision | None = result["parsed"]

        if parsed is not None:
            logger.info(
                "_decide_model: decision for node '%s' (theme=%s, class=%s, confidence=%s)",
                node_id, theme_node.path, parsed.matched_model_class, parsed.confidence,
            )
            return parsed, None

        # Parse failed — enter repair loop
        current_raw = extract_raw_payload(result["raw"])
        current_error = (
            str(result["parsing_error"]) if result.get("parsing_error") else "Unknown parsing error"
        )
        logger.warning(
            "_decide_model: parse failed for node '%s', starting repair loop", node_id
        )

        repaired = await run_repair_loop(
            schema=AnnotationDecision,
            initial_raw=current_raw,
            initial_error=current_error,
            context_label=node_id,
        )
        if repaired is not None:
            logger.info("_decide_model: repair successful for node '%s'", node_id)
            return repaired, None

        logger.error("_decide_model: all repair attempts exhausted for node '%s'", node_id)
        return None, f"decide_model failed for node {node_id}"


# ── Private helper: write decision ────────────────────────────────────────────

async def _write_decision(
    driver,
    full_id: str,
    decision: AnnotationDecision,
    document_name: str,
    node_id: str,
) -> str | None:
    """Write the AnnotationDecision to Neo4j.

    Args:
        driver: Async Neo4j driver instance.
        full_id: Full StructureNode.id as stored in Neo4j.
        decision: Parsed AnnotationDecision to persist.
        document_name: Neo4j Document.name.
        node_id: Node identifier string (for logging).

    Returns:
        Error message on failure, None on success.
    """
    try:
        await write_annotation(driver, full_id, decision, document_name)
        logger.info(
            "_write_decision: wrote AnnotationDecision for node '%s' (class=%s, confidence=%s)",
            node_id,
            decision.matched_model_class,
            decision.confidence,
        )
        return None
    except Exception as exc:
        logger.error(
            "_write_decision: Neo4j write failed for '%s': %s", node_id, exc
        )
        return f"write failed for {node_id}: {exc}"


# ── Slim orchestrator ─────────────────────────────────────────────────────────

async def process_single_annotation_node(
    node_data: dict,
    document_name: str,
    bedrock_semaphore: asyncio.Semaphore,
    user_context: str = "",
) -> dict:
    """Process a single StructureNode through the full annotation cycle.

    Encapsulates: fetch context → read theme → decide model → write decision.
    Intended for use with asyncio.gather() for intra-document parallelism.

    The Bedrock call (decide_model + repair loop) is executed while holding
    ``bedrock_semaphore``, bounding total concurrent Bedrock calls.
    Neo4j reads and writes are each wrapped in ``get_neo4j_semaphore()`` to
    prevent connection pool saturation under parallel load.

    Args:
        node_data: Node dict as returned by fetch_nodes_to_annotate.
            Expected keys: full_id, node_id, title, role, theme.
        document_name: Neo4j Document.name — passed through to write_annotation.
        bedrock_semaphore: Process-wide asyncio.Semaphore controlling Bedrock concurrency.
        user_context: Optional freeform context string prepended to the LLM human message.

    Returns:
        ``{"node_id": str, "decision": AnnotationDecision | None, "error": str | None}``
    """
    node_id = node_data.get("node_id", "unknown")
    driver = get_async_driver()
    neo4j_semaphore = get_neo4j_semaphore()

    # ── Fetch context (bounded by Neo4j semaphore) ────────────────────────────
    # The semaphore is acquired here at the root level to bound root-node
    # concurrency; individual session opens inside _fetch_node_context are also
    # guarded by the same semaphore for the recursive sub-queries.
    async with neo4j_semaphore:
        ctx = await _fetch_node_context(node_data, driver)
    if ctx is None:
        return {"node_id": node_id, "decision": None, "error": f"fetch_context failed for {node_id}"}

    theme = _read_theme(node_data)
    decision, error = await _decide_model(ctx, node_id, theme, user_context, bedrock_semaphore)

    if decision is not None:
        # ── Write decision (bounded by Neo4j semaphore) ───────────────────────
        async with neo4j_semaphore:
            write_error = await _write_decision(driver, node_data["full_id"], decision, document_name, node_id)
        if write_error:
            error = write_error

    return {"node_id": node_id, "decision": decision, "error": error}
