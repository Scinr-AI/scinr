"""Second call of ``info_units_mode="auto"``.

In auto mode the structure-extraction call decides, per node, ``verbatim`` or ``summary``.
Once the whole document tree is final (after step F), this module builds the InfoUnits:

* ``verbatim`` nodes get the original paragraphs as InfoUnits (no LLM).
* ``summary`` nodes get one extra LLM call (or several, when the text is long) that
  writes the InfoUnits from the original paragraphs of the node.

A failed second call never fails the document: the node falls back to ``verbatim``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from scinr.newton.config import get_llm_semaphore, get_prompt_family, make_system_message
from scinr.newton.extraction.paragraphs import (
    ParagraphIndex,
    build_verbatim_info_units,
    merge_ranges,
    strip_markers,
)
from scinr.newton.models.document_structure import (
    Document,
    InfoUnit,
    InfoUnitsOutput,
    StructureNode,
)
from scinr.newton.prompts.info_units_prompt import build_info_units_prompt
from scinr.newton.utils.llm_repair import extract_raw_payload, run_repair_loop
from scinr.newton.utils.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)


@dataclass
class AutoStats:
    summary_nodes: int = 0
    verbatim_nodes: int = 0
    calls: int = 0
    fallbacks: int = 0


def group_paragraphs(
    numbers: Sequence[int], index: ParagraphIndex, max_chars: int
) -> list[list[int]]:
    """Consecutive groups of *numbers* whose texts add up to at most *max_chars* (D-15).

    A paragraph longer than *max_chars* is a group of its own. Never returns an empty group.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_len = 0
    for n in numbers:
        length = len(index.text(n))
        if current and current_len + length > max_chars:
            groups.append(current)
            current, current_len = [], 0
        current.append(n)
        current_len += length
    if current:
        groups.append(current)
    return groups


def build_info_units_human_message(
    node: StructureNode,
    texts: Sequence[str],
    *,
    title_path: str,
    user_context: str,
) -> str:
    prefix = f"<user_context>\n{user_context}\n</user_context>\n\n" if user_context else ""
    body = "\n\n".join(texts)
    return (
        f"{prefix}"
        "<node>\n"
        f"<title_path>{title_path}</title_path>\n"
        f"<title>{node.title or ''}</title>\n"
        f"<role>{node.role}</role>\n"
        "</node>\n\n"
        f"<paragraphs>\n{body}\n</paragraphs>"
    )


def _paragraph_numbers(node: StructureNode, index: ParagraphIndex) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for r in merge_ranges(node.paragraph_ranges):
        for n in range(r.start, r.end + 1):
            if 1 <= n <= index.total and n not in seen:
                seen.add(n)
                numbers.append(n)
    return numbers


async def extract_node_info_units(
    node: StructureNode,
    index: ParagraphIndex,
    *,
    title_path: str,
    user_context: str,
    llm: Any,
    max_input_chars: int,
    stats: AutoStats,
    log_context: str,
) -> list[InfoUnit] | None:
    """InfoUnits of one ``summary`` node, written by the LLM from its original paragraphs.

    Returns ``None`` when a call cannot be parsed even after the repair loop. An empty list
    is a valid answer (the LLM found nothing worth an InfoUnit).
    """
    groups = group_paragraphs(_paragraph_numbers(node, index), index, max_input_chars)
    units: list[InfoUnit] = []
    structured_llm = llm.with_structured_output(InfoUnitsOutput, include_raw=True)
    system_message = make_system_message(
        build_info_units_prompt(get_prompt_family(stage="extraction")), stage="extraction"
    )

    for group in groups:
        # The message is built inside the semaphore so only `llm_concurrency` texts are alive.
        async with get_llm_semaphore():
            human = build_info_units_human_message(
                node,
                [index.text(n) for n in group],
                title_path=title_path,
                user_context=user_context,
            )
            messages = [system_message, HumanMessage(content=human)]
            result = await with_llm_retry(lambda m=messages: structured_llm.ainvoke(m))
            stats.calls += 1
            parsed = result["parsed"]
            if parsed is None:
                logger.warning(
                    "Auto info units: structured output failed for node %r (%s), "
                    "running repair loop.",
                    node.node_id,
                    log_context,
                )
                parsed = await run_repair_loop(
                    schema=InfoUnitsOutput,
                    initial_raw=extract_raw_payload(result["raw"]),
                    initial_error=str(result.get("parsing_error", "Unknown parse error")),
                    context_label="auto-info-units",
                )
                if parsed is None:
                    return None
        units.extend(parsed.info_units)

    for order, unit in enumerate(units):
        unit.title = strip_markers(unit.title)
        unit.description = strip_markers(unit.description)
        unit.order = order
    return units


def _walk_with_path(
    nodes: Iterable[StructureNode], ancestors: tuple[StructureNode, ...] = ()
) -> Iterable[tuple[StructureNode, str]]:
    path = " › ".join(a.title or a.node_id for a in ancestors)
    for node in nodes:
        yield node, path
        yield from _walk_with_path(node.children, (*ancestors, node))


async def resolve_auto_info_units(
    document: Document,
    index: ParagraphIndex,
    *,
    llm: Any,
    max_input_chars: int,
    verbatim_max_chars: int,
    stats: AutoStats,
) -> None:
    """Build the InfoUnits of every node of *document* according to its decided mode.

    Afterwards every node has ``info_units_mode`` ``"summary"`` or ``"verbatim"``.
    """
    log_context = f"document={document.document_name!r}"
    user_context = document.context_instructions or ""

    def verbatim_units(node: StructureNode) -> list[InfoUnit]:
        return build_verbatim_info_units(
            node.paragraph_ranges,
            index,
            node_title=node.title or node.node_id,
            max_chars=verbatim_max_chars,
        )

    pending: list[tuple[StructureNode, str]] = []
    for node, path in _walk_with_path(document.document_structure):
        if not node.paragraph_ranges:
            node.info_units_mode = "verbatim"
            node.info_units = []
            stats.verbatim_nodes += 1
        elif node.info_units_mode == "verbatim":
            node.info_units = verbatim_units(node)
            stats.verbatim_nodes += 1
        else:
            pending.append((node, path))

    results = await asyncio.gather(
        *(
            extract_node_info_units(
                node,
                index,
                title_path=path,
                user_context=user_context,
                llm=llm,
                max_input_chars=max_input_chars,
                stats=stats,
                log_context=log_context,
            )
            for node, path in pending
        ),
        return_exceptions=True,
    )

    for (node, _path), result in zip(pending, results, strict=True):
        if isinstance(result, list):
            node.info_units_mode = "summary"
            node.info_units = result
            stats.summary_nodes += 1
            continue
        reason = result if isinstance(result, BaseException) else "all repair attempts failed"
        logger.warning(
            "auto info units failed for node %r (%s): %s — falling back to verbatim",
            node.node_id,
            log_context,
            reason,
        )
        node.info_units_mode = "verbatim"
        node.info_units = verbatim_units(node)
        stats.fallbacks += 1
        stats.verbatim_nodes += 1

    logger.info(
        "Auto info units for %r: %d summary nodes, %d verbatim nodes, %d calls, %d fallbacks",
        document.document_name,
        stats.summary_nodes,
        stats.verbatim_nodes,
        stats.calls,
        stats.fallbacks,
    )
