"""
extraction.py

Async LLM extraction for one sliding-window chunk.

Public API
----------
    async def extract_chunk(
        prev_page: str,
        curr_pages: list[str],
        active_hierarchy: str,
        llm: Any | None = None,
    ) -> list[StructureNode]

Architecture
------------
Phase 1: call the LLM with full context (prev_page + curr_pages + active_hierarchy).
  - On success: return parsed nodes.
  - On parse failure: delegate to run_repair_loop() from utils.llm_repair.
    On repair success: return nodes.
    On repair failure: fall through to Phase 2.

Phase 2: retry Phase 1 with prev_page="" (no previous page context).
  - Same success/repair/failure logic as Phase 1.
  - If Phase 2 also exhausts all retries: raise ExtractionMaxRetriesError.
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from scinr.newton.config import get_prompt_family, make_system_message
from scinr.newton.extraction.paragraphs import (
    ParagraphIndex,
    ParagraphStats,
    merge_ranges,
    strip_markers,
)
from scinr.newton.models.document_structure import (
    DocumentStructureSummaryOutput,
    DocumentStructureVerbatimOutput,
    InfoUnitsMode,
    StructureNode,
    StructureNodeSummaryOutput,
    StructureNodeVerbatimOutput,
)
from scinr.newton.prompts.system_prompt import build_extraction_prompt
from scinr.newton.utils.llm_repair import MAX_REPAIR_RETRIES, extract_raw_payload, run_repair_loop
from scinr.newton.utils.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

load_dotenv()
_PHASE1_TEMPERATURE = 0.0
_INFO_UNITS_MODES = ("summary", "verbatim")


# ── Error class ───────────────────────────────────────────────────────────────


class ExtractionMaxRetriesError(Exception):
    """Raised when all extraction and repair attempts fail for a chunk."""


# ── Dynamic output class (built once at module load) ──────────────────────────


def _build_extraction_output_class(info_units_mode: str = "summary") -> type:
    """Builds the LLM output schema with `theme` validated as a dynamic Literal.

    The node and document base classes depend on *info_units_mode*: summary mode uses
    ``StructureNodeSummaryOutput`` / ``DocumentStructureSummaryOutput``; verbatim mode uses
    the ``Verbatim`` variants, which have no ``info_units``.

    Called lazily, once per mode. The returned class is used as the schema
    for with_structured_output(), which translates the Literal type into an
    `{"enum": [...]}` constraint in the Bedrock tool_use JSON schema — preventing
    the LLM from hallucinating theme values.

    Since the node class is recursive, we use a subclass approach rather than
    pydantic.create_model to preserve the recursive children reference.
    """
    from typing import Literal

    from scinr.newton.utils.theme_registry import get_theme_registry
    if info_units_mode == "verbatim":
        node_base: type = StructureNodeVerbatimOutput
        document_base: type = DocumentStructureVerbatimOutput
    else:
        node_base = StructureNodeSummaryOutput
        document_base = DocumentStructureSummaryOutput

    theme_paths = tuple(get_theme_registry().get_all_theme_paths())
    if not theme_paths:
        # Fallback: no themes discovered, use the plain document output
        return document_base

    # make_literal unpacks a tuple as separate Literal arguments
    # e.g. make_literal("a", "b") → Literal["a", "b"]
    def make_literal(*args: str):
        return Literal[args]

    ThemeLiteral = make_literal(*theme_paths)

    class ExtractionStructureNode(node_base):  # type: ignore[valid-type,misc]
        """Node output subclass with theme validated as a Literal enum."""
        theme: ThemeLiteral = theme_paths[0]  # type: ignore[valid-type]
        children: list[ExtractionStructureNode] = []

    ExtractionStructureNode.model_rebuild()

    import typing as _typing
    # Verify the Literal was constructed correctly (not as Literal[tuple])
    _theme_annotation = ExtractionStructureNode.model_fields["theme"].annotation
    _actual_args = set(_typing.get_args(_theme_annotation))
    _expected_args = set(theme_paths)
    assert _actual_args == _expected_args, (
        f"Theme Literal was not built correctly. "
        f"Expected args: {_expected_args}, got: {_actual_args}. "
        f"This may be a Python version incompatibility with Literal[tuple] subscript. "
        f"Please report this as a bug."
    )

    class ExtractionDocumentStructure(document_base):  # type: ignore[valid-type,misc]
        """Document output subclass using ExtractionStructureNode for theme validation."""
        nodes: list[ExtractionStructureNode]

    return ExtractionDocumentStructure


# Lazy caches — built on first use after configure() has been called.
# Reset by reset_extraction_cache() which is called from reset_theme_registry().
_EXTRACTION_OUTPUT_CLASSES: dict[str, type] = {}  # key: info_units_mode
_EXTRACTION_PROMPTS: dict[tuple[str, str], str] = {}  # key: (family.value, info_units_mode)


def _get_extraction_output_class(info_units_mode: str = "summary") -> type:
    if info_units_mode not in _EXTRACTION_OUTPUT_CLASSES:
        _EXTRACTION_OUTPUT_CLASSES[info_units_mode] = _build_extraction_output_class(
            info_units_mode
        )
    return _EXTRACTION_OUTPUT_CLASSES[info_units_mode]


def _get_extraction_prompt(info_units_mode: str = "summary") -> str:
    family = get_prompt_family(stage="extraction")
    key = (family.value, info_units_mode)
    if key not in _EXTRACTION_PROMPTS:
        from scinr.newton.utils.theme_registry import get_theme_registry
        _EXTRACTION_PROMPTS[key] = build_extraction_prompt(
            get_theme_registry().build_theme_section_for_extraction_prompt(),
            family=family,
            info_units_mode=info_units_mode,
        )
    return _EXTRACTION_PROMPTS[key]


def reset_extraction_cache() -> None:
    """Invalidate lazy caches. Called from reset_theme_registry() after configure()."""
    _EXTRACTION_OUTPUT_CLASSES.clear()
    _EXTRACTION_PROMPTS.clear()


# ── Private helpers ───────────────────────────────────────────────────────────


_DEFERRED_HIERARCHY_NOTE = "(none)"
"""Sentinel substituted for `<active_hierarchy>`'s content when `defer_hierarchy=True`.

Deliberately reuses the same `"(none)"` sentinel `get_active_hierarchy()` already
returns for an empty tree, rather than a novel placeholder, so every existing
prompt rule that already handles `"(none)"` correctly (not just the ORPHANED
branch) applies uniformly — no new/undefined state is introduced anywhere in
the prompt.
"""

_FAST_EXTRACTION_MODE_NOTE = (
    "<extraction_mode>\n"
    "fast — this chunk is processed in isolation, without visibility into the "
    "real document hierarchy beyond the current batch of pages. Because of "
    "this, when heading-level ambiguity cannot be resolved using "
    "<previous_page> alone, prefer the lower-level role (subsection or "
    "freeform_block) over section. A continuation misclassified as a new "
    "top-level section here becomes permanent and cannot be corrected by a "
    "later consolidation step.\n"
    "</extraction_mode>"
)
"""`<extraction_mode>` block injected into the human message when `defer_hierarchy=True`.

Counteracts the extraction prompt's own ambiguous-role fallback, which — absent
this note — defaults to the higher-level role (section) whenever heading-level
ambiguity cannot be resolved from `<previous_page>` alone. That default is safe
in the legacy sequential path (a wrong SECTION can still be corrected by later
chunks with full context) but is actively harmful under `fast_extraction=True`:
each chunk is processed in isolation with no downstream visibility into the
real hierarchy, structure_consolidation.py never re-examines nodes already
classified as SECTION/APPENDIX (they are top-level roles by definition), so a
continuation-of-an-open-section wrongly promoted to SECTION here is a permanent,
unrecoverable error. This note only appears when `defer_hierarchy=True` — the
legacy path's human message is left byte-for-byte unchanged.
"""


def _build_human_message(
    prev_page: str,
    curr_pages: list[str],
    active_hierarchy: str,
    user_context: str = "",
    *,
    defer_hierarchy: bool = False,
) -> str:
    """Builds the human message for the extraction LLM call.

    Args:
        prev_page: Context page (previous batch's last page). Empty string for first batch.
        curr_pages: One or more consecutive pages to extract from.
        active_hierarchy: Current structural path string from get_active_hierarchy().
        user_context: Optional free-text context supplied by the caller (e.g. document
            description, domain hints). When non-empty it is prepended as a
            ``<user_context>`` XML block before the rest of the message.
        defer_hierarchy: When ``True``, the ``<active_hierarchy>`` block content is
            replaced with the ``"(none)"`` sentinel instead of *active_hierarchy*
            (which is ignored in this mode — callers may pass ``""``). Used by the
            ``fast_extraction=True`` parallel-chunk path, where cross-chunk hierarchy
            resolution is deferred to a separate consolidation step. Defaults to
            ``False`` (unchanged behavior — the real *active_hierarchy* is always used).
            When ``True``, an additional ``<extraction_mode>`` block (see
            ``_FAST_EXTRACTION_MODE_NOTE``) is also inserted immediately before the
            ``<active_hierarchy>`` block, telling the model to prefer the lower-level
            role on unresolved heading-level ambiguity. When ``False``, no
            ``<extraction_mode>`` block is added at all — the legacy message is
            byte-for-byte unchanged.
    """
    prefix = f"<user_context>\n{user_context}\n</user_context>\n\n" if user_context else ""
    pages_xml = "\n\n".join(
        f"<page_{i + 1}>\n{page}\n</page_{i + 1}>"
        for i, page in enumerate(curr_pages)
    )
    hierarchy_content = _DEFERRED_HIERARCHY_NOTE if defer_hierarchy else active_hierarchy
    extraction_mode_block = f"{_FAST_EXTRACTION_MODE_NOTE}\n\n" if defer_hierarchy else ""
    return (
        f"{prefix}"
        f"<previous_page>\n{prev_page}\n</previous_page>\n\n"
        f"{pages_xml}\n\n"
        f"{extraction_mode_block}"
        f"<active_hierarchy>\n{hierarchy_content}\n</active_hierarchy>"
    )


# ── Public API ────────────────────────────────────────────────────────────────


def _assign_page_ids_recursive(
    nodes: list[StructureNode],
    page_ids: list[str],
) -> None:
    """Assign source_page_ids to all nodes in the subtree recursively.

    Called immediately after a successful LLM parse to stamp each node
    with the MongoDB page IDs of the batch that produced it.  Any value
    previously set by the LLM (the field defaults to []) is overwritten.

    Args:
        nodes: Top-level nodes returned by the LLM for this chunk.
        page_ids: MongoDB ObjectId strings of the pages in the current batch.
    """
    for node in nodes:
        node.source_page_ids = list(page_ids)
        _assign_page_ids_recursive(node.children, page_ids)


def _walk(nodes: list[StructureNode]):
    """Yield every node of the subtrees, parents before children."""
    for node in nodes:
        yield node
        yield from _walk(node.children)


def _postprocess_nodes(
    llm_nodes: list,
    *,
    curr_page_ids: list[str] | None,
    allowed: tuple[int, int] | None,
    has_index: bool,
    paragraph_stats: ParagraphStats | None,
    log_context: str,
) -> list[StructureNode]:
    """Convert the LLM output nodes to StructureNode, clean markers and validate ranges.

    Args:
        llm_nodes: Nodes parsed from the LLM output (output-schema instances).
        curr_page_ids: Page ids of the chunk; stamped as ``source_page_ids`` fallback.
        allowed: Inclusive (lo, hi) paragraph numbers shown to the LLM, or None.
        has_index: Whether anchors were shown to the LLM at all.
        paragraph_stats: Counter shared by the document's chunks, if any.
        log_context: Label included in the discard warnings.
    """
    nodes = [StructureNode.model_validate(n.model_dump()) for n in llm_nodes]
    for node in _walk(nodes):
        if node.title:
            node.title = strip_markers(node.title) or None
        for unit in node.info_units:
            unit.title = strip_markers(unit.title)
            unit.description = strip_markers(unit.description)

        valid = []
        for r in node.paragraph_ranges:
            if not has_index:
                continue
            if r.start > r.end:
                reason = "start > end"
            elif allowed is None:
                reason = "no paragraphs in the current pages"
            elif r.start < allowed[0] or r.end > allowed[1]:
                reason = f"outside the current pages (allowed {allowed[0]}-{allowed[1]})"
            else:
                valid.append(r)
                continue
            logger.warning(
                "Discarded paragraph range %d-%d of node %r (%s): %s",
                r.start, r.end, node.node_id, log_context, reason,
            )
            if paragraph_stats is not None:
                paragraph_stats.discarded_ranges += 1
        node.paragraph_ranges = merge_ranges(valid)
    if curr_page_ids:
        _assign_page_ids_recursive(nodes, curr_page_ids)
    return nodes


async def extract_chunk(
    prev_page: str,
    curr_pages: list[str],
    active_hierarchy: str,
    llm: Any | None = None,
    curr_page_ids: list[str] | None = None,
    user_context: str = "",
    *,
    defer_hierarchy: bool = False,
    paragraph_index: ParagraphIndex | None = None,
    curr_positions: list[int] | None = None,
    info_units_mode: InfoUnitsMode = "summary",
    paragraph_stats: ParagraphStats | None = None,
    log_context: str = "",
) -> list[StructureNode]:
    """
    Extract structured nodes from one sliding-window chunk.

    Args:
        prev_page: Markdown text of the previous page (may be empty string for the first chunk).
        curr_pages: Markdown text of one or more consecutive pages to extract from.
        active_hierarchy: Formatted string describing the rightmost open-node path from the growing
            document tree. Produced by ``get_active_hierarchy()``.
        llm: Pre-configured LangChain BaseChatModel instance. If None, uses the
            configured LLM from scinr_config. The function uses it for Phase 1;
            Phase 2 reuses the same instance with adjusted temperature via bind().
        curr_page_ids: MongoDB ``page_id`` strings for the pages in ``curr_pages``.
            When provided, all returned nodes (and their children recursively)
            will have ``source_page_ids`` set to this list.
            Defaults to ``None`` (backward-compatible — ``source_page_ids`` stays ``[]``).
        user_context: Optional free-text context supplied by the caller (e.g. document description,
            domain hints). When non-empty, prepended as a ``<user_context>`` XML block at
            the top of every HumanMessage sent to the LLM in both Phase 1 and Phase 2.
            Defaults to ``""`` (no block added).
        defer_hierarchy: When ``True``, both Phase 1 and Phase 2 human messages replace
            the ``<active_hierarchy>`` block with the ``"(none)"`` sentinel instead
            of *active_hierarchy* (which is ignored — callers may pass ``active_hierarchy=""``).
            Used by the ``fast_extraction=True`` parallel-chunk path. An
            ``<extraction_mode>`` block is also inserted before ``<active_hierarchy>``
            in this mode, instructing the model to prefer the lower-level role
            (subsection/freeform_block) over section when heading-level ambiguity
            cannot be resolved from ``<previous_page>`` alone — see
            ``_FAST_EXTRACTION_MODE_NOTE``. Defaults to ``False`` (unchanged behavior;
            no ``<extraction_mode>`` block is added).
        paragraph_index: Global paragraph numbering of the document. When provided, the
            current pages are shown to the LLM with ``[[n]]`` anchors (not
            ``prev_page``) and the returned nodes carry validated ``paragraph_ranges``.
        curr_positions: Positions of ``curr_pages`` inside ``paragraph_index``. Required
            with ``paragraph_index``; must have the same length as ``curr_pages``.
        info_units_mode: ``"summary"`` (the LLM writes InfoUnits) or ``"verbatim"`` (the
            LLM only cites paragraph ranges; requires ``paragraph_index``).
        paragraph_stats: Counter updated with every discarded paragraph range.
        log_context: Label (e.g. the chunk) included in range-discard warnings.

    Returns:
        Merged, parsed nodes for this chunk.

    Raises:
        ExtractionMaxRetriesError: If both phases exhaust all repair retries without a successful parse.
        ValueError: On inconsistent paragraph_index / curr_positions / info_units_mode arguments.
    """
    if info_units_mode not in _INFO_UNITS_MODES:
        raise ValueError(f"Unknown info_units_mode {info_units_mode!r}")
    if paragraph_index is not None and (
        curr_positions is None or len(curr_positions) != len(curr_pages)
    ):
        raise ValueError(
            "paragraph_index requires curr_positions with one position per page in curr_pages"
        )
    if info_units_mode == "verbatim" and paragraph_index is None:
        raise ValueError("info_units_mode='verbatim' requires paragraph_index (the anchors)")

    allowed = (
        paragraph_index.number_range(curr_positions) if paragraph_index is not None else None
    )
    pages_for_prompt = (
        [paragraph_index.marked_page(p) for p in curr_positions]
        if paragraph_index is not None
        else curr_pages
    )
    output_class = _get_extraction_output_class(info_units_mode)

    def postprocess(llm_nodes: list) -> list[StructureNode]:
        return _postprocess_nodes(
            llm_nodes,
            curr_page_ids=curr_page_ids,
            allowed=allowed,
            has_index=paragraph_index is not None,
            paragraph_stats=paragraph_stats,
            log_context=log_context,
        )

    if llm is None:
        from scinr.newton.config import get_llm
        llm = get_llm(stage="extraction")
    structured_llm = llm.with_structured_output(output_class, include_raw=True)

    # ── Phase 1: full context ──────────────────────────────────────────────────
    human_text_p1 = _build_human_message(
        prev_page, pages_for_prompt, active_hierarchy, user_context, defer_hierarchy=defer_hierarchy
    )
    messages_p1 = [
        make_system_message(_get_extraction_prompt(info_units_mode), stage="extraction"),
        HumanMessage(content=human_text_p1),
    ]

    logger.info("Phase 1: calling LLM for chunk extraction.")
    result_p1 = await with_llm_retry(lambda: structured_llm.ainvoke(messages_p1))

    if result_p1["parsed"] is not None:
        logger.info("Phase 1: structured output parsed successfully.")
        return postprocess(result_p1["parsed"].nodes)

    logger.warning("Phase 1: structured output failed, running repair loop.")
    broken_json_p1 = extract_raw_payload(result_p1["raw"])
    error_p1 = result_p1.get("parsing_error", "Unknown parse error")

    repaired_p1 = await run_repair_loop(
        schema=output_class,
        initial_raw=broken_json_p1,
        initial_error=str(error_p1),
        context_label="extraction-phase1",
    )
    if repaired_p1 is not None:
        return postprocess(repaired_p1.nodes)

    # ── Phase 2: without previous page ────────────────────────────────────────
    logger.warning(
        "All Phase 1 repair attempts failed. "
        "Falling back to Phase 2 (no previous page context)."
    )

    structured_llm_p2 = llm.with_structured_output(output_class, include_raw=True)

    human_text_p2 = _build_human_message(
        "", pages_for_prompt, active_hierarchy, user_context, defer_hierarchy=defer_hierarchy
    )
    messages_p2 = [
        make_system_message(_get_extraction_prompt(info_units_mode), stage="extraction"),
        HumanMessage(content=human_text_p2),
    ]

    logger.info("Phase 2: calling LLM without previous page context.")
    result_p2 = await with_llm_retry(lambda: structured_llm_p2.ainvoke(messages_p2))

    if result_p2["parsed"] is not None:
        logger.info("Phase 2: structured output parsed successfully.")
        return postprocess(result_p2["parsed"].nodes)

    logger.warning("Phase 2: structured output also failed, running repair loop.")
    broken_json_p2 = extract_raw_payload(result_p2["raw"])
    error_p2 = result_p2.get("parsing_error", "Unknown parse error")

    repaired_p2 = await run_repair_loop(
        schema=output_class,
        initial_raw=broken_json_p2,
        initial_error=str(error_p2),
        context_label="extraction-phase2",
    )
    if repaired_p2 is not None:
        return postprocess(repaired_p2.nodes)

    logger.error(
        "All Phase 1 and Phase 2 repair attempts exhausted. "
        "Raising ExtractionMaxRetriesError."
    )
    raise ExtractionMaxRetriesError(
        f"Extraction failed in both phases after {MAX_REPAIR_RETRIES} repair "
        "attempts each."
    )
