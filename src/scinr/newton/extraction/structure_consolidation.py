"""
extraction/structure_consolidation.py

Post-extraction consolidation for Stage 1's ``fast_extraction=True`` path.

When ``fast_extraction=True``, every chunk is extracted independently and in
parallel (each ``extract_chunk()`` call uses ``defer_hierarchy=True`` and never
sees the growing document tree). This module is what stitches those
independent chunk outputs back into one coherent tree:

  1. ``namespace_node_ids()`` — rewrites every node_id (recursively, including
     descendants) to be globally unique across chunks, by prefixing with the
     chunk's first absolute page index.
  2. ``consolidate_structure()`` — identifies "orphan" top-level nodes (nodes
     whose role implies they must nest under something, but whose chunk
     never saw their real parent) and resolves each orphan's true parent
     from the cross-chunk node pool. For large documents, this pool is
     walked in contiguous backward-looking sliding-window batches (each
     batch sees only its own chunks plus the immediately preceding batch as
     context) rather than in a single call, to stay under the model's
     context window; unresolved orphans are carried forward and retried in
     later batches before ever falling back to root.
  3. ``assemble_tree()`` — deterministically applies those decisions (with
     cycle detection) to build the final ``document_structure`` root list.
  4. ``write_map_checkpoint()`` / ``delete_map_checkpoint()`` — crash-safety
     I/O around the (parallel, potentially slow) Map phase; write-then-delete
     only, no automatic resume logic in this pass.

Intentionally fully decoupled from ``compact_extraction.py`` (the
``fast_extraction=False`` code path) — nothing here is imported from or by
that module, and ``compact_extraction.py`` is never touched.

Public API
----------
    def namespace_node_ids(nodes: list[StructureNode], page_number: int) -> None
    async def consolidate_structure(all_chunks: list[list[StructureNode]], llm=None) -> list[ParentDecision]
    def assemble_tree(all_chunks: list[list[StructureNode]], decisions: list[ParentDecision]) -> list[StructureNode]
    def write_map_checkpoint(path: Path, all_chunks: list[list[StructureNode]]) -> None
    def delete_map_checkpoint(path: Path) -> None
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from scinr.newton.config import get_config, get_llm_semaphore, make_system_message
from scinr.newton.models.consolidation import ConsolidationOutput, NodeCandidate, ParentDecision
from scinr.newton.models.document_structure import NodeRole, StructureNode
from scinr.newton.prompts.consolidation_prompt import build_consolidation_prompt
from scinr.newton.utils.llm_repair import extract_raw_payload, run_repair_loop
from scinr.newton.utils.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

# ── Role categories (duplicated from compact_extraction.py's sets by design —
# this module is intentionally fully decoupled from that one; see module
# docstring). ─────────────────────────────────────────────────────────────────
_TOP_LEVEL_ROLES = {NodeRole.SECTION, NodeRole.APPENDIX}
_ORPHAN_ROLES = {
    NodeRole.SUBSECTION,
    NodeRole.TABLE,
    NodeRole.FIELD_GROUP,
    NodeRole.FREEFORM_BLOCK,
}


# ── Namespacing ───────────────────────────────────────────────────────────────


def namespace_node_ids(nodes: list[StructureNode], page_number: int) -> None:
    """Rewrite node_id in-place for *nodes* and every descendant, recursively.

    Each node_id becomes ``f"page-{page_number}::{node.node_id}"``. Does not
    touch ``parent_id`` (the Map-phase LLM never populates it meaningfully
    when ``defer_hierarchy=True`` — left as-is).

    Args:
        nodes: Top-level nodes returned by one ``extract_chunk()`` call.
        page_number: Absolute page index — for multi-page chunks, the caller
            passes the first absolute page index of that chunk's batch for
            every node the chunk returned (simplest rule, accepted tradeoff).
    """
    for node in nodes:
        node.node_id = f"page-{page_number}::{node.node_id}"
        namespace_node_ids(node.children, page_number)


# ── Token estimation (real tiktoken counts — see module comment below) ──────

# There is no public Claude tokenizer in tiktoken; "o200k_base" is the
# deliberately-accepted, more-precise-than-char-proxy stand-in agreed with
# the user for estimating (not exactly matching) Bedrock/Claude token counts.
_TIKTOKEN_ENCODING: Any | None = None


def _get_encoding() -> Any:
    """Lazy singleton for the tiktoken encoding used to estimate token counts.

    Uses o200k_base as an approximation — there is no public Claude tokenizer
    in tiktoken; this is a deliberately-accepted, more-precise-than-char-proxy
    stand-in agreed with the user. Cached at module level so the BPE is loaded
    once per process, not once per call.
    """
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        import tiktoken

        _TIKTOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
    return _TIKTOKEN_ENCODING


def _estimate_tokens(text: str) -> int:
    """Real token count (tiktoken, o200k_base) — see ``_get_encoding()``."""
    return len(_get_encoding().encode(text))


def _iter_all_nodes(nodes: list[StructureNode]):
    """Recursively yield every node in *nodes*, at every depth."""
    for node in nodes:
        yield node
        yield from _iter_all_nodes(node.children)


def _estimate_decision_output_tokens(
    orphans: list[StructureNode], all_chunks: list[list[StructureNode]]
) -> int:
    """Estimate of the LLM's output size for a set of orphan decisions.

    The actual ``decided_parent_id`` values are unknown before the call (that
    is what we're asking the LLM to produce), so this uses the longest
    node_id present anywhere in the pool as a conservative stand-in for every
    decision's ``decided_parent_id`` field, plus a fixed per-entry overhead
    for JSON structure/keys — then measures the resulting representative
    text with the real tokenizer (``_estimate_tokens()``) instead of a
    char-count formula. Only the counting mechanism changed here; the
    content-estimation heuristic (longest-id stand-in) is unchanged.
    """
    all_ids = [n.node_id for chunk in all_chunks for n in _iter_all_nodes(chunk)]
    longest_id = max(all_ids, key=len, default="")
    overhead = '{"node_id": "", "decided_parent_id": ""}'  # per-entry JSON structure/keys stand-in
    representative_text = "".join(
        orphan.node_id + longest_id + overhead for orphan in orphans
    )
    return _estimate_tokens(representative_text)


# ── Prompt rendering ──────────────────────────────────────────────────────────


def _render_pool(
    all_chunks: list[list[StructureNode]], orphan_ids: set[str], start_index: int = 0
) -> str:
    """Render a cross-chunk node pool, grouped by chunk, with each chunk's
    local nesting preserved via indentation. Marks each node as ORPHAN (needs
    a decision) or REFERENCE (already placed, valid as a decision target).

    *all_chunks* may be the full document's chunk list or an already-filtered
    subset (see ``consolidate_structure()``'s sliding-window batching) — this
    function itself only ever iterates whatever it is given. *start_index*
    lets a subset be labelled with its true absolute chunk index (rather than
    always starting the "Chunk N:" numbering at 0) without changing this
    function's core iteration; defaults to 0 to keep the original single-call
    behaviour byte-for-byte unchanged for callers that pass the full list.
    """
    lines: list[str] = []
    for offset, chunk_nodes in enumerate(all_chunks):
        lines.append(f"Chunk {start_index + offset}:")
        _render_nodes(chunk_nodes, orphan_ids, depth=1, lines=lines)
    return "\n".join(lines)


def _render_nodes(
    nodes: list[StructureNode],
    orphan_ids: set[str],
    depth: int,
    lines: list[str],
) -> None:
    indent = "  " * depth
    for node in nodes:
        candidate = NodeCandidate(node_id=node.node_id, role=node.role, title=node.title)
        tag = "ORPHAN" if candidate.node_id in orphan_ids else "REFERENCE"
        lines.append(
            f'{indent}- node_id="{candidate.node_id}" role="{candidate.role}" '
            f'title="{candidate.title or ""}" [{tag}]'
        )
        _render_nodes(node.children, orphan_ids, depth + 1, lines)


def _render_must_decide(orphan_subset: list[StructureNode]) -> str:
    return "\n".join(
        f'- node_id="{n.node_id}" role="{n.role}" title="{n.title or ""}"'
        for n in orphan_subset
    )


def _build_consolidation_human_message(
    pool_text: str, orphan_subset: list[StructureNode]
) -> str:
    must_decide_text = _render_must_decide(orphan_subset)
    return (
        f"<node_pool>\n{pool_text}\n</node_pool>\n\n"
        f"<must_decide>\n{must_decide_text}\n</must_decide>\n\n"
        "Return exactly one ParentDecision per node_id listed in <must_decide>. "
        "node_id values must be copied verbatim from the pool above."
    )


# ── Structured-output schema (lazy singleton, mirrors extraction.py's
# _get_extraction_output_class() caching pattern) ────────────────────────────

# Currently trivial (ConsolidationOutput needs no dynamic Literal/enum
# construction, unlike the extraction schema's theme Literal) — kept as a
# lazy accessor for consistency with extraction.py and as a single reset
# point if a future variant ever needs dynamic construction.
_CONSOLIDATION_OUTPUT_CLASS: type | None = None


def _get_consolidation_output_class() -> type:
    global _CONSOLIDATION_OUTPUT_CLASS
    if _CONSOLIDATION_OUTPUT_CLASS is None:
        _CONSOLIDATION_OUTPUT_CLASS = ConsolidationOutput
    return _CONSOLIDATION_OUTPUT_CLASS


# ── LLM call plumbing ─────────────────────────────────────────────────────────


async def _run_consolidation_call(
    system_prompt: str,
    pool_text: str,
    orphan_subset: list[StructureNode],
    llm: Any,
) -> list[ParentDecision]:
    """Run one consolidation LLM call for *orphan_subset*, bounded by the
    global LLM semaphore for its full duration (initial call + repair loop).

    Returns an empty list (logged) if the call and its repair loop both fail
    — the caller's validation step then treats every orphan in *orphan_subset*
    as having no decision, which falls back to root placement in
    ``assemble_tree()``.
    """
    async with get_llm_semaphore():
        structured_llm = llm.with_structured_output(
            _get_consolidation_output_class(), include_raw=True
        )
        human_text = _build_consolidation_human_message(pool_text, orphan_subset)
        messages = [make_system_message(system_prompt), HumanMessage(content=human_text)]

        result = await with_llm_retry(lambda: structured_llm.ainvoke(messages))

        if result["parsed"] is not None:
            return result["parsed"].decisions

        logger.warning(
            "consolidate_structure: structured output failed for a partition of "
            "%d orphan(s), running repair loop.", len(orphan_subset),
        )
        broken_raw = extract_raw_payload(result["raw"])
        error = result.get("parsing_error", "Unknown parse error")

        repaired = await run_repair_loop(
            schema=_get_consolidation_output_class(),
            initial_raw=broken_raw,
            initial_error=str(error),
            context_label="consolidation",
        )
        if repaired is not None:
            return repaired.decisions

        logger.error(
            "consolidate_structure: LLM call failed for a partition of %d orphan(s) "
            "after repair — all will fall back to root placement.", len(orphan_subset),
        )
        return []


def _validate_decisions(
    decisions: list[ParentDecision], orphan_ids: set[str]
) -> list[ParentDecision]:
    """Deduplicate (first-wins, logged) and log missing decisions.

    Missing decisions are not synthesized here — a missing node_id simply has
    no entry in the returned list, which ``assemble_tree()`` treats as a root
    fallback (matching a decision with ``decided_parent_id=None``).
    """
    seen: set[str] = set()
    deduped: list[ParentDecision] = []
    for decision in decisions:
        if decision.node_id in seen:
            logger.warning(
                "consolidate_structure: duplicate decision for node_id=%r — "
                "keeping the first, discarding the later one.", decision.node_id,
            )
            continue
        seen.add(decision.node_id)
        deduped.append(decision)

    missing = orphan_ids - seen
    for node_id in missing:
        logger.warning(
            "consolidate_structure: no decision returned for orphan node_id=%r — "
            "falling back to root (decided_parent_id=None).", node_id,
        )
    return deduped


# ── Public API ────────────────────────────────────────────────────────────────


def _partition_orphans_for_output(
    orphans: list[StructureNode],
    all_chunks: list[list[StructureNode]],
    output_ceiling: float,
) -> list[list[StructureNode]]:
    """Split *orphans* into sub-partitions so each partition's estimated
    output stays under *output_ceiling*, using the same "longest node_id
    stand-in" heuristic as ``_estimate_decision_output_tokens()`` (which
    always looks at the full, global *all_chunks* — not just whatever batch
    *orphans* happens to come from — as the conservative source of the
    longest node_id). Returns a single-element list (no split) when
    *orphans* already fits comfortably.
    """
    if not orphans:
        return []
    output_tokens_est = _estimate_decision_output_tokens(orphans, all_chunks)
    if output_tokens_est <= output_ceiling:
        return [orphans]
    avg_output_tokens_per_orphan = max(1, output_tokens_est // len(orphans))
    max_orphans_per_call = max(1, int(output_ceiling // avg_output_tokens_per_orphan))
    n_partitions = math.ceil(len(orphans) / max_orphans_per_call)
    logger.warning(
        "consolidate_structure: estimated output tokens (%d) exceed ceiling (%d) — "
        "partitioning %d orphan(s) into %d call(s) of up to %d orphan(s) each.",
        output_tokens_est, output_ceiling, len(orphans), n_partitions, max_orphans_per_call,
    )
    return [
        orphans[i:i + max_orphans_per_call]
        for i in range(0, len(orphans), max_orphans_per_call)
    ]


async def _run_output_partitioned_calls(
    system_prompt: str,
    pool_text: str,
    orphans: list[StructureNode],
    all_chunks: list[list[StructureNode]],
    output_ceiling: float,
    llm: Any,
) -> list[ParentDecision]:
    """Run one or more concurrent ``_run_consolidation_call()``s for
    *orphans*, splitting by the output-size ceiling only (see
    ``_partition_orphans_for_output()``) — every partition shares the same
    *pool_text*, so cross-partition context is preserved.
    """
    partitions = _partition_orphans_for_output(orphans, all_chunks, output_ceiling)
    results = await asyncio.gather(
        *[
            _run_consolidation_call(system_prompt, pool_text, partition, llm)
            for partition in partitions
        ]
    )
    return [d for sub in results for d in sub]


async def consolidate_structure(
    all_chunks: list[list[StructureNode]], llm: Any = None
) -> list[ParentDecision]:
    """Decide the true parent (or root) for every orphan node across *all_chunks*.

    Processes ``all_chunks`` as one or more contiguous batches (a backward-
    looking sliding window, in document order) instead of always rendering
    every chunk's node pool into a single call — this is what keeps large
    documents under the model's context window (see module docstring; this
    redesign fixes a real-world Bedrock/Claude ``ValidationException`` caused
    by the previous single-call design). Each batch's LLM call(s) see only:

      - its own new chunks' rendered pool, and
      - a "backward buffer" consisting of ONLY the immediately preceding
        batch's rendered pool — not an unbounded accumulation of every prior
        batch (see the design note below for why).

    Any orphan whose batch-of-origin resolves it to ``decided_parent_id=None``
    (or that is missing entirely from the parsed response) is not treated as
    final — it is carried forward (``pending_orphans``) and retried in the
    next batch, and the one after that, and so on. Only once the whole
    document has been walked does a still-unresolved orphan fall back to
    root — the same safe fallback this function has always had, just with
    several chances across batches instead of a single one. ``ParentDecision``
    itself is unchanged (still just ``node_id`` + ``decided_parent_id``) —
    the retry-vs-final rule lives entirely in this function's orchestration,
    never in the LLM response schema.

    If the whole document's pool comfortably fits under
    ``cfg.consolidation_max_input_tokens`` in a single batch, this collapses
    to exactly one batch — the same single-call behaviour (prompt, pool
    render, output-size partitioning) this function had before this
    redesign, verified equivalent by test (see ``fits_single_batch`` below).

    Design note — why the backward buffer is not cumulative: carrying every
    prior batch forward indefinitely would silently reintroduce the original
    unbounded-input-size problem for large documents. The accepted
    consequence is that an orphan can only find its real parent if that
    parent is visible in the same batch or the immediately preceding one —
    not further back. In this domain a node's real parent almost always
    appears *earlier* in the document (a section heading is seen before its
    contents), and an unresolved orphan keeps getting retried in every
    subsequent batch anyway, so this covers the general case correctly. The
    rare case of a parent more than one batch away falls back to root, by
    design — the same safe fallback this function already had before this
    change, just reached after several attempts instead of one.

    Args:
        all_chunks: Namespaced top-level node lists, one per chunk, in chunk order.
        llm: Pre-configured LangChain BaseChatModel. Defaults to the configured LLM.

    Returns:
        Flat list of ParentDecision, one per orphan that ever received a
        decision (including a root fallback once every batch has been
        exhausted) — a duplicate response for the same node_id within one
        LLM call is logged, not raised, first-wins (see
        ``_validate_decisions()``, applied once on the final merged list).
    """
    if llm is None:
        from scinr.newton.config import get_llm
        llm = get_llm()

    orphans_by_chunk_index: list[list[StructureNode]] = []
    orphan_ids: set[str] = set()
    total_orphans = 0
    for chunk_nodes in all_chunks:
        chunk_orphans = [node for node in chunk_nodes if node.role in _ORPHAN_ROLES]
        orphans_by_chunk_index.append(chunk_orphans)
        for node in chunk_orphans:
            orphan_ids.add(node.node_id)
        total_orphans += len(chunk_orphans)

    if total_orphans == 0:
        logger.info("consolidate_structure: no orphan nodes found — skipping LLM call.")
        return []

    cfg = get_config()
    output_ceiling = (
        cfg.consolidation_max_output_tokens
        if cfg.consolidation_max_output_tokens is not None
        else cfg.max_tokens * cfg.consolidation_token_safety_margin
    )
    input_ceiling = cfg.consolidation_max_input_tokens

    # ── Decide once, up front, whether the whole document fits in a single
    # batch. This is the simpler of the two alternatives the design allowed
    # for choosing `partial_visibility` (see build_consolidation_prompt()):
    # estimate the full-pool cost with the non-partial-visibility prompt; if
    # it already fits under the ceiling, run the exact pre-redesign
    # single-call path below (verified byte-for-byte equivalent by test).
    # Only when it does not fit do we pay for the partial-visibility prompt
    # variant and the sliding-window loop — chosen over recomputing
    # partial_visibility mid-loop because it is simpler to reason about
    # correctly: one fixed prompt for the whole call, decided once. ──
    single_batch_system_prompt = build_consolidation_prompt(partial_visibility=False)
    full_pool_text = _render_pool(all_chunks, orphan_ids)
    full_input_est = _estimate_tokens(single_batch_system_prompt) + _estimate_tokens(full_pool_text)
    fits_single_batch = input_ceiling is None or full_input_est <= input_ceiling

    if fits_single_batch:
        orphan_refs = [
            orphan for chunk_orphans in orphans_by_chunk_index for orphan in chunk_orphans
        ]
        decisions = await _run_output_partitioned_calls(
            single_batch_system_prompt, full_pool_text, orphan_refs, all_chunks, output_ceiling, llm
        )
        return _validate_decisions(decisions, orphan_ids)

    # ── Sliding-window batching path (input pool does not fit in one call) ─
    system_prompt = build_consolidation_prompt(partial_visibility=True)
    margin_tokens = 500  # headroom for <node_pool>/<must_decide> XML wrapper overhead per call
    available_for_pool = input_ceiling - _estimate_tokens(system_prompt) - margin_tokens
    if available_for_pool <= 0:
        logger.error(
            "consolidate_structure: consolidation_max_input_tokens (%d) leaves no room "
            "for any node pool once the system prompt and margin are reserved — this is "
            "a misconfiguration. Falling back to an emergency minimum pool budget so "
            "batching can still make progress instead of crashing.",
            input_ceiling,
        )
        available_for_pool = 4000  # emergency minimum — never zero or negative.

    n_chunks = len(all_chunks)
    per_chunk_tokens = [
        _estimate_tokens(_render_pool([all_chunks[idx]], orphan_ids, start_index=idx))
        for idx in range(n_chunks)
    ]

    backward_pool_text = ""
    backward_pool_tokens = 0
    pending_orphans: list[StructureNode] = []
    finalized_decisions: dict[str, ParentDecision] = {}

    i = 0
    while i < n_chunks:
        remaining_budget = available_for_pool - backward_pool_tokens
        if per_chunk_tokens[i] > remaining_budget:
            # Atomic best-effort unit — same "best effort, log and proceed"
            # principle this module already used for output partitioning.
            # This is also what guarantees at least one chunk of progress
            # per outer-loop iteration, so the loop can never stall.
            logger.warning(
                "consolidate_structure: chunk index %d alone (~%d tokens) exceeds the "
                "remaining batch budget (~%d tokens after the backward buffer) — "
                "including it anyway as an atomic best-effort batch of size 1.",
                i, per_chunk_tokens[i], remaining_budget,
            )
            batch_indices = [i]
        else:
            batch_indices = []
            batch_tokens = 0
            j = i
            while (
                j < n_chunks
                and backward_pool_tokens + batch_tokens + per_chunk_tokens[j] <= available_for_pool
            ):
                batch_indices.append(j)
                batch_tokens += per_chunk_tokens[j]
                j += 1
            # batch_indices is guaranteed non-empty: the `if` branch above
            # already handles the only case where chunk i itself would not
            # fit, so this loop's first iteration always succeeds.

        batch_chunks = [all_chunks[idx] for idx in batch_indices]
        batch_pool_text = _render_pool(batch_chunks, orphan_ids, start_index=batch_indices[0])

        new_batch_orphans = [
            orphan for idx in batch_indices for orphan in orphans_by_chunk_index[idx]
        ]
        must_decide = pending_orphans + new_batch_orphans

        if must_decide:
            pool_text = (
                f"{backward_pool_text}\n{batch_pool_text}"
                if backward_pool_text
                else batch_pool_text
            )
            batch_decisions = await _run_output_partitioned_calls(
                system_prompt, pool_text, must_decide, all_chunks, output_ceiling, llm
            )

            decided_by_id: dict[str, str | None] = {}
            for decision in batch_decisions:
                if decision.node_id in decided_by_id:
                    logger.warning(
                        "consolidate_structure: duplicate decision within one batch's "
                        "call(s) for node_id=%r — keeping the first.", decision.node_id,
                    )
                    continue
                decided_by_id[decision.node_id] = decision.decided_parent_id

            still_pending: list[StructureNode] = []
            for orphan in must_decide:
                resolved_parent = decided_by_id.get(orphan.node_id)
                if orphan.node_id in decided_by_id and resolved_parent is not None:
                    finalized_decisions[orphan.node_id] = ParentDecision(
                        node_id=orphan.node_id, decided_parent_id=resolved_parent
                    )
                else:
                    # Missing from the response, or explicitly null — not
                    # final yet, retried in the next batch (see docstring).
                    still_pending.append(orphan)
            pending_orphans = still_pending

        # Replace (never accumulate) the backward buffer with this batch —
        # see the "why the backward buffer is not cumulative" design note
        # in this function's docstring.
        backward_pool_text = batch_pool_text
        backward_pool_tokens = _estimate_tokens(batch_pool_text)

        i = batch_indices[-1] + 1

    for orphan in pending_orphans:
        logger.warning(
            "consolidate_structure: exhausted all batches without a parent match for "
            "node_id=%r — falling back to root.", orphan.node_id,
        )
        finalized_decisions[orphan.node_id] = ParentDecision(
            node_id=orphan.node_id, decided_parent_id=None
        )

    return _validate_decisions(list(finalized_decisions.values()), orphan_ids)


def assemble_tree(
    all_chunks: list[list[StructureNode]], decisions: list[ParentDecision]
) -> list[StructureNode]:
    """Deterministically assemble the final ``document_structure`` root list.

    Args:
        all_chunks: Namespaced top-level node lists, one per chunk, in chunk order.
        decisions: ParentDecision list from ``consolidate_structure()``.

    Returns:
        The new document_structure root list (top-level StructureNodes).
    """
    node_by_id: dict[str, StructureNode] = {}
    local_parent_by_id: dict[str, str | None] = {}

    def _walk(nodes: list[StructureNode], parent_id: str | None) -> None:
        for node in nodes:
            node_by_id[node.node_id] = node
            local_parent_by_id[node.node_id] = parent_id
            _walk(node.children, node.node_id)

    for chunk_nodes in all_chunks:
        _walk(chunk_nodes, None)

    # Genuine chunk-top-level orphan ids — same role-based check
    # consolidate_structure() uses to compute orphan_ids. A decision keyed by
    # any other node_id (a SECTION/APPENDIX id, or a nested/non-top-level
    # node's id) is schema-valid but instruction-violating, and must be
    # discarded before it ever reaches decision_by_id/_effective_parent() —
    # otherwise it could silently reroute another orphan's ancestor-chain
    # walk during cycle detection (see M-2).
    orphan_ids: set[str] = {
        node.node_id
        for chunk_nodes in all_chunks
        for node in chunk_nodes
        if node.role in _ORPHAN_ROLES
    }

    # First-wins on duplicates (defensive — consolidate_structure() already
    # deduplicates via _validate_decisions(), but assemble_tree() may be
    # called directly, e.g. from tests, with raw/unvalidated decisions).
    decision_by_id: dict[str, str | None] = {}
    for decision in decisions:
        if decision.node_id not in orphan_ids:
            logger.warning(
                "assemble_tree: discarding decision for node_id=%r — not a "
                "genuine chunk-top-level orphan (schema-valid but "
                "instruction-violating; would otherwise be usable by "
                "_effective_parent() during cycle detection).", decision.node_id,
            )
            continue
        if decision.node_id not in decision_by_id:
            decision_by_id[decision.node_id] = decision.decided_parent_id

    def _effective_parent(node_id: str) -> str | None:
        if node_id in decision_by_id:
            return decision_by_id[node_id]
        return local_parent_by_id.get(node_id)

    max_walk = len(node_by_id) + 1

    def _has_cycle(start_id: str) -> bool:
        visited: set[str] = {start_id}
        current = start_id
        steps = 0
        while True:
            steps += 1
            if steps > max_walk:
                return True
            nxt = _effective_parent(current)
            if nxt is None:
                return False
            if nxt in visited:
                return True
            visited.add(nxt)
            current = nxt

    root: list[StructureNode] = []

    for chunk_nodes in all_chunks:
        for node in chunk_nodes:
            if node.role in _TOP_LEVEL_ROLES:
                root.append(node)
                continue

            target_id = decision_by_id.get(node.node_id)
            if target_id is None or target_id not in node_by_id:
                root.append(node)
                continue

            if _has_cycle(node.node_id):
                logger.warning(
                    "assemble_tree: cycle detected walking the effective-parent "
                    "chain from orphan node_id=%r — falling back to root.",
                    node.node_id,
                )
                root.append(node)
                continue

            node_by_id[target_id].children.append(node)

    _renumber_nodes(root)
    return root


def _renumber_nodes(nodes: list[StructureNode]) -> None:
    """In-place: reassign 1-based appearance_order within each sibling list,
    recursively. Copied (not imported) from compact_extraction.py by design —
    see module docstring for why this module stays fully decoupled from it.
    """
    for order, node in enumerate(nodes, start=1):
        node.appearance_order = order
        if node.children:
            _renumber_nodes(node.children)


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def write_map_checkpoint(path: Path, all_chunks: list[list[StructureNode]]) -> None:
    """Write the namespaced Map-phase output to *path* as JSON.

    Content is a JSON list of chunks, each chunk being its own namespaced
    top-level nodes list (as ``StructureNode.model_dump()`` — local nesting
    fully intact). No automatic read/resume logic — write only.
    """
    payload = [[node.model_dump() for node in chunk_nodes] for chunk_nodes in all_chunks]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def delete_map_checkpoint(path: Path) -> None:
    """Delete the checkpoint file at *path* if it exists, else no-op."""
    if path.exists():
        path.unlink()
