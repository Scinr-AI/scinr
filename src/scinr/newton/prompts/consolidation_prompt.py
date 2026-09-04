"""
prompts/consolidation_prompt.py

System prompt for the Stage 1 ``fast_extraction=True`` post-extraction
consolidation LLM call (see ``extraction/structure_consolidation.py``).

Design choice — single generic prompt, no per-PromptFamily variants
---------------------------------------------------------------------
Unlike ``system_prompt.py`` (the per-chunk extraction prompt dispatcher),
this module intentionally does NOT branch on ``PromptFamily``. The
consolidation task is narrow and mechanical — given a flat pool of already-
extracted nodes (grouped by chunk, with each chunk's local nesting already
fixed), decide the parent (or root) for a small "orphan" subset — and does
not benefit from the family-specific reasoning scaffolding (XML protocols
for Claude, goal-based Markdown for GPT reasoning models) that the main
extraction prompt uses to steer multi-step structural judgment calls over
raw document text. A single well-structured generic prompt is sufficient
here; `build_consolidation_prompt()` is still exposed as a dispatcher (empty
of branching today) so a per-family variant can be added later without
changing any call site, mirroring the shape of `build_extraction_prompt()`.

Public API
----------
    def build_consolidation_prompt(partial_visibility: bool = False) -> str
"""

from __future__ import annotations

_PARTIAL_VISIBILITY_NOTICE = """\

## Partial visibility notice

This call may only see a partial, bounded window of the full document's node \
pool — not necessarily every chunk. It is correct and expected to return \
`decided_parent_id: null` for any orphan where you don't find a genuinely \
confident match in what's currently visible. Do NOT force a placement onto \
the closest available candidate just to avoid returning null — an incorrect \
forced placement is permanent and worse than null. Returning null here is \
not final: this orphan will be reconsidered in a later call with additional \
document context; it only becomes a permanent root-level placement if no \
confident match is ever found by the end of the document."""

_CONSOLIDATION_SYSTEM_PROMPT = """\
You are a structure-consolidation engine operating inside an automated document \
ingestion pipeline. A document was extracted in independent, parallel chunks — \
each chunk's own local nesting (which nodes are children of which other nodes \
within that same chunk) is already fixed and correct. Some nodes could not be \
placed under a parent because their parent heading was not visible within their \
own chunk — these are called ORPHANS, and your sole task is to decide, for each \
listed orphan, which node (from ANY chunk, at ANY depth) is its true parent, or \
whether it belongs at the document root.

## Input you receive

A rendered POOL of every node across every chunk, grouped by chunk, with each \
chunk's own local nesting shown via indentation. Each node is shown with its \
node_id, role, and title. Two kinds of nodes appear in the pool:

- REFERENCE nodes: already correctly placed by the chunk-local extraction. \
  They are valid decision targets (a candidate parent) for any orphan, at any \
  depth, in any chunk — but you do not need to (and must not) make a decision \
  about a REFERENCE node itself.
- ORPHAN nodes: explicitly listed in a separate "must decide" section. For \
  every orphan listed there, you must return exactly one decision.

## Your task

For each ORPHAN in the "must decide" list, decide `decided_parent_id`:

- Set it to the exact `node_id` of whichever node in the POOL (REFERENCE or \
  ORPHAN, any chunk, any depth) is genuinely this orphan's structural parent — \
  the node whose heading/section this orphan's content logically nests under, \
  based on numbering, title continuity, and topical fit.
- Set it to `null` when the orphan genuinely belongs at the document root (no \
  real parent exists anywhere in the pool).

## Hard rules

1. `node_id` values in your output must be copied verbatim from the pool. Never \
   invent, reformat, abbreviate, or guess a node_id that is not exactly present \
   in the input.
2. Return exactly one decision per orphan listed in the "must decide" section — \
   no more, no fewer, no duplicates.
3. Do not return a decision for any node not listed in the "must decide" \
   section (this includes REFERENCE nodes and orphans from a different \
   partition of the same document, if this call covers only part of the full \
   orphan set).
4. Never choose another ORPHAN from the same "must decide" list as a parent — \
   an orphan may only be attached to a node that is either a REFERENCE node or \
   an orphan that is NOT itself in the same "must decide" section being resolved \
   in this call, when such a node is a genuinely better structural fit than any \
   REFERENCE node. When in doubt between two candidates, prefer the REFERENCE \
   node.
5. `decided_parent_id` is never the orphan's own node_id.
6. When no confident structural match exists anywhere in the pool, return \
   `null` rather than guessing — a wrong parent is worse than a root-level \
   fallback.

Return only the structured decisions. Do not include commentary."""


def build_consolidation_prompt(partial_visibility: bool = False) -> str:
    """Return the system prompt for the consolidation LLM call.

    Currently a single generic prompt used for every ``PromptFamily`` — see the
    module docstring for why per-family variants were judged unnecessary for
    this narrower task. Kept as a function (rather than a bare module constant)
    so a future per-family dispatch can be added here without changing any
    call site in ``structure_consolidation.py``.

    Args:
        partial_visibility: When ``True``, appends a notice explaining that
            this call may only see a bounded window of the full document's
            node pool (sliding-window batching in
            ``consolidate_structure()``) — returning ``null`` for an orphan
            with no confident match is expected and not final, since the
            orphan is carried forward and retried in later batches. When
            ``False`` (default), the prompt is byte-for-byte identical to
            the pre-batching prompt.
    """
    if partial_visibility:
        return _CONSOLIDATION_SYSTEM_PROMPT + _PARTIAL_VISIBILITY_NOTICE
    return _CONSOLIDATION_SYSTEM_PROMPT
