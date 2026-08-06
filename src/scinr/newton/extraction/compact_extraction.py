"""
compact_extraction.py

Merges per-chunk extraction results into a single coherent document tree
and provides the active-hierarchy context string for the next LLM call.

Public API
----------
    def get_active_hierarchy(document: Document) -> str
    def compact_extraction(document: Document, new_nodes: list[StructureNode]) -> Document

Algorithm overview
------------------
1. ``get_active_hierarchy`` walks the rightmost (last child) path of the growing
   document tree and formats it as a newline-separated string injected into
   the LLM prompt as <active_hierarchy>.

2. ``compact_extraction`` merges each node in ``new_nodes`` into the growing
   ``document.document_structure``:
   - SECTION / SUPPLEMENT / ADDENDUM → appended to root level.
   - SUBSECTION / TABLE / FREEFORM_BLOCK / FIELD_GROUP:
       • With parent_id: attached to the matching parent found by BFS.
       • Without parent_id (FORMAT A node_id only): underscore-prefix matching
         to find the deepest ancestor; FORMAT B falls back to root.

3. After all nodes are merged:
   - ``_dedup_siblings`` collapses same-node_id siblings at every level
     (merging info_units and children into the first occurrence).
   - ``_renumber_nodes`` reassigns appearance_order as 1-based sequential
     integers within each sibling list.
"""

from __future__ import annotations

from collections import deque

from scinr.newton.models.document_structure import Document, NodeRole, StructureNode

# ── Role categories ───────────────────────────────────────────────────────────

_TOP_LEVEL_ROLES = {NodeRole.SECTION, NodeRole.APPENDIX}
_NESTED_ROLES = {
    NodeRole.SUBSECTION,
    NodeRole.TABLE,
    NodeRole.FREEFORM_BLOCK,
    NodeRole.FIELD_GROUP,
}


# ── Public API ────────────────────────────────────────────────────────────────


def get_active_hierarchy(document: Document) -> str:
    """
    Extract the rightmost open-node path from the growing Document tree.

    Walks last-child at each level from the root to the deepest leaf.
    For each node on this path produces a line:
        - node_id="{node.node_id}" role="{node.role.value}" title="{node.title or ''}"

    Returns ``"(none)"`` when the document has no structure yet.
    """
    if not document.document_structure:
        return "(none)"

    lines: list[str] = []
    current: StructureNode = document.document_structure[-1]
    while True:
        lines.append(
            f'- node_id="{current.node_id}" '
            f'role="{current.role}" '
            f'title="{current.title or ""}"'
        )
        if current.children:
            current = current.children[-1]
        else:
            break

    return "\n".join(lines)


def compact_extraction(
    document: Document,
    new_nodes: list[StructureNode],
) -> Document:
    """
    Merge new_nodes (extracted from one chunk) into the growing Document tree.

    Args:
        document: The accumulator Document being built across all chunks.
        new_nodes: Nodes returned by a single ``extract_chunk()`` call.

    Returns:
        The same ``document`` object, mutated in-place and returned.
    """
    for node in new_nodes:
        if node.role in _TOP_LEVEL_ROLES:
            # Top-level structural divisions always go at root
            document.document_structure.append(node)

        elif node.role in _NESTED_ROLES:
            if node.parent_id is not None:
                # Explicit parent hint from the LLM
                parent = _find_node_by_id(document.document_structure, node.parent_id)
                if parent is not None:
                    parent.children.append(node)
                else:
                    # parent_id provided but not found in the tree — fall back to root
                    document.document_structure.append(node)
            else:
                # No explicit parent: try prefix matching (FORMAT A only)
                parent = _prefix_match_parent(document.document_structure, node.node_id)
                if parent is not None:
                    parent.children.append(node)
                else:
                    document.document_structure.append(node)

    _dedup_siblings(document.document_structure)
    _renumber_nodes(document.document_structure)
    return document


# ── Private helpers ───────────────────────────────────────────────────────────


def _find_node_by_id(
    nodes: list[StructureNode], node_id: str
) -> StructureNode | None:
    """BFS search for a node by node_id anywhere in the subtree."""
    queue: deque[StructureNode] = deque(nodes)
    while queue:
        current = queue.popleft()
        if current.node_id == node_id:
            return current
        queue.extend(current.children)
    return None


def _dedup_siblings(nodes: list[StructureNode]) -> None:
    """
    In-place: for each level, merge duplicate node_ids.

    The first occurrence is kept. All info_units and children from duplicates
    are appended to the first occurrence (info_units are concatenated without
    deduplication; SHA-256 ids handle dedup at ingestion time).
    Recurse into children after deduplication.
    """
    seen: dict[str, int] = {}       # node_id → index of first occurrence
    to_remove: list[int] = []

    for idx, node in enumerate(nodes):
        if node.node_id not in seen:
            seen[node.node_id] = idx
        else:
            first = nodes[seen[node.node_id]]
            # Merge info_units (concatenate, no dedup here)
            first.info_units.extend(node.info_units)
            # Theme merge policy: first non-"default" theme wins.
            # If the kept node has "default" but a duplicate has a specific theme, adopt it.
            kept_node = first
            dup_node = node
            if kept_node.theme == "default" and dup_node.theme != "default":
                kept_node.theme = dup_node.theme
            # Merge children recursively via the same dedup pass below
            first.children.extend(node.children)
            # Merge source_page_ids: union of both sets (preserve order, no duplicates)
            first.source_page_ids = list(
                dict.fromkeys(first.source_page_ids + node.source_page_ids)
            )
            to_remove.append(idx)

    for i in reversed(to_remove):
        nodes.pop(i)

    for node in nodes:
        _dedup_siblings(node.children)


def _renumber_nodes(nodes: list[StructureNode]) -> None:
    """
    In-place: reassign appearance_order as 1-based sequential integers within
    each sibling list, preserving relative order.
    Recurse into children after renumbering.
    """
    for order, node in enumerate(nodes, start=1):
        node.appearance_order = order
        if node.children:
            _renumber_nodes(node.children)


def _prefix_match_parent(
    nodes: list[StructureNode], node_id: str
) -> StructureNode | None:
    """
    Find the deepest node whose node_id is a proper underscore-prefix of node_id.

    Only applies to FORMAT A node_ids (containing underscores, no hyphens).
    FORMAT B node_ids (containing hyphens) are never prefix-matched.

    A node A is an underscore-prefix of node B when:
        B.startswith(A.node_id + "_")

    Among all matching nodes the one with the longest node_id wins (deepest match).

    Returns None if:
    - ``node_id`` is FORMAT B (contains a hyphen), or
    - no prefix match is found anywhere in the tree.
    """
    # FORMAT B: hyphens present → no prefix matching
    if "-" in node_id:
        return None

    best: StructureNode | None = None
    best_len: int = 0

    stack: list[StructureNode] = list(nodes)
    while stack:
        current = stack.pop()
        # Only compare FORMAT A nodes as potential parents
        if "-" not in current.node_id and node_id.startswith(current.node_id + "_"):
            if len(current.node_id) > best_len:
                best = current
                best_len = len(current.node_id)
        stack.extend(current.children)

    return best
