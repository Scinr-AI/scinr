"""
utils/document_resolver.py — Resolves document hierarchies via IS_COMPOSED_OF.

Given a root document of one tenant, returns all leaf documents reachable
through the IS_COMPOSED_OF relationship in the graph. Leaf documents are those
that have no outgoing IS_COMPOSED_OF relationships.

The root is selected within a single tenant (``None`` = public), by path or —
for callers that only know a display name — by name. Leaves are returned as
:class:`LeafDocument` ``(name, path)`` pairs so that later stages select each
leaf by ``(tenant_id, path)`` rather than by its non-unique name.

The graph query lives in the graph store
(``get_graph_store().documents.resolve_leaf_documents``); this module builds the
:class:`~scinr.newton.graph.types.DocumentSelector` and applies the fallback for
a root that is not in the graph.
"""
from __future__ import annotations

import logging

from scinr.newton.graph import get_graph_store
from scinr.newton.graph.types import DocumentSelector, LeafDocument

logger = logging.getLogger(__name__)

__all__ = ["LeafDocument", "resolve_leaf_documents", "resolve_leaf_documents_async"]


def _selector(
    tenant_id: str | None, doc_path: str | None, document_name: str | None
) -> DocumentSelector:
    if doc_path is None and document_name is None:
        raise ValueError("Either doc_path or document_name is required.")
    return DocumentSelector(
        tenant_id=tenant_id, path=doc_path, document_name=document_name, latest=True
    )


def _with_fallback(
    leaves: list[LeafDocument],
    tenant_id: str | None,
    doc_path: str | None,
    document_name: str | None,
) -> list[LeafDocument]:
    selector = f"path={doc_path!r}" if doc_path is not None else f"name={document_name!r}"
    if not leaves:
        logger.warning(
            "Document %s (tenant=%r) not found in Neo4j. Proceeding with it as-is.",
            selector,
            tenant_id,
        )
        name = document_name if document_name is not None else doc_path.rsplit("/", 1)[-1]
        path = doc_path if doc_path is not None else document_name
        return [LeafDocument(name, path)]

    if len(leaves) > 1:
        logger.info(
            "Document %s (tenant=%r) resolved to %d leaf documents: %s",
            selector,
            tenant_id,
            len(leaves),
            [leaf.path for leaf in leaves],
        )
    return leaves


def resolve_leaf_documents(
    *,
    tenant_id: str | None,
    doc_path: str | None = None,
    document_name: str | None = None,
) -> list[LeafDocument]:
    """
    Return every leaf document reachable via IS_COMPOSED_OF from a root
    document of *tenant_id*.

    The root is the ``latest`` :Document of *tenant_id* at *doc_path* or, when
    only *document_name* is given, **every** ``latest`` :Document of
    *tenant_id* with that name (a name is not unique; each match is expanded
    and the leaves are merged, deduplicated by path). Another tenant's
    documents are never returned — public ones included, when *tenant_id* is
    a real tenant.

    A root with no children is its own (single) leaf.

    The traversal is performed entirely in the graph engine (variable-length
    path match), so it handles arbitrarily deep hierarchies without
    Python-level recursion. It uses the connection of the surrounding
    ``store.sync_scope()``, or opens and closes its own.

    Args:
        tenant_id: Owner of the documents, ``None`` for public documents.
        doc_path: Path of the root document (preferred selector).
        document_name: Display name of the root document(s), used when
            *doc_path* is not given.

    Returns:
        Leaf documents to process, ordered by root path. Always contains at
        least one entry: when nothing matches, the selector itself (path
        derived from the name, or name from the last path segment).

    Raises:
        ValueError: If neither *doc_path* nor *document_name* is given, or
            *tenant_id* is invalid (see ``utils.tenancy.tenant_key``).
        GraphError: If the graph engine fails.
    """
    leaves = get_graph_store().documents.resolve_leaf_documents(
        _selector(tenant_id, doc_path, document_name)
    )
    return _with_fallback(leaves, tenant_id, doc_path, document_name)


async def resolve_leaf_documents_async(
    *,
    tenant_id: str | None,
    doc_path: str | None = None,
    document_name: str | None = None,
) -> list[LeafDocument]:
    """Async version of :func:`resolve_leaf_documents` (same arguments,
    semantics and return value), for the shared async connection."""
    leaves = await get_graph_store().documents.resolve_leaf_documents_async(
        _selector(tenant_id, doc_path, document_name)
    )
    return _with_fallback(leaves, tenant_id, doc_path, document_name)
