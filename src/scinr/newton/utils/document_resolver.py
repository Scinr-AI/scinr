"""
utils/document_resolver.py — Resolves document hierarchies via IS_COMPOSED_OF.

Given a root document of one tenant, returns all leaf documents reachable
through the IS_COMPOSED_OF relationship in Neo4j. Leaf documents are those
that have no outgoing IS_COMPOSED_OF relationships.

The root is selected within a single tenant (``None`` = public), by path or —
for callers that only know a display name — by name. Leaves are returned as
:class:`LeafDocument` ``(name, path)`` pairs so that later stages select each
leaf by ``(tenant_id, path)`` rather than by its non-unique name.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

from neo4j import AsyncDriver, Driver

from scinr.newton.config import get_config
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)


class LeafDocument(NamedTuple):
    """A leaf document to process: its display ``name`` and its ``path`` (the
    selector, together with the tenant, of its ``latest`` :Document)."""

    name: str
    path: str


def latest_document_pattern(
    alias: str,
    *,
    tenant_id: str | None,
    doc_path: str | None = None,
    document_name: str | None = None,
) -> tuple[str, dict]:
    """Cypher node pattern (and its parameters) matching the ``latest``
    :Document(s) of *tenant_id* (``None`` = public) at *doc_path*, or — when
    only *document_name* is given — every one with that name.

    Returns e.g. ``("(d:Document {tenant_id: $tenant_id, path: $doc_selector,
    latest: true})", {"tenant_id": "acme", "doc_selector": "A/b"})``.

    Raises:
        ValueError: If neither *doc_path* nor *document_name* is given, or
            *tenant_id* is invalid (see ``utils.tenancy.tenant_key``).
    """
    if doc_path is None and document_name is None:
        raise ValueError("Either doc_path or document_name is required.")
    # Hard-coded property name, never user input.
    prop, value = ("path", doc_path) if doc_path is not None else ("name", document_name)
    pattern = f"({alias}:Document {{tenant_id: $tenant_id, {prop}: $doc_selector, latest: true}})"
    return pattern, {"tenant_id": tenant_key(tenant_id), "doc_selector": value}


# Root and leaves are both pinned to the tenant: IS_COMPOSED_OF never crosses
# tenants by construction (each tenant has its own folders), the leaf filter
# is defence in depth.
_LEAVES_QUERY = """
MATCH {root}
OPTIONAL MATCH (root)-[:IS_COMPOSED_OF*1..]->(leaf:Document {{tenant_id: $tenant_id, latest: true}})
WHERE NOT (leaf)-[:IS_COMPOSED_OF]->(:Document)
WITH root, collect(DISTINCT leaf) AS leaves
RETURN CASE
  WHEN size(leaves) > 0 THEN [l IN leaves | {{name: l.name, path: l.path}}]
  ELSE [{{name: root.name, path: root.path}}]
END AS docs
ORDER BY root.path
"""


def _query_and_params(
    tenant_id: str | None, doc_path: str | None, document_name: str | None
) -> tuple[str, dict]:
    root, params = latest_document_pattern(
        "root", tenant_id=tenant_id, doc_path=doc_path, document_name=document_name
    )
    return _LEAVES_QUERY.format(root=root), params


def _collect(
    records: list[list[dict]],
    tenant_id: str | None,
    doc_path: str | None,
    document_name: str | None,
) -> list[LeafDocument]:
    leaves: dict[str, LeafDocument] = {}
    for docs in records:
        for d in docs:
            leaves.setdefault(d["path"], LeafDocument(d["name"], d["path"]))

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

    result = list(leaves.values())
    if len(result) > 1:
        logger.info(
            "Document %s (tenant=%r) resolved to %d leaf documents: %s",
            selector,
            tenant_id,
            len(result),
            [leaf.path for leaf in result],
        )
    return result


def resolve_leaf_documents(
    driver: Driver,
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

    The traversal is performed entirely in Cypher (variable-length path match),
    so it handles arbitrarily deep hierarchies without Python-level recursion.

    Args:
        driver: An open Neo4j driver instance.
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
    """
    query, params = _query_and_params(tenant_id, doc_path, document_name)
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        result = session.run(query, **params)
        records = [row["docs"] for row in result.data()]
    return _collect(records, tenant_id, doc_path, document_name)


async def resolve_leaf_documents_async(
    driver: AsyncDriver,
    *,
    tenant_id: str | None,
    doc_path: str | None = None,
    document_name: str | None = None,
) -> list[LeafDocument]:
    """Async version of :func:`resolve_leaf_documents` (same arguments,
    semantics and return value), for the singleton async driver."""
    query, params = _query_and_params(tenant_id, doc_path, document_name)
    cfg = get_config()
    async with driver.session(database=cfg.neo4j_database) as session:
        result = await session.run(query, **params)
        records = [row["docs"] for row in await result.data()]
    return _collect(records, tenant_id, doc_path, document_name)
