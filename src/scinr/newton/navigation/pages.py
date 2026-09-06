"""
navigation/pages.py — Source-text bridge (Group I).

Resolves the verbatim converted source pages behind a structure node / info unit
/ document. Uses the **already-abstract** storage layer
(``storage.factory.get_storage``), so it stays engine-agnostic on the graph
side: it only calls the public :class:`GraphNavigator` methods plus the page
repository.

Every function needs a configured, non-``none`` storage backend — a graph
ingested with ``storage_backend="none"`` has no page content to return.
"""

from __future__ import annotations

from scinr.newton.exceptions import StorageError
from scinr.newton.navigation.base import GraphNavigator
from scinr.newton.navigation.models import PageText


def _page_repo():
    from scinr.newton.config import get_config
    from scinr.newton.storage.factory import get_storage

    if get_config().storage_backend == "none":
        raise StorageError(
            "Source-text access needs a persistent storage backend; "
            "configure(storage_backend='mongodb', …) before calling navigation.pages.*"
        )
    _raw_repo, page_repo = get_storage()
    return page_repo


async def get_node_source_page_ids(nav: GraphNavigator, node_id: str) -> list[str]:
    """Return the raw ``source_page_ids`` recorded on *node_id* (no storage needed)."""
    node = await nav.get_structure_node(node_id)
    return list(node.source_page_ids) if node else []


async def _pages_for(nav: GraphNavigator, node_id: str) -> list[PageText]:
    node = await nav.get_structure_node(node_id)
    if node is None or not node.source_page_ids:
        return []
    doc = await nav.get_document_of_node(node_id)
    raw_file_id = doc.raw_file_id if doc else None
    if not raw_file_id:
        return []
    wanted = set(node.source_page_ids)
    repo = _page_repo()
    records = await repo.get_pages(raw_file_id)
    return [
        PageText(raw={"id": r.id}, page_id=r.id, index=r.page_index, markdown=r.markdown)
        for r in records
        if r.id in wanted
    ]


async def get_node_source_text(nav: GraphNavigator, node_id: str) -> list[PageText]:
    """Return the verbatim converted markdown pages behind *node_id*.

    Raises:
        StorageError: If no persistent storage backend is configured.
    """
    return await _pages_for(nav, node_id)


async def get_info_unit_source_text(nav: GraphNavigator, uid: str) -> list[PageText]:
    """Return the source pages behind the structure node that owns info unit *uid*."""
    node = await nav.get_node_for_info_unit(uid)
    if node is None:
        return []
    return await _pages_for(nav, node.id)


async def get_document_source_text(
    nav: GraphNavigator, document: str, *, version: int | None = None
) -> list[PageText]:
    """Return every converted page of *document*, ordered by page index.

    Raises:
        StorageError: If no persistent storage backend is configured.
    """
    from scinr.newton.navigation.neo4j._common import selector_path

    path = selector_path(document)
    doc = (
        await nav.get_latest_version(path)
        if version is None
        else await nav.get_one_document(path, version)
    )
    if doc is None or not doc.raw_file_id:
        return []
    repo = _page_repo()
    records = await repo.get_pages(doc.raw_file_id)
    return [
        PageText(raw={"id": r.id}, page_id=r.id, index=r.page_index, markdown=r.markdown)
        for r in sorted(records, key=lambda r: r.page_index)
    ]


__all__ = [
    "get_node_source_page_ids",
    "get_node_source_text",
    "get_info_unit_source_text",
    "get_document_source_text",
]
