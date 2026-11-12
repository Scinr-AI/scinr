"""
navigation/pages.py — Source-text bridge (Group I).

Resolves the verbatim converted source pages — and the original uploaded file —
behind structure nodes / an info unit / a document. Uses the **already-abstract**
storage layer (``storage.factory.get_storage``), so it stays engine-agnostic on
the graph side: it only calls the public :class:`GraphNavigator` methods plus
the storage repositories.

Every function needs a configured, non-``none`` storage backend — a graph
ingested with ``storage_backend="none"`` has no page content to return. To get
a structure node's page ids without storage, read ``source_page_ids`` from
:meth:`GraphNavigator.get_structure_nodes_by_ids`.

Stored images (described embedded images and video frames, 0.5.0):
``include_images=True`` on the three page functions fills ``PageText.images``
(one ``list_images`` per ``(tenant, raw_file_id)``, in the resolved node's or
document's tenant); :func:`get_document_images` lists a document's images;
:func:`get_page_image` opens one by id with the call's **full** scope (there is
no graph node to resolve from an image id).

Structure nodes are read in batches only
(:func:`get_structure_nodes_source_pages`, one id for a single node): one graph
query for all nodes, one storage read per tenant, and each shared page once.

The same operations are available as :class:`GraphNavigator` methods
(``nav.get_structure_nodes_source_pages(node_ids)``, ``nav.get_document_original(path)``, …),
which delegate here; on a ``ScopedNavigator`` they get the view's scope.

Multi-tenancy
-------------
The graph lookup takes the usual four scope filters (``tenant_id``,
``include_public``, ``created_by_user_id``, ``job_id``) and follows the
navigator's own scope: in a multi-tenant API layer pass them, or use a
:class:`~scinr.newton.navigation.ScopedNavigator` (``nav.scoped(tenant_id=...)``),
so another tenant's node or document is never resolved.

The storage read is then **also** filtered by the stored tenant of the resolved
graph node (defense in depth). Even a page id or ``raw_file_id`` pointing at
another tenant's upload returns nothing:

- A structure node / info unit reads only its own ``source_page_ids``, by id,
  in the node's tenant (a page of another tenant counts as not found). The
  ``:Document`` is not resolved and no other page of the document is loaded.
- A document reads its upload's pages / original in the ``:Document``'s tenant.

A public node or document only reads public pages; a legacy one without a
tenant reads unfiltered. Only the tenant is checked on the storage side: the
user / job filters of the scope already selected the node or document in the
graph, and the upload's provenance may legitimately differ from it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from scinr.newton.exceptions import NavigationError, StorageError
from scinr.newton.navigation.models import (
    DocumentRef,
    OriginalFile,
    PageImageFile,
    PageImageRef,
    PageText,
    StructureNodeRef,
    StructureNodeSourcePages,
    StructureNodesSourcePages,
)
from scinr.newton.navigation.scope import make_scope

if TYPE_CHECKING:
    from scinr.newton.navigation.base import GraphNavigator


def _storage():
    from scinr.newton.config import get_config
    from scinr.newton.storage.factory import get_storage

    if get_config().storage_backend == "none":
        raise StorageError(
            "Source-text access needs a persistent storage backend; "
            "configure(storage_backend='mongodb', …) before calling navigation.pages.*"
        )
    return get_storage()


def _page_repo():
    return _storage()[1]


def _raw_repo():
    return _storage()[0]


def _scope_kw(
    tenant_id: str | None,
    include_public: bool,
    created_by_user_id: str | Sequence[str] | None,
    job_id: str | Sequence[str] | None,
) -> dict[str, Any]:
    """The four scope filters as kwargs, validated now (same errors as the navigator)."""
    make_scope(tenant_id, include_public, created_by_user_id, job_id)
    return {
        "tenant_id": tenant_id,
        "include_public": include_public,
        "created_by_user_id": created_by_user_id,
        "job_id": job_id,
    }


def _page_text(record: Any) -> PageText:
    return PageText(
        raw={"id": record.id},
        page_id=record.id,
        index=record.page_index,
        markdown=record.markdown,
        raw_file_id=record.raw_file_id,
        filename=record.filename,
        folder_path=record.folder_path,
        start_ms=getattr(record, "start_ms", None),
        end_ms=getattr(record, "end_ms", None),
        transcript=getattr(record, "transcript", None),
    )


def _image_ref(record: Any) -> PageImageRef:
    return PageImageRef(
        raw={"id": record.id},
        image_id=record.id,
        raw_file_id=record.raw_file_id,
        page_index=record.page_index,
        image_index=record.image_index,
        kind=record.kind,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        width=record.width,
        height=record.height,
        time_ms=record.time_ms,
        description=record.description,
    )


async def _with_images(groups: dict[str | None, list[PageText]]) -> dict[str, PageText]:
    """The pages of *groups* (stored tenant -> pages read in it) with their
    ``images`` filled, keyed by ``page_id``.

    One ``list_images`` per ``(tenant, raw_file_id)``, all in parallel, for
    just the page indexes read. The tenant is the one the pages were read in
    (the resolved node's or document's), so an image is never read outside
    it. A page without stored images gets ``[]``; a page without a
    ``raw_file_id`` or an index too.
    """
    # (tenant, raw_file_id) -> page indexes (ordered set)
    wanted: dict[tuple[str | None, str], dict[int, None]] = {}
    for tenant, pages in groups.items():
        for page in pages:
            if page.raw_file_id and page.index is not None:
                wanted.setdefault((tenant, page.raw_file_id), {})[page.index] = None
    found: dict[tuple[str | None, str, int], list[PageImageRef]] = {}
    if wanted:
        repo = _raw_repo()
        keys = list(wanted)
        results = await asyncio.gather(
            *(
                repo.list_images(rid, page_indexes=list(wanted[(t, rid)]), tenant_id=t)
                for t, rid in keys
            )
        )
        for (tenant, rid), records in zip(keys, results, strict=True):
            for record in records:
                found.setdefault((tenant, rid, record.page_index), []).append(_image_ref(record))
    out: dict[str, PageText] = {}
    for tenant, pages in groups.items():
        for page in pages:
            images = found.get((tenant, page.raw_file_id or "", page.index), [])
            images.sort(key=lambda ref: ref.image_index)
            out[page.page_id] = page.model_copy(update={"images": images})
    return out


async def _source_pages_for_nodes(
    nodes: Sequence[StructureNodeRef],
    *,
    not_found_structure_nodes: Sequence[str] = (),
    include_images: bool = False,
) -> StructureNodesSourcePages:
    """Source pages of already-resolved structure nodes, read by id.

    With *include_images*, each page's ``images`` is filled from one
    ``list_images`` per ``(tenant, raw_file_id)`` (see :func:`_with_images`).

    Only the nodes' ``source_page_ids`` are fetched (no document lookup, no
    other pages), with one storage read per **stored** tenant of the nodes: a
    public node only reads public pages, and a legacy node without a tenant
    reads unfiltered. A page of another tenant counts as not found for the
    node. The nodes' user / job are not applied — the upload's provenance may
    legitimately differ from the node's. Storage is not touched when there is
    no page to read.
    """
    with_pages = [n for n in nodes if n.source_page_ids]
    without_pages = [n.id for n in nodes if not n.source_page_ids]

    # tenant -> page ids to read in it (ordered set: a shared page is read once)
    groups: dict[str | None, dict[str, None]] = {}
    for node in with_pages:
        groups.setdefault(node.raw.get("tenant_id"), {}).update(
            dict.fromkeys(node.source_page_ids)
        )

    found: dict[str | None, dict[str, PageText]] = {}
    if groups:
        repo = _page_repo()
        tenants = list(groups)
        results = await asyncio.gather(
            *(repo.get_pages_by_ids(list(groups[t]), tenant_id=t) for t in tenants)
        )
        found = {
            t: {r.id: _page_text(r) for r in records}
            for t, records in zip(tenants, results, strict=True)
        }
        if include_images:
            with_images = await _with_images({t: list(found[t].values()) for t in found})
            found = {t: {pid: with_images[pid] for pid in found[t]} for t in found}

    entries: list[StructureNodeSourcePages] = []
    pages: dict[str, PageText] = {}
    for node in with_pages:
        in_group = found[node.raw.get("tenant_id")]
        requested = list(dict.fromkeys(node.source_page_ids))
        hits = sorted(
            (in_group[p] for p in requested if p in in_group),
            key=lambda pt: pt.index if pt.index is not None else -1,
        )
        for page in hits:
            pages.setdefault(page.page_id, page)
        entries.append(
            StructureNodeSourcePages(
                structure_node_id=node.id,
                page_ids=[page.page_id for page in hits],
                not_found_page_ids=[p for p in requested if p not in in_group],
            )
        )
    return StructureNodesSourcePages(
        nodes=entries,
        pages=pages,
        not_found_structure_nodes=list(not_found_structure_nodes),
        structure_nodes_without_pages=without_pages,
    )


async def _node_source_text(
    node: StructureNodeRef | None, *, include_images: bool = False
) -> list[PageText]:
    """The source pages of one already-resolved node, ordered by page index."""
    if node is None:
        return []
    result = await _source_pages_for_nodes([node], include_images=include_images)
    return [result.pages[p] for entry in result.nodes for p in entry.page_ids]


async def get_structure_nodes_source_pages(
    nav: GraphNavigator,
    node_ids: Sequence[str],
    *,
    include_images: bool = False,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> StructureNodesSourcePages:
    """Return the verbatim converted markdown pages behind structure nodes *node_ids*.

    *node_ids* are ``:StructureNode`` unique ``id`` s (``StructureNodeRef.id``),
    not their short local ``node_id``. Duplicates are ignored. The nodes are
    looked up in one graph query and their pages read with one storage read
    per tenant; a page shared by several nodes is read and returned once, in
    ``pages``, and each node lists its ``page_ids``.

    Individual ids never raise; they are reported in the envelope's status
    groups: ``not_found_structure_nodes`` (missing **or** outside the scope),
    ``structure_nodes_without_pages`` and, per node, ``not_found_page_ids``.

    With *include_images*, each page's ``images`` lists its stored images
    (described embedded images, video frames) — one storage read per
    ``(tenant, raw_file_id)``, in the node's tenant. Without it (default)
    ``images`` stays ``None`` and the image storage is not touched.

    Example::

        result = await nav.get_structure_nodes_source_pages(
            ["acme::report.pdf::v1::sec-1", "acme::report.pdf::v1::tbl-3"], tenant_id="acme"
        )
        for entry in result.nodes:
            text = "\\n\\n".join(result.pages[p].markdown for p in entry.page_ids)

    Raises:
        StorageError: If there are pages to read and no persistent storage
            backend is configured.
    """
    scope = _scope_kw(tenant_id, include_public, created_by_user_id, job_id)
    ids = list(dict.fromkeys(node_ids))
    nodes = await nav.get_structure_nodes_by_ids(ids, **scope)
    resolved = {n.id for n in nodes}
    return await _source_pages_for_nodes(
        nodes,
        not_found_structure_nodes=[i for i in ids if i not in resolved],
        include_images=include_images,
    )


async def get_info_unit_source_text(
    nav: GraphNavigator,
    uid: str,
    *,
    include_images: bool = False,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> list[PageText]:
    """Return the source pages behind the structure node that owns info unit *uid*.

    *include_images*: fill each page's ``images``, as in
    :func:`get_structure_nodes_source_pages`.

    Raises:
        StorageError: If no persistent storage backend is configured.
    """
    scope = _scope_kw(tenant_id, include_public, created_by_user_id, job_id)
    # The info unit passed the scope; the node that owns it is in the same tenant.
    return await _node_source_text(
        await nav.get_node_for_info_unit(uid, **scope), include_images=include_images
    )


async def _resolve_document(
    nav: GraphNavigator,
    document: str | DocumentRef,
    version: int | None,
    scope: dict[str, Any],
) -> DocumentRef | None:
    """Resolve a document selector (a path, or a ``DocumentRef`` whose own tenant
    is authoritative) to its latest version, or to *version*, inside *scope*."""
    kwargs = dict(scope)
    if isinstance(document, DocumentRef):
        path = document.path
        ref_tenant = document.tenant_id
        if ref_tenant is not None:
            tenants = make_scope(scope["tenant_id"], scope["include_public"]).tenants
            if tenants is not None and ref_tenant not in tenants:
                raise NavigationError(
                    f"document {path!r} belongs to tenant {ref_tenant!r}, "
                    f"which is outside the requested scope {list(tenants)}"
                )
            kwargs["tenant_id"] = ref_tenant
            kwargs["include_public"] = False
    elif isinstance(document, str):
        path = document
    else:
        raise TypeError(
            f"document selector must be a path str or DocumentRef, got {type(document).__name__}"
        )
    if version is None:
        return await nav.get_latest_version(path, **kwargs)
    return await nav.get_one_document(path, version, **kwargs)


async def get_document_source_text(
    nav: GraphNavigator,
    document: str | DocumentRef,
    *,
    version: int | None = None,
    include_images: bool = False,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> list[PageText]:
    """Return every converted page of *document*, ordered by page index.

    *include_images*: fill each page's ``images`` with its stored images
    (one storage read, in the document's tenant), as in
    :func:`get_structure_nodes_source_pages`.

    Raises:
        StorageError: If no persistent storage backend is configured.
    """
    scope = _scope_kw(tenant_id, include_public, created_by_user_id, job_id)
    doc = await _resolve_document(nav, document, version, scope)
    if doc is None or not doc.raw_file_id:
        return []
    # Storage read restricted to the document's (stored) tenant.
    records = await _page_repo().get_pages(doc.raw_file_id, tenant_id=doc.tenant_id)
    pages = [_page_text(r) for r in sorted(records, key=lambda r: r.page_index)]
    if include_images and pages:
        with_images = await _with_images({doc.tenant_id: pages})
        pages = [with_images[page.page_id] for page in pages]
    return pages


async def get_document_original(
    nav: GraphNavigator,
    document: str | DocumentRef,
    *,
    version: int | None = None,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> OriginalFile | None:
    """Return the original uploaded file of *document* (latest version, or
    *version*) as an :class:`OriginalFile` — its metadata plus a stream of
    the binary.

    Returns ``None`` when the document is not found (in the navigator's
    scope), has no stored original (``raw_file_id`` empty), or its original
    does not belong to the document's tenant.

    Example::

        original = await nav.get_document_original(
            "folder/report.pdf", tenant_id="acme", include_public=True
        )
        if original is not None:
            async for chunk in original.stream:
                response.write(chunk)

    Raises:
        StorageError: If no persistent storage backend is configured, or the
            binary is missing from the backend.
    """
    scope = _scope_kw(tenant_id, include_public, created_by_user_id, job_id)
    doc = await _resolve_document(nav, document, version, scope)
    if doc is None or not doc.raw_file_id:
        return None
    opened = await _raw_repo().open_with_record(doc.raw_file_id, tenant_id=doc.tenant_id)
    if opened is None:
        return None
    record, stream = opened
    return OriginalFile(record=record, stream=stream)


async def get_document_images(
    nav: GraphNavigator,
    document: str | DocumentRef,
    *,
    version: int | None = None,
    page_indexes: Sequence[int] | None = None,
    kind: Literal["embedded", "frame"] | None = None,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> list[PageImageRef]:
    """Return the stored images of *document* (latest version, or *version*),
    ordered by ``(page_index, image_index)``.

    *page_indexes* restricts them to some pages; *kind* to ``"embedded"``
    images or video ``"frame"`` s. The document is resolved as in
    :func:`get_document_original` (a ``DocumentRef`` of another tenant raises
    :class:`~scinr.newton.exceptions.NavigationError`), and the storage read
    is restricted to the document's tenant. Returns ``[]`` when the document
    is not found in the scope or has no ``raw_file_id``.

    Raises:
        StorageError: If no persistent storage backend is configured.
    """
    scope = _scope_kw(tenant_id, include_public, created_by_user_id, job_id)
    doc = await _resolve_document(nav, document, version, scope)
    if doc is None or not doc.raw_file_id:
        return []
    records = await _raw_repo().list_images(
        doc.raw_file_id, page_indexes=page_indexes, kind=kind, tenant_id=doc.tenant_id
    )
    return [_image_ref(r) for r in records]


async def get_page_image(
    nav: GraphNavigator,
    image_id: str,
    *,
    tenant_id: str | None = None,
    include_public: bool = False,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
) -> PageImageFile | None:
    """Open the stored image *image_id* (a ``PageImageRef.image_id``) as a
    :class:`PageImageFile` — its record plus a stream of the bytes. The
    natural way to serve ``/images/{image_id}`` from an API.

    Tenancy: there is no graph node to resolve from an image id, so the
    storage read applies the **full scope** of the call (tenant,
    ``include_public``, user and job — matched against the provenance of the
    upload). An image of another tenant, or outside the scope, returns
    ``None``, exactly like one that does not exist. On a ``ScopedNavigator``
    the view's scope is applied automatically.

    Example::

        image = await nav.get_page_image(ref.image_id, tenant_id="acme")
        if image is not None:
            async for chunk in image.stream:
                response.write(chunk)

    Raises:
        StorageError: If no persistent storage backend is configured, or the
            record exists but its binary is missing from the backend.
    """
    scope = _scope_kw(tenant_id, include_public, created_by_user_id, job_id)
    opened = await _raw_repo().open_image(image_id, **scope)
    if opened is None:
        return None
    record, stream = opened
    return PageImageFile(record=record, stream=stream)


__all__ = [
    "get_structure_nodes_source_pages",
    "get_info_unit_source_text",
    "get_document_source_text",
    "get_document_original",
    "get_document_images",
    "get_page_image",
]
