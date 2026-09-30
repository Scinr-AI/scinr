"""
storage/mongodb/pages.py — MongoDB implementation of PageRepository.

Converted pages (Markdown text) are stored as plain documents in the
``converted_pages`` collection.  Each document maps 1-to-1 to an
:class:`~storage.models.ConvertedPageRecord` and references its parent
raw file via ``raw_file_id``.

Every page carries the stored ``tenant_id`` (``"__public__"`` for public
uploads), ``created_by_user_id`` and ``job_id`` of its upload; reads and
deletes filter by the requested scope (see :mod:`scinr.newton.storage.base`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from bson.errors import InvalidId
from bson.objectid import ObjectId

from scinr.newton.storage.base import PageRepository
from scinr.newton.storage.filters import mongo_scope_filter
from scinr.newton.storage.models import ConvertedPageRecord
from scinr.newton.storage.mongodb.client import get_db
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)


def _to_record(doc: dict) -> ConvertedPageRecord:
    return ConvertedPageRecord(
        id=str(doc["_id"]),
        raw_file_id=doc["raw_file_id"],
        filename=doc["filename"],
        folder_path=doc.get("folder_path"),
        page_index=doc["page_index"],
        markdown=doc["markdown"],
        converted_at=doc["converted_at"],
        tenant_id=doc.get("tenant_id"),
        created_by_user_id=doc.get("created_by_user_id"),
        job_id=doc.get("job_id"),
    )


class MongoDBPageRepository(PageRepository):
    """Stores and retrieves converted pages in the ``converted_pages`` collection.

    Each document in the collection represents a single page of a converted
    document.  Pages are ordered by ``page_index`` which mirrors
    :attr:`~converters.base.IntermediatePage.index`.
    """

    async def store_page(
        self,
        raw_file_id: str,
        filename: str,
        folder_path: str | None,
        page_index: int,
        markdown: str,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Persist a single converted page in MongoDB.

        Parameters
        ----------
        raw_file_id:
            ID of the parent :class:`~storage.models.RawFileRecord`.
        filename:
            Stem of the source file without extension, e.g. ``"3.2.P.1"``.
        folder_path:
            Relative path of the containing folder, or ``None``.
        page_index:
            Zero-based page index.
        markdown:
            Full Markdown text of this page.
        tenant_id:
            Owning tenant (``None`` / ``"__public__"`` = public), stored as
            :func:`~scinr.newton.utils.tenancy.tenant_key`.
        created_by_user_id, job_id:
            Provenance of the upload, stored verbatim.

        Returns
        -------
        str
            The ``page_id``: ``str(ObjectId)`` of the newly inserted document.
        """
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        doc = {
            "raw_file_id": raw_file_id,
            "filename": filename,
            "folder_path": folder_path,
            "page_index": page_index,
            "markdown": markdown,
            "converted_at": datetime.now(UTC),
            "tenant_id": tenant_key(tenant_id),
            "created_by_user_id": created_by_user_id,
            "job_id": job_id,
        }
        result = await db[cfg.mongodb_pages_collection].insert_one(doc)
        page_id = str(result.inserted_id)

        logger.debug(
            "Stored page %d of '%s' → page_id=%s",
            page_index,
            filename,
            page_id,
        )
        return page_id

    async def get_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        """Retrieve the pages of a raw file inside the scope, ordered by page index.

        Parameters
        ----------
        raw_file_id:
            ID of the :class:`~storage.models.RawFileRecord` whose pages
            should be retrieved.
        tenant_id, include_public, created_by_user_id, job_id:
            Scope filters (see :mod:`scinr.newton.storage.base`).

        Returns
        -------
        list[ConvertedPageRecord]
            Pages sorted by ``page_index`` ascending.  Returns an empty list
            if no pages have been stored for this ``raw_file_id`` or they lie
            outside the scope.
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        cursor = db[cfg.mongodb_pages_collection].find(
            {"raw_file_id": raw_file_id, **scope_filter},
            sort=[("page_index", 1)],
        )
        return [_to_record(doc) async for doc in cursor]

    async def get_pages_by_ids(
        self,
        page_ids: Sequence[str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        """Retrieve the pages whose ``page_id`` is in *page_ids*, inside the
        scope, ordered by page index.

        A single ``_id $in`` lookup (default ``_id`` index). Ids that are not
        valid ObjectIds, do not exist or lie outside the scope are skipped.
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        object_ids: list[ObjectId] = []
        for page_id in dict.fromkeys(page_ids):
            try:
                object_ids.append(ObjectId(page_id))
            except (InvalidId, TypeError):
                continue
        if not object_ids:
            return []
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        cursor = db[cfg.mongodb_pages_collection].find(
            {"_id": {"$in": object_ids}, **scope_filter},
            sort=[("page_index", 1)],
        )
        return [_to_record(doc) async for doc in cursor]

    async def delete_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Delete the converted pages of *raw_file_id* inside the scope.

        Not an error if no pages exist for this ``raw_file_id`` — returns
        ``0`` in that case rather than raising.

        Parameters
        ----------
        raw_file_id:
            ID of the :class:`~storage.models.RawFileRecord` whose pages
            should be deleted.
        tenant_id, include_public, created_by_user_id, job_id:
            Scope filters (see :mod:`scinr.newton.storage.base`).

        Returns
        -------
        int
            Number of pages deleted (``0`` if none matched).
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        result = await db[cfg.mongodb_pages_collection].delete_many(
            {"raw_file_id": raw_file_id, **scope_filter}
        )
        logger.debug(
            "Deleted %d converted page(s) for raw_file_id=%s",
            result.deleted_count,
            raw_file_id,
        )
        return result.deleted_count
