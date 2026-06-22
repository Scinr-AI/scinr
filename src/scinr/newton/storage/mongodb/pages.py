"""
storage/mongodb/pages.py — MongoDB implementation of PageRepository.

Converted pages (Markdown text) are stored as plain documents in the
``converted_pages`` collection.  Each document maps 1-to-1 to an
:class:`~storage.models.ConvertedPageRecord` and references its parent
raw file via ``raw_file_id``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from scinr.newton.storage.base import PageRepository
from scinr.newton.storage.config import PAGES_COLLECTION
from scinr.newton.storage.models import ConvertedPageRecord
from scinr.newton.storage.mongodb.client import get_db

logger = logging.getLogger(__name__)


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

        Returns
        -------
        str
            The ``page_id``: ``str(ObjectId)`` of the newly inserted document.
        """
        db = get_db()
        doc = {
            "raw_file_id": raw_file_id,
            "filename": filename,
            "folder_path": folder_path,
            "page_index": page_index,
            "markdown": markdown,
            "converted_at": datetime.now(UTC),
        }
        result = await db[PAGES_COLLECTION].insert_one(doc)
        page_id = str(result.inserted_id)

        logger.debug(
            "Stored page %d of '%s' → page_id=%s",
            page_index,
            filename,
            page_id,
        )
        return page_id

    async def get_pages(self, raw_file_id: str) -> list[ConvertedPageRecord]:
        """Retrieve all pages for a given raw file, ordered by page index.

        Parameters
        ----------
        raw_file_id:
            ID of the :class:`~storage.models.RawFileRecord` whose pages
            should be retrieved.

        Returns
        -------
        list[ConvertedPageRecord]
            Pages sorted by ``page_index`` ascending.  Returns an empty list
            if no pages have been stored for this ``raw_file_id``.
        """
        db = get_db()
        cursor = db[PAGES_COLLECTION].find(
            {"raw_file_id": raw_file_id},
            sort=[("page_index", 1)],
        )
        records: list[ConvertedPageRecord] = []
        async for doc in cursor:
            records.append(
                ConvertedPageRecord(
                    id=str(doc["_id"]),
                    raw_file_id=doc["raw_file_id"],
                    filename=doc["filename"],
                    folder_path=doc.get("folder_path"),
                    page_index=doc["page_index"],
                    markdown=doc["markdown"],
                    converted_at=doc["converted_at"],
                )
            )
        return records
