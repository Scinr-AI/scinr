"""
storage/sql/pages.py — SQL (Postgres / SQLite) implementation of PageRepository.

Mirror of ``storage/mongodb/pages.py``: one row of ``pages`` per converted page, referencing
its raw file by ``raw_file_id`` and carrying the tenant and provenance of its upload.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

from scinr.newton.models.transcript import TranscriptCue
from scinr.newton.sql.ids import new_id
from scinr.newton.sql.tables import pages
from scinr.newton.storage.base import PageRepository
from scinr.newton.storage.models import ConvertedPageRecord
from scinr.newton.storage.sql import _common
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

__all__ = ["SqlPageRepository"]


def _to_record(row: Any) -> ConvertedPageRecord:
    cues = row["transcript"]
    return ConvertedPageRecord(
        id=row["id"],
        raw_file_id=row["raw_file_id"],
        filename=row["filename"],
        folder_path=row["folder_path"],
        page_index=row["page_index"],
        markdown=row["markdown"],
        converted_at=row["converted_at"],
        tenant_id=row["tenant_id"],
        created_by_user_id=row["created_by_user_id"],
        job_id=row["job_id"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        transcript=[TranscriptCue.model_validate(cue) for cue in cues] if cues else None,
    )


class SqlPageRepository(PageRepository):
    """Converted pages in the ``pages`` table."""

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
        start_ms: int | None = None,
        end_ms: int | None = None,
        transcript: list[dict] | None = None,
    ) -> str:
        """Persist one converted page and return its ``page_id`` (24 hex chars).

        The tenant is stored as :func:`~scinr.newton.utils.tenancy.tenant_key`. The transcript
        cues are validated and stored as JSON (``TranscriptCue.model_dump(mode="json")``).
        """
        page_id = new_id()
        row = {
            "id": page_id,
            "raw_file_id": raw_file_id,
            "filename": filename,
            "folder_path": folder_path,
            "page_index": page_index,
            "markdown": markdown,
            "converted_at": datetime.now(UTC),
            "tenant_id": tenant_key(tenant_id),
            "created_by_user_id": created_by_user_id,
            "job_id": job_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "transcript": (
                None
                if transcript is None
                else [
                    TranscriptCue.model_validate(cue).model_dump(mode="json") for cue in transcript
                ]
            ),
        }
        await _common.execute(sa.insert(pages).values(**row))
        logger.debug("Stored page %d of '%s' -> page_id=%s", page_index, filename, page_id)
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
        """Pages of *raw_file_id* inside the scope, ``page_index`` ascending (empty if none
        were stored or they lie outside the scope).

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = pages
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        rows = await _common.fetch_all(
            sa.select(table)
            .where(table.c.raw_file_id == raw_file_id, *where)
            .order_by(table.c.page_index.asc(), table.c.id.asc())
        )
        return [_to_record(row) for row in rows]

    async def get_pages_by_ids(
        self,
        page_ids: Sequence[str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        """Pages whose id is in *page_ids* inside the scope, ``page_index`` ascending.

        Ids that do not exist, lie outside the scope or are not strings are skipped; the
        order of *page_ids* does not matter (nor does repeating an id).

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = pages
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        ids = [page_id for page_id in dict.fromkeys(page_ids) if isinstance(page_id, str)]
        found: list[Any] = []
        for start in range(0, len(ids), _common.IN_BATCH):
            batch = ids[start : start + _common.IN_BATCH]
            found.extend(
                await _common.fetch_all(sa.select(table).where(table.c.id.in_(batch), *where))
            )
        found.sort(key=lambda row: (row["page_index"], row["id"]))
        return [_to_record(row) for row in found]

    async def delete_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Delete the pages of *raw_file_id* inside the scope and return how many (``0`` when
        there are none).

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = pages
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        rows = await _common.execute(
            sa.delete(table).where(table.c.raw_file_id == raw_file_id, *where).returning(table.c.id)
        )
        logger.debug("Deleted %d converted page(s) for raw_file_id=%s", len(rows), raw_file_id)
        return len(rows)
