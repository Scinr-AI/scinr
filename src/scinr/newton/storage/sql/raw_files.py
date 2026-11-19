"""
storage/sql/raw_files.py — SQL (Postgres / SQLite) implementation of RawFileRepository.

Mirror of ``storage/mongodb/raw_files.py``: the metadata of the original files lives in
``raw_files`` and the one of the described page images in ``page_images``; the bytes are
kept by the :class:`~scinr.newton.storage.sql.blobs.BlobStore` of the engine (chunks in
Postgres, files next to the database in SQLite). Every read and delete filters the metadata
row by the requested scope first and only then touches the bytes, through the row already
filtered. Deleting a raw file does **not** delete its pages or images (neither does MongoDB):
``delete_pages`` / ``delete_images`` do.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.ids import new_id
from scinr.newton.storage.base import RawFileRepository
from scinr.newton.storage.models import PageImageRecord, RawFileRecord
from scinr.newton.storage.sql import _common
from scinr.newton.storage.sql.blobs import BlobKind, get_blob_store, iter_bytes, iter_file
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

__all__ = ["SqlRawFileRepository"]


def _to_record(row: Any) -> RawFileRecord:
    return RawFileRecord(
        id=row["id"],
        filename=row["filename"],
        folder_path=row["folder_path"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        checksum_sha256=row["checksum_sha256"],
        stored_at=row["stored_at"],
        tenant_id=row["tenant_id"],
        created_by_user_id=row["created_by_user_id"],
        job_id=row["job_id"],
    )


def _to_image_record(row: Any) -> PageImageRecord:
    return PageImageRecord(
        id=row["id"],
        raw_file_id=row["raw_file_id"],
        page_index=row["page_index"],
        image_index=row["image_index"],
        kind=row["kind"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        checksum_sha256=row["checksum_sha256"],
        width=row["width"],
        height=row["height"],
        time_ms=row["time_ms"],
        description=row["description"],
        filename=row["filename"],
        folder_path=row["folder_path"],
        stored_at=row["stored_at"],
        tenant_id=row["tenant_id"],
        created_by_user_id=row["created_by_user_id"],
        job_id=row["job_id"],
    )


def _owner_fields(
    tenant_id: str | None, created_by_user_id: str | None, job_id: str | None
) -> dict[str, Any]:
    """Tenant / provenance columns written on every row."""
    return {
        "tenant_id": tenant_key(tenant_id),
        "created_by_user_id": created_by_user_id,
        "job_id": job_id,
    }


class SqlRawFileRepository(RawFileRepository):
    """Raw files and page images in SQL; see the module docstring."""

    # ------------------------------------------------------------------
    # Shared paths
    # ------------------------------------------------------------------

    @staticmethod
    async def _put(kind: BlobKind, row: dict[str, Any], chunks: AsyncIterator[bytes]) -> str:
        """Insert *row* and store the bytes; return the new id."""
        row = {**row, "id": new_id(), "stored_at": datetime.now(UTC)}
        async with async_sql_errors("storage"):
            size = await get_blob_store().put(kind, row, chunks)
        logger.debug("Stored %s %s (%d bytes, tenant=%s)", kind, row["id"], size, row["tenant_id"])
        return row["id"]

    @staticmethod
    async def _find(table: sa.Table, row_id: str, where: list[sa.ColumnElement[bool]]) -> Any:
        """The row of *row_id* inside the scope *where*, else ``None`` (also for a bad id)."""
        if not isinstance(row_id, str):
            return None
        return await _common.fetch_one(sa.select(table).where(table.c.id == row_id, *where))

    # ------------------------------------------------------------------
    # Raw files
    # ------------------------------------------------------------------

    def _raw_row(
        self,
        filename: str,
        content_type: str,
        folder_path: str | None,
        tenant_id: str | None,
        created_by_user_id: str | None,
        job_id: str | None,
    ) -> dict[str, Any]:
        return {
            "filename": filename,
            "folder_path": folder_path,
            "content_type": content_type,
            **_owner_fields(tenant_id, created_by_user_id, job_id),
        }

    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Persist the file and return its ``raw_file_id`` (24 hex chars).

        The tenant is stored as :func:`~scinr.newton.utils.tenancy.tenant_key`; provenance
        is stored verbatim.
        """
        row = self._raw_row(
            filename, content_type, folder_path, tenant_id, created_by_user_id, job_id
        )
        return await self._put("raw_file", row, iter_bytes(content))

    async def store_file(
        self,
        path: Path,
        filename: str,
        content_type: str,
        folder_path: str | None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Stream the file at *path* without loading it into memory (SHA-256 and size are
        computed on the fly) and return its ``raw_file_id``."""
        row = self._raw_row(
            filename, content_type, folder_path, tenant_id, created_by_user_id, job_id
        )
        return await self._put("raw_file", row, iter_file(path))

    async def get(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> RawFileRecord | None:
        """Metadata of *raw_file_id*, or ``None`` if it does not exist, the id is not valid
        or it lies outside the requested scope (indistinguishable).

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = _common.tables().raw_files
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        row = await self._find(table, raw_file_id, where)
        return _to_record(row) if row is not None else None

    async def open(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> AsyncIterator[bytes] | None:
        """Stream the original binary of *raw_file_id*, or ``None`` when the record does not
        exist or lies outside the scope.

        The bytes are located through the already-filtered row, never from the caller's id.

        Raises:
            ScopeError: If the filters are invalid.
            StorageError: If the row exists but its binary is missing.
        """
        found = await self.open_with_record(
            raw_file_id,
            tenant_id=tenant_id,
            include_public=include_public,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )
        return found[1] if found is not None else None

    async def open_with_record(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[RawFileRecord, AsyncIterator[bytes]] | None:
        """``(record, stream)`` from a single read of the row (same rules as :meth:`open`)."""
        table = _common.tables().raw_files
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        row = await self._find(table, raw_file_id, where)
        if row is None:
            return None
        async with async_sql_errors("storage"):
            stream = await get_blob_store().open("raw_file", row)
        return _to_record(row), stream

    async def list_raw_files(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
        folder_path: str | None = None,
        filename: str | None = None,
    ) -> list[RawFileRecord]:
        """Inventory of the raw files inside the scope, ``stored_at`` ascending (ties by id)."""
        table = _common.tables().raw_files
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        if folder_path is not None:
            where.append(table.c.folder_path == folder_path)
        if filename is not None:
            where.append(table.c.filename == filename)
        rows = await _common.fetch_all(
            sa.select(table).where(*where).order_by(table.c.stored_at.asc(), table.c.id.asc())
        )
        return [_to_record(row) for row in rows]

    async def delete(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> None:
        """Delete a raw file's metadata and binary; idempotent.

        Only a record inside the scope is deleted; an unknown or out-of-scope id (or a
        repeated call) logs a warning and returns. The row goes first and the bytes after it;
        a binary already gone is logged and ignored. Pages and images are not touched.

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = _common.tables().raw_files
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        if not isinstance(raw_file_id, str):
            rows: list[Any] = []
        else:
            rows = await _common.execute(
                sa.delete(table).where(table.c.id == raw_file_id, *where).returning(table)
            )
        if not rows:
            logger.warning(
                "SqlRawFileRepository.delete: raw_file_id=%r not found (already deleted, "
                "invalid or outside the requested scope); nothing to delete.",
                raw_file_id,
            )
            return
        async with async_sql_errors("storage"):
            await get_blob_store().delete("raw_file", rows[0])
        logger.debug("Deleted raw file %s", raw_file_id)

    # ------------------------------------------------------------------
    # Page images
    # ------------------------------------------------------------------

    @property
    def stores_images(self) -> bool:
        return True

    @staticmethod
    def _image_row(
        raw_file_id: str,
        page_index: int,
        image_index: int,
        kind: str,
        content_type: str,
        description: str,
        filename: str,
        folder_path: str | None,
        width: int | None,
        height: int | None,
        time_ms: int | None,
        tenant_id: str | None,
        created_by_user_id: str | None,
        job_id: str | None,
    ) -> dict[str, Any]:
        return {
            "raw_file_id": raw_file_id,
            "page_index": page_index,
            "image_index": image_index,
            "kind": kind,
            "content_type": content_type,
            "description": description,
            "filename": filename,
            "folder_path": folder_path,
            "width": width,
            "height": height,
            "time_ms": time_ms,
            **_owner_fields(tenant_id, created_by_user_id, job_id),
        }

    async def store_image(
        self,
        raw_file_id: str,
        page_index: int,
        image_index: int,
        content: bytes,
        *,
        kind: str,
        content_type: str,
        description: str,
        filename: str,
        folder_path: str | None,
        width: int | None = None,
        height: int | None = None,
        time_ms: int | None = None,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Store the original image bytes and its record; return the ``image_id``.

        A failed write leaves neither a row nor bytes behind.
        """
        row = self._image_row(
            raw_file_id, page_index, image_index, kind, content_type, description, filename,
            folder_path, width, height, time_ms, tenant_id, created_by_user_id, job_id,
        )  # fmt: skip
        return await self._put("page_image", row, iter_bytes(content))

    async def store_image_file(
        self,
        path: Path,
        raw_file_id: str,
        page_index: int,
        image_index: int,
        *,
        kind: str,
        content_type: str,
        description: str,
        filename: str,
        folder_path: str | None,
        width: int | None = None,
        height: int | None = None,
        time_ms: int | None = None,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Stream the image at *path* without loading it into memory; like :meth:`store_image`."""
        row = self._image_row(
            raw_file_id, page_index, image_index, kind, content_type, description, filename,
            folder_path, width, height, time_ms, tenant_id, created_by_user_id, job_id,
        )  # fmt: skip
        return await self._put("page_image", row, iter_file(path))

    async def list_images(
        self,
        raw_file_id: str,
        *,
        page_indexes: Sequence[int] | None = None,
        kind: str | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[PageImageRecord]:
        """Stored images of *raw_file_id* inside the scope, ordered by
        ``(page_index, image_index)``; *page_indexes* and *kind* filter (``None`` = no filter).

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = _common.tables().page_images
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        if page_indexes is not None:
            if not page_indexes:
                return []
            where.append(table.c.page_index.in_(list(page_indexes)))
        if kind is not None:
            where.append(table.c.kind == kind)
        rows = await _common.fetch_all(
            sa.select(table)
            .where(table.c.raw_file_id == raw_file_id, *where)
            .order_by(table.c.page_index.asc(), table.c.image_index.asc(), table.c.id.asc())
        )
        return [_to_image_record(row) for row in rows]

    async def open_image(
        self,
        image_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[PageImageRecord, AsyncIterator[bytes]] | None:
        """``(record, stream)`` of *image_id* from a single read of the row, or ``None`` when
        it does not exist or lies outside the scope.

        Raises:
            ScopeError: If the filters are invalid.
            StorageError: If the row exists but its binary is missing.
        """
        table = _common.tables().page_images
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        row = await self._find(table, image_id, where)
        if row is None:
            return None
        async with async_sql_errors("storage"):
            stream = await get_blob_store().open("page_image", row)
        return _to_image_record(row), stream

    async def delete_images(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Delete every stored image of *raw_file_id* inside the scope (rows first, then the
        bytes); return how many. Idempotent: nothing to delete is ``0``.

        Raises:
            ScopeError: If the filters are invalid.
        """
        table = _common.tables().page_images
        where = _common.scope_where(table, tenant_id, include_public, created_by_user_id, job_id)
        rows = await _common.execute(
            sa.delete(table).where(table.c.raw_file_id == raw_file_id, *where).returning(table)
        )
        blobs = get_blob_store()
        async with async_sql_errors("storage"):
            for row in rows:
                await blobs.delete("page_image", row)
        logger.debug("Deleted %d page image(s) for raw_file_id=%s", len(rows), raw_file_id)
        return len(rows)
