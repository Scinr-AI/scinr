"""
freeze/sql/repository.py — SQL (Postgres / SQLite) implementation of FreezeRepository.

Mirror of ``freeze/mongodb/repository.py``. A snapshot is one row of ``frozen_snapshots``
(``mode``, ``frozen_at`` and the ``documents`` list as columns; the rest of the metadata,
the SHA-256 and the selector's provenance filters in ``props``) plus its bytes, kept by the
:class:`~scinr.newton.storage.sql.blobs.BlobStore` of the engine. The ``frozen_blob_id`` is
the id of the row. Reads and deletes look the row up by ``(id, stored tenant)`` first and
only then touch the bytes, through the row already filtered.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from scinr.newton.freeze.base import FreezeRepository, SnapshotRecord
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.ids import new_id
from scinr.newton.storage.sql import _common
from scinr.newton.storage.sql.blobs import get_blob_store, iter_file
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

__all__ = ["SqlFreezeRepository"]

_SNAPSHOT_CONTENT_TYPE = "application/json"
#: Metadata keys that have a column of their own (the rest goes to ``props``).
_COLUMN_METADATA = frozenset({"mode", "frozen_at", "documents"})


def _as_stored(value: str | Sequence[str] | None) -> str | list[str] | None:
    """Keep a provenance filter as a scalar or a plain list (never a tuple)."""
    if value is None or isinstance(value, str):
        return value
    return list(value)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


class SqlFreezeRepository(FreezeRepository):
    """Frozen-document snapshots in ``frozen_snapshots``."""

    async def store_snapshot(
        self,
        path: Path,
        *,
        tenant_id: str | None,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
        metadata: dict[str, Any],
    ) -> str:
        """Stream the snapshot at *path* into the store and insert its row.

        The file is read in blocks, never whole. The row keeps ``mode``, ``frozen_at`` and
        ``documents`` as columns and everything else of *metadata* (plus the provenance
        filters and the SHA-256) in ``props``; ``None`` values are not stored.
        """
        props = {k: v for k, v in metadata.items() if k not in _COLUMN_METADATA and v is not None}
        for key, value in (
            ("created_by_user_id", _as_stored(created_by_user_id)),
            ("job_id", _as_stored(job_id)),
        ):
            if value is not None:
                props[key] = value
        row = {
            "id": new_id(),
            "tenant_id": tenant_key(tenant_id),
            "mode": metadata.get("mode"),
            "frozen_at": metadata.get("frozen_at"),
            "content_type": _SNAPSHOT_CONTENT_TYPE,
            "stored_at": datetime.now(UTC),
            "documents": list(metadata.get("documents") or []),
            "props": props,
        }
        async with async_sql_errors("storage"):
            size = await get_blob_store().put("snapshot", row, iter_file(path))
        logger.debug(
            "Stored frozen snapshot -> frozen_blob_id=%s (%d bytes, tenant=%s)",
            row["id"],
            size,
            row["tenant_id"],
        )
        return row["id"]

    @staticmethod
    async def _find_scoped(frozen_blob_id: str, tenant_id: str) -> Any:
        """The row of *frozen_blob_id* if it belongs to *tenant_id*, else ``None``."""
        table = _common.tables().frozen_snapshots
        stored = tenant_key(tenant_id)
        if not isinstance(frozen_blob_id, str):
            return None
        return await _common.fetch_one(
            sa.select(table).where(table.c.id == frozen_blob_id, table.c.tenant_id == stored)
        )

    async def read_snapshot_to_file(
        self, frozen_blob_id: str, dest_path: Path, *, tenant_id: str
    ) -> bool:
        """Download the snapshot block by block into *dest_path* (created or truncated).

        Returns ``False``, writing nothing, when the row does not exist or belongs to
        another tenant.

        Raises:
            StorageError: If the row exists but its binary is missing.
        """
        row = await self._find_scoped(frozen_blob_id, tenant_id)
        if row is None:
            return False
        async with async_sql_errors("storage"):
            stream = await get_blob_store().open("snapshot", row)
            handle = await asyncio.to_thread(dest_path.open, "wb")
            try:
                async with aclosing(stream):
                    async for block in stream:
                        await asyncio.to_thread(handle.write, block)
            finally:
                await asyncio.to_thread(handle.close)
        return True

    async def delete_snapshot(self, frozen_blob_id: str, *, tenant_id: str) -> None:
        """Delete the snapshot's row and bytes; idempotent.

        The row is deleted by ``(id, tenant)`` in one statement, so a snapshot of another
        tenant is never touched; an invalid or missing id logs a warning and returns.
        """
        table = _common.tables().frozen_snapshots
        stored = tenant_key(tenant_id)
        rows: list[Any] = []
        if isinstance(frozen_blob_id, str):
            rows = await _common.execute(
                sa.delete(table)
                .where(table.c.id == frozen_blob_id, table.c.tenant_id == stored)
                .returning(table)
            )
        if not rows:
            logger.warning(
                "SqlFreezeRepository.delete_snapshot: frozen_blob_id=%r not found "
                "(already deleted, invalid or of another tenant); nothing to delete.",
                frozen_blob_id,
            )
            return
        async with async_sql_errors("storage"):
            await get_blob_store().delete("snapshot", rows[0])
        logger.debug("Deleted frozen snapshot %s", frozen_blob_id)

    async def find_snapshots(
        self,
        *,
        tenant_id: str,
        path: str | None = None,
        version: int | None = None,
        job_id: Sequence[str] | None = None,
        created_by_user_id: Sequence[str] | None = None,
    ) -> list[SnapshotRecord]:
        """Snapshots of *tenant_id* with a matching ``documents`` entry, newest first
        (``stored_at``). Every filter must hold for the **same** entry (like the MongoDB
        ``$elemMatch``); without filters every snapshot of the tenant matches.

        SQL: the entries are matched in Python (the ``documents`` column is a JSON list in
        both engines), one snapshot row at a time.
        """
        table = _common.tables().frozen_snapshots
        rows = await _common.fetch_all(
            sa.select(table.c.id, table.c.mode, table.c.frozen_at, table.c.documents)
            .where(table.c.tenant_id == tenant_key(tenant_id))
            .order_by(table.c.stored_at.desc(), table.c.id.desc())
        )
        wanted: dict[str, Any] = {}
        if path is not None:
            wanted["path"] = path
        if version is not None:
            wanted["version"] = version
        in_filters = {
            key: set(values)
            for key, values in (("job_id", job_id), ("created_by_user_id", created_by_user_id))
            if values is not None
        }

        def entry_matches(entry: dict[str, Any]) -> bool:
            if any(entry.get(key) != value for key, value in wanted.items()):
                return False
            return all(
                any(item in allowed for item in _as_list(entry.get(key)))
                for key, allowed in in_filters.items()
            )

        no_filter = not wanted and not in_filters
        return [
            SnapshotRecord(
                frozen_blob_id=row["id"],
                mode=row["mode"],
                frozen_at=row["frozen_at"],
                documents=list(row["documents"] or []),
            )
            for row in rows
            if no_filter or any(entry_matches(entry) for entry in row["documents"] or [])
        ]
