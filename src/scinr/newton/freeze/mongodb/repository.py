"""
freeze/mongodb/repository.py — MongoDB/GridFS implementation of FreezeRepository.

The snapshot JSON is stored in its own GridFS bucket
(``cfg.mongodb_frozen_gridfs_bucket``, default ``"frozen_snapshots"``) and a
metadata document is inserted into ``cfg.mongodb_frozen_collection`` (default
``"frozen_documents"``), mirroring ``MongoDBRawFileRepository``: the
``frozen_blob_id`` is the ``str(ObjectId)`` of that metadata document.

Both the metadata document and the GridFS ``metadata`` carry the stored
``tenant_id`` (``"__public__"`` for public documents), ``created_by_user_id``
and ``job_id``. Reads and deletes look the metadata document up by
``{"_id": ..., "tenant_id": <stored key>}`` first and only then touch GridFS,
through the ``gridfs_id`` of that already-filtered document.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson.errors import InvalidId
from bson.objectid import ObjectId
from gridfs.errors import NoFile

from scinr.newton.freeze.base import FreezeRepository, SnapshotRecord
from scinr.newton.storage.mongodb.client import get_db, get_gridfs_bucket
from scinr.newton.storage.mongodb.raw_files import _HashingReader, _owner_fields
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

_SNAPSHOT_CONTENT_TYPE = "application/json"


def _as_stored(value: str | Sequence[str] | None) -> str | list[str] | None:
    """Keep a provenance filter as a scalar or a plain list (never a tuple)."""
    if value is None or isinstance(value, str):
        return value
    return list(value)


class MongoDBFreezeRepository(FreezeRepository):
    """Stores frozen-document snapshots in GridFS, with metadata in
    ``frozen_documents``."""

    @staticmethod
    def _collection():
        from scinr.newton.config import get_config

        return get_db()[get_config().mongodb_frozen_collection]

    @staticmethod
    def _bucket():
        from scinr.newton.config import get_config

        return get_gridfs_bucket(get_config().mongodb_frozen_gridfs_bucket)

    async def store_snapshot(
        self,
        path: Path,
        *,
        tenant_id: str | None,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
        metadata: dict[str, Any],
    ) -> str:
        """Stream the snapshot at *path* to GridFS and insert its metadata.

        SHA-256 and size are computed on the fly by ``_HashingReader``; the
        file is read in GridFS ``chunk_size`` blocks, never whole.
        """
        owner = _owner_fields(tenant_id, _as_stored(created_by_user_id), _as_stored(job_id))
        filename = path.name

        with path.open("rb") as fh:
            reader = _HashingReader(fh)
            gridfs_id = await self._bucket().upload_from_stream(
                filename,
                reader,
                metadata={"content_type": _SNAPSHOT_CONTENT_TYPE, **owner},
            )

        doc = {
            **metadata,
            "content_type": _SNAPSHOT_CONTENT_TYPE,
            "size_bytes": reader.size,
            "checksum_sha256": reader.hexdigest(),
            "stored_at": datetime.now(UTC),
            "gridfs_id": gridfs_id,
            **owner,
        }
        result = await self._collection().insert_one(doc)
        frozen_blob_id = str(result.inserted_id)
        logger.debug(
            "Stored frozen snapshot → frozen_blob_id=%s, gridfs_id=%s (%d bytes, tenant=%s)",
            frozen_blob_id,
            gridfs_id,
            reader.size,
            owner["tenant_id"],
        )
        return frozen_blob_id

    async def _find_scoped(self, frozen_blob_id: str, tenant_id: str) -> dict | None:
        """Return the metadata document of *frozen_blob_id* if it belongs to
        *tenant_id*, else ``None`` (also for an invalid ObjectId)."""
        try:
            object_id = ObjectId(frozen_blob_id)
        except (InvalidId, TypeError):
            return None
        return await self._collection().find_one(
            {"_id": object_id, "tenant_id": tenant_key(tenant_id)}
        )

    async def read_snapshot_to_file(
        self, frozen_blob_id: str, dest_path: Path, *, tenant_id: str
    ) -> bool:
        """Download the snapshot chunk by chunk into *dest_path*.

        Returns ``False`` (writing nothing) when the metadata document does
        not exist or belongs to another tenant. A metadata document whose
        GridFS binary is gone raises :class:`~scinr.newton.exceptions.StorageError`.
        """
        doc = await self._find_scoped(frozen_blob_id, tenant_id)
        if doc is None:
            return False

        gridfs_id = doc.get("gridfs_id")
        try:
            if gridfs_id is None:
                raise NoFile(f"frozen_blob_id={frozen_blob_id!r} has no gridfs_id")
            grid_out = await self._bucket().open_download_stream(gridfs_id)
        except NoFile as exc:
            from scinr.newton.exceptions import StorageError

            raise StorageError(
                f"The snapshot of frozen_blob_id={frozen_blob_id!r} is missing from GridFS."
            ) from exc

        try:
            with dest_path.open("wb") as out:
                while chunk := await grid_out.readchunk():
                    out.write(chunk)
        finally:
            grid_out.close()
        return True

    async def delete_snapshot(self, frozen_blob_id: str, *, tenant_id: str) -> None:
        """Delete the snapshot binary (GridFS) and its metadata document.

        Idempotent: an invalid id, a missing metadata document or one of
        another tenant logs a warning and returns; a GridFS binary already
        gone is logged and the metadata document is still deleted.
        """
        doc = await self._find_scoped(frozen_blob_id, tenant_id)
        if doc is None:
            logger.warning(
                "MongoDBFreezeRepository.delete_snapshot: frozen_blob_id=%r not found "
                "(already deleted, invalid or of another tenant); nothing to delete.",
                frozen_blob_id,
            )
            return

        gridfs_id = doc.get("gridfs_id")
        if gridfs_id is not None:
            try:
                await self._bucket().delete(gridfs_id)
            except NoFile:
                logger.warning(
                    "MongoDBFreezeRepository.delete_snapshot: GridFS binary for "
                    "frozen_blob_id=%r (gridfs_id=%r) is already gone; deleting "
                    "metadata only.",
                    frozen_blob_id,
                    gridfs_id,
                )

        await self._collection().delete_one({"_id": doc["_id"], "tenant_id": doc["tenant_id"]})
        logger.debug("Deleted frozen snapshot metadata for frozen_blob_id=%s", frozen_blob_id)

    async def find_snapshots(
        self,
        *,
        tenant_id: str,
        path: str | None = None,
        version: int | None = None,
        job_id: Sequence[str] | None = None,
        created_by_user_id: Sequence[str] | None = None,
    ) -> list[SnapshotRecord]:
        """Snapshots of *tenant_id* with a matching ``documents`` entry, newest
        first (``stored_at``). One ``$elemMatch``, so every filter applies to
        the same document entry."""
        element: dict[str, Any] = {}
        if path is not None:
            element["path"] = path
        if version is not None:
            element["version"] = version
        if job_id is not None:
            element["job_id"] = {"$in": list(job_id)}
        if created_by_user_id is not None:
            element["created_by_user_id"] = {"$in": list(created_by_user_id)}
        query: dict[str, Any] = {"tenant_id": tenant_key(tenant_id)}
        if element:
            query["documents"] = {"$elemMatch": element}

        cursor = self._collection().find(
            query,
            projection={"mode": 1, "frozen_at": 1, "documents": 1},
            sort=[("stored_at", -1)],
        )
        return [
            SnapshotRecord(
                frozen_blob_id=str(doc["_id"]),
                mode=doc.get("mode"),
                frozen_at=doc.get("frozen_at"),
                documents=list(doc.get("documents") or []),
            )
            async for doc in cursor
        ]
