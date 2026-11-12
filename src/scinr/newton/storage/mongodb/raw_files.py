"""
storage/mongodb/raw_files.py — MongoDB/GridFS implementation of RawFileRepository.

Binary content is stored in GridFS to support files of arbitrary size
(including PDFs larger than the 16 MB BSON document limit).  A lightweight
metadata document is inserted into the ``raw_files`` collection so that
records can be queried by checksum, filename, or folder path without
fetching the full binary from GridFS.

Described page images and video frames follow the same pattern with their
own pair of destinations: the metadata lives in the ``page_images``
collection and the original bytes in the ``page_image_binaries`` GridFS
bucket (never mixed with the raw-file inventory).  A unique index on
``(tenant_id, raw_file_id, page_index, image_index)`` acts as a safety net:
a duplicate is a bug and must surface as a write error.

Both the metadata documents and the GridFS ``metadata`` carry the stored
``tenant_id`` (``"__public__"`` for public uploads), ``created_by_user_id``
and ``job_id``. Every read / delete filters the metadata document by the
requested scope first and only then touches GridFS, through the
``gridfs_id`` of that already-filtered record.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from bson.errors import InvalidId
from bson.objectid import ObjectId
from gridfs.errors import NoFile

from scinr.newton.storage.base import RawFileRepository
from scinr.newton.storage.filters import mongo_scope_filter
from scinr.newton.storage.models import PageImageRecord, RawFileRecord
from scinr.newton.storage.mongodb.client import get_db, get_gridfs_bucket
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)


class _HashingReader:
    """File-like wrapper that hashes and counts the bytes read through it.

    GridFS reads the source in ``chunk_size`` blocks, so wrapping the file
    handle lets us compute the SHA-256 and size in the same pass as the upload,
    without ever holding the whole file in memory.
    """

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh
        self._sha256 = hashlib.sha256()
        self.size = 0

    def read(self, n: int = -1) -> bytes:
        data = self._fh.read(n)
        self._sha256.update(data)
        self.size += len(data)
        return data

    def hexdigest(self) -> str:
        return self._sha256.hexdigest()


def _owner_fields(
    tenant_id: str | None, created_by_user_id: str | None, job_id: str | None
) -> dict[str, Any]:
    """Tenant / provenance fields written on every raw-file record."""
    return {
        "tenant_id": tenant_key(tenant_id),
        "created_by_user_id": created_by_user_id,
        "job_id": job_id,
    }


def _collection_name(cfg, coll_attr: str | None) -> str:
    """Collection for a metadata read/write: raw files by default, or the one
    named by the ``ScinrConfig`` attribute *coll_attr* (e.g. page images)."""
    if coll_attr is None:
        return cfg.mongodb_raw_files_collection
    return getattr(cfg, coll_attr)


def _to_record(doc: dict) -> RawFileRecord:
    return RawFileRecord(
        id=str(doc["_id"]),
        filename=doc["filename"],
        folder_path=doc.get("folder_path"),
        content_type=doc["content_type"],
        size_bytes=doc["size_bytes"],
        checksum_sha256=doc["checksum_sha256"],
        stored_at=doc["stored_at"],
        tenant_id=doc.get("tenant_id"),
        created_by_user_id=doc.get("created_by_user_id"),
        job_id=doc.get("job_id"),
    )


def _to_image_record(doc: dict) -> PageImageRecord:
    return PageImageRecord(
        id=str(doc["_id"]),
        raw_file_id=doc["raw_file_id"],
        page_index=doc["page_index"],
        image_index=doc["image_index"],
        kind=doc["kind"],
        content_type=doc["content_type"],
        size_bytes=doc["size_bytes"],
        checksum_sha256=doc["checksum_sha256"],
        width=doc.get("width"),
        height=doc.get("height"),
        time_ms=doc.get("time_ms"),
        description=doc["description"],
        filename=doc["filename"],
        folder_path=doc.get("folder_path"),
        stored_at=doc["stored_at"],
        tenant_id=doc.get("tenant_id"),
        created_by_user_id=doc.get("created_by_user_id"),
        job_id=doc.get("job_id"),
    )


class MongoDBRawFileRepository(RawFileRepository):
    """Stores binary files in GridFS with metadata in ``raw_files``.

    GridFS splits large files into 255 kB chunks and stores them across
    two internal collections (``<bucket>.files`` and ``<bucket>.chunks``),
    removing the 16 MB BSON size constraint.

    The ``raw_files`` collection holds only metadata plus a ``gridfs_id``
    reference so callers can retrieve the binary when needed.
    """

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
        """Upload a binary file to GridFS and persist its metadata.

        Parameters
        ----------
        filename:
            Original filename including extension, e.g. ``"3.2.P.1.pdf"``.
        content:
            Raw binary content of the file.
        content_type:
            MIME type, e.g. ``"application/pdf"``.
        folder_path:
            Relative path of the containing folder from the ingestion root,
            or ``None`` for files at the root.
        tenant_id:
            Owning tenant (``None`` / ``"__public__"`` = public), stored as
            :func:`~scinr.newton.utils.tenancy.tenant_key`.
        created_by_user_id, job_id:
            Provenance of the upload, stored verbatim.

        Returns
        -------
        str
            The ``raw_file_id``: ``str(ObjectId)`` of the newly inserted
            document in the ``raw_files`` collection.
        """
        owner = _owner_fields(tenant_id, created_by_user_id, job_id)
        bucket = get_gridfs_bucket()

        # 1. Compute SHA-256 checksum before uploading
        checksum = hashlib.sha256(content).hexdigest()

        # 2. Upload binary to GridFS; attach lightweight metadata for traceability
        gridfs_id = await bucket.upload_from_stream(
            filename,
            content,
            metadata={"content_type": content_type, "folder_path": folder_path, **owner},
        )

        # 3. Insert metadata document into the raw_files collection
        return await self._insert_metadata(
            filename=filename,
            folder_path=folder_path,
            content_type=content_type,
            size_bytes=len(content),
            checksum=checksum,
            gridfs_id=gridfs_id,
            owner=owner,
        )

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
        """Stream the file at *path* to GridFS without loading it into memory.

        Motor runs ``upload_from_stream`` in an executor thread, so the reads
        from the file handle happen off the event loop and in ``chunk_size``
        blocks. SHA-256 and size are computed on the fly by :class:`_HashingReader`.

        Parameters
        ----------
        path:
            Path of the file to upload.
        filename, content_type, folder_path, tenant_id, created_by_user_id, job_id:
            Same meaning as in :meth:`store`.

        Returns
        -------
        str
            The ``raw_file_id``: ``str(ObjectId)`` of the metadata document.
        """
        owner = _owner_fields(tenant_id, created_by_user_id, job_id)
        bucket = get_gridfs_bucket()

        with path.open("rb") as fh:
            reader = _HashingReader(fh)
            gridfs_id = await bucket.upload_from_stream(
                filename,
                reader,
                metadata={"content_type": content_type, "folder_path": folder_path, **owner},
            )

        return await self._insert_metadata(
            filename=filename,
            folder_path=folder_path,
            content_type=content_type,
            size_bytes=reader.size,
            checksum=reader.hexdigest(),
            gridfs_id=gridfs_id,
            owner=owner,
        )

    @staticmethod
    async def _insert_metadata(
        *,
        filename: str,
        folder_path: str | None,
        content_type: str,
        size_bytes: int,
        checksum: str,
        gridfs_id: ObjectId,
        owner: dict[str, Any],
    ) -> str:
        """Insert the ``raw_files`` metadata document and return its id as str."""
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()

        doc = {
            "filename": filename,
            "folder_path": folder_path,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "checksum_sha256": checksum,
            "stored_at": datetime.now(UTC),
            "gridfs_id": gridfs_id,
            **owner,
        }
        result = await db[cfg.mongodb_raw_files_collection].insert_one(doc)
        raw_file_id = str(result.inserted_id)

        logger.debug(
            "Stored raw file '%s' → raw_file_id=%s, gridfs_id=%s (%d bytes, tenant=%s)",
            filename,
            raw_file_id,
            gridfs_id,
            size_bytes,
            owner["tenant_id"],
        )
        return raw_file_id

    @staticmethod
    async def _find_scoped(
        raw_file_id: str,
        scope_filter: dict[str, Any],
        coll_attr: str | None = None,
    ) -> dict | None:
        """Return the document for *raw_file_id* if it is inside
        *scope_filter*, else ``None`` (also for an invalid ObjectId).

        Reads from the raw-files collection by default; *coll_attr* names the
        ``ScinrConfig`` attribute of another metadata collection (e.g.
        ``"mongodb_images_collection"`` for page images).
        """
        try:
            object_id = ObjectId(raw_file_id)
        except (InvalidId, TypeError):
            return None
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        return await db[_collection_name(cfg, coll_attr)].find_one(
            {"_id": object_id, **scope_filter}
        )

    async def get(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> RawFileRecord | None:
        """Return the metadata of *raw_file_id*, or ``None`` if it does not
        exist or lies outside the requested scope."""
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        doc = await self._find_scoped(raw_file_id, scope_filter)
        return _to_record(doc) if doc is not None else None

    async def open(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> AsyncIterator[bytes] | None:
        """Stream the original binary of *raw_file_id* in GridFS chunks.

        Returns ``None`` when the record does not exist or lies outside the
        scope. The GridFS stream is opened through the ``gridfs_id`` of the
        already-filtered record; a record whose binary is gone raises
        :class:`~scinr.newton.exceptions.StorageError`.
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        doc = await self._find_scoped(raw_file_id, scope_filter)
        if doc is None:
            return None
        return await self._open_stream(raw_file_id, doc)

    async def open_with_record(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[RawFileRecord, AsyncIterator[bytes]] | None:
        """Metadata and stream of *raw_file_id* from a single ``raw_files``
        lookup (same rules as :meth:`get` and :meth:`open`)."""
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        doc = await self._find_scoped(raw_file_id, scope_filter)
        if doc is None:
            return None
        return _to_record(doc), await self._open_stream(raw_file_id, doc)

    @staticmethod
    async def _open_stream(
        raw_file_id: str,
        doc: dict,
        bucket_name: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Open the GridFS binary referenced by the already-filtered record *doc*.

        The binary is always located through the ``gridfs_id`` of that
        record, never from a caller-supplied id.  *bucket_name* is the GridFS
        bucket of the binary (``None`` = the raw-files bucket).
        """
        gridfs_id = doc.get("gridfs_id")
        try:
            if gridfs_id is None:
                raise NoFile(f"raw_file_id={raw_file_id!r} has no gridfs_id")
            bucket = (
                get_gridfs_bucket(bucket_name) if bucket_name is not None else get_gridfs_bucket()
            )
            grid_out = await bucket.open_download_stream(gridfs_id)
        except NoFile as exc:
            from scinr.newton.exceptions import StorageError

            raise StorageError(
                f"The original binary of raw_file_id={raw_file_id!r} is missing from GridFS."
            ) from exc

        async def _chunks() -> AsyncIterator[bytes]:
            try:
                while chunk := await grid_out.readchunk():
                    yield chunk
            finally:
                grid_out.close()

        return _chunks()

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
        """Inventory of the raw files inside the scope, oldest first."""
        query = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        if folder_path is not None:
            query["folder_path"] = folder_path
        if filename is not None:
            query["filename"] = filename
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        cursor = db[cfg.mongodb_raw_files_collection].find(query, sort=[("stored_at", 1)])
        return [_to_record(doc) async for doc in cursor]

    async def delete(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> None:
        """Delete a raw file's binary (GridFS) and metadata (``raw_files``).

        Only a record inside the requested scope is deleted; outside it the
        record is treated as missing.

        Idempotent: if *raw_file_id* is not a valid ObjectId, or no matching
        metadata document is found (already deleted, out of scope, or a
        repeated call), this logs a warning and returns without raising. If the metadata
        document exists but its GridFS binary is already gone
        (``gridfs.errors.NoFile``), that is also logged and swallowed —
        the metadata document is still deleted.

        Parameters
        ----------
        raw_file_id:
            The ``raw_file_id`` (``str(ObjectId)``) to delete.
        tenant_id, include_public, created_by_user_id, job_id:
            Scope filters (see :mod:`scinr.newton.storage.base`).
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        try:
            object_id = ObjectId(raw_file_id)
        except InvalidId:
            logger.warning(
                "MongoDBRawFileRepository.delete: raw_file_id=%r is not a valid "
                "ObjectId; nothing to delete.",
                raw_file_id,
            )
            return

        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()

        doc = await db[cfg.mongodb_raw_files_collection].find_one(
            {"_id": object_id, **scope_filter}
        )
        if doc is None:
            logger.warning(
                "MongoDBRawFileRepository.delete: raw_file_id=%r not found "
                "(already deleted, invalid or outside the requested scope); "
                "nothing to delete.",
                raw_file_id,
            )
            return

        bucket = get_gridfs_bucket()
        gridfs_id = doc.get("gridfs_id")
        if gridfs_id is not None:
            try:
                await bucket.delete(gridfs_id)
            except NoFile:
                logger.warning(
                    "MongoDBRawFileRepository.delete: GridFS binary for "
                    "raw_file_id=%r (gridfs_id=%r) is already gone; deleting "
                    "metadata only.",
                    raw_file_id,
                    gridfs_id,
                )

        await db[cfg.mongodb_raw_files_collection].delete_one({"_id": object_id, **scope_filter})
        logger.debug("Deleted raw file metadata for raw_file_id=%s", raw_file_id)

    # ------------------------------------------------------------------
    # Imágenes de página (colección y bucket propios; ver el docstring del módulo)
    # ------------------------------------------------------------------

    @property
    def stores_images(self) -> bool:
        return True

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
        """Upload the original image bytes to the images bucket and persist
        the record in the ``page_images`` collection.

        If the metadata insert fails, the already-uploaded GridFS blob is
        deleted on a best-effort basis (the error is logged, never raised
        over the original one) so a failed write never leaves an orphan
        binary behind.

        Returns
        -------
        str
            The ``image_id``: ``str(ObjectId)`` of the new ``page_images``
            document.
        """
        owner = _owner_fields(tenant_id, created_by_user_id, job_id)
        from scinr.newton.config import get_config
        cfg = get_config()
        bucket = get_gridfs_bucket(cfg.mongodb_images_gridfs_bucket)

        # 1. Compute SHA-256 checksum before uploading
        checksum = hashlib.sha256(content).hexdigest()

        # 2. Upload binary to the images bucket
        gridfs_id = await bucket.upload_from_stream(
            filename,
            content,
            metadata={
                "raw_file_id": raw_file_id,
                "page_index": page_index,
                "image_index": image_index,
                "kind": kind,
                **owner,
            },
        )

        # 3. Insert the metadata document into page_images
        return await self._insert_image_metadata(
            bucket,
            raw_file_id=raw_file_id,
            page_index=page_index,
            image_index=image_index,
            kind=kind,
            content_type=content_type,
            description=description,
            filename=filename,
            folder_path=folder_path,
            width=width,
            height=height,
            time_ms=time_ms,
            size_bytes=len(content),
            checksum=checksum,
            gridfs_id=gridfs_id,
            owner=owner,
        )

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
        """Stream the image at *path* to the images bucket without loading it
        into memory, igual que :meth:`store_file` respecto de :meth:`store`.

        SHA-256 and size are computed on the fly by :class:`_HashingReader`.
        """
        owner = _owner_fields(tenant_id, created_by_user_id, job_id)
        from scinr.newton.config import get_config
        cfg = get_config()
        bucket = get_gridfs_bucket(cfg.mongodb_images_gridfs_bucket)

        with path.open("rb") as fh:
            reader = _HashingReader(fh)
            gridfs_id = await bucket.upload_from_stream(
                filename,
                reader,
                metadata={
                    "raw_file_id": raw_file_id,
                    "page_index": page_index,
                    "image_index": image_index,
                    "kind": kind,
                    **owner,
                },
            )

        return await self._insert_image_metadata(
            bucket,
            raw_file_id=raw_file_id,
            page_index=page_index,
            image_index=image_index,
            kind=kind,
            content_type=content_type,
            description=description,
            filename=filename,
            folder_path=folder_path,
            width=width,
            height=height,
            time_ms=time_ms,
            size_bytes=reader.size,
            checksum=reader.hexdigest(),
            gridfs_id=gridfs_id,
            owner=owner,
        )

    @staticmethod
    async def _insert_image_metadata(
        bucket,
        *,
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
        size_bytes: int,
        checksum: str,
        gridfs_id: ObjectId,
        owner: dict[str, Any],
    ) -> str:
        """Insert the ``page_images`` metadata document and return its id as str.

        On failure the already-uploaded GridFS blob *gridfs_id* is deleted on
        a best-effort basis (logged, never raised over the original error):
        a failed write must not leave an orphan binary behind.
        """
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()

        doc = {
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
            "size_bytes": size_bytes,
            "checksum_sha256": checksum,
            "stored_at": datetime.now(UTC),
            "gridfs_id": gridfs_id,
            **owner,
        }
        try:
            result = await db[cfg.mongodb_images_collection].insert_one(doc)
        except Exception:
            try:
                await bucket.delete(gridfs_id)
            except Exception:
                logger.error(
                    "Could not delete the orphaned GridFS blob %s of image "
                    "(%d, %d) of raw_file_id=%s after the metadata insert failed.",
                    gridfs_id,
                    page_index,
                    image_index,
                    raw_file_id,
                    exc_info=True,
                )
            raise
        image_id = str(result.inserted_id)

        logger.debug(
            "Stored image (%d, %d) of '%s' → image_id=%s, gridfs_id=%s (%d bytes, tenant=%s)",
            page_index,
            image_index,
            filename,
            image_id,
            gridfs_id,
            size_bytes,
            owner["tenant_id"],
        )
        return image_id

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
        """Inventory of the stored images of *raw_file_id* inside the scope,
        ordered by ``(page_index, image_index)``."""
        query = {
            "raw_file_id": raw_file_id,
            **mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id),
        }
        if page_indexes is not None:
            query["page_index"] = {"$in": list(page_indexes)}
        if kind is not None:
            query["kind"] = kind
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()
        cursor = db[cfg.mongodb_images_collection].find(
            query,
            sort=[("page_index", 1), ("image_index", 1)],
        )
        return [_to_image_record(doc) async for doc in cursor]

    async def open_image(
        self,
        image_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[PageImageRecord, AsyncIterator[bytes]] | None:
        """Record and stream of *image_id* from a single ``page_images``
        lookup (same rules as :meth:`open_with_record`).

        The binary is opened through the ``gridfs_id`` of the already-filtered
        record, never from a caller-supplied id; a record whose binary is
        gone raises :class:`~scinr.newton.exceptions.StorageError`.
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        doc = await self._find_scoped(image_id, scope_filter, "mongodb_images_collection")
        if doc is None:
            return None
        from scinr.newton.config import get_config
        cfg = get_config()
        stream = await self._open_stream(
            image_id, doc, bucket_name=cfg.mongodb_images_gridfs_bucket
        )
        return _to_image_record(doc), stream

    async def delete_images(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Delete every stored image of *raw_file_id* inside the scope
        (GridFS binaries first, then the ``page_images`` records).

        Idempotent: no matching record is ``0``, never an error.  A binary
        already gone (``gridfs.errors.NoFile``) is logged and ignored — the
        record is still deleted.

        Returns
        -------
        int
            Number of records deleted.
        """
        scope_filter = mongo_scope_filter(tenant_id, include_public, created_by_user_id, job_id)
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()

        query = {"raw_file_id": raw_file_id, **scope_filter}
        docs = await db[cfg.mongodb_images_collection].find(query).to_list(length=None)
        if not docs:
            return 0

        bucket = get_gridfs_bucket(cfg.mongodb_images_gridfs_bucket)
        for doc in docs:
            gridfs_id = doc.get("gridfs_id")
            if gridfs_id is None:
                continue
            try:
                await bucket.delete(gridfs_id)
            except NoFile:
                logger.warning(
                    "MongoDBRawFileRepository.delete_images: GridFS binary of image "
                    "(%d, %d) of raw_file_id=%r (gridfs_id=%r) is already gone; "
                    "deleting the record only.",
                    doc.get("page_index"),
                    doc.get("image_index"),
                    raw_file_id,
                    gridfs_id,
                )

        result = await db[cfg.mongodb_images_collection].delete_many(query)
        logger.debug(
            "Deleted %d page image(s) for raw_file_id=%s",
            result.deleted_count,
            raw_file_id,
        )
        return result.deleted_count
