"""
storage/mongodb/raw_files.py — MongoDB/GridFS implementation of RawFileRepository.

Binary content is stored in GridFS to support files of arbitrary size
(including PDFs larger than the 16 MB BSON document limit).  A lightweight
metadata document is inserted into the ``raw_files`` collection so that
records can be queried by checksum, filename, or folder path without
fetching the full binary from GridFS.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from bson.errors import InvalidId
from bson.objectid import ObjectId
from gridfs.errors import NoFile

from scinr.newton.storage.base import RawFileRepository
from scinr.newton.storage.mongodb.client import get_db, get_gridfs_bucket

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

        Returns
        -------
        str
            The ``raw_file_id``: ``str(ObjectId)`` of the newly inserted
            document in the ``raw_files`` collection.
        """
        bucket = get_gridfs_bucket()

        # 1. Compute SHA-256 checksum before uploading
        checksum = hashlib.sha256(content).hexdigest()

        # 2. Upload binary to GridFS; attach lightweight metadata for traceability
        gridfs_id = await bucket.upload_from_stream(
            filename,
            content,
            metadata={"content_type": content_type, "folder_path": folder_path},
        )

        # 3. Insert metadata document into the raw_files collection
        return await self._insert_metadata(
            filename=filename,
            folder_path=folder_path,
            content_type=content_type,
            size_bytes=len(content),
            checksum=checksum,
            gridfs_id=gridfs_id,
        )

    async def store_file(
        self,
        path: Path,
        filename: str,
        content_type: str,
        folder_path: str | None,
    ) -> str:
        """Stream the file at *path* to GridFS without loading it into memory.

        Motor runs ``upload_from_stream`` in an executor thread, so the reads
        from the file handle happen off the event loop and in ``chunk_size``
        blocks. SHA-256 and size are computed on the fly by :class:`_HashingReader`.

        Parameters
        ----------
        path:
            Path of the file to upload.
        filename, content_type, folder_path:
            Same meaning as in :meth:`store`.

        Returns
        -------
        str
            The ``raw_file_id``: ``str(ObjectId)`` of the metadata document.
        """
        bucket = get_gridfs_bucket()

        with path.open("rb") as fh:
            reader = _HashingReader(fh)
            gridfs_id = await bucket.upload_from_stream(
                filename,
                reader,
                metadata={"content_type": content_type, "folder_path": folder_path},
            )

        return await self._insert_metadata(
            filename=filename,
            folder_path=folder_path,
            content_type=content_type,
            size_bytes=reader.size,
            checksum=reader.hexdigest(),
            gridfs_id=gridfs_id,
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
        }
        result = await db[cfg.mongodb_raw_files_collection].insert_one(doc)
        raw_file_id = str(result.inserted_id)

        logger.debug(
            "Stored raw file '%s' → raw_file_id=%s, gridfs_id=%s (%d bytes)",
            filename,
            raw_file_id,
            gridfs_id,
            size_bytes,
        )
        return raw_file_id

    async def delete(self, raw_file_id: str) -> None:
        """Delete a raw file's binary (GridFS) and metadata (``raw_files``).

        Idempotent: if *raw_file_id* is not a valid ObjectId, or no matching
        metadata document is found (already deleted, or a repeated call),
        this logs a warning and returns without raising. If the metadata
        document exists but its GridFS binary is already gone
        (``gridfs.errors.NoFile``), that is also logged and swallowed —
        the metadata document is still deleted.

        Parameters
        ----------
        raw_file_id:
            The ``raw_file_id`` (``str(ObjectId)``) to delete.
        """
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

        doc = await db[cfg.mongodb_raw_files_collection].find_one({"_id": object_id})
        if doc is None:
            logger.warning(
                "MongoDBRawFileRepository.delete: raw_file_id=%r not found "
                "(already deleted or invalid); nothing to delete.",
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

        await db[cfg.mongodb_raw_files_collection].delete_one({"_id": object_id})
        logger.debug("Deleted raw file metadata for raw_file_id=%s", raw_file_id)
