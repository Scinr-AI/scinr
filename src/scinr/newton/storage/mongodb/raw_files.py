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

from scinr.newton.storage.base import RawFileRepository
from scinr.newton.storage.mongodb.client import get_db, get_gridfs_bucket

logger = logging.getLogger(__name__)


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
        db = get_db()
        from scinr.newton.config import get_config
        cfg = get_config()

        # 1. Compute SHA-256 checksum before uploading
        checksum = hashlib.sha256(content).hexdigest()

        # 2. Upload binary to GridFS; attach lightweight metadata for traceability
        gridfs_id = await bucket.upload_from_stream(
            filename,
            content,
            metadata={"content_type": content_type, "folder_path": folder_path},
        )

        # 3. Insert metadata document into the raw_files collection
        doc = {
            "filename": filename,
            "folder_path": folder_path,
            "content_type": content_type,
            "size_bytes": len(content),
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
            len(content),
        )
        return raw_file_id
