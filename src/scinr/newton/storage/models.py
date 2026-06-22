"""
storage/models.py — Pydantic models for persisted storage records.

These models represent the data returned when reading records from the
storage backend.  Binary content is NOT inlined here; the raw bytes live
in GridFS and are identified by the ``id`` field of :class:`RawFileRecord`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RawFileRecord(BaseModel):
    """Metadata record for a raw (binary) file stored in GridFS.

    Attributes
    ----------
    id:
        ``str(ObjectId)`` of the document in the ``raw_files`` MongoDB
        collection.  Use this value as ``raw_file_id`` when referencing
        this record from :class:`ConvertedPageRecord`.
    filename:
        Original filename including extension, e.g. ``"3.2.P.1.pdf"``.
    folder_path:
        Relative path of the containing folder from the ingestion root,
        e.g. ``"ModuloA/SubModulo"``.  ``None`` for files at the root.
    content_type:
        MIME type of the original file, e.g. ``"application/pdf"``.
    size_bytes:
        Size of the binary content in bytes.
    checksum_sha256:
        Full hex-encoded SHA-256 digest of the original binary content.
        Used for deduplication and integrity checks.
    stored_at:
        UTC timestamp when this record was persisted.
    """

    id: str
    filename: str
    folder_path: str | None
    content_type: str
    size_bytes: int
    checksum_sha256: str
    stored_at: datetime


class ConvertedPageRecord(BaseModel):
    """Record for a single converted page stored in ``converted_pages``.

    Attributes
    ----------
    id:
        ``str(ObjectId)`` of the document in the ``converted_pages``
        MongoDB collection.
    raw_file_id:
        Reference to the parent :attr:`RawFileRecord.id`.
    filename:
        Stem of the source file without extension, e.g. ``"3.2.P.1"``.
    folder_path:
        Relative path of the containing folder, or ``None``.
    page_index:
        Zero-based page index, identical to
        :attr:`~converters.base.IntermediatePage.index`.
    markdown:
        Full Markdown text of this page as produced by the converter.
    converted_at:
        UTC timestamp when this page was persisted.
    """

    id: str
    raw_file_id: str
    filename: str
    folder_path: str | None
    page_index: int
    markdown: str
    converted_at: datetime
