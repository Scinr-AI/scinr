"""
storage/models.py — Pydantic models for persisted storage records.

These models represent the data returned when reading records from the
storage backend.  Binary content is NOT inlined here; the raw bytes live
in GridFS and are identified by the ``id`` field of :class:`RawFileRecord`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from scinr.newton.models.transcript import TranscriptCue


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
    tenant_id:
        Stored tenant (``"__public__"`` for a public upload). ``None`` only for
        legacy records written before multi-tenancy.
    created_by_user_id, job_id:
        Provenance of the upload (``None`` when not supplied).
    """

    id: str
    filename: str
    folder_path: str | None
    content_type: str
    size_bytes: int
    checksum_sha256: str
    stored_at: datetime
    tenant_id: str | None = None
    created_by_user_id: str | None = None
    job_id: str | None = None


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
        Name of the source file with its extension, e.g. ``"3.2.P.1.pdf"``
        (records written before 0.5.0 hold the name without extension).
    folder_path:
        Relative path of the containing folder, or ``None``.
    page_index:
        Zero-based page index, identical to
        :attr:`~converters.base.IntermediatePage.index`.
    markdown:
        Full Markdown text of this page as produced by the converter.
    converted_at:
        UTC timestamp when this page was persisted.
    tenant_id, created_by_user_id, job_id:
        Same as in :class:`RawFileRecord` (copied from the upload).
    start_ms, end_ms:
        Time range ``[start_ms, end_ms)`` of the page in the source medium
        (audio / video segments). ``None`` for documents without a time axis
        and for records written before 0.5.0.
    transcript:
        Transcribed sentences of the page with their absolute times (audio /
        video; decision D7). ``None`` for documents without a time axis and
        for records written before this change.
    """

    id: str
    raw_file_id: str
    filename: str
    folder_path: str | None
    page_index: int
    markdown: str
    converted_at: datetime
    tenant_id: str | None = None
    created_by_user_id: str | None = None
    job_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    transcript: list[TranscriptCue] | None = None
    transcript: list[TranscriptCue] | None = None


class PageImageRecord(BaseModel):
    """Metadata record for a described page image stored in ``page_images``.

    Each embedded image or video frame that got a non-empty description is
    persisted (original bytes in GridFS, metadata here), linked to the page
    where it appears. Images that were discarded (too small, repeated, above
    the per-page limit, illegible or left without a description) are not
    stored.

    Attributes
    ----------
    id:
        ``str(ObjectId)`` of the document in the ``page_images`` MongoDB
        collection. Use this value as ``image_id`` when opening the binary.
    raw_file_id:
        Reference to the parent :attr:`RawFileRecord.id`.
    page_index:
        Zero-based page index, identical to
        :attr:`~converters.base.IntermediatePage.index` (the segment index
        for video).
    image_index:
        Index of the image within its page, i.e. the ``index`` of the
        ``<image index="N">`` block of that page.
    kind:
        ``"embedded"`` for an image embedded in a document, ``"frame"`` for
        a video frame.
    content_type:
        MIME type of the stored image, e.g. ``"image/png"`` or
        ``"image/jpeg"``.
    size_bytes:
        Size of the stored binary in bytes.
    checksum_sha256:
        Full hex-encoded SHA-256 digest of the stored binary.
    width, height:
        Pixel dimensions of the stored image, ``None`` when unknown.
    time_ms:
        Instant of the frame in the video, in milliseconds. Only set for
        ``kind="frame"``; ``None`` for embedded images.
    description:
        The description produced by the vision model — the same text that
        appears in the ``<image>`` block of the page markdown.
    filename:
        Name of the source document with its extension, e.g.
        ``"3.2.P.1.pdf"``.
    folder_path:
        Relative path of the containing folder of the source document, or
        ``None``.
    stored_at:
        UTC timestamp when this record was persisted.
    tenant_id, created_by_user_id, job_id:
        Same as in :class:`RawFileRecord` (copied from the upload).
    """

    id: str
    raw_file_id: str
    page_index: int
    image_index: int
    kind: Literal["embedded", "frame"]
    content_type: str
    size_bytes: int
    checksum_sha256: str
    width: int | None = None
    height: int | None = None
    time_ms: int | None = None
    description: str
    filename: str
    folder_path: str | None
    stored_at: datetime
    tenant_id: str | None = None
    created_by_user_id: str | None = None
    job_id: str | None = None
