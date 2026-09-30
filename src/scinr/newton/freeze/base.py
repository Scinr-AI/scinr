"""
freeze/base.py — Abstract repository interface for frozen-document snapshots.

A snapshot is the streaming JSON written by
:func:`scinr.newton.ingest.freeze.export_document_snapshot` (see
``docs/user-guides/document-freezing.md`` for its format). The freeze backend
stores it so that :func:`~scinr.newton.ingest.freeze.freeze_document` can
remove the document subtree from the graph and
:func:`~scinr.newton.ingest.restore.restore_document` can rebuild it later.

This interface is deliberately separate from
:class:`~scinr.newton.storage.base.RawFileRepository`: raw files never need
to be read back through the repository, snapshots always do.

Multi-tenancy
-------------
A snapshot belongs to exactly one tenant (the freeze selector fixes a single
tenant), and follows the same tenant contract as the storage repositories:

- **Writes** (:meth:`FreezeRepository.store_snapshot`) take ``tenant_id`` in
  its public-API form: ``None`` and ``"__public__"`` both mean public and are
  stored as :func:`~scinr.newton.utils.tenancy.tenant_key` (``"__public__"``).
- **Reads, lookups and deletes** take the **stored** tenant key (the
  ``tenant_id`` of the frozen ``:Document`` stub) and treat a snapshot of
  another tenant exactly as a snapshot that does not exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SnapshotRecord:
    """A stored snapshot, as :meth:`FreezeRepository.find_snapshots` lists it.

    Attributes:
        frozen_blob_id: Id to read / delete the snapshot with.
        mode: ``"freeze"``, ``"backup"`` or ``"export"`` (the call that stored it).
        frozen_at: ISO timestamp of the export.
        documents: One ``{"path", "version", "job_id", "created_by_user_id"}``
            dict per document entry of the snapshot (``job_id`` /
            ``created_by_user_id`` only in snapshots stored since they were
            recorded).
    """

    frozen_blob_id: str
    mode: str | None
    frozen_at: str | None
    documents: list[dict[str, Any]] = field(default_factory=list)


class FreezeRepository(ABC):
    """Stores the JSON snapshots of frozen (or backed-up) documents."""

    @abstractmethod
    async def store_snapshot(
        self,
        path: Path,
        *,
        tenant_id: str | None,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
        metadata: dict[str, Any],
    ) -> str:
        """Upload the snapshot file at *path* and return its ``frozen_blob_id``.

        The file is already fully written; backends must upload it in
        streaming (never read it whole into memory) — see
        ``MongoDBRawFileRepository.store_file`` for the reference pattern.

        Parameters
        ----------
        path:
            Local JSON snapshot file.
        tenant_id:
            Owning tenant (``None`` or ``"__public__"`` = public), stored as
            :func:`~scinr.newton.utils.tenancy.tenant_key`. Mandatory: a
            snapshot belongs to a single tenant.
        created_by_user_id, job_id:
            The freeze selector's provenance filters (one value or several),
            kept for audit / listing.
        metadata:
            Descriptive fields about the snapshot content (documents,
            ``schema_version``, ``frozen_at``, ``keep_flags``…), stored as is.

        Returns
        -------
        str
            The ``frozen_blob_id`` assigned by the backend.
        """

    @abstractmethod
    async def read_snapshot_to_file(
        self, frozen_blob_id: str, dest_path: Path, *, tenant_id: str
    ) -> bool:
        """Download the snapshot *frozen_blob_id* to *dest_path* in streaming.

        Parameters
        ----------
        frozen_blob_id:
            Id returned by :meth:`store_snapshot`.
        dest_path:
            Local file to write (created or truncated).
        tenant_id:
            **Stored** tenant key of the frozen document.

        Returns
        -------
        bool
            ``True`` when the snapshot was written. ``False`` — and nothing
            written — when it does not exist, the id is invalid for the
            backend, or it belongs to another tenant (indistinguishable).
        """

    @abstractmethod
    async def delete_snapshot(self, frozen_blob_id: str, *, tenant_id: str) -> None:
        """Delete the snapshot *frozen_blob_id* (binary and metadata).

        Must be idempotent: when the snapshot does not exist (already
        deleted, invalid id, repeated call) or belongs to another tenant, it
        logs and returns without raising — same contract as
        :meth:`~scinr.newton.storage.base.RawFileRepository.delete`.

        Parameters
        ----------
        frozen_blob_id:
            Id returned by :meth:`store_snapshot`.
        tenant_id:
            **Stored** tenant key of the frozen document.
        """

    async def find_snapshots(
        self,
        *,
        tenant_id: str,
        path: str | None = None,
        version: int | None = None,
        job_id: Sequence[str] | None = None,
        created_by_user_id: Sequence[str] | None = None,
    ) -> list[SnapshotRecord]:
        """List the snapshots of *tenant_id* holding a matching document, newest first.

        Used by ``restore_document()`` when the selected document has no
        frozen stub in the graph (it was deleted, or the snapshot is a
        backup). A document matches when one entry of the snapshot's
        ``documents`` metadata satisfies every filter given (*path* /
        *version*, or *job_id*; *created_by_user_id* on top).

        Optional for custom backends: the default raises
        :class:`~scinr.newton.exceptions.FreezeError`, and callers must then
        name the snapshot explicitly (``restore_document(frozen_blob_id=...)``).

        Parameters
        ----------
        tenant_id:
            **Stored** tenant key.
        job_id, created_by_user_id:
            Lists of values (matched with IN), as ``delete_document()`` takes them.
        """
        from scinr.newton.exceptions import FreezeError

        raise FreezeError(
            f"{type(self).__name__} cannot look snapshots up; pass frozen_blob_id= "
            "to restore_document() to restore a document that has no frozen stub."
        )
