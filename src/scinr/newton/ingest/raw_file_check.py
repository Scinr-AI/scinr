"""
ingest/raw_file_check.py — Verify that a Document's ``raw_file_id`` belongs to its tenant.

A ``:Document`` points at its original upload through ``raw_file_id``. That id
comes from the converted / extracted JSON, which a caller can edit: a tenant-B
``extract-*.json`` carrying the ``raw_file_id`` of a tenant-A file would give B
read access to A's pages (``navigation.pages``) and, on ``delete_document``,
delete A's original. So before a Document with a non-empty ``raw_file_id`` is
written to the graph, the stored raw-file record must exist **and** carry the
Document's tenant:

- the lookup passes the **stored** tenant key (``tenant_key``), never the API
  value: on reads ``None`` means "every tenant", while a public document must
  only accept a public file;
- only the tenant must match — ``created_by_user_id`` / ``job_id`` of the upload
  may legitimately differ from the Document's (preprocess and ingestion can run
  in different jobs);
- a public file is never "claimed" by a tenant document;
- with ``storage_backend="none"`` there is nothing to verify against, so a
  non-empty ``raw_file_id`` is rejected (empty it in the JSON, or configure the
  storage backend the file was converted with).

Storage repositories are async (Motor binds to the event loop it first runs
on), so the check runs in the async ingestion entry points — ``ingest_one``,
``ingest_one_from_path`` and ``run_ingestion`` — before the synchronous graph
write is dispatched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scinr.newton.exceptions import IngestionError
from scinr.newton.utils.tenancy import tenant_key


async def verify_raw_file_owner(
    raw_file_id: str | None, tenant_id: str | None, *, document: str
) -> None:
    """Raise :class:`IngestionError` unless *raw_file_id* is empty or is a stored
    raw file of *tenant_id* (public-API form, ``None`` = public).

    Parameters
    ----------
    raw_file_id:
        The Document's ``raw_file_id`` (``""`` / ``None`` = no stored original,
        nothing to check).
    tenant_id:
        The Document's **effective** tenant — after the caller's override.
    document:
        Name / path used in the error message.
    """
    if not raw_file_id:
        return

    from scinr.newton.config import get_config

    if get_config().storage_backend == "none":
        raise IngestionError(
            f"Document {document!r} references raw_file_id={raw_file_id!r}, but "
            "storage_backend='none' cannot verify it belongs to the document's tenant. "
            "Configure the storage backend the file was converted with, or clear "
            "raw_file_id in the JSON."
        )

    from scinr.newton.storage.factory import get_storage

    raw_repo, _ = get_storage()
    record = await raw_repo.get(raw_file_id, tenant_id=tenant_key(tenant_id))
    if record is None:
        raise IngestionError(
            f"Document {document!r}: raw_file_id={raw_file_id!r} does not exist or does "
            f"not belong to the document's tenant ({tenant_key(tenant_id)!r}). "
            "Refusing to link another tenant's original file."
        )


def effective_tenant(override: str | None, baked: str | None) -> str | None:
    """The tenant a Document is ingested under: the caller's *override* when
    given, otherwise the value *baked* into the Document / JSON (see
    ``ingest.loader._apply_metadata_overrides``)."""
    return override if override is not None else baked


async def verify_document(doc: Any, tenant_id: str | None) -> None:
    """:func:`verify_raw_file_owner` for an in-memory ``Document`` ingested with
    the caller's *tenant_id* override."""
    await verify_raw_file_owner(
        getattr(doc, "raw_file_id", None),
        effective_tenant(tenant_id, getattr(doc, "tenant_id", None)),
        document=getattr(doc, "doc_path", None) or getattr(doc, "document_name", "?"),
    )


async def verify_json_file(path: Path, tenant_id: str | None) -> None:
    """:func:`verify_raw_file_owner` for an ``extract-*.json`` ingested with the
    caller's *tenant_id* override.

    Only the two fields needed are read. An unreadable / invalid file is not
    reported here — it fails with its own, clearer error on full validation,
    before anything is written.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    await verify_raw_file_owner(
        raw.get("raw_file_id"),
        effective_tenant(tenant_id, raw.get("tenant_id")),
        document=str(path),
    )


__all__ = [
    "effective_tenant",
    "verify_document",
    "verify_json_file",
    "verify_raw_file_owner",
]
