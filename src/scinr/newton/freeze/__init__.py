"""
freeze — Pluggable backend for frozen-document snapshots.

Mirrors ``storage/``: an abstract repository (:class:`FreezeRepository`), a
factory driven by ``ScinrConfig.freeze_backend`` and a MongoDB/GridFS
implementation. The graph side (export, freeze, restore) lives in
``ingest/freeze.py`` and ``ingest/restore.py``.

Usage
-----
from scinr.newton.freeze import get_freeze_storage
repo = get_freeze_storage()

blob_id = await repo.store_snapshot(tmp_path, tenant_id="acme", metadata={...})  # None = public
found   = await repo.read_snapshot_to_file(blob_id, dest, tenant_id="acme")      # stored key
await repo.delete_snapshot(blob_id, tenant_id="acme")                             # idempotent
records = await repo.find_snapshots(tenant_id="acme", path="a.pdf")              # newest first
"""

from __future__ import annotations

from scinr.newton.freeze.base import FreezeRepository, SnapshotRecord
from scinr.newton.freeze.factory import get_freeze_storage

__all__ = ["FreezeRepository", "SnapshotRecord", "get_freeze_storage"]
