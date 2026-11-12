"""
storage — Database abstraction layer for scinr-ingest.

Provides repository interfaces and implementations for persisting:
- Raw files (binary originals) via GridFS
- Converted pages (Markdown text) via a MongoDB collection
- Described page images and video frames (optional for custom backends,
  ``RawFileRepository.stores_images``) via their own collection and bucket

Usage
-----
from scinr.newton.storage.factory import get_storage
raw_repo, page_repo = get_storage()

# Writes: tenant_id=None / "__public__" = public (stored as "__public__")
raw_file_id = await raw_repo.store(filename, content, content_type, folder_path,
                                   tenant_id="acme", created_by_user_id="u1", job_id="j1")
page_id     = await page_repo.store_page(raw_file_id, filename, folder_path, page_index,
                                         markdown, tenant_id="acme")

# Reads / deletes: same filters as the navigation API (tenant_id=None = all tenants)
record = await raw_repo.get(raw_file_id, tenant_id="acme")        # None if absent / out of scope
stream = await raw_repo.open(raw_file_id, tenant_id="acme")       # async iterator of bytes
files  = await raw_repo.list_raw_files(tenant_id="acme", job_id=["j1", "j2"])
pages  = await page_repo.get_pages(raw_file_id, tenant_id="acme")
await raw_repo.delete(raw_file_id, tenant_id="acme")              # idempotent; no-op if gone
pages_deleted = await page_repo.delete_pages(raw_file_id, tenant_id="acme")

# Stored images (when raw_repo.stores_images)
images = await raw_repo.list_images(raw_file_id, page_indexes=[0, 1], tenant_id="acme")
opened = await raw_repo.open_image(images[0].id, tenant_id="acme")  # (record, stream) | None
images_deleted = await raw_repo.delete_images(raw_file_id, tenant_id="acme")
"""

from __future__ import annotations

from scinr.newton.storage.factory import get_storage

__all__ = ["get_storage"]
