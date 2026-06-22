"""
storage — Database abstraction layer for scinr-ingest.

Provides repository interfaces and implementations for persisting:
- Raw files (binary originals) via GridFS
- Converted pages (Markdown text) via a MongoDB collection

Usage
-----
from scinr.newton.storage.factory import get_storage
raw_file_id = await raw_repo.store(filename, content, content_type, folder_path)
page_id     = await page_repo.store_page(raw_file_id, filename, folder_path, page_index, markdown)
pages       = await page_repo.get_pages(raw_file_id)
"""

from __future__ import annotations

from scinr.newton.storage.factory import get_storage

__all__ = ["get_storage"]
