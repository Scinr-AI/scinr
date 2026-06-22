"""
storage/null.py — No-op storage repositories for when storage_backend='none'.

These implementations satisfy the RawFileRepository and PageRepository interfaces
without performing any I/O. They are used as the default when no storage backend
is configured, eliminating the need for None checks throughout the codebase.
"""
from scinr.newton.storage.base import PageRepository, RawFileRepository
from scinr.newton.storage.models import ConvertedPageRecord


class NullRawFileRepository(RawFileRepository):
    """No-op implementation. All writes are silently discarded."""

    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None = None,
    ) -> str:
        return ""  # raw_file_id empty string — documented no-storage sentinel


class NullPageRepository(PageRepository):
    """No-op implementation. All writes are silently discarded."""

    async def store_page(
        self,
        raw_file_id: str,
        filename: str,
        folder_path: str | None,
        page_index: int,
        markdown: str,
    ) -> str:
        return ""

    async def get_pages(self, raw_file_id: str) -> list[ConvertedPageRecord]:
        return []
