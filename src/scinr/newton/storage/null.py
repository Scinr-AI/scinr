"""
storage/null.py — No-op storage repositories for when storage_backend='none'.

These implementations satisfy the RawFileRepository and PageRepository interfaces
without performing any I/O. They are used as the default when no storage backend
is configured, eliminating the need for None checks throughout the codebase.

Arguments are still validated like in a real backend (an empty ``tenant_id`` on
write, an invalid read scope) so that a misuse does not go unnoticed only
because storage is disabled.
"""
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from scinr.newton.exceptions import StorageError
from scinr.newton.storage.base import PageRepository, RawFileRepository
from scinr.newton.storage.models import ConvertedPageRecord, RawFileRecord
from scinr.newton.utils.scope import make_scope
from scinr.newton.utils.tenancy import tenant_key


class NullRawFileRepository(RawFileRepository):
    """No-op implementation. All writes are silently discarded."""

    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None = None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        tenant_key(tenant_id)
        return ""  # raw_file_id empty string — documented no-storage sentinel

    async def store_file(
        self,
        path: Path,
        filename: str,
        content_type: str,
        folder_path: str | None = None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        tenant_key(tenant_id)
        return ""  # same sentinel as store(); never opens *path*

    async def get(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> RawFileRecord | None:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return None  # nothing is ever stored

    async def open(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> AsyncIterator[bytes] | None:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        raise StorageError(
            "storage_backend='none' keeps no original files; configure a persistent "
            "storage backend to read them."
        )

    async def open_with_record(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[RawFileRecord, AsyncIterator[bytes]] | None:
        await self.open(
            raw_file_id,
            tenant_id=tenant_id,
            include_public=include_public,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )
        return None  # unreachable: open() always raises

    async def list_raw_files(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
        folder_path: str | None = None,
        filename: str | None = None,
    ) -> list[RawFileRecord]:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return []

    async def delete(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> None:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return None  # no-op: nothing is ever stored, so nothing to delete


class NullPageRepository(PageRepository):
    """No-op implementation. All writes are silently discarded."""

    async def store_page(
        self,
        raw_file_id: str,
        filename: str,
        folder_path: str | None,
        page_index: int,
        markdown: str,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        tenant_key(tenant_id)
        return ""

    async def get_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return []

    async def get_pages_by_ids(
        self,
        page_ids: Sequence[str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return []

    async def delete_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return 0  # no-op: nothing is ever stored, so nothing to delete
