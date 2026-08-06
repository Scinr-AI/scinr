"""
tests/unit/test_storage_null.py — Unit tests for the no-op storage
repositories in scinr.newton.storage.null (used when storage_backend='none').
"""

from __future__ import annotations

from scinr.newton.storage.null import NullPageRepository, NullRawFileRepository


class TestNullRawFileRepository:
    async def test_store_returns_empty_string_sentinel(self):
        repo = NullRawFileRepository()

        raw_file_id = await repo.store(
            filename="doc.pdf",
            content=b"binary content",
            content_type="application/pdf",
            folder_path=None,
        )

        assert raw_file_id == ""

    async def test_delete_is_a_safe_noop_for_any_raw_file_id(self):
        repo = NullRawFileRepository()

        # Must not raise for an empty id, a plausible-looking id, or a
        # garbage value — nothing is ever stored, so nothing to delete.
        assert await repo.delete("") is None
        assert await repo.delete("507f1f77bcf86cd799439011") is None
        assert await repo.delete("not-an-object-id") is None

    async def test_delete_called_repeatedly_stays_a_noop(self):
        repo = NullRawFileRepository()

        for _ in range(3):
            assert await repo.delete("some-id") is None


class TestNullPageRepository:
    async def test_store_page_returns_empty_string_sentinel(self):
        repo = NullPageRepository()

        page_id = await repo.store_page(
            raw_file_id="",
            filename="doc",
            folder_path=None,
            page_index=0,
            markdown="# Hello",
        )

        assert page_id == ""

    async def test_get_pages_returns_empty_list(self):
        repo = NullPageRepository()

        pages = await repo.get_pages("some-id")

        assert pages == []

    async def test_delete_pages_returns_zero_and_never_raises(self):
        repo = NullPageRepository()

        assert await repo.delete_pages("") == 0
        assert await repo.delete_pages("507f1f77bcf86cd799439011") == 0
        assert await repo.delete_pages("not-an-object-id") == 0

    async def test_delete_pages_called_repeatedly_stays_zero(self):
        repo = NullPageRepository()

        for _ in range(3):
            assert await repo.delete_pages("some-id") == 0
