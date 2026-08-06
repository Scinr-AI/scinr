"""
tests/unit/test_storage_mongodb_delete.py — Unit tests for
MongoDBRawFileRepository.delete() and MongoDBPageRepository.delete_pages().

No real MongoDB is used. get_db()/get_gridfs_bucket() are monkeypatched with
AsyncMock/MagicMock stand-ins, and get_config() is monkeypatched to avoid
requiring a prior configure() call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson.objectid import ObjectId
from gridfs.errors import NoFile

from scinr.newton.storage.mongodb.pages import MongoDBPageRepository
from scinr.newton.storage.mongodb.raw_files import MongoDBRawFileRepository


class _FakeConfig:
    mongodb_raw_files_collection = "raw_files"
    mongodb_pages_collection = "converted_pages"


@pytest.fixture(autouse=True)
def patch_get_config(monkeypatch):
    """Both raw_files.py and pages.py do a lazy
    `from scinr.newton.config import get_config` inside each method, so
    patching the module-level attribute is enough — no prior configure()
    call is required for these tests.
    """
    monkeypatch.setattr("scinr.newton.config.get_config", lambda: _FakeConfig())


# ---------------------------------------------------------------------------
# MongoDBRawFileRepository.delete()
# ---------------------------------------------------------------------------


class TestMongoDBRawFileRepositoryDelete:
    async def test_normal_case_deletes_gridfs_binary_and_metadata(self, monkeypatch):
        object_id = ObjectId()
        gridfs_id = ObjectId()

        collection = MagicMock()
        collection.find_one = AsyncMock(return_value={"_id": object_id, "gridfs_id": gridfs_id})
        collection.delete_one = AsyncMock()
        db = {"raw_files": collection}

        bucket = MagicMock()
        bucket.delete = AsyncMock()

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", lambda: bucket
        )

        repo = MongoDBRawFileRepository()
        await repo.delete(str(object_id))

        collection.find_one.assert_awaited_once_with({"_id": object_id})
        bucket.delete.assert_awaited_once_with(gridfs_id)
        collection.delete_one.assert_awaited_once_with({"_id": object_id})

    async def test_metadata_not_found_is_a_noop_without_raising(self, monkeypatch):
        collection = MagicMock()
        collection.find_one = AsyncMock(return_value=None)
        collection.delete_one = AsyncMock()
        db = {"raw_files": collection}

        bucket = MagicMock()
        bucket.delete = AsyncMock()

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", lambda: bucket
        )

        repo = MongoDBRawFileRepository()
        await repo.delete(str(ObjectId()))  # must not raise

        collection.delete_one.assert_not_awaited()
        bucket.delete.assert_not_awaited()

    async def test_gridfs_nofile_is_swallowed_and_metadata_still_deleted(self, monkeypatch):
        object_id = ObjectId()
        gridfs_id = ObjectId()

        collection = MagicMock()
        collection.find_one = AsyncMock(return_value={"_id": object_id, "gridfs_id": gridfs_id})
        collection.delete_one = AsyncMock()
        db = {"raw_files": collection}

        bucket = MagicMock()
        bucket.delete = AsyncMock(side_effect=NoFile("binary already gone"))

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", lambda: bucket
        )

        repo = MongoDBRawFileRepository()
        await repo.delete(str(object_id))  # must not raise despite NoFile

        bucket.delete.assert_awaited_once_with(gridfs_id)
        collection.delete_one.assert_awaited_once_with({"_id": object_id})

    async def test_invalid_object_id_is_a_noop_and_never_touches_db(self, monkeypatch):
        get_db_mock = MagicMock()
        get_bucket_mock = MagicMock()
        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", get_db_mock)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", get_bucket_mock
        )

        repo = MongoDBRawFileRepository()
        await repo.delete("not-a-valid-object-id")  # must not raise

        get_db_mock.assert_not_called()
        get_bucket_mock.assert_not_called()


# ---------------------------------------------------------------------------
# MongoDBPageRepository.delete_pages()
# ---------------------------------------------------------------------------


class TestMongoDBPageRepositoryDeletePages:
    async def test_zero_matches_returns_zero_without_raising(self, monkeypatch):
        collection = MagicMock()
        collection.delete_many = AsyncMock(return_value=MagicMock(deleted_count=0))
        db = {"converted_pages": collection}

        monkeypatch.setattr("scinr.newton.storage.mongodb.pages.get_db", lambda: db)

        repo = MongoDBPageRepository()
        deleted_count = await repo.delete_pages("raw-file-with-no-pages")

        assert deleted_count == 0
        collection.delete_many.assert_awaited_once_with(
            {"raw_file_id": "raw-file-with-no-pages"}
        )

    async def test_multiple_matches_returns_deleted_count(self, monkeypatch):
        collection = MagicMock()
        collection.delete_many = AsyncMock(return_value=MagicMock(deleted_count=7))
        db = {"converted_pages": collection}

        monkeypatch.setattr("scinr.newton.storage.mongodb.pages.get_db", lambda: db)

        repo = MongoDBPageRepository()
        deleted_count = await repo.delete_pages("raw-file-with-pages")

        assert deleted_count == 7
        collection.delete_many.assert_awaited_once_with({"raw_file_id": "raw-file-with-pages"})
