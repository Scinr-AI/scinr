"""
tests/unit/test_storage_raw_files_streaming.py — Unit tests for the streaming
upload path of raw files (``RawFileRepository.store_file``).

No real MongoDB is used: get_db()/get_gridfs_bucket()/get_config() are patched.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson.objectid import ObjectId

from scinr.newton.storage.base import RawFileRepository
from scinr.newton.storage.mongodb.raw_files import MongoDBRawFileRepository, _HashingReader


class _FakeConfig:
    mongodb_raw_files_collection = "raw_files"


@pytest.fixture(autouse=True)
def patch_get_config(monkeypatch):
    monkeypatch.setattr("scinr.newton.config.get_config", lambda: _FakeConfig())


@pytest.fixture
def payload() -> bytes:
    # Not a multiple of any read size used below.
    return bytes(range(256)) * 1000 + b"tail"


class TestHashingReader:
    @pytest.mark.parametrize("read_size", [1, 7, 255, 4096, 10**6])
    def test_sha256_and_size_match_hashlib_for_varied_read_sizes(self, payload, read_size):
        reader = _HashingReader(io.BytesIO(payload))

        chunks = []
        while chunk := reader.read(read_size):
            chunks.append(chunk)

        assert b"".join(chunks) == payload
        assert reader.size == len(payload)
        assert reader.hexdigest() == hashlib.sha256(payload).hexdigest()

    def test_read_all_with_default_and_negative_size(self, payload):
        for args in ((), (-1,)):
            reader = _HashingReader(io.BytesIO(payload))
            assert reader.read(*args) == payload
            assert reader.size == len(payload)
            assert reader.hexdigest() == hashlib.sha256(payload).hexdigest()


class TestMongoDBStoreFile:
    async def test_streams_a_file_object_and_records_correct_metadata(
        self, monkeypatch, tmp_path: Path, payload
    ):
        path = tmp_path / "big.pdf"
        path.write_bytes(payload)
        gridfs_id = ObjectId()
        inserted_id = ObjectId()

        uploaded: dict = {}

        async def upload_from_stream(filename, source, metadata=None):
            uploaded["filename"] = filename
            uploaded["source"] = source
            uploaded["metadata"] = metadata
            # Emulate GridIn: consume the source in chunk_size blocks.
            while source.read(255 * 1024):
                pass
            return gridfs_id

        bucket = MagicMock()
        bucket.upload_from_stream = upload_from_stream
        collection = MagicMock()
        collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))
        db = {"raw_files": collection}

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", lambda: bucket
        )

        raw_file_id = await MongoDBRawFileRepository().store_file(
            path=path,
            filename="big.pdf",
            content_type="application/pdf",
            folder_path="a/b",
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )

        assert raw_file_id == str(inserted_id)
        assert not isinstance(uploaded["source"], (bytes, bytearray))
        assert uploaded["metadata"] == {
            "content_type": "application/pdf",
            "folder_path": "a/b",
            "tenant_id": "acme",
            "created_by_user_id": "u1",
            "job_id": "j1",
        }
        doc = collection.insert_one.await_args.args[0]
        assert doc["size_bytes"] == len(payload)
        assert doc["checksum_sha256"] == hashlib.sha256(payload).hexdigest()
        assert doc["gridfs_id"] == gridfs_id
        assert doc["filename"] == "big.pdf"
        assert doc["folder_path"] == "a/b"
        assert (doc["tenant_id"], doc["created_by_user_id"], doc["job_id"]) == ("acme", "u1", "j1")

    async def test_store_still_uploads_bytes_and_shares_metadata_insertion(self, monkeypatch):
        gridfs_id = ObjectId()
        bucket = MagicMock()
        bucket.upload_from_stream = AsyncMock(return_value=gridfs_id)
        collection = MagicMock()
        collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db", lambda: {"raw_files": collection}
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", lambda: bucket
        )

        await MongoDBRawFileRepository().store(
            filename="a.txt", content=b"hello", content_type="text/plain", folder_path=None
        )

        assert bucket.upload_from_stream.await_args.args[1] == b"hello"
        # No tenant = public, stored as the reserved key (never null).
        assert bucket.upload_from_stream.await_args.kwargs["metadata"]["tenant_id"] == "__public__"
        doc = collection.insert_one.await_args.args[0]
        assert doc["tenant_id"] == "__public__"
        assert doc["size_bytes"] == 5
        assert doc["checksum_sha256"] == hashlib.sha256(b"hello").hexdigest()


class TestDefaultStoreFile:
    async def test_custom_repo_implementing_only_store_receives_the_file_bytes(
        self, tmp_path: Path
    ):
        received: dict = {}

        class _CustomRepo(RawFileRepository):
            async def store(self, filename, content, content_type, folder_path, **owner):
                received.update(
                    filename=filename,
                    content=content,
                    content_type=content_type,
                    folder_path=folder_path,
                    **owner,
                )
                return "custom-id"

            async def get(self, raw_file_id, **scope):
                return None

            async def open(self, raw_file_id, **scope):
                return None

            async def open_with_record(self, raw_file_id, **scope):
                return None

            async def list_raw_files(self, **scope):
                return []

            async def delete(self, raw_file_id, **scope):
                return None

        path = tmp_path / "doc.bin"
        path.write_bytes(b"\x00\x01\x02")

        raw_file_id = await _CustomRepo().store_file(
            path=path,
            filename="doc.bin",
            content_type="application/x",
            folder_path=None,
            tenant_id="acme",
            job_id="j1",
        )

        assert raw_file_id == "custom-id"
        assert received == {
            "filename": "doc.bin",
            "content": b"\x00\x01\x02",
            "content_type": "application/x",
            "folder_path": None,
            "tenant_id": "acme",
            "created_by_user_id": None,
            "job_id": "j1",
        }
