"""
tests/unit/test_stored_images_storage.py — Optional page-image methods of the
storage layer (WP1 and WP2 of plans/stored-images-implementation-plan.md).

Covers:

- A custom ``RawFileRepository`` that only implements the pre-0.5.0 abstract
  methods still instantiates, and its ``stores_images`` is ``False`` (D4: the
  image methods are optional, not abstract).
- The no-op repositories (``storage/null.py``): what they return and how they
  validate the scope, like the rest of their methods.
- The default ``RawFileRepository.store_image_file`` reads the file and
  delegates to ``store_image``.
- The MongoDB implementation (WP2): order of the writes, the orphan-blob
  cleanup when the insert fails, the scope filter on reads and deletes, and
  an invalid ``image_id`` returning ``None`` — with Motor stand-ins, like the
  other ``raw_files`` unit tests (no real MongoDB).
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson.objectid import ObjectId

from scinr.newton.exceptions import ScopeError, StorageError
from scinr.newton.storage.base import RawFileRepository
from scinr.newton.storage.models import RawFileRecord
from scinr.newton.storage.mongodb.raw_files import MongoDBRawFileRepository
from scinr.newton.storage.null import NullRawFileRepository


class _LegacyRawFileRepository(RawFileRepository):
    """Custom backend written before 0.5.0: only the abstract methods."""

    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        return "legacy-id"

    async def get(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> RawFileRecord | None:
        return None

    async def open(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> AsyncIterator[bytes] | None:
        return None

    async def open_with_record(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[RawFileRecord, AsyncIterator[bytes]] | None:
        return None

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
        return None


class _RecordingRawFileRepository(_LegacyRawFileRepository):
    """Records the ``store_image`` calls it receives."""

    def __init__(self) -> None:
        self.stored: list[dict] = []

    async def store_image(
        self,
        raw_file_id: str,
        page_index: int,
        image_index: int,
        content: bytes,
        **kwargs,
    ) -> str:
        self.stored.append(
            {
                "raw_file_id": raw_file_id,
                "page_index": page_index,
                "image_index": image_index,
                "content": content,
                **kwargs,
            }
        )
        return "image-id-1"


# ---------------------------------------------------------------------------
# D4: a custom backend without the image methods keeps working
# ---------------------------------------------------------------------------


class TestOptionalImageMethods:
    def test_backend_without_image_methods_instantiates(self):
        assert _LegacyRawFileRepository() is not None

    def test_default_stores_images_is_false(self):
        assert _LegacyRawFileRepository().stores_images is False

    async def test_default_store_image_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            await _LegacyRawFileRepository().store_image(
                "rid",
                0,
                0,
                b"png",
                kind="embedded",
                content_type="image/png",
                description="a logo",
                filename="doc.pdf",
                folder_path=None,
            )

    async def test_default_list_images_returns_empty_list(self):
        assert await _LegacyRawFileRepository().list_images("rid") == []

    async def test_default_open_image_returns_none(self):
        assert await _LegacyRawFileRepository().open_image("image-id") is None

    async def test_default_delete_images_returns_zero(self):
        assert await _LegacyRawFileRepository().delete_images("rid") == 0


class TestDefaultStoreImageFile:
    async def test_reads_the_file_and_delegates_to_store_image(self, tmp_path: Path):
        repo = _RecordingRawFileRepository()
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"jpeg bytes")

        image_id = await repo.store_image_file(
            path,
            "rid",
            2,
            1,
            kind="frame",
            content_type="image/jpeg",
            description="a frame",
            filename="clip.mp4",
            folder_path="videos",
            width=640,
            height=360,
            time_ms=12_000,
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )

        assert image_id == "image-id-1"
        assert repo.stored == [
            {
                "raw_file_id": "rid",
                "page_index": 2,
                "image_index": 1,
                "content": b"jpeg bytes",
                "kind": "frame",
                "content_type": "image/jpeg",
                "description": "a frame",
                "filename": "clip.mp4",
                "folder_path": "videos",
                "width": 640,
                "height": 360,
                "time_ms": 12_000,
                "tenant_id": "acme",
                "created_by_user_id": "u1",
                "job_id": "j1",
            }
        ]

    async def test_delegates_to_a_backend_without_store_image(self, tmp_path: Path):
        path = tmp_path / "img.png"
        path.write_bytes(b"png bytes")

        with pytest.raises(NotImplementedError):
            await _LegacyRawFileRepository().store_image_file(
                path,
                "rid",
                0,
                0,
                kind="embedded",
                content_type="image/png",
                description="an image",
                filename="doc.pdf",
                folder_path=None,
            )


# ---------------------------------------------------------------------------
# Null repository: no-op, but validates like a real backend
# ---------------------------------------------------------------------------


class TestNullRepositoryImages:
    def test_stores_images_is_false(self):
        assert NullRawFileRepository().stores_images is False

    async def test_store_image_returns_empty_string_sentinel(self):
        image_id = await NullRawFileRepository().store_image(
            "rid",
            0,
            0,
            b"png",
            kind="embedded",
            content_type="image/png",
            description="a logo",
            filename="doc.pdf",
            folder_path=None,
        )
        assert image_id == ""

    async def test_store_image_file_returns_sentinel_without_opening_the_file(
        self, tmp_path: Path
    ):
        missing = tmp_path / "does-not-exist.png"  # opening it would raise

        image_id = await NullRawFileRepository().store_image_file(
            missing,
            "rid",
            0,
            0,
            kind="embedded",
            content_type="image/png",
            description="a logo",
            filename="doc.pdf",
            folder_path=None,
        )
        assert image_id == ""

    async def test_list_images_returns_empty_list(self):
        assert await NullRawFileRepository().list_images("rid") == []
        assert await NullRawFileRepository().list_images(
            "rid", page_indexes=[0, 1], kind="embedded", tenant_id="acme"
        ) == []

    async def test_open_image_returns_none(self):
        assert await NullRawFileRepository().open_image("image-id") is None

    async def test_delete_images_is_a_safe_noop(self):
        assert await NullRawFileRepository().delete_images("rid") == 0
        assert await NullRawFileRepository().delete_images("not-an-oid") == 0

    @pytest.mark.parametrize(
        "scope",
        [
            {"tenant_id": ""},
            {"created_by_user_id": []},
            {"job_id": []},
            {"job_id": ["j1", ""]},
        ],
    )
    async def test_invalid_scopes_are_rejected(self, scope):
        repo = NullRawFileRepository()
        with pytest.raises(ScopeError):
            await repo.list_images("rid", **scope)
        with pytest.raises(ScopeError):
            await repo.open_image("image-id", **scope)
        with pytest.raises(ScopeError):
            await repo.delete_images("rid", **scope)

    async def test_empty_tenant_on_write_is_rejected(self):
        repo = NullRawFileRepository()
        with pytest.raises(ValueError, match="tenant_id"):
            await repo.store_image(
                "rid",
                0,
                0,
                b"png",
                kind="embedded",
                content_type="image/png",
                description="a logo",
                filename="doc.pdf",
                folder_path=None,
                tenant_id="",
            )
        with pytest.raises(ValueError, match="tenant_id"):
            await repo.store_image_file(
                Path("x.png"),
                "rid",
                0,
                0,
                kind="embedded",
                content_type="image/png",
                description="a logo",
                filename="doc.pdf",
                folder_path=None,
                tenant_id="",
            )


# ---------------------------------------------------------------------------
# MongoDB implementation (WP2) — Motor stand-ins, no real server
# ---------------------------------------------------------------------------


class _FakeConfig:
    mongodb_raw_files_collection = "raw_files"
    mongodb_images_collection = "page_images"
    mongodb_images_gridfs_bucket = "page_image_binaries"


@pytest.fixture(autouse=True)
def patch_mongo_config(monkeypatch):
    """raw_files.py resolves ``get_config`` lazily inside each method, so
    patching the module attribute is enough — no configure() call needed."""
    monkeypatch.setattr("scinr.newton.config.get_config", lambda: _FakeConfig())


class _AsyncCursor:
    """Async iteration over an in-memory list of documents (Motor stand-in)."""

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def __aiter__(self):
        async def _gen():
            for doc in self._docs:
                yield doc

        return _gen()

    async def to_list(self, length=None):
        return list(self._docs)


def _image_doc(**overrides) -> dict:
    doc = {
        "_id": ObjectId(),
        "raw_file_id": "rid",
        "page_index": 0,
        "image_index": 0,
        "kind": "embedded",
        "content_type": "image/png",
        "description": "a logo",
        "filename": "doc.pdf",
        "folder_path": None,
        "width": 10,
        "height": 20,
        "time_ms": None,
        "size_bytes": 3,
        "checksum_sha256": hashlib.sha256(b"png").hexdigest(),
        "stored_at": datetime(2026, 1, 1, tzinfo=UTC),
        "gridfs_id": ObjectId(),
        "tenant_id": "acme",
        "created_by_user_id": None,
        "job_id": None,
    }
    doc.update(overrides)
    return doc


class TestMongoDBStoresImages:
    def test_stores_images_is_true(self):
        assert MongoDBRawFileRepository().stores_images is True

    async def test_store_image_uploads_first_and_then_inserts(self, monkeypatch):
        """Order of the writes: GridFS first, then the record, into the images
        bucket and collection — never into the raw-files ones."""
        gridfs_id = ObjectId()
        inserted_id = ObjectId()
        events: list[str] = []

        async def upload_from_stream(filename, content, metadata=None):
            events.append("upload")
            return gridfs_id

        bucket = MagicMock()
        bucket.upload_from_stream = upload_from_stream
        bucket.delete = AsyncMock()
        images_coll = MagicMock()

        async def insert_one(doc):
            events.append("insert")
            return MagicMock(inserted_id=inserted_id)

        images_coll.insert_one = insert_one
        db = {"page_images": images_coll}

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        bucket_calls: list[str | None] = []
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket_calls.append(name) or bucket,
        )

        image_id = await MongoDBRawFileRepository().store_image(
            "rid",
            1,
            2,
            b"png bytes",
            kind="embedded",
            content_type="image/png",
            description="a logo",
            filename="doc.pdf",
            folder_path="docs",
            width=10,
            height=20,
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )

        assert image_id == str(inserted_id)
        assert events == ["upload", "insert"]
        # The images bucket, not the raw-files one (None).
        assert bucket_calls == ["page_image_binaries"]

    async def test_store_image_full_document_and_metadata(self, monkeypatch):
        """The record lands in ``page_images`` with the gridfs_id and every
        field; the GridFS metadata carries the raw-file link and the owner."""
        gridfs_id = ObjectId()
        inserted_id = ObjectId()
        uploaded: dict = {}

        async def upload_from_stream(filename, content, metadata=None):
            uploaded.update(filename=filename, content=content, metadata=metadata)
            return gridfs_id

        bucket = MagicMock()
        bucket.upload_from_stream = upload_from_stream
        images_coll = MagicMock()
        images_coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))
        db = {"page_images": images_coll}

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        image_id = await MongoDBRawFileRepository().store_image(
            "rid",
            1,
            2,
            b"png bytes",
            kind="embedded",
            content_type="image/png",
            description="a logo",
            filename="doc.pdf",
            folder_path="docs",
            width=10,
            height=20,
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )

        assert image_id == str(inserted_id)
        assert uploaded["filename"] == "doc.pdf"
        assert uploaded["content"] == b"png bytes"
        assert uploaded["metadata"] == {
            "raw_file_id": "rid",
            "page_index": 1,
            "image_index": 2,
            "kind": "embedded",
            "tenant_id": "acme",
            "created_by_user_id": "u1",
            "job_id": "j1",
        }
        doc = images_coll.insert_one.await_args.args[0]
        assert doc["gridfs_id"] == gridfs_id
        assert doc["raw_file_id"] == "rid"
        assert (doc["page_index"], doc["image_index"]) == (1, 2)
        assert doc["kind"] == "embedded"
        assert doc["description"] == "a logo"
        assert doc["size_bytes"] == len(b"png bytes")
        assert doc["checksum_sha256"] == hashlib.sha256(b"png bytes").hexdigest()
        assert doc["tenant_id"] == "acme"

    async def test_store_image_deletes_the_blob_when_the_insert_fails(self, monkeypatch):
        """A failed insert must not leave an orphan binary: the blob is
        deleted (best effort, logged) and the original error propagates."""
        gridfs_id = ObjectId()
        bucket = MagicMock()
        bucket.upload_from_stream = AsyncMock(return_value=gridfs_id)
        bucket.delete = AsyncMock()
        images_coll = MagicMock()
        images_coll.insert_one = AsyncMock(side_effect=RuntimeError("insert boom"))
        db = {"page_images": images_coll}

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        with pytest.raises(RuntimeError, match="insert boom"):
            await MongoDBRawFileRepository().store_image(
                "rid",
                0,
                0,
                b"png",
                kind="embedded",
                content_type="image/png",
                description="a logo",
                filename="doc.pdf",
                folder_path=None,
            )

        bucket.delete.assert_awaited_once_with(gridfs_id)

    async def test_store_image_file_streams_with_the_hashing_reader(
        self, monkeypatch, tmp_path: Path
    ):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"jpeg bytes")
        gridfs_id = ObjectId()
        uploaded: dict = {}

        async def upload_from_stream(filename, source, metadata=None):
            uploaded.update(filename=filename, source=source, metadata=metadata)
            while source.read(4):
                pass
            return gridfs_id

        bucket = MagicMock()
        bucket.upload_from_stream = upload_from_stream
        images_coll = MagicMock()
        images_coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))
        db = {"page_images": images_coll}

        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        image_id = await MongoDBRawFileRepository().store_image_file(
            path,
            "rid",
            0,
            3,
            kind="frame",
            content_type="image/jpeg",
            description="a frame",
            filename="clip.mp4",
            folder_path=None,
            time_ms=12_000,
        )

        assert isinstance(image_id, str)
        assert not isinstance(uploaded["source"], (bytes, bytearray))
        doc = images_coll.insert_one.await_args.args[0]
        assert doc["size_bytes"] == len(b"jpeg bytes")
        assert doc["checksum_sha256"] == hashlib.sha256(b"jpeg bytes").hexdigest()
        assert doc["gridfs_id"] == gridfs_id
        assert doc["time_ms"] == 12_000


class TestMongoDBListImages:
    async def test_filters_by_raw_file_scope_and_sorts(self, monkeypatch):
        docs = [_image_doc()]
        find_calls: list[dict] = {}

        class _Collection:
            def find(self, query, sort=None):
                find_calls["query"] = query
                find_calls["sort"] = sort
                return _AsyncCursor(docs)

        db = {"page_images": _Collection()}
        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)

        repo = MongoDBRawFileRepository()
        records = await repo.list_images(
            "rid", page_indexes=[0, 3], kind="embedded", tenant_id="acme"
        )

        assert find_calls["query"] == {
            "raw_file_id": "rid",
            "tenant_id": {"$in": ["acme"]},
            "page_index": {"$in": [0, 3]},
            "kind": "embedded",
        }
        assert find_calls["sort"] == [("page_index", 1), ("image_index", 1)]
        assert len(records) == 1
        record = records[0]
        assert record.raw_file_id == "rid"
        assert record.description == "a logo"
        assert (record.width, record.height) == (10, 20)

    async def test_without_optional_filters_queries_only_scope_and_raw_file(
        self, monkeypatch
    ):
        find_calls: list[dict] = {}

        class _Collection:
            def find(self, query, sort=None):
                find_calls["query"] = query
                return _AsyncCursor([])

        db = {"page_images": _Collection()}
        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", lambda: db)

        assert await MongoDBRawFileRepository().list_images("rid", tenant_id="acme") == []
        assert find_calls["query"] == {
            "raw_file_id": "rid",
            "tenant_id": {"$in": ["acme"]},
        }


class TestMongoDBOpenImage:
    async def test_record_and_stream_from_the_filtered_lookup(self, monkeypatch):
        gridfs_id = ObjectId()
        doc = _image_doc(gridfs_id=gridfs_id)
        find_calls: list[dict] = {}

        class _Collection:
            async def find_one(self, query):
                find_calls["query"] = query
                return doc

        grid_out = MagicMock()
        chunks = iter([b"png-bytes"])

        async def readchunk():
            return next(chunks, b"")

        close_calls: list[bool] = []

        async def open_download_stream(gid):
            assert gid == gridfs_id
            return grid_out

        grid_out.readchunk = readchunk
        grid_out.close = lambda: close_calls.append(True)
        bucket = MagicMock()
        bucket.open_download_stream = open_download_stream

        bucket_calls: list[str | None] = []
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db",
            lambda: {"page_images": _Collection()},
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket_calls.append(name) or bucket,
        )

        repo = MongoDBRawFileRepository()
        result = await repo.open_image(str(doc["_id"]), tenant_id="acme")

        assert result is not None
        record, stream = result
        assert record.id == str(doc["_id"])
        assert find_calls["query"] == {"_id": doc["_id"], "tenant_id": {"$in": ["acme"]}}
        assert bucket_calls == ["page_image_binaries"]
        # The binary is opened through the gridfs_id of the filtered record.
        assert [chunk async for chunk in stream] == [b"png-bytes"]
        assert close_calls == [True]

    async def test_invalid_image_id_returns_none_and_never_touches_db(self, monkeypatch):
        get_db_mock = MagicMock()
        get_bucket_mock = MagicMock()
        monkeypatch.setattr("scinr.newton.storage.mongodb.raw_files.get_db", get_db_mock)
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", get_bucket_mock
        )

        assert await MongoDBRawFileRepository().open_image("not-an-oid") is None

        get_db_mock.assert_not_called()
        get_bucket_mock.assert_not_called()

    async def test_record_outside_the_scope_returns_none(self, monkeypatch):
        class _Collection:
            async def find_one(self, query):
                return None

        bucket = MagicMock()
        bucket.open_download_stream = AsyncMock()
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db",
            lambda: {"page_images": _Collection()},
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        repo = MongoDBRawFileRepository()
        assert await repo.open_image(str(ObjectId()), tenant_id="other") is None
        bucket.open_download_stream.assert_not_called()

    async def test_a_missing_binary_is_a_storage_error(self, monkeypatch):
        from gridfs.errors import NoFile

        doc = _image_doc()

        class _Collection:
            async def find_one(self, query):
                return doc

        bucket = MagicMock()
        bucket.open_download_stream = AsyncMock(side_effect=NoFile("gone"))
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db",
            lambda: {"page_images": _Collection()},
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        with pytest.raises(StorageError):
            await MongoDBRawFileRepository().open_image(str(doc["_id"]))


class TestMongoDBDeleteImages:
    async def test_deletes_each_blob_and_then_the_records(self, monkeypatch):
        docs = [
            _image_doc(_id=ObjectId(), gridfs_id=ObjectId(), page_index=0),
            _image_doc(_id=ObjectId(), gridfs_id=ObjectId(), page_index=1),
        ]
        deleted_blobs: list[ObjectId] = []
        delete_many_calls: list[dict] = []

        class _Collection:
            def find(self, query, sort=None):
                return _AsyncCursor(docs)

            async def delete_many(self, query):
                delete_many_calls.append(query)
                return MagicMock(deleted_count=2)

        bucket = MagicMock()

        async def delete_blob(gid):
            deleted_blobs.append(gid)

        bucket.delete = delete_blob
        bucket_calls: list[str | None] = []
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db",
            lambda: {"page_images": _Collection()},
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket_calls.append(name) or bucket,
        )

        repo = MongoDBRawFileRepository()
        deleted = await repo.delete_images("rid", tenant_id="acme")

        assert deleted == 2
        assert deleted_blobs == [doc["gridfs_id"] for doc in docs]
        assert delete_many_calls == [{"raw_file_id": "rid", "tenant_id": {"$in": ["acme"]}}]
        assert bucket_calls == ["page_image_binaries"]

    async def test_a_missing_blob_is_ignored_and_the_record_is_still_deleted(
        self, monkeypatch, caplog
    ):
        from gridfs.errors import NoFile

        docs = [_image_doc(_id=ObjectId(), gridfs_id=ObjectId())]
        delete_many_calls: list[dict] = []

        class _Collection:
            def find(self, query, sort=None):
                return _AsyncCursor(docs)

            async def delete_many(self, query):
                delete_many_calls.append(query)
                return MagicMock(deleted_count=1)

        bucket = MagicMock()
        bucket.delete = AsyncMock(side_effect=NoFile("gone"))
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db",
            lambda: {"page_images": _Collection()},
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        with caplog.at_level("WARNING", logger="scinr.newton.storage.mongodb.raw_files"):
            deleted = await MongoDBRawFileRepository().delete_images("rid")

        assert deleted == 1
        assert "already gone" in caplog.text
        assert delete_many_calls == [{"raw_file_id": "rid"}]

    async def test_no_matching_records_is_a_noop_zero(self, monkeypatch):
        class _Collection:
            def find(self, query, sort=None):
                return _AsyncCursor([])

            async def delete_many(self, query):  # pragma: no cover - must not run
                raise AssertionError("delete_many must not be called")

        bucket = MagicMock()
        bucket.delete = AsyncMock()
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_db",
            lambda: {"page_images": _Collection()},
        )
        monkeypatch.setattr(
            "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket",
            lambda name=None: bucket,
        )

        assert await MongoDBRawFileRepository().delete_images("rid") == 0
        bucket.delete.assert_not_called()
