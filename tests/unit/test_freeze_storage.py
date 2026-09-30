"""
tests/unit/test_freeze_storage.py — The freeze backend: configuration cascade,
``get_freeze_storage()`` and ``MongoDBFreezeRepository``.

The repository runs against small in-memory stand-ins for the Motor
collection and the GridFS bucket (no real MongoDB), like
tests/unit/test_storage_tenancy.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from bson.objectid import ObjectId
from gridfs.errors import NoFile

from scinr.newton.config import configure
from scinr.newton.exceptions import ConfigurationError, StorageError
from scinr.newton.freeze.base import FreezeRepository
from scinr.newton.freeze.factory import get_freeze_storage
from scinr.newton.freeze.mongodb.repository import MongoDBFreezeRepository

_NEO4J = {"neo4j_uri": "bolt://localhost:7687", "neo4j_user": "u", "neo4j_password": "p"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("FREEZE_BACKEND", raising=False)


# ---------------------------------------------------------------------------
# configure() cascade
# ---------------------------------------------------------------------------


class TestFreezeBackendCascade:
    def test_inherits_storage_backend_none(self):
        cfg = configure(**_NEO4J, storage_backend="none")
        assert cfg.freeze_backend == "none"

    def test_inherits_storage_backend_mongodb(self):
        cfg = configure(**_NEO4J, storage_backend="mongodb")
        assert cfg.freeze_backend == "mongodb"

    def test_inherits_storage_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "mongodb")
        cfg = configure(**_NEO4J)
        assert cfg.freeze_backend == "mongodb"

    def test_env_overrides_inheritance(self, monkeypatch):
        monkeypatch.setenv("FREEZE_BACKEND", "none")
        cfg = configure(**_NEO4J, storage_backend="mongodb")
        assert cfg.freeze_backend == "none"

    def test_argument_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FREEZE_BACKEND", "none")
        cfg = configure(**_NEO4J, storage_backend="none", freeze_backend="mongodb")
        assert cfg.freeze_backend == "mongodb"

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ConfigurationError, match="Unknown freeze_backend"):
            configure(**_NEO4J, freeze_backend="bogus")  # type: ignore[arg-type]

    def test_collection_and_bucket_defaults(self):
        cfg = configure(**_NEO4J)
        assert cfg.mongodb_frozen_collection == "frozen_documents"
        assert cfg.mongodb_frozen_gridfs_bucket == "frozen_snapshots"

    def test_collection_and_bucket_overrides(self):
        cfg = configure(
            **_NEO4J,
            mongodb_frozen_collection="fc",
            mongodb_frozen_gridfs_bucket="fb",
        )
        assert cfg.mongodb_frozen_collection == "fc"
        assert cfg.mongodb_frozen_gridfs_bucket == "fb"


# ---------------------------------------------------------------------------
# get_freeze_storage()
# ---------------------------------------------------------------------------


class _CustomRepo(FreezeRepository):
    async def store_snapshot(self, path, *, tenant_id, created_by_user_id=None, job_id=None, metadata):
        return "x"

    async def read_snapshot_to_file(self, frozen_blob_id, dest_path, *, tenant_id):
        return False

    async def delete_snapshot(self, frozen_blob_id, *, tenant_id):
        return None


class TestGetFreezeStorage:
    def test_none_raises_configuration_error(self):
        configure(**_NEO4J, storage_backend="none")
        with pytest.raises(ConfigurationError, match="freeze backend"):
            get_freeze_storage()

    def test_custom_returns_the_instance(self):
        repo = _CustomRepo()
        configure(**_NEO4J, freeze_backend="custom", custom_freeze_storage=repo)
        assert get_freeze_storage() is repo

    def test_custom_without_instance_raises(self):
        configure(**_NEO4J, freeze_backend="custom")
        with pytest.raises(ConfigurationError, match="custom_freeze_storage"):
            get_freeze_storage()

    def test_mongodb_checks_readiness(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "scinr.newton.storage.factory._ensure_mongodb_ready", lambda cfg: seen.append(cfg)
        )
        cfg = configure(**_NEO4J, freeze_backend="mongodb")
        assert isinstance(get_freeze_storage(), MongoDBFreezeRepository)
        assert seen == [cfg]


# ---------------------------------------------------------------------------
# MongoDBFreezeRepository — in-memory Motor / GridFS stand-ins
# ---------------------------------------------------------------------------


def _cond(value: Any, cond: Any) -> bool:
    if isinstance(cond, dict) and "$in" in cond:
        return value in cond["$in"]
    return value == cond


def _matches(doc: dict, query: dict) -> bool:
    for key, cond in query.items():
        if isinstance(cond, dict) and "$elemMatch" in cond:
            element = cond["$elemMatch"]
            if not any(
                all(_cond(item.get(k), c) for k, c in element.items()) for item in doc.get(key, [])
            ):
                return False
        elif not _cond(doc.get(key), cond):
            return False
    return True


class _Cursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def __aiter__(self):
        async def _gen():
            for d in self._docs:
                yield d

        return _gen()


class _Result:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Collection:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    async def insert_one(self, doc: dict) -> _Result:
        doc = {"_id": ObjectId(), **doc}
        self.docs.append(doc)
        return _Result(inserted_id=doc["_id"])

    async def find_one(self, query: dict) -> dict | None:
        return next((d for d in self.docs if _matches(d, query)), None)

    def find(self, query: dict, projection=None, sort=None) -> _Cursor:
        out = [d for d in self.docs if _matches(d, query)]
        for key, direction in reversed(sort or []):
            out.sort(key=lambda d, k=key: d.get(k), reverse=direction < 0)
        return _Cursor(out)

    async def delete_one(self, query: dict) -> _Result:
        for d in self.docs:
            if _matches(d, query):
                self.docs.remove(d)
                return _Result(deleted_count=1)
        return _Result(deleted_count=0)


class _GridOut:
    def __init__(self, data: bytes, chunk: int = 5) -> None:
        self._data = data
        self._chunk = chunk
        self._pos = 0
        self.closed = False

    async def readchunk(self) -> bytes:
        out = self._data[self._pos : self._pos + self._chunk]
        self._pos += len(out)
        return out

    def close(self) -> None:
        self.closed = True


class _Bucket:
    def __init__(self) -> None:
        self.files: dict[ObjectId, dict] = {}
        self.reads: list[int] = []

    async def upload_from_stream(self, filename, source, metadata=None) -> ObjectId:
        chunks = []
        while chunk := source.read(7):  # GridFS reads in blocks, never whole
            self.reads.append(len(chunk))
            chunks.append(chunk)
        fid = ObjectId()
        self.files[fid] = {"filename": filename, "data": b"".join(chunks), "metadata": metadata}
        return fid

    async def open_download_stream(self, fid) -> _GridOut:
        if fid not in self.files:
            raise NoFile(str(fid))
        return _GridOut(self.files[fid]["data"])

    async def delete(self, fid) -> None:
        if fid not in self.files:
            raise NoFile(str(fid))
        del self.files[fid]


@pytest.fixture
def mongo(monkeypatch):
    collection = _Collection()
    bucket = _Bucket()
    buckets: list[str] = []
    cfg = SimpleNamespace(
        mongodb_frozen_collection="frozen_documents",
        mongodb_frozen_gridfs_bucket="frozen_snapshots",
    )
    monkeypatch.setattr("scinr.newton.config.get_config", lambda: cfg)
    monkeypatch.setattr(
        "scinr.newton.freeze.mongodb.repository.get_db",
        lambda: {"frozen_documents": collection},
    )

    def _bucket(name=None):
        buckets.append(name)
        return bucket

    monkeypatch.setattr("scinr.newton.freeze.mongodb.repository.get_gridfs_bucket", _bucket)
    return SimpleNamespace(collection=collection, bucket=bucket, buckets=buckets)


repo = MongoDBFreezeRepository()
_PAYLOAD = b'{"schema_version": 1, "documents": []}'


async def _store(tmp_path: Path, tenant: str | None, documents=None, mode="freeze", **kw) -> str:
    src = tmp_path / "snap.json"
    src.write_bytes(_PAYLOAD)
    metadata = {"schema_version": 1, "mode": mode, "frozen_at": "t"}
    if documents is not None:
        metadata["documents"] = documents
    return await repo.store_snapshot(src, tenant_id=tenant, metadata=metadata, **kw)


def _meta(path: str, version: int, job: str | None = None, user: str | None = None) -> dict:
    return {"path": path, "version": version, "job_id": job, "created_by_user_id": user}


class TestMongoDBFreezeRepository:
    async def test_store_writes_metadata_and_gridfs(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme", created_by_user_id="u1", job_id=("j1", "j2"))

        [doc] = mongo.collection.docs
        assert blob_id == str(doc["_id"])
        assert doc["tenant_id"] == "acme"
        assert doc["created_by_user_id"] == "u1"
        assert doc["job_id"] == ["j1", "j2"]
        assert doc["mode"] == "freeze" and doc["schema_version"] == 1
        assert doc["size_bytes"] == len(_PAYLOAD)
        assert len(doc["checksum_sha256"]) == 64
        stored = mongo.bucket.files[doc["gridfs_id"]]
        assert stored["data"] == _PAYLOAD
        assert stored["metadata"]["tenant_id"] == "acme"
        assert set(mongo.buckets) == {"frozen_snapshots"}
        assert max(mongo.bucket.reads) <= 7  # streamed

    async def test_public_tenant_is_stored_as_sentinel(self, mongo, tmp_path):
        await _store(tmp_path, None)
        assert mongo.collection.docs[0]["tenant_id"] == "__public__"

    async def test_read_round_trip(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        dest = tmp_path / "out.json"
        assert await repo.read_snapshot_to_file(blob_id, dest, tenant_id="acme") is True
        assert dest.read_bytes() == _PAYLOAD

    async def test_read_is_tenant_scoped(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        dest = tmp_path / "out.json"
        assert await repo.read_snapshot_to_file(blob_id, dest, tenant_id="other") is False
        assert await repo.read_snapshot_to_file(blob_id, dest, tenant_id=None) is False
        assert not dest.exists()

    @pytest.mark.parametrize("blob_id", ["not-an-object-id", str(ObjectId())])
    async def test_read_unknown_id_returns_false(self, mongo, tmp_path, blob_id):
        assert await repo.read_snapshot_to_file(blob_id, tmp_path / "o", tenant_id="acme") is False

    async def test_read_missing_binary_raises_storage_error(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        mongo.bucket.files.clear()
        with pytest.raises(StorageError, match="missing from GridFS"):
            await repo.read_snapshot_to_file(blob_id, tmp_path / "o", tenant_id="acme")

    async def test_delete_removes_both(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        await repo.delete_snapshot(blob_id, tenant_id="acme")
        assert mongo.collection.docs == []
        assert mongo.bucket.files == {}

    async def test_delete_other_tenant_is_a_noop(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        await repo.delete_snapshot(blob_id, tenant_id="other")
        assert len(mongo.collection.docs) == 1
        assert len(mongo.bucket.files) == 1

    async def test_delete_is_idempotent(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        await repo.delete_snapshot(blob_id, tenant_id="acme")
        await repo.delete_snapshot(blob_id, tenant_id="acme")
        await repo.delete_snapshot("garbage", tenant_id="acme")

    async def test_delete_with_binary_already_gone_still_removes_metadata(self, mongo, tmp_path):
        blob_id = await _store(tmp_path, "acme")
        mongo.bucket.files.clear()
        await repo.delete_snapshot(blob_id, tenant_id="acme")
        assert mongo.collection.docs == []

    async def test_find_snapshots_newest_first_and_tenant_scoped(self, mongo, tmp_path):
        old = await _store(tmp_path, "acme", documents=[_meta("a.pdf", 1)], mode="backup")
        new = await _store(tmp_path, "acme", documents=[_meta("a.pdf", 1), _meta("b.pdf", 1)])
        await _store(tmp_path, "other", documents=[_meta("a.pdf", 1)])
        mongo.collection.docs[0]["stored_at"] = mongo.collection.docs[0]["stored_at"].replace(year=2000)

        records = await repo.find_snapshots(tenant_id="acme", path="a.pdf")

        assert [r.frozen_blob_id for r in records] == [new, old]
        assert records[1].mode == "backup" and records[0].mode == "freeze"
        assert records[0].documents[1]["path"] == "b.pdf"

    async def test_find_snapshots_filters_apply_to_one_entry(self, mongo, tmp_path):
        blob = await _store(
            tmp_path, "acme", documents=[_meta("a.pdf", 1, job="j1"), _meta("b.pdf", 2, job="j2")]
        )
        found = lambda records: [r.frozen_blob_id for r in records]  # noqa: E731
        assert found(await repo.find_snapshots(tenant_id="acme", path="a.pdf", version=1)) == [blob]
        assert found(await repo.find_snapshots(tenant_id="acme", path="a.pdf", version=2)) == []
        assert found(await repo.find_snapshots(tenant_id="acme", job_id=["j2", "x"])) == [blob]
        assert found(
            await repo.find_snapshots(tenant_id="acme", path="a.pdf", job_id=["j2"])
        ) == []

    async def test_find_snapshots_public_sentinel(self, mongo, tmp_path):
        blob = await _store(tmp_path, None, documents=[_meta("a.pdf", 1)])
        records = await repo.find_snapshots(tenant_id="__public__", path="a.pdf")
        assert [r.frozen_blob_id for r in records] == [blob]

    async def test_base_repository_has_no_lookup(self):
        from scinr.newton.exceptions import FreezeError

        with pytest.raises(FreezeError, match="frozen_blob_id"):
            await _CustomRepo().find_snapshots(tenant_id="acme", path="a.pdf")
