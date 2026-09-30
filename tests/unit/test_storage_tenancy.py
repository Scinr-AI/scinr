"""
tests/unit/test_storage_tenancy.py — Multi-tenancy of the storage layer.

Exercises the MongoDB repositories against a small in-memory stand-in for
Motor collections / the GridFS bucket (no real MongoDB): what is written
(tenant + provenance on ``raw_files``, GridFS metadata and ``converted_pages``)
and the read / delete scope, which must behave exactly like the navigation
scope (``utils/scope.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from bson.objectid import ObjectId
from gridfs.errors import NoFile

from scinr.newton.exceptions import NavigationError, ScopeError, StorageError
from scinr.newton.storage.filters import mongo_scope_filter
from scinr.newton.storage.mongodb.pages import MongoDBPageRepository
from scinr.newton.storage.mongodb.raw_files import MongoDBRawFileRepository
from scinr.newton.storage.null import NullPageRepository, NullRawFileRepository

# ---------------------------------------------------------------------------
# In-memory Motor / GridFS stand-ins
# ---------------------------------------------------------------------------


def _matches(doc: dict, query: dict) -> bool:
    for key, cond in query.items():
        if isinstance(cond, dict) and "$in" in cond:
            if key not in doc or doc[key] not in cond["$in"]:
                return False
        elif doc.get(key) != cond:
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
    def __init__(self, name: str) -> None:
        self.name = name
        self.docs: list[dict] = []

    async def insert_one(self, doc: dict) -> _Result:
        doc = {"_id": ObjectId(), **doc}
        self.docs.append(doc)
        return _Result(inserted_id=doc["_id"])

    async def find_one(self, query: dict) -> dict | None:
        return next((d for d in self.docs if _matches(d, query)), None)

    def find(self, query: dict, sort: list | None = None) -> _Cursor:
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

    async def delete_many(self, query: dict) -> _Result:
        keep = [d for d in self.docs if not _matches(d, query)]
        n = len(self.docs) - len(keep)
        self.docs[:] = keep
        return _Result(deleted_count=n)


class _GridOut:
    def __init__(self, data: bytes, chunk: int = 4) -> None:
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
        self.opened: list[ObjectId] = []

    async def upload_from_stream(self, filename, source, metadata=None) -> ObjectId:
        data = source if isinstance(source, bytes) else source.read()
        fid = ObjectId()
        self.files[fid] = {"filename": filename, "data": data, "metadata": metadata}
        return fid

    async def open_download_stream(self, fid) -> _GridOut:
        if fid not in self.files:
            raise NoFile(str(fid))
        self.opened.append(fid)
        return _GridOut(self.files[fid]["data"])

    async def delete(self, fid) -> None:
        if fid not in self.files:
            raise NoFile(str(fid))
        del self.files[fid]


class _FakeConfig:
    mongodb_raw_files_collection = "raw_files"
    mongodb_pages_collection = "converted_pages"


@pytest.fixture
def mongo(monkeypatch):
    db = {"raw_files": _Collection("raw_files"), "converted_pages": _Collection("converted_pages")}
    bucket = _Bucket()
    monkeypatch.setattr("scinr.newton.config.get_config", lambda: _FakeConfig())
    for mod in ("raw_files", "pages"):
        monkeypatch.setattr(f"scinr.newton.storage.mongodb.{mod}.get_db", lambda: db)
    monkeypatch.setattr(
        "scinr.newton.storage.mongodb.raw_files.get_gridfs_bucket", lambda: bucket
    )
    return db, bucket


raw_repo = MongoDBRawFileRepository()
page_repo = MongoDBPageRepository()


async def _upload(tenant: str | None, *, user: str | None = None, job: str | None = None,
                  name: str = "f.pdf", folder: str | None = None) -> str:
    rid = await raw_repo.store(
        name, f"bytes of {tenant}".encode(), "application/pdf", folder,
        tenant_id=tenant, created_by_user_id=user, job_id=job,
    )
    await page_repo.store_page(
        rid, name.rsplit(".", 1)[0], folder, 0, f"page of {tenant}",
        tenant_id=tenant, created_by_user_id=user, job_id=job,
    )
    return rid


@pytest.fixture
async def seeded(mongo):
    """acme, globex and public uploads + one legacy record without tenant."""
    db, bucket = mongo
    ids = {
        "acme": await _upload("acme", user="u1", job="j1", folder="a"),
        "globex": await _upload("globex", user="u2", job="j2"),
        "public": await _upload(None, job="j3"),
    }
    # Legacy (pre-multitenancy) record: no tenant / provenance fields at all.
    fid = await bucket.upload_from_stream("old.pdf", b"legacy", metadata={})
    legacy = await db["raw_files"].insert_one({
        "filename": "old.pdf", "folder_path": None, "content_type": "application/pdf",
        "size_bytes": 6, "checksum_sha256": "x", "stored_at": datetime.now(UTC),
        "gridfs_id": fid,
    })
    ids["legacy"] = str(legacy.inserted_id)
    await db["converted_pages"].insert_one({
        "raw_file_id": ids["legacy"], "filename": "old", "folder_path": None,
        "page_index": 0, "markdown": "legacy page", "converted_at": datetime.now(UTC),
    })
    return ids


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


class TestWrites:
    @pytest.mark.parametrize("tenant", [None, "__public__"])
    async def test_public_is_stored_as_the_reserved_key(self, mongo, tenant):
        db, bucket = mongo
        rid = await _upload(tenant)
        (raw,) = db["raw_files"].docs
        (page,) = db["converted_pages"].docs
        (gridfs_file,) = bucket.files.values()
        assert raw["tenant_id"] == page["tenant_id"] == "__public__"
        assert gridfs_file["metadata"]["tenant_id"] == "__public__"
        assert page["raw_file_id"] == rid

    async def test_tenant_and_provenance_on_every_record(self, mongo):
        db, bucket = mongo
        await _upload("acme", user="u1", job="j1")
        (raw,) = db["raw_files"].docs
        (page,) = db["converted_pages"].docs
        (gridfs_file,) = bucket.files.values()
        for rec in (raw, page, gridfs_file["metadata"]):
            assert (rec["tenant_id"], rec["created_by_user_id"], rec["job_id"]) == (
                "acme", "u1", "j1",
            )

    async def test_store_file_streams_with_the_owner(self, mongo, tmp_path: Path):
        db, bucket = mongo
        path = tmp_path / "x.bin"
        path.write_bytes(b"abc")
        rid = await raw_repo.store_file(path, "x.bin", "application/x", None, tenant_id="acme")
        assert db["raw_files"].docs[0]["tenant_id"] == "acme"
        record = await raw_repo.get(rid, tenant_id="acme")
        assert record is not None and record.size_bytes == 3 and record.tenant_id == "acme"

    async def test_empty_tenant_on_write_is_rejected(self, mongo):
        with pytest.raises(ValueError):
            await raw_repo.store("f", b"x", "text/plain", None, tenant_id="")
        with pytest.raises(ValueError):
            await page_repo.store_page("rid", "f", None, 0, "md", tenant_id="")


# ---------------------------------------------------------------------------
# Read scope — same table as the navigation (utils/scope.py)
# ---------------------------------------------------------------------------

_SCOPE_TABLE = [
    # (scope kwargs, visible uploads)
    ({}, {"acme", "globex", "public", "legacy"}),
    ({"tenant_id": "__public__"}, {"public"}),
    ({"tenant_id": "__public__", "include_public": True}, {"public"}),
    ({"tenant_id": "acme"}, {"acme"}),
    ({"tenant_id": "acme", "include_public": True}, {"acme", "public"}),
    ({"created_by_user_id": "u1"}, {"acme"}),
    ({"created_by_user_id": ["u1", "u2"]}, {"acme", "globex"}),
    ({"job_id": ["j2", "j3"]}, {"globex", "public"}),
    ({"tenant_id": "acme", "include_public": True, "job_id": "j3"}, {"public"}),
    ({"tenant_id": "acme", "job_id": "j2"}, set()),
]


class TestReadScope:
    @pytest.mark.parametrize(("scope", "visible"), _SCOPE_TABLE)
    async def test_get(self, seeded, scope, visible):
        got = {name for name, rid in seeded.items() if await raw_repo.get(rid, **scope)}
        assert got == visible

    @pytest.mark.parametrize(("scope", "visible"), _SCOPE_TABLE)
    async def test_get_pages(self, seeded, scope, visible):
        got = {name for name, rid in seeded.items() if await page_repo.get_pages(rid, **scope)}
        assert got == visible

    @pytest.mark.parametrize(("scope", "visible"), _SCOPE_TABLE)
    async def test_list_raw_files(self, seeded, scope, visible):
        by_id = {rid: name for name, rid in seeded.items()}
        got = {by_id[r.id] for r in await raw_repo.list_raw_files(**scope)}
        assert got == visible

    @pytest.mark.parametrize(("scope", "visible"), _SCOPE_TABLE)
    async def test_open(self, seeded, scope, visible):
        got = set()
        for name, rid in seeded.items():
            stream = await raw_repo.open(rid, **scope)
            if stream is not None:
                got.add(name)
                assert b"".join([c async for c in stream])
        assert got == visible

    @pytest.mark.parametrize(("scope", "visible"), _SCOPE_TABLE)
    async def test_get_pages_by_ids(self, seeded, scope, visible):
        by_page = {p.id: name for name, rid in seeded.items() for p in await page_repo.get_pages(rid)}
        got = {by_page[p.id] for p in await page_repo.get_pages_by_ids(list(by_page), **scope)}
        assert got == visible

    @pytest.mark.parametrize(("scope", "visible"), _SCOPE_TABLE)
    async def test_open_with_record(self, seeded, scope, visible):
        got = set()
        for name, rid in seeded.items():
            opened = await raw_repo.open_with_record(rid, **scope)
            if opened is not None:
                got.add(name)
                record, stream = opened
                assert record.id == rid
                assert b"".join([c async for c in stream])
        assert got == visible

    async def test_open_streams_the_binary_of_the_filtered_record(self, seeded, mongo):
        _db, bucket = mongo
        stream = await raw_repo.open(seeded["acme"], tenant_id="acme")
        assert b"".join([c async for c in stream]) == b"bytes of acme"

    async def test_open_with_missing_binary_raises_storage_error(self, seeded, mongo):
        _db, bucket = mongo
        bucket.files.clear()
        with pytest.raises(StorageError):
            await raw_repo.open(seeded["acme"], tenant_id="acme")
        with pytest.raises(StorageError):
            await raw_repo.open_with_record(seeded["acme"], tenant_id="acme")

    async def test_open_with_record_reads_the_record_once(self, seeded, mongo, monkeypatch):
        db, _bucket = mongo
        calls = []
        find_one = db["raw_files"].find_one

        async def _counting(query):
            calls.append(query)
            return await find_one(query)

        monkeypatch.setattr(db["raw_files"], "find_one", _counting)
        record, stream = await raw_repo.open_with_record(seeded["acme"], tenant_id="acme")
        assert (record.filename, record.tenant_id) == ("f.pdf", "acme")
        assert b"".join([c async for c in stream]) == b"bytes of acme"
        assert len(calls) == 1

    async def test_get_pages_by_ids_reads_only_the_requested_pages(self, mongo):
        rid = await _upload("acme")
        for idx in (2, 1):
            await page_repo.store_page(rid, "f", None, idx, f"page {idx}", tenant_id="acme")
        pages = {p.page_index: p.id for p in await page_repo.get_pages(rid)}

        got = await page_repo.get_pages_by_ids([pages[2], pages[0], pages[2]], tenant_id="acme")
        # ordered by page_index, duplicates collapsed, page 1 not loaded
        assert [p.page_index for p in got] == [0, 2]

    async def test_get_pages_by_ids_skips_invalid_and_unknown_ids(self, seeded):
        (page,) = await page_repo.get_pages(seeded["acme"])
        got = await page_repo.get_pages_by_ids(["not-an-oid", str(ObjectId()), page.id])
        assert [p.id for p in got] == [page.id]
        assert await page_repo.get_pages_by_ids([]) == []
        assert await page_repo.get_pages_by_ids(["not-an-oid"]) == []

    async def test_record_exposes_tenant_and_provenance(self, seeded):
        record = await raw_repo.get(seeded["acme"])
        assert (record.tenant_id, record.created_by_user_id, record.job_id) == ("acme", "u1", "j1")
        (page,) = await page_repo.get_pages(seeded["acme"])
        assert (page.tenant_id, page.created_by_user_id, page.job_id) == ("acme", "u1", "j1")

    async def test_legacy_record_has_no_tenant(self, seeded):
        record = await raw_repo.get(seeded["legacy"])
        assert record is not None and record.tenant_id is None

    async def test_list_filters_by_folder_and_filename(self, seeded):
        (rec,) = await raw_repo.list_raw_files(tenant_id="acme", folder_path="a")
        assert rec.id == seeded["acme"]
        assert await raw_repo.list_raw_files(tenant_id="acme", folder_path="b") == []
        assert len(await raw_repo.list_raw_files(filename="f.pdf")) == 3

    async def test_invalid_object_id_is_not_found(self, seeded):
        assert await raw_repo.get("not-an-oid") is None
        assert await raw_repo.open("not-an-oid") is None


# ---------------------------------------------------------------------------
# Delete scope
# ---------------------------------------------------------------------------


class TestDeleteScope:
    async def test_wrong_tenant_deletes_nothing(self, seeded, mongo):
        db, bucket = mongo
        before = (len(db["raw_files"].docs), len(db["converted_pages"].docs), len(bucket.files))
        await raw_repo.delete(seeded["acme"], tenant_id="globex")
        assert await page_repo.delete_pages(seeded["acme"], tenant_id="globex") == 0
        # include_public does not reach another tenant either
        await raw_repo.delete(seeded["acme"], tenant_id="__public__")
        assert before == (len(db["raw_files"].docs), len(db["converted_pages"].docs),
                          len(bucket.files))
        assert await raw_repo.get(seeded["acme"]) is not None

    async def test_right_tenant_deletes_binary_metadata_and_pages(self, seeded, mongo):
        _db, bucket = mongo
        n_files = len(bucket.files)
        assert await page_repo.delete_pages(seeded["acme"], tenant_id="acme") == 1
        await raw_repo.delete(seeded["acme"], tenant_id="acme")
        assert await raw_repo.get(seeded["acme"]) is None
        assert await page_repo.get_pages(seeded["acme"]) == []
        assert len(bucket.files) == n_files - 1

    async def test_provenance_filters_apply_to_delete(self, seeded):
        await raw_repo.delete(seeded["acme"], tenant_id="acme", job_id="j2")
        assert await raw_repo.get(seeded["acme"]) is not None
        await raw_repo.delete(seeded["acme"], tenant_id="acme", job_id=["j1", "j2"])
        assert await raw_repo.get(seeded["acme"]) is None

    async def test_legacy_record_only_deleted_without_tenant_filter(self, seeded):
        await raw_repo.delete(seeded["legacy"], tenant_id="__public__")
        assert await page_repo.delete_pages(seeded["legacy"], tenant_id="__public__") == 0
        assert await raw_repo.get(seeded["legacy"]) is not None
        assert await page_repo.delete_pages(seeded["legacy"]) == 1
        await raw_repo.delete(seeded["legacy"])
        assert await raw_repo.get(seeded["legacy"]) is None


# ---------------------------------------------------------------------------
# Invalid scopes — same errors as the navigation
# ---------------------------------------------------------------------------

_BAD_SCOPES = [
    {"tenant_id": ""},
    {"created_by_user_id": []},
    {"job_id": []},
    {"job_id": ["j1", ""]},
]


class TestInvalidScope:
    @pytest.mark.parametrize("scope", _BAD_SCOPES)
    async def test_mongo_repositories_reject(self, mongo, scope):
        for call in (
            lambda: raw_repo.get("x", **scope),
            lambda: raw_repo.open("x", **scope),
            lambda: raw_repo.list_raw_files(**scope),
            lambda: raw_repo.delete("x", **scope),
            lambda: page_repo.get_pages("x", **scope),
            lambda: page_repo.get_pages_by_ids(["x"], **scope),
            lambda: raw_repo.open_with_record("x", **scope),
            lambda: page_repo.delete_pages("x", **scope),
        ):
            with pytest.raises(ScopeError):
                await call()

    @pytest.mark.parametrize("scope", _BAD_SCOPES)
    async def test_null_repositories_reject(self, scope):
        with pytest.raises(ScopeError):
            await NullRawFileRepository().get("x", **scope)
        with pytest.raises(ScopeError):
            await NullPageRepository().get_pages("x", **scope)
        with pytest.raises(ScopeError):
            await NullPageRepository().get_pages_by_ids(["x"], **scope)

    def test_scope_error_is_both_a_storage_and_a_navigation_error(self):
        with pytest.raises(StorageError):
            mongo_scope_filter(tenant_id="")
        with pytest.raises(NavigationError):
            mongo_scope_filter(tenant_id="")


class TestScopeToMongo:
    def test_unfiltered_is_empty(self):
        assert mongo_scope_filter() == {}

    def test_full_render(self):
        assert mongo_scope_filter(
            tenant_id="acme", include_public=True, created_by_user_id="u1", job_id=["j1", "j2"]
        ) == {
            "tenant_id": {"$in": ["acme", "__public__"]},
            "created_by_user_id": {"$in": ["u1"]},
            "job_id": {"$in": ["j1", "j2"]},
        }


class TestNullRepositories:
    async def test_reads_find_nothing_and_open_raises(self):
        assert await NullRawFileRepository().get("x", tenant_id="acme") is None
        assert await NullRawFileRepository().list_raw_files(tenant_id="acme") == []
        assert await NullPageRepository().get_pages_by_ids(["x"], tenant_id="acme") == []
        with pytest.raises(StorageError):
            await NullRawFileRepository().open("x", tenant_id="acme")
        with pytest.raises(StorageError):
            await NullRawFileRepository().open_with_record("x", tenant_id="acme")
