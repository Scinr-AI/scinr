"""
tests/storage_contract/test_storage_contract.py -- The storage and freeze contract.

One test per case of the plan (files, scope, deletion, pages, images, snapshots, memory); each
runs against every available backend (see ``conftest.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scinr.newton.exceptions import ScopeError
from scinr.newton.storage.models import ConvertedPageRecord, PageImageRecord, RawFileRecord

MIB = 1024 * 1024
ID_SHAPE = re.compile(r"^[0-9a-f]{24}$")
PDF = "application/pdf"


async def drain(stream) -> bytes:
    parts = []
    async for block in stream:
        parts.append(block)
    return b"".join(parts)


def write_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def as_utc(value: datetime, backend: str) -> datetime:
    """Return ``value`` as an aware datetime for the comparison.

    The MongoDB client is not ``tz_aware``, so it returns naive datetimes that are UTC by
    construction (BSON dates are UTC). The owner decided not to change the MongoDB backend:
    only for that backend a naive value is read as UTC. The SQL backends stay strict.
    """
    if backend == "mongodb" and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class TestFiles:
    async def test_store_and_get(self, stores) -> None:
        file_id = await stores.raw.store(
            "a.pdf",
            b"hello",
            PDF,
            "mod/sub",
            tenant_id="acme",
            created_by_user_id="u1",
            job_id="j1",
        )
        assert ID_SHAPE.match(file_id)
        record = await stores.raw.get(file_id)
        assert isinstance(record, RawFileRecord)
        assert record.id == file_id
        assert (record.filename, record.folder_path, record.content_type) == (
            "a.pdf",
            "mod/sub",
            PDF,
        )
        assert record.size_bytes == 5
        assert record.checksum_sha256 == hashlib.sha256(b"hello").hexdigest()
        assert (record.tenant_id, record.created_by_user_id, record.job_id) == ("acme", "u1", "j1")
        assert as_utc(record.stored_at, stores.backend).tzinfo is not None

    async def test_public_upload_is_stored_as_public_tenant(self, stores) -> None:
        file_id = await stores.raw.store("p.pdf", b"x", PDF, None)
        record = await stores.raw.get(file_id)
        assert record.tenant_id == "__public__"
        assert record.folder_path is None
        explicit = await stores.raw.store("q.pdf", b"x", PDF, None, tenant_id="__public__")
        assert (await stores.raw.get(explicit)).tenant_id == "__public__"

    async def test_store_file_streams_from_disk(self, stores) -> None:
        content = b"abc" * 1000
        path = write_file(stores.tmp_path / "src.bin", content)
        file_id = await stores.raw.store_file(path, "src.bin", "application/octet-stream", None)
        record = await stores.raw.get(file_id)
        assert record.size_bytes == len(content)
        assert record.checksum_sha256 == hashlib.sha256(content).hexdigest()
        assert await drain(await stores.raw.open(file_id)) == content

    async def test_open_returns_the_same_bytes(self, stores) -> None:
        content = bytes(range(256)) * 100
        file_id = await stores.raw.store("b.bin", content, "application/octet-stream", None)
        assert await drain(await stores.raw.open(file_id)) == content

    async def test_open_with_record(self, stores) -> None:
        file_id = await stores.raw.store("c.txt", b"data", "text/plain", "f", tenant_id="acme")
        record, stream = await stores.raw.open_with_record(file_id, tenant_id="acme")
        assert record.id == file_id
        assert await drain(stream) == b"data"
        assert await stores.raw.open_with_record(file_id, tenant_id="other") is None

    async def test_empty_file(self, stores) -> None:
        file_id = await stores.raw.store("empty.txt", b"", "text/plain", None)
        record = await stores.raw.get(file_id)
        assert record.size_bytes == 0
        assert record.checksum_sha256 == hashlib.sha256(b"").hexdigest()
        assert await drain(await stores.raw.open(file_id)) == b""

    async def test_file_of_three_chunks(self, stores) -> None:
        content = (b"0123456789abcdef" * (MIB // 16)) * 2 + b"x" * 1000  # 2 MiB + 1000 bytes
        path = write_file(stores.tmp_path / "big.bin", content)
        file_id = await stores.raw.store_file(path, "big.bin", "application/octet-stream", None)
        assert (await stores.raw.get(file_id)).size_bytes == len(content)
        assert await drain(await stores.raw.open(file_id)) == content

    async def test_list_is_oldest_first_and_filters_by_folder_and_name(self, stores) -> None:
        first = await stores.raw.store("a.pdf", b"1", PDF, "f1", tenant_id="acme")
        await asyncio.sleep(0.01)
        second = await stores.raw.store("b.pdf", b"2", PDF, "f2", tenant_id="acme")
        await asyncio.sleep(0.01)
        third = await stores.raw.store("a.pdf", b"3", PDF, "f2", tenant_id="acme")
        listed = await stores.raw.list_raw_files(tenant_id="acme")
        assert [r.id for r in listed] == [first, second, third]
        assert [
            r.id for r in await stores.raw.list_raw_files(tenant_id="acme", folder_path="f2")
        ] == [
            second,
            third,
        ]
        assert [
            r.id for r in await stores.raw.list_raw_files(tenant_id="acme", filename="a.pdf")
        ] == [
            first,
            third,
        ]
        assert (
            await stores.raw.list_raw_files(tenant_id="acme", filename="a.pdf", folder_path="f1")
            != []
        )

    async def test_unknown_and_invalid_ids(self, stores) -> None:
        assert await stores.raw.get("0" * 24) is None
        assert await stores.raw.get("not-an-id") is None
        assert await stores.raw.open("0" * 24) is None
        assert await stores.raw.open_with_record("not-an-id") is None


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class TestScope:
    async def test_other_tenant_sees_nothing(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"secret", PDF, None, tenant_id="acme")
        assert await stores.raw.get(file_id, tenant_id="other") is None
        assert await stores.raw.open(file_id, tenant_id="other") is None
        assert await stores.raw.list_raw_files(tenant_id="other") == []
        await stores.raw.delete(file_id, tenant_id="other")
        assert await stores.raw.get(file_id, tenant_id="acme") is not None
        assert await drain(await stores.raw.open(file_id, tenant_id="acme")) == b"secret"

    async def test_tenant_none_reads_every_tenant(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        assert (await stores.raw.get(file_id)).id == file_id
        assert [r.id for r in await stores.raw.list_raw_files()] == [file_id]

    async def test_include_public(self, stores) -> None:
        public = await stores.raw.store("pub.pdf", b"1", PDF, None)
        assert await stores.raw.get(public, tenant_id="acme") is None
        assert (await stores.raw.get(public, tenant_id="acme", include_public=True)).id == public
        assert (await stores.raw.get(public, tenant_id="__public__")).id == public
        mine = await stores.raw.store("mine.pdf", b"1", PDF, None, tenant_id="acme")
        listed = await stores.raw.list_raw_files(tenant_id="acme", include_public=True)
        assert {r.id for r in listed} == {public, mine}
        assert [r.id for r in await stores.raw.list_raw_files(tenant_id="acme")] == [mine]

    async def test_user_and_job_filters(self, stores) -> None:
        one = await stores.raw.store(
            "1.pdf", b"1", PDF, None, tenant_id="t", created_by_user_id="u1", job_id="j1"
        )
        two = await stores.raw.store(
            "2.pdf", b"2", PDF, None, tenant_id="t", created_by_user_id="u2", job_id="j2"
        )
        assert (await stores.raw.get(one, created_by_user_id="u1")).id == one
        assert await stores.raw.get(one, created_by_user_id="u2") is None
        assert await stores.raw.get(one, job_id="j2") is None
        assert (
            await stores.raw.get(two, created_by_user_id=["u1", "u2"], job_id=["j2", "j9"])
        ).id == two
        assert [r.id for r in await stores.raw.list_raw_files(job_id="j1")] == [one]
        assert [r.id for r in await stores.raw.list_raw_files(created_by_user_id=["u1", "u2"])] == [
            one,
            two,
        ]
        await stores.raw.delete(one, job_id="j2")
        assert await stores.raw.get(one) is not None

    async def test_invalid_scope_raises_scope_error(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None)
        for call in (
            stores.raw.get(file_id, tenant_id=""),
            stores.raw.open(file_id, tenant_id=""),
            stores.raw.open_with_record(file_id, tenant_id=""),
            stores.raw.list_raw_files(tenant_id=""),
            stores.raw.delete(file_id, tenant_id=""),
            stores.raw.get(file_id, created_by_user_id=[]),
            stores.raw.list_raw_files(job_id=[]),
            stores.pages.get_pages(file_id, tenant_id=""),
            stores.pages.get_pages_by_ids([file_id], job_id=[]),
            stores.pages.delete_pages(file_id, tenant_id=""),
            stores.raw.list_images(file_id, tenant_id=""),
            stores.raw.open_image(file_id, tenant_id=""),
            stores.raw.delete_images(file_id, created_by_user_id=[]),
        ):
            with pytest.raises(ScopeError):
                await call


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_twice_does_not_raise(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None)
        await stores.raw.delete(file_id)
        assert await stores.raw.get(file_id) is None
        assert await stores.raw.open(file_id) is None
        await stores.raw.delete(file_id)
        await stores.raw.delete("not-an-id")
        await stores.raw.delete("0" * 24)

    async def test_delete_does_not_cascade_to_pages_or_images(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        await stores.pages.store_page(file_id, "a.pdf", None, 0, "# one", tenant_id="acme")
        await stores.raw.store_image(
            file_id, 0, 0, b"png", kind="embedded", content_type="image/png",
            description="d", filename="a.pdf", folder_path=None, tenant_id="acme",
        )  # fmt: skip
        await stores.raw.delete(file_id, tenant_id="acme")
        assert len(await stores.pages.get_pages(file_id, tenant_id="acme")) == 1
        assert len(await stores.raw.list_images(file_id, tenant_id="acme")) == 1


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class TestPages:
    async def test_store_and_get_in_page_order(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, "f", tenant_id="acme")
        ids = {}
        for index in (2, 0, 1):
            ids[index] = await stores.pages.store_page(
                file_id, "a.pdf", "f", index, f"page {index}", tenant_id="acme",
                created_by_user_id="u1", job_id="j1",
            )  # fmt: skip
        assert all(ID_SHAPE.match(i) for i in ids.values())
        pages = await stores.pages.get_pages(file_id, tenant_id="acme")
        assert [p.page_index for p in pages] == [0, 1, 2]
        first = pages[0]
        assert isinstance(first, ConvertedPageRecord)
        assert (first.id, first.raw_file_id, first.filename, first.folder_path) == (
            ids[0],
            file_id,
            "a.pdf",
            "f",
        )
        assert (first.markdown, first.tenant_id, first.created_by_user_id, first.job_id) == (
            "page 0",
            "acme",
            "u1",
            "j1",
        )
        assert as_utc(first.converted_at, stores.backend).tzinfo is not None
        assert (first.start_ms, first.end_ms, first.transcript) == (None, None, None)
        assert await stores.pages.get_pages(file_id, tenant_id="other") == []
        assert await stores.pages.get_pages("0" * 24) == []

    async def test_get_pages_by_ids(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        p1 = await stores.pages.store_page(file_id, "a.pdf", None, 1, "one", tenant_id="acme")
        p0 = await stores.pages.store_page(file_id, "a.pdf", None, 0, "zero", tenant_id="acme")
        found = await stores.pages.get_pages_by_ids(
            [p1, "0" * 24, "not-an-id", p0, p1], tenant_id="acme"
        )
        assert [p.id for p in found] == [p0, p1]
        assert await stores.pages.get_pages_by_ids([]) == []
        assert await stores.pages.get_pages_by_ids([p0], tenant_id="other") == []
        assert await stores.pages.get_pages_by_ids(["not-an-id"]) == []

    async def test_time_range_and_transcript_round_trip(self, stores) -> None:
        file_id = await stores.raw.store("v.mp4", b"1", "video/mp4", None)
        cues = [
            {"start_ms": 0, "end_ms": 1500, "text": "hola"},
            {"start_ms": 1500, "end_ms": 3000, "text": "mundo"},
        ]
        page_id = await stores.pages.store_page(
            file_id, "v.mp4", None, 0, "hola\nmundo", start_ms=0, end_ms=3000, transcript=cues
        )
        (page,) = await stores.pages.get_pages_by_ids([page_id])
        assert (page.start_ms, page.end_ms) == (0, 3000)
        assert [c.model_dump() for c in page.transcript] == cues

    async def test_delete_pages(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        for index in range(3):
            await stores.pages.store_page(file_id, "a.pdf", None, index, "x", tenant_id="acme")
        assert await stores.pages.delete_pages(file_id, tenant_id="other") == 0
        assert len(await stores.pages.get_pages(file_id, tenant_id="acme")) == 3
        assert await stores.pages.delete_pages(file_id, tenant_id="acme") == 3
        assert await stores.pages.delete_pages(file_id, tenant_id="acme") == 0
        assert await stores.pages.get_pages(file_id) == []

    async def test_duplicate_page_index_is_rejected_in_sql(self, stores) -> None:
        # The functional spec (sql-graph-backends.md §6) makes (raw_file_id, page_index)
        # unique in SQL. MongoDB has no such index and accepts the duplicate: documented
        # difference, so this case only runs on the SQL backends.
        if stores.backend == "mongodb":
            pytest.skip("MongoDB has no unique index on (raw_file_id, page_index)")
        from scinr.newton.exceptions import StorageError

        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        first = await stores.pages.store_page(file_id, "a.pdf", None, 0, "one", tenant_id="acme")
        with pytest.raises(StorageError):
            await stores.pages.store_page(file_id, "a.pdf", None, 0, "two", tenant_id="acme")
        (page,) = await stores.pages.get_pages(file_id)
        assert (page.id, page.markdown) == (first, "one")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


async def put_image(stores, file_id, page, index, content=b"img", **kwargs):
    options = {
        "kind": "embedded", "content_type": "image/png", "description": f"image {page}.{index}",
        "filename": "a.pdf", "folder_path": "f", "tenant_id": "acme",
    }  # fmt: skip
    options.update(kwargs)
    return await stores.raw.store_image(file_id, page, index, content, **options)


class TestImages:
    async def test_stores_images(self, stores) -> None:
        assert stores.raw.stores_images is True

    async def test_store_list_open_delete(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, "f", tenant_id="acme")
        b = await put_image(stores, file_id, 1, 0, b"second", width=10, height=20)
        a = await put_image(stores, file_id, 0, 1, b"first-b")
        z = await put_image(stores, file_id, 0, 0, b"first-a", created_by_user_id="u1", job_id="j1")
        listed = await stores.raw.list_images(file_id, tenant_id="acme")
        assert [(i.page_index, i.image_index) for i in listed] == [(0, 0), (0, 1), (1, 0)]
        assert [i.id for i in listed] == [z, a, b]
        image = listed[2]
        assert isinstance(image, PageImageRecord)
        assert (image.width, image.height, image.time_ms) == (10, 20, None)
        assert (image.kind, image.content_type, image.description) == (
            "embedded",
            "image/png",
            "image 1.0",
        )
        assert (image.size_bytes, image.checksum_sha256) == (
            6,
            hashlib.sha256(b"second").hexdigest(),
        )
        assert (image.filename, image.folder_path, image.tenant_id) == ("a.pdf", "f", "acme")
        record, stream = await stores.raw.open_image(b, tenant_id="acme")
        assert record.id == b
        assert await drain(stream) == b"second"
        assert await stores.raw.delete_images(file_id, tenant_id="other") == 0
        assert await stores.raw.delete_images(file_id, tenant_id="acme") == 3
        assert await stores.raw.delete_images(file_id, tenant_id="acme") == 0
        assert await stores.raw.list_images(file_id) == []
        assert await stores.raw.open_image(b) is None

    async def test_frame_with_time(self, stores, tmp_path) -> None:
        file_id = await stores.raw.store("v.mp4", b"1", "video/mp4", None, tenant_id="acme")
        frame = write_file(tmp_path / "frame.jpg", b"\xff\xd8jpeg")
        image_id = await stores.raw.store_image_file(
            frame, file_id, 2, 0, kind="frame", content_type="image/jpeg", description="a frame",
            filename="v.mp4", folder_path=None, width=640, height=360, time_ms=12_500, tenant_id="acme",
        )  # fmt: skip
        (image,) = await stores.raw.list_images(file_id, kind="frame")
        assert image.id == image_id
        assert (image.kind, image.time_ms, image.width, image.height) == ("frame", 12_500, 640, 360)
        _, stream = await stores.raw.open_image(image_id)
        assert await drain(stream) == b"\xff\xd8jpeg"

    async def test_filters_and_scope(self, stores) -> None:
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        await put_image(stores, file_id, 0, 0)
        await put_image(stores, file_id, 1, 0, kind="frame", time_ms=5)
        await put_image(stores, file_id, 2, 0)
        assert [
            i.page_index for i in await stores.raw.list_images(file_id, page_indexes=[0, 2])
        ] == [0, 2]
        assert [i.page_index for i in await stores.raw.list_images(file_id, kind="frame")] == [1]
        assert await stores.raw.list_images(file_id, page_indexes=[]) == []
        assert await stores.raw.list_images(file_id, tenant_id="other") == []
        assert await stores.raw.list_images("0" * 24) == []
        image_id = (await stores.raw.list_images(file_id))[0].id
        assert await stores.raw.open_image(image_id, tenant_id="other") is None
        assert await stores.raw.open_image("not-an-id") is None

    async def test_duplicate_image_identity_fails_and_keeps_the_first(self, stores) -> None:
        # MongoDB: unique index (tenant, raw_file_id, page_index, image_index); a duplicate is
        # a bug and must surface as a write error, never as a silent overwrite.
        file_id = await stores.raw.store("a.pdf", b"1", PDF, None, tenant_id="acme")
        first = await put_image(stores, file_id, 0, 0, b"first")
        with pytest.raises(Exception):  # noqa: B017 — DuplicateKeyError / StorageError
            await put_image(stores, file_id, 0, 0, b"second")
        (image,) = await stores.raw.list_images(file_id)
        assert image.id == first
        _, stream = await stores.raw.open_image(first)
        assert await drain(stream) == b"first"
        # Another tenant may use the same (raw_file_id, page, image) identity.
        await put_image(stores, file_id, 0, 0, b"other", tenant_id="other")


# ---------------------------------------------------------------------------
# Snapshots (freeze)
# ---------------------------------------------------------------------------


def snapshot_metadata(*documents: dict, mode: str = "freeze") -> dict:
    return {
        "schema_version": 1,
        "frozen_at": "2026-11-19T10:00:00+00:00",
        "keep_flags": {"structure_nodes": False},
        "mode": mode,
        "path": None,
        "version": None,
        "documents": list(documents),
    }


def doc(path: str, version: int = 1, job_id: str | None = "j1", user: str | None = "u1") -> dict:
    return {"path": path, "version": version, "job_id": job_id, "created_by_user_id": user}


class TestSnapshots:
    async def test_store_and_read_back_identical(self, stores) -> None:
        content = json.dumps({"nodes": list(range(5000))}).encode()
        source = write_file(stores.tmp_path / "snap.json", content)
        blob_id = await stores.freeze.store_snapshot(
            source, tenant_id="acme", created_by_user_id="u1", job_id=["j1"],
            metadata=snapshot_metadata(doc("a/b")),
        )  # fmt: skip
        assert ID_SHAPE.match(blob_id)
        dest = stores.tmp_path / "out.json"
        assert await stores.freeze.read_snapshot_to_file(blob_id, dest, tenant_id="acme") is True
        assert dest.read_bytes() == content

    async def test_read_unknown_or_other_tenant_writes_nothing(self, stores) -> None:
        source = write_file(stores.tmp_path / "snap.json", b"{}")
        blob_id = await stores.freeze.store_snapshot(
            source, tenant_id="acme", metadata=snapshot_metadata()
        )
        for target, tenant in ((blob_id, "other"), ("0" * 24, "acme"), ("not-an-id", "acme")):
            dest = stores.tmp_path / f"{tenant}-{target[:4]}.json"
            assert (
                await stores.freeze.read_snapshot_to_file(target, dest, tenant_id=tenant) is False
            )
            assert not dest.exists()

    async def test_public_snapshot_uses_the_public_tenant(self, stores) -> None:
        source = write_file(stores.tmp_path / "snap.json", b"{}")
        blob_id = await stores.freeze.store_snapshot(
            source, tenant_id=None, metadata=snapshot_metadata(doc("p"))
        )
        dest = stores.tmp_path / "out.json"
        assert (
            await stores.freeze.read_snapshot_to_file(blob_id, dest, tenant_id="__public__") is True
        )
        found = await stores.freeze.find_snapshots(tenant_id="__public__", path="p")
        assert [s.frozen_blob_id for s in found] == [blob_id]

    async def test_find_snapshots_filters_and_order(self, stores) -> None:
        source = write_file(stores.tmp_path / "snap.json", b"{}")

        async def store(*documents, mode="freeze", tenant="acme"):
            blob = await stores.freeze.store_snapshot(
                source, tenant_id=tenant, metadata=snapshot_metadata(*documents, mode=mode)
            )
            await asyncio.sleep(0.01)
            return blob

        first = await store(doc("a/b", 1, "j1", "u1"), doc("a/c", 1, "j2", "u2"))
        second = await store(doc("a/b", 2, "j3", "u1"), mode="backup")
        third = await store(doc("x/y", 1, "j1", "u1"), tenant="other")

        everything = await stores.freeze.find_snapshots(tenant_id="acme")
        assert [s.frozen_blob_id for s in everything] == [second, first]  # newest first
        assert (everything[0].mode, everything[1].mode) == ("backup", "freeze")
        assert everything[1].frozen_at == "2026-11-19T10:00:00+00:00"
        assert everything[1].documents == [doc("a/b", 1, "j1", "u1"), doc("a/c", 1, "j2", "u2")]

        def ids(found):
            return [s.frozen_blob_id for s in found]

        assert ids(await stores.freeze.find_snapshots(tenant_id="acme", path="a/b")) == [
            second,
            first,
        ]
        assert ids(await stores.freeze.find_snapshots(tenant_id="acme", path="a/b", version=2)) == [
            second
        ]
        assert ids(await stores.freeze.find_snapshots(tenant_id="acme", job_id=["j2", "j9"])) == [
            first
        ]
        assert ids(
            await stores.freeze.find_snapshots(tenant_id="acme", created_by_user_id=["u2"])
        ) == [first]
        # every filter must hold for the SAME document entry
        assert (
            ids(await stores.freeze.find_snapshots(tenant_id="acme", path="a/b", job_id=["j2"]))
            == []
        )
        assert ids(await stores.freeze.find_snapshots(tenant_id="acme", path="nope")) == []
        assert ids(await stores.freeze.find_snapshots(tenant_id="other")) == [third]

    async def test_delete_snapshot(self, stores) -> None:
        source = write_file(stores.tmp_path / "snap.json", b"{}")
        blob_id = await stores.freeze.store_snapshot(
            source, tenant_id="acme", metadata=snapshot_metadata(doc("a"))
        )
        await stores.freeze.delete_snapshot(blob_id, tenant_id="other")  # another tenant: untouched
        dest = stores.tmp_path / "out.json"
        assert await stores.freeze.read_snapshot_to_file(blob_id, dest, tenant_id="acme") is True
        await stores.freeze.delete_snapshot(blob_id, tenant_id="acme")
        assert await stores.freeze.read_snapshot_to_file(blob_id, dest, tenant_id="acme") is False
        assert await stores.freeze.find_snapshots(tenant_id="acme") == []
        await stores.freeze.delete_snapshot(blob_id, tenant_id="acme")  # idempotent
        await stores.freeze.delete_snapshot("not-an-id", tenant_id="acme")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def test_fifty_mib_up_and_down_stays_flat(stores, request: pytest.FixtureRequest) -> None:
    """Uploading and reading 50 MiB keeps the Python heap within 3 MiB of the baseline."""
    if stores.backend == "mongodb":
        request.applymarker(
            pytest.mark.xfail(
                strict=False,
                reason=(
                    "MongoDB backend unchanged by decision: the GridFS upload through Motor "
                    "buffers the source (peak ~90 MiB for 50 MiB), so the flat-memory bound "
                    "only binds the SQL backends"
                ),
            )
        )
    block = b"m" * MIB
    source = stores.tmp_path / "fifty.bin"
    with source.open("wb") as handle:
        for _ in range(50):
            handle.write(block)
    del block
    await stores.raw.store("warm.bin", b"warm-up", "application/octet-stream", None)  # connections
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        file_id = await stores.raw.store_file(source, "fifty.bin", "application/octet-stream", None)
        up_peak = tracemalloc.get_traced_memory()[1] - baseline
        tracemalloc.reset_peak()
        total = 0
        async for part in await stores.raw.open(file_id):
            total += len(part)
        down_peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        tracemalloc.stop()
    assert total == 50 * MIB
    assert up_peak < 3 * MIB, f"upload peak {up_peak / MIB:.1f} MiB"
    assert down_peak < 3 * MIB, f"download peak {down_peak / MIB:.1f} MiB"
