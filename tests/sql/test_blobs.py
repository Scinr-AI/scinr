"""tests/sql/test_blobs.py -- ``storage/sql/blobs.py`` on both engines.

Postgres keeps the bytes in ``*_chunks`` tables, SQLite in files next to the database; the
behaviour seen through :class:`BlobStore` must be the same.
"""

from __future__ import annotations

import hashlib
import tracemalloc
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa

from scinr.newton.exceptions import StorageError
from scinr.newton.sql.engine import aclose_engines, get_async_engine, get_dialect_for_config
from scinr.newton.sql.ids import new_id
from scinr.newton.sql.schema import ensure_schema
from scinr.newton.sql.tables import storage_tables
from scinr.newton.storage.sql.blobs import (
    CHUNK_SIZE,
    files_root,
    find_inconsistencies,
    get_blob_store,
    iter_bytes,
)

MIB = 1024 * 1024


@pytest.fixture
async def target(sql_backend):
    ensure_schema()
    yield sql_backend
    await aclose_engines()


def _row(**extra) -> dict:
    return {
        "id": new_id(),
        "tenant_id": "acme",
        "filename": "a.bin",
        "content_type": "application/octet-stream",
        "stored_at": datetime.now(UTC),
        **extra,
    }


async def _drain(stream) -> bytes:
    parts = []
    async for block in stream:
        parts.append(block)
    return b"".join(parts)


async def _fetch_row(table, row_id) -> dict:
    async with get_async_engine().connect() as connection:
        return dict(
            (await connection.execute(sa.select(table).where(table.c.id == row_id)))
            .mappings()
            .one()
        )


async def _count(table) -> int:
    async with get_async_engine().connect() as connection:
        return (
            await connection.execute(sa.select(sa.func.count()).select_from(table))
        ).scalar_one()


def _raw_files():
    return storage_tables(get_dialect_for_config().name).raw_files


async def test_five_mib_round_trip(target) -> None:
    content = bytes(range(256)) * (5 * MIB // 256) + b"tail"
    row = _row()
    size = await get_blob_store().put("raw_file", row, iter_bytes(content))
    assert size == len(content)
    stored = await _fetch_row(_raw_files(), row["id"])
    assert stored["size_bytes"] == len(content)
    assert stored["checksum_sha256"] == hashlib.sha256(content).hexdigest()
    assert await _drain(await get_blob_store().open("raw_file", stored)) == content


async def test_empty_content(target) -> None:
    row = _row()
    assert await get_blob_store().put("raw_file", row, iter_bytes(b"")) == 0
    stored = await _fetch_row(_raw_files(), row["id"])
    assert stored["checksum_sha256"] == hashlib.sha256(b"").hexdigest()
    assert await _drain(await get_blob_store().open("raw_file", stored)) == b""


async def test_pieces_of_any_size_are_stored_whole(target) -> None:
    async def pieces():
        for size in (10, CHUNK_SIZE + 5, 3, 2 * CHUNK_SIZE):
            yield b"x" * size

    row = _row()
    expected = 10 + CHUNK_SIZE + 5 + 3 + 2 * CHUNK_SIZE
    assert await get_blob_store().put("raw_file", row, pieces()) == expected
    stored = await _fetch_row(_raw_files(), row["id"])
    assert len(await _drain(await get_blob_store().open("raw_file", stored))) == expected


async def test_chunks_are_exactly_one_mib(target) -> None:
    if get_dialect_for_config().name != "postgres":
        pytest.skip("chunk tables exist only in Postgres")
    row = _row()
    await get_blob_store().put("raw_file", row, iter_bytes(b"y" * (2 * CHUNK_SIZE + 7)))
    chunks = storage_tables("postgres").raw_file_chunks
    async with get_async_engine().connect() as connection:
        rows = (
            await connection.execute(
                sa.select(chunks.c.n, sa.func.length(chunks.c.data), chunks.c.tenant_id)
                .where(chunks.c.file_id == row["id"])
                .order_by(chunks.c.n)
            )
        ).all()
    assert [(n, length) for n, length, _ in rows] == [(0, CHUNK_SIZE), (1, CHUNK_SIZE), (2, 7)]
    assert {tenant for _, _, tenant in rows} == {"acme"}


async def test_failure_midway_leaves_nothing(target) -> None:
    async def failing():
        yield b"a" * CHUNK_SIZE
        yield b"b" * 10
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await get_blob_store().put("raw_file", _row(), failing())
    assert await _count(_raw_files()) == 0
    if get_dialect_for_config().name == "postgres":
        assert await _count(storage_tables("postgres").raw_file_chunks) == 0
    else:
        root = files_root()
        assert not root.exists() or not [p for p in root.rglob("*") if p.is_file()]


async def test_failed_row_insert_removes_the_file(target) -> None:
    # A row without its mandatory columns fails the INSERT after the bytes were written.
    bad = {"id": new_id(), "tenant_id": "acme"}
    with pytest.raises(Exception):  # noqa: B017, PT011 — the driver's own error
        await get_blob_store().put("raw_file", bad, iter_bytes(b"data"))
    assert await _count(_raw_files()) == 0
    if get_dialect_for_config().name == "sqlite":
        root = files_root()
        assert not root.exists() or not [p for p in root.rglob("*") if p.is_file()]


async def test_missing_binary_raises_storage_error(target) -> None:
    row = _row()
    await get_blob_store().put("raw_file", row, iter_bytes(b"x" * 100))
    stored = await _fetch_row(_raw_files(), row["id"])
    if get_dialect_for_config().name == "postgres":
        chunks = storage_tables("postgres").raw_file_chunks
        async with get_async_engine().begin() as connection:
            await connection.execute(sa.delete(chunks))
    else:
        (files_root() / stored["_file_path"]).unlink()
    with pytest.raises(StorageError):
        await get_blob_store().open("raw_file", stored)


async def test_snapshot_checksum_goes_to_props(target) -> None:
    content = b"{}" * 1000
    row = {
        "id": new_id(),
        "tenant_id": "acme",
        "content_type": "application/json",
        "stored_at": datetime.now(UTC),
        "documents": [],
        "props": {"mode": "x"},
    }
    await get_blob_store().put("snapshot", row, iter_bytes(content))
    table = storage_tables(get_dialect_for_config().name).frozen_snapshots
    stored = await _fetch_row(table, row["id"])
    assert stored["props"] == {"mode": "x", "checksum_sha256": hashlib.sha256(content).hexdigest()}
    assert await _drain(await get_blob_store().open("snapshot", stored)) == content


async def test_delete_removes_the_bytes(target) -> None:
    row = _row()
    await get_blob_store().put("raw_file", row, iter_bytes(b"z" * 10))
    table = _raw_files()
    async with get_async_engine().begin() as connection:
        deleted = (
            (
                await connection.execute(
                    sa.delete(table).where(table.c.id == row["id"]).returning(table)
                )
            )
            .mappings()
            .one()
        )
    await get_blob_store().delete("raw_file", deleted)
    if get_dialect_for_config().name == "postgres":
        assert await _count(storage_tables("postgres").raw_file_chunks) == 0
    else:
        assert not (files_root() / deleted["_file_path"]).exists()
        # Deleting again is not an error (the file is already gone).
        await get_blob_store().delete("raw_file", deleted)


async def test_file_paths_come_from_the_id(target) -> None:
    if get_dialect_for_config().name != "sqlite":
        pytest.skip("files exist only in SQLite")
    row = _row(filename="../../evil.bin")
    await get_blob_store().put("raw_file", row, iter_bytes(b"1"))
    stored = await _fetch_row(_raw_files(), row["id"])
    assert stored["_file_path"] == f"raw_files/{row['id'][:2]}/{row['id'][2:4]}/{row['id']}.bin"
    with pytest.raises(ValueError):
        await get_blob_store().put("raw_file", _row(id="../../x"), iter_bytes(b"1"))
    escaping = {**stored, "_file_path": "../outside.bin"}
    with pytest.raises(StorageError):
        await get_blob_store().open("raw_file", escaping)


async def test_find_inconsistencies(target) -> None:
    if get_dialect_for_config().name == "postgres":
        assert (await find_inconsistencies()).clean
        return
    kept, lost = _row(), _row()
    await get_blob_store().put("raw_file", kept, iter_bytes(b"kept"))
    await get_blob_store().put("raw_file", lost, iter_bytes(b"lost"))
    assert (await find_inconsistencies()).clean

    lost_path = files_root() / (await _fetch_row(_raw_files(), lost["id"]))["_file_path"]
    lost_path.unlink()
    orphan = files_root() / "raw_files" / "ff" / "ff" / ("f" * 24 + ".bin")
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    leftover = orphan.with_name(orphan.name + ".tmp")
    leftover.write_bytes(b"half")

    found = await find_inconsistencies()
    assert found.rows_without_file == [("raw_file", lost["id"])]
    assert found.files_without_row == sorted(
        [orphan.relative_to(files_root()).as_posix(), leftover.relative_to(files_root()).as_posix()]
    )
    assert not found.clean


async def test_memory_peak_is_bounded(target) -> None:
    """Writing and reading 50 MiB never holds more than ~3 chunks in memory."""
    block = b"m" * CHUNK_SIZE

    async def fifty_mib():
        for _ in range(50):
            yield block

    row = _row()
    await get_blob_store().put("raw_file", _row(), iter_bytes(b"warm-up"))  # connections, caches
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        await get_blob_store().put("raw_file", row, fifty_mib())
        write_peak = tracemalloc.get_traced_memory()[1] - baseline

        stored = await _fetch_row(_raw_files(), row["id"])
        tracemalloc.reset_peak()
        total = 0
        async for part in await get_blob_store().open("raw_file", stored):
            total += len(part)
        read_peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        tracemalloc.stop()
    assert total == 50 * MIB
    assert write_peak < 3 * CHUNK_SIZE, f"write peak {write_peak / MIB:.1f} MiB"
    assert read_peak < 3 * CHUNK_SIZE, f"read peak {read_peak / MIB:.1f} MiB"
