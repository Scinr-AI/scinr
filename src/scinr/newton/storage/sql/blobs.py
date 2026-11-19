"""
storage/sql/blobs.py — Where the bytes of raw files, page images and snapshots live.

One interface, :class:`BlobStore`, with two implementations that the storage tables decide
(``tables.storage_tables(dialect.name)``: the engines that have ``*_chunks`` tables use
:class:`ChunkBlobStore`, the others :class:`FileBlobStore`):

* **Postgres**: the binary is split in chunks of :data:`CHUNK_SIZE` and stored in the
  ``*_chunks`` table of its kind, in the **same transaction** as the owner row. A failure
  leaves neither row nor chunks, and deleting the owner row deletes the chunks (``ON DELETE
  CASCADE``). Never more than one chunk (plus the one being filled) is held in memory.
* **SQLite**: one file per binary in a folder next to the ``.db`` (decision 10), named after
  the id. The file is written first (to ``<name>.tmp``, ``fsync``, ``os.replace``) and the row
  after it; if the insert fails the file is removed. Deleting removes the row first and the
  file after. A crash in between leaves an orphan file or a row without file;
  :func:`find_inconsistencies` reports both.

File paths are always built from a validated id (``^[0-9a-f]{24}$``), never from user data
such as the file name, and a stored path is checked to stay under the files folder.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

import sqlalchemy as sa
from sqlalchemy.engine import make_url

from scinr.newton.exceptions import ConfigurationError, StorageError
from scinr.newton.sql.engine import get_async_engine, get_dialect_for_config
from scinr.newton.sql.ids import is_valid_id
from scinr.newton.sql.tables import storage_tables

logger = logging.getLogger(__name__)

__all__ = [
    "CHUNK_SIZE",
    "BlobKind",
    "BlobStore",
    "ChunkBlobStore",
    "FileBlobStore",
    "Inconsistencies",
    "find_inconsistencies",
    "get_blob_store",
    "iter_bytes",
    "iter_file",
]

BlobKind = Literal["raw_file", "page_image", "snapshot"]

CHUNK_SIZE = 1024 * 1024
"""Size of a Postgres chunk and of every read / write block."""

_FILES_SUFFIX = ".files"
_SUBDIRS: dict[str, str] = {
    "raw_file": "raw_files",
    "page_image": "page_images",
    "snapshot": "snapshots",
}
_TABLE_ATTRS: dict[str, tuple[str, str]] = {
    # kind -> (owner table attribute, chunk table attribute) of ``StorageTables``
    "raw_file": ("raw_files", "raw_file_chunks"),
    "page_image": ("page_images", "page_image_chunks"),
    "snapshot": ("frozen_snapshots", "frozen_snapshot_chunks"),
}
_CHECKSUM_PROP = "checksum_sha256"


class BlobStore(Protocol):
    """Stores, opens and deletes the bytes that belong to a storage row."""

    async def put(
        self, kind: BlobKind, row: Mapping[str, Any], chunks: AsyncIterator[bytes]
    ) -> int:
        """Insert the owner row and store the bytes, atomically as far as the engine allows.

        Args:
            kind: Which table the row belongs to.
            row: The owner row without ``size_bytes``, ``checksum_sha256`` and the location
                column (``_chunk_count`` / ``_file_path``); this method fills them in. It
                must hold ``id`` (from :func:`~scinr.newton.sql.ids.new_id`) and
                ``tenant_id``. A table without ``checksum_sha256`` (snapshots) gets the
                digest in ``props``.
            chunks: The content, in blocks of any size up to :data:`CHUNK_SIZE`.

        Returns:
            The number of bytes stored.
        """

    async def open(self, kind: BlobKind, owner_row: Mapping[str, Any]) -> AsyncIterator[bytes]:
        """Stream the bytes of an already-filtered owner row.

        Raises:
            StorageError: If the bytes are missing.
        """

    async def delete(self, kind: BlobKind, owner_row: Mapping[str, Any]) -> None:
        """Remove the bytes of an owner row that has just been deleted (never raises if
        they are already gone)."""


@dataclass(frozen=True)
class Inconsistencies:
    """Result of :func:`find_inconsistencies`."""

    files_without_row: list[str] = field(default_factory=list)
    """Paths (relative to the files folder) of files no row points to, temporary files included."""
    rows_without_file: list[tuple[str, str]] = field(default_factory=list)
    """``(kind, id)`` of the rows whose file is missing."""

    @property
    def clean(self) -> bool:
        return not self.files_without_row and not self.rows_without_file


# ── Sources of bytes ──────────────────────────────────────────────────────────


async def iter_bytes(content: bytes) -> AsyncIterator[bytes]:
    """Yield *content* in blocks of :data:`CHUNK_SIZE`."""
    for start in range(0, len(content), CHUNK_SIZE):
        yield content[start : start + CHUNK_SIZE]


async def iter_file(path: Path) -> AsyncIterator[bytes]:
    """Yield the file at *path* in blocks of :data:`CHUNK_SIZE`, reading off the event loop."""
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while block := await asyncio.to_thread(handle.read, CHUNK_SIZE):
            yield block
    finally:
        await asyncio.to_thread(handle.close)


async def _rechunk(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Re-slice *source* in blocks of exactly :data:`CHUNK_SIZE` (the last one may be shorter)."""
    buffer = bytearray()
    async for piece in source:
        if not buffer and len(piece) == CHUNK_SIZE:
            yield piece
            continue
        buffer += piece
        while len(buffer) >= CHUNK_SIZE:
            yield bytes(buffer[:CHUNK_SIZE])
            del buffer[:CHUNK_SIZE]
    if buffer:
        yield bytes(buffer)


def _tables(kind: BlobKind) -> tuple[sa.Table, sa.Table | None]:
    owner_attr, chunk_attr = _TABLE_ATTRS[kind]
    tables = storage_tables(get_dialect_for_config().name)
    return getattr(tables, owner_attr), getattr(tables, chunk_attr)


def _read_engine():
    """The async engine for reads: SQLite starts a plain ``BEGIN`` (no writer queueing)."""
    return get_async_engine().execution_options(sqlite_deferred=True)


# ── Postgres: chunks in a table ───────────────────────────────────────────────


class ChunkBlobStore:
    """Binaries as ``BYTEA`` chunks of :data:`CHUNK_SIZE`, in the owner's transaction."""

    async def put(
        self, kind: BlobKind, row: Mapping[str, Any], chunks: AsyncIterator[bytes]
    ) -> int:
        table, chunk_table = _tables(kind)
        assert chunk_table is not None
        owner_column = chunk_table.c[0].name
        owner_id, tenant_id = row["id"], row["tenant_id"]
        digest = hashlib.sha256()
        size = 0
        count = 0
        placeholders: dict[str, Any] = {"size_bytes": 0, "_chunk_count": 0}
        if "checksum_sha256" in table.c:
            placeholders["checksum_sha256"] = ""
        async with get_async_engine().begin() as connection:
            # The row goes first: the chunks reference it (ON DELETE CASCADE).
            await connection.execute(sa.insert(table).values(**row, **placeholders))
            async with aclosing(chunks), aclosing(_rechunk(chunks)) as blocks:
                async for block in blocks:
                    digest.update(block)
                    size += len(block)
                    await connection.execute(
                        sa.insert(chunk_table).values(
                            {
                                owner_column: owner_id,
                                "n": count,
                                "tenant_id": tenant_id,
                                "data": block,
                            }
                        )
                    )
                    count += 1
            final: dict[str, Any] = {"size_bytes": size, "_chunk_count": count}
            if "checksum_sha256" in table.c:
                final["checksum_sha256"] = digest.hexdigest()
            else:
                final["props"] = {**(row.get("props") or {}), _CHECKSUM_PROP: digest.hexdigest()}
            await connection.execute(sa.update(table).where(table.c.id == owner_id).values(**final))
        return size

    async def open(self, kind: BlobKind, owner_row: Mapping[str, Any]) -> AsyncIterator[bytes]:
        _, chunk_table = _tables(kind)
        assert chunk_table is not None
        owner_id = owner_row["id"]
        count = owner_row["_chunk_count"]
        if count == 0:
            if owner_row["size_bytes"] > 0:
                raise StorageError(f"The binary of {kind} {owner_id!r} is missing (no chunks).")
            return _empty()
        # Fetched eagerly so that a missing binary fails here, like the GridFS open does.
        first = await self._read_chunk(chunk_table, owner_id, 0)
        return self._stream(chunk_table, owner_id, count, first)

    @staticmethod
    async def _read_chunk(
        chunk_table: sa.Table, owner_id: str, n: int, connection: Any = None
    ) -> bytes:
        owner_column = chunk_table.c[0]
        statement = sa.select(chunk_table.c.data).where(
            owner_column == owner_id, chunk_table.c.n == n
        )
        if connection is None:
            async with _read_engine().connect() as own:
                data = (await own.execute(statement)).scalar_one_or_none()
        else:
            data = (await connection.execute(statement)).scalar_one_or_none()
        if data is None:
            raise StorageError(f"Chunk {n} of the binary {owner_id!r} is missing.")
        return data if isinstance(data, bytes) else bytes(data)

    async def _stream(
        self, chunk_table: sa.Table, owner_id: str, count: int, first: bytes
    ) -> AsyncIterator[bytes]:
        yield first
        del first
        if count == 1:
            return
        async with _read_engine().connect() as connection:
            for n in range(1, count):
                yield await self._read_chunk(chunk_table, owner_id, n, connection)

    async def delete(self, kind: BlobKind, owner_row: Mapping[str, Any]) -> None:
        return None  # ON DELETE CASCADE removed the chunks with the owner row


async def _empty() -> AsyncIterator[bytes]:
    """An empty stream."""
    return
    yield b""  # unreachable: makes this an async generator


# ── SQLite: files next to the database ────────────────────────────────────────


def files_root() -> Path:
    """The folder of the binaries: ``sqlite_files_dir`` or ``<database without extension>.files``.

    Raises:
        ConfigurationError: For an in-memory database without ``sqlite_files_dir``.
    """
    from scinr.newton.config import get_config

    cfg = get_config()
    if cfg.sqlite_files_dir:
        return Path(cfg.sqlite_files_dir)
    database = make_url(cfg.sql_url).database
    if not database or database == ":memory:":
        raise ConfigurationError(
            "An in-memory SQLite database has nowhere to keep binary files: set sqlite_files_dir."
        )
    return Path(database).with_suffix(_FILES_SUFFIX)


def _relative_path(kind: BlobKind, blob_id: str) -> str:
    if not is_valid_id(blob_id):
        raise ValueError(f"Invalid id for a stored file: {blob_id!r}")
    return f"{_SUBDIRS[kind]}/{blob_id[:2]}/{blob_id[2:4]}/{blob_id}.bin"


def _resolve(root: Path, relative: str) -> Path:
    """The absolute path of a stored relative path, refusing anything outside *root*."""
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise StorageError(f"Stored file path {relative!r} points outside the files folder.")
    return path


def _open_for_write(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")


def _write_block(handle: BinaryIO, block: bytes, digest: Any) -> None:
    handle.write(block)
    digest.update(block)


def _finish_write(handle: BinaryIO, temporary: Path, final: Path) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(temporary, final)


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


class FileBlobStore:
    """Binaries as files in a folder next to the SQLite database."""

    async def put(
        self, kind: BlobKind, row: Mapping[str, Any], chunks: AsyncIterator[bytes]
    ) -> int:
        table, _ = _tables(kind)
        root = files_root()
        relative = _relative_path(kind, row["id"])
        final = root / relative
        temporary = final.with_name(final.name + ".tmp")
        size, checksum = await self._write_file(chunks, temporary, final)
        values: dict[str, Any] = {**row, "size_bytes": size, "_file_path": relative}
        if "checksum_sha256" in table.c:
            values["checksum_sha256"] = checksum
        else:
            values["props"] = {**(row.get("props") or {}), _CHECKSUM_PROP: checksum}
        try:
            async with get_async_engine().begin() as connection:
                await connection.execute(sa.insert(table).values(**values))
        except BaseException:
            await asyncio.to_thread(_unlink, final)
            raise
        return size

    @staticmethod
    async def _write_file(
        chunks: AsyncIterator[bytes], temporary: Path, final: Path
    ) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        try:
            handle = await asyncio.to_thread(_open_for_write, temporary)
        except OSError as exc:
            raise StorageError(f"Cannot write the file {final.name}: {exc}") from exc
        try:
            async with aclosing(chunks):
                async for block in chunks:
                    await asyncio.to_thread(_write_block, handle, block, digest)
                    size += len(block)
            await asyncio.to_thread(_finish_write, handle, temporary, final)
        except BaseException as exc:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(_unlink, temporary)
            if isinstance(exc, OSError):
                raise StorageError(f"Cannot write the file {final.name}: {exc}") from exc
            raise
        return size, digest.hexdigest()

    async def open(self, kind: BlobKind, owner_row: Mapping[str, Any]) -> AsyncIterator[bytes]:
        path = _resolve(files_root(), owner_row["_file_path"])
        try:
            handle = await asyncio.to_thread(path.open, "rb")
        except FileNotFoundError as exc:
            raise StorageError(
                f"The binary of {kind} {owner_row['id']!r} is missing: {owner_row['_file_path']}."
            ) from exc
        except OSError as exc:
            raise StorageError(
                f"Cannot read the binary of {kind} {owner_row['id']!r}: {exc}"
            ) from exc
        return self._stream(handle)

    @staticmethod
    async def _stream(handle: BinaryIO) -> AsyncIterator[bytes]:
        try:
            while block := await asyncio.to_thread(handle.read, CHUNK_SIZE):
                yield block
        finally:
            await asyncio.to_thread(handle.close)

    async def delete(self, kind: BlobKind, owner_row: Mapping[str, Any]) -> None:
        path = _resolve(files_root(), owner_row["_file_path"])
        try:
            removed = await asyncio.to_thread(_unlink, path)
        except OSError as exc:
            raise StorageError(
                f"Cannot delete the binary of {kind} {owner_row['id']!r}: {exc}"
            ) from exc
        if not removed:
            logger.warning(
                "The file %s of %s %r is already gone; its row was deleted anyway.",
                owner_row["_file_path"],
                kind,
                owner_row["id"],
            )


def get_blob_store() -> BlobStore:
    """The :class:`BlobStore` of the configured engine."""
    tables = storage_tables(get_dialect_for_config().name)
    return FileBlobStore() if tables.raw_file_chunks is None else ChunkBlobStore()


# ── Consistency check (SQLite) ────────────────────────────────────────────────


def _list_files(root: Path) -> set[str]:
    found: set[str] = set()
    for directory, _, names in os.walk(root):
        for name in names:
            found.add(Path(directory, name).relative_to(root).as_posix())
    return found


async def find_inconsistencies() -> Inconsistencies:
    """Files without a row and rows without a file (SQLite storage).

    A crash between writing a file and inserting its row (or deleting the row and the file)
    leaves one of the two behind. The library never calls this function: run it after a
    crash and delete what it reports. Postgres keeps the bytes in the owner's transaction,
    so it can never be inconsistent and the result is always clean.
    """
    tables = storage_tables(get_dialect_for_config().name)
    if tables.raw_file_chunks is not None:
        return Inconsistencies()
    root = files_root()
    on_disk = await asyncio.to_thread(_list_files, root) if root.is_dir() else set()
    referenced: set[str] = set()
    missing: list[tuple[str, str]] = []
    async with _read_engine().connect() as connection:
        for kind, (owner_attr, _) in _TABLE_ATTRS.items():
            table = getattr(tables, owner_attr)
            result = await connection.execute(sa.select(table.c.id, table.c["_file_path"]))
            for row_id, relative in result:
                referenced.add(relative)
                if relative not in on_disk:
                    missing.append((kind, row_id))
    return Inconsistencies(
        files_without_row=sorted(on_disk - referenced), rows_without_file=sorted(missing)
    )
