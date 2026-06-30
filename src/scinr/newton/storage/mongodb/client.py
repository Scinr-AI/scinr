"""
storage/mongodb/client.py — Motor async client singleton + GridFS access.

A single :class:`~motor.motor_asyncio.AsyncIOMotorClient` instance is kept
per process.  All repositories in this backend share it to avoid exhausting
connection-pool resources.

Public API
----------
get_client()
    Return the singleton Motor client (creates it on first call).
get_db()
    Return the configured Motor database object.
get_gridfs_bucket()
    Return an :class:`~motor.motor_asyncio.AsyncIOMotorGridFSBucket` for
    binary file storage.
ensure_indexes()
    Coroutine — create all required indexes (idempotent).  Call once at
    application startup.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return the Motor singleton client, creating it on first call.

    The client is module-level so that it survives across coroutine calls
    within the same process and reuses the underlying connection pool.

    Returns
    -------
    AsyncIOMotorClient
        The shared Motor client instance.
    """
    global _client
    if _client is None:
        from scinr.newton.config import get_config
        cfg = get_config()
        logger.debug("Creating MongoDB Motor client (URI: %s)", cfg.mongodb_uri)
        _client = AsyncIOMotorClient(cfg.mongodb_uri)
    return _client


def reset_client() -> None:
    """Reset the singleton Motor client.

    Forces :func:`get_client` to create a new client on the next call.
    Must be called whenever the configuration changes (e.g. after
    :func:`~scinr.newton.config.configure`) so that the new URI and
    database settings are picked up.
    """
    global _client
    _client = None


def get_db():
    """Return the Motor database object for the configured database.

    Returns
    -------
    AsyncIOMotorDatabase
        The Motor database identified by ``cfg.mongodb_database``.
    """
    from scinr.newton.config import get_config
    cfg = get_config()
    return get_client()[cfg.mongodb_database]


def get_gridfs_bucket() -> AsyncIOMotorGridFSBucket:
    """Return the GridFS bucket for binary file storage.

    The bucket name is read from ``cfg.mongodb_gridfs_bucket``
    (default ``"raw_binaries"``).

    Returns
    -------
    AsyncIOMotorGridFSBucket
        GridFS bucket ready for upload/download operations.
    """
    from scinr.newton.config import get_config
    cfg = get_config()
    return AsyncIOMotorGridFSBucket(get_db(), bucket_name=cfg.mongodb_gridfs_bucket)


# ---------------------------------------------------------------------------
# Index bootstrap
# ---------------------------------------------------------------------------


async def ensure_indexes() -> None:
    """Create all required MongoDB indexes (idempotent).

    Uses Motor's ``create_index`` which maps to a ``createIndex`` command
    that is a no-op if the index already exists with the same key pattern
    and name.

    Should be called once at application startup to guarantee optimal query
    performance from the first request.
    """
    from scinr.newton.config import get_config
    cfg = get_config()

    db = get_db()

    # converted_pages: primary lookup by raw_file_id + page_index ordering
    await db[cfg.mongodb_pages_collection].create_index(
        [("raw_file_id", 1), ("page_index", 1)],
        name="pages_by_raw_file_and_index",
    )
    # converted_pages: secondary lookup by filename + folder_path
    await db[cfg.mongodb_pages_collection].create_index(
        [("filename", 1), ("folder_path", 1)],
        name="pages_by_filename_folder",
    )
    # raw_files: deduplication by SHA-256 checksum
    await db[cfg.mongodb_raw_files_collection].create_index(
        [("checksum_sha256", 1)],
        name="raw_files_by_checksum",
    )

    logger.debug("MongoDB indexes ensured.")
