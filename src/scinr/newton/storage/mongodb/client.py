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
    Coroutine — create all required indexes (idempotent).
ensure_indexes_sync()
    Same indexes with a synchronous pymongo client.  Called automatically
    by :func:`~scinr.newton.storage.factory.get_storage` once per process.
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
        from scinr.newton.utils.redaction import redact_uri
        cfg = get_config()
        logger.debug("Creating MongoDB Motor client (URI: %s)", redact_uri(cfg.mongodb_uri))
        _client = AsyncIOMotorClient(cfg.mongodb_uri)
    return _client


def reset_client() -> None:
    """Reset the singleton Motor client.

    Forces :func:`get_client` to create a new client on the next call.
    Must be called whenever the configuration changes (e.g. after
    :func:`~scinr.newton.config.configure`) so that the new URI and
    database settings are picked up.  Also clears the per-process cache of
    verified connections, so the next :func:`get_storage` call pings the
    server and ensures the indexes again.
    """
    global _client
    _client = None
    from scinr.newton.storage.factory import reset_mongodb_readiness
    reset_mongodb_readiness()


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

# Single source of truth for the indexes of this backend, shared by the async
# (Motor) and sync (pymongo) bootstraps.  Each entry is
# ``(config attribute holding the collection name, keys, index name)``.
# Every index is prefixed by ``tenant_id`` so tenant-scoped reads are seeks.
_INDEX_SPECS: tuple[tuple[str, list[tuple[str, int]], str], ...] = (
    # converted_pages: primary lookup by (tenant, raw_file_id) + page_index ordering
    (
        "mongodb_pages_collection",
        [("tenant_id", 1), ("raw_file_id", 1), ("page_index", 1)],
        "pages_by_tenant_raw_file_and_index",
    ),
    # converted_pages: secondary lookup by filename + folder_path within a tenant
    (
        "mongodb_pages_collection",
        [("tenant_id", 1), ("filename", 1), ("folder_path", 1)],
        "pages_by_tenant_filename_folder",
    ),
    # raw_files: integrity / duplicate checks by SHA-256 within a tenant
    (
        "mongodb_raw_files_collection",
        [("tenant_id", 1), ("checksum_sha256", 1)],
        "raw_files_by_tenant_checksum",
    ),
    # raw_files: inventory / audit by provenance (list_raw_files)
    (
        "mongodb_raw_files_collection",
        [("tenant_id", 1), ("created_by_user_id", 1)],
        "raw_files_by_tenant_user",
    ),
    (
        "mongodb_raw_files_collection",
        [("tenant_id", 1), ("job_id", 1)],
        "raw_files_by_tenant_job",
    ),
)

# Pre-multitenancy indexes, replaced by their tenant-prefixed versions above.
_OBSOLETE_INDEXES: tuple[tuple[str, str], ...] = (
    ("mongodb_pages_collection", "pages_by_raw_file_and_index"),
    ("mongodb_pages_collection", "pages_by_filename_folder"),
    ("mongodb_raw_files_collection", "raw_files_by_checksum"),
)


async def ensure_indexes() -> None:
    """Create all required MongoDB indexes (idempotent).

    Uses Motor's ``create_index`` which maps to a ``createIndex`` command
    that is a no-op if the index already exists with the same key pattern
    and name.

    :func:`~scinr.newton.storage.factory.get_storage` already creates the
    indexes (synchronously, once per process) unless
    ``mongodb_ensure_indexes=False``; call this only when managing them
    explicitly from async code.
    """
    from scinr.newton.config import get_config
    cfg = get_config()

    db = get_db()
    for coll_attr, name in _OBSOLETE_INDEXES:
        await _drop_index_if_exists(db[getattr(cfg, coll_attr)], name)
    for coll_attr, keys, name in _INDEX_SPECS:
        await db[getattr(cfg, coll_attr)].create_index(keys, name=name)

    logger.debug("MongoDB indexes ensured.")


def ensure_indexes_sync(cfg, client=None) -> None:
    """Create all required MongoDB indexes with a synchronous pymongo client.

    Same indexes as :func:`ensure_indexes`.  pymongo is used instead of Motor
    because this runs from :func:`~scinr.newton.storage.factory.get_storage`,
    which is called inside running event loops, and a Motor client binds to
    the first loop it is used on.

    Parameters
    ----------
    cfg : ScinrConfig
        Configuration providing the URI, database and collection names.
    client : pymongo.MongoClient, optional
        Client to reuse.  When omitted a temporary one is created and closed.

    Raises
    ------
    pymongo.errors.PyMongoError
        If an index cannot be created (e.g. ``OperationFailure`` when the user
        lacks the ``createIndex`` privilege).
    """
    from pymongo import MongoClient

    own_client = client is None
    if own_client:
        client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        db = client[cfg.mongodb_database]
        for coll_attr, name in _OBSOLETE_INDEXES:
            collection = db[getattr(cfg, coll_attr)]
            if name in collection.index_information():
                collection.drop_index(name)
                logger.info("Dropped obsolete MongoDB index %s.%s", collection.name, name)
        for coll_attr, keys, name in _INDEX_SPECS:
            db[getattr(cfg, coll_attr)].create_index(keys, name=name)
    finally:
        if own_client:
            client.close()

    logger.debug("MongoDB indexes ensured.")


async def _drop_index_if_exists(collection, name: str) -> None:
    """Drop index *name* from *collection*, ignoring "index not found"."""
    from pymongo.errors import OperationFailure

    try:
        await collection.drop_index(name)
        logger.info("Dropped obsolete MongoDB index %s.%s", collection.name, name)
    except OperationFailure:
        pass
