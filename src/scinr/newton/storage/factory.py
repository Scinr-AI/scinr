"""
storage/factory.py — Repository factory.

Call get_storage() to obtain the configured pair of repository implementations.
The backend is determined by ScinrConfig (from scinr_config) which reads
STORAGE_BACKEND from the environment or from configure().
"""

from __future__ import annotations

import logging
import threading

from scinr.newton.storage.base import PageRepository, RawFileRepository

logger = logging.getLogger(__name__)

# MongoDB targets already pinged (and indexed) in this process.  get_storage()
# runs on every source-text read, so the check must not repeat per call.
_mongodb_ready: set[tuple] = set()
_mongodb_ready_lock = threading.Lock()


def get_storage() -> tuple[RawFileRepository, PageRepository]:
    """Return the repository implementations for the configured backend.

    Returns
    -------
    tuple[RawFileRepository, PageRepository]
        A (raw_file_repo, page_repo) pair ready to use.

    Raises
    ------
    ConfigurationError
        If the storage backend is unknown or misconfigured.
        If backend='custom' but custom_storage was not provided.
    StorageError
        If backend='mongodb' but the MongoDB server is unreachable.

    Notes
    -----
    With backend='mongodb' the first call per process (and per MongoDB
    target) pings the server and, unless ``mongodb_ensure_indexes=False``,
    creates the required indexes.  Later calls skip both.
    """
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError
    cfg = get_config()
    backend = cfg.storage_backend

    if backend == "none":
        from scinr.newton.storage.null import NullPageRepository, NullRawFileRepository
        return NullRawFileRepository(), NullPageRepository()

    if backend == "mongodb":
        _ensure_mongodb_ready(cfg)
        from scinr.newton.storage.mongodb.pages import MongoDBPageRepository
        from scinr.newton.storage.mongodb.raw_files import MongoDBRawFileRepository
        return MongoDBRawFileRepository(), MongoDBPageRepository()

    if backend == "custom":
        if cfg.custom_storage is None:
            raise ConfigurationError(
                "storage_backend='custom' requires passing custom_storage=(raw_repo, page_repo) "
                "to configure()."
            )
        return cfg.custom_storage

    raise ConfigurationError(
        f"Unknown storage_backend: {backend!r}. "
        f"Valid values: 'none', 'mongodb', 'custom'."
    )


def reset_mongodb_readiness() -> None:
    """Forget which MongoDB targets were already verified in this process.

    Called by :func:`~scinr.newton.storage.mongodb.client.reset_client`
    (hence by ``configure()``), so a configuration change re-checks the
    connection and the indexes.
    """
    with _mongodb_ready_lock:
        _mongodb_ready.clear()


def _ensure_mongodb_ready(cfg) -> None:
    """Run :func:`_check_mongodb_connection` once per process and MongoDB target."""
    key = (
        cfg.mongodb_uri,
        cfg.mongodb_database,
        cfg.mongodb_raw_files_collection,
        cfg.mongodb_pages_collection,
        cfg.mongodb_ensure_indexes,
    )
    if key in _mongodb_ready:
        return
    with _mongodb_ready_lock:
        if key in _mongodb_ready:
            return
        _check_mongodb_connection(cfg)
        _mongodb_ready.add(key)


def _check_mongodb_connection(cfg) -> None:
    """Ping MongoDB using the synchronous pymongo client to verify connectivity.

    Uses pymongo (not motor) deliberately — this function is always called from
    inside an async context (under asyncio.run), so creating a new event loop
    here would raise RuntimeError. pymongo is a synchronous client with no event
    loop dependency, and is always available as a transitive dependency of motor.

    When ``cfg.mongodb_ensure_indexes`` is true, the same client then creates
    the required indexes.  A failure there (typically ``OperationFailure``
    because the user lacks the ``createIndex`` privilege) is logged as a
    warning and does not block storage access.
    """
    from scinr.newton.exceptions import StorageError
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
        client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
            if getattr(cfg, "mongodb_ensure_indexes", True):
                _ensure_indexes_best_effort(cfg, client)
        finally:
            client.close()
    except ImportError:
        pass  # pymongo/motor not installed — will fail later with a clear error
    except PyMongoError as exc:
        from scinr.newton.utils.redaction import redact_secrets, redact_uri

        # The URI embeds the credentials: this message reaches logs and, through
        # StageResult.errors, API callers — never include it unredacted.
        uri_display = redact_uri(cfg.mongodb_uri or "mongodb://localhost:27017")
        raise StorageError(
            f"Cannot connect to MongoDB at '{uri_display}': {redact_secrets(str(exc))}\n"
            f"Ensure MongoDB is running or use storage_backend='none'."
        ) from exc


def _ensure_indexes_best_effort(cfg, client) -> None:
    """Create the MongoDB indexes, logging a warning instead of failing."""
    from pymongo.errors import PyMongoError

    from scinr.newton.storage.mongodb.client import ensure_indexes_sync

    try:
        ensure_indexes_sync(cfg, client=client)
    except PyMongoError as exc:
        from scinr.newton.utils.redaction import redact_secrets

        logger.warning(
            "Could not create MongoDB indexes in database %r: %s. Storage keeps "
            "working, but reads may scan whole collections. Grant the createIndex "
            "privilege, create the indexes out of band, or set "
            "mongodb_ensure_indexes=False to silence this warning.",
            cfg.mongodb_database,
            redact_secrets(str(exc)),
        )
