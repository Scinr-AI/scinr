"""
storage/factory.py — Repository factory.

Call get_storage() to obtain the configured pair of repository implementations.
The backend is determined by ScinrConfig (from scinr_config) which reads
STORAGE_BACKEND from the environment or from configure().
"""

from __future__ import annotations

from scinr.newton.storage.base import PageRepository, RawFileRepository


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
    """
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError
    cfg = get_config()
    backend = cfg.storage_backend

    if backend == "none":
        from scinr.newton.storage.null import NullPageRepository, NullRawFileRepository
        return NullRawFileRepository(), NullPageRepository()

    if backend == "mongodb":
        _check_mongodb_connection(cfg)
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


def _check_mongodb_connection(cfg) -> None:
    """Ping MongoDB using the synchronous pymongo client to verify connectivity.

    Uses pymongo (not motor) deliberately — this function is always called from
    inside an async context (under asyncio.run), so creating a new event loop
    here would raise RuntimeError. pymongo is a synchronous client with no event
    loop dependency, and is always available as a transitive dependency of motor.
    """
    from scinr.newton.exceptions import StorageError
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
        client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
        finally:
            client.close()
    except ImportError:
        pass  # pymongo/motor not installed — will fail later with a clear error
    except PyMongoError as exc:
        uri_display = cfg.mongodb_uri or "mongodb://localhost:27017"
        raise StorageError(
            f"Cannot connect to MongoDB at '{uri_display}': {exc}\n"
            f"Ensure MongoDB is running or use storage_backend='none'."
        ) from exc
