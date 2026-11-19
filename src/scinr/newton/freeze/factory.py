"""
freeze/factory.py — Freeze-backend factory.

Call get_freeze_storage() to obtain the configured
:class:`~scinr.newton.freeze.base.FreezeRepository`. The backend is
``ScinrConfig.freeze_backend``, already resolved by ``configure()``: the
``freeze_backend`` argument, else env ``FREEZE_BACKEND``, else the resolved
``storage_backend``.
"""

from __future__ import annotations

from scinr.newton.freeze.base import FreezeRepository


def get_freeze_storage() -> FreezeRepository:
    """Return the freeze repository for the configured backend.

    Raises
    ------
    ConfigurationError
        If the backend resolved to ``'none'``, is unknown, or is ``'custom'``
        without ``custom_freeze_storage``.
    StorageError
        If backend='mongodb' but the MongoDB server is unreachable, or a SQL
        backend cannot create or reach its schema.

    Notes
    -----
    With backend='mongodb' the first call per process (and per MongoDB
    target) pings the server and creates the indexes — the same readiness
    check as :func:`~scinr.newton.storage.factory.get_storage`, shared with it. With
    backend='postgres' or 'sqlite' the first call per process creates, migrates or
    verifies the SQL schema, likewise shared.
    """
    from scinr.newton.config import get_config
    from scinr.newton.exceptions import ConfigurationError

    cfg = get_config()
    backend = cfg.freeze_backend

    if backend == "none":
        raise ConfigurationError(
            "freeze_backend resolved to 'none' — freeze_document(), restore_document() "
            "and export_document_snapshot(destination='storage') need a freeze backend: "
            "configure storage_backend='mongodb' / 'postgres' / 'sqlite' (inherited) or an "
            "explicit freeze_backend='mongodb' / 'postgres' / 'sqlite' / 'custom'."
        )

    if backend == "mongodb":
        from scinr.newton.freeze.mongodb.repository import MongoDBFreezeRepository
        from scinr.newton.storage.factory import _ensure_mongodb_ready

        _ensure_mongodb_ready(cfg)
        return MongoDBFreezeRepository()

    if backend in ("postgres", "sqlite"):
        from scinr.newton.freeze.sql.repository import SqlFreezeRepository
        from scinr.newton.storage.factory import _ensure_sql_ready

        _ensure_sql_ready()
        return SqlFreezeRepository()

    if backend == "custom":
        if cfg.custom_freeze_storage is None:
            raise ConfigurationError(
                "freeze_backend='custom' requires passing custom_freeze_storage="
                "<FreezeRepository> to configure()."
            )
        return cfg.custom_freeze_storage

    raise ConfigurationError(
        f"Unknown freeze_backend: {backend!r}. "
        f"Valid values: 'none', 'mongodb', 'postgres', 'sqlite', 'custom'."
    )
