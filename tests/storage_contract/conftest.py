"""
tests/storage_contract/conftest.py -- Fixtures of the storage contract suite.

The suite checks the behaviour that **every** backend of ``RawFileRepository``,
``PageRepository`` and ``FreezeRepository`` must have, through the factories
(``get_storage()`` / ``get_freeze_storage()``) only. The reference is the MongoDB
implementation: a case that passes in SQL and fails in MongoDB (or the other way round) is a
hole in the contract, not a test to adapt.

Backends
--------
* ``sqlite``: a database and its files folder in ``tmp_path``. Always available.
* ``postgres``: the disposable ``postgres-test`` container (see ``tests/sql/README.md``);
  skipped with the reason when it is not there.
* ``mongodb``: the disposable MongoDB of the golden harness (``localhost:27028``) with
  ``SCINR_GOLDEN_ALLOW_WIPE=1``; **its collections are wiped before each test**. Skipped
  with the reason otherwise.

To add a backend: write a function that configures the library (and raises ``Unavailable``
when it must not run) and add it to :data:`BACKENDS`; every test then runs against it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from tests.sql.conftest import pg_schema, pg_url  # noqa: F401  (fixtures used by name)

_NEO4J = {"neo4j_uri": "bolt://localhost:7687", "neo4j_user": "u", "neo4j_password": "p"}
_GOLDEN_ALLOW = "SCINR_GOLDEN_ALLOW_WIPE"
_GOLDEN_MONGO_PORT = 27028
_GOLDEN_MONGO_URI = os.getenv(
    "SCINR_GOLDEN_MONGODB_URI", f"mongodb://localhost:{_GOLDEN_MONGO_PORT}"
)
_GOLDEN_MONGO_DATABASE = "scinr_golden"


class Unavailable(Exception):
    """The backend cannot (or must not) be used here; the message says why."""


@dataclass(frozen=True)
class Stores:
    """The three repositories of one backend, as the factories return them."""

    backend: str
    raw: Any
    pages: Any
    freeze: Any
    tmp_path: Path


def _sqlite(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    from scinr.newton import configure

    configure(
        graph_backend="sqlite",
        storage_backend="sqlite",
        freeze_backend="sqlite",
        sql_url=f"sqlite:///{tmp_path}/newton.db",
    )


def _postgres(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    from scinr.newton import configure

    url = request.getfixturevalue("pg_url")
    schema = request.getfixturevalue("pg_schema")
    configure(
        graph_backend="postgres",
        storage_backend="postgres",
        freeze_backend="postgres",
        sql_url=url,
        sql_schema=schema,
    )


def _mongodb(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    if os.getenv(_GOLDEN_ALLOW) != "1":
        raise Unavailable(f"{_GOLDEN_ALLOW}=1 not set (the suite wipes its MongoDB)")
    parsed = urlparse(_GOLDEN_MONGO_URI)
    if (
        parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.port != _GOLDEN_MONGO_PORT
    ):
        raise Unavailable(
            f"{_GOLDEN_MONGO_URI!r} is not the disposable golden MongoDB; refusing to wipe it"
        )
    try:
        import pymongo

        client = pymongo.MongoClient(_GOLDEN_MONGO_URI, serverSelectionTimeoutMS=2000)
    except ImportError as exc:
        raise Unavailable("pymongo is not installed") from exc
    try:
        database = client[_GOLDEN_MONGO_DATABASE]
        for name in database.list_collection_names():
            database[name].delete_many({})
        for bucket in ("fs.files", "fs.chunks"):
            database[bucket].delete_many({})
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(f"MongoDB not reachable at {_GOLDEN_MONGO_URI}: {exc}") from exc
    finally:
        client.close()
    from scinr.newton import configure

    configure(
        **_NEO4J,
        storage_backend="mongodb",
        freeze_backend="mongodb",
        mongodb_uri=_GOLDEN_MONGO_URI,
        mongodb_database=_GOLDEN_MONGO_DATABASE,
    )


BACKENDS: dict[str, Callable[[pytest.FixtureRequest, Path], None]] = {
    "sqlite": _sqlite,
    "postgres": _postgres,
    "mongodb": _mongodb,
}


@pytest.fixture(params=list(BACKENDS))
async def stores(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Stores]:
    """Configure the library for one backend and return its repositories."""
    from scinr.newton.freeze.factory import get_freeze_storage
    from scinr.newton.storage.factory import get_storage

    name = request.param
    try:
        BACKENDS[name](request, tmp_path)
    except Unavailable as exc:
        pytest.skip(f"{name}: {exc}")
    raw, pages = get_storage()
    try:
        yield Stores(name, raw, pages, get_freeze_storage(), tmp_path)
    finally:
        if name == "mongodb":
            from scinr.newton.storage.mongodb.client import reset_client

            reset_client()
        else:
            from scinr.newton.sql.engine import aclose_engines, reset_engines

            await aclose_engines()
            reset_engines()
