"""
tests/sql_rls/conftest.py -- Fixtures of the Row-Level-Security suite (PostgreSQL only).

The suite runs on the disposable ``postgres-test`` container (see ``tests/sql/conftest.py``:
``SCINR_SQL_PG_URL`` and ``SCINR_SQL_ALLOW_WIPE=1``) and is skipped, with the reason, without
it. Every test gets its own schema in RLS mode and two restricted, non-owner roles created by
the test itself (newton never creates roles). Run only this suite with ``-m sql_rls``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest

from scinr.newton import configure
from scinr.newton.sql.engine import reset_engines
from tests.sql.conftest import _skip_unless_pg_wipeable, pg_schema, pg_url  # noqa: F401

from ._support import RlsDb, RlsRole, provision, teardown


@pytest.fixture
def rls_db() -> Iterator[RlsDb]:
    """A schema in RLS mode and two restricted roles; everything is dropped afterwards."""
    from scinr.newton.graph import reset_graph_store

    admin_url = _skip_unless_pg_wipeable()
    db = provision(admin_url)
    try:
        yield db
    finally:
        reset_graph_store()
        reset_engines()
        teardown(db)


@pytest.fixture
def connect(rls_db: RlsDb) -> Callable[[RlsRole], None]:
    """``connect(role)``: configure the library to run as a restricted *role*."""
    from scinr.newton.graph import reset_graph_store

    def _connect(role: RlsRole, **overrides: Any) -> None:
        reset_graph_store()
        reset_engines()
        configure(
            graph_backend="postgres",
            storage_backend="postgres",
            freeze_backend="postgres",
            sql_url=overrides.pop("sql_url", rls_db.url_for(role)),
            sql_schema=rls_db.schema,
            **overrides,
        )

    return _connect


@pytest.fixture
async def store_as(connect: Callable[[RlsRole], None]) -> AsyncIterator[Callable[[RlsRole], Any]]:
    """``store_as(role)``: the ``GraphStore`` of the process, running as *role*."""
    from scinr.newton.graph import get_graph_store, reset_graph_store

    opened: list[Any] = []

    def _store(role: RlsRole) -> Any:
        connect(role)
        store = get_graph_store()
        store.schema.setup()
        opened.append(store)
        return store

    yield _store
    for store in opened:
        await store.aclose()
    reset_graph_store()


@pytest.fixture
async def navigator_as(
    connect: Callable[[RlsRole], None],
) -> AsyncIterator[Callable[[RlsRole], Any]]:
    """``await navigator_as(role)``: a connected ``SqlGraphNavigator`` running as *role*."""
    from scinr.newton.navigation.sql.navigator import SqlGraphNavigator

    opened: list[SqlGraphNavigator] = []

    async def _navigator(role: RlsRole) -> SqlGraphNavigator:
        connect(role)
        navigator = SqlGraphNavigator()
        await navigator.connect()
        opened.append(navigator)
        return navigator

    yield _navigator
    for navigator in opened:
        await navigator.close()
