"""
tests/contract/conftest.py -- Fixtures of the ``GraphStore`` contract suite.

The suite checks the behaviour that **every** backend of
:class:`~scinr.newton.graph.base.GraphStore` must have: each method of the interface
works, writes are idempotent, tenants are isolated and engine failures leave as the
exceptions of ``scinr.newton.exceptions``. The tests talk to the store through
``get_graph_store()`` only (never Cypher, never ``execute_raw``, never driver objects)
and observe the result with a neutral dump of the graph, the same format as
``tests/golden/_dump.py``::

    {"nodes": [{"key": "...", "labels": [...], "props": {...}}],
     "relationships": [{"type": "...", "from": "<key>", "to": "<key>", "props": {...}}]}

How a new backend plugs in
--------------------------
1. Write a function that returns a :class:`Backend` (see :func:`_neo4j_backend`):

   * ``prepare()``: refuse to run (raise :class:`BackendUnavailable`) unless it is safe to
     wipe the backend, wipe it, and ``configure()`` the library for it (including
     ``graph_backend``). The tests never wipe anything by themselves.
   * ``dump()``: the neutral dump of the whole graph, read by a channel independent of the
     store under test.
   * ``broken_configs``: ``configure()`` keyword arguments for the error-contract cases
     every engine can reproduce: ``unreachable`` (a host nobody listens on) and
     ``bad_credentials`` (an engine that rejects the credentials).
   * ``drop_credentials()``: make the live configuration lack the credentials (``configure()``
     itself refuses that, so the test removes them after configuring), for the
     ``GraphConfigurationError`` case.

2. Add it to :data:`BACKENDS`. Every test of the suite then runs against it, with the
   ``graph_store`` fixture.

Safety: the Neo4j backend wipes its graph and the MongoDB collections before **each** test,
so it only runs on the disposable containers of the golden harness (Neo4j on 7698, MongoDB
on 27028) with ``SCINR_GOLDEN_ALLOW_WIPE=1`` and ``PYTHONHASHSEED=0``; otherwise every test
is skipped with the reason. Start them as described in
``tests/integration/test_graph_store_golden.py``. Never run this suite and the golden one at
the same time: they share the databases.

The SQL backends (graph, storage and freeze in the same engine): ``sqlite`` uses a new file
per test and always runs; ``postgres`` drops the schema ``newton_test_<pid>`` of the
disposable ``postgres-test`` container (``SCINR_SQL_ALLOW_WIPE=1``, see
``tests/sql/conftest.py``) before each test, and is skipped without it. After every test of
a SQL backend the invariants of ``tests/sql/_invariants.py`` are checked.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from tests.golden import _runner
from tests.golden._dump import dump_graph, normalize_artifacts
from tests.golden._scenarios import TENANT_A, TENANT_B, Env, prepare_script


class BackendUnavailable(Exception):
    """The backend cannot (or must not) be used here; the message says why."""


@dataclass(frozen=True)
class Backend:
    """What the suite needs from one engine besides its ``GraphStore``."""

    name: str
    prepare: Callable[[], None]
    dump: Callable[[], dict[str, Any]]
    broken_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    drop_credentials: Callable[[], None] = lambda: None
    #: ``configure_harness()`` arguments of the backend; the broken configs go on top.
    settings: Callable[[], dict[str, Any]] = dict
    #: The credential of ``bad_credentials`` that must never appear in a message.
    secret: str | None = None
    #: Error-contract cases the engine cannot reproduce, with the reason (skipped).
    unsupported: dict[str, str] = field(default_factory=dict)
    #: Called after each test with the backend still in place (SQL: the invariants).
    after_test: Callable[[], None] = lambda: None


# ---------------------------------------------------------------------------
# Neo4j (the disposable container of the golden harness)
# ---------------------------------------------------------------------------


def _neo4j_prepare() -> None:
    try:
        _runner.guard()
        _runner.wipe_databases()
    except _runner.GoldenUnavailable as exc:
        raise BackendUnavailable(str(exc)) from exc
    _runner.configure_harness()
    prepare_script()


def _neo4j_dump() -> dict[str, Any]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        _runner.NEO4J_URI, auth=(_runner.NEO4J_USER, _runner.NEO4J_PASSWORD)
    )
    try:
        return dump_graph(driver, _runner.NEO4J_DATABASE)
    finally:
        driver.close()


def _neo4j_drop_credentials() -> None:
    from scinr.newton.config import get_config

    get_config().neo4j_user = None
    get_config().neo4j_password = None


def _neo4j_backend() -> Backend:
    return Backend(
        name="neo4j",
        prepare=_neo4j_prepare,
        dump=_neo4j_dump,
        broken_configs={
            # Nothing listens on port 1 (never 7687/7688).
            "unreachable": {"neo4j_uri": "bolt://127.0.0.1:1"},
            # An unknown user: failed logins of the real one would lock it out for a while.
            "bad_credentials": {
                "neo4j_user": "contract-no-such-user",
                "neo4j_password": "definitely-not-the-password",
            },
        },
        drop_credentials=_neo4j_drop_credentials,
        secret="definitely-not-the-password",
    )


# ---------------------------------------------------------------------------
# SQL (SQLite file per test; Postgres schema of the disposable container)
# ---------------------------------------------------------------------------


def _sql_drop_credentials() -> None:
    from scinr.newton.config import get_config

    get_config().sql_url = ""


def _sql_backend(engine: str, *, rls: bool = False) -> Backend:
    """A SQL backend; with ``rls`` the process runs as a restricted role of a RLS schema.

    The RLS variant (``postgres_rls``, marker ``sql_rls``) creates the schema in RLS mode and
    a restricted role with write access to the tenants the suite uses (``golden-a``,
    ``golden-b``, ``t`` and the public data); the dumps and the invariants are read as the
    administrator, which bypasses the policies.
    """
    state: dict[str, Any] = {}

    def settings() -> dict[str, Any]:
        return {
            "graph_backend": engine,
            "storage_backend": engine,
            "freeze_backend": engine,
            "sql_url": state["url"],
            "sql_schema": state["schema"],
        }

    def prepare() -> None:
        from scinr.newton.sql.engine import reset_engines

        reset_engines()
        if engine == "sqlite":
            folder = Path(tempfile.mkdtemp(prefix="scinr_contract_sqlite_"))
            state["url"] = f"sqlite:///{folder}/newton.db"
            state["schema"] = "newton"
        else:
            from tests.sql.conftest import (
                WIPE_SCHEMA_PREFIX,
                _skip_unless_pg_wipeable,
                drop_pg_schema,
            )

            try:
                url = _skip_unless_pg_wipeable()
            except pytest.skip.Exception as exc:
                raise BackendUnavailable(str(exc)) from exc
            state["url"] = state["admin_url"] = url
            state["schema"] = f"{WIPE_SCHEMA_PREFIX}{os.getpid()}"
            drop_pg_schema(url, state["schema"])
            if rls:
                from tests.sql_rls._support import provision_runtime_role

                state["url"] = provision_runtime_role(
                    url, state["schema"], tenants=(TENANT_A, TENANT_B, "t", None)
                )
        _runner.configure_harness(**settings())
        prepare_script()

    def dump() -> dict[str, Any]:
        from tests.sql._dump import dump_sql

        url = state.get("admin_url", state["url"])
        return dump_sql(url, state["schema"] if engine == "postgres" else None)

    def after_test() -> None:
        from tests.sql._invariants import check_all

        url = state.get("admin_url", state["url"])
        check_all(url, state["schema"] if engine == "postgres" else None)

    if engine == "sqlite":
        broken: dict[str, dict[str, Any]] = {
            # A file inside a folder that does not exist: SQLite cannot open it.
            "unreachable": {"sql_url": "sqlite:////nonexistent-scinr-contract-dir/x/newton.db"},
        }
        unsupported = {"bad_credentials": "SQLite has no credentials to reject"}
        secret = None
    else:
        from tests.sql.conftest import _pg_base_url

        secret = "contract-not-the-password"
        container = urlsplit(_pg_base_url())
        host = f"{container.hostname}:{container.port}{container.path}"
        broken = {
            # Nothing listens on port 1.
            "unreachable": {"sql_url": f"postgresql://newton:x@127.0.0.1:1{container.path}"},
            # An unknown user of the test container.
            "bad_credentials": {"sql_url": f"postgresql://contract_no_such_user:{secret}@{host}"},
        }
        unsupported = {}
    return Backend(
        name="postgres_rls" if rls else engine,
        prepare=prepare,
        dump=dump,
        broken_configs=broken,
        drop_credentials=_sql_drop_credentials,
        settings=settings,
        secret=secret,
        unsupported=unsupported,
        after_test=after_test,
    )


BACKENDS: dict[str, Callable[[], Backend]] = {
    "neo4j": _neo4j_backend,
    "sqlite": lambda: _sql_backend("sqlite"),
    "postgres": lambda: _sql_backend("postgres"),
    "postgres_rls": lambda: _sql_backend("postgres", rls=True),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rls_selected(config: pytest.Config) -> bool:
    """Whether the run was asked for the ``sql_rls`` marker (``-m sql_rls``)."""
    return "sql_rls" in (config.getoption("markexpr") or "").replace("not sql_rls", "")


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.sql_rls) if name == "postgres_rls" else name
        for name in BACKENDS
    ],
    ids=list(BACKENDS),
)
def backend(request: pytest.FixtureRequest) -> Backend:
    """The backend under test; skips the test when it is not safe to use it."""
    if request.param == "postgres_rls" and not _rls_selected(request.config):
        pytest.skip("postgres_rls contract backend only runs with -m sql_rls")
    candidate = BACKENDS[request.param]()
    try:
        candidate.prepare()
    except BackendUnavailable as exc:
        pytest.skip(f"{candidate.name} contract backend unavailable: {exc}")
    return candidate


@pytest.fixture(autouse=True)
def _backend_after_test(request: pytest.FixtureRequest) -> Iterator[None]:
    """Run the backend's post-test checks (the SQL invariants) after every test that has one.

    Torn down after ``graph_store`` (set up later), so the store is closed by then.
    """
    if "backend" not in request.fixturenames:
        yield
        return
    candidate = request.getfixturevalue("backend")
    yield
    candidate.after_test()


@pytest.fixture
async def graph_store(backend: Backend) -> AsyncIterator[Any]:
    """The process-wide ``GraphStore`` of the (wiped and configured) backend."""
    from scinr.newton.graph import get_graph_store, reset_graph_store

    reset_graph_store()
    store = get_graph_store()
    store.schema.setup()
    yield store
    await store.aclose()
    reset_graph_store()


@pytest.fixture
def dump(backend: Backend) -> Callable[[], dict[str, Any]]:
    """Callable returning the neutral dump of the graph, volatile values normalized.

    Timestamps, engine ids and ObjectIds are replaced by placeholders (same rules as the
    golden artifacts), so two dumps of the same state compare equal.
    """

    def _dump() -> dict[str, Any]:
        _, graph = normalize_artifacts([], backend.dump(), {})
        return graph

    return _dump


@pytest.fixture
def broken_store(graph_store: Any, backend: Backend) -> Callable[[str], Any]:
    """Factory of stores configured with one of the ``broken_configs`` of the backend.

    The returned store is the process-wide one (``get_graph_store()``), as the library
    uses it; the next ``configure()`` of a test discards it.
    """
    from scinr.newton.graph import get_graph_store, reset_graph_store

    def _make(case: str) -> Any:
        if case in backend.unsupported:
            pytest.skip(f"{backend.name}: {backend.unsupported[case]}")
        if case == "no_credentials":
            _runner.configure_harness(**backend.settings())
            backend.drop_credentials()
        else:
            _runner.configure_harness(**{**backend.settings(), **backend.broken_configs[case]})
        reset_graph_store()
        return get_graph_store()

    return _make


@pytest.fixture
def env(graph_store: Any, tmp_path: Path) -> Env:
    """Environment of the golden scenarios (raw-file storage, ingestion, annotation...)."""
    return Env(tmp=tmp_path)
