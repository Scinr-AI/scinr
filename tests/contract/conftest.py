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
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.golden import _runner
from tests.golden._dump import dump_graph, normalize_artifacts
from tests.golden._scenarios import Env, prepare_script


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
    )


BACKENDS: dict[str, Callable[[], Backend]] = {"neo4j": _neo4j_backend}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def backend(request: pytest.FixtureRequest) -> Backend:
    """The backend under test; skips the test when it is not safe to use it."""
    candidate = BACKENDS[request.param]()
    try:
        candidate.prepare()
    except BackendUnavailable as exc:
        pytest.skip(f"{candidate.name} contract backend unavailable: {exc}")
    return candidate


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
        if case == "no_credentials":
            _runner.configure_harness()
            backend.drop_credentials()
        else:
            _runner.configure_harness(**backend.broken_configs[case])
        reset_graph_store()
        return get_graph_store()

    return _make


@pytest.fixture
def env(graph_store: Any, tmp_path: Path) -> Env:
    """Environment of the golden scenarios (raw-file storage, ingestion, annotation...)."""
    return Env(tmp=tmp_path)
