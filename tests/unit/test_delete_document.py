"""
tests/unit/test_delete_document.py — Unit tests for
scinr.newton.ingest.deletion.delete_document().

No real Neo4j is used. A minimal fake driver/session/transaction stack is
used instead, keyed off the literal Cypher query text (existence check vs.
the cascade's queries vs. each GC pass), mirroring the mocking style used in
tests/unit/test_ingest_one.py and tests/unit/test_pipeline_orchestration.py
(MagicMock/monkeypatch-based, no network).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scinr.newton.ingest import _cascade, _gc, deletion
from scinr.newton.ingest.deletion import GC_MAX_PASSES, collect_orphans, delete_document
from scinr.newton.results import DeletionResult, OrphanCollectionResult

# ---------------------------------------------------------------------------
# Minimal fake Neo4j driver/session/transaction stack
# ---------------------------------------------------------------------------


class _FakeResult:
    """Mimics enough of neo4j.Result for our purposes: iteration + .single()."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _FakeTx:
    """Mimics enough of neo4j.Transaction for our purposes."""

    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    def __enter__(self) -> _FakeTx:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def run(self, query: str, **params) -> _FakeResult:
        self.driver.calls.append(("tx.run", query, params))
        if query == _cascade.DOCUMENTS_DELETE_QUERY:
            return _FakeResult([{"n": len(params["keys"])}])
        if query in _ORPHAN_DELETE_QUERIES:
            # The orphans read just before are deleted (re-checked).
            deleted = [self.driver.orphans.pop(i) for i in params["ids"] if i in self.driver.orphans]
            # A write retried after its commit response was lost finds nothing.
            return _FakeResult([{"n": 0 if self.driver.lost_commit else len(deleted)}])
        raise AssertionError(f"Unexpected tx.run query: {query}")

    def commit(self) -> None:
        self.driver.calls.append(("tx.commit", "", {}))

    def rollback(self) -> None:
        self.driver.calls.append(("tx.rollback", "", {}))


class _FakeSession:
    """Mimics enough of neo4j.Session for our purposes."""

    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def run(self, query: str, **params) -> _FakeResult:
        self.driver.calls.append(("session.run", query, params))
        if "RETURN d.version AS version" in query:
            if self.driver.existence_error is not None:
                raise self.driver.existence_error
            return _FakeResult(self.driver.existence_rows)
        if "RETURN DISTINCT n.raw_file_id AS raw_file_id" in query:
            if self.driver.raw_file_ids_error is not None:
                raise self.driver.raw_file_ids_error
            # Rows omitting tenant_id stand for public nodes (the stored key).
            return _FakeResult(
                [{"tenant_id": "__public__", **row} for row in self.driver.raw_file_id_rows]
            )
        if deletion._DOCUMENT_KEYS_TAIL in query:
            return _FakeResult(self.driver.document_rows_for(params))
        if query == deletion._MARK_PENDING_QUERY:
            return _FakeResult([{"n": len(params["keys"])}])
        if query == _gc.GC_SEEDS_QUERY:
            return _FakeResult(self.driver.seed_rows)
        if query in _CANDIDATE_QUERIES:
            # One batch of GC candidates: the orphans among them, each with
            # what it points at.
            if self.driver.gc_error is not None:
                raise self.driver.gc_error
            return _FakeResult(
                [
                    {"id": element_id, "dependents": self.driver.orphans[element_id]}
                    for element_id in params["ids"]
                    if element_id in self.driver.orphans
                ]
            )
        if query in _STEP_COUNTERS:
            # One step of the cascade (auto-commit: CALL ... IN TRANSACTIONS).
            if self.driver.cascade_error is not None:
                raise self.driver.cascade_error
            return _FakeResult([{"n": self.driver.step_counts.get(_STEP_COUNTERS[query], 0)}])
        if query == _gc.GC_ENTITY_MODEL_INSTANCE_QUERY:
            return _FakeResult([{"borrados": self.driver.pop_gc_emi()}])
        if query == _gc.GC_LABELED_ENTITY_QUERY:
            return _FakeResult([{"borrados": self.driver.pop_gc_le()}])
        raise AssertionError(f"Unexpected session.run query: {query}")

    def begin_transaction(self) -> _FakeTx:
        return _FakeTx(self.driver)

    def execute_write(self, fn):
        self.driver.calls.append(("session.execute_write", "", {}))
        return fn(_FakeTx(self.driver))


class _FakeDriver:
    """Fake Neo4j driver: records every call and serves canned responses."""

    def __init__(
        self,
        existence_rows: list[dict] | None = None,
        raw_file_id_rows: list[dict] | None = None,
        step_counts: dict[str, int] | None = None,
        document_rows: list[dict] | None = None,
        gc_emi_sequence: list[int] | None = None,
        gc_le_sequence: list[int] | None = None,
        existence_error: Exception | None = None,
        raw_file_ids_error: Exception | None = None,
        cascade_error: Exception | None = None,
        interrupted: bool = False,
        seed_rows: list[dict] | None = None,
        orphans: dict[str, list] | None = None,
        gc_error: Exception | None = None,
    ) -> None:
        self.existence_rows = existence_rows if existence_rows is not None else []
        self.raw_file_id_rows = raw_file_id_rows if raw_file_id_rows is not None else []
        # What each step of the cascade reports per query, by counter name.
        self.step_counts = step_counts or {}
        # Rows of the document-set query; by default one matched Document per
        # existence row (see document_rows_for).
        self.document_rows = document_rows
        # The tenant sweep (collect_orphans(), or a delete that finds the
        # mark of an interrupted one): nodes deleted by each iteration.
        self._gc_emi_sequence = list(gc_emi_sequence if gc_emi_sequence is not None else [0])
        self._gc_le_sequence = list(gc_le_sequence if gc_le_sequence is not None else [0])
        # Whether the documents carry the mark of an interrupted delete/freeze.
        self.interrupted = interrupted
        # The scoped GC: what the ExtractionResults point at ({"id", "labeled"}
        # rows), and the orphans — element id -> [[id, labeled], ...] it points at.
        self.seed_rows = seed_rows or []
        self.orphans = dict(orphans or {})
        self.gc_error = gc_error
        self.lost_commit = False
        self.calls: list[tuple[str, str, dict]] = []
        self.closed = False
        # Optional exceptions to simulate failures at specific points.
        self.existence_error = existence_error
        self.raw_file_ids_error = raw_file_ids_error
        self.cascade_error = cascade_error

    def session(self, **kwargs) -> _FakeSession:
        return _FakeSession(self)

    def close(self) -> None:
        self.closed = True

    def document_rows_for(self, params: dict) -> list[dict]:
        if self.document_rows is not None:
            return [{"interrupted": self.interrupted, **row} for row in self.document_rows]
        return [
            {
                "path": params.get("path", f"job-doc-{index}"),
                "version": row["version"],
                "tenant_id": params["tenant_id"],
                "is_seed": True,
                "interrupted": self.interrupted,
            }
            for index, row in enumerate(self.existence_rows)
        ]

    def pop_gc_emi(self) -> int:
        return self._gc_emi_sequence.pop(0)

    def pop_gc_le(self) -> int:
        return self._gc_le_sequence.pop(0)

    # -- Convenience assertions ------------------------------------------------

    def tx_run_queries(self) -> list[str]:
        return [query for kind, query, _ in self.calls if kind == "tx.run"]

    def session_run_queries(self) -> list[str]:
        return [query for kind, query, _ in self.calls if kind == "session.run"]


# Step query -> counter it feeds (None for the steps without one).
_STEP_COUNTERS = {query: counter for counter, query in _cascade.subtree_steps()}

# Scoped GC: each batch of candidates is read (which are orphans, and what do
# they point at), then the orphans are deleted.
_CANDIDATE_QUERIES = (
    _gc.GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY,
    _gc.GC_LABELED_ENTITY_ORPHANS_QUERY,
)
_ORPHAN_DELETE_QUERIES = (
    _gc.GC_ENTITY_MODEL_INSTANCE_DELETE_QUERY,
    _gc.GC_LABELED_ENTITY_DELETE_QUERY,
)
_SWEEP_QUERIES = (_gc.GC_ENTITY_MODEL_INSTANCE_QUERY, _gc.GC_LABELED_ENTITY_QUERY)


def _keys_calls(driver: _FakeDriver) -> list[tuple[str, dict]]:
    """The cascade's selector-driven query: the document set it works on."""
    return [
        (query, params)
        for kind, query, params in driver.calls
        if kind == "session.run" and deletion._DOCUMENT_KEYS_TAIL in query
    ]


def _step_calls(driver: _FakeDriver) -> list[tuple[str, dict]]:
    return [
        (query, params)
        for kind, query, params in driver.calls
        if kind == "session.run" and query in _STEP_COUNTERS
    ]


def _document_delete_calls(driver: _FakeDriver) -> list[dict]:
    return [
        params
        for kind, query, params in driver.calls
        if kind == "tx.run" and query == _cascade.DOCUMENTS_DELETE_QUERY
    ]


def _cascade_indices(driver: _FakeDriver) -> list[int]:
    """Positions in driver.calls of everything the cascade delete runs."""
    return [
        index
        for index, (kind, query, _) in enumerate(driver.calls)
        if (kind == "session.run" and deletion._DOCUMENT_KEYS_TAIL in query)
        or (kind == "session.run" and query == deletion._MARK_PENDING_QUERY)
        or (kind == "session.run" and query == _gc.GC_SEEDS_QUERY)
        or (kind == "session.run" and query in _STEP_COUNTERS)
        or (kind == "tx.run" and query == _cascade.DOCUMENTS_DELETE_QUERY)
    ]


def _gc_calls(driver: _FakeDriver) -> list[tuple[str, dict]]:
    """Everything the GC runs: the scoped one (seeds, candidate batches) and
    the tenant sweep."""
    return [
        (query, params)
        for kind, query, params in driver.calls
        if query == _gc.GC_SEEDS_QUERY
        or query in _CANDIDATE_QUERIES
        or query in _ORPHAN_DELETE_QUERIES
        or query in _SWEEP_QUERIES
    ]


def _sweep_calls(driver: _FakeDriver) -> list[tuple[str, dict]]:
    return [(query, params) for query, params in _gc_calls(driver) if query in _SWEEP_QUERIES]


def _candidate_calls(driver: _FakeDriver) -> list[tuple[str, list[str]]]:
    """("emi" | "le", ids) of every batch of candidates checked, in order."""
    return [
        ("emi" if query == _gc.GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY else "le", params["ids"])
        for query, params in _gc_calls(driver)
        if query in _CANDIDATE_QUERIES
    ]


def _orphan_delete_calls(driver: _FakeDriver) -> list[list[str]]:
    """ids of every batch of orphans deleted, in order."""
    return [params["ids"] for query, params in _gc_calls(driver) if query in _ORPHAN_DELETE_QUERIES]


@pytest.fixture(autouse=True)
def _stub_deletion_config(monkeypatch):
    """delete_document()'s helpers call get_config().neo4j_database to pick the
    Neo4j database for each session. These unit tests never call configure(),
    so stub the get_config reference imported into the deletion module with a
    minimal fake exposing just neo4j_database.
    """
    fake_cfg = SimpleNamespace(neo4j_database="neo4j")
    monkeypatch.setattr(deletion, "get_config", lambda: fake_cfg)


@pytest.fixture
def patch_driver(monkeypatch):
    """Monkeypatch scinr.newton.ingest.deletion.get_driver to return a given fake driver."""

    def _patch(fake_driver: _FakeDriver) -> _FakeDriver:
        monkeypatch.setattr(deletion, "get_driver", lambda: fake_driver)
        return fake_driver

    return _patch


# ---------------------------------------------------------------------------
# Fake storage repositories
# ---------------------------------------------------------------------------


class _FakeRawFileRepo:
    """Fake RawFileRepository: instrumented, records calls into a shared
    driver.calls log (kind="storage.raw_delete") so ordering relative to
    Neo4j calls can be asserted the same way GC-pass ordering is asserted
    elsewhere in this file.

    ``raise_exc`` raises unconditionally on every call (used by the
    single-id fail-fast tests). ``raise_on_id`` restricts that same
    exception to only fire when ``raw_file_id == raise_on_id``, letting
    tests simulate a failure part-way through a multi-id list while still
    recording calls for ids processed before the failure.
    """

    def __init__(
        self,
        driver: _FakeDriver,
        raise_exc: Exception | None = None,
        raise_on_id: str | None = None,
    ) -> None:
        self.driver = driver
        self.raise_exc = raise_exc
        self.raise_on_id = raise_on_id
        self.deleted_ids: list[str] = []
        self.scopes: list[dict] = []

    async def delete(self, raw_file_id: str, **scope) -> None:
        self.scopes.append(scope)
        self.driver.calls.append(("storage.raw_delete", raw_file_id, {}))
        if self.raise_exc is not None and (
            self.raise_on_id is None or raw_file_id == self.raise_on_id
        ):
            raise self.raise_exc
        self.deleted_ids.append(raw_file_id)


class _FakePageRepo:
    """Fake PageRepository: instrumented the same way as _FakeRawFileRepo.

    See _FakeRawFileRepo for the ``raise_exc``/``raise_on_id`` semantics.
    """

    def __init__(
        self,
        driver: _FakeDriver,
        pages_per_id: dict[str, int] | None = None,
        raise_exc: Exception | None = None,
        raise_on_id: str | None = None,
    ) -> None:
        self.driver = driver
        self.pages_per_id = pages_per_id or {}
        self.raise_exc = raise_exc
        self.raise_on_id = raise_on_id
        self.deleted_ids: list[str] = []
        self.scopes: list[dict] = []

    async def delete_pages(self, raw_file_id: str, **scope) -> int:
        self.scopes.append(scope)
        self.driver.calls.append(("storage.page_delete", raw_file_id, {}))
        if self.raise_exc is not None and (
            self.raise_on_id is None or raw_file_id == self.raise_on_id
        ):
            raise self.raise_exc
        self.deleted_ids.append(raw_file_id)
        return self.pages_per_id.get(raw_file_id, 0)


@pytest.fixture
def patch_storage(monkeypatch):
    """Monkeypatch scinr.newton.storage.factory.get_storage to return a given
    (raw_file_repo, page_repo) pair, as imported lazily inside
    deletion._delete_storage_for_raw_file_ids().
    """

    def _patch(raw_file_repo, page_repo) -> None:
        monkeypatch.setattr(
            "scinr.newton.storage.factory.get_storage",
            lambda: (raw_file_repo, page_repo),
        )

    return _patch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeleteDocumentNotFound:
    async def test_no_matching_document_returns_found_false_with_zero_counters(
        self, patch_driver
    ):
        patch_driver(_FakeDriver(existence_rows=[]))

        result = await delete_document("some/path", version=None, tenant_id=None)

        assert isinstance(result, DeletionResult)
        assert result.found is False
        assert result.versions_deleted == []
        assert result.documents_deleted == 0
        assert result.structure_nodes_deleted == 0
        assert result.info_units_deleted == 0
        assert result.model_decisions_deleted == 0
        assert result.proposed_models_deleted == 0
        assert result.proposed_fields_deleted == 0
        assert result.extraction_results_deleted == 0
        assert result.gc_entity_model_instance_deleted == 0
        assert result.gc_entity_model_instance_passes == 0
        assert result.gc_labeled_entity_deleted == 0
        assert result.gc_labeled_entity_passes == 0
        assert result.raw_files_deleted == 0
        assert result.converted_pages_deleted == 0

    async def test_no_matching_document_does_not_run_delete_or_gc_queries(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        await delete_document("some/path", version=None, tenant_id=None)

        # Only the read-only existence check should have run.
        execute_write_calls = [c for c in fake_driver.calls if c[0] == "session.execute_write"]
        tx_run_calls = [c for c in fake_driver.calls if c[0] == "tx.run"]
        assert execute_write_calls == []
        assert tx_run_calls == []

        session_run_calls = [c for c in fake_driver.calls if c[0] == "session.run"]
        assert len(session_run_calls) == 1

    async def test_no_matching_document_never_calls_get_storage(
        self, patch_driver, monkeypatch
    ):
        """found=False must short-circuit before even the raw_file_ids
        lookup or get_storage() are reached."""
        from unittest.mock import MagicMock

        patch_driver(_FakeDriver(existence_rows=[]))
        fake_get_storage = MagicMock()
        monkeypatch.setattr("scinr.newton.storage.factory.get_storage", fake_get_storage)

        await delete_document("some/path", version=None, tenant_id=None)

        fake_get_storage.assert_not_called()

    async def test_driver_is_closed_even_when_nothing_found(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        await delete_document("some/path", tenant_id=None)

        assert fake_driver.closed is True

    async def test_empty_string_path_is_passed_through_unchanged(self, patch_driver):
        """An empty-string path is not special-cased: it is passed through
        verbatim to the existence query, and (since it matches nothing)
        results in found=False rather than raising or silently defaulting.
        """
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        result = await delete_document("", tenant_id=None)

        assert result.path == ""
        assert result.found is False

        existence_calls = [
            params for kind, query, params in fake_driver.calls if kind == "session.run"
        ]
        assert existence_calls[0] == {"tenant_id": "__public__", "path": ""}


class TestDeleteDocumentCascade:
    async def test_found_document_runs_cascade_delete_with_correct_params(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                step_counts={
                    "structure_nodes_deleted": 3,
                    "info_units_deleted": 2,
                    "model_decisions_deleted": 1,
                    "proposed_models_deleted": 1,
                    "proposed_fields_deleted": 4,
                    "extraction_results_deleted": 1,
                },
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/a", version=1, tenant_id=None)

        assert result.found is True
        assert result.versions_deleted == [1]
        assert result.documents_deleted == 1
        assert result.structure_nodes_deleted == 3
        assert result.info_units_deleted == 2
        assert result.model_decisions_deleted == 1
        assert result.proposed_models_deleted == 1
        assert result.proposed_fields_deleted == 4
        assert result.extraction_results_deleted == 1

        # The selector resolves the document set once...
        [(_, keys_params)] = _keys_calls(fake_driver)
        assert keys_params == {"tenant_id": "__public__", "path": "docs/a", "version": 1}
        # ...and every step, then the Document delete, works on those keys.
        expected = {"tenant_id": "__public__", "keys": [{"path": "docs/a", "version": 1}]}
        # The documents are marked before anything is deleted.
        first = fake_driver.calls[_cascade_indices(fake_driver)[1]]
        assert first == ("session.run", deletion._MARK_PENDING_QUERY, expected)
        assert "SET d.deletion_pending = true" in deletion._MARK_PENDING_QUERY
        steps = _step_calls(fake_driver)
        assert [query for query, _ in steps] == [query for _, query in _cascade.subtree_steps()]
        assert all(params == expected for _, params in steps)
        assert _document_delete_calls(fake_driver) == [expected]

    async def test_counters_are_summed_across_document_chunks(self, patch_driver):
        total = _cascade.DOCUMENTS_PER_QUERY + 1
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                document_rows=[
                    {"path": f"docs/{i}", "version": 1, "tenant_id": "__public__", "is_seed": True}
                    for i in range(total)
                ],
                step_counts={"structure_nodes_deleted": 2, "info_units_deleted": 5},
            )
        )

        result = await delete_document(job_id="job-1", tenant_id=None)

        # Two chunks of documents: every step runs once per chunk.
        assert len(_step_calls(fake_driver)) == 2 * len(_STEP_COUNTERS)
        assert [len(params["keys"]) for _, params in _step_calls(fake_driver)[:: len(_STEP_COUNTERS)]] == [
            _cascade.DOCUMENTS_PER_QUERY,
            1,
        ]
        assert result.structure_nodes_deleted == 4
        assert result.info_units_deleted == 10
        assert result.documents_deleted == total

    async def test_documents_are_deleted_last_and_descendants_first(self, patch_driver):
        """The Document nodes go after their whole subtree, and the matched
        ones after their descendants: whatever a failure leaves behind is
        still reachable by the same selector."""
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                document_rows=[
                    {"path": "folder", "version": 1, "tenant_id": "acme", "is_seed": True},
                    {"path": "folder/a", "version": 1, "tenant_id": "acme", "is_seed": False},
                    {"path": "folder/b", "version": 1, "tenant_id": "acme", "is_seed": False},
                ],
            )
        )

        result = await delete_document("folder", tenant_id="acme")

        [params] = _document_delete_calls(fake_driver)
        assert [key["path"] for key in params["keys"]] == ["folder/a", "folder/b", "folder"]
        delete_index = next(
            i
            for i, (kind, query, _) in enumerate(fake_driver.calls)
            if query == _cascade.DOCUMENTS_DELETE_QUERY
        )
        step_indices = [i for i, (_, query, _) in enumerate(fake_driver.calls) if query in _STEP_COUNTERS]
        assert max(step_indices) < delete_index
        assert result.documents_deleted == 3

    async def test_documents_that_cannot_be_addressed_by_key_are_left_alone(
        self, patch_driver, caplog
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                document_rows=[
                    {"path": "docs/a", "version": 1, "tenant_id": "acme", "is_seed": True},
                    {"path": "docs/a/x", "version": None, "tenant_id": "acme", "is_seed": False},
                    {"path": "docs/a/y", "version": 1, "tenant_id": "other", "is_seed": False},
                ],
            )
        )

        with caplog.at_level("WARNING", logger="scinr.newton.ingest.deletion"):
            result = await delete_document("docs/a", tenant_id="acme")

        [params] = _document_delete_calls(fake_driver)
        assert params["keys"] == [{"path": "docs/a", "version": 1}]
        assert result.documents_deleted == 1
        assert "2 Document(s)" in caplog.text

    async def test_version_none_is_omitted_from_the_bound_params(self, patch_driver):
        """When version is not given, no ``version`` condition/param is emitted
        (the WHERE stays a plain equality conjunction so the :Document indexes
        can be used) — only the tenant and ``path`` are bound.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        await delete_document("docs/c", tenant_id=None)  # version defaults to None

        existence_calls = [
            (query, params)
            for kind, query, params in fake_driver.calls
            if kind == "session.run"
        ]
        assert existence_calls[0][1] == {"tenant_id": "__public__", "path": "docs/c"}
        assert "$version" not in existence_calls[0][0]
        assert "IS NULL" not in existence_calls[0][0]

        [(_, keys_params)] = _keys_calls(fake_driver)
        assert keys_params == {"tenant_id": "__public__", "path": "docs/c"}

    async def test_explicit_version_passed_through(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 3}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        await delete_document("docs/d", version=3, tenant_id=None)

        [(_, keys_params)] = _keys_calls(fake_driver)
        assert keys_params == {"tenant_id": "__public__", "path": "docs/d", "version": 3}

    async def test_driver_is_closed_after_successful_deletion(self, patch_driver):
        """The happy path (found=True, cascade + GC all run) must still
        close the driver, not just the not-found early-return path.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        await delete_document("docs/i", version=1, tenant_id=None)

        assert fake_driver.closed is True

    async def test_existence_query_exception_propagates_and_still_closes_driver(self, patch_driver):
        """If the existence check itself raises (e.g. a Neo4j connectivity
        error), delete_document must not swallow it: it should propagate to
        the caller, while still closing the driver via the `finally` block.
        """
        fake_driver = patch_driver(
            _FakeDriver(existence_error=RuntimeError("neo4j unavailable"))
        )

        with pytest.raises(RuntimeError, match="neo4j unavailable"):
            await delete_document("docs/broken", tenant_id=None)

        assert fake_driver.closed is True
        # No cascade or GC work should have been attempted.
        assert _cascade_indices(fake_driver) == []
        assert _gc_calls(fake_driver) == []

    async def test_cascade_delete_exception_propagates_and_keeps_the_documents(
        self, patch_driver, caplog
    ):
        """A failure in a step of the cascade must propagate — not be
        swallowed into a successful DeletionResult — and stop there: the
        Document nodes are not deleted (so the same call can finish the job
        later) and nothing is collected.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                cascade_error=ValueError("cascade write failed"),
            )
        )

        with caplog.at_level("ERROR", logger="scinr.newton.ingest.deletion"):
            with pytest.raises(ValueError, match="cascade write failed"):
                await delete_document("docs/j", version=1, tenant_id=None)

        assert len(_step_calls(fake_driver)) == 1
        assert _document_delete_calls(fake_driver) == []
        assert _candidate_calls(fake_driver) == []
        assert _sweep_calls(fake_driver) == []
        assert "calling delete_document() again" in caplog.text
        assert fake_driver.closed is True

    async def test_structure_less_documents_are_deleted_too(self, patch_driver):
        """A Document with no :StructureNode of its own (every folder-parent
        Document, plus any leaf never fully processed) must still be deleted:
        the Documents are deleted by key, whatever the subtree steps found.
        (An earlier single-query cascade silently skipped them.)
        """
        fake_driver = patch_driver(
            _FakeDriver(existence_rows=[{"version": 1}], gc_emi_sequence=[0], gc_le_sequence=[0])
        )

        result = await delete_document("docs/structureless", version=1, tenant_id=None)

        assert all(count == 0 for count in fake_driver.step_counts.values())
        assert _document_delete_calls(fake_driver) == [
            {"tenant_id": "__public__", "keys": [{"path": "docs/structureless", "version": 1}]}
        ]
        assert "HAS_STRUCTURE" not in _cascade.DOCUMENTS_DELETE_QUERY
        assert result.documents_deleted == 1


class TestDeleteDocumentGarbageCollection:
    """The GC is scoped to the deletion: it only checks what the deleted
    ExtractionResults pointed at, and what the orphans among those pointed
    at — never the whole tenant."""

    async def test_orphans_are_collected_round_after_round(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                seed_rows=[
                    {"id": "mi-1", "labeled": False},
                    {"id": "mi-shared", "labeled": False},
                    {"id": "le-shared", "labeled": True},
                ],
                # mi-shared / le-shared are still in use: not orphans.
                orphans={
                    "mi-1": [["entity-1", False], ["le-1", True]],
                    "entity-1": [],
                    "le-1": [["le-2", True]],
                    "le-2": [],
                },
            )
        )

        result = await delete_document("docs/e", version=1, tenant_id="acme")

        assert _candidate_calls(fake_driver) == [
            ("emi", ["mi-1", "mi-shared"]),
            ("le", ["le-1", "le-shared"]),
            ("emi", ["entity-1"]),
            ("le", ["le-2"]),
        ]
        # Only the orphans are deleted, each batch in its own write transaction.
        assert _orphan_delete_calls(fake_driver) == [["mi-1"], ["le-1"], ["entity-1"], ["le-2"]]
        kinds = [kind for kind, query, _ in fake_driver.calls if query in _ORPHAN_DELETE_QUERIES]
        assert kinds == ["tx.run"] * 4
        assert result.gc_entity_model_instance_deleted == 2
        assert result.gc_entity_model_instance_passes == 2
        assert result.gc_labeled_entity_deleted == 2
        assert result.gc_labeled_entity_passes == 2
        assert fake_driver.orphans == {}
        assert _sweep_calls(fake_driver) == []

    async def test_nothing_to_collect_runs_no_gc_round(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[{"version": 1}]))

        result = await delete_document("docs/e", version=1, tenant_id=None)

        assert [query for query, _ in _gc_calls(fake_driver)] == [_gc.GC_SEEDS_QUERY]
        assert result.gc_entity_model_instance_deleted == 0
        assert result.gc_entity_model_instance_passes == 0
        assert result.gc_labeled_entity_deleted == 0
        assert result.gc_labeled_entity_passes == 0

    async def test_candidates_are_read_before_the_extraction_results_are_deleted(
        self, patch_driver
    ):
        """Once an ExtractionResult is gone nothing says what it pointed at:
        each chunk of documents is asked before its steps run."""
        total = _cascade.DOCUMENTS_PER_QUERY + 1
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                document_rows=[
                    {"path": f"folder/{i}", "version": 1, "tenant_id": "acme", "is_seed": i == 0}
                    for i in range(total)
                ],
            )
        )

        await delete_document("folder", tenant_id="acme")

        order = [
            "seeds" if query == _gc.GC_SEEDS_QUERY else "step"
            for _, query, _ in fake_driver.calls
            if query == _gc.GC_SEEDS_QUERY or query in _STEP_COUNTERS
        ]
        steps = ["step"] * len(_STEP_COUNTERS)
        assert order == ["seeds", *steps, "seeds", *steps]
        seed_keys = [p["keys"] for q, p in _gc_calls(fake_driver) if q == _gc.GC_SEEDS_QUERY]
        assert [len(keys) for keys in seed_keys] == [_cascade.DOCUMENTS_PER_QUERY, 1]

    async def test_orphans_are_collected_before_the_documents_are_deleted(self, patch_driver):
        """The Documents carry the mark that makes an interrupted run
        detectable: they go only once the GC is done."""
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                seed_rows=[{"id": "mi-1", "labeled": False}],
                orphans={"mi-1": []},
            )
        )

        await delete_document("docs/e", version=1, tenant_id=None)

        queries = [query for _, query, _ in fake_driver.calls]
        assert queries.index(_gc.GC_ENTITY_MODEL_INSTANCE_DELETE_QUERY) < queries.index(
            _cascade.DOCUMENTS_DELETE_QUERY
        )

    async def test_candidates_go_in_bounded_transactions(self, patch_driver):
        total = 2 * _cascade.DELETE_BATCH_ROWS + 1
        ids = [f"mi-{i:05d}" for i in range(total)]
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                seed_rows=[{"id": element_id, "labeled": False} for element_id in ids],
                orphans={element_id: [] for element_id in ids},
            )
        )

        result = await delete_document("docs/e", version=1, tenant_id=None)

        sizes = [_cascade.DELETE_BATCH_ROWS, _cascade.DELETE_BATCH_ROWS, 1]
        assert [len(batch) for _, batch in _candidate_calls(fake_driver)] == sizes
        assert [len(batch) for batch in _orphan_delete_calls(fake_driver)] == sizes
        assert result.gc_entity_model_instance_deleted == total
        assert result.gc_entity_model_instance_passes == 1

    async def test_candidates_are_collected_early_once_enough_pile_up(
        self, patch_driver, monkeypatch
    ):
        """The candidates are held in memory: past GC_FLUSH_CANDIDATES they
        are collected at the end of the chunk instead of at the end."""
        monkeypatch.setattr(_gc, "GC_FLUSH_CANDIDATES", 1)
        total = _cascade.DOCUMENTS_PER_QUERY + 1
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                document_rows=[
                    {"path": f"folder/{i}", "version": 1, "tenant_id": "acme", "is_seed": i == 0}
                    for i in range(total)
                ],
                seed_rows=[{"id": "mi-1", "labeled": False}],
            )
        )

        await delete_document("folder", tenant_id="acme")

        order = [
            "seeds" if query == _gc.GC_SEEDS_QUERY else "candidates"
            for query, _ in _gc_calls(fake_driver)
        ]
        assert order == ["seeds", "candidates", "seeds", "candidates"]

    async def test_a_lost_commit_does_not_lose_the_next_candidates(self, patch_driver):
        """A delete retried after its commit response was lost finds the
        orphans already gone. What they pointed at was read beforehand, so it
        is still checked; only the counter misses them."""
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                seed_rows=[{"id": "mi-1", "labeled": False}],
                orphans={"mi-1": [["le-1", True]], "le-1": []},
            )
        )
        fake_driver.lost_commit = True

        result = await delete_document("docs/e", version=1, tenant_id=None)

        assert _candidate_calls(fake_driver) == [("emi", ["mi-1"]), ("le", ["le-1"])]
        assert fake_driver.orphans == {}
        assert result.gc_entity_model_instance_deleted == 0  # a lower bound
        assert result.gc_entity_model_instance_passes == 1

    async def test_gc_failure_propagates_and_keeps_the_documents(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                seed_rows=[{"id": "mi-1", "labeled": False}],
                gc_error=ValueError("gc failed"),
            )
        )

        with pytest.raises(ValueError, match="gc failed"):
            await delete_document("docs/e", version=1, tenant_id=None)

        assert _document_delete_calls(fake_driver) == []
        assert fake_driver.closed is True

    def test_candidate_queries_never_start_from_the_tenant_indexes(self):
        """A plain ``mi:Entity`` / ``c.tenant_id`` predicate lets the planner
        start from the tenant's indexes — the scan of the whole tenant the
        scoped GC exists to avoid. The candidates are looked up by id, and
        their labels tested through ``labels()``."""
        for query in (*_CANDIDATE_QUERIES, *_ORPHAN_DELETE_QUERIES):
            assert query.lstrip().startswith("MATCH (mi) WHERE elementId(mi) IN $ids")
            assert "labels(mi)" in query
            assert "mi.tenant_id = $tenant_id" in query
            assert "MATCH (mi:" not in query
        for read, delete in zip(_CANDIDATE_QUERIES, _ORPHAN_DELETE_QUERIES, strict=True):
            # The delete re-checks exactly what the read found.
            assert read.split("RETURN")[0] == delete.split("DETACH DELETE")[0]
            assert "DETACH DELETE" not in read
            assert "[(mi)-->(c) WHERE" in read and "labels(c)" in read
            assert delete.rstrip().endswith("DETACH DELETE mi\nRETURN count(mi) AS n")
        assert "[*1.." in _gc.GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY
        assert "NOT EXISTS { (mi)<--() }" in _gc.GC_LABELED_ENTITY_ORPHANS_QUERY
        assert "MATCH (er)-->(c)" in _gc.GC_SEEDS_QUERY
        assert "labels(c)" in _gc.GC_SEEDS_QUERY


class TestDeleteDocumentResumed:
    """The candidates of the scoped GC only live in the process that gathered
    them. A delete that finds the mark of an interrupted delete or freeze
    sweeps the whole tenant instead."""

    async def test_interrupted_documents_trigger_the_tenant_sweep(self, patch_driver, caplog):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                interrupted=True,
                seed_rows=[{"id": "mi-1", "labeled": False}],
                gc_emi_sequence=[5, 0],
                gc_le_sequence=[2, 0],
            )
        )

        with caplog.at_level("WARNING", logger="scinr.newton.ingest.deletion"):
            result = await delete_document("docs/e", version=1, tenant_id="acme")

        assert [query for query, _ in _gc_calls(fake_driver)] == [
            _gc.GC_ENTITY_MODEL_INSTANCE_QUERY,
            _gc.GC_ENTITY_MODEL_INSTANCE_QUERY,
            _gc.GC_LABELED_ENTITY_QUERY,
            _gc.GC_LABELED_ENTITY_QUERY,
        ]
        assert all(params == {"tenant_id": "acme"} for _, params in _sweep_calls(fake_driver))
        assert result.gc_entity_model_instance_deleted == 5
        assert result.gc_labeled_entity_deleted == 2
        assert result.documents_deleted == 1
        assert "was interrupted" in caplog.text

    async def test_the_sweep_runs_before_the_documents_are_deleted(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[{"version": 1}], interrupted=True))

        await delete_document("docs/e", version=1, tenant_id=None)

        queries = [query for _, query, _ in fake_driver.calls]
        assert queries.index(_gc.GC_LABELED_ENTITY_QUERY) < queries.index(
            _cascade.DOCUMENTS_DELETE_QUERY
        )

    def test_the_mark_of_an_interrupted_freeze_counts_too(self):
        assert (
            "coalesce(doc.deletion_pending, doc.frozen_cleanup_pending, false) AS interrupted"
            in deletion._DOCUMENT_KEYS_TAIL
        )

    async def test_gc_pass_stops_at_first_zero(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                interrupted=True,
                gc_emi_sequence=[5, 2, 0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/e", version=1, tenant_id=None)

        assert result.gc_entity_model_instance_deleted == 7
        assert result.gc_entity_model_instance_passes == 3

        emi_run_count = sum(
            1 for query, _ in _sweep_calls(fake_driver) if query == _gc.GC_ENTITY_MODEL_INSTANCE_QUERY
        )
        assert emi_run_count == 3

    async def test_gc_pass_caps_at_gc_max_passes_when_never_reaching_zero(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                interrupted=True,
                gc_emi_sequence=[1] * GC_MAX_PASSES,
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/f", version=1, tenant_id=None)

        assert result.gc_entity_model_instance_passes == GC_MAX_PASSES
        assert result.gc_entity_model_instance_deleted == GC_MAX_PASSES

        emi_run_count = sum(
            1 for query, _ in _sweep_calls(fake_driver) if query == _gc.GC_ENTITY_MODEL_INSTANCE_QUERY
        )
        assert emi_run_count == GC_MAX_PASSES

    async def test_labeled_entity_pass_runs_only_after_entity_model_instance_pass_completes(
        self, patch_driver
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                interrupted=True,
                gc_emi_sequence=[3, 1, 0],
                gc_le_sequence=[2, 0],
            )
        )

        result = await delete_document("docs/g", version=1, tenant_id=None)

        assert result.gc_entity_model_instance_deleted == 4
        assert result.gc_entity_model_instance_passes == 3
        assert result.gc_labeled_entity_deleted == 2
        assert result.gc_labeled_entity_passes == 2

        # Order: all Entity|ModelInstance passes must precede all
        # LabeledEntity passes.
        labels = [
            "emi" if query == _gc.GC_ENTITY_MODEL_INSTANCE_QUERY else "le"
            for query, _ in _sweep_calls(fake_driver)
        ]
        assert labels == ["emi", "emi", "emi", "le", "le"]

    async def test_gc_second_pass_independent_max_passes(self, patch_driver):
        patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                interrupted=True,
                gc_emi_sequence=[0],
                gc_le_sequence=[1] * GC_MAX_PASSES,
            )
        )

        result = await delete_document("docs/h", version=1, tenant_id=None)

        assert result.gc_entity_model_instance_passes == 1
        assert result.gc_labeled_entity_passes == GC_MAX_PASSES
        assert result.gc_labeled_entity_deleted == GC_MAX_PASSES


class TestCollectOrphans:
    """collect_orphans(): the tenant sweep as a maintenance call."""

    async def test_sweeps_the_tenant_and_reports_the_counters(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(gc_emi_sequence=[4, 0], gc_le_sequence=[3, 1, 0]))

        result = await collect_orphans(tenant_id="acme")

        assert result == OrphanCollectionResult(
            tenant_id="acme",
            gc_entity_model_instance_deleted=4,
            gc_entity_model_instance_passes=2,
            gc_labeled_entity_deleted=4,
            gc_labeled_entity_passes=3,
        )
        assert [(kind, query) for kind, query, _ in fake_driver.calls] == [
            ("session.run", _gc.GC_ENTITY_MODEL_INSTANCE_QUERY),
            ("session.run", _gc.GC_ENTITY_MODEL_INSTANCE_QUERY),
            ("session.run", _gc.GC_LABELED_ENTITY_QUERY),
            ("session.run", _gc.GC_LABELED_ENTITY_QUERY),
            ("session.run", _gc.GC_LABELED_ENTITY_QUERY),
        ]
        assert all(params == {"tenant_id": "acme"} for _, _, params in fake_driver.calls)
        assert fake_driver.closed is True

    async def test_none_means_the_public_documents(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver())

        result = await collect_orphans(tenant_id=None)

        assert all(params == {"tenant_id": "__public__"} for _, _, params in fake_driver.calls)
        assert result.tenant_id is None

    async def test_tenant_is_mandatory(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver())
        with pytest.raises(TypeError):
            await collect_orphans()  # type: ignore[call-arg]
        with pytest.raises(ValueError):
            await collect_orphans(tenant_id="")
        assert fake_driver.calls == []

    async def test_driver_is_closed_when_the_sweep_fails(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(gc_emi_sequence=[]))

        with pytest.raises(IndexError):
            await collect_orphans(tenant_id="acme")

        assert fake_driver.closed is True

    def test_is_exported(self):
        import scinr.newton

        assert scinr.newton.collect_orphans is collect_orphans
        assert scinr.newton.OrphanCollectionResult is OrphanCollectionResult


class TestDeleteDocumentStorageCleanup:
    """Tests for the pre-cascade documental storage cleanup step."""

    async def test_deletes_storage_for_each_distinct_raw_file_id_before_cascade(
        self, patch_driver, patch_storage
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[{"raw_file_id": "rid1"}, {"raw_file_id": "rid2"}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver)
        page_repo = _FakePageRepo(fake_driver, pages_per_id={"rid1": 3, "rid2": 5})
        patch_storage(raw_repo, page_repo)

        result = await delete_document("docs/k", version=1, tenant_id=None)

        assert raw_repo.deleted_ids == ["rid1", "rid2"]
        assert page_repo.deleted_ids == ["rid1", "rid2"]
        assert result.raw_files_deleted == 2
        assert result.converted_pages_deleted == 8

        # All storage deletion calls must complete before the cascade
        # delete ever runs.
        kinds_in_order = [kind for kind, _, _ in fake_driver.calls]
        storage_indices = [
            i
            for i, k in enumerate(kinds_in_order)
            if k in ("storage.raw_delete", "storage.page_delete")
        ]
        cascade_indices = _cascade_indices(fake_driver)
        assert storage_indices, "expected storage deletion calls to have happened"
        assert cascade_indices, "expected the cascade delete to have run"
        assert max(storage_indices) < min(cascade_indices)

    async def test_storage_deletes_are_scoped_to_the_tenant_of_each_node(
        self, patch_driver, patch_storage
    ):
        """A raw_file_id is deleted only within the tenant of the graph node
        carrying it: a forged id pointing at another tenant's upload deletes
        nothing in storage. No user / job filter is applied here."""
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[{"raw_file_id": "rid1", "tenant_id": "acme"}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver)
        page_repo = _FakePageRepo(fake_driver)
        patch_storage(raw_repo, page_repo)

        await delete_document("docs/m", version=1, tenant_id="acme", created_by_user_id="u1")

        assert raw_repo.scopes == [{"tenant_id": "acme"}]
        assert page_repo.scopes == [{"tenant_id": "acme"}]

    async def test_documents_with_empty_raw_file_id_are_excluded(
        self, patch_driver, patch_storage
    ):
        """The Cypher query itself filters out empty raw_file_id values
        (folders / storage_backend='none' documents); this test pins that
        only the non-empty ids returned by the query trigger storage
        deletion calls.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                # Simulates the server-side WHERE n.raw_file_id <> '' filter:
                # the folder-parent Document (empty raw_file_id) never
                # appears in these rows.
                raw_file_id_rows=[{"raw_file_id": "rid1"}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver)
        page_repo = _FakePageRepo(fake_driver)
        patch_storage(raw_repo, page_repo)

        result = await delete_document("docs/l", version=1, tenant_id=None)

        assert raw_repo.deleted_ids == ["rid1"]
        assert page_repo.deleted_ids == ["rid1"]
        assert "" not in raw_repo.deleted_ids
        assert result.raw_files_deleted == 1

    async def test_page_repo_exception_propagates_and_skips_cascade_delete(
        self, patch_driver, patch_storage
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[{"raw_file_id": "rid1"}],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver)
        page_repo = _FakePageRepo(fake_driver, raise_exc=RuntimeError("mongo down"))
        patch_storage(raw_repo, page_repo)

        with pytest.raises(RuntimeError, match="mongo down"):
            await delete_document("docs/m", version=1, tenant_id=None)

        assert fake_driver.closed is True
        assert _cascade_indices(fake_driver) == []
        # raw_repo.delete() must never have been reached for this id since
        # delete_pages() is called first and raised.
        assert raw_repo.deleted_ids == []

    async def test_raw_file_repo_exception_propagates_and_skips_cascade_delete(
        self, patch_driver, patch_storage
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[{"raw_file_id": "rid1"}],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver, raise_exc=ValueError("gridfs error"))
        page_repo = _FakePageRepo(fake_driver, pages_per_id={"rid1": 2})
        patch_storage(raw_repo, page_repo)

        with pytest.raises(ValueError, match="gridfs error"):
            await delete_document("docs/n", version=1, tenant_id=None)

        assert fake_driver.closed is True
        assert _cascade_indices(fake_driver) == []
        # delete_pages() should have run (and returned normally) before
        # delete() raised.
        assert page_repo.deleted_ids == ["rid1"]

    async def test_no_raw_file_ids_never_calls_get_storage(self, patch_driver, monkeypatch):
        """When every Document in scope has an empty raw_file_id (so the
        raw_file_ids query returns no rows), get_storage() must never be
        called at all.
        """
        from unittest.mock import MagicMock

        patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )
        fake_get_storage = MagicMock()
        monkeypatch.setattr("scinr.newton.storage.factory.get_storage", fake_get_storage)

        result = await delete_document("docs/o", version=1, tenant_id=None)

        fake_get_storage.assert_not_called()
        assert result.raw_files_deleted == 0
        assert result.converted_pages_deleted == 0

    async def test_converted_pages_deleted_sums_across_multiple_raw_file_ids(
        self, patch_driver, patch_storage
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[
                    {"raw_file_id": "rid1"},
                    {"raw_file_id": "rid2"},
                    {"raw_file_id": "rid3"},
                ],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver)
        page_repo = _FakePageRepo(fake_driver, pages_per_id={"rid1": 1, "rid2": 0, "rid3": 4})
        patch_storage(raw_repo, page_repo)

        result = await delete_document("docs/p", version=1, tenant_id=None)

        assert result.raw_files_deleted == 3
        assert result.converted_pages_deleted == 5

    async def test_raw_file_ids_query_receives_correct_params(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 2}],
                raw_file_id_rows=[],
            )
        )

        await delete_document("docs/q", version=2, tenant_id=None)

        raw_file_id_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "session.run" and "RETURN DISTINCT n.raw_file_id AS raw_file_id" in query
        ]
        assert len(raw_file_id_calls) == 1
        assert raw_file_id_calls[0] == {"tenant_id": "__public__", "path": "docs/q", "version": 2}

    async def test_page_repo_failure_mid_list_stops_before_processing_later_ids(
        self, patch_driver, patch_storage
    ):
        """With three raw_file_ids, a delete_pages() failure on the second
        one must stop the loop immediately: the third id's delete_pages()
        and delete() must never be called, and the second id's raw_file_repo
        .delete() (which runs after delete_pages() for that same id) must
        never be called either, since the exception happens first.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[
                    {"raw_file_id": "rid1"},
                    {"raw_file_id": "rid2"},
                    {"raw_file_id": "rid3"},
                ],
            )
        )
        raw_repo = _FakeRawFileRepo(fake_driver)
        page_repo = _FakePageRepo(
            fake_driver,
            pages_per_id={"rid1": 1, "rid3": 9},
            raise_exc=RuntimeError("mongo down on rid2"),
            raise_on_id="rid2",
        )
        patch_storage(raw_repo, page_repo)

        with pytest.raises(RuntimeError, match="mongo down on rid2"):
            await delete_document("docs/r", version=1, tenant_id=None)

        assert fake_driver.closed is True
        # rid1 was fully processed (both page delete and raw delete).
        assert raw_repo.deleted_ids == ["rid1"]
        # page_repo saw rid1 (succeeded) then rid2 (raised) — rid3 never reached.
        page_delete_calls = [
            rid for kind, rid, _ in fake_driver.calls if kind == "storage.page_delete"
        ]
        assert page_delete_calls == ["rid1", "rid2"]
        raw_delete_calls = [
            rid for kind, rid, _ in fake_driver.calls if kind == "storage.raw_delete"
        ]
        # rid2's raw_file_repo.delete() must never be reached: delete_pages()
        # raised before raw_file_repo.delete(rid2) could run.
        assert raw_delete_calls == ["rid1"]
        # No cascade delete should have run.
        assert _cascade_indices(fake_driver) == []

    async def test_raw_file_repo_failure_mid_list_stops_before_processing_later_ids(
        self, patch_driver, patch_storage
    ):
        """With three raw_file_ids, a raw_file_repo.delete() failure on the
        second one (after its own delete_pages() succeeded) must stop the
        loop before the third id is ever touched.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                raw_file_id_rows=[
                    {"raw_file_id": "rid1"},
                    {"raw_file_id": "rid2"},
                    {"raw_file_id": "rid3"},
                ],
            )
        )
        raw_repo = _FakeRawFileRepo(
            fake_driver,
            raise_exc=ValueError("gridfs error on rid2"),
            raise_on_id="rid2",
        )
        page_repo = _FakePageRepo(
            fake_driver, pages_per_id={"rid1": 1, "rid2": 2, "rid3": 9}
        )
        patch_storage(raw_repo, page_repo)

        with pytest.raises(ValueError, match="gridfs error on rid2"):
            await delete_document("docs/s", version=1, tenant_id=None)

        assert fake_driver.closed is True
        # delete_pages() ran for rid1 and rid2, but never for rid3.
        page_delete_calls = [
            rid for kind, rid, _ in fake_driver.calls if kind == "storage.page_delete"
        ]
        assert page_delete_calls == ["rid1", "rid2"]
        # raw_file_repo.delete() ran for rid1 (succeeded) and rid2 (raised),
        # but never for rid3.
        raw_delete_calls = [
            rid for kind, rid, _ in fake_driver.calls if kind == "storage.raw_delete"
        ]
        assert raw_delete_calls == ["rid1", "rid2"]
        assert raw_repo.deleted_ids == ["rid1"]
        assert _cascade_indices(fake_driver) == []


class TestDeleteDocumentSelectorValidation:
    async def test_neither_path_nor_job_id_raises_value_error(self, patch_driver):
        patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError, match="exactly one of 'path' or 'job_id'"):
            await delete_document(tenant_id=None)

    async def test_both_path_and_job_id_raises_value_error(self, patch_driver):
        patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError, match="exactly one of 'path' or 'job_id'"):
            await delete_document("docs/a", job_id="job-1", tenant_id=None)

    async def test_value_error_raised_before_any_neo4j_call(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError):
            await delete_document(tenant_id=None)
        assert fake_driver.calls == []


class TestDeleteDocumentByJobId:
    async def test_job_id_selector_binds_only_job_id_and_echoes_it(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}, {"version": 1}, {"version": 2}],
                step_counts={"structure_nodes_deleted": 9},
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document(job_id="job-xyz", tenant_id=None)

        assert result.found is True
        assert result.path is None
        assert result.job_id == "job-xyz"
        assert result.documents_deleted == 3
        # Only the tenant (always) and the supplied selector are emitted — no
        # path/version conditions, and no `IS NULL` disjunction (so the
        # indexes can be used).
        existence_calls = [
            (query, params)
            for kind, query, params in fake_driver.calls
            if kind == "session.run"
        ]
        assert existence_calls[0][1] == {"tenant_id": "__public__", "job_id": ["job-xyz"]}
        assert "d.job_id IN $job_id" in existence_calls[0][0]
        assert "IS NULL" not in existence_calls[0][0]
        [(_, keys_params)] = _keys_calls(fake_driver)
        assert keys_params == {"tenant_id": "__public__", "job_id": ["job-xyz"]}
        assert result.structure_nodes_deleted == 9

    async def test_tenant_and_user_filters_are_forwarded_in_path_mode(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        await delete_document(
            "docs/a",
            version=1,
            tenant_id="tenant-7",
            created_by_user_id="user-42",
        )

        [(query, params)] = _keys_calls(fake_driver)
        assert params == {
            "path": "docs/a",
            "version": 1,
            "tenant_id": "tenant-7",
            "created_by_user_id": ["user-42"],
        }
        # job_id was not supplied → no job_id condition at all
        assert "job_id" not in query
        for cond in (
            "d.path = $path",
            "d.version = $version",
            "d.tenant_id = $tenant_id",
            "d.created_by_user_id IN $created_by_user_id",
        ):
            assert cond in query

    async def test_job_id_no_match_returns_found_false_and_echoes_selector(self, patch_driver):
        patch_driver(_FakeDriver(existence_rows=[]))

        result = await delete_document(job_id="missing-job", tenant_id="tenant-1")

        assert result.found is False
        assert result.path is None
        assert result.job_id == "missing-job"
        assert result.tenant_id == "tenant-1"
        assert result.documents_deleted == 0


class TestDeleteDocumentTenantScope:
    """The tenant is a mandatory, always-applied scope (see WP6 of
    plans/multitenancy-document-identity-plan.md)."""

    async def test_omitting_tenant_id_is_a_type_error(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(TypeError):
            await delete_document("docs/a")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            await delete_document(job_id="job-1")  # type: ignore[call-arg]
        assert fake_driver.calls == []

    async def test_tenant_none_targets_public_documents(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        result = await delete_document("docs/a", tenant_id=None)

        query, params = next(
            (q, p) for kind, q, p in fake_driver.calls if kind == "session.run"
        )
        assert "d.tenant_id = $tenant_id" in query
        assert params["tenant_id"] == "__public__"
        assert result.tenant_id is None

    @pytest.mark.parametrize("selector", [{"path": "docs/a"}, {"job_id": "job-1"}])
    async def test_tenant_filter_is_in_every_query(self, patch_driver, selector):
        fake_driver = patch_driver(
            _FakeDriver(existence_rows=[{"version": 1}], gc_emi_sequence=[0], gc_le_sequence=[0])
        )

        await delete_document(**selector, tenant_id="acme")

        doc_queries = [
            (q, p) for kind, q, p in fake_driver.calls
            if kind in ("session.run", "tx.run") and "MATCH (d:Document)" in q
        ]
        # existence check, raw_file_id lookup, the cascade's document set
        assert len(doc_queries) == 3
        for q, p in doc_queries:
            assert "d.tenant_id = $tenant_id" in q
            assert p["tenant_id"] == "acme"
        # The cascade itself (mark, GC seeds, steps, Document delete) addresses
        # the documents by key, tenant included.
        by_key = [(q, p) for _, q, p in fake_driver.calls if "UNWIND $keys AS k" in q]
        assert len(by_key) == len(_STEP_COUNTERS) + 3
        for q, p in by_key:
            assert "MATCH (d:Document {tenant_id: $tenant_id, path: k.path, version: k.version})" in q
            assert p["tenant_id"] == "acme"

    async def test_gc_is_scoped_to_the_tenant(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                seed_rows=[{"id": "mi-1", "labeled": False}, {"id": "le-1", "labeled": True}],
            )
        )

        await delete_document("docs/a", tenant_id="acme")

        gc_calls = _gc_calls(fake_driver)
        assert len(gc_calls) == 3  # seeds, one batch of each kind
        for q, p in gc_calls:
            assert "tenant_id = $tenant_id" in q
            assert p["tenant_id"] == "acme"
        for q in (*_CANDIDATE_QUERIES, *_ORPHAN_DELETE_QUERIES):
            assert "mi.tenant_id = $tenant_id" in q
        for q in _SWEEP_QUERIES:
            assert "mi.tenant_id = $tenant_id" in q

    async def test_empty_tenant_is_rejected_before_neo4j(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError):
            await delete_document("docs/a", tenant_id="")
        assert fake_driver.calls == []

    async def test_public_sentinel_is_the_same_as_none(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        await delete_document("docs/a", tenant_id="__public__")
        query, params = next((q, p) for kind, q, p in fake_driver.calls if kind == "session.run")
        assert params["tenant_id"] == "__public__"

    async def test_job_and_user_lists_use_in(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        await delete_document(job_id=["j1", "j2"], created_by_user_id=("u1",), tenant_id="acme")
        query, params = next((q, p) for kind, q, p in fake_driver.calls if kind == "session.run")
        assert "d.job_id IN $job_id" in query
        assert "d.created_by_user_id IN $created_by_user_id" in query
        assert params["job_id"] == ["j1", "j2"]
        assert params["created_by_user_id"] == ["u1"]

    @pytest.mark.parametrize("kw", [{"job_id": []}, {"path": "p", "created_by_user_id": []}])
    async def test_empty_lists_are_rejected(self, patch_driver, kw):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError, match="empty list"):
            await delete_document(tenant_id="acme", **kw)
        assert fake_driver.calls == []


class TestBuildDocMatch:
    """Unit tests for deletion._build_doc_match() — the dynamic WHERE builder."""

    def test_only_supplied_filters_become_conditions(self):
        match, params = deletion._build_doc_match(
            {"path": None, "version": None, "tenant_id": "__public__",
             "created_by_user_id": None, "job_id": ["j1"]}
        )
        assert params == {"tenant_id": "__public__", "job_id": ["j1"]}
        assert (
            match.strip()
            == "MATCH (d:Document)\nWHERE d.tenant_id = $tenant_id AND d.job_id IN $job_id"
        )

    def test_multiple_filters_are_anded_in_fixed_order(self):
        match, params = deletion._build_doc_match(
            {"path": "p", "version": 2, "tenant_id": "t",
             "created_by_user_id": ["u", "v"], "job_id": None}
        )
        assert params == {"tenant_id": "t", "path": "p", "version": 2, "created_by_user_id": ["u", "v"]}
        assert (
            "WHERE d.tenant_id = $tenant_id AND d.path = $path AND d.version = $version"
            " AND d.created_by_user_id IN $created_by_user_id"
            in match
        )

    def test_tenant_is_mandatory(self):
        with pytest.raises(ValueError):
            deletion._build_doc_match({"path": "p", "tenant_id": None})
        with pytest.raises(ValueError):
            deletion._build_doc_match({"path": "p"})

    def test_no_is_null_disjunction_is_ever_emitted(self):
        match, _ = deletion._build_doc_match({"path": "p", "tenant_id": "t"})
        assert "IS NULL" not in match
        assert " OR " not in match

    def test_falsy_but_non_none_values_are_kept(self):
        match, params = deletion._build_doc_match({"path": "", "version": 0, "tenant_id": "t"})
        assert params == {"tenant_id": "t", "path": "", "version": 0}
        assert "d.path = $path" in match
        assert "d.version = $version" in match
