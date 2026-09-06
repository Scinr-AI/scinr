"""
tests/unit/test_delete_document.py — Unit tests for
scinr.newton.ingest.deletion.delete_document().

No real Neo4j is used. A minimal fake driver/session/transaction stack is
used instead, keyed off the literal Cypher query text (existence check vs.
cascade delete vs. each GC pass), mirroring the mocking style used in
tests/unit/test_ingest_one.py and tests/unit/test_pipeline_orchestration.py
(MagicMock/monkeypatch-based, no network).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scinr.newton.ingest import deletion
from scinr.newton.ingest.deletion import GC_MAX_PASSES, delete_document
from scinr.newton.results import DeletionResult

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
        if "documents_deleted" in query:
            # Cascade delete query.
            if self.driver.cascade_error is not None:
                raise self.driver.cascade_error
            return _FakeResult(self.driver.pop_cascade_rows())
        if "Entity|ModelInstance" in query:
            return _FakeResult([{"borrados": self.driver.pop_gc_emi()}])
        if "LabeledEntity" in query:
            return _FakeResult([{"borrados": self.driver.pop_gc_le()}])
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
            return _FakeResult(self.driver.raw_file_id_rows)
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
        cascade_rows: list[dict] | None = None,
        gc_emi_sequence: list[int] | None = None,
        gc_le_sequence: list[int] | None = None,
        existence_error: Exception | None = None,
        raw_file_ids_error: Exception | None = None,
        cascade_error: Exception | None = None,
    ) -> None:
        self.existence_rows = existence_rows if existence_rows is not None else []
        self.raw_file_id_rows = raw_file_id_rows if raw_file_id_rows is not None else []
        self._cascade_rows = cascade_rows if cascade_rows is not None else [
            {
                "documents_deleted": 0,
                "structure_nodes_deleted": 0,
                "info_units_deleted": 0,
                "model_decisions_deleted": 0,
                "proposed_models_deleted": 0,
                "proposed_fields_deleted": 0,
                "extraction_results_deleted": 0,
            }
        ]
        self._gc_emi_sequence = list(gc_emi_sequence if gc_emi_sequence is not None else [0])
        self._gc_le_sequence = list(gc_le_sequence if gc_le_sequence is not None else [0])
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

    def pop_cascade_rows(self) -> list[dict]:
        return self._cascade_rows

    def pop_gc_emi(self) -> int:
        return self._gc_emi_sequence.pop(0)

    def pop_gc_le(self) -> int:
        return self._gc_le_sequence.pop(0)

    # -- Convenience assertions ------------------------------------------------

    def tx_run_queries(self) -> list[str]:
        return [query for kind, query, _ in self.calls if kind == "tx.run"]

    def session_run_queries(self) -> list[str]:
        return [query for kind, query, _ in self.calls if kind == "session.run"]


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

    async def delete(self, raw_file_id: str) -> None:
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

    async def delete_pages(self, raw_file_id: str) -> int:
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

        result = await delete_document("some/path", version=None)

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

        await delete_document("some/path", version=None)

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

        await delete_document("some/path", version=None)

        fake_get_storage.assert_not_called()

    async def test_driver_is_closed_even_when_nothing_found(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        await delete_document("some/path")

        assert fake_driver.closed is True

    async def test_empty_string_path_is_passed_through_unchanged(self, patch_driver):
        """An empty-string path is not special-cased: it is passed through
        verbatim to the existence query, and (since it matches nothing)
        results in found=False rather than raising or silently defaulting.
        """
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        result = await delete_document("")

        assert result.path == ""
        assert result.found is False

        existence_calls = [
            params for kind, query, params in fake_driver.calls if kind == "session.run"
        ]
        assert existence_calls[0] == {"path": ""}


class TestDeleteDocumentCascade:
    async def test_found_document_runs_cascade_delete_with_correct_params(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                cascade_rows=[
                    {
                        "documents_deleted": 1,
                        "structure_nodes_deleted": 3,
                        "info_units_deleted": 2,
                        "model_decisions_deleted": 1,
                        "proposed_models_deleted": 1,
                        "proposed_fields_deleted": 4,
                        "extraction_results_deleted": 1,
                    }
                ],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/a", version=1)

        assert result.found is True
        assert result.versions_deleted == [1]
        assert result.documents_deleted == 1
        assert result.structure_nodes_deleted == 3
        assert result.info_units_deleted == 2
        assert result.model_decisions_deleted == 1
        assert result.proposed_models_deleted == 1
        assert result.proposed_fields_deleted == 4
        assert result.extraction_results_deleted == 1

        # Verify the cascade delete tx.run call received the right params.
        cascade_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert len(cascade_calls) == 1
        assert cascade_calls[0] == {"path": "docs/a", "version": 1}

    async def test_multiple_cascade_rows_are_summed(self, patch_driver):
        patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}, {"version": 2}],
                cascade_rows=[
                    {
                        "documents_deleted": 1,
                        "structure_nodes_deleted": 2,
                        "info_units_deleted": 1,
                        "model_decisions_deleted": 0,
                        "proposed_models_deleted": 0,
                        "proposed_fields_deleted": 0,
                        "extraction_results_deleted": 1,
                    },
                    {
                        "documents_deleted": 1,
                        "structure_nodes_deleted": 5,
                        "info_units_deleted": 0,
                        "model_decisions_deleted": 2,
                        "proposed_models_deleted": 1,
                        "proposed_fields_deleted": 3,
                        "extraction_results_deleted": 0,
                    },
                ],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/b")

        assert result.found is True
        assert result.versions_deleted == [1, 2]
        assert result.documents_deleted == 2
        assert result.structure_nodes_deleted == 7
        assert result.info_units_deleted == 1
        assert result.model_decisions_deleted == 2
        assert result.proposed_models_deleted == 1
        assert result.proposed_fields_deleted == 3
        assert result.extraction_results_deleted == 1

    async def test_version_none_is_omitted_from_the_bound_params(self, patch_driver):
        """When version is not given, no ``version`` condition/param is emitted
        (the WHERE stays a plain equality conjunction so the :Document indexes
        can be used) — only ``path`` is bound.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        await delete_document("docs/c")  # version defaults to None

        existence_calls = [
            (query, params)
            for kind, query, params in fake_driver.calls
            if kind == "session.run"
        ]
        assert existence_calls[0][1] == {"path": "docs/c"}
        assert "$version" not in existence_calls[0][0]
        assert "IS NULL" not in existence_calls[0][0]

        cascade_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls[0] == {"path": "docs/c"}

    async def test_explicit_version_passed_through(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 3}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        await delete_document("docs/d", version=3)

        cascade_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls[0] == {"path": "docs/d", "version": 3}

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

        await delete_document("docs/i", version=1)

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
            await delete_document("docs/broken")

        assert fake_driver.closed is True
        # No cascade or GC work should have been attempted.
        assert [c for c in fake_driver.calls if c[0] == "tx.run"] == []

    async def test_cascade_delete_exception_rolls_back_and_propagates(self, patch_driver):
        """A failure inside the cascade-delete write transaction must roll
        back that transaction and re-raise it must not be swallowed into
        a successful DeletionResult, and no commit should have happened.
        """
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                cascade_error=ValueError("cascade write failed"),
            )
        )

        with pytest.raises(ValueError, match="cascade write failed"):
            await delete_document("docs/j", version=1)

        rollback_calls = [c for c in fake_driver.calls if c[0] == "tx.rollback"]
        commit_calls = [c for c in fake_driver.calls if c[0] == "tx.commit"]
        assert len(rollback_calls) == 1
        assert commit_calls == []
        # GC passes must never run if the cascade delete itself failed.
        gc_calls = [
            query
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and ("Entity|ModelInstance" in query or "LabeledEntity" in query)
        ]
        assert gc_calls == []
        assert fake_driver.closed is True


class TestDeleteDocumentGarbageCollection:
    async def test_gc_pass_stops_at_first_zero(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[5, 2, 0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/e", version=1)

        assert result.gc_entity_model_instance_deleted == 7
        assert result.gc_entity_model_instance_passes == 3

        emi_run_count = sum(
            1
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "Entity|ModelInstance" in query
        )
        assert emi_run_count == 3

    async def test_gc_pass_caps_at_gc_max_passes_when_never_reaching_zero(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[1] * GC_MAX_PASSES,
                gc_le_sequence=[0],
            )
        )

        result = await delete_document("docs/f", version=1)

        assert result.gc_entity_model_instance_passes == GC_MAX_PASSES
        assert result.gc_entity_model_instance_deleted == GC_MAX_PASSES

        emi_run_count = sum(
            1
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "Entity|ModelInstance" in query
        )
        assert emi_run_count == GC_MAX_PASSES

    async def test_labeled_entity_pass_runs_only_after_entity_model_instance_pass_completes(
        self, patch_driver
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[3, 1, 0],
                gc_le_sequence=[2, 0],
            )
        )

        result = await delete_document("docs/g", version=1)

        assert result.gc_entity_model_instance_deleted == 4
        assert result.gc_entity_model_instance_passes == 3
        assert result.gc_labeled_entity_deleted == 2
        assert result.gc_labeled_entity_passes == 2

        # Order: all Entity|ModelInstance tx.run calls must precede all
        # LabeledEntity tx.run calls.
        gc_queries = [
            query
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and ("Entity|ModelInstance" in query or "LabeledEntity" in query)
        ]
        labels = [
            "emi" if "Entity|ModelInstance" in q else "le"
            for q in gc_queries
        ]
        assert labels == ["emi", "emi", "emi", "le", "le"]

    async def test_gc_second_pass_independent_max_passes(self, patch_driver):
        patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[1] * GC_MAX_PASSES,
            )
        )

        result = await delete_document("docs/h", version=1)

        assert result.gc_entity_model_instance_passes == 1
        assert result.gc_labeled_entity_passes == GC_MAX_PASSES
        assert result.gc_labeled_entity_deleted == GC_MAX_PASSES


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

        result = await delete_document("docs/k", version=1)

        assert raw_repo.deleted_ids == ["rid1", "rid2"]
        assert page_repo.deleted_ids == ["rid1", "rid2"]
        assert result.raw_files_deleted == 2
        assert result.converted_pages_deleted == 8

        # All storage deletion calls must complete before the cascade
        # delete (tx.run containing "documents_deleted") ever runs.
        kinds_in_order = [kind for kind, _, _ in fake_driver.calls]
        storage_indices = [
            i
            for i, k in enumerate(kinds_in_order)
            if k in ("storage.raw_delete", "storage.page_delete")
        ]
        cascade_indices = [
            i
            for i, (kind, query, _) in enumerate(fake_driver.calls)
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert storage_indices, "expected storage deletion calls to have happened"
        assert cascade_indices, "expected the cascade delete to have run"
        assert max(storage_indices) < min(cascade_indices)

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

        result = await delete_document("docs/l", version=1)

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
            await delete_document("docs/m", version=1)

        assert fake_driver.closed is True
        cascade_calls = [
            query
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls == []
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
            await delete_document("docs/n", version=1)

        assert fake_driver.closed is True
        cascade_calls = [
            query
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls == []
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

        result = await delete_document("docs/o", version=1)

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

        result = await delete_document("docs/p", version=1)

        assert result.raw_files_deleted == 3
        assert result.converted_pages_deleted == 5

    async def test_raw_file_ids_query_receives_correct_params(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 2}],
                raw_file_id_rows=[],
            )
        )

        await delete_document("docs/q", version=2)

        raw_file_id_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "session.run" and "RETURN DISTINCT n.raw_file_id AS raw_file_id" in query
        ]
        assert len(raw_file_id_calls) == 1
        assert raw_file_id_calls[0] == {"path": "docs/q", "version": 2}

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
            await delete_document("docs/r", version=1)

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
        cascade_calls = [
            query
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls == []

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
            await delete_document("docs/s", version=1)

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
        cascade_calls = [
            query
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls == []


class TestDeleteDocumentSelectorValidation:
    async def test_neither_path_nor_job_id_raises_value_error(self, patch_driver):
        patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError, match="exactly one of 'path' or 'job_id'"):
            await delete_document()

    async def test_both_path_and_job_id_raises_value_error(self, patch_driver):
        patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError, match="exactly one of 'path' or 'job_id'"):
            await delete_document("docs/a", job_id="job-1")

    async def test_value_error_raised_before_any_neo4j_call(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))
        with pytest.raises(ValueError):
            await delete_document()
        assert fake_driver.calls == []


class TestDeleteDocumentByJobId:
    async def test_job_id_selector_binds_only_job_id_and_echoes_it(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}, {"version": 1}, {"version": 2}],
                cascade_rows=[
                    {
                        "documents_deleted": 3,
                        "structure_nodes_deleted": 9,
                        "info_units_deleted": 0,
                        "model_decisions_deleted": 0,
                        "proposed_models_deleted": 0,
                        "proposed_fields_deleted": 0,
                        "extraction_results_deleted": 0,
                    }
                ],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        result = await delete_document(job_id="job-xyz")

        assert result.found is True
        assert result.path is None
        assert result.job_id == "job-xyz"
        assert result.documents_deleted == 3
        # Only the supplied selector is emitted — no path/version/tenant
        # conditions, and no `IS NULL` disjunction (so idx_document_job_id
        # can be used).
        existence_calls = [
            (query, params)
            for kind, query, params in fake_driver.calls
            if kind == "session.run"
        ]
        assert existence_calls[0][1] == {"job_id": "job-xyz"}
        assert "d.job_id = $job_id" in existence_calls[0][0]
        assert "IS NULL" not in existence_calls[0][0]
        cascade_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls[0] == {"job_id": "job-xyz"}

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

        cascade_calls = [
            (query, params)
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        query, params = cascade_calls[0]
        assert params == {
            "path": "docs/a",
            "version": 1,
            "tenant_id": "tenant-7",
            "created_by_user_id": "user-42",
        }
        # job_id was not supplied → no job_id condition at all
        assert "job_id" not in query
        for cond in (
            "d.path = $path",
            "d.version = $version",
            "d.tenant_id = $tenant_id",
            "d.created_by_user_id = $created_by_user_id",
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


class TestBuildDocMatch:
    """Unit tests for deletion._build_doc_match() — the dynamic WHERE builder."""

    def test_only_supplied_filters_become_conditions(self):
        match, params = deletion._build_doc_match(
            {"path": None, "version": None, "tenant_id": None,
             "created_by_user_id": None, "job_id": "j1"}
        )
        assert params == {"job_id": "j1"}
        assert match.strip() == "MATCH (d:Document)\nWHERE d.job_id = $job_id"

    def test_multiple_filters_are_anded_in_fixed_order(self):
        match, params = deletion._build_doc_match(
            {"path": "p", "version": 2, "tenant_id": "t",
             "created_by_user_id": None, "job_id": None}
        )
        assert params == {"path": "p", "version": 2, "tenant_id": "t"}
        assert (
            "WHERE d.path = $path AND d.version = $version AND d.tenant_id = $tenant_id"
            in match
        )

    def test_no_is_null_disjunction_is_ever_emitted(self):
        match, _ = deletion._build_doc_match({"path": "p"})
        assert "IS NULL" not in match
        assert " OR " not in match

    def test_falsy_but_non_none_values_are_kept(self):
        match, params = deletion._build_doc_match({"path": "", "version": 0})
        assert params == {"path": "", "version": 0}
        assert "d.path = $path" in match
        assert "d.version = $version" in match
