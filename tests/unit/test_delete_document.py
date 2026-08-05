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
        cascade_rows: list[dict] | None = None,
        gc_emi_sequence: list[int] | None = None,
        gc_le_sequence: list[int] | None = None,
        existence_error: Exception | None = None,
        cascade_error: Exception | None = None,
    ) -> None:
        self.existence_rows = existence_rows if existence_rows is not None else []
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
        self.cascade_error = cascade_error

    def session(self) -> _FakeSession:
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


@pytest.fixture
def patch_driver(monkeypatch):
    """Monkeypatch scinr.newton.ingest.deletion.get_driver to return a given fake driver."""

    def _patch(fake_driver: _FakeDriver) -> _FakeDriver:
        monkeypatch.setattr(deletion, "get_driver", lambda: fake_driver)
        return fake_driver

    return _patch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeleteDocumentNotFound:
    def test_no_matching_document_returns_found_false_with_zero_counters(self, patch_driver):
        patch_driver(_FakeDriver(existence_rows=[]))

        result = delete_document("some/path", version=None)

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

    def test_no_matching_document_does_not_run_delete_or_gc_queries(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        delete_document("some/path", version=None)

        # Only the read-only existence check should have run.
        execute_write_calls = [c for c in fake_driver.calls if c[0] == "session.execute_write"]
        tx_run_calls = [c for c in fake_driver.calls if c[0] == "tx.run"]
        assert execute_write_calls == []
        assert tx_run_calls == []

        session_run_calls = [c for c in fake_driver.calls if c[0] == "session.run"]
        assert len(session_run_calls) == 1

    def test_driver_is_closed_even_when_nothing_found(self, patch_driver):
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        delete_document("some/path")

        assert fake_driver.closed is True

    def test_empty_string_path_is_passed_through_unchanged(self, patch_driver):
        """An empty-string path is not special-cased: it is passed through
        verbatim to the existence query, and (since it matches nothing)
        results in found=False rather than raising or silently defaulting.
        """
        fake_driver = patch_driver(_FakeDriver(existence_rows=[]))

        result = delete_document("")

        assert result.path == ""
        assert result.found is False

        existence_calls = [
            params for kind, query, params in fake_driver.calls if kind == "session.run"
        ]
        assert existence_calls[0] == {"path": "", "version": None}


class TestDeleteDocumentCascade:
    def test_found_document_runs_cascade_delete_with_correct_params(self, patch_driver):
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

        result = delete_document("docs/a", version=1)

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

    def test_multiple_cascade_rows_are_summed(self, patch_driver):
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

        result = delete_document("docs/b")

        assert result.found is True
        assert result.versions_deleted == [1, 2]
        assert result.documents_deleted == 2
        assert result.structure_nodes_deleted == 7
        assert result.info_units_deleted == 1
        assert result.model_decisions_deleted == 2
        assert result.proposed_models_deleted == 1
        assert result.proposed_fields_deleted == 3
        assert result.extraction_results_deleted == 1

    def test_version_none_passed_through_as_null(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        delete_document("docs/c")  # version defaults to None

        existence_calls = [
            params for kind, query, params in fake_driver.calls if kind == "session.run"
        ]
        assert existence_calls[0] == {"path": "docs/c", "version": None}

        cascade_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls[0] == {"path": "docs/c", "version": None}

    def test_explicit_version_passed_through(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 3}],
                gc_emi_sequence=[0],
                gc_le_sequence=[0],
            )
        )

        delete_document("docs/d", version=3)

        cascade_calls = [
            params
            for kind, query, params in fake_driver.calls
            if kind == "tx.run" and "documents_deleted" in query
        ]
        assert cascade_calls[0] == {"path": "docs/d", "version": 3}

    def test_driver_is_closed_after_successful_deletion(self, patch_driver):
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

        delete_document("docs/i", version=1)

        assert fake_driver.closed is True

    def test_existence_query_exception_propagates_and_still_closes_driver(self, patch_driver):
        """If the existence check itself raises (e.g. a Neo4j connectivity
        error), delete_document must not swallow it: it should propagate to
        the caller, while still closing the driver via the `finally` block.
        """
        fake_driver = patch_driver(
            _FakeDriver(existence_error=RuntimeError("neo4j unavailable"))
        )

        with pytest.raises(RuntimeError, match="neo4j unavailable"):
            delete_document("docs/broken")

        assert fake_driver.closed is True
        # No cascade or GC work should have been attempted.
        assert [c for c in fake_driver.calls if c[0] == "tx.run"] == []

    def test_cascade_delete_exception_rolls_back_and_propagates(self, patch_driver):
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
            delete_document("docs/j", version=1)

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
    def test_gc_pass_stops_at_first_zero(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[5, 2, 0],
                gc_le_sequence=[0],
            )
        )

        result = delete_document("docs/e", version=1)

        assert result.gc_entity_model_instance_deleted == 7
        assert result.gc_entity_model_instance_passes == 3

        emi_run_count = sum(
            1
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "Entity|ModelInstance" in query
        )
        assert emi_run_count == 3

    def test_gc_pass_caps_at_gc_max_passes_when_never_reaching_zero(self, patch_driver):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[1] * GC_MAX_PASSES,
                gc_le_sequence=[0],
            )
        )

        result = delete_document("docs/f", version=1)

        assert result.gc_entity_model_instance_passes == GC_MAX_PASSES
        assert result.gc_entity_model_instance_deleted == GC_MAX_PASSES

        emi_run_count = sum(
            1
            for kind, query, _ in fake_driver.calls
            if kind == "tx.run" and "Entity|ModelInstance" in query
        )
        assert emi_run_count == GC_MAX_PASSES

    def test_labeled_entity_pass_runs_only_after_entity_model_instance_pass_completes(
        self, patch_driver
    ):
        fake_driver = patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[3, 1, 0],
                gc_le_sequence=[2, 0],
            )
        )

        result = delete_document("docs/g", version=1)

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

    def test_gc_second_pass_independent_max_passes(self, patch_driver):
        patch_driver(
            _FakeDriver(
                existence_rows=[{"version": 1}],
                gc_emi_sequence=[0],
                gc_le_sequence=[1] * GC_MAX_PASSES,
            )
        )

        result = delete_document("docs/h", version=1)

        assert result.gc_entity_model_instance_passes == 1
        assert result.gc_labeled_entity_passes == GC_MAX_PASSES
        assert result.gc_labeled_entity_deleted == GC_MAX_PASSES
