"""
tests/unit/graph_neo4j/test_errors.py — Translation of ``neo4j.exceptions`` to the
graph exception hierarchy (``graph/neo4j/errors.py``).

Each row of the table in the plan (section 2.5) is checked through a fake driver
that raises the driver's exception, run inside the real retry wrapper and the
real ``neo4j_errors()`` boundary: type, message, ``__cause__``, a clean
hierarchy (no ``IngestionError`` / ``ValueError`` / ``NavigationError`` bases),
and that the retry happens *before* the translation (same number of attempts as
without it).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from neo4j.exceptions import (
    AuthError,
    ClientError,
    DriverError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)

from scinr.newton.exceptions import (
    DocumentVersionConflict,
    GraphAuthenticationError,
    GraphConnectionError,
    GraphConstraintError,
    GraphError,
    GraphQueryError,
    GraphResourceError,
    IngestionError,
    NavigationError,
    ScinrError,
    TransientGraphError,
)
from scinr.newton.graph.neo4j.errors import neo4j_errors, translate
from scinr.newton.graph.neo4j.retry import (
    TOTAL_RETRIES,
    with_neo4j_retry,
    with_neo4j_retry_sync,
)

VERSION_CONFLICT_MESSAGE = (
    "Version conflict: version 3 was already ingested by a concurrent process. "
    "Retry the ingestion."
)


class _FakeSession:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver

    def run(self, query: str, **params):
        self._driver.attempts += 1
        raise self._driver.error

    async def arun(self, query: str, **params):
        self._driver.attempts += 1
        raise self._driver.error


class _FakeDriver:
    """A driver whose every statement raises *error*; counts the attempts."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0

    def session(self) -> _FakeSession:
        return _FakeSession(self)


def _hydrate(code: str, message: str) -> Neo4jError:
    return Neo4jError._hydrate_neo4j(code=code, message=message)


def _run_sync(driver: _FakeDriver, **kwargs):
    with neo4j_errors(**kwargs):
        return with_neo4j_retry_sync(lambda: driver.session().run("RETURN 1"))


async def _run_async(driver: _FakeDriver, **kwargs):
    with neo4j_errors(**kwargs):
        return await with_neo4j_retry(lambda: driver.session().arun("RETURN 1"))


def _assert_clean(exc: BaseException) -> None:
    assert not isinstance(exc, (IngestionError, ValueError, NavigationError))
    assert isinstance(exc, GraphError)


# (label, driver exception, expected class, attempts, keyword arguments of neo4j_errors)
_CASES = [
    (
        "constraint-version",
        _hydrate(
            "Neo.ClientError.Schema.ConstraintValidationFailed",
            "Node already exists with constraint_document_tenant_path_version",
        ),
        DocumentVersionConflict,
        1,
        {"version_conflict": VERSION_CONFLICT_MESSAGE},
    ),
    (
        "constraint-other",
        _hydrate(
            "Neo.ClientError.Schema.ConstraintValidationFailed",
            "Node already exists with label `Theme`",
        ),
        GraphConstraintError,
        1,
        {"version_conflict": VERSION_CONFLICT_MESSAGE},
    ),
    (
        "constraint-no-insert-context",
        _hydrate(
            "Neo.ClientError.Schema.ConstraintValidationFailed",
            "something about version 3",
        ),
        GraphConstraintError,
        1,
        {},
    ),
    (
        "auth",
        _hydrate("Neo.ClientError.Security.Unauthorized", "The client is unauthorized"),
        GraphAuthenticationError,
        1,
        {},
    ),
    ("service-unavailable", ServiceUnavailable("Failed to read from defunct connection"),
     GraphConnectionError, TOTAL_RETRIES + 1, {}),
    ("session-expired", SessionExpired("Server no longer the leader"),
     GraphConnectionError, TOTAL_RETRIES + 1, {}),
    (
        "out-of-memory",
        _hydrate(
            "Neo.TransientError.General.MemoryPoolOutOfMemoryError",
            "The allocation of an extra 2.0 MiB would use more than the limit",
        ),
        GraphResourceError,
        TOTAL_RETRIES + 1,
        {},
    ),
    (
        "transient",
        _hydrate("Neo.TransientError.Transaction.DeadlockDetected", "deadlock"),
        TransientGraphError,
        TOTAL_RETRIES + 1,
        {},
    ),
    (
        "syntax",
        _hydrate("Neo.ClientError.Statement.SyntaxError", "Invalid input"),
        GraphQueryError,
        1,
        {},
    ),
    ("driver-error", DriverError("bad driver usage"), GraphQueryError, 1, {}),
]


@pytest.mark.parametrize(
    ("error", "expected", "attempts", "kwargs"),
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
class TestTranslation:
    def test_sync(self, error, expected, attempts, kwargs):
        driver = _FakeDriver(error)
        with patch("scinr.newton.graph.neo4j.retry.time.sleep"):
            with pytest.raises(GraphError) as info:
                _run_sync(driver, **kwargs)
        exc = info.value
        assert type(exc) is expected
        _assert_clean(exc)
        assert exc.__cause__ is error
        expected_message = (
            VERSION_CONFLICT_MESSAGE if expected is DocumentVersionConflict else str(error)
        )
        assert str(exc) == expected_message
        assert driver.attempts == attempts

    async def test_async(self, error, expected, attempts, kwargs):
        driver = _FakeDriver(error)

        async def _no_sleep(_delay):
            return None

        with patch("scinr.newton.graph.neo4j.retry.asyncio.sleep", _no_sleep):
            with pytest.raises(GraphError) as info:
                await _run_async(driver, **kwargs)
        assert type(info.value) is expected
        assert info.value.__cause__ is error
        assert driver.attempts == attempts


class TestHierarchy:
    def test_clean_parents(self):
        assert GraphConnectionError.__bases__ == (GraphError,)
        assert GraphAuthenticationError.__bases__ == (GraphConnectionError,)
        assert GraphConstraintError.__bases__ == (GraphError,)
        assert DocumentVersionConflict.__bases__ == (GraphConstraintError,)
        for cls in (TransientGraphError, GraphResourceError, GraphQueryError):
            assert cls.__bases__ == (GraphError,)
        assert GraphError.__bases__ == (ScinrError,)

    def test_connection_error_is_not_a_navigation_error(self):
        assert not issubclass(GraphConnectionError, NavigationError)

    def test_version_conflict_is_not_an_ingestion_error(self):
        assert not issubclass(DocumentVersionConflict, IngestionError)

    def test_graph_configuration_error_is_not_a_value_error(self):
        from scinr.newton.exceptions import ConfigurationError, GraphConfigurationError

        assert GraphConfigurationError.__bases__ == (ConfigurationError,)
        assert not issubclass(GraphConfigurationError, ValueError)


class TestPassThrough:
    def test_translate_ignores_foreign_exceptions(self):
        assert translate(ValueError("x")) is None
        assert translate(RuntimeError("x")) is None
        assert translate(GraphQueryError("already translated")) is None

    def test_context_manager_does_not_touch_foreign_exceptions(self):
        boom = ValueError("not from the engine")
        with pytest.raises(ValueError) as info:
            with neo4j_errors():
                raise boom
        assert info.value is boom

    def test_already_translated_exceptions_are_not_wrapped_twice(self):
        original = GraphConnectionError("down")
        with pytest.raises(GraphConnectionError) as info:
            with neo4j_errors():
                raise original
        assert info.value is original

    def test_exception_that_is_not_an_exception_subclass_passes(self):
        with pytest.raises(KeyboardInterrupt):
            with neo4j_errors():
                raise KeyboardInterrupt

    def test_client_error_is_a_query_error(self):
        err = ClientError("generic client error")
        translated = translate(err)
        assert isinstance(translated, GraphQueryError)


class TestDecorator:
    def test_sync_function(self):
        calls = []

        @neo4j_errors()
        def method(arg):
            calls.append(arg)
            raise AuthError("bad credentials")

        with pytest.raises(GraphAuthenticationError):
            method(1)
        assert calls == [1]
        assert method.__name__ == "method"

    async def test_async_function(self):
        @neo4j_errors()
        async def method():
            raise ServiceUnavailable("down")

        with pytest.raises(GraphConnectionError):
            await method()

    def test_returns_value(self):
        @neo4j_errors()
        def method():
            return 42

        assert method() == 42


class TestCallbackErrors:
    def test_callback_error_passes_untouched_and_is_not_retried(self):
        """An error of the caller's callback that is not from the engine is not retried
        and not translated (``export_snapshot`` contract)."""
        attempts = []

        def attempt():
            attempts.append(1)
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            with neo4j_errors():
                with_neo4j_retry_sync(attempt)
        assert attempts == [1]

    def test_driver_error_in_callback_is_retried_then_translated(self):
        driver = _FakeDriver(ServiceUnavailable("Failed to write to defunct connection"))

        def attempt():
            driver.session().run("RETURN 1")

        with patch("scinr.newton.graph.neo4j.retry.time.sleep"):
            with pytest.raises(GraphConnectionError):
                with neo4j_errors():
                    with_neo4j_retry_sync(attempt)
        assert driver.attempts == TOTAL_RETRIES + 1


# ---------------------------------------------------------------------------
# Store methods (work packages 2 to 7)
# ---------------------------------------------------------------------------


class _Tx:
    """Sync session/transaction whose every statement raises *error*."""

    def __init__(self, driver: _StatementDriver) -> None:
        self._driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):
        self._driver.attempts += 1
        raise self._driver.error

    def execute_write(self, fn):
        return fn(self)

    def begin_transaction(self):
        return self

    def commit(self):
        self._driver.commits += 1

    def rollback(self):
        self._driver.rollbacks += 1


class _AsyncSession:
    def __init__(self, driver: _StatementDriver) -> None:
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query: str, **params):
        self._driver.attempts += 1
        raise self._driver.error


class _StatementDriver:
    """Sync and async driver whose every statement raises *error*."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def session(self, **kwargs):
        # The async driver's ``session`` is the same plain call returning an
        # async context manager; the sync code path enters it with ``with``.
        return _SyncOrAsyncSession(self)

    def close(self):
        self.closed += 1


class _Done:
    """Awaitable that completes at once; what an async transaction's commit/rollback return."""

    def __await__(self):
        return
        yield


class _SyncOrAsyncSession(_Tx, _AsyncSession):
    """Supports ``with`` and ``async with``; as a transaction, also awaitable steps."""

    def __init__(self, driver: _StatementDriver) -> None:
        _Tx.__init__(self, driver)
        _AsyncSession.__init__(self, driver)

    def __await__(self):
        # ``await session.begin_transaction()`` resolves to the session itself.
        return self
        yield

    def commit(self):
        _Tx.commit(self)
        return _Done()

    def rollback(self):
        _Tx.rollback(self)
        return _Done()


def _document():
    from scinr.newton.models.document_structure import Document

    return Document(
        document_name="doc",
        document_type="pdf",
        doc_path="F/doc",
        raw_file_id="",
        document_structure=[],
    )


async def _ensure_catalog_through_sink(store):
    from scinr.newton.annotation.catalog import ensure_catalog_models

    async with store.annotation.catalog_sink() as sink:
        await ensure_catalog_models(_Registry(), sink)


async def _ensure_themes_through_sink(store):
    from scinr.newton.annotation.catalog import ensure_theme_structure

    async with store.annotation.catalog_sink() as sink:
        await ensure_theme_structure(_Registry(), sink)


async def _write_extraction_through_sink(store):
    from scinr.newton.entity_extraction.graph_mapper import write_extraction_subgraph
    from scinr.newton.graph import reset_graph_store, set_graph_store

    set_graph_store(store)
    try:
        await write_extraction_subgraph(
            "acme::F/doc::1::n", _Extracted(), "Product", [], "doc", "er-uid"
        )
    finally:
        reset_graph_store()


async def _write_triple_through_sink(store):
    from scinr.newton.entity_extraction.graph_mapper import write_triple_subgraph
    from scinr.newton.graph import reset_graph_store, set_graph_store

    set_graph_store(store)
    try:
        await write_triple_subgraph("acme::F/doc::1::n", _Extracted(), "doc", "er-uid")
    finally:
        reset_graph_store()


async def _write_tabular_through_sink(store):
    from scinr.newton.graph import reset_graph_store, set_graph_store
    from scinr.newton.tabular.graph_writer import write_tabular_subgraph

    set_graph_store(store)
    try:
        await write_tabular_subgraph(
            "F/sheet",
            "sheet",
            1,
            {"sheet_name": "s", "headers": ["c"], "total_rows": 0},
            lambda n: iter(()),
            0,
            _decision(),
            None,
            tenant_id="acme",
        )
    finally:
        reset_graph_store()


class _Registry:
    """Just enough of a ThemeRegistry for ``ensure_catalog_models`` / ``ensure_theme_structure``."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        from pydantic import BaseModel

        class Model(BaseModel):
            """A model."""

            code: str = ""

        self._themes = {"default": SimpleNamespace(models=[Model])}

    def get_neo4j_theme_structure(self):
        return [{"path": "default", "name": "default", "parent_path": None, "model_names": ["M"]}]


def _decision():
    from scinr.newton.annotation.models import AnnotationDecision

    return AnnotationDecision(
        matched_model_class="Product",
        confidence="high",
        rationale="r",
        coverage_gaps=[],
        propose_new_model=False,
    )


_DOC_KEY = {"path": "F/doc", "version": 1}


def _delete_selector():
    from scinr.newton.graph.types import DocumentSelector

    return DocumentSelector(tenant_id="acme", path="F/doc")


def _doc_info():
    from scinr.newton.graph.types import DocumentInfo

    return DocumentInfo(
        path="F/doc",
        version=1,
        tenant_id="acme",
        is_seed=True,
        properties={"path": "F/doc", "version": 1, "tenant_id": "acme"},
    )


def _export_everything(store):
    """An ``export_snapshot`` attempt that reads the nodes of the document (the
    first statement fails inside the callback, while the reader is iterated)."""
    doc = _doc_info()
    return store.freeze.export_snapshot([doc], "acme", lambda reader: list(reader.subtree_nodes(doc)))


def _collect_with_a_candidate(store):
    collector = store.gc.collector("acme", "t")
    collector._add("4:db:1", False)
    collector.collect()


def _selector():
    from scinr.newton.graph.types import DocumentSelector

    return DocumentSelector(tenant_id="acme", path="F/doc", latest=True)


class _Extracted:
    """Stands for the populated model of an extraction: the write fails before reading it."""

    model_fields: dict = {}
    triples: list = []


# (id, async?, callable(store) -> result/awaitable, retries?)
_STORE_CALLS = [
    ("schema.setup", False, lambda s: s.schema.setup(), False),
    ("documents.next_version", False, lambda s: s.documents.next_version("p", "acme"), False),
    (
        "documents.current_latest_version",
        False,
        lambda s: s.documents.current_latest_version("p", "acme"),
        False,
    ),
    (
        "documents.resolve_batch_version",
        False,
        lambda s: s.documents.resolve_batch_version(["p"], ["acme"], False),
        False,
    ),
    (
        "documents.insert_document_graph",
        False,
        lambda s: s.documents.insert_document_graph(_document(), 1, False),
        True,
    ),
    (
        "documents.create_tabular_document",
        False,
        lambda s: s.documents.create_tabular_document(
            document_name="t.csv",
            doc_path="t.csv",
            version=1,
            raw_file_id="",
            tenant="acme",
            created_by_user_id=None,
            job_id=None,
        ),
        False,
    ),
    ("documents.preflight_replaces", False, lambda s: s.documents.preflight_replaces("n", "acme"), False),
    (
        "documents.apply_replacement",
        False,
        lambda s: s.documents.apply_replacement("n", ["new"], tenant="acme", replaced_path=None),
        False,
    ),
    (
        "documents.resolve_leaf_documents",
        False,
        lambda s: s.documents.resolve_leaf_documents(_selector()),
        False,
    ),
    (
        "documents.resolve_leaf_documents_async",
        True,
        lambda s: s.documents.resolve_leaf_documents_async(_selector()),
        False,
    ),
    (
        "documents.count_latest_documents",
        True,
        lambda s: s.documents.count_latest_documents(_selector()),
        False,
    ),
    (
        "annotation.count_annotated_nodes",
        True,
        lambda s: s.annotation.count_annotated_nodes(_selector()),
        False,
    ),
    (
        "annotation.catalog_sink.ensure_catalog_models",
        True,
        lambda s: _ensure_catalog_through_sink(s),
        False,
    ),
    (
        "annotation.catalog_sink.ensure_theme_structure",
        True,
        lambda s: _ensure_themes_through_sink(s),
        False,
    ),
    (
        "annotation.fetch_nodes_to_annotate",
        True,
        lambda s: s.annotation.fetch_nodes_to_annotate(tenant_id="acme", doc_path="F/doc"),
        False,
    ),
    (
        "annotation.fetch_node_context",
        True,
        lambda s: s.annotation.fetch_node_context("acme::F/doc::1::n", "n"),
        False,
    ),
    (
        "annotation.fetch_document_context_instructions",
        True,
        lambda s: s.annotation.fetch_document_context_instructions(
            tenant_id="acme", doc_path="F/doc"
        ),
        False,
    ),
    (
        "annotation.write_annotation",
        True,
        lambda s: s.annotation.write_annotation("acme::F/doc::1::n", _decision(), "doc"),
        False,
    ),
    (
        "annotation.write_manual_annotation",
        True,
        lambda s: s.annotation.write_manual_annotation(
            "doc", "Product", tenant_id="acme", doc_path="F/doc"
        ),
        False,
    ),
    (
        "extraction.fetch_targets",
        True,
        lambda s: s.extraction.fetch_targets(tenant_id="acme", doc_path="F/doc"),
        False,
    ),
    (
        "extraction.mark_info_units_extracted",
        True,
        lambda s: s.extraction.mark_info_units_extracted("acme::F/doc::1::n"),
        False,
    ),
    (
        "extraction.sink.write_extraction_subgraph",
        True,
        lambda s: _write_extraction_through_sink(s),
        False,
    ),
    (
        "extraction.sink.write_triple_subgraph",
        True,
        lambda s: _write_triple_through_sink(s),
        False,
    ),
    (
        "tabular.sink.write_tabular_subgraph",
        True,
        lambda s: _write_tabular_through_sink(s),
        False,
    ),
    (
        "tabular.delete_tabular_subgraph",
        True,
        lambda s: s.tabular.delete_tabular_subgraph("F/sheet", 1, 0, tenant_id="acme"),
        False,
    ),
    (
        "deletion.fetch_document_keys",
        False,
        lambda s: s.deletion.fetch_document_keys(_delete_selector()),
        True,
    ),
    (
        "deletion.fetch_existing_versions",
        False,
        lambda s: s.deletion.fetch_existing_versions(_delete_selector()),
        True,
    ),
    (
        "deletion.fetch_raw_file_ids",
        False,
        lambda s: s.deletion.fetch_raw_file_ids(_delete_selector()),
        True,
    ),
    (
        "deletion.mark_deletion_pending",
        False,
        lambda s: s.deletion.mark_deletion_pending("acme", [_DOC_KEY]),
        True,
    ),
    (
        "deletion.cascade_delete",
        False,
        lambda s: s.deletion.cascade_delete("acme", [_DOC_KEY], None, caller="t"),
        True,
    ),
    (
        "deletion.delete_documents",
        False,
        lambda s: s.deletion.delete_documents("acme", [_DOC_KEY]),
        True,
    ),
    (
        "gc.collector.gather",
        False,
        lambda s: s.gc.collector("acme", "t").gather([_DOC_KEY]),
        True,
    ),
    (
        "gc.collector.collect",
        False,
        lambda s: _collect_with_a_candidate(s),
        True,
    ),
    (
        "gc.sweep_tenant",
        False,
        lambda s: s.gc.sweep_tenant("acme", "t"),
        True,
    ),
    (
        "gc.run_pass",
        False,
        lambda s: s.gc.run_pass("labeled_entity", "acme", "t"),
        True,
    ),
    (
        "freeze.fetch_document_set",
        False,
        lambda s: s.freeze.fetch_document_set(_delete_selector()),
        True,
    ),
    (
        "freeze.export_snapshot",
        False,
        lambda s: _export_everything(s),
        True,
    ),
    (
        "freeze.mark_frozen",
        False,
        lambda s: s.freeze.mark_frozen(
            "acme",
            [_doc_info()],
            blob_id="b",
            frozen_at="2026-01-01T00:00:00+00:00",
            keep_flags=(False, False, False),
        ),
        True,
    ),
    (
        "freeze.mark_backup",
        False,
        lambda s: s.freeze.mark_backup("acme", [_doc_info()], "b", "2026-01-01T00:00:00+00:00"),
        True,
    ),
    (
        "freeze.reduce_to_stubs",
        False,
        lambda s: s.freeze.reduce_to_stubs(
            "acme", [_doc_info()], (False, False, False), None, caller="t"
        ),
        True,
    ),
    (
        "freeze.clear_cleanup_pending",
        False,
        lambda s: s.freeze.clear_cleanup_pending("acme", [_doc_info()]),
        True,
    ),
    (
        "restore.existing_documents",
        False,
        lambda s: s.restore.existing_documents("acme", [_DOC_KEY]),
        True,
    ),
    (
        "restore.frozen_stubs_of_blob",
        False,
        lambda s: s.restore.frozen_stubs_of_blob("acme", "b"),
        True,
    ),
    (
        "restore.finish_restore",
        False,
        lambda s: s.restore.finish_restore("acme", [_doc_info()], ["F/doc"]),
        True,
    ),
    (
        "restore.count_blob_references",
        False,
        lambda s: s.restore.count_blob_references("acme", "b"),
        True,
    ),
    (
        "restore.has_stale_extraction_results",
        False,
        lambda s: s.restore.has_stale_extraction_results([["acme::F/doc::1::n"]]),
        True,
    ),
]


@pytest.fixture
def failing_store(monkeypatch):
    """A ``Neo4jGraphStore`` whose sync and async drivers are one failing fake."""
    from types import SimpleNamespace

    from scinr.newton.annotation import catalog as catalog_mod
    from scinr.newton.graph.neo4j import annotation as annotation_mod
    from scinr.newton.graph.neo4j import deletion as deletion_mod
    from scinr.newton.graph.neo4j import documents as documents_mod
    from scinr.newton.graph.neo4j import driver as driver_mod
    from scinr.newton.graph.neo4j import extraction as extraction_mod
    from scinr.newton.graph.neo4j import freeze as freeze_mod
    from scinr.newton.graph.neo4j import gc as gc_mod
    from scinr.newton.graph.neo4j import restore as restore_mod
    from scinr.newton.graph.neo4j import schema as schema_mod
    from scinr.newton.graph.neo4j import tabular as tabular_mod
    from scinr.newton.graph.neo4j.store import Neo4jGraphStore

    holder: dict[str, _StatementDriver] = {}
    cfg = SimpleNamespace(neo4j_database="neo4j", full_docstring=False)
    for module in (
        annotation_mod,
        deletion_mod,
        documents_mod,
        extraction_mod,
        freeze_mod,
        gc_mod,
        restore_mod,
        schema_mod,
        tabular_mod,
    ):
        monkeypatch.setattr(module, "get_config", lambda: cfg)
    monkeypatch.setattr(catalog_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(driver_mod, "get_driver", lambda: holder["driver"])
    monkeypatch.setattr(driver_mod, "get_async_driver", lambda: holder["driver"])

    def _make(error: Exception) -> tuple[Neo4jGraphStore, _StatementDriver]:
        holder["driver"] = _StatementDriver(error)
        return Neo4jGraphStore(), holder["driver"]

    return _make


_TRANSIENT = _hydrate("Neo.TransientError.Transaction.DeadlockDetected", "deadlock")
_SYNTAX = _hydrate("Neo.ClientError.Statement.SyntaxError", "Invalid input")


@pytest.mark.parametrize(
    ("error", "expected"),
    [(_TRANSIENT, TransientGraphError), (_SYNTAX, GraphQueryError)],
    ids=["transient", "non-transient"],
)
@pytest.mark.parametrize(
    ("is_async", "call", "retries"),
    [c[1:] for c in _STORE_CALLS],
    ids=[c[0] for c in _STORE_CALLS],
)
class TestStoreMethodsTranslateEngineErrors:
    async def test_error_is_translated_after_the_retries(
        self, failing_store, error, expected, is_async, call, retries
    ):
        store, driver = failing_store(error)
        with patch("scinr.newton.graph.neo4j.retry.time.sleep"):
            with pytest.raises(GraphError) as info:
                if is_async:
                    await call(store)
                else:
                    call(store)
        exc = info.value
        assert type(exc) is expected
        _assert_clean(exc)
        assert exc.__cause__ is error
        assert str(exc) == str(error)
        if retries and expected is TransientGraphError:
            assert driver.attempts == TOTAL_RETRIES + 1
        else:
            assert driver.attempts >= 1


def _extraction_sink_calls():
    from scinr.newton.graph.types import Provenance

    prov = Provenance(tenant_id="acme", created_by_user_id="u1", job_id="j1")
    return [
        # (name, retried, call)
        ("read_node_provenance", False, lambda k: k.read_node_provenance("n")),
        ("delete_stale_extraction_result", False, lambda k: k.delete_stale_extraction_result("n")),
        (
            "create_extraction_result",
            False,
            lambda k: k.create_extraction_result("n", "u", "doc", "M", "ts", prov),
        ),
        ("link_catalog_model", True, lambda k: k.link_catalog_model("u", "M", primary=True)),
        (
            "merge_keyed_instance",
            False,
            lambda k: k.merge_keyed_instance("ModelInstance", "p", "HAS_X", "c", "M", {"a": "b"}, prov, 0),
        ),
        (
            "create_instance",
            False,
            lambda k: k.create_instance("ExtractionResult", "p", "HAS_X", "c", "M", prov, None),
        ),
        ("merge_labeled_entity", True, lambda k: k.merge_labeled_entity("u", "L", "v", "v", prov)),
        (
            "link_reference",
            False,
            lambda k: k.link_reference("ModelInstance", "p", "le", "f", 1),
        ),
        (
            "merge_instance_shell",
            True,
            lambda k: k.merge_instance_shell("t", "M", {"a": "b"}, prov),
        ),
        (
            "link_instance_relationship",
            True,
            lambda k: k.link_instance_relationship("ModelInstance", "s", "t", "REL"),
        ),
        (
            "set_instance_properties",
            False,
            lambda k: k.set_instance_properties("ModelInstance", "p", {"a": 1}),
        ),
        ("link_field_relationship", False, lambda k: k.link_field_relationship("s", "t", "REL")),
        ("delete_triple_extraction_result", False, lambda k: k.delete_triple_extraction_result("n")),
        (
            "create_triple_extraction_result",
            False,
            lambda k: k.create_triple_extraction_result("n", "u", "doc", "ts", prov),
        ),
        ("merge_entity", True, lambda k: k.merge_entity("u", "v", "v", prov)),
        ("link_triple_entity_subject", False, lambda k: k.link_triple_entity("er", "e", "subject")),
        ("link_triple_entity_object", False, lambda k: k.link_triple_entity("er", "e", "object")),
        ("merge_predicate", False, lambda k: k.merge_predicate("s", "o", "REL", "rel")),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [(_TRANSIENT, TransientGraphError), (_SYNTAX, GraphQueryError)],
    ids=["transient", "non-transient"],
)
@pytest.mark.parametrize(
    ("retried", "call"),
    [c[1:] for c in _extraction_sink_calls()],
    ids=[c[0] for c in _extraction_sink_calls()],
)
class TestExtractionSinkErrors:
    async def test_each_sink_method_translates_the_engine_error(
        self, failing_store, error, expected, retried, call
    ):
        store, driver = failing_store(error)
        with patch("scinr.newton.graph.neo4j.retry.asyncio.sleep", new=AsyncMock()):
            async with store.extraction.sink() as sink:
                with pytest.raises(GraphError) as info:
                    await call(sink)
        exc = info.value
        assert type(exc) is expected
        _assert_clean(exc)
        assert exc.__cause__ is error
        assert str(exc) == str(error)
        # Only the statements that were retried before the split are still retried.
        assert driver.attempts == (TOTAL_RETRIES + 1 if retried and expected is TransientGraphError else 1)


def _tabular_sink_calls():
    from scinr.newton.graph.types import Provenance, RowData, TableNode

    prov = Provenance(tenant_id="acme", created_by_user_id="u1", job_id="j1")
    table = TableNode(
        id="t",
        node_id="table_1",
        title="s",
        order=1,
        theme="d",
        col_count=1,
        row_count=0,
        doc_key=("acme", "F/sheet", 1),
        page_id="",
    )
    row = RowData(
        id="t/row_1", node_id="row_1", row_index=0, appearance_order=1, info_uid="u", description="d"
    )
    return [
        # (name, retried, call)
        ("create_table", False, lambda k: k.create_table(table)),
        ("write_row_batch", False, lambda k: k.write_row_batch("t", [row], "md", "d", "")),
        (
            "replace_raw_extraction_result",
            False,
            lambda k: k.replace_raw_extraction_result("t/row_1", "er", "doc", "ts"),
        ),
        (
            "link_raw_column_value",
            True,
            lambda k: k.link_raw_column_value("er", "HAS_C", "e", "v", "v", prov),
        ),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [(_TRANSIENT, TransientGraphError), (_SYNTAX, GraphQueryError)],
    ids=["transient", "non-transient"],
)
@pytest.mark.parametrize(
    ("retried", "call"),
    [c[1:] for c in _tabular_sink_calls()],
    ids=[c[0] for c in _tabular_sink_calls()],
)
class TestTabularSinkErrors:
    async def test_each_sink_method_translates_the_engine_error(
        self, failing_store, error, expected, retried, call
    ):
        store, driver = failing_store(error)
        with patch("scinr.newton.graph.neo4j.retry.asyncio.sleep", new=AsyncMock()):
            async with store.tabular.sink() as sink:
                with pytest.raises(GraphError) as info:
                    await call(sink)
        exc = info.value
        assert type(exc) is expected
        _assert_clean(exc)
        assert exc.__cause__ is error
        assert str(exc) == str(error)
        # Only the statement that was retried before the split is still retried.
        assert driver.attempts == (TOTAL_RETRIES + 1 if retried and expected is TransientGraphError else 1)


class TestInsertDocumentGraphErrors:
    def test_version_conflict_keeps_the_message_of_the_old_ingestion_error(self, failing_store):
        conflict = _hydrate(
            "Neo.ClientError.Schema.ConstraintValidationFailed",
            "Node already exists with constraint_document_tenant_path_version",
        )
        store, driver = failing_store(conflict)

        with pytest.raises(DocumentVersionConflict) as info:
            store.documents.insert_document_graph(_document(), 3, False)

        assert str(info.value) == VERSION_CONFLICT_MESSAGE
        assert info.value.__cause__ is conflict
        assert not isinstance(info.value, IngestionError)
        assert driver.rollbacks == 1 and driver.commits == 0
        assert driver.attempts == 1  # a constraint error is not retried

    def test_other_constraint_error_is_a_constraint_error(self, failing_store):
        other = _hydrate(
            "Neo.ClientError.Schema.ConstraintValidationFailed", "Node already exists with label `Theme`"
        )
        store, _ = failing_store(other)

        with pytest.raises(GraphConstraintError) as info:
            store.documents.insert_document_graph(_document(), 3, False)

        assert type(info.value) is GraphConstraintError

    def test_connection_failure_is_retried_and_one_driver_serves_every_attempt(self, failing_store):
        store, driver = failing_store(ServiceUnavailable("down"))

        with patch("scinr.newton.graph.neo4j.retry.time.sleep"):
            with pytest.raises(GraphConnectionError):
                store.documents.insert_document_graph(_document(), 1, False)

        assert driver.attempts == TOTAL_RETRIES + 1
        assert driver.closed == 1


class TestEnsureIndexesIsBestEffort:
    @pytest.mark.parametrize("error", [_TRANSIENT, _SYNTAX], ids=["transient", "non-transient"])
    def test_engine_errors_are_logged_not_raised(self, failing_store, error, caplog):
        store, _ = failing_store(error)

        assert store.schema.ensure_indexes(("idx_model_decision_uid",)) == 0
        assert "Could not ensure indexes" in caplog.text

    def test_unknown_index_name_passes_through_untranslated(self, failing_store):
        store, _ = failing_store(_SYNTAX)

        with pytest.raises(ValueError, match="Unknown index"):
            store.schema.ensure_indexes(("nope",))


class TestExportSnapshotCallbackSemantics:
    """``export_snapshot`` retries the whole attempt (the callback opens the file inside it)
    and translates on the way out; what is not an engine error passes through."""

    def test_callback_error_is_not_retried_nor_translated(self, failing_store):
        store, driver = failing_store(_TRANSIENT)
        attempts = []

        def attempt(reader):
            attempts.append(1)
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            store.freeze.export_snapshot([_doc_info()], "acme", attempt)
        assert attempts == [1]
        assert driver.attempts == 0

    def test_engine_error_while_iterating_restarts_the_attempt_then_is_translated(
        self, failing_store
    ):
        store, driver = failing_store(_TRANSIENT)
        starts = []
        doc = _doc_info()

        def attempt(reader):
            starts.append(1)  # the caller opens its output file here
            return list(reader.subtree_nodes(doc))

        with patch("scinr.newton.graph.neo4j.retry.time.sleep"):
            with pytest.raises(TransientGraphError) as info:
                store.freeze.export_snapshot([doc], "acme", attempt)
        assert len(starts) == TOTAL_RETRIES + 1
        assert driver.attempts == TOTAL_RETRIES + 1
        assert info.value.__cause__ is _TRANSIENT

    def test_non_transient_engine_error_is_not_retried(self, failing_store):
        store, driver = failing_store(_SYNTAX)

        with pytest.raises(GraphQueryError):
            _export_everything(store)
        assert driver.attempts == 1


class _FailingBatchSession:
    def __init__(self, driver: _FailingBatchDriver) -> None:
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute_write(self, fn, *args):
        self._driver.attempts += 1
        raise self._driver.error


class _FailingBatchDriver:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0

    def session(self, **kwargs):
        return _FailingBatchSession(self)


@pytest.mark.parametrize(
    ("error", "expected"),
    [(_TRANSIENT, TransientGraphError), (_SYNTAX, GraphQueryError)],
    ids=["transient", "non-transient"],
)
class TestRestoreWriterErrors:
    """A failed batch is kept; ``drain()`` (or the next submit) translates and raises the first one."""

    @pytest.fixture
    def writer(self, monkeypatch, error):
        from types import SimpleNamespace

        from scinr.newton.graph.neo4j import restore as restore_mod
        from scinr.newton.graph.neo4j.store import Neo4jGraphStore

        driver = _FailingBatchDriver(error)
        monkeypatch.setattr(restore_mod, "get_config", lambda: SimpleNamespace(neo4j_database="neo4j"))
        monkeypatch.setattr(restore_mod, "get_async_driver", lambda: driver)
        return Neo4jGraphStore().restore.batch_writer(2), driver

    async def test_drain_translates_the_first_saved_error(self, writer, error, expected):
        batch_writer, driver = writer
        await batch_writer.merge_nodes(("nodes", ("InfoUnit",)), ("InfoUnit",), [], lambda _: None)

        with pytest.raises(GraphError) as info:
            await batch_writer.drain()
        assert type(info.value) is expected
        _assert_clean(info.value)
        assert info.value.__cause__ is error
        assert str(info.value) == str(error)
        assert driver.attempts == 1  # the writer never retried; neither does it now

    async def test_the_next_submit_raises_the_translated_error_too(self, writer, error, expected):
        batch_writer, _ = writer
        await batch_writer.delete_stale(
            ("stale",), ["n1"], model_decisions=True, extraction_results=False
        )
        with pytest.raises(expected):
            await batch_writer.drain()
        with pytest.raises(expected):
            await batch_writer.create_relationships(
                ("relationships", ("T", "A", "B")), ("T", "InfoUnit", "InfoUnit"), [], lambda _: None
            )

    async def test_abort_never_raises_the_saved_error(self, writer, error, expected):
        batch_writer, _ = writer
        await batch_writer.merge_nodes(("nodes", ("InfoUnit",)), ("InfoUnit",), [], lambda _: None)

        await batch_writer.abort()
