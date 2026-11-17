"""
tests/unit/graph_neo4j/test_documents_ops.py — ``Neo4jDocumentOps`` (``store.documents``)
and ``Neo4jAnnotationOps.count_annotated_nodes`` against a fake driver.

The fakes record every statement and transaction boundary, so the tests pin the
Cypher, the parameters, the transaction scope and how many drivers are opened.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scinr.newton.exceptions import GraphQueryError
from scinr.newton.graph import reset_graph_store, set_graph_store
from scinr.newton.graph.neo4j import annotation as annotation_mod
from scinr.newton.graph.neo4j import documents as documents_mod
from scinr.newton.graph.neo4j import driver as driver_mod
from scinr.newton.graph.neo4j.store import Neo4jGraphStore
from scinr.newton.graph.types import DocumentSelector, LeafDocument
from scinr.newton.ingest import loader
from scinr.newton.models.document_structure import Document


class _Result:
    def __init__(self, single=None, data=None, rows=None):
        self._single, self._data, self._rows = single, data or [], rows or []

    def single(self):
        return self._single

    def data(self):
        return self._data

    def __iter__(self):
        return iter(self._rows)


class _SyncSession:
    """Sync session and transaction in one: records statements and boundaries."""

    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin_transaction(self):
        self.driver.events.append("begin")
        return self

    def commit(self):
        self.driver.events.append("commit")

    def rollback(self):
        self.driver.events.append("rollback")

    def run(self, query, **params):
        self.driver.calls.append((query, params))
        self.driver.events.append("run")
        if self.driver.fail_on and self.driver.fail_on in query:
            raise self.driver.error
        return self.driver.respond(query, params)


class _AsyncSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.driver.calls.append((query, params))
        result = self.driver.respond(query, params)

        async def single():
            return result.single()

        async def data():
            return result.data()

        return SimpleNamespace(single=single, data=data)


class _Driver:
    def __init__(self, respond=None, fail_on=None, error=None):
        self.calls: list[tuple[str, dict]] = []
        self.events: list[str] = []
        self.respond = respond or (lambda q, p: _Result())
        self.fail_on, self.error = fail_on, error
        self.closed = 0

    def session(self, **kwargs):
        self.database = kwargs.get("database")
        return _SyncSession(self)

    def asession(self):
        return _AsyncSession(self)

    def close(self):
        self.closed += 1


class _AsyncDriverAdapter:
    def __init__(self, driver):
        self.driver = driver

    def session(self, **kwargs):
        return self.driver.asession()


@pytest.fixture
def env(monkeypatch):
    """The real store with a recording sync driver (``drivers`` counts how many are opened)."""
    cfg = SimpleNamespace(neo4j_database="graphdb")
    for module in (documents_mod, annotation_mod):
        monkeypatch.setattr(module, "get_config", lambda: cfg)

    state = SimpleNamespace(driver=_Driver(), drivers=0)

    def _get_driver():
        state.drivers += 1
        return state.driver

    monkeypatch.setattr(driver_mod, "get_driver", _get_driver)
    monkeypatch.setattr(
        driver_mod, "get_async_driver", lambda: _AsyncDriverAdapter(state.driver)
    )
    store = Neo4jGraphStore()
    set_graph_store(store)
    state.store = store
    yield state
    reset_graph_store()


def _doc(**kw) -> Document:
    return Document(
        document_name="doc",
        document_type="pdf",
        doc_path="F/doc",
        raw_file_id="",
        document_structure=[],
        **kw,
    )


class TestVersions:
    def test_next_version(self, env):
        env.driver.respond = lambda q, p: _Result(single={"max_version": 4})
        assert env.store.documents.next_version("F/doc", "acme") == 5
        query, params = env.driver.calls[0]
        assert "MATCH (d:Document {tenant_id: $tenant_id, path: $path})" in query
        assert params == {"tenant_id": "acme", "path": "F/doc"}
        assert env.driver.database == "graphdb"
        assert env.driver.closed == 1  # a stand-alone call closes its driver

    def test_next_version_without_documents_is_one(self, env):
        env.driver.respond = lambda q, p: _Result(single={"max_version": None})
        assert env.store.documents.next_version("p", "acme") == 1

    def test_current_latest_version(self, env):
        env.driver.respond = lambda q, p: _Result(single={"version": 2})
        assert env.store.documents.current_latest_version("p", "acme") == 2
        env.driver.respond = lambda q, p: _Result(single=None)
        assert env.store.documents.current_latest_version("p", "acme") is None

    @pytest.mark.parametrize(("update_mode", "expected"), [(False, 4), (True, 3)])
    def test_resolve_batch_version(self, env, update_mode, expected):
        env.driver.respond = lambda q, p: _Result(single={"max_version": 3})
        assert env.store.documents.resolve_batch_version(["a", "a/b"], [None], update_mode) == expected
        query, params = env.driver.calls[0]
        assert ("latest: true" in query) is update_mode
        assert params == {"tenant_ids": ["__public__"], "paths": ["a", "a/b"]}

    def test_resolve_batch_version_without_paths_runs_no_query(self, env):
        assert env.store.documents.resolve_batch_version([], ["acme"], False) == 1
        assert env.driver.calls == []

    def test_sync_scope_shares_one_driver(self, env):
        env.driver.respond = lambda q, p: _Result(single={"max_version": None})
        with env.store.sync_scope():
            env.store.documents.next_version("a", "acme")
            env.store.documents.next_version("b", "acme")
            assert env.driver.closed == 0
        assert env.drivers == 1
        assert env.driver.closed == 1


class TestInsertDocumentGraph:
    def test_commits_one_explicit_transaction(self, env):
        env.store.documents.insert_document_graph(_doc(), 2, False)
        events = env.driver.events
        assert events[0] == "begin" and events[-1] == "commit"
        assert "rollback" not in events
        assert env.driver.closed == 1

    def test_failure_rolls_back_and_is_translated(self, env):
        from neo4j.exceptions import ClientError

        env.driver.fail_on = "MERGE (d:Document"
        env.driver.error = ClientError("bad cypher")

        with pytest.raises(GraphQueryError):
            env.store.documents.insert_document_graph(_doc(), 2, False)

        assert env.driver.events[-1] == "rollback"
        assert "commit" not in env.driver.events

    def test_errors_that_are_not_the_engines_pass_through(self, env):
        with pytest.raises(ValueError):
            env.store.documents.insert_document_graph(_doc(tenant_id=""), 2, False)
        assert env.driver.events[-1] == "rollback"


class TestCreateTabularDocument:
    def _create(self, env, doc_path="Folder/data.csv"):
        env.store.documents.create_tabular_document(
            document_name="data.csv",
            doc_path=doc_path,
            version=3,
            raw_file_id="raw-1",
            tenant="acme",
            created_by_user_id="u1",
            job_id="j1",
        )

    def test_one_transaction_with_hierarchy_document_versioning_and_link(self, env):
        self._create(env)
        events = env.driver.events
        assert events[0] == "begin" and events[-1] == "commit"
        queries = " ".join(q for q, _ in env.driver.calls)
        for needle in ("MERGE (f:Document", "MERGE (d:Document", "HAS_NEWER_VERSION", "IS_COMPOSED_OF"):
            assert needle in queries
        document = next(p for q, p in env.driver.calls if "MERGE (d:Document" in q)
        assert document["raw_file_id"] == "raw-1" and document["tenant_id"] == "acme"
        assert document["context_instructions"] is None

    def test_root_file_creates_no_folders(self, env):
        self._create(env, doc_path="data.csv")
        assert not any("MERGE (f:Document" in q for q, _ in env.driver.calls)

    def test_failure_rolls_back_and_raises(self, env):
        from neo4j.exceptions import ClientError

        env.driver.fail_on = "HAS_NEWER_VERSION"
        env.driver.error = ClientError("boom")

        with pytest.raises(GraphQueryError):
            self._create(env)

        assert env.driver.events[-1] == "rollback"
        assert "commit" not in env.driver.events


class TestReplacement:
    def test_preflight_replaces_returns_the_rows(self, env):
        env.driver.respond = lambda q, p: _Result(rows=[{"path": "p", "version": 1}])
        assert env.store.documents.preflight_replaces("old", "acme") == [{"path": "p", "version": 1}]
        assert env.driver.calls[0][1] == {"tenant_id": "acme", "name": "old"}

    def test_apply_replacement_without_new_roots_writes_nothing(self, env, caplog):
        env.store.documents.apply_replacement("old", ["new"], tenant="acme", replaced_path=None)
        assert len(env.driver.calls) == 1  # only the lookup of the new roots
        assert "could not find any root documents" in caplog.text

    def test_apply_replacement_selects_the_old_document_by_name_without_a_path(self, env):
        answers = iter([_Result(rows=[{"path": "n", "version": 2, "name": "new"}]), _Result()])
        env.driver.respond = lambda q, p: next(answers)
        env.store.documents.apply_replacement("old", ["new"], tenant="acme", replaced_path=None)
        query, params = env.driver.calls[1]
        assert "name: $old_name" in query and "path: $old_path" not in query
        assert params["old_name"] == "old" and params["new_version"] == 2


class TestLatestSelector:
    async def test_count_latest_documents(self, env):
        env.driver.respond = lambda q, p: _Result(single={"n": 2})
        selector = DocumentSelector(tenant_id="acme", path="F/doc", latest=True)
        assert await env.store.documents.count_latest_documents(selector) == 2
        query, params = env.driver.calls[0]
        assert query == (
            "MATCH (d:Document {tenant_id: $tenant_id, path: $doc_selector, latest: true}) "
            "RETURN count(d) AS n"
        )
        assert params == {"tenant_id": "acme", "doc_selector": "F/doc"}

    async def test_by_name_for_the_public_tenant(self, env):
        env.driver.respond = lambda q, p: _Result(single={"n": 1})
        selector = DocumentSelector(tenant_id=None, document_name="doc", latest=True)
        await env.store.documents.count_latest_documents(selector)
        query, params = env.driver.calls[0]
        assert "name: $doc_selector" in query
        assert params == {"tenant_id": "__public__", "doc_selector": "doc"}

    @pytest.mark.parametrize(
        "selector",
        [
            DocumentSelector(tenant_id="acme", path="p"),
            DocumentSelector(tenant_id="acme", latest=True),
            DocumentSelector(tenant_id="", path="p", latest=True),
        ],
        ids=["not-latest", "no-path-nor-name", "empty-tenant"],
    )
    async def test_invalid_selector_is_a_value_error_and_runs_nothing(self, env, selector):
        with pytest.raises(ValueError):
            await env.store.documents.count_latest_documents(selector)
        assert env.driver.calls == []

    async def test_count_annotated_nodes(self, env):
        env.driver.respond = lambda q, p: _Result(single={"n": 0})
        selector = DocumentSelector(tenant_id="acme", path="F/doc", latest=True)
        assert await env.store.annotation.count_annotated_nodes(selector) == 0
        query, params = env.driver.calls[0]
        assert query == (
            "MATCH (d:Document {tenant_id: $tenant_id, path: $doc_selector, latest: true})"
            "-[:HAS_STRUCTURE|HAS_CHILD*1..]->(sn:StructureNode)"
            "-[:HAS_MODEL_DECISION]->() RETURN count(sn) AS n"
        )
        assert params == {"tenant_id": "acme", "doc_selector": "F/doc"}


class TestLeafResolution:
    _ROWS = [
        {"docs": [{"name": "a", "path": "F/a"}]},
        {"docs": [{"name": "b", "path": "F/b"}, {"name": "a", "path": "F/a"}]},
    ]

    def test_sync_deduplicates_by_path(self, env):
        env.driver.respond = lambda q, p: _Result(data=self._ROWS)
        selector = DocumentSelector(tenant_id="acme", path="F", latest=True)
        assert env.store.documents.resolve_leaf_documents(selector) == [
            LeafDocument("a", "F/a"),
            LeafDocument("b", "F/b"),
        ]
        assert env.driver.calls[0][1] == {"tenant_id": "acme", "doc_selector": "F"}

    async def test_async_matches_the_sync_result(self, env):
        env.driver.respond = lambda q, p: _Result(data=self._ROWS)
        selector = DocumentSelector(tenant_id="acme", path="F", latest=True)
        assert await env.store.documents.resolve_leaf_documents_async(selector) == [
            LeafDocument("a", "F/a"),
            LeafDocument("b", "F/b"),
        ]

    def test_nothing_found_is_an_empty_list_and_the_resolver_falls_back(self, env):
        from scinr.newton.utils.document_resolver import resolve_leaf_documents

        env.driver.respond = lambda q, p: _Result(data=[])
        selector = DocumentSelector(tenant_id="acme", path="F/doc", latest=True)
        assert env.store.documents.resolve_leaf_documents(selector) == []
        assert resolve_leaf_documents(tenant_id="acme", doc_path="F/doc") == [
            LeafDocument("doc", "F/doc")
        ]


class TestLoaderUsesOneConnection:
    def test_load_file_resolves_the_version_and_inserts_with_one_driver(self, env, tmp_path):
        env.driver.respond = lambda q, p: _Result(single={"max_version": 6})
        path = tmp_path / "extract-doc.json"
        path.write_text(_doc().model_dump_json(), encoding="utf-8")

        assert loader.load_file(path) == "doc"

        assert env.drivers == 1
        assert env.driver.closed == 1
        assert env.driver.calls[0][1] == {"tenant_id": "__public__", "path": "F/doc"}
        assert env.driver.events[-1] == "commit"
        version = next(p["version"] for q, p in env.driver.calls if "MERGE (d:Document" in q)
        assert version == 7

    def test_load_documents_shares_the_driver_and_the_batch_version(self, env):
        env.driver.respond = lambda q, p: _Result(single={"max_version": 1})

        names = loader.load_documents([_doc(), _doc()], tenant_id="acme")

        assert names == ["doc", "doc"]
        assert env.drivers == 1
        assert env.driver.events.count("commit") == 2
        assert sum("RETURN max(d.version)" in q for q, _ in env.driver.calls) == 1
