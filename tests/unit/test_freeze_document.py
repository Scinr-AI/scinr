"""
tests/unit/test_freeze_document.py — Unit tests for
scinr.newton.ingest.freeze: freeze_document() and export_document_snapshot().

No real Neo4j or MongoDB: a fake sync driver serves canned rows keyed off the
query text (the same style as tests/unit/test_delete_document.py) and a fake
FreezeRepository captures the snapshot file at upload time.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neo4j.time import DateTime

from scinr.newton.exceptions import ConfigurationError, FreezeError
from scinr.newton.ingest import _cascade, _gc, freeze
from scinr.newton.ingest._gc import GC_REACHABILITY_MAX_HOPS
from scinr.newton.ingest.freeze import (
    SNAPSHOT_SCHEMA_VERSION,
    export_document_snapshot,
    freeze_document,
)
from scinr.newton.results import FreezeResult

# ---------------------------------------------------------------------------
# Fake graph / driver
# ---------------------------------------------------------------------------


def _doc(path: str, version: int, *, tenant: str = "acme", seed: bool = True, **props) -> dict:
    return {
        "properties": {"path": path, "version": version, "tenant_id": tenant, **props},
        "is_seed": seed,
    }


def _node(labels: list[str], **props) -> dict:
    return {"labels": labels, "properties": props}


def _key(n: dict) -> dict:
    p = n["properties"]
    return {k: p.get(k) for k in ("id", "uid", "name", "model", "label", "path", "tenant_id", "version")}


def _rel(rel_type: str, source: dict, target: dict, **props) -> dict:
    return {
        "type": rel_type,
        "from_labels": source["labels"],
        "from_key": _key(source),
        "to_labels": target["labels"],
        "to_key": _key(target),
        "properties": props,
    }


class _FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None

    def consume(self):
        return None


class _FakeTx:
    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params) -> _FakeResult:
        d = self.driver
        if query in _ORPHAN_DELETE_QUERIES:
            # Scoped GC: the orphans just read are deleted (a managed write
            # transaction, logged apart from the stub marking).
            d.calls.append(("gc.run", query, params))
            deleted = [d.orphans.pop(i) for i in params["ids"] if i in d.orphans]
            return _FakeResult([{"n": len(deleted)}])
        d.calls.append(("tx.run", query, params))
        if query == freeze._FREEZE_GUARD_QUERY:
            return _FakeResult([d.guard or {"found": len(params["keys"]), "frozen": 0}])
        if query in (freeze._MARK_FROZEN_QUERY, freeze._MARK_BACKUP_QUERY):
            return _FakeResult([{"n": len(params["keys"])}])
        raise AssertionError(f"Unexpected tx.run query: {query}")

    def commit(self):
        self.driver.calls.append(("tx.commit", "", {}))

    def rollback(self):
        self.driver.calls.append(("tx.rollback", "", {}))


class _FakeSession:
    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params) -> _FakeResult:
        d = self.driver
        d.calls.append(("session.run", query, params))
        key = (params.get("path"), params.get("version"))
        if freeze._DOCUMENT_SET_TAIL in query:
            return _FakeResult(d.docs)
        if query == freeze._SUBTREE_NODES_EXPORT_QUERY:
            if d.export_error is not None:
                raise d.export_error
            return _FakeResult(d.nodes.get(key, []))
        if query == freeze._DOCUMENT_RELATIONSHIPS_EXPORT_QUERY:
            return _FakeResult(d.doc_rels.get(key, []))
        if query == freeze._SUBTREE_RELATIONSHIPS_EXPORT_QUERY:
            return _FakeResult(d.rels.get(key, []))
        if query == _gc.GC_SEEDS_QUERY:
            return _FakeResult(d.seed_rows)
        if query in _CANDIDATE_QUERIES:
            # Scoped GC: the orphans among a batch of candidates, each with
            # what it points at.
            if d.gc_error is not None:
                raise d.gc_error
            return _FakeResult(
                [{"id": i, "dependents": d.orphans[i]} for i in params["ids"] if i in d.orphans]
            )
        if query == _gc.GC_ENTITY_MODEL_INSTANCE_QUERY:
            if d.gc_error is not None:
                raise d.gc_error
            return _FakeResult([{"borrados": d.gc_emi.pop(0) if d.gc_emi else 0}])
        if query == _gc.GC_LABELED_ENTITY_QUERY:
            return _FakeResult([{"borrados": 0}])
        if query == freeze._CLEAR_PENDING_QUERY:
            return _FakeResult([{"n": len(params["keys"])}])
        if query in _STEP_QUERIES:
            if d.step_error is not None and "DETACH DELETE" in query:
                raise d.step_error
            return _FakeResult([{"n": d.step_count}])
        raise AssertionError(f"Unexpected session.run query: {query}")

    def begin_transaction(self):
        return _FakeTx(self.driver)

    def execute_write(self, fn):
        return fn(_FakeTx(self.driver))


class _FakeDriver:
    def __init__(self, docs=None, nodes=None, rels=None, doc_rels=None) -> None:
        self.docs = docs or []
        self.nodes = nodes or {}
        self.rels = rels or {}
        self.doc_rels = doc_rels or {}
        self.guard: dict | None = None
        self.step_count = 3
        self.step_error: Exception | None = None
        self.gc_error: Exception | None = None
        self.export_error: Exception | None = None
        # Tenant sweep (a resumed freeze): nodes deleted by each iteration.
        self.gc_emi: list[int] = [2, 0]
        # Scoped GC: what the ExtractionResults point at, and the orphans
        # (element id -> [[id, labeled], ...] it points at).
        self.seed_rows = [{"id": "mi-1", "labeled": False}, {"id": "mi-shared", "labeled": False}]
        self.orphans: dict[str, list] = {"mi-1": [["le-1", True]], "le-1": []}
        self.calls: list[tuple[str, str, dict]] = []
        self.closed = False

    def session(self, **kwargs):
        return _FakeSession(self)

    def close(self):
        self.closed = True

    def tx_queries(self) -> list[str]:
        return [q for kind, q, _ in self.calls if kind == "tx.run"]

    def write_queries(self) -> list[str]:
        """Everything run after the export, in order: the stub marking
        (tx.run), the batched cleanup and pending clear (session.run) and the
        GC (session.run; its deletes are gc.run)."""
        return [
            q
            for kind, q, _ in self.calls
            if kind.endswith(".run") and not _is_export_read(q)
        ]

    def step_queries(self) -> list[str]:
        return [q for kind, q, _ in self.calls if kind == "session.run" and q in _STEP_QUERIES]


# Every query the batched cleanup can run, whatever the keep_* flags.
_STEP_QUERIES = {
    query
    for ksn in (False, True)
    for ka in (False, True)
    for ker in (False, True)
    for _, query in _cascade.subtree_steps(ksn, ka, ker)
}


_CANDIDATE_QUERIES = (
    _gc.GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY,
    _gc.GC_LABELED_ENTITY_ORPHANS_QUERY,
)
_ORPHAN_DELETE_QUERIES = (
    _gc.GC_ENTITY_MODEL_INSTANCE_DELETE_QUERY,
    _gc.GC_LABELED_ENTITY_DELETE_QUERY,
)


def _is_export_read(query: str) -> bool:
    return freeze._DOCUMENT_SET_TAIL in query or query in (
        freeze._SUBTREE_NODES_EXPORT_QUERY,
        freeze._DOCUMENT_RELATIONSHIPS_EXPORT_QUERY,
        freeze._SUBTREE_RELATIONSHIPS_EXPORT_QUERY,
    )


class _FakeFreezeRepo:
    def __init__(self, error: Exception | None = None) -> None:
        self.stored: list[dict[str, Any]] = []
        self.error = error

    async def store_snapshot(self, path, *, tenant_id, created_by_user_id=None, job_id=None, metadata):
        if self.error is not None:
            raise self.error
        self.stored.append(
            {
                "path": Path(path),
                "snapshot": json.loads(Path(path).read_text(encoding="utf-8")),
                "tenant_id": tenant_id,
                "created_by_user_id": created_by_user_id,
                "job_id": job_id,
                "metadata": metadata,
            }
        )
        return f"blob-{len(self.stored)}"


# One document with a small but complete subtree.
_ROOT = _node(["StructureNode", "Section"], id="acme|a.pdf|1|root", tenant_id="acme", title="R")
_CHILD = _node(["StructureNode", "Paragraph"], id="acme|a.pdf|1|p1", tenant_id="acme")
_IU = _node(["InfoUnit"], uid="iu-1", tenant_id="acme", text="hello")
_MD = _node(["ModelDecision"], uid="md-1", tenant_id="acme", confidence=0.5)
_ER = _node(["ExtractionResult"], uid="er-1", tenant_id="acme")
_MI = _node(["ModelInstance", "Product"], uid="mi-1", tenant_id="acme", job_ids=["j1"])
_CATALOG = _node(["CatalogModel"], name="Product")
_DOCUMENT_NODE = _node(["Document"], path="a.pdf", version=1, tenant_id="acme")
_OUTSIDE = _node(["ModelInstance"], uid="mi-far", tenant_id="acme")  # beyond the hop bound


def _graph(**doc_props) -> _FakeDriver:
    key = ("a.pdf", 1)
    return _FakeDriver(
        docs=[_doc("a.pdf", 1, **doc_props)],
        nodes={key: [_ROOT, _CHILD, _IU, _MD, _ER, _MI, _MI]},  # _MI twice: two paths
        doc_rels={key: [_rel("HAS_STRUCTURE", _DOCUMENT_NODE, _ROOT)]},
        rels={
            key: [
                _rel("HAS_CHILD", _ROOT, _CHILD, order=0),
                _rel("HAS_INFO_UNIT", _CHILD, _IU),
                _rel("HAS_MODEL_DECISION", _CHILD, _MD),
                _rel("MATCHED_MODEL", _MD, _CATALOG),
                _rel("HAS_EXTRACTION", _CHILD, _ER),
                _rel("USES_PRIMARY_MODEL", _ER, _CATALOG),
                _rel("HAS_INSTANCE", _ER, _MI),
                _rel("RELATED", _MI, _OUTSIDE),
            ]
        },
    )


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    monkeypatch.setattr(freeze, "get_config", lambda: SimpleNamespace(neo4j_database="neo4j"))


@pytest.fixture
def env(monkeypatch):
    def _setup(driver: _FakeDriver, repo: _FakeFreezeRepo | None = None):
        repo = repo or _FakeFreezeRepo()
        monkeypatch.setattr(freeze, "get_driver", lambda: driver)
        monkeypatch.setattr("scinr.newton.freeze.factory.get_freeze_storage", lambda: repo)
        return driver, repo

    return _setup


# ---------------------------------------------------------------------------
# export_document_snapshot()
# ---------------------------------------------------------------------------


class TestExportSnapshot:
    async def test_dict_snapshot_shape(self, env):
        driver, _ = env(_graph())

        snap = await export_document_snapshot("a.pdf", 1, tenant_id="acme")

        assert snap["schema_version"] == SNAPSHOT_SCHEMA_VERSION
        assert snap["tenant_id"] == "acme"
        assert snap["keep_flags"] is None
        assert snap["mode"] == "export"
        assert isinstance(snap["frozen_at"], str)
        [entry] = snap["documents"]
        assert entry["document"] == {"path": "a.pdf", "version": 1, "tenant_id": "acme"}
        assert driver.closed

    async def test_nodes_are_deduplicated_and_keys_split_from_properties(self, env):
        env(_graph())
        snap = await export_document_snapshot("a.pdf", 1, tenant_id="acme")
        nodes = snap["documents"][0]["nodes"]

        assert len(nodes) == 6  # _MI returned twice, emitted once
        root = nodes[0]
        assert root["labels"] == ["StructureNode", "Section"]
        assert root["key"] == {"id": "acme|a.pdf|1|root"}
        assert root["properties"] == {"tenant_id": "acme", "title": "R"}
        mi = next(n for n in nodes if n["key"] == {"uid": "mi-1"})
        assert mi["labels"] == ["ModelInstance", "Product"]
        assert not any(n["labels"][0] == "CatalogModel" for n in nodes)

    async def test_links_to_other_documents_are_kept(self, env):
        driver = _graph(job_id="j1", created_by_user_id="u1")
        folder = _node(["Document"], path="folder", version=1, tenant_id="acme")
        older = _node(["Document"], path="a.pdf", version=0, tenant_id="acme")
        driver.doc_rels[("a.pdf", 1)] += [
            _rel("IS_COMPOSED_OF", folder, _DOCUMENT_NODE),
            _rel("HAS_NEWER_VERSION", older, _DOCUMENT_NODE),
        ]
        _, repo = env(driver)
        await export_document_snapshot("a.pdf", 1, tenant_id="acme", destination="storage")
        [stored] = repo.stored
        rels = stored["snapshot"]["documents"][0]["relationships"]
        composed = next(r for r in rels if r["type"] == "IS_COMPOSED_OF")
        assert composed["from"]["key"] == {"tenant_id": "acme", "path": "folder", "version": 1}
        assert composed["to"]["key"] == {"tenant_id": "acme", "path": "a.pdf", "version": 1}
        assert any(r["type"] == "HAS_NEWER_VERSION" for r in rels)
        assert stored["metadata"]["documents"] == [
            {"path": "a.pdf", "version": 1, "job_id": "j1", "created_by_user_id": "u1"}
        ]

    def test_document_links_are_tenant_scoped(self):
        query = freeze._DOCUMENT_RELATIONSHIPS_EXPORT_QUERY
        assert "m.tenant_id = $tenant_id" in query and "n.tenant_id = $tenant_id" in query
        assert "IS_COMPOSED_OF|HAS_NEWER_VERSION" in query

    async def test_relationships_keep_catalog_refs_and_drop_outside_targets(self, env):
        env(_graph())
        snap = await export_document_snapshot("a.pdf", 1, tenant_id="acme")
        rels = snap["documents"][0]["relationships"]

        types = [r["type"] for r in rels]
        assert types[0] == "HAS_STRUCTURE"
        assert rels[0]["from"] == {
            "labels": ["Document"],
            "key": {"tenant_id": "acme", "path": "a.pdf", "version": 1},
        }
        assert "MATCHED_MODEL" in types and "USES_PRIMARY_MODEL" in types
        matched = next(r for r in rels if r["type"] == "MATCHED_MODEL")
        assert matched["to"] == {"labels": ["CatalogModel"], "key": {"name": "Product"}}
        assert "RELATED" not in types  # target beyond the entry's nodes
        assert next(r for r in rels if r["type"] == "HAS_CHILD")["properties"] == {"order": 0}

    async def test_file_destination_matches_dict(self, env, tmp_path):
        env(_graph())
        as_dict = await export_document_snapshot("a.pdf", 1, tenant_id="acme")
        dest = await export_document_snapshot(
            "a.pdf", 1, tenant_id="acme", destination="file", file_path=tmp_path / "s.json"
        )
        assert dest == tmp_path / "s.json"
        assert json.loads(dest.read_text())["documents"] == as_dict["documents"]

    async def test_storage_destination_uploads_and_removes_temp_file(self, env):
        _, repo = env(_graph())
        blob = await export_document_snapshot(
            "a.pdf", 1, tenant_id="acme", job_id=None, destination="storage"
        )
        assert blob == "blob-1"
        [stored] = repo.stored
        assert stored["metadata"]["mode"] == "export"
        assert stored["metadata"]["documents"] == [
            {"path": "a.pdf", "version": 1, "job_id": None, "created_by_user_id": None}
        ]
        assert stored["snapshot"]["documents"][0]["document"]["path"] == "a.pdf"
        assert not stored["path"].exists()

    async def test_storage_destination_without_backend(self, env, monkeypatch):
        driver, _ = env(_graph())

        def _none():
            raise ConfigurationError("none")

        monkeypatch.setattr("scinr.newton.freeze.factory.get_freeze_storage", _none)
        with pytest.raises(ConfigurationError):
            await export_document_snapshot("a.pdf", tenant_id="acme", destination="storage")
        assert not any(q == freeze._SUBTREE_NODES_EXPORT_QUERY for _, q, _ in driver.calls)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"destination": "s3"},
            {"destination": "file"},
            {"destination": "dict", "file_path": "x.json"},
        ],
    )
    async def test_invalid_destination_arguments(self, env, kwargs):
        driver, _ = env(_graph())
        with pytest.raises(ValueError):
            await export_document_snapshot("a.pdf", tenant_id="acme", **kwargs)
        assert driver.calls == []

    async def test_no_match_raises(self, env):
        env(_FakeDriver())
        with pytest.raises(FreezeError, match="no Document found"):
            await export_document_snapshot("missing.pdf", tenant_id="acme")

    async def test_frozen_document_is_refused(self, env):
        env(_graph(frozen=True, frozen_blob_id="b"))
        with pytest.raises(FreezeError, match="already frozen"):
            await export_document_snapshot("a.pdf", tenant_id="acme")

    @pytest.mark.parametrize("tenant", ["other", None])
    async def test_node_of_another_or_no_tenant_raises(self, env, tenant):
        driver = _graph()
        bad = _node(["InfoUnit"], uid="iu-x", tenant_id=tenant)
        driver.nodes[("a.pdf", 1)].append(bad)
        env(driver)
        with pytest.raises(FreezeError, match="tenant"):
            await export_document_snapshot("a.pdf", 1, tenant_id="acme")

    async def test_node_without_key_raises(self, env):
        driver = _graph()
        driver.nodes[("a.pdf", 1)].append(_node(["InfoUnit"], tenant_id="acme"))
        env(driver)
        with pytest.raises(FreezeError, match="business key"):
            await export_document_snapshot("a.pdf", 1, tenant_id="acme")

    async def test_unknown_node_type_raises(self, env):
        driver = _graph()
        driver.nodes[("a.pdf", 1)].append(_node(["Mystery"], tenant_id="acme"))
        env(driver)
        with pytest.raises(FreezeError, match="no known business key"):
            await export_document_snapshot("a.pdf", 1, tenant_id="acme")

    async def test_descendant_of_another_tenant_raises(self, env):
        driver = _graph()
        driver.docs.append(_doc("a.pdf/child", 1, tenant="other", seed=False))
        env(driver)
        with pytest.raises(FreezeError, match="IS_COMPOSED_OF"):
            await export_document_snapshot("a.pdf", tenant_id="acme")

    async def test_failed_file_export_removes_partial_file(self, env, tmp_path):
        driver = _graph()
        driver.export_error = RuntimeError("boom")
        env(driver)
        dest = tmp_path / "s.json"
        with pytest.raises(RuntimeError):
            await export_document_snapshot(
                "a.pdf", tenant_id="acme", destination="file", file_path=dest
            )
        assert not dest.exists()

    async def test_temporal_properties_become_iso_strings(self, env):
        driver = _graph()
        driver.nodes[("a.pdf", 1)] = [
            _node(["InfoUnit"], uid="iu-t", tenant_id="acme", at=DateTime(2026, 1, 2, 3, 4, 5))
        ]
        env(driver)
        snap = await export_document_snapshot("a.pdf", 1, tenant_id="acme")
        assert snap["documents"][0]["nodes"][0]["properties"]["at"].startswith("2026-01-02T03:04:05")

    async def test_folder_exports_every_descendant_entry(self, env):
        driver = _graph()
        driver.docs.append(_doc("a.pdf/child", 1, seed=False))
        env(driver)
        snap = await export_document_snapshot("a.pdf", tenant_id="acme")
        assert [e["document"]["path"] for e in snap["documents"]] == ["a.pdf", "a.pdf/child"]

    def test_subtree_query_is_hop_bounded_and_skips_catalog(self):
        fragment = freeze._SUBTREE_NODES_FRAGMENT
        assert f"[*1..{GC_REACHABILITY_MAX_HOPS}]" in fragment
        for label in freeze.CATALOG_KEY_FIELDS:
            assert f"n:{label}" in fragment
        assert "tenant_id: $tenant_id" in fragment

    async def test_every_query_carries_the_tenant(self, env):
        driver, _ = env(_graph())
        await export_document_snapshot("a.pdf", tenant_id="acme")
        for _, query, params in driver.calls:
            assert params["tenant_id"] == "acme"
            assert "$tenant_id" in query


# ---------------------------------------------------------------------------
# freeze_document()
# ---------------------------------------------------------------------------

_KEEP_COMBINATIONS = [
    (ksn, ka, ker) for ksn in (False, True) for ka in (False, True) for ker in (False, True)
]


def _index(queries: list[str], query: str) -> int:
    return queries.index(query)


class TestFreezeDocument:
    async def test_not_found_touches_nothing(self, env):
        driver, repo = env(_FakeDriver())
        result = await freeze_document("missing.pdf", tenant_id="acme")
        assert isinstance(result, FreezeResult)
        assert result.found is False
        assert result.frozen_blob_id is None
        assert result.documents_frozen == 0
        assert repo.stored == []
        assert driver.write_queries() == []
        assert driver.closed

    async def test_backup_mode_only_marks(self, env):
        driver, repo = env(_graph())
        result = await freeze_document("a.pdf", tenant_id="acme", delete_after_export=False)

        assert result.mode == "backup"
        assert result.found and result.frozen_blob_id == "blob-1"
        assert result.structure_nodes_deleted == 0
        assert driver.tx_queries() == [freeze._MARK_BACKUP_QUERY]
        [(_, _, params)] = [c for c in driver.calls if c[0] == "tx.run"]
        assert params["blob_id"] == "blob-1"
        assert params["tenant_id"] == "acme"
        assert repo.stored[0]["metadata"]["mode"] == "backup"
        assert repo.stored[0]["snapshot"]["mode"] == "backup"

    @pytest.mark.parametrize("flag", ["keep_structure_nodes", "keep_annotations", "keep_extraction_results"])
    async def test_backup_mode_rejects_keep_flags(self, env, flag):
        driver, repo = env(_graph())
        with pytest.raises(ValueError, match="backup"):
            await freeze_document("a.pdf", tenant_id="acme", delete_after_export=False, **{flag: True})
        assert driver.calls == [] and repo.stored == []

    async def test_already_frozen_is_refused_before_export(self, env):
        driver, repo = env(_graph(frozen=True, frozen_blob_id="b0"))
        with pytest.raises(FreezeError, match="already frozen"):
            await freeze_document("a.pdf", tenant_id="acme")
        assert repo.stored == [] and driver.write_queries() == []

    async def test_frozen_descendant_is_refused(self, env):
        driver = _graph()
        driver.docs.append(_doc("a.pdf/child", 1, seed=False, frozen=True))
        _, repo = env(driver)
        with pytest.raises(FreezeError, match="already frozen"):
            await freeze_document("a.pdf", tenant_id="acme")
        assert repo.stored == []

    async def test_export_failure_leaves_the_graph_untouched(self, env):
        driver, _ = env(_graph(), _FakeFreezeRepo(error=RuntimeError("mongo down")))
        with pytest.raises(RuntimeError, match="mongo down"):
            await freeze_document("a.pdf", tenant_id="acme")
        assert driver.write_queries() == []

    @pytest.mark.parametrize("ksn,ka,ker", _KEEP_COMBINATIONS)
    async def test_keep_combinations(self, env, ksn, ka, ker):
        driver, repo = env(_graph())

        result = await freeze_document(
            "a.pdf",
            tenant_id="acme",
            keep_structure_nodes=ksn,
            keep_annotations=ka,
            keep_extraction_results=ker,
        )
        queries = driver.write_queries()
        sub = _cascade.subtree_query
        md_delete = sub(_cascade.MODEL_DECISION_PATTERN, "delete")
        er_delete = sub(_cascade.EXTRACTION_RESULT_PATTERN, "delete")
        iu_delete = sub(_cascade.INFO_UNIT_PATTERN, "delete")

        # The snapshot is always complete, whatever the flags.
        [stored] = repo.stored
        assert len(stored["snapshot"]["documents"][0]["nodes"]) == 6
        assert stored["snapshot"]["mode"] == "freeze"
        assert stored["snapshot"]["keep_flags"] == {
            "structure_nodes": ksn,
            "annotations": ka,
            "extraction_results": ker,
        }

        # The stubs are marked first, in their own transaction: from then on
        # every document points at the snapshot, whatever happens next.
        assert queries[:2] == [freeze._FREEZE_GUARD_QUERY, freeze._MARK_FROZEN_QUERY]
        assert driver.tx_queries() == queries[:2]
        kinds = [kind for kind, _, _ in driver.calls if kind.startswith("tx.")]
        assert kinds == ["tx.run", "tx.run", "tx.commit"]
        assert "d.frozen_cleanup_pending = true" in freeze._MARK_FROZEN_QUERY

        assert iu_delete in queries
        assert (md_delete in queries) is (not ka)
        assert (sub(_cascade.PROPOSED_FIELD_PATTERN, "delete") in queries) is (not ka)
        assert (er_delete in queries) is (not ker)
        assert (_cascade.STRUCTURE_NODES_DELETE_QUERY in queries) is (not ksn)
        assert (_cascade.STRUCTURE_NODES_COUNT_QUERY in queries) is ksn
        assert (_cascade.RELINK_MODEL_DECISIONS_QUERY in queries) is (not ksn and ka)
        assert (_cascade.RELINK_EXTRACTION_RESULTS_QUERY in queries) is (not ksn and ker)

        # Order: relinks before the StructureNodes go, children before parents,
        # then the GC of what the deleted ExtractionResults pointed at (read
        # before they go), and the pending mark is cleared last.
        if not ksn:
            sn = _index(queries, _cascade.STRUCTURE_NODES_DELETE_QUERY)
            assert _index(queries, iu_delete) < sn
            if ka:
                assert _index(queries, _cascade.RELINK_MODEL_DECISIONS_QUERY) < sn
            if ker:
                assert _index(queries, _cascade.RELINK_EXTRACTION_RESULTS_QUERY) < sn
        if not ka:
            assert _index(queries, sub(_cascade.PROPOSED_FIELD_PATTERN, "delete")) < _index(
                queries, sub(_cascade.PROPOSED_MODEL_PATTERN, "delete")
            ) < _index(queries, md_delete)
        steps = driver.step_queries()
        cleanup = queries[2:]
        if not ker:
            assert cleanup[0] == _gc.GC_SEEDS_QUERY
            cleanup = cleanup[1:]
        assert cleanup[: len(steps)] == steps
        tail = cleanup[len(steps) :]
        # Kept ExtractionResults orphan nothing: no candidates, no GC.
        scoped_gc = [
            _gc.GC_ENTITY_MODEL_INSTANCE_ORPHANS_QUERY,
            _gc.GC_ENTITY_MODEL_INSTANCE_DELETE_QUERY,
            _gc.GC_LABELED_ENTITY_ORPHANS_QUERY,
            _gc.GC_LABELED_ENTITY_DELETE_QUERY,
        ]
        assert tail[:-1] == ([] if ker else scoped_gc)
        assert tail[-1] == freeze._CLEAR_PENDING_QUERY
        # Never a scan of the whole tenant.
        assert not any("borrados" in q for q in queries)
        candidates = [p["ids"] for _, q, p in driver.calls if q in _CANDIDATE_QUERIES]
        assert candidates == ([] if ker else [["mi-1", "mi-shared"], ["le-1"]])
        deleted = [p["ids"] for kind, _, p in driver.calls if kind == "gc.run"]
        assert deleted == ([] if ker else [["mi-1"], ["le-1"]])

        mark_params = next(p for kind, q, p in driver.calls if q == freeze._MARK_FROZEN_QUERY)
        assert mark_params["blob_id"] == "blob-1"
        assert (
            mark_params["keep_structure_nodes"],
            mark_params["keep_annotations"],
            mark_params["keep_extraction_results"],
        ) == (ksn, ka, ker)
        assert mark_params["frozen_at"] == stored["snapshot"]["frozen_at"]

        assert result.mode == "freeze"
        assert result.info_units_deleted == 3
        assert result.structure_nodes_deleted == (0 if ksn else 3)
        assert result.structure_nodes_kept == (3 if ksn else 0)
        assert result.model_decisions_deleted == (0 if ka else 3)
        assert result.model_decisions_kept == (3 if ka else 0)
        assert result.proposed_fields_deleted == (0 if ka else 3)
        assert result.extraction_results_deleted == (0 if ker else 3)
        assert result.extraction_results_kept == (3 if ker else 0)
        collected = 0 if ker else 1  # mi-1 and the le-1 it pointed at; mi-shared is in use
        assert result.gc_entity_model_instance_deleted == collected
        assert result.gc_entity_model_instance_passes == collected
        assert result.gc_labeled_entity_deleted == collected
        assert result.gc_labeled_entity_passes == collected

    async def test_every_query_is_tenant_scoped(self, env):
        driver, _ = env(_graph())
        await freeze_document("a.pdf", tenant_id="acme")
        runs = [c for c in driver.calls if c[0].endswith(".run")]
        assert len(runs) > 10
        for _, query, params in runs:
            assert params["tenant_id"] == "acme", query
            assert "$tenant_id" in query

    async def test_public_tenant_uses_the_stored_key(self, env):
        driver = _graph()
        driver.docs = [_doc("a.pdf", 1, tenant="__public__")]
        driver.nodes = {}
        env(driver)
        await freeze_document("a.pdf", tenant_id=None)
        assert {p["tenant_id"] for k, _, p in driver.calls if k.endswith(".run")} == {"__public__"}

    async def test_folder_freezes_descendants_but_reports_seed_versions(self, env):
        driver = _graph()
        driver.docs = [
            _doc("folder", 2),
            _doc("folder", 1),
            _doc("folder/leaf", 7, seed=False),
        ]
        env(driver)
        result = await freeze_document("folder", tenant_id="acme")
        assert result.versions_frozen == [1, 2]
        assert result.documents_frozen == 3
        mark_params = next(p for _, q, p in driver.calls if q == freeze._MARK_FROZEN_QUERY)
        assert mark_params["keys"] == [
            {"path": "folder", "version": 2},
            {"path": "folder", "version": 1},
            {"path": "folder/leaf", "version": 7},
        ]

    async def test_guard_failure_rolls_back(self, env):
        driver = _graph()
        driver.guard = {"found": 1, "frozen": 1}
        env(driver)
        with pytest.raises(FreezeError, match="changed during the export"):
            await freeze_document("a.pdf", tenant_id="acme")
        kinds = [kind for kind, _, _ in driver.calls if kind.startswith("tx.")]
        assert "tx.rollback" in kinds and "tx.commit" not in kinds
        assert driver.write_queries() == [freeze._FREEZE_GUARD_QUERY]

    async def test_step_failure_leaves_frozen_stubs_pending(self, env, caplog):
        """The subtree is deleted in batches, not atomically: when a step
        fails the stubs are already marked (they point at the complete
        snapshot), the GC does not run and the pending mark stays."""
        driver = _graph()
        driver.step_error = RuntimeError("deadlock")
        env(driver)
        with caplog.at_level("ERROR", logger="scinr.newton.ingest.freeze"):
            with pytest.raises(RuntimeError, match="deadlock"):
                await freeze_document("a.pdf", tenant_id="acme")
        kinds = [kind for kind, _, _ in driver.calls if kind.startswith("tx.")]
        assert kinds == ["tx.run", "tx.run", "tx.commit"]  # guard, mark — committed
        queries = driver.write_queries()
        assert not any("borrados" in q or q in _CANDIDATE_QUERIES for q in queries)
        assert freeze._CLEAR_PENDING_QUERY not in queries
        assert "Call freeze_document() again" in caplog.text
        assert "restore_document()" in caplog.text

    async def test_gc_failure_keeps_the_pending_mark(self, env):
        driver = _graph()
        driver.gc_error = RuntimeError("gc failed")
        env(driver)
        with pytest.raises(RuntimeError, match="gc failed"):
            await freeze_document("a.pdf", tenant_id="acme")
        assert freeze._CLEAR_PENDING_QUERY not in driver.write_queries()

    async def test_selector_validation(self, env):
        driver, _ = env(_graph())
        with pytest.raises(ValueError, match="exactly one"):
            await freeze_document(tenant_id="acme")
        with pytest.raises(ValueError, match="exactly one"):
            await freeze_document("a.pdf", job_id="j", tenant_id="acme")
        with pytest.raises(ValueError):
            await freeze_document("a.pdf", tenant_id="")
        with pytest.raises(TypeError):
            await freeze_document("a.pdf")  # type: ignore[call-arg]
        assert driver.calls == []

    async def test_job_selector_and_provenance_reach_storage(self, env):
        driver, repo = env(_graph())
        result = await freeze_document(job_id=["j1", "j2"], created_by_user_id="u1", tenant_id="acme")
        query, params = next((q, p) for kind, q, p in driver.calls if kind == "session.run")
        assert "d.job_id IN $job_id" in query and params["job_id"] == ["j1", "j2"]
        assert repo.stored[0]["job_id"] == ["j1", "j2"]
        assert repo.stored[0]["created_by_user_id"] == "u1"
        assert result.job_id == ["j1", "j2"] and result.path is None


_PENDING = {
    "frozen": True,
    "frozen_blob_id": "b0",
    "frozen_cleanup_pending": True,
    "frozen_keep_structure_nodes": False,
    "frozen_keep_annotations": True,
    "frozen_keep_extraction_results": False,
}


class TestResumeInterruptedFreeze:
    """A freeze interrupted while deleting leaves stubs marked
    ``frozen_cleanup_pending``: freeze_document() finishes them."""

    async def test_pending_stubs_are_finished_without_a_new_export(self, env):
        driver, repo = env(_graph(**_PENDING))

        result = await freeze_document("a.pdf", tenant_id="acme")

        assert repo.stored == []
        assert driver.tx_queries() == []  # neither guard nor mark: already stubs
        assert not any(_is_export_read(q) for _, q, _ in driver.calls[1:])
        queries = driver.write_queries()
        assert queries[-1] == freeze._CLEAR_PENDING_QUERY
        # The candidates of the interrupted run died with it: the whole
        # tenant is swept instead (2 + 0 Entity|ModelInstance, 0 LabeledEntity).
        assert sum("borrados" in q for q in queries) == 3
        assert _gc.GC_SEEDS_QUERY not in queries
        assert not any(q in _CANDIDATE_QUERIES for q in queries)
        assert result.gc_entity_model_instance_deleted == 2
        assert result.found and result.mode == "freeze"
        assert result.frozen_blob_id == "b0"
        assert result.documents_frozen == 1
        assert result.versions_frozen == [1]

    async def test_the_flags_of_the_first_call_win(self, env):
        # Frozen with keep_annotations=True; this call passes no flag.
        driver, _ = env(_graph(**_PENDING))

        result = await freeze_document("a.pdf", tenant_id="acme", keep_extraction_results=True)

        assert driver.step_queries() == [
            query for _, query in _cascade.subtree_steps(False, True, False)
        ]
        assert _cascade.RELINK_MODEL_DECISIONS_QUERY in driver.step_queries()
        assert result.model_decisions_kept == 3
        assert result.extraction_results_kept == 0

    async def test_new_documents_are_frozen_along_with_the_pending_ones(self, env):
        driver = _graph()
        driver.docs = [_doc("folder", 1), _doc("folder/old", 1, seed=False, **_PENDING)]
        _, repo = env(driver)

        result = await freeze_document("folder", tenant_id="acme", keep_structure_nodes=True)

        # Only the document that was not frozen yet is exported and marked.
        [stored] = repo.stored
        assert [e["document"]["path"] for e in stored["snapshot"]["documents"]] == ["folder"]
        mark_params = next(p for _, q, p in driver.calls if q == freeze._MARK_FROZEN_QUERY)
        assert mark_params["keys"] == [{"path": "folder", "version": 1}]
        # Each document is cleaned up with its own flags...
        step_keys = {
            q: p["keys"] for kind, q, p in driver.calls if kind == "session.run" and q in _STEP_QUERIES
        }
        assert step_keys[_cascade.STRUCTURE_NODES_COUNT_QUERY] == [{"path": "folder", "version": 1}]
        assert step_keys[_cascade.STRUCTURE_NODES_DELETE_QUERY] == [
            {"path": "folder/old", "version": 1}
        ]
        # ...and the pending mark is cleared on both.
        clear_params = next(p for _, q, p in driver.calls if q == freeze._CLEAR_PENDING_QUERY)
        assert clear_params["keys"] == [
            {"path": "folder", "version": 1},
            {"path": "folder/old", "version": 1},
        ]
        assert result.frozen_blob_id == "blob-1"
        assert result.documents_frozen == 2

    async def test_a_pending_stub_makes_the_whole_call_sweep(self, env):
        """One sweep covers the new documents too: no candidates are gathered."""
        driver = _graph()
        driver.docs = [_doc("folder", 1), _doc("folder/old", 1, seed=False, **_PENDING)]
        env(driver)

        await freeze_document("folder", tenant_id="acme")

        queries = driver.write_queries()
        assert _gc.GC_SEEDS_QUERY not in queries
        assert sum("borrados" in q for q in queries) == 3

    async def test_a_half_deleted_document_is_swept_and_unmarked(self, env):
        """A document left by an interrupted delete_document() lost the
        candidates of that run too."""
        driver, repo = env(_graph(deletion_pending=True))

        result = await freeze_document("a.pdf", tenant_id="acme")

        queries = driver.write_queries()
        assert len(repo.stored) == 1  # not a stub: exported and frozen as usual
        assert _gc.GC_SEEDS_QUERY not in queries
        assert sum("borrados" in q for q in queries) == 3
        assert queries[-1] == freeze._CLEAR_PENDING_QUERY
        assert "d.deletion_pending" in freeze._CLEAR_PENDING_QUERY
        assert result.gc_entity_model_instance_deleted == 2

    async def test_backup_mode_refuses_pending_stubs(self, env):
        driver, repo = env(_graph(**_PENDING))
        with pytest.raises(FreezeError, match="already frozen"):
            await freeze_document("a.pdf", tenant_id="acme", delete_after_export=False)
        assert repo.stored == [] and driver.write_queries() == []

    async def test_export_refuses_pending_stubs(self, env):
        env(_graph(**_PENDING))
        with pytest.raises(FreezeError, match="already frozen"):
            await export_document_snapshot("a.pdf", tenant_id="acme")

    async def test_a_finished_stub_is_still_refused(self, env):
        driver, _ = env(_graph(**{**_PENDING, "frozen_cleanup_pending": None}))
        with pytest.raises(FreezeError, match="already frozen"):
            await freeze_document("a.pdf", tenant_id="acme")
        assert driver.write_queries() == []
