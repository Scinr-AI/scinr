"""
tests/unit/test_freeze_restore_orchestration.py — freeze_document(),
export_document_snapshot() and restore_document() against a ``FakeGraphStore``.

What is called, in which order, inside which ``sync_scope()``, and what the
neutral snapshot code does with what the ``ExportReader`` hands out. The Cypher
behind those calls is covered by test_freeze_document.py / test_restore_document.py
(through the real Neo4j store over a fake driver) and tests/unit/graph_neo4j/.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scinr.newton.exceptions import FreezeError
from scinr.newton.graph.types import DocumentInfo, RawNode, RawRelationship
from scinr.newton.ingest.freeze import export_document_snapshot, freeze_document
from scinr.newton.ingest.restore import restore_document

GC_COUNTS = {
    "gc_entity_model_instance_deleted": 0,
    "gc_entity_model_instance_passes": 0,
    "gc_labeled_entity_deleted": 0,
    "gc_labeled_entity_passes": 0,
}
SUBTREE_COUNTS = {"structure_nodes_deleted": 3, "info_units_deleted": 2}


def _info(path="a.pdf", version=1, seed=True, **props) -> DocumentInfo:
    properties = {"path": path, "version": version, "tenant_id": "acme", **props}
    return DocumentInfo(path, version, "acme", seed, properties)


class _Reader:
    """An ``ExportReader`` over canned data; counts how it was consumed."""

    def __init__(self, nodes, doc_rels=(), sub_rels=()):
        self._nodes, self._doc_rels, self._sub_rels = nodes, doc_rels, sub_rels
        self.log: list[str] = []

    def subtree_nodes(self, doc):
        self.log.append("nodes")
        return iter(self._nodes)

    def document_relationships(self, doc):
        self.log.append("doc_rels")
        return iter(self._doc_rels)

    def subtree_relationships(self, doc):
        self.log.append("sub_rels")
        return iter(self._sub_rels)


def _node(labels, **props) -> RawNode:
    return RawNode(tuple(labels), props)


_ROOT = _node(["StructureNode", "Section"], id="acme|a|1|root", tenant_id="acme")
_IU = _node(["InfoUnit"], uid="iu-1", tenant_id="acme")


@pytest.fixture
def repo(monkeypatch):
    stored = []

    class _Repo:
        async def store_snapshot(self, path, *, tenant_id, created_by_user_id=None, job_id=None, metadata):
            stored.append(json.loads(Path(path).read_text(encoding="utf-8")))
            return "blob-1"

    monkeypatch.setattr("scinr.newton.freeze.factory.get_freeze_storage", lambda: _Repo())
    return stored


@pytest.fixture
def store(fake_graph_store):
    """The fake store, with a scope that records when it is open, and an
    ``export_snapshot`` that runs the caller's attempt on a reader."""
    fake_graph_store.log = log = []
    fake_graph_store.reader = _Reader([_ROOT, _IU, _IU])

    @contextmanager
    def scope():
        log.append("scope-open")
        try:
            yield
        finally:
            log.append("scope-close")

    fake_graph_store.sync_scope = scope
    fake_graph_store.freeze.export_snapshot.side_effect = (
        lambda docs, tenant, attempt: attempt(fake_graph_store.reader)
    )
    collector = MagicMock(name="collector")
    collector.counts.return_value = GC_COUNTS
    fake_graph_store.gc.collector.return_value = collector
    fake_graph_store.gc.sweep_tenant.return_value = GC_COUNTS
    fake_graph_store.freeze.reduce_to_stubs.return_value = SUBTREE_COUNTS
    fake_graph_store.collector = collector
    return fake_graph_store


class TestExportDocumentSnapshot:
    async def test_dict_export_streams_through_the_reader_inside_one_scope(self, store):
        store.freeze.fetch_document_set.return_value = [_info()]

        snap = await export_document_snapshot("a.pdf", tenant_id="acme")

        assert [n["key"] for n in snap["documents"][0]["nodes"]] == [{"id": "acme|a|1|root"}, {"uid": "iu-1"}]
        assert store.reader.log == ["nodes", "doc_rels", "sub_rels"]
        assert store.log == ["scope-open", "scope-close"]
        selector = store.freeze.fetch_document_set.call_args.args[0]
        assert (selector.tenant_id, selector.path) == ("acme", "a.pdf")

    async def test_relationship_filter_keeps_document_links_and_catalog_refs(self, store):
        catalog = ("CatalogModel",), {"name": "Product"}
        rels = [
            RawRelationship("HAS_INFO_UNIT", ("StructureNode",), {"id": "acme|a|1|root"}, ("InfoUnit",), {"uid": "iu-1"}),
            RawRelationship("MATCHED_MODEL", ("StructureNode",), {"id": "acme|a|1|root"}, *catalog),
            RawRelationship("RELATED", ("StructureNode",), {"id": "acme|a|1|root"}, ("InfoUnit",), {"uid": "far"}),
        ]
        doc_rel = RawRelationship(
            "HAS_STRUCTURE", ("Document",), {"tenant_id": "acme", "path": "a.pdf", "version": 1},
            ("StructureNode",), {"id": "acme|a|1|root"},
        )
        store.reader = _Reader([_ROOT, _IU], [doc_rel], rels)
        store.freeze.fetch_document_set.return_value = [_info()]

        snap = await export_document_snapshot("a.pdf", tenant_id="acme")

        types = [r["type"] for r in snap["documents"][0]["relationships"]]
        assert types == ["HAS_STRUCTURE", "HAS_INFO_UNIT", "MATCHED_MODEL"]

    async def test_other_tenant_node_is_refused(self, store):
        store.reader = _Reader([_node(["InfoUnit"], uid="x", tenant_id="other")])
        store.freeze.fetch_document_set.return_value = [_info()]

        with pytest.raises(FreezeError, match="tenant"):
            await export_document_snapshot("a.pdf", tenant_id="acme")

    async def test_failed_file_export_removes_the_partial_file(self, store, tmp_path):
        store.reader = _Reader([_node(["InfoUnit"], uid="x", tenant_id="other")])
        store.freeze.fetch_document_set.return_value = [_info()]
        dest = tmp_path / "s.json"

        with pytest.raises(FreezeError):
            await export_document_snapshot("a.pdf", tenant_id="acme", destination="file", file_path=dest)
        assert not dest.exists()

    async def test_a_retried_attempt_starts_the_file_over(self, store, tmp_path):
        """The backend retries the whole callback; the file is opened inside it."""
        store.freeze.fetch_document_set.return_value = [_info()]
        store.freeze.export_snapshot.side_effect = lambda docs, tenant, attempt: (
            attempt(_Reader([_ROOT])) and attempt(store.reader)
        )
        dest = tmp_path / "s.json"

        await export_document_snapshot("a.pdf", tenant_id="acme", destination="file", file_path=dest)

        nodes = json.loads(dest.read_text())["documents"][0]["nodes"]
        assert len(nodes) == 2  # only the second attempt's content


class TestFreezeDocument:
    async def test_order_of_a_full_freeze(self, store, repo):
        doc = _info()
        store.freeze.fetch_document_set.return_value = [doc]
        for name in ("mark_frozen", "reduce_to_stubs", "clear_cleanup_pending"):
            getattr(store.freeze, name).side_effect = (
                lambda *a, _n=name, **k: store.log.append(_n) or (SUBTREE_COUNTS if _n == "reduce_to_stubs" else None)
            )
        store.collector.collect.side_effect = lambda: store.log.append("collect")

        result = await freeze_document("a.pdf", tenant_id="acme")

        assert store.log == ["scope-open", "mark_frozen", "reduce_to_stubs", "collect", "clear_cleanup_pending", "scope-close"]
        assert repo[0]["mode"] == "freeze"
        assert result.frozen_blob_id == "blob-1" and result.structure_nodes_deleted == 3
        _, args, kwargs = store.freeze.mark_frozen.mock_calls[0]
        assert args == ("acme", [doc]) and kwargs["blob_id"] == "blob-1"
        store.gc.collector.assert_called_once_with("acme", "freeze_document")
        store.gc.sweep_tenant.assert_not_called()

    async def test_kept_extraction_results_hand_no_collector_to_the_reduction(self, store, repo):
        store.freeze.fetch_document_set.return_value = [_info()]

        await freeze_document("a.pdf", tenant_id="acme", keep_extraction_results=True)

        _, args, kwargs = store.freeze.reduce_to_stubs.mock_calls[0]
        assert args[2:] == ((False, False, True), None) and kwargs == {"caller": "freeze_document"}

    async def test_a_resumed_freeze_sweeps_the_tenant_instead_of_collecting(self, store, repo):
        pending = _info(
            frozen=True, frozen_blob_id="b0", frozen_cleanup_pending=True, frozen_keep_annotations=True
        )
        store.freeze.fetch_document_set.return_value = [pending]

        result = await freeze_document("a.pdf", tenant_id="acme")

        store.freeze.export_snapshot.assert_not_called()
        store.freeze.mark_frozen.assert_not_called()
        store.gc.collector.assert_not_called()
        store.gc.sweep_tenant.assert_called_once_with("acme", "freeze_document")
        _, args, _ = store.freeze.reduce_to_stubs.mock_calls[0]
        assert args[2] == (False, True, False) and args[3] is None
        store.freeze.clear_cleanup_pending.assert_called_once()
        assert result.frozen_blob_id == "b0"

    async def test_backup_mode_only_marks(self, store, repo):
        doc = _info()
        store.freeze.fetch_document_set.return_value = [doc]

        result = await freeze_document("a.pdf", tenant_id="acme", delete_after_export=False)

        store.freeze.mark_backup.assert_called_once()
        assert store.freeze.mark_backup.call_args.args[:3] == ("acme", [doc], "blob-1")
        store.freeze.mark_frozen.assert_not_called()
        store.freeze.reduce_to_stubs.assert_not_called()
        assert result.mode == "backup"

    async def test_nothing_found_touches_nothing(self, store):
        store.freeze.fetch_document_set.return_value = []

        result = await freeze_document("missing.pdf", tenant_id="acme")

        assert result.found is False
        store.freeze.export_snapshot.assert_not_called()
        assert store.log == ["scope-open", "scope-close"]

    async def test_descendant_of_another_tenant_is_refused(self, store):
        other = DocumentInfo("a.pdf/c", 1, "other", False, {"path": "a.pdf/c", "version": 1, "tenant_id": "other"})
        store.freeze.fetch_document_set.return_value = [_info(), other]

        with pytest.raises(FreezeError, match="IS_COMPOSED_OF"):
            await freeze_document("a.pdf", tenant_id="acme")


class TestRestoreDocument:
    @pytest.fixture
    def writer(self, store):
        writer = MagicMock(name="writer")
        for name in ("delete_stale", "merge_nodes", "create_relationships", "drain", "abort"):
            setattr(writer, name, AsyncMock())
        store.restore.batch_writer.return_value = writer
        return writer

    @pytest.fixture
    def blob(self, monkeypatch, tmp_path):
        monkeypatch.setattr("scinr.newton.ingest.restore.get_config", lambda: SimpleNamespace(storage_backend="none"))
        snapshot = {
            "schema_version": 1,
            "tenant_id": "acme",
            "frozen_at": "2026-01-01T00:00:00+00:00",
            "mode": "freeze",
            "keep_flags": None,
            "documents": [
                {
                    "document": {"path": "a.pdf", "version": 1, "tenant_id": "acme"},
                    "nodes": [{"labels": ["InfoUnit"], "key": {"uid": "iu"}, "properties": {"tenant_id": "acme"}}],
                    "relationships": [],
                }
            ],
        }

        class _Repo:
            async def read_snapshot_to_file(self, blob_id, dest, *, tenant_id):
                Path(dest).write_text(json.dumps(snapshot))
                return True

            async def delete_snapshot(self, blob_id, *, tenant_id):
                self.deleted = blob_id

        repo = _Repo()
        monkeypatch.setattr("scinr.newton.freeze.factory.get_freeze_storage", lambda: repo)
        return repo

    async def test_rebuild_goes_through_one_writer_and_finishes_in_order(self, store, writer, blob):
        stub = _info(frozen=True, frozen_blob_id="b1")
        store.freeze.fetch_document_set.return_value = [stub]
        store.restore.has_stale_extraction_results.return_value = False
        store.restore.count_blob_references.return_value = 0

        result = await restore_document("a.pdf", tenant_id="acme", batch_size=10, concurrency=2)

        store.restore.batch_writer.assert_called_once_with(2)
        writer.merge_nodes.assert_awaited_once()
        lane, labels, rows, on_done = writer.merge_nodes.await_args.args
        assert (lane, labels, rows) == (("nodes", ("InfoUnit",)), ("InfoUnit",), [{"key": {"uid": "iu"}, "props": {"tenant_id": "acme"}}])
        writer.drain.assert_awaited()
        store.restore.finish_restore.assert_called_once_with("acme", [stub], [])
        store.restore.count_blob_references.assert_called_once_with("acme", "b1")
        store.gc.sweep_tenant.assert_not_called()
        assert blob.deleted == "b1" and result.snapshots_deleted == 1
        assert store.log[0] == "scope-open" and store.log[-1] == "scope-close"

    async def test_gc_sweeps_only_when_stale_extraction_results_exist(self, store, writer, blob):
        store.freeze.fetch_document_set.return_value = [_info(frozen=True, frozen_blob_id="b1")]
        store.restore.has_stale_extraction_results.return_value = True
        store.restore.count_blob_references.return_value = 1

        await restore_document("a.pdf", tenant_id="acme")

        store.gc.sweep_tenant.assert_called_once_with("acme", "restore_document")

    async def test_a_failed_batch_aborts_the_writer_and_finishes_nothing(self, store, writer, blob):
        store.freeze.fetch_document_set.return_value = [_info(frozen=True, frozen_blob_id="b1")]
        store.restore.has_stale_extraction_results.return_value = False
        writer.drain.side_effect = RuntimeError("batch failed")

        with pytest.raises(RuntimeError, match="batch failed"):
            await restore_document("a.pdf", tenant_id="acme")

        writer.abort.assert_awaited_once()
        store.restore.finish_restore.assert_not_called()
