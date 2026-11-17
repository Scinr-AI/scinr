"""
tests/unit/test_delete_document_orchestration.py — delete_document() and
collect_orphans() against a ``FakeGraphStore``.

What is called, in which order and with which selector. The Cypher behind those
calls is covered by test_delete_document.py (through the real Neo4j store over a
fake driver) and tests/unit/graph_neo4j/test_deletion_ops.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scinr.newton.graph.types import DocumentSelector
from scinr.newton.ingest.deletion import collect_orphans, delete_document
from scinr.newton.results import DeletedDocument

KEYS = [{"path": "docs/a/x", "version": 1}, {"path": "docs/a", "version": 1}]
DOCUMENTS = [DeletedDocument(path=k["path"], version=k["version"], is_folder=False) for k in KEYS]
SUBTREE_COUNTS = {"structure_nodes_deleted": 4, "info_units_deleted": 2}
GC_COUNTS = {
    "gc_entity_model_instance_deleted": 3,
    "gc_entity_model_instance_passes": 1,
    "gc_labeled_entity_deleted": 1,
    "gc_labeled_entity_passes": 1,
}


@pytest.fixture
def calls(fake_graph_store):
    """The fake store with canned answers; ``calls`` records the order of the calls."""
    log: list[str] = []
    deletion, gc = fake_graph_store.deletion, fake_graph_store.gc
    collector = MagicMock(name="collector")
    collector.counts.return_value = GC_COUNTS

    def record(name, result=None):
        def _call(*args, **kwargs):
            log.append(name)
            return result

        return _call

    deletion.fetch_existing_versions.side_effect = record("versions", [1])
    deletion.fetch_raw_file_ids.side_effect = record("raw_file_ids", [])
    deletion.fetch_document_keys.side_effect = record("keys", (KEYS, DOCUMENTS, False))
    deletion.mark_deletion_pending.side_effect = record("mark")
    deletion.cascade_delete.side_effect = record("cascade", SUBTREE_COUNTS)
    deletion.delete_documents.side_effect = record("delete_documents", 2)
    gc.collector.side_effect = record("collector", collector)
    gc.sweep_tenant.side_effect = record("sweep", GC_COUNTS)
    collector.collect.side_effect = record("collect")
    fake_graph_store.collector = collector
    fake_graph_store.log = log
    return fake_graph_store


class TestDeleteDocument:
    async def test_runs_the_steps_in_order_inside_one_sync_scope(self, calls):
        result = await delete_document("docs/a", tenant_id="acme")

        assert calls.log == [
            "versions",
            "raw_file_ids",
            "keys",
            "collector",
            "mark",
            "cascade",
            "collect",
            "delete_documents",
        ]
        calls.sync_scope_calls.assert_called_once_with()
        assert result.found is True
        assert result.documents == DOCUMENTS
        assert result.documents_deleted == 2
        assert result.structure_nodes_deleted == 4
        assert result.gc_entity_model_instance_deleted == 3

    async def test_the_selector_carries_the_stored_tenant_and_the_filters(self, calls):
        await delete_document(
            job_id=["j1", "j2"], tenant_id=None, version=2, created_by_user_id="u1"
        )

        expected = DocumentSelector(
            tenant_id="__public__", version=2, job_id=["j1", "j2"], created_by_user_id=["u1"]
        )
        calls.deletion.fetch_existing_versions.assert_called_once_with(expected)
        calls.deletion.fetch_raw_file_ids.assert_called_once_with(expected)
        calls.deletion.fetch_document_keys.assert_called_once_with(expected)

    async def test_the_collector_is_shared_by_the_cascade_and_the_collect(self, calls):
        await delete_document("docs/a", tenant_id="acme")

        calls.gc.collector.assert_called_once_with("acme", "delete_document")
        calls.deletion.mark_deletion_pending.assert_called_once_with("acme", KEYS)
        calls.deletion.cascade_delete.assert_called_once_with(
            "acme", KEYS, calls.collector, caller="delete_document"
        )
        calls.deletion.delete_documents.assert_called_once_with("acme", KEYS)
        calls.gc.sweep_tenant.assert_not_called()

    async def test_an_interrupted_delete_sweeps_the_tenant_before_the_documents_go(self, calls):
        calls.deletion.fetch_document_keys.side_effect = lambda selector: (
            calls.log.append("keys"),
            (KEYS, DOCUMENTS, True),
        )[1]

        await delete_document("docs/a", tenant_id="acme")

        calls.gc.collector.assert_not_called()
        calls.deletion.cascade_delete.assert_called_once_with(
            "acme", KEYS, None, caller="delete_document"
        )
        assert calls.log[-3:] == ["cascade", "sweep", "delete_documents"]
        calls.gc.sweep_tenant.assert_called_once_with("acme", "delete_document")

    async def test_nothing_found_touches_neither_storage_nor_the_cascade(self, calls):
        calls.deletion.fetch_existing_versions.side_effect = lambda selector: []

        result = await delete_document("docs/a", tenant_id="acme")

        assert result.found is False
        calls.deletion.fetch_raw_file_ids.assert_not_called()
        calls.deletion.cascade_delete.assert_not_called()

    async def test_storage_is_cleaned_before_the_graph_cascade(self, calls):
        calls.deletion.fetch_raw_file_ids.side_effect = lambda selector: [("raw-1", "acme")]

        async def _delete(*args, **kwargs):
            calls.log.append("storage")
            return 0

        raw_repo = MagicMock()
        raw_repo.delete_images = _delete
        raw_repo.delete = _delete
        page_repo = MagicMock()
        page_repo.delete_pages = _delete
        with patch("scinr.newton.storage.factory.get_storage", return_value=(raw_repo, page_repo)):
            await delete_document("docs/a", tenant_id="acme")

        assert calls.log.index("storage") < calls.log.index("keys")

    async def test_a_failing_cascade_propagates_and_keeps_the_documents(self, calls):
        calls.deletion.cascade_delete.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await delete_document("docs/a", tenant_id="acme")

        calls.deletion.delete_documents.assert_not_called()

    @pytest.mark.parametrize("kw", [{}, {"path": "p", "job_id": "j"}])
    async def test_exactly_one_of_path_or_job_id(self, calls, kw):
        with pytest.raises(ValueError, match="exactly one"):
            await delete_document(tenant_id="acme", **kw)
        calls.deletion.fetch_existing_versions.assert_not_called()


class TestCollectOrphans:
    async def test_sweeps_the_tenant_inside_a_sync_scope(self, calls):
        result = await collect_orphans(tenant_id="acme")

        calls.gc.sweep_tenant.assert_called_once_with("acme", "collect_orphans")
        calls.sync_scope_calls.assert_called_once_with()
        assert result.gc_entity_model_instance_deleted == 3
        assert result.tenant_id == "acme"

    async def test_none_means_the_public_documents(self, calls):
        await collect_orphans(tenant_id=None)

        calls.gc.sweep_tenant.assert_called_once_with("__public__", "collect_orphans")
