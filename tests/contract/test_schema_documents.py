"""
tests/contract/test_schema_documents.py -- ``store.schema`` and ``store.documents``.
"""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import DocumentVersionConflict
from scinr.newton.graph.types import DocumentSelector
from scinr.newton.models.document_structure import InfoUnit, NodeRole, StructureNode
from tests.golden._scenarios import TENANT_A, TENANT_B, product_structure

from ._helpers import documents, labels, make_document, rels, twice


class TestSchema:
    async def test_setup_is_idempotent(self, graph_store, dump):
        before = dump()
        graph_store.schema.setup()
        graph_store.schema.setup()
        assert dump() == before == {"nodes": [], "relationships": []}

    async def test_ensure_indexes_is_best_effort_and_idempotent(self, graph_store):
        assert graph_store.schema.ensure_indexes([]) == 0

    async def test_ensure_indexes_rejects_unknown_name(self, graph_store):
        with pytest.raises(ValueError):
            graph_store.schema.ensure_indexes(["no_such_index"])


class TestVersions:
    async def test_next_and_current_version(self, graph_store):
        docs = graph_store.documents
        assert docs.next_version("a.txt", TENANT_A) == 1
        assert docs.current_latest_version("a.txt", TENANT_A) is None
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 1, False)
        assert docs.next_version("a.txt", TENANT_A) == 2
        assert docs.current_latest_version("a.txt", TENANT_A) == 1
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 2, False)
        assert docs.next_version("a.txt", TENANT_A) == 3
        assert docs.current_latest_version("a.txt", TENANT_A) == 2

    async def test_versions_are_per_tenant(self, graph_store):
        docs = graph_store.documents
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 1, False)
        assert docs.next_version("a.txt", TENANT_B) == 1
        assert docs.current_latest_version("a.txt", TENANT_B) is None
        assert docs.next_version("a.txt", "__public__") == 1

    async def test_resolve_batch_version(self, graph_store):
        docs = graph_store.documents
        paths = ["F/a.txt", "F"]
        assert docs.resolve_batch_version(paths, [TENANT_A], False) == 1
        assert docs.resolve_batch_version(paths, [TENANT_A], True) == 1
        docs.insert_document_graph(make_document("a.txt", folder="F", tenant=TENANT_A), 1, False)
        docs.insert_document_graph(make_document("a.txt", folder="F", tenant=TENANT_A), 2, False)
        assert docs.resolve_batch_version(paths, [TENANT_A], False) == 3
        assert docs.resolve_batch_version(paths, [TENANT_A], True) == 2
        # Another tenant (and the public one, ``None``) does not see them.
        assert docs.resolve_batch_version(paths, [TENANT_B], False) == 1
        assert docs.resolve_batch_version(paths, [None], False) == 1
        assert docs.resolve_batch_version(paths, [TENANT_B, TENANT_A], False) == 3


class TestInsertDocumentGraph:
    async def test_writes_hierarchy_structure_and_info_units(self, graph_store, dump):
        graph_store.documents.insert_document_graph(
            make_document("a.txt", folder="Docs/Sub", tenant=TENANT_A, job_id="j", created_by="u"),
            1,
            False,
        )
        graph = dump()
        paths = {d["props"]["path"] for d in documents(graph, tenant=TENANT_A)}
        assert paths == {"Docs", "Docs/Sub", "Docs/Sub/a.txt"}
        leaf = next(d for d in documents(graph) if d["props"]["path"] == "Docs/Sub/a.txt")
        assert leaf["props"]["version"] == 1
        assert leaf["props"]["latest"] is True
        assert leaf["props"]["job_id"] == "j"
        assert leaf["props"]["created_by_user_id"] == "u"
        assert len(rels(graph, "IS_COMPOSED_OF")) >= 2
        assert labels(graph, "InfoUnit")

    async def test_writes_paragraph_reference_properties(self, graph_store, dump):
        child = StructureNode(
            node_id="child",
            title="Child",
            role=NodeRole.SECTION,
            appearance_order=1,
            info_units=[InfoUnit(title="Child", order=0, description="Child text.")],
        )
        parent = StructureNode(
            node_id="parent",
            title="Parent",
            role=NodeRole.SECTION,
            appearance_order=1,
            source_paragraphs=["1:1-2", "2:1-1"],
            paragraph_segmenter_version=1,
            info_units_mode="verbatim",
            info_units=[InfoUnit(title="Parent", order=0, description="Parent text.")],
            children=[child],
        )
        graph_store.documents.insert_document_graph(
            make_document("a.txt", tenant=TENANT_A, structure=[parent]), 1, False
        )
        nodes = {n["props"]["node_id"]: n["props"] for n in labels(dump(), "StructureNode")}
        assert list(nodes["parent"]["source_paragraphs"]) == ["1:1-2", "2:1-1"]
        assert nodes["parent"]["paragraph_segmenter_version"] == 1
        assert nodes["parent"]["info_units_mode"] == "verbatim"
        assert nodes["child"]["info_units_mode"] == "summary"
        assert list(nodes["child"]["source_paragraphs"]) == []
        assert nodes["child"].get("paragraph_segmenter_version") is None

    async def test_is_idempotent_in_update_mode(self, graph_store, dump):
        doc = make_document("a.txt", tenant=TENANT_A)

        async def run():
            graph_store.documents.insert_document_graph(doc, 1, True)

        graph = await twice(dump, run)
        assert len(documents(graph, tenant=TENANT_A)) == 1

    async def test_update_mode_replaces_content_of_same_version(self, graph_store, dump):
        docs = graph_store.documents
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 1, False)
        before = len(labels(dump(), "InfoUnit"))
        revised = make_document("a.txt", tenant=TENANT_A, structure=product_structure(2))
        docs.insert_document_graph(revised, 1, True)
        graph = dump()
        assert len(documents(graph, tenant=TENANT_A)) == 1
        assert len(labels(graph, "InfoUnit")) == before
        descriptions = {n["props"].get("description") for n in labels(graph, "InfoUnit")}
        assert "Catalogue of widgets, revision 2." in descriptions

    async def test_same_version_twice_without_update_mode(self, graph_store, dump):
        """Loading a version that is already there either conflicts or changes nothing.

        The contract error is ``DocumentVersionConflict`` (a concurrent loader won the
        race for the version). Neo4j's ``MERGE`` makes a *sequential* second load a no-op,
        so that is the outcome observed there; a backend that detects the duplicate must
        raise the conflict, never another error and never a half-written graph.
        """
        docs = graph_store.documents
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 1, False)
        before = dump()
        try:
            docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 1, False)
        except DocumentVersionConflict as exc:
            assert exc.__cause__ is not None
        assert dump() == before

    async def test_new_version_flips_latest(self, graph_store, dump):
        docs = graph_store.documents
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 1, False)
        docs.insert_document_graph(make_document("a.txt", tenant=TENANT_A), 2, False)
        latest = {
            d["props"]["version"]: d["props"]["latest"]
            for d in documents(dump(), tenant=TENANT_A)
            if d["props"]["path"] == "a.txt"
        }
        assert latest == {1: False, 2: True}
        assert len(rels(dump(), "HAS_NEWER_VERSION")) == 1


class TestTabularDocument:
    def _create(self, store, tenant=TENANT_A):
        store.documents.create_tabular_document(
            document_name="t.csv",
            doc_path="Sheets/t.csv",
            version=1,
            raw_file_id="raw-t",
            tenant=tenant,
            created_by_user_id="u",
            job_id="j",
        )

    async def test_creates_hierarchy_and_document(self, graph_store, dump):
        self._create(graph_store)
        paths = {d["props"]["path"] for d in documents(dump(), tenant=TENANT_A)}
        assert paths == {"Sheets", "Sheets/t.csv"}
        assert graph_store.documents.current_latest_version("Sheets/t.csv", TENANT_A) == 1

    async def test_is_idempotent(self, graph_store, dump):
        async def run():
            self._create(graph_store)

        await twice(dump, run)


class TestReplacement:
    async def test_preflight_and_apply(self, graph_store, dump):
        docs = graph_store.documents
        docs.insert_document_graph(make_document("old.txt", tenant=TENANT_A), 1, False)
        assert docs.preflight_replaces("old.txt", TENANT_A) == [{"path": "old.txt", "version": 1}]
        assert docs.preflight_replaces("old.txt", TENANT_B) == []
        assert docs.preflight_replaces("ghost.txt", TENANT_A) == []

        docs.insert_document_graph(make_document("new.txt", tenant=TENANT_A), 1, False)

        async def run():
            docs.apply_replacement("old.txt", ["new.txt"], tenant=TENANT_A, replaced_path="old.txt")

        graph = await twice(dump, run)
        old = next(d for d in documents(graph) if d["props"]["path"] == "old.txt")
        assert old["props"]["latest"] is False
        assert len(rels(graph, "HAS_NEWER_VERSION")) == 1
        assert docs.preflight_replaces("old.txt", TENANT_A) == []

    async def test_other_tenant_is_untouched(self, graph_store, dump):
        docs = graph_store.documents
        docs.insert_document_graph(make_document("old.txt", tenant=TENANT_A), 1, False)
        docs.insert_document_graph(make_document("new.txt", tenant=TENANT_B), 1, False)
        before = dump()
        docs.apply_replacement("old.txt", ["new.txt"], tenant=TENANT_B, replaced_path=None)
        assert dump() == before


class TestLeavesAndCounts:
    async def _seed(self, store):
        store.documents.insert_document_graph(
            make_document("a.txt", folder="Docs", tenant=TENANT_A), 1, False
        )
        store.documents.insert_document_graph(
            make_document("b.txt", folder="Docs", tenant=TENANT_A), 1, False
        )

    async def test_resolve_leaf_documents_sync_and_async(self, graph_store):
        await self._seed(graph_store)
        selector = DocumentSelector(tenant_id=TENANT_A, path="Docs", latest=True)
        sync_leaves = graph_store.documents.resolve_leaf_documents(selector)
        async_leaves = await graph_store.documents.resolve_leaf_documents_async(selector)
        assert sorted(sync_leaves) == sorted(async_leaves)
        assert sorted(leaf.path for leaf in sync_leaves) == ["Docs/a.txt", "Docs/b.txt"]
        assert sorted(leaf.name for leaf in sync_leaves) == ["a.txt", "b.txt"]

    async def test_leaf_is_its_own_root_and_misses_are_empty(self, graph_store):
        await self._seed(graph_store)
        by_name = DocumentSelector(tenant_id=TENANT_A, document_name="a.txt", latest=True)
        assert [leaf.path for leaf in graph_store.documents.resolve_leaf_documents(by_name)] == [
            "Docs/a.txt"
        ]
        missing = DocumentSelector(tenant_id=TENANT_A, path="Nope", latest=True)
        assert graph_store.documents.resolve_leaf_documents(missing) == []
        assert await graph_store.documents.resolve_leaf_documents_async(missing) == []
        other_tenant = DocumentSelector(tenant_id=TENANT_B, path="Docs", latest=True)
        assert graph_store.documents.resolve_leaf_documents(other_tenant) == []

    async def test_count_latest_documents(self, graph_store):
        docs = graph_store.documents
        await self._seed(graph_store)
        docs.insert_document_graph(make_document("a.txt", folder="Docs", tenant=TENANT_A), 2, False)
        count = docs.count_latest_documents
        assert (
            await count(DocumentSelector(tenant_id=TENANT_A, path="Docs/a.txt", latest=True)) == 1
        )
        assert (
            await count(DocumentSelector(tenant_id=TENANT_A, document_name="a.txt", latest=True))
            == 1
        )
        assert (
            await count(DocumentSelector(tenant_id=TENANT_B, path="Docs/a.txt", latest=True)) == 0
        )
        assert await count(DocumentSelector(tenant_id=None, path="Docs/a.txt", latest=True)) == 0
