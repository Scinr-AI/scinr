"""
tests/contract/test_annotation_extraction.py -- ``store.annotation`` and ``store.extraction``.

The sinks are opened through the domain functions that use them in production
(``ensure_catalog_models``, ``ensure_theme_structure``, ``write_extraction_subgraph``,
``write_triple_subgraph``).
"""

from __future__ import annotations

import pytest

from scinr.newton.annotation.catalog import ensure_catalog_models, ensure_theme_structure
from scinr.newton.annotation.models import AnnotationDecision
from scinr.newton.entity_extraction.graph_mapper import (
    write_extraction_subgraph,
    write_triple_subgraph,
)
from scinr.newton.entity_extraction.model_resolver import resolve_model_class
from scinr.newton.graph.types import DocumentSelector
from scinr.newton.models.default.triple import Triple, TripleItem
from scinr.newton.utils.theme_registry import get_theme_registry
from tests.golden._scenarios import TENANT_A, TENANT_B

from ._helpers import labels, make_document, rels, twice

DOC = "annot.txt"
NODE_IDS = {"sec_1", "prod_a", "combo", "misc"}


def selector(tenant: str | None = TENANT_A, path: str = DOC) -> DocumentSelector:
    return DocumentSelector(tenant_id=tenant, path=path, latest=True)


async def seed(store, *, tenant: str = TENANT_A, name: str = DOC) -> list[dict]:
    """Insert the product document and return its annotatable nodes."""
    store.documents.insert_document_graph(make_document(name, tenant=tenant), 1, False)
    return await store.annotation.fetch_nodes_to_annotate(tenant_id=tenant, doc_path=name)


async def write_catalog(store) -> None:
    registry = get_theme_registry()
    async with store.annotation.catalog_sink() as sink:
        await ensure_catalog_models(registry, sink)
    async with store.annotation.catalog_sink() as sink:
        await ensure_theme_structure(registry, sink)


def decision(model: str | None = "GoldenProduct") -> AnnotationDecision:
    return AnnotationDecision(
        matched_model_class=model, confidence="high", rationale="Contract test."
    )


def by_id(nodes: list[dict]) -> dict[str, dict]:
    return {n["node_id"]: n for n in nodes}


class TestCatalog:
    async def test_catalog_and_themes_are_written_and_idempotent(self, graph_store, dump):
        await twice(dump, lambda: write_catalog(graph_store))
        graph = dump()
        names = {n["props"]["name"] for n in labels(graph, "CatalogModel")}
        assert {"GoldenProduct", "GoldenSupplier", "GoldenRow"} <= names
        assert labels(graph, "ModelField")
        assert labels(graph, "Theme")
        assert rels(graph, "HAS_FIELD")
        assert rels(graph, "BELONGS_TO_THEME")


class TestAnnotation:
    async def test_fetch_nodes_to_annotate(self, graph_store):
        nodes = await seed(graph_store)
        assert {n["node_id"] for n in nodes} == NODE_IDS
        assert all(n["full_id"].startswith(f"{TENANT_A}::{DOC}::1::") for n in nodes)
        assert {"full_id", "node_id", "title", "role", "appearance_order", "theme"} <= set(nodes[0])
        # Another tenant sees nothing; so does the public scope.
        assert (
            await graph_store.annotation.fetch_nodes_to_annotate(tenant_id=TENANT_B, doc_path=DOC)
            == []
        )
        assert (
            await graph_store.annotation.fetch_nodes_to_annotate(tenant_id=None, doc_path=DOC) == []
        )

    async def test_fetch_node_context(self, graph_store):
        nodes = by_id(await seed(graph_store))
        top = nodes["sec_1"]
        context = await graph_store.annotation.fetch_node_context(
            top["full_id"], top["node_id"], top["title"], top["role"]
        )
        assert context["node_id"] == "sec_1"
        assert context["full_id"] == top["full_id"]
        assert [u["title"] for u in context["info_units"]] == ["Overview"]
        assert context["depth"] == 0
        assert context["children"] == []  # only freeform blocks, tables and field groups qualify

    async def test_fetch_document_context_instructions(self, graph_store):
        doc = make_document(DOC, tenant=TENANT_A)
        doc.context_instructions = "Focus on suppliers."
        graph_store.documents.insert_document_graph(doc, 1, False)
        fetch = graph_store.annotation.fetch_document_context_instructions
        assert await fetch(tenant_id=TENANT_A, doc_path=DOC) == "Focus on suppliers."
        assert await fetch(tenant_id=TENANT_B, doc_path=DOC) is None
        assert await fetch(tenant_id=TENANT_A, doc_path="ghost.txt") is None

    async def test_write_annotation_counts_and_is_idempotent(self, graph_store, dump):
        nodes = by_id(await seed(graph_store))
        await write_catalog(graph_store)
        store = graph_store.annotation
        assert await store.count_annotated_nodes(selector()) == 0

        async def run():
            await store.write_annotation(nodes["prod_a"]["full_id"], decision(), DOC)

        graph = await twice(dump, run)
        assert await store.count_annotated_nodes(selector()) == 1
        assert await store.count_annotated_nodes(selector(TENANT_B)) == 0
        assert len(labels(graph, "ModelDecision")) == 1
        assert rels(graph, "HAS_MODEL_DECISION")

        remaining = await store.fetch_nodes_to_annotate(
            tenant_id=TENANT_A, doc_path=DOC, only_unannotated=True
        )
        assert {n["node_id"] for n in remaining} == NODE_IDS - {"prod_a"}

    async def test_write_annotation_without_node_fails(self, graph_store):
        await seed(graph_store)
        with pytest.raises(RuntimeError):
            await graph_store.annotation.write_annotation(
                f"{TENANT_A}::{DOC}::1::nope", decision(), DOC
            )

    async def test_write_manual_annotation(self, graph_store, dump):
        await seed(graph_store)
        await write_catalog(graph_store)
        store = graph_store.annotation
        counts: list[int] = []

        async def run():
            counts.append(
                await store.write_manual_annotation(
                    DOC, "GoldenSupplier", tenant_id=TENANT_A, doc_path=DOC
                )
            )

        await twice(dump, run)
        assert counts == [4, 4]
        assert await store.count_annotated_nodes(selector()) == 4
        assert (
            await store.write_manual_annotation(
                DOC, "GoldenSupplier", tenant_id=TENANT_B, doc_path=DOC
            )
            == 0
        )


class TestExtraction:
    async def _annotated(self, store) -> dict[str, dict]:
        nodes = by_id(await seed(store))
        await write_catalog(store)
        await store.annotation.write_manual_annotation(
            DOC, "GoldenSupplier", tenant_id=TENANT_A, doc_path=DOC
        )
        return nodes

    async def test_fetch_targets(self, graph_store):
        nodes = by_id(await seed(graph_store))
        fetch = graph_store.extraction.fetch_targets
        assert await fetch(tenant_id=TENANT_A, doc_path=DOC) == []  # nothing annotated yet
        await write_catalog(graph_store)
        await graph_store.annotation.write_annotation(nodes["prod_a"]["full_id"], decision(), DOC)
        await graph_store.annotation.write_annotation(nodes["misc"]["full_id"], decision(None), DOC)
        targets = await fetch(tenant_id=TENANT_A, doc_path=DOC)
        assert [t["node_full_id"] for t in targets] == [
            nodes["prod_a"]["full_id"],
            nodes["misc"]["full_id"],
        ]
        assert [t["model_class"] for t in targets] == ["GoldenProduct", None]
        assert [u["title"] for u in targets[0]["info_units"]] == ["Widget A"]
        assert await fetch(tenant_id=TENANT_B, doc_path=DOC) == []

    async def test_mark_info_units_extracted_is_idempotent(self, graph_store, dump):
        nodes = await self._annotated(graph_store)
        node = nodes["prod_a"]["full_id"]
        store = graph_store.extraction
        before = {
            t["node_full_id"]
            for t in await store.fetch_targets(
                tenant_id=TENANT_A, doc_path=DOC, only_unextracted=True
            )
        }
        assert node in before

        async def run():
            await store.mark_info_units_extracted(node)

        graph = await twice(dump, run)
        marked = [n for n in labels(graph, "InfoUnit") if n["props"].get("extracted")]
        assert len(marked) == 1  # the node's only information unit
        assert marked[0]["props"]["extracted"] == "<ts>"
        # The flag is not what ``only_unextracted`` looks at: that is the ExtractionResult.
        assert before == {
            t["node_full_id"]
            for t in await store.fetch_targets(
                tenant_id=TENANT_A, doc_path=DOC, only_unextracted=True
            )
        }

    async def test_triple_sink_is_idempotent(self, graph_store, dump):
        nodes = await self._annotated(graph_store)
        node = nodes["misc"]["full_id"]
        triples = Triple(
            triples=[
                TripleItem(subject="Widget", predicate="is made by", object="Acme Corp"),
                TripleItem(subject="Widget", predicate="ships with", object="Manual"),
            ]
        )

        async def run():
            await write_triple_subgraph(node, triples, DOC, "er-uid-1")

        graph = await twice(dump, run)
        assert len(labels(graph, "ExtractionResult")) == 1
        assert len(labels(graph, "Entity")) == 3
        entity_keys = {n["key"] for n in labels(graph, "Entity")}
        between_entities = [
            r for r in graph["relationships"] if r["from"] in entity_keys and r["to"] in entity_keys
        ]
        assert len(between_entities) == 2
        assert {
            t["node_full_id"]
            for t in await graph_store.extraction.fetch_targets(
                tenant_id=TENANT_A, doc_path=DOC, only_unextracted=True
            )
        } == set(f["full_id"] for f in nodes.values()) - {node}

    async def test_extraction_sink_is_idempotent(self, graph_store, dump):
        nodes = await self._annotated(graph_store)
        node = nodes["prod_a"]["full_id"]
        supplier = resolve_model_class("GoldenSupplier")(
            supplier_code="S-1", supplier_name="Supply Co"
        )

        async def run():
            await write_extraction_subgraph(node, supplier, "GoldenSupplier", [], DOC, "er-uid-2")

        graph = await twice(dump, run)
        assert len(labels(graph, "ExtractionResult")) == 1
        assert len(labels(graph, "LabeledEntity")) == 1
        assert rels(graph, "HAS_EXTRACTION")

    async def test_sink_on_missing_node_fails(self, graph_store):
        await self._annotated(graph_store)
        with pytest.raises(RuntimeError):
            await write_triple_subgraph(
                f"{TENANT_A}::{DOC}::1::nope", Triple(triples=[]), DOC, "er-uid-3"
            )
