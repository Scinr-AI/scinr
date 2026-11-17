"""
tests/contract/test_tenants.py -- Isolation between two tenants and the public documents.
"""

from __future__ import annotations

import asyncio

from scinr.newton.graph.types import DocumentSelector
from tests.golden._scenarios import TENANT_A, TENANT_B

from ._helpers import documents, labels, make_document

PUBLIC = "__public__"
DOC = "Shared/shared.txt"
TENANTS = (TENANT_A, TENANT_B, PUBLIC)


def of(graph, tenant: str, label: str = "Document"):
    return [n for n in labels(graph, label) if n["props"].get("tenant_id") == tenant]


async def test_same_path_is_independent_per_tenant(graph_store, dump):
    docs = graph_store.documents
    for tenant in (TENANT_A, TENANT_B, None):
        docs.insert_document_graph(
            make_document("shared.txt", folder="Shared", tenant=tenant), 1, False
        )
    docs.insert_document_graph(
        make_document("shared.txt", folder="Shared", tenant=TENANT_A), 2, False
    )

    graph = dump()
    for tenant in TENANTS:
        assert {d["props"]["path"] for d in documents(graph, tenant=tenant)} >= {"Shared", DOC}
    assert docs.current_latest_version(DOC, TENANT_A) == 2
    assert docs.current_latest_version(DOC, TENANT_B) == 1
    assert docs.current_latest_version(DOC, PUBLIC) == 1
    # Structure is per tenant too: ids carry the tenant.
    for tenant in TENANTS:
        assert of(graph, tenant, "StructureNode")
        assert all(tenant in n["props"]["id"] for n in of(graph, tenant, "StructureNode"))


async def test_selectors_see_only_their_tenant(graph_store):
    docs = graph_store.documents
    docs.insert_document_graph(
        make_document("shared.txt", folder="Shared", tenant=TENANT_A), 1, False
    )
    docs.insert_document_graph(make_document("shared.txt", folder="Shared", tenant=None), 1, False)
    count = docs.count_latest_documents
    assert await count(DocumentSelector(tenant_id=TENANT_A, path=DOC, latest=True)) == 1
    assert await count(DocumentSelector(tenant_id=None, path=DOC, latest=True)) == 1
    assert await count(DocumentSelector(tenant_id=TENANT_B, path=DOC, latest=True)) == 0
    for tenant in (TENANT_A, None):
        leaves = docs.resolve_leaf_documents(
            DocumentSelector(tenant_id=tenant, path="Shared", latest=True)
        )
        assert [leaf.path for leaf in leaves] == [DOC]
    a = await graph_store.annotation.fetch_nodes_to_annotate(tenant_id=TENANT_A, doc_path=DOC)
    public = await graph_store.annotation.fetch_nodes_to_annotate(tenant_id=None, doc_path=DOC)
    assert a and public
    assert {n["full_id"] for n in a}.isdisjoint(n["full_id"] for n in public)


async def test_deleting_one_tenant_leaves_the_others(graph_store, env, dump):
    for tenant in (TENANT_A, TENANT_B, None):
        await env.full("shared.txt", folder="Shared", tenant=tenant)
    before = dump()
    deletion = graph_store.deletion
    for victim in (TENANT_A, PUBLIC):
        keys, _, _ = deletion.fetch_document_keys(DocumentSelector(tenant_id=victim, path=DOC))
        await asyncio.to_thread(deletion.mark_deletion_pending, victim, keys)
        await asyncio.to_thread(deletion.cascade_delete, victim, keys, None, caller="contract")
        await asyncio.to_thread(deletion.delete_documents, victim, keys)
        await asyncio.to_thread(graph_store.gc.sweep_tenant, victim, "contract")
        graph = dump()
        assert not [d for d in of(graph, victim) if d["props"]["path"] == DOC]
        assert of(graph, TENANT_B, "StructureNode") == of(before, TENANT_B, "StructureNode")
        assert of(graph, TENANT_B, "ExtractionResult") == of(before, TENANT_B, "ExtractionResult")
    assert len(of(dump(), TENANT_B, "InfoUnit")) == len(of(before, TENANT_B, "InfoUnit"))
