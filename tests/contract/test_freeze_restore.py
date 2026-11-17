"""
tests/contract/test_freeze_restore.py -- ``store.freeze`` and ``store.restore``.

Documents are seeded with the full pipeline (ingest, annotate, extract; scripted LLM). The
tests drive the store methods in the order ``freeze_document`` / ``restore_document`` use
them, and rebuild a deleted document through ``restore.batch_writer`` from what
``freeze.export_snapshot`` read.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from scinr.newton.exceptions import FreezeError
from scinr.newton.graph.types import DocumentSelector
from scinr.newton.ingest.freeze import node_identity
from tests.golden._scenarios import TENANT_A, TENANT_B

from ._helpers import diff, documents, labels, same

BLOB = "blob-contract-1"
FROZEN_AT = "2026-01-01T00:00:00+00:00"
NO_KEEP = (False, False, False)
KEEP_ALL = (True, True, True)


def select(tenant: str = TENANT_A, **filters: Any) -> DocumentSelector:
    return DocumentSelector(tenant_id=tenant, **filters)


async def in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def seed(env, name: str = "frz.txt", tenant: str | None = TENANT_A) -> str:
    return await env.full(name, tenant=tenant)


def doc_props(graph: dict[str, Any], path: str, tenant: str = TENANT_A) -> dict[str, Any]:
    return next(d for d in documents(graph, tenant=tenant) if d["props"]["path"] == path)["props"]


class TestFetchAndExport:
    async def test_fetch_document_set(self, graph_store, env):
        await env.full("a.txt", folder="Frz", tenant=TENANT_A)
        await env.full("b.txt", folder="Frz", tenant=TENANT_A)
        await env.ingest("a.txt", tenant=TENANT_B)
        fetch = graph_store.freeze.fetch_document_set

        docs = fetch(select(path="Frz"))
        assert {d.path for d in docs} == {"Frz", "Frz/a.txt", "Frz/b.txt"}
        assert len({(d.path, d.version) for d in docs}) == len(docs)  # each once
        assert {d.path for d in docs if d.is_seed} == {"Frz"}
        assert all(d.tenant_id == TENANT_A and not d.frozen for d in docs)
        assert [d.path for d in fetch(select(path="a.txt"))] == []
        assert [d.path for d in fetch(select(TENANT_B, path="a.txt"))] == ["a.txt"]
        assert fetch(select(path="ghost.txt")) == []
        with pytest.raises(ValueError):
            fetch(DocumentSelector(path="Frz"))

    async def test_export_snapshot_streams_and_returns_the_callback_value(
        self, graph_store, env, dump
    ):
        await seed(env)
        docs = graph_store.freeze.fetch_document_set(select(path="frz.txt"))
        before = dump()

        def attempt(reader):
            nodes = list(reader.subtree_nodes(docs[0]))
            doc_rels = list(reader.document_relationships(docs[0]))
            sub_rels = list(reader.subtree_relationships(docs[0]))
            return len(nodes), len(doc_rels), len(sub_rels), nodes[0]

        for _ in range(2):  # read-only: twice, same answer
            count, doc_rels, sub_rels, first = await in_thread(
                graph_store.freeze.export_snapshot, docs, TENANT_A, attempt
            )
            assert count > 0 and doc_rels > 0 and sub_rels > 0
            assert first.labels and first.properties
        assert same(dump(), before)

    async def test_export_snapshot_callback_errors_pass_through(self, graph_store, env):
        await seed(env)
        docs = graph_store.freeze.fetch_document_set(select(path="frz.txt"))

        class CallerError(Exception):
            pass

        calls: list[int] = []

        def attempt(reader):
            calls.append(1)
            list(reader.subtree_nodes(docs[0]))
            raise CallerError("not an engine error")

        with pytest.raises(CallerError):
            await in_thread(graph_store.freeze.export_snapshot, docs, TENANT_A, attempt)
        assert len(calls) == 1  # not retried: it is not an engine error


class TestFreezeSteps:
    async def test_backup_freeze_reduce_and_clear(self, graph_store, env, dump):
        await seed(env)
        freeze, gc = graph_store.freeze, graph_store.gc
        docs = freeze.fetch_document_set(select(path="frz.txt"))

        # mark_backup: only records the last backup; idempotent.
        await in_thread(freeze.mark_backup, TENANT_A, docs, BLOB, FROZEN_AT)
        first = dump()
        await in_thread(freeze.mark_backup, TENANT_A, docs, BLOB, FROZEN_AT)
        assert same(dump(), first)
        assert doc_props(first, "frz.txt")["last_backup_blob_id"] == BLOB
        assert not doc_props(first, "frz.txt").get("frozen")

        # mark_frozen: before anything is deleted; a second call finds them frozen.
        before_freeze = dump()
        await in_thread(
            freeze.mark_frozen,
            TENANT_A,
            docs,
            blob_id=BLOB,
            frozen_at=FROZEN_AT,
            keep_flags=NO_KEEP,
        )
        marked = dump()
        assert doc_props(marked, "frz.txt")["frozen"] is True
        assert doc_props(marked, "frz.txt")["frozen_blob_id"] == BLOB
        assert not same(before_freeze, marked)
        with pytest.raises(FreezeError):
            await in_thread(
                freeze.mark_frozen,
                TENANT_A,
                docs,
                blob_id=BLOB,
                frozen_at=FROZEN_AT,
                keep_flags=NO_KEEP,
            )
        assert same(dump(), marked)  # the failed call marked nothing

        # reduce_to_stubs: deletes the subtree; resumable, so the second run is a no-op.
        stubs = graph_store.restore.frozen_stubs_of_blob(TENANT_A, BLOB)
        assert [d.path for d in stubs] == ["frz.txt"] and stubs[0].frozen
        collector = gc.collector(TENANT_A, "contract")
        counts = await in_thread(
            freeze.reduce_to_stubs, TENANT_A, stubs, NO_KEEP, collector, caller="contract"
        )
        assert any(counts.values())
        await in_thread(collector.collect)
        reduced = dump()
        assert not labels(reduced, "StructureNode")
        assert not labels(reduced, "InfoUnit")
        assert not labels(reduced, "ExtractionResult")
        assert [d["props"]["path"] for d in documents(reduced)] == ["frz.txt"]
        again = await in_thread(
            freeze.reduce_to_stubs, TENANT_A, stubs, NO_KEEP, None, caller="contract"
        )
        assert not any(again.values())
        assert same(dump(), reduced)

        # clear_cleanup_pending: idempotent.
        await in_thread(freeze.clear_cleanup_pending, TENANT_A, stubs)
        cleared = dump()
        await in_thread(freeze.clear_cleanup_pending, TENANT_A, stubs)
        assert same(dump(), cleared)

    async def test_keep_flags_keep_the_structure(self, graph_store, env, dump):
        await seed(env)
        freeze = graph_store.freeze
        docs = freeze.fetch_document_set(select(path="frz.txt"))
        await in_thread(
            freeze.mark_frozen,
            TENANT_A,
            docs,
            blob_id=BLOB,
            frozen_at=FROZEN_AT,
            keep_flags=KEEP_ALL,
        )
        stubs = graph_store.restore.frozen_stubs_of_blob(TENANT_A, BLOB)
        assert stubs[0].keep_flags == KEEP_ALL
        await in_thread(freeze.reduce_to_stubs, TENANT_A, stubs, KEEP_ALL, None, caller="contract")
        graph = dump()
        assert labels(graph, "StructureNode")
        assert labels(graph, "ExtractionResult")

    async def test_freezing_one_tenant_leaves_the_other(self, graph_store, env, dump):
        await seed(env, tenant=TENANT_A)
        await seed(env, tenant=TENANT_B)
        freeze = graph_store.freeze
        docs = freeze.fetch_document_set(select(path="frz.txt"))
        before_b = [n for n in labels(dump(), "StructureNode") if TENANT_B in n["key"]]
        await in_thread(
            freeze.mark_frozen,
            TENANT_A,
            docs,
            blob_id=BLOB,
            frozen_at=FROZEN_AT,
            keep_flags=NO_KEEP,
        )
        await in_thread(freeze.reduce_to_stubs, TENANT_A, docs, NO_KEEP, None, caller="contract")
        graph = dump()
        assert [n for n in labels(graph, "StructureNode") if TENANT_B in n["key"]] == before_b
        assert not doc_props(graph, "frz.txt", TENANT_B).get("frozen")


class TestRestore:
    async def test_reads_and_finish(self, graph_store, env, dump):
        await seed(env)
        freeze, restore = graph_store.freeze, graph_store.restore
        docs = freeze.fetch_document_set(select(path="frz.txt"))
        keys = [d.key for d in docs]

        assert [d.path for d in restore.existing_documents(TENANT_A, keys)] == ["frz.txt"]
        assert restore.existing_documents(TENANT_B, keys) == []
        assert restore.existing_documents(TENANT_A, [{"path": "ghost.txt", "version": 1}]) == []
        assert restore.frozen_stubs_of_blob(TENANT_A, BLOB) == []
        assert restore.count_blob_references(TENANT_A, BLOB) == 0

        await in_thread(
            freeze.mark_frozen,
            TENANT_A,
            docs,
            blob_id=BLOB,
            frozen_at=FROZEN_AT,
            keep_flags=NO_KEEP,
        )
        assert restore.count_blob_references(TENANT_A, BLOB) == 1
        assert restore.count_blob_references(TENANT_B, BLOB) == 0
        stubs = restore.frozen_stubs_of_blob(TENANT_A, BLOB)
        assert restore.frozen_stubs_of_blob(TENANT_B, BLOB) == []

        async def finish():
            await in_thread(restore.finish_restore, TENANT_A, stubs, ["frz.txt"])

        await finish()
        done = dump()
        assert not doc_props(done, "frz.txt").get("frozen")
        assert doc_props(done, "frz.txt")["latest"] is True
        assert restore.count_blob_references(TENANT_A, BLOB) == 0
        await finish()
        assert same(dump(), done)

    async def test_has_stale_extraction_results(self, graph_store, env):
        await seed(env)
        restore = graph_store.restore
        nodes = await graph_store.annotation.fetch_nodes_to_annotate(
            tenant_id=TENANT_A, doc_path="frz.txt"
        )
        ids = [n["full_id"] for n in nodes]
        assert restore.has_stale_extraction_results([ids]) is True
        assert restore.has_stale_extraction_results([]) is False
        assert restore.has_stale_extraction_results([["nope::nope::1::nope"]]) is False

    async def test_delete_stale_subtrees(self, graph_store, env, dump):
        await seed(env)
        nodes = await graph_store.annotation.fetch_nodes_to_annotate(
            tenant_id=TENANT_A, doc_path="frz.txt"
        )
        ids = [n["full_id"] for n in nodes]
        assert labels(dump(), "ModelDecision") and labels(dump(), "ExtractionResult")

        async def run():
            writer = graph_store.restore.batch_writer(2)
            await writer.delete_stale(
                ("stale",), ids, model_decisions=False, extraction_results=True
            )
            await writer.drain()

        await run()
        graph = dump()
        assert not labels(graph, "ExtractionResult")
        assert labels(graph, "ModelDecision")

        async def run_all():
            writer = graph_store.restore.batch_writer(2)
            await writer.delete_stale(
                ("stale",), ids, model_decisions=True, extraction_results=True
            )
            await writer.drain()

        await run_all()
        await run_all()
        assert not labels(dump(), "ModelDecision")
        assert labels(dump(), "StructureNode")

    async def test_rebuild_from_a_snapshot_through_the_batch_writer(self, graph_store, env, dump):
        await seed(env)
        original = dump()
        store = graph_store
        docs = store.freeze.fetch_document_set(select(path="frz.txt"))

        def attempt(reader):
            nodes = [n for doc in docs for n in reader.subtree_nodes(doc)]
            rels = [
                r
                for doc in docs
                for r in (*reader.document_relationships(doc), *reader.subtree_relationships(doc))
            ]
            return nodes, rels

        nodes, rels = await in_thread(store.freeze.export_snapshot, docs, TENANT_A, attempt)

        keys, _, _ = store.deletion.fetch_document_keys(select(path="frz.txt"))
        await in_thread(store.deletion.mark_deletion_pending, TENANT_A, keys)
        await in_thread(store.deletion.cascade_delete, TENANT_A, keys, None, caller="contract")
        await in_thread(store.deletion.delete_documents, TENANT_A, keys)
        assert not documents(dump(), tenant=TENANT_A)

        counters: list[Any] = []

        async def rebuild() -> None:
            writer = store.restore.batch_writer(2)
            groups: dict[tuple[str, ...], list[dict[str, Any]]] = {("Document",): []}
            for doc in docs:  # the Document nodes are not part of their own subtree
                key = {"tenant_id": doc.tenant_id, "path": doc.path, "version": doc.version}
                props = {k: v for k, v in doc.properties.items() if k not in key}
                groups[("Document",)].append({"key": key, "props": props})
            for raw in nodes:
                primary, key_fields = node_identity(raw.labels)
                key = {f: raw.properties.get(f) for f in key_fields}
                props = {k: v for k, v in raw.properties.items() if k not in key_fields}
                groups.setdefault(tuple(raw.labels), []).append({"key": key, "props": props})
            for node_labels, rows in groups.items():
                await writer.merge_nodes(("nodes", node_labels), node_labels, rows, counters.append)
            await writer.drain()
            rel_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            for rel in rels:
                from_primary, from_fields = node_identity(rel.from_labels)
                to_primary, _ = node_identity(rel.to_labels)
                rel_groups.setdefault((rel.type, from_primary, to_primary), []).append(
                    {"from": rel.from_key, "to": rel.to_key, "props": rel.properties}
                )
            for group, rows in rel_groups.items():
                await writer.create_relationships(
                    ("relationships", group), group, rows, counters.append
                )
            await writer.drain()

        await rebuild()
        assert sum(c.nodes_created for c in counters) > 0
        assert sum(c.relationships_created for c in counters) > 0
        rebuilt = dump()
        assert same(rebuilt, original), diff(original, rebuilt)

        counters.clear()
        await rebuild()
        assert sum(c.nodes_created for c in counters) == 0
        assert sum(c.relationships_created for c in counters) == 0
        assert same(dump(), original)

    async def test_abort_cancels_what_is_in_flight(self, graph_store):
        writer = graph_store.restore.batch_writer(1)
        await writer.abort()
        await writer.drain()
