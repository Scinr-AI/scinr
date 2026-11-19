"""
tests/contract/test_deletion_gc.py -- ``store.deletion`` and ``store.gc``.

Documents are seeded with the full pipeline (ingest, annotate, extract; scripted LLM), so
there is a real subtree and real extraction results to delete and collect.
"""

from __future__ import annotations

import asyncio
from typing import Any

from scinr.newton.graph.types import DocumentSelector
from tests.golden._scenarios import TENANT_A, TENANT_B, USER

from ._helpers import documents, labels, same, twice

SUBTREE_LABELS = ("StructureNode", "InfoUnit", "ExtractionResult", "ModelDecision")


def select(tenant: str = TENANT_A, **filters: Any) -> DocumentSelector:
    return DocumentSelector(tenant_id=tenant, **filters)


def tenant_nodes(graph: dict[str, Any], tenant: str) -> list[dict[str, Any]]:
    """Nodes of one tenant, without the raw-file ids (their placeholder numbering depends
    on which other tenants are in the dump)."""
    return [
        {**n, "props": {k: v for k, v in n["props"].items() if k != "raw_file_id"}}
        for n in graph["nodes"]
        if tenant in n["key"]
    ]


async def in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


class TestFetch:
    async def test_fetch_document_keys_by_every_filter(self, graph_store, env):
        await env.full("a.txt", folder="Docs", tenant=TENANT_A, created_by=USER, job_id="job-1")
        await env.full("b.txt", folder="Other", tenant=TENANT_A, created_by="other", job_id="job-2")
        await env.ingest("a.txt", folder="Docs", tenant=TENANT_A, job_id="job-1")  # version 2
        await env.full("a.txt", tenant=TENANT_B, created_by=USER, job_id="job-1")
        fetch = graph_store.deletion.fetch_document_keys

        keys, deleted, interrupted = fetch(select(path="Docs/a.txt"))
        assert interrupted is False
        assert sorted(k["path"] for k in keys) == ["Docs/a.txt", "Docs/a.txt"]
        assert sorted(d.path for d in deleted) == ["Docs/a.txt", "Docs/a.txt"]
        assert {k["version"] for k in keys} == {1, 2}

        keys, _, _ = fetch(select(path="Docs/a.txt", version=1))
        assert [(k["path"], k["version"]) for k in keys] == [("Docs/a.txt", 1)]

        # A folder selects its descendants, descendants first.
        keys, _, _ = fetch(select(path="Docs", version=1))
        paths = [k["path"] for k in keys]
        assert set(paths) == {"Docs", "Docs/a.txt"}
        assert paths[-1] == "Docs"

        # The domain always hands over lists (a bare string is a Neo4j query error: see report).
        keys, _, _ = fetch(select(job_id=["job-2"]))
        assert {k["path"] for k in keys} == {"Other", "Other/b.txt"}
        keys, _, _ = fetch(select(job_id=["job-1", "job-2"], version=1))
        assert {"Docs/a.txt", "Other/b.txt"} <= {k["path"] for k in keys}
        keys, _, _ = fetch(select(created_by_user_id=["other"]))
        assert {k["path"] for k in keys} == {"Other", "Other/b.txt"}

        assert fetch(select(path="ghost.txt")) == ([], [], False)

    async def test_tenants_do_not_see_each_other(self, graph_store, env):
        await env.ingest("a.txt", tenant=TENANT_A)
        await env.ingest("a.txt", tenant=TENANT_B)
        await env.ingest("a.txt", tenant=None)
        deletion = graph_store.deletion
        for tenant in (TENANT_A, TENANT_B, "__public__"):
            keys, deleted, _ = deletion.fetch_document_keys(select(tenant, path="a.txt"))
            assert len(keys) == len(deleted) == 1
            assert deletion.fetch_existing_versions(select(tenant, path="a.txt")) == [1]
        assert deletion.fetch_existing_versions(select("nobody", path="a.txt")) == []

    async def test_fetch_existing_versions_and_raw_file_ids(self, graph_store, env):
        await env.ingest("a.txt", folder="Docs", tenant=TENANT_A)
        await env.ingest("a.txt", folder="Docs", tenant=TENANT_A)
        await env.ingest("b.txt", folder="Docs", tenant=TENANT_A)
        deletion = graph_store.deletion
        assert deletion.fetch_existing_versions(select(path="Docs/a.txt")) == [1, 2]
        pairs = deletion.fetch_raw_file_ids(select(path="Docs"))
        assert len(pairs) == 3 and len(set(pairs)) == 3
        assert {tenant for _, tenant in pairs} == {TENANT_A}
        assert deletion.fetch_raw_file_ids(select(TENANT_B, path="Docs")) == []


class TestDelete:
    async def test_mark_pending_cascade_and_delete_are_idempotent(self, graph_store, env, dump):
        await env.full("keep.txt", tenant=TENANT_A)
        await env.full("gone.txt", tenant=TENANT_A)
        deletion = graph_store.deletion
        keys, _, _ = deletion.fetch_document_keys(select(path="gone.txt"))

        async def mark():
            await in_thread(deletion.mark_deletion_pending, TENANT_A, keys)

        graph = await twice(dump, mark)
        pending = [d for d in documents(graph) if d["props"].get("deletion_pending")]
        assert [d["props"]["path"] for d in pending] == ["gone.txt"]
        assert deletion.fetch_document_keys(select(path="gone.txt"))[2] is True  # interrupted

        counts = await in_thread(deletion.cascade_delete, TENANT_A, keys, None, caller="contract")
        assert counts and any(counts.values())
        after_first = dump()
        again = await in_thread(deletion.cascade_delete, TENANT_A, keys, None, caller="contract")
        assert not any(again.values())
        assert same(dump(), after_first)
        # The subtree of gone.txt is gone, its Document and keep.txt are not.
        assert {d["props"]["path"] for d in documents(after_first)} == {"keep.txt", "gone.txt"}
        assert all("gone.txt" not in n["key"] for n in labels(after_first, "StructureNode"))
        assert labels(after_first, "StructureNode")

        assert await in_thread(deletion.delete_documents, TENANT_A, keys) == 1
        assert await in_thread(deletion.delete_documents, TENANT_A, keys) == 0
        assert {d["props"]["path"] for d in documents(dump())} == {"keep.txt"}

    async def test_other_tenant_is_untouched(self, graph_store, env, dump):
        await env.full("shared.txt", tenant=TENANT_A)
        await env.full("shared.txt", tenant=TENANT_B)
        deletion = graph_store.deletion
        keys, _, _ = deletion.fetch_document_keys(select(path="shared.txt"))
        before_b = tenant_nodes(dump(), TENANT_B)
        await in_thread(deletion.mark_deletion_pending, TENANT_A, keys)
        await in_thread(deletion.cascade_delete, TENANT_A, keys, None, caller="contract")
        await in_thread(deletion.delete_documents, TENANT_A, keys)
        graph = dump()
        assert tenant_nodes(graph, TENANT_B) == before_b
        assert not tenant_nodes(graph, TENANT_A)


#: The subtree of a Document as the Cypher of ``graph/neo4j/deletion.py`` defines it: the
#: StructureNodes (``HAS_STRUCTURE`` then ``HAS_CHILD*``) and, from each one, these paths.
_SUBTREE_PATHS: tuple[tuple[str, ...], ...] = (
    ("HAS_INFO_UNIT",),
    ("HAS_MODEL_DECISION",),
    ("HAS_MODEL_DECISION", "HAS_PROPOSED_MODEL"),
    ("HAS_MODEL_DECISION", "HAS_PROPOSED_MODEL", "HAS_PROPOSED_FIELD"),
    ("HAS_MODEL_DECISION", "HAS_COMPLEMENTARY_MATCH"),
    ("HAS_MODEL_DECISION", "HAS_SUPPLEMENTARY_FIELD"),
    ("HAS_EXTRACTION",),
)

#: Every relationship type of those paths (what ``reachable`` may walk in the SQL check).
_OWNERSHIP_TYPES = frozenset({"HAS_STRUCTURE", "HAS_CHILD"} | {t for p in _SUBTREE_PATHS for t in p})


def cypher_subtree(graph: dict[str, Any], document_key: str) -> set[str]:
    """Dump keys of the subtree of a Document, walking the dump like the Cypher does."""
    out: dict[tuple[str, str], list[str]] = {}
    for rel in graph["relationships"]:
        out.setdefault((rel["from"], rel["type"]), []).append(rel["to"])

    def step(keys: set[str], rel_type: str) -> set[str]:
        return {dst for key in keys for dst in out.get((key, rel_type), [])}

    structure = step({document_key}, "HAS_STRUCTURE")
    frontier = set(structure)
    while frontier:
        frontier = step(frontier, "HAS_CHILD") - structure
        structure |= frontier
    found = set(structure)
    for path in _SUBTREE_PATHS:
        keys = set(structure)
        for rel_type in path:
            keys = step(keys, rel_type)
        found |= keys
    return found


def sql_reachable_subtree(backend_name: str, tenant: str, path: str, version: int) -> set[str]:
    """Dump keys of what ``reachable()`` finds below the Document over the ownership types."""
    import sqlalchemy as sa

    from scinr.newton.graph.sql import _common as c
    from scinr.newton.graph.sql._traversal import reachable
    from scinr.newton.sql.keys import decode_key
    from tests.golden._dump import node_key

    start = c.nodes_with_keys("Document", [c.document_key(tenant, path, version)])
    below = reachable(start, max_hops=100, rel_types=_OWNERSHIP_TYPES, name="subtree")
    statement = sa.select(below.c.label, below.c.key).distinct()
    rows = c.sync_read(lambda connection: connection.execute(statement).all())
    return {node_key([label], decode_key(label, key)) for label, key in rows}


class TestCascadeScope:
    async def test_cascade_deletes_exactly_the_cypher_subtree(
        self, graph_store, backend, env, dump, tmp_path
    ):
        """``cascade_delete`` removes exactly what the Cypher subtree patterns reach.

        The SQL store selects the subtree by ``_document_key`` instead of walking the
        relationships; this checks that both definitions give the same nodes (a full
        pipeline document and a tabular one, next to a document that stays).
        """
        from scinr.newton.stages import run_tabular_pipeline

        await env.full("cut.txt", tenant=TENANT_A)
        await env.full("stay.txt", tenant=TENANT_A)
        root = tmp_path / "tab"
        root.mkdir()
        (root / "plain.csv").write_text("sku,item,qty,x_note\nS1,Bolt,10,a\n", encoding="utf-8")
        result = await run_tabular_pipeline(
            input_raw=str(root), tenant_id=TENANT_A, created_by_user_id=USER, job_id="job-tab"
        )
        assert result.success, [d.errors for d in result.documents]

        deletion = graph_store.deletion
        for path in ("cut.txt", "plain.csv"):
            before = dump()
            doc = next(d for d in documents(before) if d["props"]["path"].endswith(path))
            expected = cypher_subtree(before, doc["key"])
            assert expected, f"{path}: empty subtree"
            if backend.name != "neo4j":
                props = doc["props"]
                found = sql_reachable_subtree(
                    backend.name, TENANT_A, props["path"], props["version"]
                )
                assert found == expected
            keys, _, _ = deletion.fetch_document_keys(select(path=doc["props"]["path"]))
            await in_thread(deletion.cascade_delete, TENANT_A, keys, None, caller="contract")
            after = dump()
            deleted = {n["key"] for n in before["nodes"]} - {n["key"] for n in after["nodes"]}
            assert deleted == expected, (sorted(deleted - expected), sorted(expected - deleted))


class TestGc:
    async def _orphaned(self, env) -> None:
        """A document whose extraction results were deleted by an update-mode reload,
        leaving entities nobody points at."""
        await env.full("orphans.txt", tenant=TENANT_A)
        await env.ingest("orphans.txt", tenant=TENANT_A, update_mode=True)

    async def test_sweep_tenant_collects_orphans_and_is_idempotent(self, graph_store, env, dump):
        await self._orphaned(env)
        before = dump()
        assert labels(before, "LabeledEntity") or labels(before, "Entity")
        counts = await in_thread(graph_store.gc.sweep_tenant, TENANT_A, "contract")
        assert any(v for v in counts.values())
        swept = dump()
        assert not labels(swept, "LabeledEntity")
        assert not labels(swept, "Entity")

        async def sweep():
            await in_thread(graph_store.gc.sweep_tenant, TENANT_A, "contract")

        await twice(dump, sweep)
        assert same(dump(), swept)

    async def test_sweep_does_not_touch_other_tenants(self, graph_store, env, dump):
        await env.full("x.txt", tenant=TENANT_A)
        await env.full("x.txt", tenant=TENANT_B)
        await env.ingest("x.txt", tenant=TENANT_A, update_mode=True)
        before_b = tenant_nodes(dump(), TENANT_B)
        await in_thread(graph_store.gc.sweep_tenant, TENANT_A, "contract")
        assert tenant_nodes(dump(), TENANT_B) == before_b

    async def test_run_pass(self, graph_store, env, dump):
        await self._orphaned(env)
        gc = graph_store.gc
        for kind in ("entity_model_instance", "labeled_entity"):
            deleted, iterations = await in_thread(gc.run_pass, kind, TENANT_A, "contract")
            assert deleted >= 0 and iterations >= 1
        graph = dump()
        deleted, _ = await in_thread(gc.run_pass, "labeled_entity", TENANT_A, "contract")
        assert deleted == 0
        assert same(dump(), graph)

    async def test_collector_collects_what_the_operation_orphans(self, graph_store, env, dump):
        await env.full("only.txt", tenant=TENANT_A)
        deletion, gc = graph_store.deletion, graph_store.gc
        keys, _, _ = deletion.fetch_document_keys(select(path="only.txt"))
        assert labels(dump(), "LabeledEntity")

        collector = gc.collector(TENANT_A, "contract")
        await in_thread(deletion.mark_deletion_pending, TENANT_A, keys)
        await in_thread(deletion.cascade_delete, TENANT_A, keys, collector, caller="contract")
        await in_thread(collector.collect)
        counts = collector.counts()
        assert any(v for v in counts.values())
        graph = dump()
        assert not labels(graph, "LabeledEntity")
        assert not labels(graph, "ExtractionResult")

        # Collecting again, or gathering nothing, changes nothing.
        await in_thread(collector.gather, [])
        await in_thread(collector.chunk_done)
        await in_thread(collector.collect)
        assert same(dump(), graph)
