"""
tests/unit/test_restore_document.py — Unit tests for
scinr.newton.ingest.restore: the snapshot reader, validation and
restore_document().

No real Neo4j or MongoDB: a fake sync driver serves the document set, the
finishing query, the snapshot-reference count and the GC; a fake async
driver records the rebuild batches (and reports ``nodes_created`` for keys
not already "in the graph"); a fake FreezeRepository serves snapshots.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scinr.newton.exceptions import FreezeError
from scinr.newton.ingest import restore
from scinr.newton.ingest.freeze import SNAPSHOT_SCHEMA_VERSION
from scinr.newton.ingest.restore import iter_snapshot, restore_document
from scinr.newton.results import RestoreResult

# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _n(labels: list[str], key: dict, tenant: str = "acme", **props) -> dict:
    return {"labels": labels, "key": key, "properties": {"tenant_id": tenant, **props}}


def _ref(node: dict) -> dict:
    return {"labels": node["labels"], "key": node["key"]}


def _doc_ref(path: str, version: int, tenant: str = "acme") -> dict:
    return {"labels": ["Document"], "key": {"tenant_id": tenant, "path": path, "version": version}}


def _r(rel_type: str, source: dict, target: dict, **props) -> dict:
    return {"type": rel_type, "from": source, "to": target, "properties": props}


def _entry(path: str, version: int, prefix: str, tenant: str = "acme") -> dict:
    root = _n(["StructureNode", "Section"], {"id": f"{prefix}|root"}, tenant, title="R")
    child = _n(["StructureNode", "Paragraph"], {"id": f"{prefix}|p1"}, tenant)
    iu = _n(["InfoUnit"], {"uid": f"{prefix}-iu"}, tenant, text="t")
    md = _n(["ModelDecision"], {"uid": f"{prefix}-md"}, tenant, confidence=0.25)
    er = _n(["ExtractionResult"], {"uid": f"{prefix}-er"}, tenant)
    mi = _n(["ModelInstance", "Product"], {"uid": "shared-mi"}, tenant, job_ids=["j1"])
    catalog = {"labels": ["CatalogModel"], "key": {"name": "Product"}}
    return {
        "document": {"path": path, "version": version, "tenant_id": tenant},
        "nodes": [root, child, iu, md, er, mi],
        "relationships": [
            _r("HAS_STRUCTURE", _doc_ref(path, version, tenant), _ref(root)),
            _r("HAS_CHILD", _ref(root), _ref(child), order=0),
            _r("HAS_INFO_UNIT", _ref(child), _ref(iu)),
            _r("HAS_MODEL_DECISION", _ref(child), _ref(md)),
            _r("MATCHED_MODEL", _ref(md), catalog),
            _r("HAS_EXTRACTION", _ref(child), _ref(er)),
            _r("USES_PRIMARY_MODEL", _ref(er), catalog),
            _r("HAS_INSTANCE", _ref(er), _ref(mi)),
        ],
    }


def _snapshot(*entries: dict, tenant: str = "acme", **header) -> dict:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "tenant_id": tenant,
        "frozen_at": "2026-09-29T10:00:00+00:00",
        "mode": "freeze",
        "keep_flags": {"structure_nodes": False, "annotations": False, "extraction_results": False},
        **header,
        "documents": list(entries),
    }


def _stub(path: str, version: int, *, blob: str | None = "blob-1", seed: bool = True,
          tenant: str = "acme", ksn: bool = False, ka: bool = False, ker: bool = False) -> dict:
    props: dict[str, Any] = {"path": path, "version": version, "tenant_id": tenant}
    if blob is not None:
        props.update(
            frozen=True,
            frozen_blob_id=blob,
            frozen_keep_structure_nodes=ksn,
            frozen_keep_annotations=ka,
            frozen_keep_extraction_results=ker,
        )
    return {"properties": props, "is_seed": seed}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _SyncResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None

    def consume(self):
        return None


class _SyncTx:
    def __init__(self, driver: _SyncDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params) -> _SyncResult:
        self.driver.calls.append((query, params))
        if query == restore._RESTORE_FINISH_QUERY:
            for key in params["keys"]:
                blob = self.driver.blob_of.get((key["path"], key["version"]))
                if blob is not None:
                    self.driver.frozen_blobs[blob] -= 1
            return _SyncResult([{"n": len(params["keys"])}])
        if query == restore._RECOMPUTE_LATEST_QUERY:
            return _SyncResult([])
        raise AssertionError(f"Unexpected tx.run: {query}")


class _SyncSession:
    def __init__(self, driver: _SyncDriver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params) -> _SyncResult:
        self.driver.calls.append((query, params))
        if "IS_COMPOSED_OF*" in query:
            return _SyncResult(self.driver.docs)
        if "borrados" in query:  # GC: auto-commit (CALL ... IN TRANSACTIONS)
            return _SyncResult([{"borrados": 0}])
        if query == restore._TEMPORARY_LINKS_DELETE_QUERY:
            return _SyncResult([{"n": 0}])
        if query == restore._STALE_EXTRACTION_RESULTS_QUERY:
            return _SyncResult([{"n": self.driver.stale_extraction_results}])
        if query == restore._BLOB_REFERENCES_QUERY:
            return _SyncResult([{"n": self.driver.frozen_blobs.get(params["blob_id"], 0)}])
        if query == restore._STUBS_OF_BLOB_QUERY:
            return _SyncResult(
                [
                    {"properties": d["properties"]}
                    for d in self.driver.docs
                    if d["properties"].get("frozen")
                    and d["properties"].get("frozen_blob_id") == params["blob_id"]
                ]
            )
        if query == restore._EXISTING_DOCUMENTS_QUERY:
            wanted = {(k["path"], k["version"]) for k in params["keys"]}
            return _SyncResult(
                [
                    {"properties": d["properties"]}
                    for d in self.driver.docs
                    if (d["properties"]["path"], d["properties"]["version"]) in wanted
                ]
            )
        raise AssertionError(f"Unexpected session.run: {query}")

    def execute_write(self, fn):
        return fn(_SyncTx(self.driver))


class _SyncDriver:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        # ExtractionResults found on the kept StructureNodes of a stub
        # (something extracted onto a frozen document).
        self.stale_extraction_results = 0
        # Documents still frozen per blob, as the reference count sees them.
        self.frozen_blobs: dict[str, int] = {}
        self.blob_of: dict[tuple[str, int], str] = {}
        for d in docs:
            blob = d["properties"].get("frozen_blob_id")
            if blob:
                self.frozen_blobs[blob] = self.frozen_blobs.get(blob, 0) + 1
                self.blob_of[(d["properties"]["path"], d["properties"]["version"])] = blob

    def session(self, **kwargs):
        return _SyncSession(self)

    def close(self):
        self.closed = True

    def queries(self) -> list[str]:
        return [q for q, _ in self.calls]


class _AsyncResult:
    def __init__(self, counters: SimpleNamespace) -> None:
        self._counters = counters

    async def consume(self):
        return SimpleNamespace(counters=self._counters)


class _AsyncTx:
    def __init__(self, driver: _AsyncDriver) -> None:
        self.driver = driver

    async def run(self, query: str, **params) -> _AsyncResult:
        self.driver.calls.append((query, params))
        created = rels = 0
        if query.startswith("UNWIND $rows") and "MERGE (n:" in query:
            for row in params["rows"]:
                key = json.dumps(row["key"], sort_keys=True)
                if key not in self.driver.existing:
                    self.driver.existing.add(key)
                    created += 1
        elif "CREATE (a)-[r:" in query:
            rels = len(params["rows"])
        return _AsyncResult(SimpleNamespace(nodes_created=created, relationships_created=rels))


class _AsyncSession:
    def __init__(self, driver: _AsyncDriver) -> None:
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute_write(self, fn, *args):
        d = self.driver
        d.transactions += 1
        d.in_flight += 1
        d.max_in_flight = max(d.max_in_flight, d.in_flight)
        try:
            await asyncio.sleep(0)  # let other batches start: exposes the real concurrency
            if d.fail_on is not None and d.fail_on in (args[0] if args else ""):
                raise RuntimeError("batch failed")
            return await fn(_AsyncTx(d), *args)
        finally:
            d.in_flight -= 1


class _AsyncDriver:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.existing = existing or set()
        self.transactions = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.fail_on: str | None = None  # substring of a query whose batch fails

    def session(self, **kwargs):
        return _AsyncSession(self)


class _FakeFreezeRepo:
    def __init__(self, snapshots: dict[str, tuple[str, dict | str]]) -> None:
        self.snapshots = snapshots  # blob → (tenant, snapshot dict or raw text)
        self.reads: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self.lookups: list[dict] = []
        self.metadata: dict[str, list[dict]] = {}  # blob → metadata documents override

    async def read_snapshot_to_file(self, blob_id, dest_path, *, tenant_id):
        self.reads.append((blob_id, tenant_id))
        if blob_id not in self.snapshots or self.snapshots[blob_id][0] != tenant_id:
            return False
        content = self.snapshots[blob_id][1]
        Path(dest_path).write_text(content if isinstance(content, str) else json.dumps(content))
        return True

    async def find_snapshots(self, *, tenant_id, path=None, version=None, job_id=None,
                             created_by_user_id=None):
        from scinr.newton.freeze.base import SnapshotRecord

        self.lookups.append({"tenant_id": tenant_id, "path": path, "version": version,
                             "job_id": job_id, "created_by_user_id": created_by_user_id})
        records = []
        for blob_id, (tenant, content) in reversed(list(self.snapshots.items())):  # newest first
            if tenant != tenant_id or isinstance(content, str):
                continue
            documents = self.metadata.get(blob_id) or [
                {k: e["document"].get(k) for k in ("path", "version", "job_id", "created_by_user_id")}
                for e in content["documents"]
            ]
            if any(
                (path is None or d.get("path") == path)
                and (version is None or d.get("version") == version)
                and (job_id is None or d.get("job_id") in job_id)
                for d in documents
            ):
                records.append(SnapshotRecord(blob_id, content.get("mode"), content["frozen_at"],
                                              documents))
        return records

    async def delete_snapshot(self, blob_id, *, tenant_id):
        self.deleted.append((blob_id, tenant_id))


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    monkeypatch.setattr(restore, "get_config", lambda: SimpleNamespace(neo4j_database="neo4j"))


@pytest.fixture
def env(monkeypatch):
    def _setup(docs: list[dict], snapshots: dict, existing: set[str] | None = None):
        sync, adrv = _SyncDriver(docs), _AsyncDriver(existing)
        repo = _FakeFreezeRepo(snapshots)
        monkeypatch.setattr(restore, "get_driver", lambda: sync)
        monkeypatch.setattr(restore, "get_async_driver", lambda: adrv)
        monkeypatch.setattr("scinr.newton.freeze.factory.get_freeze_storage", lambda: repo)
        indexes: list[tuple] = []
        monkeypatch.setattr(
            restore,
            "ensure_indexes",
            lambda driver, names, *, database: indexes.append((driver, names, database)) or 0,
        )
        return SimpleNamespace(sync=sync, adrv=adrv, repo=repo, indexes=indexes)

    return _setup


# ---------------------------------------------------------------------------
# Snapshot reader
# ---------------------------------------------------------------------------


class TestIterSnapshot:
    def test_rebuilds_the_same_content_as_json_load(self, tmp_path):
        snap = _snapshot(_entry("a.pdf", 1, "a1"), _entry("a.pdf", 2, "a2"))
        snap["documents"][0]["nodes"][0]["properties"]["nested"] = {"x": [1, 2.5, None, True]}
        path = tmp_path / "s.json"
        path.write_text(json.dumps(snap))

        rebuilt: dict[str, Any] = {"documents": []}
        with path.open("rb") as fh:
            for kind, entry, value in iter_snapshot(fh):
                if kind == "header":
                    rebuilt.update(value)
                elif kind == "document":
                    assert entry == len(rebuilt["documents"])
                    rebuilt["documents"].append({"document": value, "nodes": [], "relationships": []})
                else:
                    rebuilt["documents"][entry][kind + "s"].append(value)
        assert rebuilt == snap

    def test_numbers_are_never_decimal(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps(_snapshot(_entry("a.pdf", 1, "a1"))))
        with path.open("rb") as fh:
            nodes = [v for kind, _, v in iter_snapshot(fh) if kind == "node"]
        md = next(n for n in nodes if n["labels"] == ["ModelDecision"])
        assert type(md["properties"]["confidence"]) is float


# ---------------------------------------------------------------------------
# restore_document()
# ---------------------------------------------------------------------------


def _async_queries(e) -> list[str]:
    return [q for q, _ in e.adrv.calls]


class TestRestoreDocument:
    async def test_not_found(self, env):
        e = env([], {})
        result = await restore_document("a.pdf", tenant_id="acme")
        assert isinstance(result, RestoreResult)
        assert result.found is False and result.documents_restored == 0
        assert e.repo.reads == [] and e.adrv.calls == []
        assert e.sync.closed

    async def test_not_frozen_raises(self, env):
        e = env([_stub("a.pdf", 1, blob=None)], {})
        with pytest.raises(FreezeError, match="nothing to restore"):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.repo.reads == []

    async def test_backed_up_only_is_not_restorable(self, env):
        docs = [_stub("a.pdf", 1, blob=None)]
        docs[0]["properties"]["last_backup_blob_id"] = "blob-9"
        env(docs, {})
        with pytest.raises(FreezeError, match="nothing to restore"):
            await restore_document("a.pdf", tenant_id="acme")

    async def test_full_restore(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})

        result = await restore_document("a.pdf", tenant_id="acme")

        assert result.found is True
        assert result.versions_restored == [1]
        assert result.documents_restored == 1
        assert result.frozen_blob_ids == ["blob-1"]
        assert result.nodes_created == 6 and result.nodes_reused == 0
        assert result.relationships_created == 8
        assert result.snapshots_deleted == 1
        assert e.repo.reads == [("blob-1", "acme")]
        assert e.repo.deleted == [("blob-1", "acme")]

        queries = _async_queries(e)
        # No stale deletes: the StructureNodes were not kept.
        assert all(q.startswith("UNWIND $rows") for q in queries)
        # Every node batch before the first relationship batch.
        first_rel = next(i for i, q in enumerate(queries) if "CREATE (a)-[r:" in q)
        assert all("MERGE (n:" in q for q in queries[:first_rel])
        assert all("CREATE (a)-[r:" in q for q in queries[first_rel:])
        # Catalog targets are merged, data targets matched.
        matched = next(q for q in queries if "`MATCHED_MODEL`" in q)
        assert "MERGE (b:`CatalogModel` {name: row.to.name})" in matched
        has_child = next(q for q in queries if "`HAS_CHILD`" in q)
        assert "MATCH (b:`StructureNode` {id: row.to.id})" in has_child
        has_structure = next(q for q in queries if "`HAS_STRUCTURE`" in q)
        assert "MATCH (a:`Document` {tenant_id: row.from.tenant_id" in has_structure

        # Temporary links → unmark → reference count, all on the tenant. No
        # GC: this restore deleted nothing, so it cannot have orphaned anything.
        sync_queries = e.sync.queries()
        finish = sync_queries.index(restore._RESTORE_FINISH_QUERY)
        assert sync_queries[finish - 1] == restore._TEMPORARY_LINKS_DELETE_QUERY
        assert sync_queries[finish + 1 :] == [restore._BLOB_REFERENCES_QUERY]
        assert not any("borrados" in q for q in sync_queries)
        assert result.gc_entity_model_instance_passes == 0
        assert result.gc_labeled_entity_passes == 0
        finish_params = e.sync.calls[finish][1]
        assert finish_params == {"tenant_id": "acme", "keys": [{"path": "a.pdf", "version": 1}]}
        assert all(p["tenant_id"] == "acme" for _, p in e.sync.calls)

    async def test_stub_of_an_interrupted_freeze_is_restored_like_any_other(self, env):
        """freeze_document() marks the stubs before deleting their subtree; a
        stub left ``frozen_cleanup_pending`` still points at the complete
        snapshot, and restoring it clears that mark with the other ones."""
        stub = _stub("a.pdf", 1)
        stub["properties"]["frozen_cleanup_pending"] = True
        e = env([stub], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})

        result = await restore_document("a.pdf", tenant_id="acme")

        assert result.documents_restored == 1
        assert result.nodes_created == 6
        assert "d.frozen_cleanup_pending" in restore._RESTORE_FINISH_QUERY.split("REMOVE")[1]
        assert e.repo.deleted == [("blob-1", "acme")]

    async def test_node_rows_split_key_and_props(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        await restore_document("a.pdf", tenant_id="acme")
        q, params = next((q, p) for q, p in e.adrv.calls if "MERGE (n:`StructureNode`" in q and "Section" in q)
        assert params["rows"] == [
            {"key": {"id": "a1|root"}, "props": {"tenant_id": "acme", "title": "R"}}
        ]
        assert "ON CREATE SET n += row.props, n:`Section`" in q

    async def test_existing_nodes_are_reused(self, env):
        existing = {json.dumps({"uid": "shared-mi"}), json.dumps({"id": "a1|root"})}
        e = env(
            [_stub("a.pdf", 1)],
            {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))},
            existing=existing,
        )
        result = await restore_document("a.pdf", tenant_id="acme")
        assert result.nodes_created == 4 and result.nodes_reused == 2
        mi_query = next(q for q, _ in e.adrv.calls if "MERGE (n:`ModelInstance`" in q)
        assert "ON MATCH SET n.created_by_user_ids = reduce(" in mi_query
        assert "n.job_ids = reduce(" in mi_query
        md_query = next(q for q, _ in e.adrv.calls if "MERGE (n:`ModelDecision`" in q)
        assert "ON MATCH" not in md_query

    async def test_relationships_are_not_duplicated(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        await restore_document("a.pdf", tenant_id="acme")
        for q, _ in e.adrv.calls:
            if "CREATE (a)-[r:" in q:
                assert "WHERE NOT EXISTS { MATCH (a)-[x:" in q
                assert "properties(x) = row.props" in q

    @pytest.mark.parametrize(
        "ka,ker,expect_md,expect_er",
        [(False, False, True, True), (True, False, False, True), (False, True, True, False),
         (True, True, False, False)],
    )
    async def test_kept_structure_gets_stale_families_deleted_first(self, env, ka, ker, expect_md, expect_er):
        e = env(
            [_stub("a.pdf", 1, ksn=True, ka=ka, ker=ker)],
            {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))},
        )
        await restore_document("a.pdf", tenant_id="acme")
        queries = _async_queries(e)
        first_merge = next(i for i, q in enumerate(queries) if q.startswith("UNWIND $rows"))
        stale = [(q, p) for q, p in e.adrv.calls[:first_merge]]
        md = [p for q, p in stale if "HAS_MODEL_DECISION" in q]
        er = [p for q, p in stale if "HAS_EXTRACTION" in q]
        assert bool(md) is expect_md
        assert bool(er) is expect_er
        for p in md + er:
            assert p["node_ids"] == ["a1|root", "a1|p1"]
        # Nothing but stale deletes before the first merge.
        assert len(stale) == len(md) + len(er)

    async def test_structure_not_kept_means_no_stale_deletes(self, env):
        e = env(
            [_stub("a.pdf", 1, ksn=False, ka=True, ker=True)],
            {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))},
        )
        await restore_document("a.pdf", tenant_id="acme")
        assert all(q.startswith("UNWIND $rows") for q in _async_queries(e))

    async def test_gc_runs_only_when_stale_extraction_results_were_deleted(self, env):
        """The GC scans every :Entity / :ModelInstance of the tenant. A restore
        can only orphan some by deleting ExtractionResults it finds on kept
        StructureNodes (the freeze removed that family, so they were written
        since); otherwise it just creates."""
        snapshot = {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))}
        stale, first_merge = restore._STALE_EXTRACTION_RESULTS_QUERY, "UNWIND $rows"

        # The usual case: the kept StructureNodes carry no ExtractionResult.
        e = env([_stub("a.pdf", 1, ksn=True)], dict(snapshot))
        result = await restore_document("a.pdf", tenant_id="acme")
        [params] = [p for q, p in e.sync.calls if q == stale]
        assert params == {"node_ids": ["a1|root", "a1|p1"]}
        assert not any("borrados" in q for q in e.sync.queries())
        assert result.gc_entity_model_instance_passes == 0

        # Some were found (and deleted before the rebuild): the tenant is swept.
        e = env([_stub("a.pdf", 1, ksn=True)], dict(snapshot))
        e.sync.stale_extraction_results = 3
        result = await restore_document("a.pdf", tenant_id="acme")
        assert sum("borrados" in q for q in e.sync.queries()) == 2
        assert result.gc_entity_model_instance_passes == 1
        # They are counted before the rebuild deletes them.
        assert e.adrv.calls  # the rebuild ran
        assert stale in e.sync.queries()[: e.sync.queries().index(restore._RESTORE_FINISH_QUERY)]
        assert any(q.startswith(first_merge) for q in _async_queries(e))

        # StructureNodes kept and ExtractionResults kept too: none is stale,
        # nothing to look for (stale ModelDecisions orphan nothing).
        for flags in ({"ka": True, "ker": True}, {"ka": False, "ker": True}):
            e = env([_stub("a.pdf", 1, ksn=True, **flags)], dict(snapshot))
            e.sync.stale_extraction_results = 3
            result = await restore_document("a.pdf", tenant_id="acme")
            assert stale not in e.sync.queries()
            assert not any("borrados" in q for q in e.sync.queries())
            assert result.gc_entity_model_instance_passes == 0

    async def test_finish_is_bounded_per_chunk_of_documents(self, env):
        """The temporary links (one per kept ModelDecision / ExtractionResult)
        are deleted in batches, a chunk of documents at a time, and each chunk
        is unmarked only after its links are gone."""
        total = restore.DOCUMENTS_PER_QUERY + 1
        e = env(
            [_stub(f"d{i}.pdf", 1, ka=True, ker=True) for i in range(total)],
            {
                "blob-1": (
                    "acme",
                    _snapshot(*[_entry(f"d{i}.pdf", 1, f"n{i}") for i in range(total)]),
                )
            },
        )

        result = await restore_document(job_id="j1", tenant_id="acme")

        assert result.documents_restored == total
        finish = [
            (q, len(p["keys"]))
            for q, p in e.sync.calls
            if q in (restore._TEMPORARY_LINKS_DELETE_QUERY, restore._RESTORE_FINISH_QUERY)
        ]
        links, unmark = restore._TEMPORARY_LINKS_DELETE_QUERY, restore._RESTORE_FINISH_QUERY
        assert finish == [
            (links, restore.DOCUMENTS_PER_QUERY),
            (unmark, restore.DOCUMENTS_PER_QUERY),
            (links, 1),
            (unmark, 1),
        ]
        assert f"IN TRANSACTIONS OF {restore.DELETE_BATCH_ROWS} ROWS" in links
        assert "collect(" not in links
        assert "DELETE link" not in unmark

    async def test_batches_respect_the_row_limit(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["nodes"] += [_n(["InfoUnit"], {"uid": f"iu-{i}"}) for i in range(3)]
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(entry))})
        result = await restore_document("a.pdf", tenant_id="acme", batch_size=1)
        assert all(len(p["rows"]) == 1 for q, p in e.adrv.calls)
        assert result.nodes_created == 9
        assert result.relationships_created == 8

    async def test_ensures_the_lookup_indexes_first(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        await restore_document("a.pdf", tenant_id="acme")
        [(driver, names, database)] = e.indexes
        assert driver is e.sync and database == "neo4j"
        assert "idx_model_decision_uid" in names and "idx_catalog_model_name" in names

    @pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"concurrency": 0}])
    async def test_invalid_batch_settings(self, env, kwargs):
        e = env([_stub("a.pdf", 1)], {})
        with pytest.raises(ValueError, match="batch_size and concurrency"):
            await restore_document("a.pdf", tenant_id="acme", **kwargs)
        assert e.sync.calls == []

    @pytest.mark.parametrize("concurrency", [1, 2, 4])
    async def test_concurrency_is_bounded(self, env, concurrency):
        entry = _entry("a.pdf", 1, "a1")
        entry["nodes"] += [_n(["InfoUnit"], {"uid": f"iu-{i}"}) for i in range(20)]
        entry["nodes"] += [_n(["ModelDecision"], {"uid": f"md-{i}"}) for i in range(20)]
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(entry))})
        result = await restore_document(
            "a.pdf", tenant_id="acme", batch_size=2, concurrency=concurrency
        )
        assert e.adrv.max_in_flight == concurrency
        assert result.nodes_created == 46

    async def test_same_label_batches_never_overlap(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["nodes"] = [_n(["ModelDecision"], {"uid": f"md-{i}"}) for i in range(10)]
        entry["relationships"] = []
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(entry))})
        await restore_document("a.pdf", tenant_id="acme", batch_size=1, concurrency=8)
        assert e.adrv.max_in_flight == 1  # one lane: one label set

    async def test_every_node_batch_ends_before_relationships(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["nodes"] += [_n(["InfoUnit"], {"uid": f"iu-{i}"}) for i in range(10)]
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(entry))})
        await restore_document("a.pdf", tenant_id="acme", batch_size=1, concurrency=8)
        queries = _async_queries(e)
        first_rel = next(i for i, q in enumerate(queries) if "CREATE (a)-[r:" in q)
        assert all("MERGE (n:" in q for q in queries[:first_rel])
        assert all("CREATE (a)-[r:" in q for q in queries[first_rel:])

    async def test_rows_are_sorted_by_key_inside_a_batch(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["nodes"] = [_n(["InfoUnit"], {"uid": f"iu-{i}"}) for i in (3, 1, 2)]
        entry["relationships"] = []
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(entry))})
        await restore_document("a.pdf", tenant_id="acme")
        [(_, params)] = e.adrv.calls
        assert [row["key"]["uid"] for row in params["rows"]] == ["iu-1", "iu-2", "iu-3"]

    async def test_failed_batch_stops_the_restore_before_finishing(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        e.adrv.fail_on = "`InfoUnit`"
        with pytest.raises(RuntimeError, match="batch failed"):
            await restore_document("a.pdf", tenant_id="acme", concurrency=4)
        assert restore._RESTORE_FINISH_QUERY not in e.sync.queries()
        assert e.repo.deleted == []
        assert e.adrv.in_flight == 0
        assert not any("CREATE (a)-[r:" in q for q in _async_queries(e))

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda s: s.update(tenant_id="other"), "belongs to tenant"),
            (lambda s: s.update(schema_version=99), "schema_version"),
            (lambda s: s["documents"][0]["document"].update(tenant_id="other"), "document entry"),
            (lambda s: s["documents"][0]["nodes"][2]["properties"].update(tenant_id="other"), "tenant_id="),
            (lambda s: s["documents"][0]["nodes"][2]["properties"].pop("tenant_id"), "tenant_id="),
            (lambda s: s["documents"][0]["nodes"][2].update(key={}), "without its key"),
            (lambda s: s["documents"][0]["nodes"][2].update(labels=["Bad Label"]), "business key"),
            (lambda s: s["documents"][0]["nodes"][1].update(labels=["StructureNode", "x`y"]),
             "Invalid label"),
            (lambda s: s["documents"][0]["relationships"][1].update(type="HAS CHILD"), "Invalid"),
            (lambda s: s["documents"][0]["relationships"][0].update({"from": _doc_ref("b.pdf", 1)}),
             "outside its entry"),
            (lambda s: s["documents"][0]["nodes"].append(
                {"labels": ["CatalogModel"], "key": {"name": "X"}, "properties": {}}),
             "Unexpected CatalogModel"),
            (lambda s: s["documents"].__setitem__(0, _entry("z.pdf", 1, "z")), "no entry"),
        ],
    )
    async def test_invalid_snapshot_writes_nothing(self, env, mutate, match):
        snap = _snapshot(_entry("a.pdf", 1, "a1"))
        mutate(snap)
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", snap)})
        with pytest.raises(FreezeError, match=match):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.adrv.calls == []
        assert restore._RESTORE_FINISH_QUERY not in e.sync.queries()
        assert e.repo.deleted == []

    async def test_malformed_json_writes_nothing(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", '{"schema_version": 1, "tenant_id": "acme"}')})
        with pytest.raises(FreezeError, match="Malformed"):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.adrv.calls == []

    async def test_missing_snapshot_writes_nothing(self, env):
        e = env([_stub("a.pdf", 1)], {})
        with pytest.raises(FreezeError, match="not found in the freeze backend"):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.adrv.calls == []

    async def test_snapshot_read_is_tenant_scoped(self, env):
        e = env([_stub("a.pdf", 1)], {"blob-1": ("other", _snapshot(_entry("a.pdf", 1, "a1")))})
        with pytest.raises(FreezeError):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.repo.reads == [("blob-1", "acme")]

    async def test_every_blob_validated_before_any_write(self, env):
        good = _snapshot(_entry("a.pdf", 1, "a1"))
        bad = _snapshot(_entry("a.pdf", 2, "a2"), tenant="other")
        e = env(
            [_stub("a.pdf", 1, blob="blob-1"), _stub("a.pdf", 2, blob="blob-2")],
            {"blob-1": ("acme", good), "blob-2": ("acme", bad)},
        )
        with pytest.raises(FreezeError):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.adrv.calls == []

    async def test_one_version_of_a_shared_snapshot_keeps_the_blob(self, env):
        snap = _snapshot(_entry("a.pdf", 1, "a1"), _entry("a.pdf", 2, "a2"))
        e = env([_stub("a.pdf", 1)], {"blob-1": ("acme", snap)})
        e.sync.frozen_blobs["blob-1"] = 2  # v2 is still frozen with the same blob
        result = await restore_document("a.pdf", version=1, tenant_id="acme")
        assert result.documents_restored == 1
        assert result.snapshots_deleted == 0 and e.repo.deleted == []
        # Only the v1 entry was written.
        rows = [row for q, p in e.adrv.calls if "MERGE (n:" in q for row in p["rows"]]
        assert all(not str(row["key"]).startswith("{'id': 'a2") for row in rows)
        assert not any(row["key"].get("uid", "").startswith("a2") for row in rows)

    async def test_folder_restores_descendants_frozen_with_it(self, env):
        snap = _snapshot(_entry("folder", 1, "f"), _entry("folder/leaf", 1, "l"))
        e = env(
            [
                _stub("folder", 1),
                _stub("folder/leaf", 1, seed=False),
                _stub("folder/other", 1, seed=False, blob="blob-other"),
                _stub("folder/plain", 1, seed=False, blob=None),
            ],
            {"blob-1": ("acme", snap)},
        )
        result = await restore_document("folder", tenant_id="acme")
        assert result.versions_restored == [1]
        assert result.documents_restored == 2
        assert result.frozen_blob_ids == ["blob-1"]
        finish = next(p for q, p in e.sync.calls if q == restore._RESTORE_FINISH_QUERY)
        assert finish["keys"] == [{"path": "folder", "version": 1}, {"path": "folder/leaf", "version": 1}]
        assert e.repo.reads == [("blob-1", "acme")]

    async def test_seeds_from_several_snapshots(self, env):
        e = env(
            [_stub("a.pdf", 1, blob="blob-1"), _stub("a.pdf", 2, blob="blob-2")],
            {
                "blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1"))),
                "blob-2": ("acme", _snapshot(_entry("a.pdf", 2, "a2"))),
            },
        )
        result = await restore_document("a.pdf", tenant_id="acme")
        assert result.frozen_blob_ids == ["blob-1", "blob-2"]
        assert result.versions_restored == [1, 2]
        assert result.snapshots_deleted == 2
        assert sorted(e.repo.deleted) == [("blob-1", "acme"), ("blob-2", "acme")]

    async def test_public_tenant(self, env):
        snap = _snapshot(_entry("a.pdf", 1, "p", tenant="__public__"), tenant="__public__")
        e = env([_stub("a.pdf", 1, tenant="__public__")], {"blob-1": ("__public__", snap)})
        result = await restore_document("a.pdf", tenant_id=None)
        assert result.found and result.tenant_id is None
        assert e.repo.deleted == [("blob-1", "__public__")]

    async def test_selector_validation(self, env):
        e = env([], {})
        with pytest.raises(ValueError):
            await restore_document(tenant_id="acme")
        with pytest.raises(ValueError):
            await restore_document("a.pdf", tenant_id="")
        with pytest.raises(TypeError):
            await restore_document("a.pdf")  # type: ignore[call-arg]
        assert e.sync.calls == []


class TestQueries:
    def test_merged_labels_union_provenance(self):
        q = restore._node_merge_query(("Entity",))
        assert "MERGE (n:`Entity` {uid: row.key.uid})" in q
        assert "CASE WHEN x IN acc THEN acc ELSE acc + x END" in q

    def test_catalog_key_with_two_fields(self):
        q = restore._relationship_query("HAS_FIELD", "CatalogModel", "ModelField")
        assert "MERGE (b:`ModelField` {name: row.to.name, model: row.to.model})" in q

    def test_unknown_labels_are_rejected(self):
        with pytest.raises(FreezeError):
            restore._node_merge_query(("Mystery",))


class TestEnsureIndexes:
    class _Session:
        def __init__(self, outer) -> None:
            self.outer = outer

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute_write(self, fn):
            return fn(self)

        def run(self, query, **params):
            outer = self.outer
            if outer.error is not None:
                raise outer.error
            outer.queries.append((query, params))
            added = 1 if query.startswith("CREATE INDEX") and outer.missing else 0
            return SimpleNamespace(
                consume=lambda: SimpleNamespace(counters=SimpleNamespace(indexes_added=added))
            )

    class _Driver:
        def __init__(self, missing: bool, error: Exception | None = None) -> None:
            self.missing = missing
            self.error = error
            self.queries: list[tuple[str, dict]] = []

        def session(self, **kwargs):
            return TestEnsureIndexes._Session(self)

    def test_existing_indexes_do_not_wait(self):
        from scinr.newton.ingest.schema import ensure_indexes

        driver = self._Driver(missing=False)
        assert ensure_indexes(driver, restore._RESTORE_INDEXES, database="neo4j") == 0
        assert len(driver.queries) == len(restore._RESTORE_INDEXES)
        assert all(q.startswith("CREATE INDEX") and "IF NOT EXISTS" in q for q, _ in driver.queries)

    def test_created_indexes_are_awaited(self):
        from scinr.newton.ingest.schema import ensure_indexes

        driver = self._Driver(missing=True)
        created = ensure_indexes(driver, ("idx_model_decision_uid",), database="neo4j")
        assert created == 1
        assert driver.queries[-1] == ("CALL db.awaitIndexes($timeout)", {"timeout": 300})

    def test_failure_is_logged_not_raised(self, caplog):
        from scinr.newton.ingest.schema import ensure_indexes

        driver = self._Driver(missing=True, error=RuntimeError("no schema privilege"))
        assert ensure_indexes(driver, ("idx_model_decision_uid",), database="neo4j") == 0
        assert "Could not ensure indexes" in caplog.text

    def test_unknown_name_is_rejected(self):
        from scinr.newton.ingest.schema import ensure_indexes

        with pytest.raises(ValueError, match="Unknown index"):
            ensure_indexes(self._Driver(missing=False), ("nope",), database="neo4j")


def _folder_snapshot(**header) -> dict:
    folder = _entry("folder", 1, "f")
    leaf = _entry("folder/leaf", 1, "l")
    link = _r("IS_COMPOSED_OF", _doc_ref("folder", 1), _doc_ref("folder/leaf", 1))
    folder["relationships"].append(link)
    leaf["relationships"].append(link)
    return _snapshot(folder, leaf, **header)


def _document_merges(e) -> list[dict]:
    return [row for q, p in e.adrv.calls if "MERGE (n:`Document`" in q for row in p["rows"]]


class TestRestoreWithoutStub:
    async def test_deleted_document_is_recreated_from_the_newest_snapshot(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["document"].update(name="a", latest=True, raw_file_id="rf-1", job_id="j1")
        e = env([], {"blob-1": ("acme", _snapshot(entry))})

        result = await restore_document("a.pdf", tenant_id="acme")

        assert result.found and result.documents_restored == 1
        assert result.documents_recreated == 1
        assert result.versions_restored == [1]
        assert e.repo.lookups[0]["path"] == "a.pdf"
        [doc_row] = _document_merges(e)
        assert doc_row["key"] == {"tenant_id": "acme", "path": "a.pdf", "version": 1}
        assert doc_row["props"]["raw_file_id"] == "rf-1"
        # Document merged before any relationship (HAS_STRUCTURE needs it).
        queries = _async_queries(e)
        doc_i = next(i for i, q in enumerate(queries) if "MERGE (n:`Document`" in q)
        first_rel = next(i for i, q in enumerate(queries) if "CREATE (a)-[r:" in q)
        assert doc_i < first_rel
        latest = next(p for q, p in e.sync.calls if q == restore._RECOMPUTE_LATEST_QUERY)
        assert latest == {"tenant_id": "acme", "paths": ["a.pdf"]}
        # A freeze snapshot nobody references any more is deleted.
        assert result.snapshots_deleted == 1 and e.repo.deleted == [("blob-1", "acme")]

    async def test_frozen_flags_are_never_recreated(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["document"].update(frozen=False, frozen_at="x", frozen_keep_annotations=True)
        e = env([], {"blob-1": ("acme", _snapshot(entry))})
        await restore_document("a.pdf", tenant_id="acme")
        [doc_row] = _document_merges(e)
        assert not any(name.startswith("frozen") for name in doc_row["props"])

    @pytest.mark.parametrize("mode", ["backup", "export", None])
    async def test_backups_and_exports_are_kept(self, env, mode):
        snap = _snapshot(_entry("a.pdf", 1, "a1"), mode=mode)
        e = env([], {"blob-1": ("acme", snap)})
        result = await restore_document("a.pdf", tenant_id="acme")
        assert result.documents_recreated == 1
        assert result.snapshots_deleted == 0 and e.repo.deleted == []

    async def test_newest_snapshot_wins(self, env):
        e = env(
            [],
            {
                "old": ("acme", _snapshot(_entry("a.pdf", 1, "old"), mode="backup")),
                "new": ("acme", _snapshot(_entry("a.pdf", 1, "new"), mode="backup")),
            },
        )
        result = await restore_document("a.pdf", version=1, tenant_id="acme")
        assert result.frozen_blob_ids == ["new"]
        assert e.repo.reads == [("new", "acme")]

    async def test_each_version_from_its_newest_snapshot(self, env):
        e = env(
            [],
            {
                "v1": ("acme", _snapshot(_entry("a.pdf", 1, "a1"), mode="backup")),
                "v2": ("acme", _snapshot(_entry("a.pdf", 2, "a2"), mode="backup")),
            },
        )
        result = await restore_document("a.pdf", tenant_id="acme")
        assert sorted(result.frozen_blob_ids) == ["v1", "v2"]
        assert result.versions_restored == [1, 2]
        assert result.documents_recreated == 2
        assert sorted(e.repo.reads) == [("v1", "acme"), ("v2", "acme")]

    async def test_nothing_anywhere_is_not_found(self, env):
        e = env([], {"blob-1": ("acme", _snapshot(_entry("b.pdf", 1, "b")))})
        result = await restore_document("a.pdf", tenant_id="acme")
        assert result.found is False
        assert e.repo.reads == [] and e.adrv.calls == []

    async def test_job_id_lookup(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["document"]["job_id"] = "j1"
        e = env([], {"blob-1": ("acme", _snapshot(entry))})
        result = await restore_document(job_id="j1", tenant_id="acme")
        assert result.documents_recreated == 1
        assert e.repo.lookups[0]["job_id"] == ["j1"]

    async def test_old_metadata_without_job_id_is_not_found_by_job(self, env):
        e = env([], {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        e.repo.metadata["blob-1"] = [{"path": "a.pdf", "version": 1}]
        result = await restore_document(job_id="j1", tenant_id="acme")
        assert result.found is False

    async def test_folder_restores_its_children_and_relinks_them(self, env):
        e = env([], {"blob-1": ("acme", _folder_snapshot())})
        result = await restore_document("folder", tenant_id="acme")
        assert result.documents_recreated == 2
        assert result.versions_restored == [1]
        assert {row["key"]["path"] for row in _document_merges(e)} == {"folder", "folder/leaf"}
        composed = [p for q, p in e.adrv.calls if "`IS_COMPOSED_OF`" in q]
        assert composed and "MATCH (b:`Document`" in next(
            q for q, _ in e.adrv.calls if "`IS_COMPOSED_OF`" in q
        )

    async def test_explicit_blob_restores_matching_entries_only(self, env):
        snap = _snapshot(_entry("a.pdf", 1, "a1"), _entry("b.pdf", 1, "b1"), mode="backup")
        e = env([], {"blob-9": ("acme", snap)})
        result = await restore_document("b.pdf", tenant_id="acme", frozen_blob_id="blob-9")
        assert e.repo.lookups == []
        assert [row["key"]["path"] for row in _document_merges(e)] == ["b.pdf"]
        assert result.frozen_blob_ids == ["blob-9"]

    async def test_explicit_blob_without_a_matching_entry(self, env):
        e = env([], {"blob-9": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        with pytest.raises(FreezeError, match="has no document matching"):
            await restore_document("z.pdf", tenant_id="acme", frozen_blob_id="blob-9")
        assert e.adrv.calls == []

    async def test_explicit_blob_on_its_own_frozen_stub_works_as_usual(self, env):
        e = env([_stub("a.pdf", 1, blob="blob-1", ksn=True)],
                {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1")))})
        result = await restore_document("a.pdf", tenant_id="acme", frozen_blob_id="blob-1")
        assert result.documents_restored == 1 and result.documents_recreated == 0
        assert _document_merges(e) == []
        # the stub's kept structure gets its stale families removed first
        assert any("HAS_MODEL_DECISION" in q and "node_ids" in p for q, p in e.adrv.calls)
        assert result.snapshots_deleted == 1

    @pytest.mark.parametrize("frozen_blob_id", [None, "blob-1"])
    async def test_existing_unfrozen_document_is_refused(self, env, frozen_blob_id):
        e = env([_stub("a.pdf", 1, blob=None)],
                {"blob-1": ("acme", _snapshot(_entry("a.pdf", 1, "a1"), mode="backup"))})
        with pytest.raises(FreezeError, match="delete it first"):
            await restore_document("a.pdf", tenant_id="acme", frozen_blob_id=frozen_blob_id)
        assert e.adrv.calls == []

    async def test_stub_frozen_with_another_snapshot_is_refused(self, env):
        e = env([_stub("a.pdf", 1, blob="blob-1")],
                {"blob-2": ("acme", _snapshot(_entry("a.pdf", 1, "a1"), mode="backup"))})
        with pytest.raises(FreezeError, match="frozen with another snapshot"):
            await restore_document("a.pdf", tenant_id="acme", frozen_blob_id="blob-2")
        assert e.adrv.calls == []

    async def test_backend_without_lookup(self, env, monkeypatch):
        from scinr.newton.freeze.base import FreezeRepository

        e = env([], {})
        monkeypatch.setattr(type(e.repo), "find_snapshots", FreezeRepository.find_snapshots)
        with pytest.raises(FreezeError, match="pass frozen_blob_id"):
            await restore_document("a.pdf", tenant_id="acme")

    async def test_document_link_of_another_tenant_is_rejected(self, env):
        entry = _entry("a.pdf", 1, "a1")
        entry["relationships"].append(
            _r("IS_COMPOSED_OF", _doc_ref("folder", 1, tenant="other"), _doc_ref("a.pdf", 1))
        )
        e = env([], {"blob-1": ("acme", _snapshot(entry))})
        with pytest.raises(FreezeError, match="another tenant"):
            await restore_document("a.pdf", tenant_id="acme")
        assert e.adrv.calls == []

    async def test_missing_raw_file_is_only_a_warning(self, env, monkeypatch, caplog):
        entry = _entry("a.pdf", 1, "a1")
        entry["document"]["raw_file_id"] = "rf-gone"
        env([], {"blob-1": ("acme", _snapshot(entry))})
        monkeypatch.setattr(
            restore,
            "get_config",
            lambda: SimpleNamespace(neo4j_database="neo4j", storage_backend="mongodb"),
        )

        class _RawFiles:
            async def get(self, raw_file_id, *, tenant_id):
                return None

        monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (_RawFiles(), None))
        result = await restore_document("a.pdf", tenant_id="acme")
        assert result.documents_recreated == 1
        assert "rf-gone" in caplog.text and "no longer exists in storage" in caplog.text
