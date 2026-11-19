"""
tests/equivalence/test_snapshot_portability.py -- A Neo4j snapshot restores the same graph in SQL.

Requirement 7 of ``plans/sql-graph-backends.md``: the SQL backends read the snapshot format
of Neo4j (``SNAPSHOT_SCHEMA_VERSION = 1``). A folder with two fully processed documents is
exported from Neo4j to a file. As the reference, Neo4j deletes the folder and restores it
from that file (``restore_document(frozen_blob_id=...)``); the same file is then stored in
the SQL freeze backend and restored the same way into an empty SQL graph. The data nodes and
the relationships of both restored graphs must be equal (catalog nodes are not part of a
snapshot: the restore only creates them by key, so they and the relationships between them
are left out of the comparison). The Neo4j restore is also checked against the graph it was
exported from (equal except for the ``frozen = false`` a restore leaves on the Documents).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from scinr.newton.ingest.freeze import CATALOG_KEY_FIELDS
from tests.contract._helpers import diff, same
from tests.golden import _runner
from tests.golden._dump import normalize_artifacts
from tests.golden._llm import deterministic_uuid4
from tests.golden._scenarios import TENANT_A, USER, Env, prepare_script

pytestmark = pytest.mark.integration


def _catalog(key: str) -> bool:
    return key.split("|", 1)[0] in CATALOG_KEY_FIELDS


def _data_part(graph: dict[str, Any]) -> dict[str, Any]:
    """The normalized dump without catalog nodes and catalog <-> catalog relationships."""
    _, graph = normalize_artifacts([], graph, {})
    return {
        "nodes": [n for n in graph["nodes"] if not _catalog(n["key"])],
        "relationships": [
            r for r in graph["relationships"] if not (_catalog(r["from"]) and _catalog(r["to"]))
        ],
    }


_SNAPSHOT_METADATA = {
    "frozen_at": "2026-01-01T00:00:00+00:00",
    "keep_flags": None,
    "mode": "export",
    "path": "Port",
    "version": None,
    "documents": [],
}


async def _restore(snapshot: Path) -> Any:
    """Store *snapshot* in the configured freeze backend and restore the folder from it."""
    from scinr.newton.freeze.factory import get_freeze_storage
    from scinr.newton.ingest.freeze import SNAPSHOT_SCHEMA_VERSION
    from scinr.newton.ingest.restore import restore_document

    blob_id = await get_freeze_storage().store_snapshot(
        snapshot,
        tenant_id=TENANT_A,
        metadata={"schema_version": SNAPSHOT_SCHEMA_VERSION, **_SNAPSHOT_METADATA},
    )
    return await restore_document("Port", tenant_id=TENANT_A, frozen_blob_id=blob_id, concurrency=4)


def _neo4j_dump() -> dict[str, Any]:
    from neo4j import GraphDatabase

    from tests.golden._dump import dump_graph

    driver = GraphDatabase.driver(
        _runner.NEO4J_URI, auth=(_runner.NEO4J_USER, _runner.NEO4J_PASSWORD)
    )
    try:
        return dump_graph(driver, _runner.NEO4J_DATABASE)
    finally:
        driver.close()


def _unfrozen(graph: dict[str, Any]) -> dict[str, Any]:
    """*graph* with ``frozen = false`` removed from the Documents (what a restore adds)."""
    nodes = [
        {**n, "props": {k: v for k, v in n["props"].items() if (k, v) != ("frozen", False)}}
        for n in graph["nodes"]
    ]
    return {**graph, "nodes": nodes}


async def _neo4j_reference(tmp: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Seed Neo4j, export the folder, then delete and restore it there.

    Returns the snapshot file, the graph it was exported from and the restored graph.
    """
    from scinr.newton.ingest.deletion import delete_document
    from scinr.newton.ingest.freeze import export_document_snapshot

    _runner.configure_harness()
    prepare_script()
    env = Env(tmp=tmp)
    with deterministic_uuid4():
        await env.full("one.txt", folder="Port", tenant=TENANT_A, created_by=USER, job_id="job-p")
        await env.full("two.txt", folder="Port", tenant=TENANT_A, created_by=USER, job_id="job-p")
    snapshot = tmp / "snapshot.json"
    await export_document_snapshot(
        "Port", tenant_id=TENANT_A, destination="file", file_path=snapshot
    )
    exported = _neo4j_dump()
    await delete_document("Port", tenant_id=TENANT_A)
    await _restore(snapshot)
    return snapshot, exported, _neo4j_dump()


async def test_neo4j_snapshot_restores_in_sql(neo4j_available, sql_engine):
    from tests.sql._invariants import check_all

    with tempfile.TemporaryDirectory(prefix="scinr_portability_") as tmp:
        snapshot, exported, neo4j_restored = await _neo4j_reference(Path(tmp))
        reference = _data_part(neo4j_restored)
        assert same(_data_part(exported), _unfrozen(reference)), diff(
            _data_part(exported), _unfrozen(reference)
        )

        _runner.configure_harness(**sql_engine.settings)
        prepare_script()
        result = await _restore(snapshot)
    documents = [n for n in exported["nodes"] if "Document" in n["labels"]]
    assert result.documents_restored == result.documents_recreated == len(documents), result
    actual = _data_part(sql_engine.dump())
    assert reference["nodes"] and reference["relationships"]
    assert same(reference, actual), diff(reference, actual)
    settings = sql_engine.settings
    check_all(
        settings["sql_url"], settings["sql_schema"] if sql_engine.name == "postgres" else None
    )
