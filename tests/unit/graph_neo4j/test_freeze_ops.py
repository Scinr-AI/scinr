"""
tests/unit/graph_neo4j/test_freeze_ops.py — Unit tests for
scinr.newton.graph.neo4j.freeze: the Cypher of the export reads, the marks and
the streaming ``ExportReader`` of ``export_snapshot``.

No real Neo4j: a fake driver records the statements and serves canned rows.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scinr.newton.graph.neo4j import driver as neo_driver
from scinr.newton.graph.neo4j import freeze as neo_freeze
from scinr.newton.graph.neo4j.gc import GC_REACHABILITY_MAX_HOPS
from scinr.newton.graph.neo4j.store import Neo4jGraphStore
from scinr.newton.graph.types import DocumentInfo, RawNode, RawRelationship
from scinr.newton.ingest import freeze as ingest_freeze


def test_document_links_are_tenant_scoped():
    query = neo_freeze._DOCUMENT_RELATIONSHIPS_EXPORT_QUERY
    assert "m.tenant_id = $tenant_id" in query and "n.tenant_id = $tenant_id" in query
    assert "IS_COMPOSED_OF|HAS_NEWER_VERSION" in query


def test_subtree_query_is_hop_bounded_and_skips_catalog():
    fragment = neo_freeze._SUBTREE_NODES_FRAGMENT
    assert f"[*1..{GC_REACHABILITY_MAX_HOPS}]" in fragment
    for label in ingest_freeze.CATALOG_KEY_FIELDS:
        assert f"n:{label}" in fragment
    assert "tenant_id: $tenant_id" in fragment


class _Session:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):
        self.driver.runs.append((query, params))
        return self.driver.rows[query]


class _Driver:
    def __init__(self, rows: dict[str, list[dict]]) -> None:
        self.rows = rows
        self.runs: list[tuple[str, dict]] = []
        self.sessions = 0
        self.closed = 0

    def session(self, **kwargs):
        self.sessions += 1
        return _Session(self)

    def close(self):
        self.closed += 1


def _doc() -> DocumentInfo:
    props = {"path": "a.pdf", "version": 1, "tenant_id": "acme"}
    return DocumentInfo("a.pdf", 1, "acme", True, props)


def _rel_row() -> dict:
    return {
        "type": "HAS_CHILD",
        "from_labels": ["StructureNode"],
        "from_key": {"id": "a"},
        "to_labels": ["StructureNode"],
        "to_key": {"id": "b"},
        "properties": {"order": 0},
    }


@pytest.fixture
def store_and_driver(monkeypatch):
    rows = {
        neo_freeze._SUBTREE_NODES_EXPORT_QUERY: [
            {"labels": ["InfoUnit"], "properties": {"uid": "iu", "tenant_id": "acme"}}
        ],
        neo_freeze._DOCUMENT_RELATIONSHIPS_EXPORT_QUERY: [_rel_row()],
        neo_freeze._SUBTREE_RELATIONSHIPS_EXPORT_QUERY: [_rel_row()],
    }
    driver = _Driver(rows)
    monkeypatch.setattr(neo_freeze, "get_config", lambda: SimpleNamespace(neo4j_database="neo4j"))
    monkeypatch.setattr(neo_driver, "get_driver", lambda: driver)
    return Neo4jGraphStore(), driver


class TestExportReader:
    def test_one_session_serves_every_read_of_the_attempt(self, store_and_driver):
        store, driver = store_and_driver
        doc = _doc()

        def attempt(reader):
            nodes = list(reader.subtree_nodes(doc))
            doc_rels = list(reader.document_relationships(doc))
            sub_rels = list(reader.subtree_relationships(doc))
            return nodes, doc_rels, sub_rels

        nodes, doc_rels, sub_rels = store.freeze.export_snapshot([doc], "acme", attempt)

        assert nodes == [RawNode(("InfoUnit",), {"uid": "iu", "tenant_id": "acme"})]
        assert doc_rels == sub_rels == [
            RawRelationship("HAS_CHILD", ("StructureNode",), {"id": "a"}, ("StructureNode",), {"id": "b"}, {"order": 0})
        ]
        assert driver.sessions == 1 and driver.closed == 1
        assert [q for q, _ in driver.runs] == [
            neo_freeze._SUBTREE_NODES_EXPORT_QUERY,
            neo_freeze._DOCUMENT_RELATIONSHIPS_EXPORT_QUERY,
            neo_freeze._SUBTREE_RELATIONSHIPS_EXPORT_QUERY,
        ]
        assert all(p == {"tenant_id": "acme", "path": "a.pdf", "version": 1} for _, p in driver.runs)

    def test_reads_are_lazy_generators(self, store_and_driver):
        """Nothing runs until the caller iterates, and rows come out one at a time."""
        store, driver = store_and_driver
        doc = _doc()

        def attempt(reader):
            iterator = reader.subtree_nodes(doc)
            assert iter(iterator) is iterator
            assert driver.runs == []
            first = next(iterator)
            assert len(driver.runs) == 1
            return first

        assert store.freeze.export_snapshot([doc], "acme", attempt).labels == ("InfoUnit",)
