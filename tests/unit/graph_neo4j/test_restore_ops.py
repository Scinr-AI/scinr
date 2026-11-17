"""
tests/unit/graph_neo4j/test_restore_ops.py — Unit tests for
scinr.newton.graph.neo4j.restore: the batch Cypher and the ``Neo4jRestoreWriter``
(lanes, concurrency, ``drain`` / ``abort``).

No real Neo4j: a fake async driver records the transactions.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scinr.newton.exceptions import FreezeError
from scinr.newton.graph.neo4j import restore as neo_restore
from scinr.newton.graph.neo4j.restore import Neo4jRestoreWriter, cypher_identifier
from scinr.newton.graph.types import RestoreCounters


class TestQueries:
    def test_merged_labels_union_provenance(self):
        q = neo_restore._node_merge_query(("Entity",))
        assert "MERGE (n:`Entity` {uid: row.key.uid})" in q
        assert "CASE WHEN x IN acc THEN acc ELSE acc + x END" in q

    def test_catalog_key_with_two_fields(self):
        q = neo_restore._relationship_query("HAS_FIELD", "CatalogModel", "ModelField")
        assert "MERGE (b:`ModelField` {name: row.to.name, model: row.to.model})" in q

    def test_unknown_labels_are_rejected(self):
        with pytest.raises(FreezeError):
            neo_restore._node_merge_query(("Mystery",))


@pytest.mark.parametrize("name", ["HAS_CHILD", "5_YEAR_PERIOD_CALCULATED_FROM", "_x"])
def test_cypher_identifier_accepts_names_the_extraction_writes(name):
    assert cypher_identifier(name) == f"`{name}`"


@pytest.mark.parametrize("name", ["", "HAS CHILD", "x`y", None])
def test_cypher_identifier_rejects_unsafe_names(name):
    with pytest.raises(FreezeError, match="Invalid label"):
        cypher_identifier(name)


class _Result:
    async def consume(self):
        return SimpleNamespace(
            counters=SimpleNamespace(nodes_created=2, relationships_created=3)
        )


class _Tx:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver

    async def run(self, query: str, **params):
        self.driver.calls.append((query, params))
        return _Result()


class _Session:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute_write(self, fn, *args):
        d = self.driver
        d.in_flight += 1
        d.max_in_flight = max(d.max_in_flight, d.in_flight)
        try:
            await asyncio.sleep(0)
            return await fn(_Tx(d), *args)
        finally:
            d.in_flight -= 1


class _Driver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def session(self, **kwargs):
        return _Session(self)


class TestRestoreWriter:
    async def test_counters_reach_on_done_as_restore_counters(self):
        driver = _Driver()
        writer = Neo4jRestoreWriter(driver, "neo4j", 2)
        seen: list[RestoreCounters] = []
        await writer.merge_nodes(("nodes", ("InfoUnit",)), ("InfoUnit",), [{"key": {"uid": "u"}, "props": {}}], seen.append)
        await writer.create_relationships(
            ("relationships", ("HAS_CHILD", "StructureNode", "StructureNode")),
            ("HAS_CHILD", "StructureNode", "StructureNode"),
            [],
            seen.append,
        )
        await writer.drain()
        assert seen == [RestoreCounters(2, 3), RestoreCounters(2, 3)]

    async def test_delete_stale_runs_the_requested_families_only(self):
        driver = _Driver()
        writer = Neo4jRestoreWriter(driver, "neo4j", 1)
        await writer.delete_stale(("stale",), ["n1"], model_decisions=False, extraction_results=True)
        await writer.drain()
        assert len(driver.calls) == 1
        assert "HAS_EXTRACTION" in driver.calls[0][0]

    @pytest.mark.parametrize("concurrency", [1, 3])
    async def test_concurrency_is_bounded_by_the_producer(self, concurrency):
        driver = _Driver()
        writer = Neo4jRestoreWriter(driver, "neo4j", concurrency)
        for i in range(10):
            await writer.merge_nodes(("nodes", i), ("InfoUnit",), [], lambda _: None)
        await writer.drain()
        assert driver.max_in_flight <= concurrency

    async def test_same_lane_batches_never_overlap(self):
        driver = _Driver()
        writer = Neo4jRestoreWriter(driver, "neo4j", 4)
        for _ in range(6):
            await writer.merge_nodes(("nodes", "same"), ("InfoUnit",), [], lambda _: None)
        await writer.drain()
        assert driver.max_in_flight == 1
