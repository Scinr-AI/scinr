"""
tests/unit/test_llm_semaphore_first.py — Stages 3/4 take the LLM (Bedrock)
semaphore BEFORE building per-node context/prompt/schema.

Regression guard for the fan-out memory problem: ``asyncio.gather`` over every
node of a document used to build every node's context/prompt up front and park
them all waiting for an LLM slot. With the slot taken first, at most
``llm_concurrency`` nodes hold that memory at any time.

Everything is faked: no Neo4j, no LLM, no configure() needed.
"""

from __future__ import annotations

import asyncio

import pytest

from scinr.newton.annotation import nodes as ann_nodes
from scinr.newton.entity_extraction import nodes as ent_nodes

_N_NODES = 60


class _Counter:
    """Tracks how many coroutines are between ``enter()`` and ``leave()``."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    def enter(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def leave(self) -> None:
        self.in_flight -= 1


# ── Annotation (Stage 3) ──────────────────────────────────────────────────────


@pytest.fixture
def annotation_env(monkeypatch):
    counter = _Counter()
    written: list[str] = []

    async def fake_fetch_context(node_data, driver):
        counter.enter()  # context now exists
        await asyncio.sleep(0)
        return {"node": node_data["node_id"]}

    async def fake_decide_model(ctx, node_id, theme, user_context):
        await asyncio.sleep(0.005)
        counter.leave()  # prompt/LLM call finished, context released
        return object(), None

    async def fake_write_decision(driver, full_id, decision, document_name, node_id):
        written.append(node_id)
        return None

    monkeypatch.setattr(ann_nodes, "get_async_driver", lambda: object())
    monkeypatch.setattr(ann_nodes, "get_neo4j_semaphore", lambda: asyncio.Semaphore(10))
    monkeypatch.setattr(ann_nodes, "_fetch_node_context", fake_fetch_context)
    monkeypatch.setattr(ann_nodes, "_decide_model", fake_decide_model)
    monkeypatch.setattr(ann_nodes, "_write_decision", fake_write_decision)
    return counter, written


def _ann_node(i: int) -> dict:
    return {"node_id": f"n{i}", "full_id": f"doc/n{i}", "theme": "default"}


class TestAnnotationSemaphoreFirst:
    @pytest.mark.parametrize("slots", [1, 3])
    async def test_contexts_alive_never_exceed_llm_slots(self, annotation_env, slots):
        counter, written = annotation_env
        sem = asyncio.Semaphore(slots)

        results = await asyncio.wait_for(
            asyncio.gather(
                *[
                    ann_nodes.process_single_annotation_node(_ann_node(i), "doc", sem)
                    for i in range(_N_NODES)
                ]
            ),
            timeout=5,
        )

        assert counter.max_in_flight <= slots
        assert len(results) == _N_NODES
        assert all(r["error"] is None and r["decision"] is not None for r in results)
        assert len(written) == _N_NODES

    async def test_failed_fetch_returns_error_and_releases_the_slot(
        self, annotation_env, monkeypatch
    ):
        counter, _ = annotation_env

        async def fetch_none_for_first(node_data, driver):
            if node_data["node_id"] == "n0":
                return None
            counter.enter()
            return {"node": node_data["node_id"]}

        monkeypatch.setattr(ann_nodes, "_fetch_node_context", fetch_none_for_first)
        sem = asyncio.Semaphore(1)

        failed, ok = await asyncio.wait_for(
            asyncio.gather(
                ann_nodes.process_single_annotation_node(_ann_node(0), "doc", sem),
                ann_nodes.process_single_annotation_node(_ann_node(1), "doc", sem),
            ),
            timeout=5,
        )

        assert failed == {
            "node_id": "n0",
            "decision": None,
            "error": "fetch_context failed for n0",
        }
        assert ok["error"] is None  # second node got the slot the first one released

    async def test_neo4j_write_happens_outside_the_llm_slot(self, annotation_env, monkeypatch):
        sem = asyncio.Semaphore(1)
        held_during_write: list[bool] = []

        async def write(driver, full_id, decision, document_name, node_id):
            held_during_write.append(sem.locked())
            return None

        monkeypatch.setattr(ann_nodes, "_write_decision", write)

        await ann_nodes.process_single_annotation_node(_ann_node(0), "doc", sem)

        assert held_during_write == [False]

    async def test_write_error_is_reported(self, annotation_env, monkeypatch):
        async def write(driver, full_id, decision, document_name, node_id):
            return "write failed for n0: boom"

        monkeypatch.setattr(ann_nodes, "_write_decision", write)

        result = await ann_nodes.process_single_annotation_node(
            _ann_node(0), "doc", asyncio.Semaphore(1)
        )

        assert result["error"] == "write failed for n0: boom"


# ── Entity extraction (Stage 4) ───────────────────────────────────────────────


@pytest.fixture
def entity_env(monkeypatch):
    counter = _Counter()
    marked: list[str] = []

    def fake_compose_schema(target):
        counter.enter()  # schema now exists
        return object, None

    async def fake_extract_entities(schema, info_units, node_full_id, node_id=None, node_title=None):
        await asyncio.sleep(0.005)
        counter.leave()  # LLM call finished, schema/prompt released
        return {"entities": []}, None

    async def fake_write_entities(driver, target, extraction, document_name):
        return None

    async def fake_mark_extracted(driver, node_full_id):
        marked.append(node_full_id)
        return None

    monkeypatch.setattr(ent_nodes, "get_async_driver", lambda: object())
    monkeypatch.setattr(ent_nodes, "get_neo4j_semaphore", lambda: asyncio.Semaphore(10))
    monkeypatch.setattr(ent_nodes, "_compose_schema", fake_compose_schema)
    monkeypatch.setattr(ent_nodes, "_extract_entities", fake_extract_entities)
    monkeypatch.setattr(ent_nodes, "_write_entities", fake_write_entities)
    monkeypatch.setattr(ent_nodes, "_mark_extracted", fake_mark_extracted)
    return counter, marked


def _target(i: int) -> dict:
    return {"node_full_id": f"doc/n{i}", "node_id": f"n{i}", "node_title": "t", "info_units": []}


class TestEntityExtractionSemaphoreFirst:
    @pytest.mark.parametrize("slots", [1, 3])
    async def test_schemas_alive_never_exceed_llm_slots(self, entity_env, slots):
        counter, marked = entity_env
        sem = asyncio.Semaphore(slots)

        results = await asyncio.wait_for(
            asyncio.gather(
                *[
                    ent_nodes.process_single_extraction_target(_target(i), "doc", sem)
                    for i in range(_N_NODES)
                ]
            ),
            timeout=5,
        )

        assert counter.max_in_flight <= slots
        assert len(results) == _N_NODES
        assert all(r["error"] is None for r in results)
        assert len(marked) == _N_NODES

    async def test_schema_error_returns_error_and_releases_the_slot(
        self, entity_env, monkeypatch
    ):
        def compose(target):
            if target["node_id"] == "n0":
                return None, "compose failed"
            return object, None

        monkeypatch.setattr(ent_nodes, "_compose_schema", compose)
        sem = asyncio.Semaphore(1)

        failed, ok = await asyncio.wait_for(
            asyncio.gather(
                ent_nodes.process_single_extraction_target(_target(0), "doc", sem),
                ent_nodes.process_single_extraction_target(_target(1), "doc", sem),
            ),
            timeout=5,
        )

        assert failed == {"node_full_id": "doc/n0", "extraction": None, "error": "compose failed"}
        assert ok["error"] is None

    async def test_neo4j_writes_happen_outside_the_llm_slot(self, entity_env, monkeypatch):
        sem = asyncio.Semaphore(1)
        held: list[bool] = []

        async def write(driver, target, extraction, document_name):
            held.append(sem.locked())
            return None

        async def mark(driver, node_full_id):
            held.append(sem.locked())
            return None

        monkeypatch.setattr(ent_nodes, "_write_entities", write)
        monkeypatch.setattr(ent_nodes, "_mark_extracted", mark)

        await ent_nodes.process_single_extraction_target(_target(0), "doc", sem)

        assert held == [False, False]
