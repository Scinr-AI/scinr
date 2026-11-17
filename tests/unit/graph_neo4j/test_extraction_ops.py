"""
tests/unit/graph_neo4j/test_extraction_ops.py — ``store.extraction`` over Neo4j.

The Cypher of the extraction sink (``Neo4jExtractionSink``) and of the target
reader, exercised with mocked sessions driven by the domain traversal of
``entity_extraction/graph_mapper.py`` and, through ``Neo4jGraphStore``, a fake
async driver that records every statement. The traversal itself (which calls the
sink emits) is in ``tests/unit/test_graph_mapper_traversal.py``. Tenancy of
``fetch_targets`` is in ``test_document_tenancy.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field, create_model

from scinr.newton.entity_extraction.graph_mapper import (
    _merge_labeled_entity,
    _write_model_fields,
    write_extraction_subgraph,
    write_triple_subgraph,
)
from scinr.newton.graph import get_graph_store, reset_graph_store, set_graph_store
from scinr.newton.graph.neo4j import driver as driver_mod
from scinr.newton.graph.neo4j import extraction as extraction_mod
from scinr.newton.graph.neo4j.extraction import Neo4jExtractionSink
from scinr.newton.graph.neo4j.store import Neo4jGraphStore
from scinr.newton.graph.types import Provenance

# ---------------------------------------------------------------------------
# _write_model_fields — end-to-end regression test for the
# `scalar_props[field_name] = scalarValues` fix (previously `= value`)
# ---------------------------------------------------------------------------


async def test_write_model_fields_writes_flattened_scalar_values_for_mixed_list() -> None:
    """
    Regression test for the `_write_model_fields()` list-case bug: a `list`
    field whose real runtime value mixes a raw dict, a plain string, and a
    None (simulating something that reaches this function despite not being
    caught by the schema_composer.py layer-1/layer-2 defenses — e.g. a
    catalog model field that isn't a supplementary field at all) must be
    written to Neo4j using the *flattened* `scalarValues` accumulator (dict
    -> "k: v" via `_stringify_if_dict`, None dropped), not the raw original
    `value` list. The buggy version wrote `scalar_props[field_name] = value`,
    i.e. the untouched `[{"a": "1"}, "plain", None]`, which would have
    raised `Neo.ClientError.Statement.TypeError` against a real Neo4j
    session because Neo4j properties cannot contain dicts/None inside a
    list.

    Uses a dynamically-created Pydantic model (no dependency on any real
    catalog model) and a minimal `AsyncMock` session — no real Neo4j driver.
    """
    DynamicModel = create_model("DynamicModel", mixed_field=(list[Any], ...))
    instance = DynamicModel(mixed_field=[{"a": "1"}, "plain", None])

    mock_session = AsyncMock()

    # Should not raise.
    await _write_model_fields(
        sink=Neo4jExtractionSink(mock_session),
        instance=instance,
        parent_uid="test-uid",
        parent_label="ExtractionResult",
        field_path_prefix="",
        entity_nodes={},
        depth=0,
        prov=Provenance(tenant_id="__public__", created_by_user_id=None, job_id=None),
    )

    # Locate the final batch-SET call: it is the one whose Cypher query
    # contains "SET" and whose kwargs include our field name.
    set_calls = [
        call
        for call in mock_session.run.call_args_list
        if "SET" in call.args[0] and "mixed_field" in call.kwargs
    ]
    assert len(set_calls) == 1, (
        f"expected exactly one SET call carrying 'mixed_field', "
        f"got {len(set_calls)} (all calls: {mock_session.run.call_args_list!r})"
    )
    assert set_calls[0].kwargs["mixed_field"] == ["a: 1", "plain"]
    assert set_calls[0].kwargs["parent_uid"] == "test-uid"


# ---------------------------------------------------------------------------
# Multi-tenant isolation — tenant_id is folded into the uid of every
# merge-deduplicated node (ModelInstance with instance_key, LabeledEntity,
# Entity), and provenance is written on every node.
# ---------------------------------------------------------------------------


class _Country(BaseModel):
    code: str = Field(json_schema_extra={"instance_key": True})
    name: str = Field(json_schema_extra={"entity_label": "CountryName"})


class _Wrapper(BaseModel):
    country: _Country
    related: list[str] = Field(json_schema_extra={"instance_relationships": [
        {"target_model": "_Country", "join_via": {"related": "code"}, "rel_type": "RELATED"}
    ]})


async def _merged_uids(tenant_id: str | None, user: str = "u1", job: str = "j1") -> tuple[dict, list]:
    """Run _write_model_fields for the same content and return
    {cypher-kind: uid} for every MERGE'd node plus the raw calls."""
    session = AsyncMock()
    await _write_model_fields(
        sink=Neo4jExtractionSink(session),
        instance=_Wrapper(country=_Country(code="ES", name="Spain"), related=["FR"]),
        parent_uid="er",
        parent_label="ExtractionResult",
        field_path_prefix="",
        entity_nodes={},
        depth=0,
        prov=Provenance(tenant_id=tenant_id, created_by_user_id=user, job_id=job),
    )
    uids: dict[str, str] = {}
    for call in session.run.call_args_list:
        q, kw = call.args[0], call.kwargs
        if "MERGE (child:ModelInstance" in q:
            uids["mi_child"] = kw["child_uid"]
        elif "MERGE (tgt:ModelInstance" in q:
            uids["mi_target"] = kw["tgt_uid"]
        elif "MERGE (le:LabeledEntity" in q:
            uids["labeled_entity"] = kw["uid"]
    return uids, session.run.call_args_list


async def test_same_content_different_tenants_never_share_merged_nodes() -> None:
    a, _ = await _merged_uids("tenant_A")
    b, _ = await _merged_uids("tenant_B")
    assert set(a) == {"mi_child", "mi_target", "labeled_entity"}
    for kind in a:
        assert a[kind] != b[kind], f"{kind} collides across tenants"


async def test_same_tenant_different_job_dedups_to_same_nodes() -> None:
    first, _ = await _merged_uids("tenant_A", user="u1", job="j1")
    second, _ = await _merged_uids("tenant_A", user="u2", job="j2")
    assert first == second


async def test_merged_nodes_write_provenance_on_create_and_on_match() -> None:
    _, calls = await _merged_uids("tenant_A", user="u1", job="j1")
    merges = [c for c in calls if "MERGE (child:ModelInstance" in c.args[0]
              or "MERGE (tgt:ModelInstance" in c.args[0]
              or "MERGE (le:LabeledEntity" in c.args[0]]
    assert len(merges) == 3
    for c in merges:
        q = c.args[0]
        on_match = q.split("ON MATCH", 1)[1]
        assert ".tenant_id = $tenant_id" in on_match
        assert ".created_by_user_ids = CASE" in on_match
        assert ".job_ids = CASE" in on_match
        assert c.kwargs["tenant_id"] == "tenant_A"
        assert c.kwargs["created_by_user_id"] == "u1"
        assert c.kwargs["job_id"] == "j1"


async def test_labeled_entity_is_merged_by_tenant_scoped_uid() -> None:
    """MERGE must match on the tenant-scoped uid, not on (label, normalized_value) —
    otherwise two tenants' entities would still collapse onto one node."""
    session = AsyncMock()
    sink = Neo4jExtractionSink(session)
    prov_a = Provenance(tenant_id="tenant_A", created_by_user_id=None, job_id=None)
    prov_b = Provenance(tenant_id="tenant_B", created_by_user_id=None, job_id=None)
    uid_a = await _merge_labeled_entity(sink, "CountryName", "Spain", prov_a)
    uid_b = await _merge_labeled_entity(sink, "CountryName", "Spain", prov_b)
    assert uid_a != uid_b
    for call in session.run.call_args_list:
        assert "MERGE (le:LabeledEntity {uid: $uid})" in call.args[0]


# ---------------------------------------------------------------------------
# Labelled lookups — every MATCH on the parent / relationship source carries
# its label, so it is a seek on the uid constraint, not a whole-graph scan.
# ---------------------------------------------------------------------------


async def test_parent_matches_carry_the_parent_label() -> None:
    _, calls = await _merged_uids("tenant_A")
    parent_matches = [c.args[0] for c in calls if "MATCH (parent" in c.args[0]]
    assert parent_matches, "expected MATCHes on the parent node"
    for q in parent_matches:
        assert "MATCH (parent:ExtractionResult {uid: $parent_uid})" in q or (
            "MATCH (parent:ModelInstance {uid: $parent_uid})" in q
        ), q
    # The nested _Country hangs from a ModelInstance parent (its REFERENCES write).
    assert any("MATCH (parent:ModelInstance {uid: $parent_uid})" in q for q in parent_matches)
    assert any("MATCH (parent:ExtractionResult {uid: $parent_uid})" in q for q in parent_matches)


async def test_instance_relationship_source_carries_its_label() -> None:
    _, calls = await _merged_uids("tenant_A")
    src_matches = [c.args[0] for c in calls if "MATCH (src" in c.args[0]]
    assert len(src_matches) == 1
    # _Wrapper is the root model: the relationship source is the ExtractionResult.
    assert "MATCH (src:ExtractionResult {uid: $src_uid})" in src_matches[0]


async def test_write_model_fields_rejects_unknown_parent_label() -> None:
    import pytest

    with pytest.raises(ValueError, match="parent label"):
        await _write_model_fields(
            sink=Neo4jExtractionSink(AsyncMock()),
            instance=_Country(code="ES", name="Spain"),
            parent_uid="x",
            parent_label="Document) DETACH DELETE (n",
            field_path_prefix="",
            entity_nodes={},
            depth=0,
            prov=Provenance(tenant_id="__public__", created_by_user_id=None, job_id=None),
        )


# ---------------------------------------------------------------------------
# Through the store, with a fake async driver that records every statement
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, single=None, data=None):
        self._single = single
        self._data = data or []

    async def single(self):
        return self._single

    async def data(self):
        return self._data


class _Session:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.driver.calls.append((" ".join(query.split()), params))
        return self.driver.respond(" ".join(query.split()), params)


class _Driver:
    def __init__(self, respond=None):
        self.calls: list[tuple[str, dict]] = []
        self.respond = respond or (lambda q, p: _Result())

    def session(self, **kwargs):
        return _Session(self)


@pytest.fixture(autouse=True)
def _store(monkeypatch):
    monkeypatch.setattr(
        extraction_mod, "get_config", lambda: SimpleNamespace(neo4j_database="neo4j")
    )
    set_graph_store(Neo4jGraphStore())
    try:
        yield
    finally:
        reset_graph_store()


def _use(monkeypatch, respond=None) -> _Driver:
    driver = _Driver(respond)
    monkeypatch.setattr(driver_mod, "get_async_driver", lambda: driver)
    return driver


_PROVENANCE = {"cnt": 1, "tenant_id": "acme", "created_by_user_id": "u1", "job_id": "j1"}


def _node_found(query, params):
    return _Result(single=_PROVENANCE) if "RETURN count(n) AS cnt" in query else _Result()


class TestFetchTargets:
    async def test_falls_back_to_the_query_without_apoc(self, monkeypatch, caplog):
        def respond(query, params):
            if "apoc.coll.sortMaps" in query:
                raise RuntimeError("apoc not installed")
            return _Result(
                data=[
                    {
                        "node_full_id": "acme::F/doc::1::n",
                        "node_id": "n",
                        "node_title": "T",
                        "model_class": None,
                        "complementary": [],
                        "supplementary": [],
                        "info_units": [{"uid": "b", "order": 2}, {"uid": "a", "order": 1}],
                    }
                ]
            )

        driver = _use(monkeypatch, respond)
        targets = await get_graph_store().extraction.fetch_targets(
            tenant_id="acme", doc_path="F/doc", only_unextracted=True
        )

        assert len(driver.calls) == 2
        assert "apoc.coll.sortMaps" not in driver.calls[1][0]
        assert "NOT EXISTS { MATCH (n)-[:HAS_EXTRACTION]->(:ExtractionResult) }" in driver.calls[1][0]
        assert "falling back to unsorted query" in caplog.text
        assert [iu["uid"] for iu in targets[0]["info_units"]] == ["a", "b"]
        assert targets[0]["model_class"] is None


class TestMarkInfoUnitsExtracted:
    async def test_sets_the_timestamp_on_unextracted_units_only(self, monkeypatch):
        driver = _use(monkeypatch, lambda q, p: _Result(single={"updated": 2}))
        await get_graph_store().extraction.mark_info_units_extracted("acme::F/doc::1::n")

        query, params = driver.calls[0]
        assert "WHERE iu.extracted IS NULL SET iu.extracted = $timestamp" in query
        assert params["nid"] == "acme::F/doc::1::n"


class TestWriteSubgraphs:
    async def test_triple_subgraph_statement_order(self, monkeypatch):
        driver = _use(monkeypatch, _node_found)
        triples = SimpleNamespace(
            triples=[
                SimpleNamespace(subject="Aspirin", predicate="is made by", object="Bayer"),
                SimpleNamespace(subject="", predicate="x", object="y"),
            ]
        )
        await write_triple_subgraph(
            "acme::F/doc::1::n", triples, "doc", "er-uid"
        )

        queries = [q for q, _ in driver.calls]
        assert "RETURN count(n) AS cnt" in queries[0]
        assert "DETACH DELETE er" in queries[1]
        assert "model_class: 'Triple'" in queries[2]
        assert sum("MERGE (e:Entity" in q for q in queries) == 2
        assert "{role: 'subject'}" in queries[5] and "{role: 'object'}" in queries[6]
        assert "MERGE (subj)-[r:`IS_MADE_BY`]->(obj)" in queries[7]
        assert len(queries) == 8
        assert driver.calls[2][1]["tenant_id"] == "acme"

    async def test_extraction_subgraph_writes_result_catalog_links_and_fields(self, monkeypatch):
        driver = _use(monkeypatch, _node_found)

        class _Primary(BaseModel):
            code: str = Field(json_schema_extra={"instance_key": True})
            note: str = "n"

        await write_extraction_subgraph(
            "acme::F/doc::1::n", _Primary(code="ES"), "_Primary", [], "doc", "er-uid"
        )

        queries = [q for q, _ in driver.calls]
        assert "RETURN count(n) AS cnt" in queries[0]
        assert any("CREATE (er:ExtractionResult" in q for q in queries)
        assert any("USES_PRIMARY_MODEL" in q for q in queries)
        assert queries[-1].endswith("SET parent.`code` = $code, parent.`note` = $note")

    @pytest.mark.parametrize("method", ["write_extraction_subgraph", "write_triple_subgraph"])
    async def test_missing_structure_node_raises_runtime_error(self, monkeypatch, method):
        _use(monkeypatch, lambda q, p: _Result(single={"cnt": 0}))
        args = (
            ("n", object(), "M", [], "doc", "uid")
            if method == "write_extraction_subgraph"
            else ("n", SimpleNamespace(triples=[]), "doc", "uid")
        )
        with pytest.raises(RuntimeError, match=f"{method}: StructureNode not found"):
            await {
                "write_extraction_subgraph": write_extraction_subgraph,
                "write_triple_subgraph": write_triple_subgraph,
            }[method](*args)
