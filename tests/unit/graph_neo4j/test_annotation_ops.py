"""
tests/unit/graph_neo4j/test_annotation_ops.py — ``store.annotation`` over Neo4j.

The Cypher that used to live in the old annotation Cypher module is exercised through
``Neo4jGraphStore`` with a fake async driver that records every statement:
the ``(tenant_id, <key>)`` instance-key indexes (moved from
``tests/unit/test_instance_key_indexes.py``), the catalog and theme setup, the
readers and the autocommit writers. Tenancy of the readers and of
``write_manual_annotation`` is in ``test_document_tenancy.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from scinr.newton.annotation import catalog as catalog_mod
from scinr.newton.annotation.models import (
    AnnotationDecision,
    ComplementaryModel,
    ProposedField,
)
from scinr.newton.graph import reset_graph_store, set_graph_store
from scinr.newton.graph.neo4j import annotation as annotation_mod
from scinr.newton.graph.neo4j import driver as driver_mod
from scinr.newton.graph.neo4j.annotation import (
    _ensure_instance_key_indexes,
    instance_key_index_name,
)
from scinr.newton.graph.neo4j.store import Neo4jGraphStore

# ---------------------------------------------------------------------------
# Recording fakes
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, single=None, data=None):
        self._single = single
        self._data = data or []

    async def single(self):
        return self._single

    async def data(self):
        return self._data

    async def consume(self):
        return None


class _Session:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.driver.calls.append((" ".join(query.split()), params))
        return self.driver.respond(query, params)


class _Driver:
    def __init__(self, respond=None):
        self.calls: list[tuple[str, dict]] = []
        self.respond = respond or (lambda q, p: _Result())

    def session(self, **kwargs):
        return _Session(self)


@pytest.fixture(autouse=True)
def _store(monkeypatch):
    monkeypatch.setattr(
        annotation_mod,
        "get_config",
        lambda: SimpleNamespace(neo4j_database="neo4j", full_docstring=False),
    )
    monkeypatch.setattr(catalog_mod, "get_config", lambda: SimpleNamespace(full_docstring=False))
    set_graph_store(Neo4jGraphStore())
    try:
        yield
    finally:
        reset_graph_store()


def _use(monkeypatch, respond=None) -> _Driver:
    driver = _Driver(respond)
    monkeypatch.setattr(driver_mod, "get_async_driver", lambda: driver)
    return driver


def _annotation_ops():
    from scinr.newton.graph import get_graph_store

    return get_graph_store().annotation


# ---------------------------------------------------------------------------
# Instance-key indexes
# ---------------------------------------------------------------------------


def test_identifier_names_map_directly():
    assert instance_key_index_name("code") == "idx_mi_key_code"
    assert instance_key_index_name("variation_code") == "idx_mi_key_variation_code"


def test_non_identifier_names_are_sanitised_with_hash_suffix():
    a = instance_key_index_name("a-b")
    b = instance_key_index_name("a b")
    assert a.startswith("idx_mi_key_a_b_") and b.startswith("idx_mi_key_a_b_")
    assert a != b
    assert all(ch.isalnum() or ch == "_" for ch in a)


async def test_one_index_per_distinct_key_property():
    session = AsyncMock()
    await _ensure_instance_key_indexes(session, {"code", "variation_code"})
    queries = [c.args[0] for c in session.run.call_args_list]
    assert queries == [
        "CREATE INDEX idx_mi_key_code IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.tenant_id, mi.`code`)",
        "CREATE INDEX idx_mi_key_variation_code IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.tenant_id, mi.`variation_code`)",
    ]


async def test_property_backticks_are_escaped():
    session = AsyncMock()
    await _ensure_instance_key_indexes(session, {"we`ird"})
    assert "mi.`we``ird`" in session.run.call_args_list[0].args[0]


async def test_index_failure_does_not_raise(caplog):
    session = AsyncMock()
    session.run.side_effect = RuntimeError("no schema privilege")
    await _ensure_instance_key_indexes(session, {"code"})
    assert "Could not create index idx_mi_key_code" in caplog.text


# ---------------------------------------------------------------------------
# Catalog and theme setup: the domain traversal over the Neo4j catalog sink
# ---------------------------------------------------------------------------


async def _ensure_catalog(registry):
    async with _annotation_ops().catalog_sink() as sink:
        await catalog_mod.ensure_catalog_models(registry, sink)


async def _ensure_themes(registry):
    async with _annotation_ops().catalog_sink() as sink:
        await catalog_mod.ensure_theme_structure(registry, sink)


class _Child(BaseModel):
    """A nested model."""

    label: str = Field(default="", json_schema_extra={"entity_label": "Label"})


class _Parent(BaseModel):
    """A model with a key, an entity label and a nested model."""

    code: str = Field(json_schema_extra={"instance_key": True, "entity_label": "Code"})
    child: _Child | None = None
    others: list[_Child] = Field(default_factory=list)


def _registry():
    theme = SimpleNamespace(models=[_Parent])
    return SimpleNamespace(_themes={"default": theme})


class TestEnsureCatalog:
    async def test_statements_in_the_order_of_the_old_function(self, monkeypatch):
        driver = _use(monkeypatch)

        await _ensure_catalog(_registry())

        queries = [q for q, _ in driver.calls]
        # CatalogModel nodes (parent first, then the discovered nested model) ...
        assert queries[0].startswith("MERGE (cm:CatalogModel {name: $name})")
        assert [p["name"] for q, p in driver.calls[:2]] == ["_Parent", "_Child"]
        # ... then HAS_FIELD, AGGREGATES, Pass A, A', B.
        first_of = {
            tag: next(i for i, q in enumerate(queries) if tag in q)
            for tag in (
                "HAS_FIELD",
                "AGGREGATES",
                "SET mf.is_instance_key",
                "CREATE INDEX",
                "PRODUCES_ENTITY",
            )
        }
        assert list(first_of.values()) == sorted(first_of.values())
        assert any("idx_mi_key_code" in q for q in queries)

    async def test_aggregates_carry_is_list_and_required(self, monkeypatch):
        driver = _use(monkeypatch)

        await _ensure_catalog(_registry())

        aggregates = {
            p["field_name"]: (p["is_list"], p["required"])
            for q, p in driver.calls
            if "AGGREGATES" in q
        }
        assert aggregates == {"child": (False, False), "others": (True, False)}

    async def test_index_failure_does_not_stop_the_catalog(self, monkeypatch):
        def respond(q, p):
            if "CREATE INDEX" in q:
                raise RuntimeError("no schema privilege")
            return _Result()

        driver = _use(monkeypatch, respond)

        await _ensure_catalog(_registry())

        assert any("PRODUCES_ENTITY" in q for q, _ in driver.calls)


class TestEnsureThemes:
    async def test_themes_subtopics_and_models(self, monkeypatch):
        driver = _use(monkeypatch)
        registry = SimpleNamespace(
            get_neo4j_theme_structure=lambda: [
                {"path": "a", "name": "a", "parent_path": None, "model_names": ["M1"]},
                {"path": "a/b", "name": "b", "parent_path": "a", "model_names": []},
            ]
        )

        await _ensure_themes(registry)

        assert [q.split(" ", 2)[0] for q, _ in driver.calls] == [
            "MERGE",
            "MERGE",
            "MATCH",
            "MERGE",
        ]
        assert driver.calls[0][1] == {"path": "a", "name": "a"}
        assert driver.calls[2][1] == {"parent_path": "a", "child_path": "a/b"}
        assert driver.calls[3][1] == {"model_name": "M1", "path": "a"}


class TestCatalogSinkStatements:
    """Each sink method is one autocommit statement with the Cypher of the old function."""

    async def _emit(self, monkeypatch, call):
        driver = _use(monkeypatch)
        async with _annotation_ops().catalog_sink() as sink:
            await call(sink)
        assert len(driver.calls) == 1
        return driver.calls[0]

    async def test_merge_catalog_model(self, monkeypatch):
        q, p = await self._emit(monkeypatch, lambda s: s.merge_catalog_model("M", "desc", True))
        assert q.startswith("MERGE (cm:CatalogModel {name: $name}) SET cm.description = $description")
        assert p == {"name": "M", "description": "desc", "selectable": True}

    async def test_merge_model_field(self, monkeypatch):
        q, p = await self._emit(
            monkeypatch, lambda s: s.merge_model_field("M", "f", "str", "d", True)
        )
        assert "MERGE (cm)-[r:HAS_FIELD]->(mf)" in q
        assert p == {
            "field_name": "f",
            "field_type": "str",
            "model_name": "M",
            "description": "d",
            "required": True,
        }

    async def test_merge_aggregates(self, monkeypatch):
        q, p = await self._emit(monkeypatch, lambda s: s.merge_aggregates("A", "B", "f", True, False))
        assert "MERGE (src)-[r:AGGREGATES {field_name: $field_name}]->(tgt)" in q
        assert p == {
            "src_name": "A",
            "tgt_name": "B",
            "field_name": "f",
            "is_list": True,
            "required": False,
        }

    async def test_set_field_metadata(self, monkeypatch):
        q, p = await self._emit(monkeypatch, lambda s: s.set_field_metadata("M", "f", True, None))
        assert "SET mf.is_instance_key = $is_instance_key, mf.entity_label = $entity_label" in q
        assert p == {
            "field_name": "f",
            "model_name": "M",
            "is_instance_key": True,
            "entity_label": None,
        }

    async def test_link_produces_entity(self, monkeypatch):
        q, p = await self._emit(monkeypatch, lambda s: s.link_produces_entity("M", "f", "Label"))
        assert "MERGE (cm)-[:PRODUCES_ENTITY {field_name: $field_name}]->(el)" in q
        assert p == {"entity_label": "Label", "model_name": "M", "field_name": "f"}

    async def test_merge_field_relationship_interpolates_the_type_in_backticks(self, monkeypatch):
        q, p = await self._emit(
            monkeypatch,
            lambda s: s.merge_field_relationship("HAS_X", "L1", "L2", "M", "from_f", "to_f"),
        )
        assert (
            "MERGE (el_src)-[r:`HAS_X` {via_model: $via_model, from_field: $from_field, "
            "to_field: $to_field}]->(el_tgt)" in q
        )
        assert p == {
            "source_label": "L1",
            "target_label": "L2",
            "via_model": "M",
            "from_field": "from_f",
            "to_field": "to_f",
        }

    async def test_merge_instance_relationship_interpolates_the_type_in_backticks(self, monkeypatch):
        q, p = await self._emit(
            monkeypatch,
            lambda s: s.merge_instance_relationship("HAS_Y", "A", "B", "f", '{"a": "b"}', False),
        )
        assert (
            "MERGE (src)-[r:`HAS_Y` {via_field: $via_field, join_via: $join_via_json}]->(tgt) "
            "SET r.composite = $is_composite" in q
        )
        assert p == {
            "src_name": "A",
            "tgt_name": "B",
            "via_field": "f",
            "join_via_json": '{"a": "b"}',
            "is_composite": False,
        }

    async def test_merge_theme_link_subtopic_and_link_model_to_theme(self, monkeypatch):
        driver = _use(monkeypatch)
        async with _annotation_ops().catalog_sink() as sink:
            await sink.merge_theme("a", "A")
            await sink.link_subtopic("a", "a/b")
            await sink.link_model_to_theme("M", "a")
        assert [q.split(" ", 1)[0] for q, _ in driver.calls] == ["MERGE", "MATCH", "MERGE"]
        assert driver.calls[0][1] == {"path": "a", "name": "A"}
        assert "HAS_SUBTOPIC" in driver.calls[1][0]
        assert "BELONGS_TO_THEME" in driver.calls[2][0]

    async def test_ensure_instance_key_indexes_is_best_effort(self, monkeypatch, caplog):
        def respond(q, p):
            raise RuntimeError("no schema privilege")

        driver = _use(monkeypatch, respond)
        async with _annotation_ops().catalog_sink() as sink:
            await sink.ensure_instance_key_indexes({"code"})
        assert len(driver.calls) == 1
        assert "Could not create index idx_mi_key_code" in caplog.text

    async def test_one_session_is_shared_by_every_call(self, monkeypatch):
        sessions = []
        driver = _use(monkeypatch)
        original = driver.session
        driver.session = lambda **kw: sessions.append(1) or original(**kw)
        async with _annotation_ops().catalog_sink() as sink:
            await sink.merge_theme("a", "A")
            await sink.merge_theme("b", "B")
        assert sessions == [1]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


class TestReaders:
    async def test_only_unannotated_adds_the_decision_filter(self, monkeypatch):
        driver = _use(monkeypatch)
        ops = _annotation_ops()

        await ops.fetch_nodes_to_annotate(tenant_id="A", doc_path="d")
        await ops.fetch_nodes_to_annotate(tenant_id="A", doc_path="d", only_unannotated=True)

        assert "NOT (n)-[:HAS_MODEL_DECISION]->()" not in driver.calls[0][0]
        assert "NOT (n)-[:HAS_MODEL_DECISION]->()" in driver.calls[1][0]

    async def test_fetch_node_context_uses_one_session_and_stops_at_max_depth(self, monkeypatch):
        sessions = []

        class _CountingDriver(_Driver):
            def session(self, **kwargs):
                sessions.append(1)
                return super().session(**kwargs)

        def respond(q, p):
            if "HAS_INFO_UNIT" in q:
                return _Result(data=[{"info_unit_id": "u", "title": "t"}])
            if "HAS_CHILD" in q:
                return _Result(data=[{"full_id": f"{p['node_id']}/c", "node_id": "c"}])
            return _Result()

        driver = _CountingDriver(respond)
        monkeypatch.setattr(driver_mod, "get_async_driver", lambda: driver)

        ctx = await _annotation_ops().fetch_node_context("root", "root", max_depth=2)

        assert len(sessions) == 1
        assert ctx["depth"] == 0 and not ctx["depth_limit_reached"]
        assert ctx["children"][0]["children"][0]["depth_limit_reached"] is True

    async def test_context_instructions_none_without_a_record(self, monkeypatch):
        _use(monkeypatch)

        assert (
            await _annotation_ops().fetch_document_context_instructions(
                tenant_id=None, doc_path="d"
            )
            is None
        )

    async def test_context_instructions_value(self, monkeypatch):
        _use(monkeypatch, lambda q, p: _Result(single={"context_instructions": "be brief"}))

        assert (
            await _annotation_ops().fetch_document_context_instructions(
                tenant_id="A", doc_path="d"
            )
            == "be brief"
        )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _decision(**kw) -> AnnotationDecision:
    base = {
        "matched_model_class": "Product",
        "confidence": "high",
        "rationale": "r",
        "coverage_gaps": [],
        "propose_new_model": False,
        "complementary_models": [],
    }
    base.update(kw)
    return AnnotationDecision(**base)


class TestWriteAnnotation:
    async def test_missing_structure_node_raises_before_any_write(self, monkeypatch):
        driver = _use(monkeypatch, lambda q, p: _Result(single={"cnt": 0}))

        with pytest.raises(RuntimeError, match="StructureNode not found"):
            await _annotation_ops().write_annotation("t::d::1::n", _decision(), "d")

        assert len(driver.calls) == 1

    async def test_matched_decision_statements(self, monkeypatch):
        driver = _use(monkeypatch, lambda q, p: _Result(single={"cnt": 1}))
        decision = _decision(
            complementary_models=[ComplementaryModel(model_class="Other", coverage_note="n")]
        )

        await _annotation_ops().write_annotation("t::d::1::n", decision, "d")

        queries = [q for q, _ in driver.calls]
        # guard, 5 stale-delete queries, decision, MATCHED_MODEL, complementary
        assert queries[0].startswith("MATCH (n:StructureNode {id: $node_id}) RETURN count(n)")
        assert all("DETACH DELETE" in q for q in queries[1:6])
        assert "CREATE (md:ModelDecision" in queries[6]
        assert "MATCHED_MODEL" in queries[7]
        assert "HAS_COMPLEMENTARY_MATCH" in queries[8]
        assert len(queries) == 9
        decision_params = driver.calls[6][1]
        assert decision_params["node_id"] == "t::d::1::n"
        assert decision_params["matched_model_class"] == "Product"

    async def test_unmatched_decision_writes_the_proposed_model(self, monkeypatch):
        driver = _use(monkeypatch, lambda q, p: _Result(single={"cnt": 1}))
        decision = _decision(
            matched_model_class=None,
            propose_new_model=True,
            proposed_schema_name="NewModel",
            proposed_schema_fields=[
                ProposedField(
                    field_name="f", field_type="str", description="d", required=True
                )
            ],
        )

        await _annotation_ops().write_annotation("t::d::1::n", decision, "d")

        queries = [q for q, _ in driver.calls]
        assert not any("MATCHED_MODEL" in q for q in queries)
        assert any("HAS_PROPOSED_MODEL]->(pm)" in q for q in queries)
        assert any("CREATE (f:ProposedField" in q for q in queries)


class TestWriteManualAnnotation:
    async def test_no_qualifying_nodes_returns_zero_without_writing(self, monkeypatch):
        driver = _use(monkeypatch)

        count = await _annotation_ops().write_manual_annotation(
            "d", "Product", tenant_id=None, doc_path="d"
        )

        assert count == 0
        assert len(driver.calls) == 1

    async def test_per_node_statements(self, monkeypatch):
        def respond(q, p):
            if "RETURN DISTINCT n.id AS full_node_id" in q:
                return _Result(data=[{"full_node_id": "n1"}, {"full_node_id": "n2"}])
            return _Result()

        driver = _use(monkeypatch, respond)

        count = await _annotation_ops().write_manual_annotation(
            "d", "Product", tenant_id="A", doc_path="d"
        )

        assert count == 2
        queries = [q for q, _ in driver.calls]
        # reader, then per node: extraction purge, 5 decision deletes, create, link
        assert len(queries) == 1 + 2 * 8
        assert "HAS_EXTRACTION" in queries[1]
        assert "CREATE (md:ModelDecision" in queries[7]
        assert "source: 'manual'" in queries[7]
        assert "MATCHED_MODEL" in queries[8]
