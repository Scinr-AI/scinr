"""
tests/unit/test_graph_mapper_traversal.py — the domain traversal of the extraction
subgraph (``entity_extraction/graph_mapper.py``) against a recording fake sink.

The traversal has no Cypher: what matters is which sink calls it emits, with which
arguments, in which order (the order of the statements of the old whole-function
writers). The Cypher of each call is in ``tests/unit/graph_neo4j/test_extraction_ops.py``.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from scinr.newton.entity_extraction import graph_mapper
from scinr.newton.entity_extraction.graph_mapper import (
    write_extraction_subgraph,
    write_triple_subgraph,
)
from scinr.newton.graph import reset_graph_store, set_graph_store
from scinr.newton.graph.types import Provenance, RawProvenance
from scinr.newton.utils.uid import make_instance_uid, make_uid


class _RecordingSink:
    """Records every call as ``(method, args, kwargs)``; reads return the configured provenance."""

    def __init__(self, raw: RawProvenance | None) -> None:
        self.raw = raw
        self.calls: list[tuple[str, tuple, dict]] = []

    async def read_node_provenance(self, node_full_id):
        self.calls.append(("read_node_provenance", (node_full_id,), {}))
        return self.raw

    def __getattr__(self, name):
        async def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return _record

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def sink_factory():
    holder: dict[str, _RecordingSink] = {}

    def _install(raw: RawProvenance | None) -> _RecordingSink:
        sink = _RecordingSink(raw)
        holder["sink"] = sink

        @asynccontextmanager
        async def _open():
            yield sink

        set_graph_store(SimpleNamespace(extraction=SimpleNamespace(sink=_open)))
        return sink

    try:
        yield _install
    finally:
        reset_graph_store()


_RAW = RawProvenance(tenant_id="acme", created_by_user_id="u1", job_id="j1")
_PROV = Provenance(tenant_id="acme", created_by_user_id="u1", job_id="j1")


class _Country(BaseModel):
    code: str = Field(json_schema_extra={"instance_key": True})
    name: str = Field(json_schema_extra={"entity_label": "CountryName"})


class _Item(BaseModel):
    label: str


class _Complementary(BaseModel):
    text: str


class _Root(BaseModel):
    country: _Country
    items: list[_Item]
    tags: list[str] = Field(json_schema_extra={"entity_label": "Tag"})
    related: list[str] = Field(
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "_Country", "join_via": {"related": "code"}, "rel_type": "RELATED"}
            ]
        }
    )
    first: str = Field(
        json_schema_extra={
            "entity_label": "First",
            "field_relationships": [{"to_field": "second", "rel_type": "NEXT"}],
        }
    )
    second: str = Field(json_schema_extra={"entity_label": "Second"})
    extra: _Complementary | None = None


async def test_extraction_emission_order_and_arguments(sink_factory):
    sink = sink_factory(_RAW)
    instance = _Root(
        country=_Country(code="ES", name="Spain"),
        items=[_Item(label="a")],
        tags=["x"],
        related=["FR"],
        first="one",
        second="two",
        extra=_Complementary(text="t"),
    )
    with patch.object(uuid, "uuid4", lambda: uuid.UUID(int=0xABC)):
        await write_extraction_subgraph("n", instance, "_Root", ["_Complementary"], "doc", "er")

    by_name = dict.fromkeys(sink.names())
    assert set(by_name) == {
        "read_node_provenance",
        "delete_stale_extraction_result",
        "create_extraction_result",
        "link_catalog_model",
        "merge_keyed_instance",
        "create_instance",
        "merge_labeled_entity",
        "link_reference",
        "set_instance_properties",
        "merge_instance_shell",
        "link_instance_relationship",
        "link_field_relationship",
    }
    calls = {n: [c for c in sink.calls if c[0] == n] for n in by_name}
    assert sink.names()[:3] == [
        "read_node_provenance",
        "delete_stale_extraction_result",
        "create_extraction_result",
    ]
    # The ExtractionResult carries the validated stored provenance and a timestamp.
    _, args, _ = calls["create_extraction_result"][0]
    assert args[:4] == ("n", "er", "doc", "_Root")
    assert args[5] == _PROV
    # Primary first, then the complementary model that was actually returned.
    assert [(c[1], c[2]) for c in calls["link_catalog_model"]] == [
        (("er", "_Root"), {"primary": True}),
        (("er", "_Complementary"), {"primary": False}),
    ]
    # Keyed sub-model: deterministic uid with the stored tenant, not in a list.
    _, args, _ = calls["merge_keyed_instance"][0]
    assert args[:6] == (
        "ExtractionResult",
        "er",
        "HAS_COUNTRY",
        make_instance_uid("_Country", {"code": "es"}, "acme"),
        "_Country",
        {"code": "es"},
    )
    assert args[7] is None
    # Keyless list item: uuid4-derived uid and its index.
    assert calls["create_instance"][0][1][:3] == ("ExtractionResult", "er", "HAS_ITEMS")
    assert calls["create_instance"][0][1][3] == uuid.UUID(int=0xABC).hex[:16]
    assert calls["create_instance"][0][1][-1] == 0
    # Entity labels: tenant-scoped uid, references with and without index.
    entity_uids = [c[1][0] for c in calls["merge_labeled_entity"]]
    assert make_uid("le", "acme", "CountryName", "spain") in entity_uids
    refs = {c[1][3]: c[1][4] for c in calls["link_reference"]}
    assert refs["tags"] == 0 and refs["first"] is None
    # Instance relationship: shell first, then the link, source is the root result.
    assert calls["merge_instance_shell"][0][1][1:3] == ("_Country", {"code": "fr"})
    assert calls["link_instance_relationship"][0][1][0:2] == ("ExtractionResult", "er")
    # Root scalars are one batch, in the order of the model fields.
    root_props = [c for c in calls["set_instance_properties"] if c[1][0] == "ExtractionResult"][0]
    assert list(root_props[1][2]) == ["tags", "related", "first", "second"]
    # Field relationships are the last thing emitted.
    assert sink.names()[-1] == "link_field_relationship"
    assert calls["link_field_relationship"][0][1][2] == "NEXT"


async def test_provenance_comes_after_the_node_check_and_before_any_write(sink_factory):
    sink = sink_factory(_RAW)
    await write_extraction_subgraph("n", _Item(label="a"), "_Item", [], "doc", "er")
    assert sink.names()[:4] == [
        "read_node_provenance",
        "delete_stale_extraction_result",
        "create_extraction_result",
        "link_catalog_model",
    ]
    assert sink.names()[-1] == "set_instance_properties"


@pytest.mark.parametrize("method", ["write_extraction_subgraph", "write_triple_subgraph"])
async def test_missing_structure_node_raises_before_any_write(sink_factory, method):
    sink = sink_factory(None)
    with pytest.raises(RuntimeError, match=f"{method}: StructureNode not found"):
        if method == "write_extraction_subgraph":
            await write_extraction_subgraph("n", _Item(label="a"), "M", [], "doc", "er")
        else:
            await write_triple_subgraph("n", SimpleNamespace(triples=[]), "doc", "er")
    assert sink.names() == ["read_node_provenance"]


@pytest.mark.parametrize("method", ["write_extraction_subgraph", "write_triple_subgraph"])
async def test_node_without_tenant_is_refused_before_any_write(sink_factory, method):
    sink = sink_factory(RawProvenance(tenant_id=None, created_by_user_id=None, job_id=None))
    with pytest.raises(RuntimeError, match="has no tenant_id"):
        if method == "write_extraction_subgraph":
            await write_extraction_subgraph("n", _Item(label="a"), "M", [], "doc", "er")
        else:
            await write_triple_subgraph("n", SimpleNamespace(triples=[]), "doc", "er")
    assert sink.names() == ["read_node_provenance"]


async def test_triple_emission_order_skips_incomplete_triples(sink_factory):
    sink = sink_factory(_RAW)
    triples = SimpleNamespace(
        triples=[
            SimpleNamespace(subject="Aspirin", predicate="is made by", object="Bayer"),
            SimpleNamespace(subject="", predicate="x", object="y"),
        ]
    )
    await write_triple_subgraph("n", triples, "doc", "er")

    assert sink.names() == [
        "read_node_provenance",
        "delete_triple_extraction_result",
        "create_triple_extraction_result",
        "merge_entity",
        "merge_entity",
        "link_triple_entity",
        "link_triple_entity",
        "merge_predicate",
    ]
    subj_uid = make_uid("entity", "acme", "aspirin")
    obj_uid = make_uid("entity", "acme", "bayer")
    assert sink.calls[3][1] == (subj_uid, "Aspirin", "aspirin", _PROV)
    assert sink.calls[4][1] == (obj_uid, "Bayer", "bayer", _PROV)
    assert sink.calls[5][1] == ("er", subj_uid, "subject")
    assert sink.calls[6][1] == ("er", obj_uid, "object")
    assert sink.calls[7][1] == (subj_uid, obj_uid, "IS_MADE_BY", "is made by")


async def test_depth_limit_stops_the_recursion(sink_factory, caplog):
    sink = sink_factory(_RAW)
    await graph_mapper._write_model_fields(
        sink=sink,
        instance=_Item(label="a"),
        parent_uid="p",
        parent_label="ModelInstance",
        field_path_prefix="",
        entity_nodes={},
        depth=11,
        prov=_PROV,
    )
    assert sink.calls == []
    assert "max depth reached" in caplog.text
