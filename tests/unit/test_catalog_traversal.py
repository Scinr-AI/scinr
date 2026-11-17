"""
tests/unit/test_catalog_traversal.py — domain traversal of the catalog and themes.

``ensure_catalog_models`` / ``ensure_theme_structure`` (``annotation/catalog.py``)
are driven with a recording sink: no engine, no Cypher. What is asserted is the
sequence of sink calls, which is the order the Neo4j statements have always
been emitted in (the graph-store golden trace depends on it).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from scinr.newton.annotation import catalog as catalog_mod


class _RecordingSink:
    """A :class:`CatalogSink` that appends ``(method, args)`` to ``calls``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        async def _record(*args):
            self.calls.append((name, args))

        return _record


class _Child(BaseModel):
    """A nested model."""

    label: str = Field(default="", json_schema_extra={"entity_label": "Label"})


class _Parent(BaseModel):
    """A model with a key, an entity label, a nested model and relationships."""

    code: str = Field(json_schema_extra={"instance_key": True, "entity_label": "Code"})
    name: str = Field(
        default="",
        json_schema_extra={
            "entity_label": "Name",
            "field_relationships": [
                {"to_field": "code", "rel_type": "IDENTIFIED_BY"},
                {"to_field": "code", "rel_type": "bad type"},
                {"to_field": "missing", "rel_type": "IGNORED"},
            ],
        },
    )
    child: _Child | None = Field(
        default=None,
        description="the child",
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "_Child", "rel_type": "HAS_CHILD", "join_via": {"a": "b", "c": "d"}},
                {"target_model": "Unknown", "rel_type": "HAS_UNKNOWN", "join_via": {"a": "b"}},
                {"target_model": "_Child", "rel_type": "lower", "join_via": {"a": "b"}},
            ]
        },
    )
    others: list[_Child] = Field(default_factory=list)


def _registry():
    return SimpleNamespace(_themes={"default": SimpleNamespace(models=[_Parent])})


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(catalog_mod, "get_config", lambda: SimpleNamespace(full_docstring=False))


class TestEnsureCatalogModels:
    async def test_emits_the_statements_in_the_order_of_the_old_function(self):
        sink = _RecordingSink()

        await catalog_mod.ensure_catalog_models(_registry(), sink)

        assert sink.calls == [
            # CatalogModel nodes: seeds first, then the discovered models
            ("merge_catalog_model", ("_Parent", "A model with a key, an entity label, a nested model and relationships.", True)),
            ("merge_catalog_model", ("_Child", "A nested model.", False)),
            # HAS_FIELD, model by model, field by field
            ("merge_model_field", ("_Parent", "code", "str", "", True)),
            ("merge_model_field", ("_Parent", "name", "str", "", False)),
            ("merge_model_field", ("_Parent", "child", f"{__name__}._Child | None", "the child", False)),
            ("merge_model_field", ("_Parent", "others", f"list[{__name__}._Child]", "", False)),
            ("merge_model_field", ("_Child", "label", "str", "", False)),
            # AGGREGATES
            ("merge_aggregates", ("_Parent", "_Child", "child", False, False)),
            ("merge_aggregates", ("_Parent", "_Child", "others", True, False)),
            # Pass A, then A' (once, with the set of keys)
            ("set_field_metadata", ("_Parent", "code", True, "Code")),
            ("set_field_metadata", ("_Parent", "name", False, "Name")),
            ("set_field_metadata", ("_Child", "label", False, "Label")),
            ("ensure_instance_key_indexes", ({"code"},)),
            # Pass B
            ("link_produces_entity", ("_Parent", "code", "Code")),
            ("link_produces_entity", ("_Parent", "name", "Name")),
            ("link_produces_entity", ("_Child", "label", "Label")),
            # Pass C: the invalid and unresolvable relationships are skipped
            ("merge_field_relationship", ("IDENTIFIED_BY", "Name", "Code", "_Parent", "name", "code")),
            # Pass D: unknown target and invalid type are skipped
            ("merge_instance_relationship", ("HAS_CHILD", "_Parent", "_Child", "child", '{"a": "b", "c": "d"}', True)),
        ]

    async def test_instance_key_indexes_are_requested_once_even_without_keys(self):
        sink = _RecordingSink()
        registry = SimpleNamespace(_themes={"default": SimpleNamespace(models=[_Child])})

        await catalog_mod.ensure_catalog_models(registry, sink)

        assert [c for c in sink.calls if c[0] == "ensure_instance_key_indexes"] == [
            ("ensure_instance_key_indexes", (set(),))
        ]


class TestEnsureThemeStructure:
    async def test_themes_then_subtopics_then_models(self):
        sink = _RecordingSink()
        registry = SimpleNamespace(
            get_neo4j_theme_structure=lambda: [
                {"path": "a", "name": "a", "parent_path": None, "model_names": ["M1", "M2"]},
                {"path": "a/b", "name": "b", "parent_path": "a", "model_names": ["M3"]},
            ]
        )

        await catalog_mod.ensure_theme_structure(registry, sink)

        assert sink.calls == [
            ("merge_theme", ("a", "a")),
            ("merge_theme", ("a/b", "b")),
            ("link_subtopic", ("a", "a/b")),
            ("link_model_to_theme", ("M1", "a")),
            ("link_model_to_theme", ("M2", "a")),
            ("link_model_to_theme", ("M3", "a/b")),
        ]


class TestMemoizedWrappers:
    @pytest.fixture
    def store(self, monkeypatch):
        sinks: list[_RecordingSink] = []

        class _Annotation:
            @asynccontextmanager
            async def catalog_sink(self):
                sink = _RecordingSink()
                sinks.append(sink)
                yield sink

        monkeypatch.setattr(catalog_mod, "get_graph_store", lambda: SimpleNamespace(annotation=_Annotation()))
        catalog_mod.reset_catalog_memoization()
        yield sinks
        catalog_mod.reset_catalog_memoization()

    async def test_each_setup_opens_its_own_sink_once_per_process(self, store):
        registry = SimpleNamespace(
            _themes={"default": SimpleNamespace(models=[_Child])},
            get_neo4j_theme_structure=lambda: [
                {"path": "a", "name": "a", "parent_path": None, "model_names": []}
            ],
        )

        for _ in range(3):
            await catalog_mod.ensure_catalog_models_once(registry)
            await catalog_mod.ensure_theme_structure_once(registry)

        assert len(store) == 2
        assert store[0].calls[0][0] == "merge_catalog_model"
        assert store[1].calls == [("merge_theme", ("a", "a"))]
