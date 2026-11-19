"""
tests/unit/test_llm_stage_call_sites.py — Every LLM call asks for the LLM of its stage (WP-B2).

Each test replaces ``get_llm`` (or ``get_prompt_family`` / ``make_system_message``) with a
spy that records the ``stage`` it receives and raises ``_Stop``, then runs the real
function up to that call.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from scinr.newton.config import configure
from scinr.newton.models.document_structure import NodeRole, StructureNode


class _Stop(Exception):
    """Raised by the spies to stop the function under test right after the call."""


class _FakeLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _configured():
    configure(
        llm=_FakeLLM(),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
    )


class _Spy:
    def __init__(self) -> None:
        self.stages: list[str | None] = []
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.stages.append(kwargs.get("stage"))
        self.calls.append((args, kwargs))
        raise _Stop


def _spy(monkeypatch, target: str) -> _Spy:
    spy = _Spy()
    monkeypatch.setattr(target, spy)
    return spy


@pytest.mark.asyncio
async def test_extract_chunk_uses_extraction(monkeypatch):
    from scinr.newton.extraction.extraction import extract_chunk

    spy = _spy(monkeypatch, "scinr.newton.config.get_llm")
    with pytest.raises(_Stop):
        await extract_chunk(prev_page="", curr_pages=["x"], active_hierarchy="(none)")
    assert spy.stages == ["extraction"]


@pytest.mark.asyncio
async def test_consolidation_uses_extraction(monkeypatch):
    from scinr.newton.extraction.structure_consolidation import consolidate_structure

    node = StructureNode(
        node_id="p1_n1", role=NodeRole.SUBSECTION, title="Heading", appearance_order=1
    )
    spy = _spy(monkeypatch, "scinr.newton.config.get_llm")
    with pytest.raises(_Stop):
        await consolidate_structure([[node]], llm=None)
    assert spy.stages == ["extraction"]


@pytest.mark.asyncio
async def test_annotation_uses_annotation(monkeypatch):
    from scinr.newton.annotation import nodes

    monkeypatch.setattr(nodes, "build_annotation_decision_prompt", lambda theme_node: "prompt")
    monkeypatch.setattr(nodes, "_build_annotation_human_message", lambda ctx, user_context: "hm")
    spy = _spy(monkeypatch, "scinr.newton.annotation.nodes.get_llm")
    with pytest.raises(_Stop):
        await nodes._decide_model({}, "id", "default", "")
    assert spy.stages == ["annotation"]


@pytest.mark.asyncio
async def test_entity_extraction_uses_entity_extraction(monkeypatch):
    from scinr.newton.entity_extraction import nodes
    from scinr.newton.models.default.triple import Triple

    spy = _spy(monkeypatch, "scinr.newton.entity_extraction.nodes.get_llm")
    with pytest.raises(_Stop):
        await nodes._extract_entities(Triple, [], "id")
    assert spy.stages == ["entity_extraction"]


@pytest.mark.asyncio
async def test_entity_extraction_system_message_uses_stage_caching(monkeypatch):
    from scinr.newton.entity_extraction import nodes
    from scinr.newton.models.default.triple import Triple

    monkeypatch.setattr(
        "scinr.newton.entity_extraction.nodes.get_llm", lambda *a, **k: _FakeLLM()
    )
    monkeypatch.setattr(_FakeLLM, "with_structured_output", lambda self, *a, **k: object(), raising=False)
    spy = _spy(monkeypatch, "scinr.newton.entity_extraction.nodes.make_system_message")
    with pytest.raises(_Stop):
        await nodes._extract_entities(Triple, [], "id")
    assert spy.stages == ["entity_extraction"]


def _sheet() -> dict:
    return {
        "sheet_name": "S",
        "headers": ["a"],
        "preview_markdown": "| a |",
        "total_rows": 1,
    }


@pytest.mark.asyncio
async def test_tabular_calls_use_tabular(monkeypatch):
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.tabular import nodes, prompts

    spy = _spy(monkeypatch, "scinr.newton.tabular.nodes.get_llm")
    monkeypatch.setattr(prompts, "build_tabular_theme_prompt", lambda **kw: "p")
    monkeypatch.setattr(nodes, "build_tabular_decision_prompt", lambda *a, **kw: "p")
    monkeypatch.setattr(nodes, "build_tabular_mapping_prompt", lambda **kw: "p")

    decision = AnnotationDecision(
        matched_model_class="M", confidence="high", rationale="r"
    )
    state = {
        "current_sheet": _sheet(),
        "document_name": "f.xlsx",
        "current_theme": "default",
        "current_decision": decision,
    }
    import scinr.newton.utils.theme_registry as theme_registry

    monkeypatch.setattr(theme_registry, "get_theme_registry", lambda: _Registry())
    for fn in (nodes.classify_theme, nodes.decide_model, nodes.map_columns):
        with pytest.raises(_Stop):
            await fn(state)
    assert spy.stages == ["tabular"] * 3


class _Registry:
    def find_best_theme(self, theme):
        return None


@pytest.mark.parametrize(
    ("module_path", "func_name", "args", "stage"),
    [
        ("scinr.newton.annotation.prompts", "_m", (), "annotation"),
        ("scinr.newton.tabular.prompts", "_m", (), "tabular"),
        (
            "scinr.newton.entity_extraction.prompts",
            "build_extraction_system_prompt",
            ("Triple",),
            "entity_extraction",
        ),
        ("scinr.newton.prompts.system_prompt", "build_extraction_prompt", ("",), "extraction"),
    ],
)
def test_prompt_dispatchers_use_their_stage(monkeypatch, module_path, func_name, args, stage):
    import importlib

    module = importlib.import_module(module_path)
    seen: list[str | None] = []

    def _get_prompt_family(*, stage=None):
        seen.append(stage)
        raise _Stop

    monkeypatch.setattr("scinr.newton.config.get_prompt_family", _get_prompt_family)
    with pytest.raises(_Stop):
        getattr(module, func_name)(*args)
    assert seen == [stage]
