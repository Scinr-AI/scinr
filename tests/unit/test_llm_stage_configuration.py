"""
tests/unit/test_llm_stage_configuration.py — Per-stage LLM configuration (WP-B1).

Imports are done from submodules directly to avoid triggering the package's
heavy import chain.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.language_models import BaseChatModel

from scinr.newton.config import (
    LLM_STAGES,
    PromptFamily,
    StageLLMConfig,
    configure,
    get_config,
    get_llm,
    get_prompt_family,
    get_stage_config,
    make_system_message,
)
from scinr.newton.exceptions import ConfigurationError

_DUMMY_URI = "bolt://localhost:7687"


class _FakeLLM(BaseChatModel):
    model_id: str = "fake"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError


def _configure(llm=None, **kwargs):
    return configure(
        llm=llm if llm is not None else _FakeLLM(model_id="global"),
        neo4j_uri=_DUMMY_URI,
        neo4j_user="neo4j",
        neo4j_password="test",
        **kwargs,
    )


def test_without_stage_configuration_everything_is_global():
    cfg = _configure()
    for stage in LLM_STAGES:
        resolved = get_stage_config(stage)
        assert resolved.llm is cfg.llm
        assert resolved.prompt_family == cfg.prompt_family
        assert resolved.prompt_caching_enabled == cfg.prompt_caching_enabled


def test_each_field_resolves_independently():
    stage_llm = _FakeLLM(model_id="stage")
    cfg = _configure(
        prompt_family="generic",
        prompt_caching_enabled=True,
        llm_stage_configuration={
            "extraction": StageLLMConfig(prompt_family="claude"),
            "annotation": StageLLMConfig(llm=stage_llm),
            "tabular": StageLLMConfig(prompt_caching_enabled=False),
        },
    )
    extraction = get_stage_config("extraction")
    assert extraction.llm is cfg.llm
    assert extraction.prompt_family == PromptFamily.CLAUDE
    assert extraction.prompt_caching_enabled is True

    annotation = get_stage_config("annotation")
    assert annotation.llm is stage_llm
    assert annotation.prompt_family == PromptFamily.GENERIC

    tabular = get_stage_config("tabular")
    assert tabular.llm is cfg.llm
    assert tabular.prompt_family == PromptFamily.GENERIC
    assert tabular.prompt_caching_enabled is False


def test_unknown_stage_is_rejected():
    with pytest.raises(ConfigurationError, match="extraction"):
        _configure(llm_stage_configuration={"consolidation": StageLLMConfig()})


def test_non_stage_config_value_is_rejected():
    with pytest.raises(ConfigurationError):
        _configure(llm_stage_configuration={"extraction": {"llm": _FakeLLM()}})


def test_invalid_prompt_family_is_rejected():
    with pytest.raises(ConfigurationError):
        _configure(llm_stage_configuration={"extraction": StageLLMConfig(prompt_family="nope")})


def test_prompt_family_string_is_normalized():
    _configure(llm_stage_configuration={"extraction": StageLLMConfig(prompt_family="claude")})
    stored = get_config().llm_stage_configuration["extraction"]
    assert stored.prompt_family is PromptFamily.CLAUDE


def test_non_bool_caching_is_rejected():
    with pytest.raises(ConfigurationError):
        _configure(
            llm_stage_configuration={"extraction": StageLLMConfig(prompt_caching_enabled="yes")}
        )


def test_get_llm_with_stage():
    stage_llm = _FakeLLM(model_id="stage")
    cfg = _configure(llm_stage_configuration={"extraction": StageLLMConfig(llm=stage_llm)})
    assert get_llm(stage="extraction") is stage_llm
    assert get_llm(stage="annotation") is cfg.llm


def test_get_llm_stage_without_any_llm_names_the_stage():
    configure(neo4j_uri=_DUMMY_URI, neo4j_user="neo4j", neo4j_password="test")
    assert get_config().llm is None
    with pytest.raises(ConfigurationError, match="Stage 'annotation'"):
        get_llm(stage="annotation")


def test_get_llm_without_stage_is_unchanged():
    cfg = _configure(llm_stage_configuration={"extraction": StageLLMConfig(llm=_FakeLLM())})
    assert get_llm() is cfg.llm
    assert get_prompt_family() == cfg.prompt_family


def test_get_stage_config_rejects_unknown_stage():
    _configure()
    with pytest.raises(ValueError):
        get_stage_config("consolidation")


def _has_cache_point(message) -> bool:
    return isinstance(message.content, list) and any(
        isinstance(block, dict) and "cachePoint" in block for block in message.content
    )


def test_make_system_message_cache_point_matrix(monkeypatch):
    import langchain_aws

    class _FakeBedrock(_FakeLLM):
        pass

    monkeypatch.setattr(langchain_aws, "ChatBedrockConverse", _FakeBedrock)

    for stage_is_bedrock in (True, False):
        for stage_caching in (True, False):
            stage_llm = _FakeBedrock() if stage_is_bedrock else _FakeLLM()
            _configure(
                llm=_FakeLLM(model_id="global"),
                llm_stage_configuration={
                    "extraction": StageLLMConfig(
                        llm=stage_llm, prompt_caching_enabled=stage_caching
                    )
                },
            )
            message = make_system_message("prompt", stage="extraction")
            assert _has_cache_point(message) is (stage_is_bedrock and stage_caching)
            # Without a stage the global LLM (not Bedrock) decides
            assert not _has_cache_point(make_system_message("prompt"))

    # Global Bedrock LLM + stage without own LLM
    _configure(llm=_FakeBedrock(), prompt_caching_enabled=True)
    assert _has_cache_point(make_system_message("prompt"))
    assert _has_cache_point(make_system_message("prompt", stage="tabular"))


def test_stage_config_is_logged_without_secrets(caplog):
    with caplog.at_level(logging.INFO, logger="scinr.newton.config"):
        _configure(
            llm_stage_configuration={
                "extraction": StageLLMConfig(llm=_FakeLLM(model_id="stage-model"))
            }
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("LLM stage configuration" in m and "stage-model" in m for m in messages)
