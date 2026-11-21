"""
tests/unit/test_graph_concurrency.py — graph_concurrency / graph_sync_concurrency
and the deprecated neo4j_* aliases (removed in 0.7.0).
"""

from __future__ import annotations

import pytest

from scinr.newton import config as config_module
from scinr.newton.config import (
    configure,
    get_config,
    get_graph_semaphore,
    get_graph_sync_semaphore,
    reset_graph_semaphore,
    reset_graph_sync_semaphore,
)
from scinr.newton.exceptions import ConfigurationError

_DUMMY_URI = "bolt://localhost:7687"


def _make_mock_llm():
    from langchain_core.language_models import BaseChatModel

    class _MockLLM(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "mock"

        def _generate(self, *args, **kwargs):
            pass

    return _MockLLM()


@pytest.fixture(autouse=True)
def _clean_env_and_semaphores(monkeypatch):
    for name in (
        "GRAPH_CONCURRENCY",
        "GRAPH_SYNC_CONCURRENCY",
        "NEO4J_CONCURRENCY",
        "NEO4J_SYNC_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_graph_semaphore()
    reset_graph_sync_semaphore()
    yield
    reset_graph_semaphore()
    reset_graph_sync_semaphore()


def _configure(**kwargs):
    return configure(
        llm=_make_mock_llm(),
        neo4j_uri=_DUMMY_URI,
        neo4j_user="neo4j",
        neo4j_password="test",
        **kwargs,
    )


class TestNewNames:
    def test_graph_concurrency_reaches_config_and_semaphore(self):
        _configure(graph_concurrency=3)
        assert get_config().graph_concurrency == 3
        assert get_graph_semaphore()._value == 3

    def test_graph_sync_concurrency_reaches_config_and_semaphore(self):
        _configure(graph_sync_concurrency=2)
        assert get_config().graph_sync_concurrency == 2
        assert get_graph_sync_semaphore()._value == 2

    def test_defaults(self):
        cfg = _configure()
        assert (cfg.graph_concurrency, cfg.graph_sync_concurrency) == (10, 8)

    def test_new_env_vars(self, monkeypatch):
        monkeypatch.setenv("GRAPH_CONCURRENCY", "4")
        monkeypatch.setenv("GRAPH_SYNC_CONCURRENCY", "6")
        cfg = _configure()
        assert (cfg.graph_concurrency, cfg.graph_sync_concurrency) == (4, 6)


class TestDeprecatedAliases:
    def test_neo4j_concurrency_argument(self):
        with pytest.warns(DeprecationWarning, match="neo4j_concurrency is deprecated"):
            _configure(neo4j_concurrency=3)
        assert get_config().graph_concurrency == 3
        assert get_graph_semaphore()._value == 3

    def test_neo4j_sync_concurrency_argument(self):
        with pytest.warns(DeprecationWarning, match="neo4j_sync_concurrency is deprecated"):
            _configure(neo4j_sync_concurrency=3)
        assert get_config().graph_sync_concurrency == 3
        assert get_graph_sync_semaphore()._value == 3

    def test_both_names_with_different_values_fail(self):
        with pytest.warns(DeprecationWarning), pytest.raises(ConfigurationError):
            _configure(graph_concurrency=3, neo4j_concurrency=4)
        with pytest.warns(DeprecationWarning), pytest.raises(ConfigurationError):
            _configure(graph_sync_concurrency=3, neo4j_sync_concurrency=4)

    def test_both_names_with_equal_values_pass(self):
        with pytest.warns(DeprecationWarning):
            cfg = _configure(graph_concurrency=3, neo4j_concurrency=3)
        assert cfg.graph_concurrency == 3

    def test_old_env_only_warns_and_applies(self, monkeypatch):
        monkeypatch.setenv("NEO4J_CONCURRENCY", "5")
        monkeypatch.setenv("NEO4J_SYNC_CONCURRENCY", "7")
        with pytest.warns(DeprecationWarning, match="NEO4J_CONCURRENCY is deprecated"):
            cfg = _configure()
        assert (cfg.graph_concurrency, cfg.graph_sync_concurrency) == (5, 7)

    def test_new_env_wins_over_old_env(self, monkeypatch, recwarn):
        monkeypatch.setenv("GRAPH_CONCURRENCY", "4")
        monkeypatch.setenv("NEO4J_CONCURRENCY", "5")
        cfg = _configure()
        assert cfg.graph_concurrency == 4
        assert not [w for w in recwarn if "deprecated, use" in str(w.message)]

    def test_config_properties_warn_and_return_new_value(self):
        cfg = _configure(graph_concurrency=3, graph_sync_concurrency=2)
        with pytest.warns(DeprecationWarning):
            assert cfg.neo4j_concurrency == 3
        with pytest.warns(DeprecationWarning):
            assert cfg.neo4j_sync_concurrency == 2

    def test_old_properties_are_not_dataclass_fields(self):
        cfg = _configure()
        assert "neo4j_concurrency" not in repr(cfg)
        assert "graph_concurrency=10" in repr(cfg)

    def test_old_semaphore_functions_share_the_new_semaphore(self):
        _configure(graph_concurrency=3, graph_sync_concurrency=2)
        with pytest.warns(DeprecationWarning):
            assert config_module.get_neo4j_semaphore() is get_graph_semaphore()
        with pytest.warns(DeprecationWarning):
            assert config_module.get_neo4j_sync_semaphore() is get_graph_sync_semaphore()
        with pytest.warns(DeprecationWarning):
            config_module.reset_neo4j_semaphore()
        with pytest.warns(DeprecationWarning):
            config_module.reset_neo4j_sync_semaphore()
        assert config_module._graph_semaphore is None
        assert config_module._graph_sync_semaphore is None

    def test_old_utils_module_still_importable(self):
        from scinr.newton.utils.neo4j_concurrency import (  # noqa: F401
            get_neo4j_semaphore,
            reset_neo4j_semaphore,
        )


class TestValidation:
    def test_zero_graph_concurrency_fails(self):
        with pytest.raises(ConfigurationError, match="graph_concurrency"):
            _configure(graph_concurrency=0)

    def test_zero_sync_env_fails(self, monkeypatch):
        monkeypatch.setenv("GRAPH_SYNC_CONCURRENCY", "0")
        with pytest.raises(ConfigurationError, match="graph_sync_concurrency"):
            _configure()

    def test_non_integer_env_fails(self, monkeypatch):
        monkeypatch.setenv("GRAPH_CONCURRENCY", "abc")
        with pytest.raises(ConfigurationError, match="GRAPH_CONCURRENCY"):
            _configure()
