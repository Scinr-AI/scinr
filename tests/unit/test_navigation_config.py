"""Unit tests for the graph_backend configuration field."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import make_fake_llm  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.exceptions import ConfigurationError  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}


def test_graph_backend_defaults_to_neo4j() -> None:
    cfg = configure(**_NEO)
    assert cfg.graph_backend == "neo4j"


def test_graph_backend_explicit() -> None:
    cfg = configure(graph_backend="neo4j", **_NEO)
    assert cfg.graph_backend == "neo4j"


def test_graph_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "neo4j")
    cfg = configure(**_NEO)
    assert cfg.graph_backend == "neo4j"


def test_graph_backend_invalid_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "titan")
    with pytest.raises(ConfigurationError):
        configure(**_NEO)


def test_graph_backend_invalid_explicit() -> None:
    with pytest.raises(ConfigurationError):
        configure(graph_backend="dgraph", **_NEO)  # type: ignore[arg-type]
