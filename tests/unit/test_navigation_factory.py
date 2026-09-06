"""Unit tests for scinr.newton.navigation.factory."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import make_fake_llm  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.exceptions import ConfigurationError  # noqa: E402
from scinr.newton.navigation.factory import get_graph_navigator, graph_navigator  # noqa: E402
from scinr.newton.navigation.neo4j.navigator import Neo4jGraphNavigator  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}


@pytest.fixture(autouse=True)
def _no_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Neo4jGraphNavigator, "connect", AsyncMock())
    monkeypatch.setattr(Neo4jGraphNavigator, "close", AsyncMock())


async def test_factory_returns_neo4j_by_default() -> None:
    configure(**_NEO)
    nav = await get_graph_navigator()
    assert isinstance(nav, Neo4jGraphNavigator)
    Neo4jGraphNavigator.connect.assert_awaited()  # type: ignore[attr-defined]


async def test_factory_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configure(**_NEO)
    monkeypatch.setattr(cfg, "graph_backend", "titan", raising=False)
    with pytest.raises(ConfigurationError):
        await get_graph_navigator()


async def test_context_manager_closes() -> None:
    configure(**_NEO)
    async with graph_navigator() as nav:
        assert isinstance(nav, Neo4jGraphNavigator)
    Neo4jGraphNavigator.close.assert_awaited()  # type: ignore[attr-defined]
