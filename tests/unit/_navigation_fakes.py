"""
tests/unit/_navigation_fakes.py — Test doubles for the navigation layer.

``FakeGraphNavigator`` implements the full :class:`GraphNavigator` ABC with
trivial in-memory behaviour, proving the interface is satisfiable without Neo4j
and giving other tests a lightweight stand-in.

``FakeAsyncDriver`` / ``FakeAsyncSession`` emulate just enough of the
``neo4j.AsyncDriver`` surface that ``_Neo4jRuntime._read`` exercises, so the
Neo4j backend can be unit-tested with canned rows.
"""

from __future__ import annotations

from typing import Any

from scinr.newton.navigation.base import GraphNavigator
from scinr.newton.navigation.models import (
    CatalogGraph,
    GraphSummary,
    NodeSelector,
)


def make_fake_llm():
    """Minimal object that passes ``config._validate_llm``."""
    try:
        from langchain_core.language_models import BaseChatModel

        class _MockLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "mock"

            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                return None

        return _MockLLM()
    except Exception:  # pragma: no cover
        from unittest.mock import MagicMock

        return MagicMock()


#: kwargs that satisfy ``configure()`` in navigation tests.
NEO_CONFIG = {"neo4j_user": "neo4j", "neo4j_password": "pw"}


class FakeGraphNavigator(GraphNavigator):
    """Every abstract method implemented with an empty / trivial result."""

    dialect = "fake"

    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def ping(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - safety net
        raise AttributeError(name)


# Fill in every remaining abstract method with a stub whose empty return value
# matches the method's annotated return type (list -> [], int -> 0, else None).
def _empty_for(name: str) -> Any:
    import inspect

    try:
        ret = inspect.signature(getattr(GraphNavigator, name)).return_annotation
    except (ValueError, TypeError):
        return None
    text = str(ret)
    if text.startswith("list[") or text == "list":
        return []
    if text == "int" or "int" in text.split("|")[0]:
        return 0
    if text == "bool":
        return False
    return None


def _make_stub(name: str):
    _val = _empty_for(name)

    async def _stub(self, *args: Any, **kwargs: Any):  # noqa: ANN001, ARG001
        return [] if _val == [] else _val

    _stub.__name__ = name
    return _stub


for _name in list(GraphNavigator.__abstractmethods__):
    if _name in {"connect", "close", "ping"}:
        continue
    setattr(FakeGraphNavigator, _name, _make_stub(_name))

FakeGraphNavigator.__abstractmethods__ = frozenset()


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def __aiter__(self) -> FakeResult:
        self._it = iter(self._rows)
        return self

    async def __anext__(self) -> FakeRecord:
        try:
            return FakeRecord(next(self._it))
        except StopIteration:  # noqa: PERF203
            raise StopAsyncIteration from None

    async def single(self) -> FakeRecord | None:
        return FakeRecord(self._rows[0]) if self._rows else None

    async def data(self) -> list[dict[str, Any]]:
        return list(self._rows)


class FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def __getitem__(self, k: str) -> Any:
        return self._data[k]


class FakeTx:
    def __init__(self, responder) -> None:  # noqa: ANN001
        self._responder = responder
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, cypher: str, **params: Any) -> FakeResult:
        self.calls.append((cypher, params))
        rows = self._responder(cypher, params)
        return FakeResult(rows or [])


class FakeAsyncSession:
    def __init__(self, responder) -> None:  # noqa: ANN001
        self._responder = responder
        self.tx = FakeTx(responder)

    async def __aenter__(self) -> FakeAsyncSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute_read(self, fn) -> Any:  # noqa: ANN001
        return await fn(self.tx)

    async def run(self, cypher: str, **params: Any) -> FakeResult:
        return await self.tx.run(cypher, **params)


class FakeAsyncDriver:
    """Minimal ``neo4j.AsyncDriver`` stand-in.

    Args:
        responder: ``callable(cypher, params) -> list[dict]`` producing the rows
            for a given query. Default returns ``[]``.
    """

    def __init__(self, responder=None) -> None:  # noqa: ANN001
        self.responder = responder or (lambda cypher, params: [])
        self.sessions: list[FakeAsyncSession] = []
        self.closed = False

    def session(self, **_kwargs: Any) -> FakeAsyncSession:
        s = FakeAsyncSession(self.responder)
        self.sessions.append(s)
        return s

    async def close(self) -> None:
        self.closed = True

    @property
    def last_queries(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        for s in self.sessions:
            out.extend(s.tx.calls)
        return out


__all__ = [
    "FakeGraphNavigator",
    "FakeAsyncDriver",
    "FakeAsyncSession",
    "FakeResult",
    "FakeRecord",
    "CatalogGraph",
    "GraphSummary",
    "NodeSelector",
]
