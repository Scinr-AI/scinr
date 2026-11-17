"""
tests/golden/_recorder.py — Records every statement sent to Neo4j.

The hook is the ``neo4j`` package itself (``GraphDatabase.driver`` and
``AsyncGraphDatabase.driver``), never a function of ``scinr``: it keeps working when
``get_driver`` moves to another module. Each driver is wrapped in a transparent proxy
(``__getattr__`` forwards everything not recorded), so results, exceptions and
retries behave exactly as without the recorder.

Every entry belongs to a **unit of work**: an autocommit statement, an explicit
transaction (``begin`` ... ``commit`` / ``rollback``) or one invocation of the work
function of ``execute_write`` / ``execute_read`` (so a retry of the driver shows as
another unit). Entries::

    {"unit": 3, "kind": "tx", "op": "run", "query": "...", "params": {...}, "lane": "Task-7"}

``driver`` is ``sync`` or ``async``. ``op`` is ``run``, ``begin``, ``commit`` or ``rollback``; ``kind`` is ``autocommit``,
``tx``, ``execute_write`` or ``execute_read``. ``lane`` (asyncio task or thread that
emitted the entry) is a diagnostic aid and is not compared.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import neo4j

from ._dump import jsonable


def normalize_query(query: Any) -> str:
    return " ".join(str(query).split())


class Recorder:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.enabled = True
        self._lock = threading.Lock()
        self._units = 0

    def new_unit(self) -> int:
        with self._lock:
            self._units += 1
            return self._units

    def add(
        self,
        unit: int,
        driver: str,
        kind: str,
        op: str,
        query: str | None = None,
        params: Any = None,
    ) -> None:
        if not self.enabled:
            return
        entry: dict[str, Any] = {"unit": unit, "driver": driver, "kind": kind, "op": op}
        if query is not None:
            entry["query"] = normalize_query(query)
            entry["params"] = jsonable(params or {})
        entry["lane"] = _lane()
        with self._lock:
            self.entries.append(entry)


def _lane() -> str:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return task.get_name() if task is not None else threading.current_thread().name


def _merge_params(parameters: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {**(dict(parameters) if parameters else {}), **kwargs}


# ---------------------------------------------------------------------------
# Sync proxies
# ---------------------------------------------------------------------------


class _Proxy:
    def __init__(self, target: Any, rec: Recorder) -> None:
        self._target = target
        self._rec = rec

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class SyncTx(_Proxy):
    DRIVER = "sync"

    def __init__(self, target: Any, rec: Recorder, unit: int, kind: str) -> None:
        super().__init__(target, rec)
        self._unit = unit
        self._kind = kind

    def run(self, query: Any, parameters: Any = None, **kwargs: Any) -> Any:
        self._rec.add(self._unit, self.DRIVER, self._kind, "run", query, _merge_params(parameters, kwargs))
        return self._target.run(query, parameters, **kwargs)

    def commit(self) -> Any:
        self._rec.add(self._unit, self.DRIVER, self._kind, "commit")
        return self._target.commit()

    def rollback(self) -> Any:
        self._rec.add(self._unit, self.DRIVER, self._kind, "rollback")
        return self._target.rollback()

    def __enter__(self) -> SyncTx:
        self._target.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._target.__exit__(*exc)


class SyncSession(_Proxy):
    DRIVER = "sync"

    def run(self, query: Any, parameters: Any = None, **kwargs: Any) -> Any:
        unit = self._rec.new_unit()
        self._rec.add(unit, self.DRIVER, "autocommit", "run", query, _merge_params(parameters, kwargs))
        return self._target.run(query, parameters, **kwargs)

    def begin_transaction(self, *args: Any, **kwargs: Any) -> SyncTx:
        unit = self._rec.new_unit()
        self._rec.add(unit, self.DRIVER, "tx", "begin")
        return SyncTx(self._target.begin_transaction(*args, **kwargs), self._rec, unit, "tx")

    def _execute(self, kind: str, method: Any, work: Any, args: Any, kwargs: Any) -> Any:
        def wrapped(tx: Any, *a: Any, **kw: Any) -> Any:
            return work(SyncTx(tx, self._rec, self._rec.new_unit(), kind), *a, **kw)

        return method(wrapped, *args, **kwargs)

    def execute_write(self, work: Any, *args: Any, **kwargs: Any) -> Any:
        return self._execute("execute_write", self._target.execute_write, work, args, kwargs)

    def execute_read(self, work: Any, *args: Any, **kwargs: Any) -> Any:
        return self._execute("execute_read", self._target.execute_read, work, args, kwargs)

    def __enter__(self) -> SyncSession:
        self._target.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._target.__exit__(*exc)


class SyncDriver(_Proxy):
    def session(self, *args: Any, **kwargs: Any) -> SyncSession:
        return SyncSession(self._target.session(*args, **kwargs), self._rec)

    def __enter__(self) -> SyncDriver:
        self._target.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._target.__exit__(*exc)


# ---------------------------------------------------------------------------
# Async proxies
# ---------------------------------------------------------------------------


class AsyncTx(_Proxy):
    DRIVER = "async"

    def __init__(self, target: Any, rec: Recorder, unit: int, kind: str) -> None:
        super().__init__(target, rec)
        self._unit = unit
        self._kind = kind

    async def run(self, query: Any, parameters: Any = None, **kwargs: Any) -> Any:
        self._rec.add(self._unit, self.DRIVER, self._kind, "run", query, _merge_params(parameters, kwargs))
        return await self._target.run(query, parameters, **kwargs)

    async def commit(self) -> Any:
        self._rec.add(self._unit, self.DRIVER, self._kind, "commit")
        return await self._target.commit()

    async def rollback(self) -> Any:
        self._rec.add(self._unit, self.DRIVER, self._kind, "rollback")
        return await self._target.rollback()

    async def __aenter__(self) -> AsyncTx:
        await self._target.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._target.__aexit__(*exc)


class AsyncSession(_Proxy):
    DRIVER = "async"

    async def run(self, query: Any, parameters: Any = None, **kwargs: Any) -> Any:
        unit = self._rec.new_unit()
        self._rec.add(unit, self.DRIVER, "autocommit", "run", query, _merge_params(parameters, kwargs))
        return await self._target.run(query, parameters, **kwargs)

    async def begin_transaction(self, *args: Any, **kwargs: Any) -> AsyncTx:
        unit = self._rec.new_unit()
        self._rec.add(unit, self.DRIVER, "tx", "begin")
        tx = await self._target.begin_transaction(*args, **kwargs)
        return AsyncTx(tx, self._rec, unit, "tx")

    async def _execute(self, kind: str, method: Any, work: Any, args: Any, kwargs: Any) -> Any:
        async def wrapped(tx: Any, *a: Any, **kw: Any) -> Any:
            return await work(AsyncTx(tx, self._rec, self._rec.new_unit(), kind), *a, **kw)

        return await method(wrapped, *args, **kwargs)

    async def execute_write(self, work: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._execute(
            "execute_write", self._target.execute_write, work, args, kwargs
        )

    async def execute_read(self, work: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._execute("execute_read", self._target.execute_read, work, args, kwargs)

    async def __aenter__(self) -> AsyncSession:
        await self._target.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._target.__aexit__(*exc)


class AsyncDriver(_Proxy):
    def session(self, *args: Any, **kwargs: Any) -> AsyncSession:
        return AsyncSession(self._target.session(*args, **kwargs), self._rec)

    async def __aenter__(self) -> AsyncDriver:
        await self._target.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._target.__aexit__(*exc)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


@contextmanager
def recording() -> Iterator[Recorder]:
    """Patch ``neo4j.GraphDatabase.driver`` / ``AsyncGraphDatabase.driver`` while active.

    Drivers created inside the context are recorded; drivers created before (or by
    :func:`real_driver`) are not.
    """
    rec = Recorder()
    # The raw class attribute (a classmethod object), not the bound method, so that it can
    # be restored as it was.
    original_sync = neo4j.GraphDatabase.__dict__["driver"]
    original_async = neo4j.AsyncGraphDatabase.__dict__["driver"]
    real_sync = neo4j.GraphDatabase.driver
    real_async = neo4j.AsyncGraphDatabase.driver

    def sync_driver(*args: Any, **kwargs: Any) -> SyncDriver:
        return SyncDriver(real_sync(*args, **kwargs), rec)

    def async_driver(*args: Any, **kwargs: Any) -> AsyncDriver:
        return AsyncDriver(real_async(*args, **kwargs), rec)

    neo4j.GraphDatabase.driver = staticmethod(sync_driver)
    neo4j.AsyncGraphDatabase.driver = staticmethod(async_driver)
    try:
        yield rec
    finally:
        neo4j.GraphDatabase.driver = original_sync
        neo4j.AsyncGraphDatabase.driver = original_async
