"""
graph/neo4j/_scope.py — The synchronous driver of a ``sync_scope()``.

One synchronous driver is shared, through a ``ContextVar``, by everything run
inside a scope: threads started with ``asyncio.to_thread`` copy the context and
see it, while other concurrent runs do not.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from neo4j import Driver

from scinr.newton.graph.neo4j import driver as _driver

_scope_driver: ContextVar[Driver | None] = ContextVar("scinr_graph_sync_driver", default=None)


@contextmanager
def open_scope() -> Iterator[None]:
    """Open a scope: create the synchronous driver and close it on exit.

    A nested scope reuses the outer driver and does not close it.

    Raises:
        GraphConfigurationError: If the credentials are not configured.
    """
    if _scope_driver.get() is not None:
        yield
        return
    driver = _driver.get_driver()
    token = _scope_driver.set(driver)
    try:
        yield
    finally:
        _scope_driver.reset(token)
        driver.close()


@contextmanager
def borrow_sync_driver() -> Iterator[Driver]:
    """Yield the driver of the current scope, or a private one closed on exit.

    The private driver reproduces today's stand-alone calls (create, use,
    close). Only the driver this call created is closed.
    """
    driver = _scope_driver.get()
    if driver is not None:
        yield driver
        return
    driver = _driver.get_driver()
    try:
        yield driver
    finally:
        driver.close()
