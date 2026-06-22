"""
utils/neo4j_retry.py — Two-phase retry wrapper for transient Neo4j errors.

Neo4j can raise transient errors (e.g. DeadlockDetected, LockClientStopped,
ServiceUnavailable / defunct connection) under write contention or parallel
read overload.  Without a retry mechanism, concurrent document ingestion would
fail immediately on the first lock conflict or socket reset.

Public API
----------
    result = await with_neo4j_retry(lambda: session.run(...))

The wrapper uses a two-phase retry strategy:

  Phase 1 — Exponential backoff with full jitter, capped at 60 s (7 attempts).
  Phase 2 — Fixed 60 s plateau, 5 additional attempts.

Total: 12 retry attempts maximum (~6.8 min worst-case wait).
Non-transient exceptions are re-raised immediately without retrying.
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────

EXP_RETRIES: int = 7
"""Number of exponential-backoff attempts (attempts 0–6, delays 1 s → 60 s)."""

PLATEAU_RETRIES: int = 5
"""Number of fixed-plateau attempts after the exponential phase (60 s each)."""

TOTAL_RETRIES: int = EXP_RETRIES + PLATEAU_RETRIES
"""Maximum total retry attempts after the initial failure (12)."""

BASE_DELAY: float = 1.0
"""Initial wait in seconds; doubles each exponential attempt."""

MAX_DELAY_EXP: float = 60.0
"""Upper cap on the exponential back-off delay and fixed plateau delay (seconds)."""

T = TypeVar("T")


# ── Transient-error detection ─────────────────────────────────────────────────


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* represents a transient Neo4j error worth retrying.

    Covers:
    - ``Neo.TransientError`` (DeadlockDetected, LockClientStopped, etc.)
    - ``DeadlockDetected`` (re-wrapped by some driver versions)
    - ``ServiceUnavailable`` (defunct connection, connection reset, socket errors)
      which manifests as "Failed to read/write from/to defunct connection"
    """
    import neo4j.exceptions as _neo4j_exc

    if isinstance(exc, _neo4j_exc.ServiceUnavailable):
        return True
    msg = str(exc)
    return (
        "Neo.TransientError" in msg
        or "DeadlockDetected" in msg
        or "defunct connection" in msg
        or "Failed to read from" in msg
        or "Failed to write to" in msg
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def with_neo4j_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    exp_retries: int = EXP_RETRIES,
    plateau_retries: int = PLATEAU_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay_exp: float = MAX_DELAY_EXP,
) -> T:
    """Await *coro_fn()* with two-phase retry on transient Neo4j errors.

    Parameters
    ----------
    coro_fn:
        A zero-argument callable that returns an awaitable (e.g.
        ``lambda: session.run(...)``).  It is called fresh on every attempt
        so that a new coroutine object is created each time.
    exp_retries:
        Number of exponential-backoff retry attempts (Phase 1).
    plateau_retries:
        Number of fixed-plateau retry attempts after the exponential phase
        (Phase 2).
    base_delay:
        Base wait time in seconds for the exponential phase.
    max_delay_exp:
        Cap on the exponential delay and the fixed plateau delay (seconds).

    Returns
    -------
    T
        Whatever *coro_fn()* returns on success.

    Raises
    ------
    Exception
        Re-raises the last transient exception once all retries are exhausted,
        or immediately re-raises any non-transient exception.
    """
    total_retries = exp_retries + plateau_retries

    for attempt in range(total_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_transient_error(exc):
                raise  # non-transient errors propagate immediately

            if attempt == total_retries:
                logger.error(
                    "Neo4j transient error: all %d retry attempts exhausted. "
                    "Re-raising last exception: %s",
                    total_retries,
                    exc,
                )
                raise

            # Phase 1: full-jitter exponential back-off
            if attempt < exp_retries:
                delay = random.uniform(0, min(base_delay * (2**attempt), max_delay_exp))
            # Phase 2: fixed plateau (contention resolved, waiting for lock release)
            else:
                delay = max_delay_exp

            logger.warning(
                "Neo4j transient error (attempt %d/%d) — waiting %.1fs before retry. Error: %s",
                attempt + 1,
                total_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    # Unreachable — satisfies type checker
    raise RuntimeError("with_neo4j_retry: unexpected exit from retry loop")
