"""
sql/retry.py — Two-phase retry wrapper for transient SQL errors.

Mirror of ``graph/neo4j/retry.py``: the same parameters, phases and logs; only the test for
"is this error transient" changes. It asks the dialect
(``dialect.classify_error(exc) == "transient"``): serialization failures, deadlocks and lock
timeouts in Postgres; ``database is locked`` / ``busy`` in SQLite.

Public API
----------
    result = await with_sql_retry(lambda: do_something())
    result = with_sql_retry_sync(lambda: do_something())  # for worker threads

  Phase 1 — Exponential backoff with full jitter, capped at 60 s (7 attempts).
  Phase 2 — Fixed 60 s plateau, 5 additional attempts.

Non-transient exceptions are re-raised immediately without retrying. Translation to the
domain exceptions (``sql/errors.py``) goes **outside** this retry.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
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
    """Return True if *exc* is a transient SQL error worth retrying."""
    from scinr.newton.sql.engine import get_dialect_for_config

    try:
        dialect = get_dialect_for_config()
    except Exception:  # noqa: BLE001 — no usable SQL configuration: nothing to retry
        return False
    return dialect.classify_error(exc) == "transient"


# ── Shared retry-loop helpers ─────────────────────────────────────────────────


def _delay_for(attempt: int, exp_retries: int, base_delay: float, max_delay_exp: float) -> float:
    """Compute the wait time for a given retry attempt (exp backoff + jitter, then plateau)."""
    if attempt < exp_retries:
        return random.uniform(0, min(base_delay * (2**attempt), max_delay_exp))
    return max_delay_exp


def _check_retry_or_raise(exc: Exception, attempt: int, total_retries: int) -> None:
    """Raise *exc* immediately if it's non-transient, or if retries are exhausted.

    Otherwise returns normally, signalling the caller should sleep and retry.
    """
    if not _is_transient_error(exc):
        raise exc  # non-transient errors propagate immediately
    if attempt == total_retries:
        logger.error(
            "SQL transient error: all %d retry attempts exhausted. Re-raising last exception: %s",
            total_retries,
            _safe_message(exc),
        )
        raise exc


def _safe_message(exc: Exception) -> str:
    from scinr.newton.utils.redaction import redact_secrets

    return redact_secrets(str(getattr(exc, "orig", None) or exc)) or ""


def _log_retry_warning(attempt: int, total_retries: int, delay: float, exc: Exception) -> None:
    logger.warning(
        "SQL transient error (attempt %d/%d) — waiting %.1fs before retry. Error: %s",
        attempt + 1,
        total_retries,
        delay,
        _safe_message(exc),
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def with_sql_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    exp_retries: int = EXP_RETRIES,
    plateau_retries: int = PLATEAU_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay_exp: float = MAX_DELAY_EXP,
) -> T:
    """Await *coro_fn()* with two-phase retry on transient SQL errors.

    Args:
        coro_fn: A zero-argument callable that returns an awaitable. It is called
            fresh on every attempt so that a new coroutine object is created each time.
        exp_retries: Number of exponential-backoff retry attempts (phase 1).
        plateau_retries: Number of fixed-plateau retry attempts (phase 2).
        base_delay: Base wait time in seconds for the exponential phase.
        max_delay_exp: Cap on the exponential delay and the fixed plateau delay (seconds).

    Returns:
        Whatever *coro_fn()* returns on success.

    Raises:
        Exception: The last transient exception once all retries are exhausted, or any
            non-transient exception immediately.
    """
    total_retries = exp_retries + plateau_retries

    for attempt in range(total_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001
            _check_retry_or_raise(exc, attempt, total_retries)
            delay = _delay_for(attempt, exp_retries, base_delay, max_delay_exp)
            _log_retry_warning(attempt, total_retries, delay, exc)
            await asyncio.sleep(delay)

    # Unreachable — satisfies type checker
    raise RuntimeError("with_sql_retry: unexpected exit from retry loop")


def with_sql_retry_sync(
    fn: Callable[[], T],
    *,
    exp_retries: int = EXP_RETRIES,
    plateau_retries: int = PLATEAU_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay_exp: float = MAX_DELAY_EXP,
) -> T:
    """Synchronous mirror of :func:`with_sql_retry` (``time.sleep`` instead of ``asyncio.sleep``).

    Designed to run inside an ``asyncio.to_thread()`` worker thread (no event loop of its
    own); it must never await. Parameters and behaviour as in :func:`with_sql_retry`.
    """
    total_retries = exp_retries + plateau_retries

    for attempt in range(total_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            _check_retry_or_raise(exc, attempt, total_retries)
            delay = _delay_for(attempt, exp_retries, base_delay, max_delay_exp)
            _log_retry_warning(attempt, total_retries, delay, exc)
            time.sleep(delay)

    # Unreachable — satisfies type checker
    raise RuntimeError("with_sql_retry_sync: unexpected exit from retry loop")
