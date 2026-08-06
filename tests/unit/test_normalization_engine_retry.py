"""
tests/unit/test_normalization_engine_retry.py — Unit tests for the
Bedrock connection-pool-exhaustion fix in
scinr.newton.tabular.normalization.engine.NormalizationEngine:

1. LLM calls in process_key_batch() / _call_llm_batch() now go through
   with_llm_retry(), so a transient retryable error (e.g. Bedrock
   throttling) is retried with backoff instead of failing the batch
   outright.
2. If retries are exhausted, the batch is still discarded gracefully
   (the existing `except Exception` around the call keeps catching the
   re-raised exception) — no exception propagates to the caller.
3. normalize_instances() no longer creates its own local
   asyncio.Semaphore(self.concurrency); it delegates to the global
   scinr.newton.config.get_llm_semaphore() instead.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from scinr.newton.tabular.normalization.engine import NormalizationEngine
from scinr.newton.tabular.normalization.models import NormalizationEntry

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _DummyTarget(BaseModel):
    """Minimal normalized target model."""

    value: str | None = None


class _DummySourceModel(BaseModel):
    """Model with one field marked for normalization, used by
    normalize_instances() end-to-end tests."""

    raw: str = "abc"
    normalized_field: _DummyTarget | None = Field(
        default=None,
        json_schema_extra={"normalization_model": True},
    )


def _make_entry(key: str = "k1") -> NormalizationEntry:
    return NormalizationEntry(
        instance_id=1,
        model_class_name="_DummySourceModel",
        field_name="normalized_field",
        target_type=_DummyTarget,
        source_values={"raw": "abc"},
        unique_key=key,
    )


@pytest.fixture
def engine_with_mock_llm():
    """A NormalizationEngine whose llm.with_structured_output() returns a
    mock structured_llm with a controllable async ainvoke()."""
    llm = MagicMock()
    structured_llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)
    engine = NormalizationEngine(llm=llm, batch_size=5)
    return engine, structured_llm


_RETRYABLE_EXC = Exception("ThrottlingException: Rate exceeded")


# ---------------------------------------------------------------------------
# (a) with_llm_retry retries on transient errors then succeeds
# ---------------------------------------------------------------------------


async def test_process_key_batch_retries_on_throttling_then_succeeds(
    engine_with_mock_llm, monkeypatch
):
    """process_key_batch() must use with_llm_retry(): a retryable error on
    the first 2 attempts must not lose the batch — the 3rd attempt succeeds
    and the final result reflects it."""
    engine, structured_llm = engine_with_mock_llm
    entry = _make_entry("k1")

    call_count = {"n": 0}

    async def _side_effect(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise _RETRYABLE_EXC
        return {"results": [{"key": "k1", "result": {"value": "normalized"}}]}

    structured_llm.ainvoke = AsyncMock(side_effect=_side_effect)

    # Avoid real exponential backoff delays (base delay 5s, would make the
    # test take ~5-10s otherwise).
    mock_sleep = AsyncMock()
    monkeypatch.setattr("scinr.newton.utils.llm_retry.asyncio.sleep", mock_sleep)

    results = await engine.process_key_batch([entry])

    assert call_count["n"] == 3, "expected 2 failed attempts + 1 successful attempt"
    assert mock_sleep.await_count == 2, "expected a backoff sleep between each retry"
    assert "k1" in results
    assert isinstance(results["k1"], _DummyTarget)
    assert results["k1"].value == "normalized"


async def test_call_llm_batch_retries_on_throttling_then_applies_result(
    engine_with_mock_llm, monkeypatch
):
    """_call_llm_batch() (the method actually used by normalize_instances())
    must also go through with_llm_retry() and, once it succeeds, apply the
    normalized result to the target instance — proving the batch was not
    silently dropped because of the earlier transient failures."""
    engine, structured_llm = engine_with_mock_llm
    entry = _make_entry("k1")

    instance = _DummySourceModel(raw="abc")
    all_instances = [(_DummySourceModel, instance)]
    key_to_targets = {"k1": [(id(instance), "normalized_field")]}

    call_count = {"n": 0}

    async def _side_effect(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise _RETRYABLE_EXC
        return {"results": [{"key": "k1", "result": {"value": "normalized"}}]}

    structured_llm.ainvoke = AsyncMock(side_effect=_side_effect)

    mock_sleep = AsyncMock()
    monkeypatch.setattr("scinr.newton.utils.llm_retry.asyncio.sleep", mock_sleep)

    await engine._call_llm_batch(entries=[entry], all_instances=all_instances, key_to_targets=key_to_targets)

    assert call_count["n"] == 3
    assert instance.normalized_field is not None
    assert instance.normalized_field.value == "normalized"


# ---------------------------------------------------------------------------
# (b) exhausted retries -> batch discarded gracefully, no propagation
# ---------------------------------------------------------------------------


async def test_process_key_batch_discards_batch_when_retries_exhausted(
    engine_with_mock_llm, monkeypatch, caplog
):
    """If a retryable error persists across all with_llm_retry attempts, the
    exception must NOT propagate out of process_key_batch() — the existing
    `except Exception` around the ainvoke() call must still catch it (now
    catching the re-raised exception from with_llm_retry instead of a direct
    ainvoke failure), the batch is dropped (empty result), and a warning is
    logged — same graceful-degradation contract as before this change."""
    engine, structured_llm = engine_with_mock_llm
    entry = _make_entry("k1")

    structured_llm.ainvoke = AsyncMock(side_effect=_RETRYABLE_EXC)

    mock_sleep = AsyncMock()
    monkeypatch.setattr("scinr.newton.utils.llm_retry.asyncio.sleep", mock_sleep)

    with caplog.at_level(logging.WARNING):
        results = await engine.process_key_batch([entry])  # must not raise

    assert results == {}
    assert structured_llm.ainvoke.await_count == 7  # max_retries=6 -> 7 attempts
    assert any(
        "Normalization batch failed" in record.message for record in caplog.records
    )


async def test_call_llm_batch_discards_batch_when_retries_exhausted(
    engine_with_mock_llm, monkeypatch, caplog
):
    """Same graceful-degradation guarantee for _call_llm_batch(): the target
    instance is left untouched (no partial/garbage write) and nothing is
    raised to the caller."""
    engine, structured_llm = engine_with_mock_llm
    entry = _make_entry("k1")

    instance = _DummySourceModel(raw="abc")
    all_instances = [(_DummySourceModel, instance)]
    key_to_targets = {"k1": [(id(instance), "normalized_field")]}

    structured_llm.ainvoke = AsyncMock(side_effect=_RETRYABLE_EXC)

    mock_sleep = AsyncMock()
    monkeypatch.setattr("scinr.newton.utils.llm_retry.asyncio.sleep", mock_sleep)

    with caplog.at_level(logging.WARNING):
        # Must not raise despite every retry failing.
        await engine._call_llm_batch(
            entries=[entry], all_instances=all_instances, key_to_targets=key_to_targets
        )

    assert instance.normalized_field is None
    assert any(
        "Normalization batch failed" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# (c) normalize_instances() delegates to the global get_llm_semaphore()
# ---------------------------------------------------------------------------


class _FakeSemaphore:
    """Minimal async-context-manager stand-in for asyncio.Semaphore, so we
    can spy on how many times it's entered without any real concurrency
    control."""

    def __init__(self) -> None:
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def test_normalize_instances_uses_global_semaphore_not_local(monkeypatch):
    """normalize_instances() must call scinr.newton.config.get_llm_semaphore()
    (the shared, global semaphore) instead of building its own
    asyncio.Semaphore(self.concurrency) — even when a NormalizationEngine is
    constructed with a large explicit `concurrency` value, that value must
    no longer govern real LLM concurrency."""
    import scinr.newton.config as config_module

    fake_sem = _FakeSemaphore()
    get_sem_mock = MagicMock(return_value=fake_sem)
    monkeypatch.setattr(config_module, "get_llm_semaphore", get_sem_mock)

    # Spy on the local asyncio.Semaphore constructor as seen from inside the
    # engine module — it must never be invoked anymore.
    semaphore_ctor_mock = MagicMock(wraps=asyncio.Semaphore)
    monkeypatch.setattr(
        "scinr.newton.tabular.normalization.engine.asyncio.Semaphore",
        semaphore_ctor_mock,
    )

    llm = MagicMock()
    structured_llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)
    structured_llm.ainvoke = AsyncMock(
        return_value={"results": [{"key": _make_entry().unique_key, "result": {"value": "x"}}]}
    )

    # Note the deliberately huge concurrency=100 — it must have zero effect.
    engine = NormalizationEngine(llm=llm, concurrency=100)
    assert engine.concurrency == 100  # stored for API compat, but unused

    instance = _DummySourceModel(raw="abc")
    instances = [(_DummySourceModel, instance)]

    result = await engine.normalize_instances(instances)

    assert result is instances
    get_sem_mock.assert_called()
    assert fake_sem.enter_count >= 1, "expected the global semaphore to be entered"
    semaphore_ctor_mock.assert_not_called()
