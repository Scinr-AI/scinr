"""
tests/unit/test_progress.py — Unit tests for scinr.newton.progress, without the
pipeline: the ProgressEvent contract, the callback base classes, the internal
dispatcher (_ProgressDispatcher) and the built-in ProgressTracker.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import get_args

import pytest

from scinr.newton.progress import (
    AsyncProgressCallback,
    ProgressCallback,
    ProgressEvent,
    ProgressEventKind,
    ProgressTracker,
    _ProgressDispatcher,
    _validate_callbacks,
)


def _event(kind: str, **fields) -> ProgressEvent:
    return ProgressEvent(kind=kind, timestamp=datetime.now(UTC), **fields)


class _RecordingSync(ProgressCallback):
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.threads: list[str] = []

    def on_event(self, event: ProgressEvent) -> None:
        self.threads.append(threading.current_thread().name)
        self.events.append(event)


class _RecordingAsync(AsyncProgressCallback):
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    async def on_event(self, event: ProgressEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# ProgressEvent
# ---------------------------------------------------------------------------


class TestProgressEvent:
    def test_fields_cannot_be_reassigned(self):
        event = _event("pipeline_started")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.kind = "pipeline_failed"  # type: ignore[misc]

    def test_metrics_is_read_only_and_detached_from_the_input(self):
        source = {"nodes_processed": 1, "nodes_failed": 0}
        event = _event("stage_completed", stage="annotation", metrics=source)
        with pytest.raises(TypeError):
            event.metrics["nodes_failed"] = 5  # type: ignore[index]
        source["nodes_failed"] = 9
        assert event.metrics["nodes_failed"] == 0

    def test_to_dict_is_json_serializable(self):
        event = _event(
            "stage_completed",
            job_id="job-1",
            document_index=0,
            stage="extraction",
            stages=("extraction", "ingestion"),
            planned_stages=["extraction"],
            metrics={"nodes_processed": 1, "nodes_failed": 0},
            duration_s=0.5,
        )
        data = event.to_dict()
        roundtrip = json.loads(json.dumps(data))
        assert roundtrip["kind"] == "stage_completed"
        assert roundtrip["metrics"] == {"nodes_processed": 1, "nodes_failed": 0}
        assert roundtrip["stages"] == ["extraction", "ingestion"]
        assert roundtrip["planned_stages"] == ["extraction"]
        assert datetime.fromisoformat(roundtrip["timestamp"]) == event.timestamp

    def test_error_is_redacted(self):
        event = _event("pipeline_failed", error="cannot reach mongodb://app:s3cr3tpw@db:27017")
        assert "s3cr3tpw" not in event.error
        assert "mongodb://app:***@db:27017" in event.error

    def test_timestamp_is_timezone_aware(self):
        assert _event("pipeline_started").timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Callback base classes
# ---------------------------------------------------------------------------


class TestCallbackBaseClasses:
    def test_sync_on_event_dispatches_to_on_kind(self):
        seen: list[str] = []

        class Cb(ProgressCallback):
            def on_stage_started(self, event):
                seen.append(event.stage)

        cb = Cb()
        cb.on_event(_event("stage_started", stage="ingestion"))
        cb.on_event(_event("stage_completed", stage="ingestion"))  # default no-op
        assert seen == ["ingestion"]

    def test_sync_default_methods_do_nothing(self):
        cb = ProgressCallback()
        for kind in get_args(ProgressEventKind):
            assert cb.on_event(_event(kind)) is None

    async def test_async_on_event_dispatches_to_on_kind(self):
        seen: list[str] = []

        class Cb(AsyncProgressCallback):
            async def on_document_finished(self, event):
                seen.append(event.status)

        cb = Cb()
        await cb.on_event(_event("document_finished", status="partial"))
        await cb.on_event(_event("pipeline_completed", success=True))  # default no-op
        assert seen == ["partial"]

    async def test_async_default_methods_do_nothing(self):
        cb = AsyncProgressCallback()
        assert await cb.on_event(_event("pipeline_failed", error="x")) is None


# ---------------------------------------------------------------------------
# _validate_callbacks
# ---------------------------------------------------------------------------


class TestValidateCallbacks:
    def test_none_and_empty_return_empty_list(self):
        assert _validate_callbacks(None) == []
        assert _validate_callbacks([]) == []

    def test_accepts_the_three_types(self):
        cbs = [ProgressTracker(), ProgressCallback(), AsyncProgressCallback()]
        assert _validate_callbacks(tuple(cbs)) == cbs

    @pytest.mark.parametrize("bad", [lambda e: None, object(), "tracker"])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(TypeError, match="ProgressCallback.*AsyncProgressCallback.*ProgressTracker"):
            _validate_callbacks([bad])

    def test_rejects_a_bare_callable_instead_of_a_sequence(self):
        with pytest.raises(TypeError):
            _validate_callbacks(lambda e: None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _ProgressDispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    async def test_fifo_delivery_to_mixed_sync_and_async_callbacks(self):
        sync_cb, async_cb = _RecordingSync(), _RecordingAsync()
        d = _ProgressDispatcher([sync_cb, async_cb], job_id="j", tenant_id="t")
        for i in range(20):
            d.emit("document_started", document_index=i)
        await d.aclose()
        assert [e.document_index for e in sync_cb.events] == list(range(20))
        assert [e.document_index for e in async_cb.events] == list(range(20))
        assert all(e.job_id == "j" and e.tenant_id == "t" for e in async_cb.events)

    async def test_explicit_fields_override_dispatcher_defaults(self):
        cb = _RecordingAsync()
        d = _ProgressDispatcher([cb], job_id="j", tenant_id="t")
        d.emit("document_started", tenant_id=None)
        await d.aclose()
        assert cb.events[0].tenant_id is None
        assert cb.events[0].job_id == "j"

    async def test_sync_callbacks_run_in_dedicated_thread(self):
        cb = _RecordingSync()
        d = _ProgressDispatcher([cb], job_id=None, tenant_id=None)
        d.emit("pipeline_started")
        await d.aclose()
        loop_thread = threading.current_thread().name
        assert cb.threads and all(t.startswith("scinr-progress") for t in cb.threads)
        assert loop_thread not in cb.threads

    async def test_slow_sync_callback_never_delays_emit(self):
        class Slow(_RecordingSync):
            def on_event(self, event):
                time.sleep(0.2)
                super().on_event(event)

        cb = Slow()
        d = _ProgressDispatcher([cb], job_id=None, tenant_id=None)
        t0 = time.monotonic()
        for i in range(5):
            d.emit("document_started", document_index=i)
        assert time.monotonic() - t0 < 0.1  # ≪ 5 × 0.2 s
        await d.aclose()
        assert [e.document_index for e in cb.events] == list(range(5))

    async def test_failing_callback_is_logged_and_others_still_receive(self, caplog):
        class Boom(AsyncProgressCallback):
            async def on_event(self, event):
                raise RuntimeError("boom")

        class BoomSync(ProgressCallback):
            def on_event(self, event):
                raise RuntimeError("boom-sync")

        good = _RecordingAsync()
        d = _ProgressDispatcher([Boom(), BoomSync(), good], job_id=None, tenant_id=None)
        with caplog.at_level(logging.ERROR, logger="scinr.newton.progress"):
            d.emit("pipeline_started")
            d.emit("pipeline_completed", success=True)
            await d.aclose()
        assert [e.kind for e in good.events] == ["pipeline_started", "pipeline_completed"]
        failures = [r for r in caplog.records if "failed on" in r.getMessage()]
        assert len(failures) == 4

    async def test_aclose_times_out_on_hung_callback(self, caplog):
        never = asyncio.Event()

        class Hung(AsyncProgressCallback):
            async def on_event(self, event):
                await never.wait()

        d = _ProgressDispatcher([Hung()], job_id=None, tenant_id=None)
        d.emit("pipeline_started")
        d.emit("pipeline_completed", success=True)
        t0 = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="scinr.newton.progress"):
            await d.aclose(timeout=0.2)
        assert time.monotonic() - t0 < 2.0
        assert any("did not finish" in r.getMessage() for r in caplog.records)
        assert d._task.done()

    async def test_no_executor_without_sync_callbacks(self):
        d = _ProgressDispatcher([_RecordingAsync()], job_id=None, tenant_id=None)
        assert d._executor is None
        await d.aclose()

    async def test_tracker_only_creates_no_queue_nor_task(self):
        tracker = ProgressTracker()
        d = _ProgressDispatcher([tracker], job_id=None, tenant_id=None)
        assert d._queue is None and d._task is None and d._executor is None
        d.emit("pipeline_started")
        assert tracker.snapshot().status == "running"
        await d.aclose()

    async def test_tracker_is_updated_inline_before_queue_delivery(self):
        tracker = ProgressTracker()
        d = _ProgressDispatcher([tracker, _RecordingAsync()], job_id=None, tenant_id=None)
        d.emit("pipeline_started")
        # No await yet: the consumer task has not run, but the tracker already knows.
        assert tracker.snapshot().status == "running"
        await d.aclose()

    async def test_aclose_is_idempotent_and_later_emits_are_ignored(self):
        cb = _RecordingSync()
        d = _ProgressDispatcher([cb], job_id=None, tenant_id=None)
        d.emit("pipeline_started")
        await d.aclose()
        await d.aclose()
        d.emit("pipeline_completed", success=True)
        assert [e.kind for e in cb.events] == ["pipeline_started"]


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


def _feed(tracker: ProgressTracker, *events: ProgressEvent) -> None:
    for e in events:
        tracker._apply(e)


def _doc_events(index: int, stages: tuple[str, ...], *, finish: str | None = None):
    events = [_event("document_started", document_index=index, planned_stages=stages)]
    for stage in stages:
        events.append(_event("stage_started", document_index=index, stage=stage))
        events.append(_event("stage_completed", document_index=index, stage=stage))
    if finish is not None:
        events.append(_event("document_finished", document_index=index, status=finish))
    return events


class TestProgressTracker:
    def test_status_lifecycle_success(self):
        t = ProgressTracker()
        assert t.snapshot().status == "pending"
        _feed(t, _event("pipeline_started", job_id="job-1"))
        snap = t.snapshot()
        assert snap.status == "running" and snap.job_id == "job-1"
        assert snap.started_at is not None and snap.finished_at is None
        _feed(t, _event("pipeline_completed", success=True))
        snap = t.snapshot()
        assert snap.status == "success" and snap.finished_at is not None

    def test_status_completed_with_errors_and_failed(self):
        t = ProgressTracker()
        _feed(t, _event("pipeline_started"), _event("pipeline_completed", success=False))
        assert t.snapshot().status == "completed_with_errors"

        t = ProgressTracker()
        _feed(t, _event("pipeline_started"), _event("pipeline_failed", error="bad stage"))
        snap = t.snapshot()
        assert snap.status == "failed" and snap.error == "bad stage"

    def test_counters_and_per_stage_maps(self):
        stages = ("extraction", "ingestion", "annotation")
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("documents_discovered", total_documents=6, stages=stages),
            *_doc_events(0, stages, finish="success"),
            *_doc_events(1, stages, finish="partial"),
            _event("document_started", document_index=2, planned_stages=stages),
            _event("stage_started", document_index=2, stage="extraction"),
            _event("stage_failed", document_index=2, stage="extraction", error="x"),
            _event(
                "document_finished",
                document_index=2,
                status="failed",
                stopped_at="extraction",
                error="x",
            ),
            _event("document_started", document_index=3, planned_stages=("preprocess",)),
            _event("stage_started", document_index=3, stage="preprocess"),
            _event("stage_completed", document_index=3, stage="preprocess"),
            _event("document_finished", document_index=3, status="skipped"),
            _event("document_started", document_index=4, planned_stages=stages),
            _event("stage_started", document_index=4, stage="extraction"),
            _event("stage_completed", document_index=4, stage="extraction"),
            _event("stage_started", document_index=4, stage="ingestion"),
        )
        snap = t.snapshot()
        assert snap.total_documents == 6
        assert snap.stages == stages
        assert snap.documents_pending == 1
        assert snap.documents_running == 1
        assert snap.documents_succeeded == 1
        assert snap.documents_partial == 1
        assert snap.documents_failed == 1
        assert snap.documents_skipped == 1
        assert dict(snap.running_by_stage) == {"ingestion": 1}
        assert dict(snap.completed_by_stage) == {
            "extraction": 3,
            "ingestion": 2,
            "annotation": 2,
            "preprocess": 1,
        }
        assert dict(snap.failed_by_stage) == {"extraction": 1}
        doc2 = snap.documents[2]
        assert doc2.status == "failed" and doc2.stopped_at == "extraction" and doc2.error == "x"
        doc4 = snap.documents[4]
        assert doc4.current_stage == "ingestion"
        assert doc4.completed_stages == ("extraction",)
        assert doc4.finished_at is None
        # 4 finished docs + 1/3 of doc 4, over 6 documents.
        assert snap.percent == round(100 * (4 + 1 / 3) / 6, 1)

    def test_document_name_follows_the_events(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("document_started", document_index=0, document_name="hint"),
            _event("stage_completed", document_index=0, stage="extraction", document_name="Real"),
        )
        assert t.snapshot().documents[0].document_name == "Real"

    def test_percent_without_discovery(self):
        t = ProgressTracker()
        _feed(t, _event("pipeline_started"))
        assert t.snapshot().percent == 0.0
        _feed(t, _event("pipeline_failed", error="boom"))
        assert t.snapshot().percent == 100.0

    def test_percent_with_zero_documents(self):
        t = ProgressTracker()
        _feed(t, _event("pipeline_started"), _event("documents_discovered", total_documents=0))
        assert t.snapshot().percent == 0.0
        _feed(t, _event("pipeline_completed", success=True))
        assert t.snapshot().percent == 100.0

    def test_percent_running_doc_with_no_planned_stages_counts_zero(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("documents_discovered", total_documents=2),
            _event("document_started", document_index=0, planned_stages=()),
        )
        assert t.snapshot().percent == 0.0

    def test_tabular_does_not_affect_percent(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("stage_started", stage="tabular"),
        )
        snap = t.snapshot()
        assert snap.tabular == "running" and snap.percent == 0.0
        _feed(
            t,
            _event(
                "stage_completed",
                stage="tabular",
                metrics={"total_processed": 3, "total_failed": 1, "documents": 2},
            ),
            _event("documents_discovered", total_documents=2),
        )
        snap = t.snapshot()
        assert snap.tabular == "completed_with_errors"
        assert snap.percent == 0.0
        assert snap.documents == {}

    def test_tabular_completed_without_failures(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("stage_started", stage="tabular"),
            _event("stage_completed", stage="tabular", metrics={"total_failed": 0}),
        )
        assert t.snapshot().tabular == "completed"

    def test_snapshot_is_a_copy(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("documents_discovered", total_documents=1, stages=("annotation",)),
            _event("document_started", document_index=0, planned_stages=("annotation",)),
            _event("stage_started", document_index=0, stage="annotation"),
        )
        before = t.snapshot()
        _feed(
            t,
            _event("stage_completed", document_index=0, stage="annotation"),
            _event("document_finished", document_index=0, status="success"),
            _event("pipeline_completed", success=True),
        )
        assert before.status == "running"
        assert before.documents[0].status == "running"
        assert before.documents[0].completed_stages == ()
        assert dict(before.running_by_stage) == {"annotation": 1}
        with pytest.raises(TypeError):
            before.documents[1] = None  # type: ignore[index]

    def test_second_pipeline_started_resets_state(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started", job_id="a"),
            _event("documents_discovered", total_documents=1),
            *_doc_events(0, ("annotation",), finish="success"),
            _event("pipeline_completed", success=True),
            _event("pipeline_started", job_id="b"),
        )
        snap = t.snapshot()
        assert snap.job_id == "b" and snap.status == "running"
        assert snap.total_documents is None and snap.documents == {}

    def test_events_for_unknown_documents_are_ignored(self):
        t = ProgressTracker()
        _feed(
            t,
            _event("pipeline_started"),
            _event("stage_started", document_index=7, stage="extraction"),
            _event("document_finished", document_index=7, status="success"),
        )
        assert t.snapshot().documents == {}

    def test_snapshot_from_another_thread_while_applying(self):
        t = ProgressTracker()
        _feed(t, _event("pipeline_started"), _event("documents_discovered", total_documents=500))
        errors: list[BaseException] = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    t.snapshot()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        th = threading.Thread(target=reader)
        th.start()
        try:
            for i in range(500):
                _feed(t, *_doc_events(i, ("extraction", "ingestion"), finish="success"))
        finally:
            stop.set()
            th.join()
        assert errors == []
        assert t.snapshot().documents_succeeded == 500
