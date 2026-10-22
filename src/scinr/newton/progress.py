"""
progress.py — Live progress reporting for :func:`~scinr.newton.pipeline.run_pipeline`.

``run_pipeline()`` only reports its outcome once it returns (``PipelineResult``).
This module lets a caller follow a run while it is in flight, at the granularity of
**one pipeline stage of one document**:

- :class:`ProgressEvent` — the small, immutable event the pipeline emits at every
  stage boundary of every document (plus pipeline-level start/end events).
- :class:`ProgressCallback` / :class:`AsyncProgressCallback` — base classes the
  caller subclasses to react to events (update a job row, push to a websocket, …).
- :class:`ProgressTracker` — built-in implementation that keeps an aggregated,
  in-memory view of the run, queryable at any time with
  :meth:`ProgressTracker.snapshot`.

Events are delivered in order by a single consumer task that reads from an
unbounded queue, so a slow callback never blocks the pipeline. Asynchronous
callbacks are awaited on the pipeline's event loop; synchronous callbacks run in a
dedicated worker thread. Errors raised by a callback are logged on this module's
logger and otherwise ignored.

Usage
-----
::

    from scinr.newton import ProgressTracker, run_pipeline

    tracker = ProgressTracker()
    task = asyncio.create_task(run_pipeline(input_raw="files/", callbacks=[tracker]))
    while not task.done():
        print(f"{tracker.snapshot().percent:.1f}%")
        await asyncio.sleep(2)
    result = await task
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol

from scinr.newton.utils.redaction import redact_secrets

logger = logging.getLogger(__name__)

ProgressEventKind = Literal[
    "pipeline_started",
    "documents_discovered",
    "document_started",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "document_finished",
    "pipeline_completed",
    "pipeline_failed",
]

DocumentStatus = Literal["success", "partial", "failed", "skipped"]

# How long run_pipeline() waits, once the run is over, for queued events to reach
# the callbacks before giving up on them.
_DRAIN_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    """One progress event emitted by :func:`~scinr.newton.pipeline.run_pipeline`.

    Only the fields relevant to ``kind`` are set; every other field is ``None``.
    The event never carries the document itself nor any extracted content. Use
    :meth:`to_dict` to serialize it (e.g. to JSON).

    Attributes:
        kind: What happened (see the "Event reference" table in the Progress Tracking
            guide).
        timestamp: When the event was emitted (timezone-aware, UTC).
        job_id: The ``job_id`` passed to ``run_pipeline()``.
        tenant_id: The document's tenant for document events, otherwise the
            ``tenant_id`` passed to ``run_pipeline()`` (``None`` = public).
        document_index: Position of the document in the list of discovered documents
            (tabular files, convertible documents and skipped files alike); unique
            within one ``run_pipeline()`` call. ``None`` for pipeline events and for
            the **batch** ``stage_started`` / ``stage_completed`` events of
            ``stage="tabular"``.
        doc_path: Path of the document; stable for the whole life of the document.
        document_name: Current document name — the name predicted from the file at
            first, the real ``document_name`` once extraction/ingestion produced it.
        source_path: File the document was discovered from, or ``None``.
        stage: Pipeline stage of a ``stage_*`` event: ``"preprocess"``,
            ``"extraction"``, ``"ingestion"``, ``"annotation"``,
            ``"entity_extraction"`` or ``"tabular"``.
        stages: ``pipeline_started``: the stages requested; ``documents_discovered``:
            the document stages that will actually run.
        planned_stages: ``document_started``: the stages this document will run.
        total_documents: ``documents_discovered``: number of documents to process.
        duration_s: Seconds spent in the stage / document / pipeline that just ended.
        metrics: ``stage_completed`` / ``stage_failed``: counters of the stage
            (``nodes_processed`` / ``nodes_failed`` for document stages — for a
            tabular file, tables written / sheets with errors;
            ``total_processed`` / ``total_failed`` / ``documents`` for the batch
            ``stage="tabular"`` event without ``document_index``). Read-only.
        status: ``document_finished``: ``"success"``, ``"partial"``, ``"failed"`` or
            ``"skipped"``.
        stopped_at: ``document_finished``: stage where the document stopped because
            of a failure, or ``None``.
        success: ``pipeline_completed``: same value as ``PipelineResult.success``.
        error: ``stage_failed`` / ``document_finished`` / ``pipeline_failed``: error
            message, with credentials redacted.
    """

    kind: ProgressEventKind
    timestamp: datetime
    job_id: str | None = None
    tenant_id: str | None = None
    # ── Document identity ────────────────────────────────────────────────────
    document_index: int | None = None
    doc_path: str | None = None
    document_name: str | None = None
    source_path: str | None = None
    # ── Stage ────────────────────────────────────────────────────────────────
    stage: str | None = None
    # ── Kind-specific data ───────────────────────────────────────────────────
    stages: tuple[str, ...] | None = None
    planned_stages: tuple[str, ...] | None = None
    total_documents: int | None = None
    duration_s: float | None = None
    metrics: Mapping[str, int] | None = None
    status: DocumentStatus | None = None
    stopped_at: str | None = None
    success: bool | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        # Frozen dataclass: normalise through object.__setattr__ so the event is
        # really immutable (read-only metrics, tuples) and never leaks credentials.
        if self.metrics is not None:
            object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        if self.stages is not None:
            object.__setattr__(self, "stages", tuple(self.stages))
        if self.planned_stages is not None:
            object.__setattr__(self, "planned_stages", tuple(self.planned_stages))
        if self.error is not None:
            object.__setattr__(self, "error", redact_secrets(self.error))

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a plain, JSON-serializable ``dict``.

        ``timestamp`` becomes an ISO 8601 string, ``metrics`` a ``dict`` and the
        stage tuples lists. (``dataclasses.asdict()`` cannot copy the read-only
        ``metrics`` mapping — use this method instead.)
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, Mapping):
                value = dict(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out


# ---------------------------------------------------------------------------
# Callback base classes
# ---------------------------------------------------------------------------


class ProgressCallback:
    """Base class for **synchronous** progress callbacks.

    Subclass it and override either :meth:`on_event` (one entry point for every
    event) or only the ``on_<kind>`` methods you care about; every method is a
    no-op by default.

    Runs in a dedicated worker thread (one per ``run_pipeline()`` call), never on
    the event loop, so it may block — call a blocking database client, ``requests``,
    etc. Events are delivered one at a time and in order, so within one
    ``run_pipeline()`` call the callback never runs concurrently with itself. A
    single instance shared by several concurrent ``run_pipeline()`` calls *can*
    run concurrently with itself: do not assume thread-safety in that case.

    Exceptions raised by a callback are logged (logger ``scinr.newton.progress``)
    and ignored; they never abort the ingestion.
    """

    def on_event(self, event: ProgressEvent) -> None:
        """Called for every event. By default dispatches to ``on_<event.kind>``."""
        handler = getattr(self, f"on_{event.kind}", None)
        if handler is not None:
            handler(event)

    def on_pipeline_started(self, event: ProgressEvent) -> None:
        """The run started (always the first event)."""
        return None

    def on_documents_discovered(self, event: ProgressEvent) -> None:
        """The documents to process are known (``total_documents``, ``stages``)."""
        return None

    def on_document_started(self, event: ProgressEvent) -> None:
        """A document started processing (``planned_stages``)."""
        return None

    def on_stage_started(self, event: ProgressEvent) -> None:
        """A document (or the tabular batch) entered ``stage``."""
        return None

    def on_stage_completed(self, event: ProgressEvent) -> None:
        """A document (or the tabular batch) finished ``stage`` (``metrics``)."""
        return None

    def on_stage_failed(self, event: ProgressEvent) -> None:
        """A document stopped at ``stage`` because of a failure (``error``)."""
        return None

    def on_document_finished(self, event: ProgressEvent) -> None:
        """A document finished processing (``status``, ``stopped_at``, ``error``)."""
        return None

    def on_pipeline_completed(self, event: ProgressEvent) -> None:
        """The run returned a ``PipelineResult`` (always the last event)."""
        return None

    def on_pipeline_failed(self, event: ProgressEvent) -> None:
        """The run raised an exception (always the last event)."""
        return None


class AsyncProgressCallback:
    """Base class for **asynchronous** progress callbacks.

    Same contract as :class:`ProgressCallback`, but every method is a coroutine
    awaited on the pipeline's own event loop — so it can use async clients bound to
    that loop (motor, httpx, the async Neo4j driver, …). Events are delivered one at
    a time and in order; within one ``run_pipeline()`` call the callback never runs
    concurrently with itself (a shared instance across concurrent calls can).

    A coroutine that blocks internally (``requests``, ``pymongo``, ``time.sleep``)
    freezes the whole event loop — and with it the ingestion. Write blocking code as
    a :class:`ProgressCallback` instead.

    Exceptions raised by a callback are logged (logger ``scinr.newton.progress``)
    and ignored; they never abort the ingestion.
    """

    async def on_event(self, event: ProgressEvent) -> None:
        """Called for every event. By default dispatches to ``on_<event.kind>``."""
        handler = getattr(self, f"on_{event.kind}", None)
        if handler is not None:
            await handler(event)

    async def on_pipeline_started(self, event: ProgressEvent) -> None:
        """The run started (always the first event)."""
        return None

    async def on_documents_discovered(self, event: ProgressEvent) -> None:
        """The documents to process are known (``total_documents``, ``stages``)."""
        return None

    async def on_document_started(self, event: ProgressEvent) -> None:
        """A document started processing (``planned_stages``)."""
        return None

    async def on_stage_started(self, event: ProgressEvent) -> None:
        """A document (or the tabular batch) entered ``stage``."""
        return None

    async def on_stage_completed(self, event: ProgressEvent) -> None:
        """A document (or the tabular batch) finished ``stage`` (``metrics``)."""
        return None

    async def on_stage_failed(self, event: ProgressEvent) -> None:
        """A document stopped at ``stage`` because of a failure (``error``)."""
        return None

    async def on_document_finished(self, event: ProgressEvent) -> None:
        """A document finished processing (``status``, ``stopped_at``, ``error``)."""
        return None

    async def on_pipeline_completed(self, event: ProgressEvent) -> None:
        """The run returned a ``PipelineResult`` (always the last event)."""
        return None

    async def on_pipeline_failed(self, event: ProgressEvent) -> None:
        """The run raised an exception (always the last event)."""
        return None


# ---------------------------------------------------------------------------
# Built-in tracker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentProgress:
    """Progress of one document, as seen by :class:`ProgressTracker`.

    Attributes:
        document_index: Position of the document among the discovered documents.
        doc_path: Path of the document.
        document_name: Latest known document name.
        tenant_id: Tenant of the document (``None`` = public).
        status: ``"running"`` until the document finishes, then ``"success"``,
            ``"partial"``, ``"failed"`` or ``"skipped"``.
        current_stage: Stage in progress right now, or ``None``.
        planned_stages: Stages this document will run.
        completed_stages: Stages this document completed, in order.
        stopped_at: Stage where the document stopped because of a failure, or ``None``.
        error: Error message when the document failed, or ``None``.
        started_at: When the document started (UTC).
        finished_at: When the document finished (UTC), or ``None`` while running.
    """

    document_index: int
    doc_path: str | None
    document_name: str | None
    tenant_id: str | None
    status: Literal["running", "success", "partial", "failed", "skipped"]
    current_stage: str | None
    planned_stages: tuple[str, ...]
    completed_stages: tuple[str, ...]
    stopped_at: str | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class ProgressSnapshot:
    """Point-in-time view of a run, returned by :meth:`ProgressTracker.snapshot`.

    Attributes:
        status: ``"pending"`` (no event yet), ``"running"``, ``"success"``,
            ``"completed_with_errors"`` (the run returned with
            ``PipelineResult.success == False``) or ``"failed"`` (the run raised).
        job_id: The ``job_id`` passed to ``run_pipeline()``.
        started_at: When the run started (UTC), or ``None`` while pending.
        finished_at: When the run ended (UTC), or ``None`` while not finished.
        elapsed_s: Seconds since the run started, up to ``finished_at`` once finished.
        stages: Document stages that will run (empty until documents are discovered).
        total_documents: Number of documents to process, or ``None`` until discovered.
        documents_pending: Documents not started yet (``0`` while unknown).
        documents_running: Documents in progress.
        documents_succeeded: Documents finished with status ``"success"``.
        documents_partial: Documents finished with status ``"partial"``.
        documents_failed: Documents finished with status ``"failed"``.
        documents_skipped: Documents finished with status ``"skipped"``.
        running_by_stage: Number of documents currently in each stage.
        completed_by_stage: Number of documents that completed each stage.
        failed_by_stage: Number of documents that failed at each stage.
        tabular: State of the tabular batch: ``"running"``, ``"completed"``,
            ``"completed_with_errors"``, or ``None`` when no tabular files are processed.
            Each tabular file is also tracked as a document of its own.
        percent: Overall document progress, 0-100, rounded to one decimal. Every
            discovered document counts: tabular files, convertible documents and
            skipped files (which count as finished).
        documents: Progress of every started document, keyed by ``document_index``.
        error: Error message when the run raised, otherwise ``None``.
    """

    status: Literal["pending", "running", "success", "completed_with_errors", "failed"]
    job_id: str | None
    started_at: datetime | None
    finished_at: datetime | None
    elapsed_s: float
    stages: tuple[str, ...]
    total_documents: int | None
    documents_pending: int
    documents_running: int
    documents_succeeded: int
    documents_partial: int
    documents_failed: int
    documents_skipped: int
    running_by_stage: Mapping[str, int]
    completed_by_stage: Mapping[str, int]
    failed_by_stage: Mapping[str, int]
    tabular: Literal["running", "completed", "completed_with_errors"] | None
    percent: float
    documents: Mapping[int, DocumentProgress]
    error: str | None


@dataclass
class _DocState:
    """Mutable per-document state kept by ProgressTracker (never exposed)."""

    document_index: int
    doc_path: str | None
    document_name: str | None
    tenant_id: str | None
    planned_stages: tuple[str, ...]
    started_at: datetime
    status: str = "running"
    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    stopped_at: str | None = None
    error: str | None = None
    finished_at: datetime | None = None

    def freeze(self) -> DocumentProgress:
        return DocumentProgress(
            document_index=self.document_index,
            doc_path=self.doc_path,
            document_name=self.document_name,
            tenant_id=self.tenant_id,
            status=self.status,  # type: ignore[arg-type]
            current_stage=self.current_stage,
            planned_stages=self.planned_stages,
            completed_stages=tuple(self.completed_stages),
            stopped_at=self.stopped_at,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


_TERMINAL_STATUSES = frozenset({"success", "completed_with_errors", "failed"})


class ProgressTracker:
    """Built-in progress callback that keeps an aggregated, in-memory view of a run.

    Pass it in ``run_pipeline(callbacks=[tracker])`` and call :meth:`snapshot` at any
    time — from the event loop or from another thread — to get a
    :class:`ProgressSnapshot`. The tracker is updated synchronously as each event is
    emitted (it does not go through the callback queue), so a snapshot is never
    behind the pipeline.

    The state lives only in the memory of the current process: it is not persisted
    and is not visible from other processes. One instance is meant for **one**
    ``run_pipeline()`` call; a second ``pipeline_started`` event resets it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._status = "pending"
        self._job_id: str | None = None
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._stages: tuple[str, ...] = ()
        self._total: int | None = None
        self._docs: dict[int, _DocState] = {}
        self._tabular: str | None = None
        self._error: str | None = None

    def _apply(self, event: ProgressEvent) -> None:
        """Fold *event* into the tracked state (called by the dispatcher)."""
        with self._lock:
            kind = event.kind
            if kind == "pipeline_started":
                self._reset()
                self._status = "running"
                self._job_id = event.job_id
                self._started_at = event.timestamp
            elif kind == "documents_discovered":
                self._total = event.total_documents
                self._stages = tuple(event.stages or ())
            elif kind == "document_started":
                if event.document_index is None:
                    return
                self._docs[event.document_index] = _DocState(
                    document_index=event.document_index,
                    doc_path=event.doc_path,
                    document_name=event.document_name,
                    tenant_id=event.tenant_id,
                    planned_stages=tuple(event.planned_stages or ()),
                    started_at=event.timestamp,
                )
            elif kind in ("stage_started", "stage_completed", "stage_failed"):
                if event.stage == "tabular" and event.document_index is None:
                    self._apply_tabular(event)
                    return
                doc = self._doc(event)
                if doc is None:
                    return
                if kind == "stage_started":
                    doc.current_stage = event.stage
                elif kind == "stage_completed":
                    doc.current_stage = None
                    if event.stage is not None:
                        doc.completed_stages.append(event.stage)
                else:
                    doc.current_stage = None
                    doc.failed_stage = event.stage
            elif kind == "document_finished":
                doc = self._doc(event)
                if doc is None:
                    return
                doc.status = event.status or "success"
                doc.current_stage = None
                doc.stopped_at = event.stopped_at
                doc.error = event.error
                doc.finished_at = event.timestamp
            elif kind == "pipeline_completed":
                self._status = "success" if event.success else "completed_with_errors"
                self._finished_at = event.timestamp
            elif kind == "pipeline_failed":
                self._status = "failed"
                self._error = event.error
                self._finished_at = event.timestamp

    def _doc(self, event: ProgressEvent) -> _DocState | None:
        if event.document_index is None:
            return None
        doc = self._docs.get(event.document_index)
        if doc is not None and event.document_name is not None:
            doc.document_name = event.document_name
        return doc

    def _apply_tabular(self, event: ProgressEvent) -> None:
        if event.kind == "stage_started":
            self._tabular = "running"
        elif event.kind == "stage_completed":
            failed = (event.metrics or {}).get("total_failed", 0)
            self._tabular = "completed_with_errors" if failed > 0 else "completed"

    def snapshot(self) -> ProgressSnapshot:
        """Return the current progress of the run. Thread-safe.

        The returned snapshot is an independent copy: later events do not change it.
        """
        with self._lock:
            docs = list(self._docs.values())
            terminal = self._status in _TERMINAL_STATUSES

            if self._started_at is None:
                elapsed = 0.0
            else:
                end = self._finished_at or datetime.now(UTC)
                elapsed = max(0.0, (end - self._started_at).total_seconds())

            counts = {"running": 0, "success": 0, "partial": 0, "failed": 0, "skipped": 0}
            running_by_stage: dict[str, int] = {}
            completed_by_stage: dict[str, int] = {}
            failed_by_stage: dict[str, int] = {}
            progress_sum = 0.0
            for doc in docs:
                counts[doc.status] = counts.get(doc.status, 0) + 1
                if doc.status == "running":
                    if doc.current_stage is not None:
                        running_by_stage[doc.current_stage] = (
                            running_by_stage.get(doc.current_stage, 0) + 1
                        )
                    if doc.planned_stages:
                        progress_sum += len(doc.completed_stages) / len(doc.planned_stages)
                else:
                    progress_sum += 1.0
                for stage in doc.completed_stages:
                    completed_by_stage[stage] = completed_by_stage.get(stage, 0) + 1
                if doc.failed_stage is not None:
                    failed_by_stage[doc.failed_stage] = (
                        failed_by_stage.get(doc.failed_stage, 0) + 1
                    )

            if self._total is None or self._total == 0:
                percent = 100.0 if terminal else 0.0
            else:
                percent = round(min(100.0, 100.0 * progress_sum / self._total), 1)

            pending = max(0, self._total - len(docs)) if self._total is not None else 0

            return ProgressSnapshot(
                status=self._status,  # type: ignore[arg-type]
                job_id=self._job_id,
                started_at=self._started_at,
                finished_at=self._finished_at,
                elapsed_s=elapsed,
                stages=self._stages,
                total_documents=self._total,
                documents_pending=pending,
                documents_running=counts["running"],
                documents_succeeded=counts["success"],
                documents_partial=counts["partial"],
                documents_failed=counts["failed"],
                documents_skipped=counts["skipped"],
                running_by_stage=MappingProxyType(running_by_stage),
                completed_by_stage=MappingProxyType(completed_by_stage),
                failed_by_stage=MappingProxyType(failed_by_stage),
                tabular=self._tabular,  # type: ignore[arg-type]
                percent=percent,
                documents=MappingProxyType({d.document_index: d.freeze() for d in docs}),
                error=self._error,
            )


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------

_AnyCallback = ProgressCallback | AsyncProgressCallback | ProgressTracker


class _ProgressEmitter(Protocol):
    def emit(self, kind: ProgressEventKind, **fields: Any) -> None: ...


class _NullEmitter:
    """Emitter used when run_pipeline() gets no callbacks: does nothing."""

    def emit(self, kind: ProgressEventKind, **fields: Any) -> None:
        return None


_NULL_EMITTER = _NullEmitter()


def _validate_callbacks(callbacks: Sequence[_AnyCallback] | None) -> list[_AnyCallback]:
    """Return *callbacks* as a list, rejecting anything that is not a callback."""
    if callbacks is None:
        return []
    if not isinstance(callbacks, Sequence) or isinstance(callbacks, (str, bytes)):
        raise TypeError(
            "callbacks must be a sequence of ProgressCallback, AsyncProgressCallback or "
            f"ProgressTracker instances, got {type(callbacks).__name__}."
        )
    validated: list[_AnyCallback] = []
    for cb in callbacks:
        if not isinstance(cb, (ProgressCallback, AsyncProgressCallback, ProgressTracker)):
            raise TypeError(
                "Each callback must be an instance of ProgressCallback, "
                f"AsyncProgressCallback or ProgressTracker, got {type(cb).__name__}."
            )
        validated.append(cb)
    return validated


class _ProgressDispatcher:
    """Builds progress events and delivers them to the callbacks of one run.

    ``ProgressTracker`` instances are updated inline in :meth:`emit`. Every other
    callback is fed by a single consumer task reading an unbounded queue, so
    :meth:`emit` never blocks: async callbacks are awaited on the event loop, sync
    callbacks run in a dedicated single-thread executor (never the loop's default
    executor, which the ingestion uses via ``asyncio.to_thread``). Must be created
    while the event loop is running.
    """

    def __init__(
        self,
        callbacks: Sequence[_AnyCallback],
        *,
        job_id: str | None,
        tenant_id: str | None,
    ) -> None:
        self._job_id = job_id
        self._tenant_id = tenant_id
        self._trackers = [cb for cb in callbacks if isinstance(cb, ProgressTracker)]
        self._queued = [cb for cb in callbacks if not isinstance(cb, ProgressTracker)]
        self._closed = False
        self._queue: asyncio.Queue[ProgressEvent | None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        if self._queued:
            if any(isinstance(cb, ProgressCallback) for cb in self._queued):
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="scinr-progress"
                )
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._consume(), name="scinr-progress-dispatcher")

    def emit(self, kind: ProgressEventKind, **fields: Any) -> None:
        """Build a ``ProgressEvent`` and hand it to every callback without blocking.

        Always called from the event loop's thread (every call site in
        ``pipeline.py`` is async code, never inside ``asyncio.to_thread``), which is
        why ``put_nowait`` is used without ``call_soon_threadsafe``. Emitting from
        another thread would require changing that.
        """
        if self._closed:
            return
        fields.setdefault("job_id", self._job_id)
        fields.setdefault("tenant_id", self._tenant_id)
        event = ProgressEvent(kind=kind, timestamp=datetime.now(UTC), **fields)
        for tracker in self._trackers:
            try:
                tracker._apply(event)
            except Exception:
                logger.exception(
                    "Progress tracker %r failed on %s event (ignored).", tracker, event.kind
                )
        if self._queue is not None:
            self._queue.put_nowait(event)

    async def _consume(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while (event := await self._queue.get()) is not None:
            for cb in self._queued:
                try:
                    if isinstance(cb, AsyncProgressCallback):
                        await cb.on_event(event)
                    else:
                        await loop.run_in_executor(self._executor, cb.on_event, event)
                except Exception:
                    logger.exception(
                        "Progress callback %r failed on %s event (ignored).", cb, event.kind
                    )

    async def aclose(self, timeout: float = _DRAIN_TIMEOUT_SECONDS) -> None:
        """Deliver the queued events, waiting at most *timeout* seconds. Idempotent.

        Past the timeout the remaining events are dropped (with a warning). A sync
        callback still running at that point cannot be interrupted: its worker
        thread stays alive until the callback returns.
        """
        if self._closed:
            return
        self._closed = True
        if self._task is None or self._queue is None:
            return
        try:
            self._queue.put_nowait(None)
            try:
                await asyncio.wait_for(self._task, timeout)
            except TimeoutError:
                logger.warning(
                    "Progress callbacks did not finish within %.1fs; %d queued event(s) "
                    "dropped.",
                    timeout,
                    max(0, self._queue.qsize() - 1),
                )
            except Exception:
                logger.exception("Progress dispatcher failed while draining (ignored).")
        finally:
            self._task.cancel()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
