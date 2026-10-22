# Progress Tracking

`run_pipeline()` returns a `PipelineResult` when it finishes, but a large ingestion can run for hours. Pass **callbacks** to follow it while it runs: the pipeline emits a small event every time a document enters or leaves a stage, plus one event when the run starts and one when it ends. Use them to update the status of an ingestion job in your database, show a percentage in a UI, or find out where documents pile up.

Progress is reported **per stage and per document** (`preprocess`, `extraction`, `ingestion`, `annotation`, `entity_extraction`, `tabular`). Every file found under `input_raw` is a document: tabular files (CSV/XLSX/XLS), convertible documents, and files the pipeline cannot interpret, which are reported as `"skipped"`. There are no events per page, chunk or node.

```python
from scinr.newton import (
    AsyncProgressCallback,
    DocumentProgress,
    ProgressCallback,
    ProgressEvent,
    ProgressSnapshot,
    ProgressTracker,
    run_pipeline,
)
```

`callbacks` is optional. Without it, `run_pipeline()` behaves exactly as before: no events are built and no extra task or thread is started.

---

## Quick start with `ProgressTracker`

`ProgressTracker` is the built-in callback. It keeps an aggregated view of the run in memory, which you read at any time with `snapshot()`:

```python
import asyncio
from scinr.newton import ProgressTracker, configure, run_pipeline

async def main():
    configure(llm=my_llm, neo4j_user="neo4j", neo4j_password="...", neo4j_database="neo4j")

    tracker = ProgressTracker()
    task = asyncio.create_task(
        run_pipeline(input_raw="./raw_docs", job_id="ingest-42", callbacks=[tracker])
    )

    while not task.done():
        snap = tracker.snapshot()
        print(
            f"{snap.percent:5.1f}%  "
            f"{snap.documents_running} running, {snap.documents_pending} pending, "
            f"{snap.documents_failed} failed — by stage: {dict(snap.running_by_stage)}"
        )
        await asyncio.sleep(2)

    result = await task          # the PipelineResult, as usual
    print(tracker.snapshot().status, result.success)

asyncio.run(main())
```

The polling loop and the pipeline share the event loop: `run_pipeline()` must run as a task (`asyncio.create_task`) so the loop can print while it works.

---

## How it works

```
 run_pipeline()                                   ┌──────────────────────────────┐
 _process_document_unit() ── emit(event) ───┬───► │ ProgressTracker (inline,     │
   (one per document, async)                │     │ under a threading.Lock)      │
                                            │     └──────────────────────────────┘
                                            │ put_nowait (never blocks)
                                            ▼
                                   asyncio.Queue (unbounded)
                                            │
                                            ▼
                                   one consumer task, events in FIFO order
                                            │
                       ┌────────────────────┴─────────────────────┐
                       ▼                                          ▼
         AsyncProgressCallback                         ProgressCallback (sync)
         awaited on the pipeline's event loop          run in a dedicated worker thread
                                                       ("scinr-progress-*", 1 per run)
```

- **The pipeline never waits for a callback.** Emitting an event only puts it on a queue. One consumer task takes the events in order and hands each to every callback, in the order of the `callbacks` list.
- **Asynchronous callbacks** (`AsyncProgressCallback`) are awaited on the pipeline's own event loop, so they can use async clients bound to that loop (motor, httpx, the async Neo4j driver).
- **Synchronous callbacks** (`ProgressCallback`) run in a worker thread created for this `run_pipeline()` call. It is not the event loop's default executor, which the ingestion stage uses.
- **`ProgressTracker` does not use the queue**: it is updated as each event is emitted, so a snapshot is never behind the pipeline.

Guarantees for each `run_pipeline(callbacks=[...])` call:

- `pipeline_started` is always the **first** event, and exactly one of `pipeline_completed` / `pipeline_failed` is always the **last** one. That terminal event is delivered to every callback **before** `run_pipeline()` returns or raises (up to 30 s, see [Slow callbacks and shutdown](#slow-callbacks-and-shutdown)).
- Every `document_started` is followed, later, by exactly one `document_finished` for the same `document_index`.
- Events of one document arrive in strict order. Events of different documents **interleave**, since documents run concurrently (`parallel_docs`).
- A callback never runs concurrently with itself within one call: the next event is delivered once the previous one has been handled by every callback.

The lifecycle of a run:

```
pipeline_started
├─ documents_discovered             (not if the run fails before discovery)
├─ for each skipped file (input_raw): document_started → document_finished(status="skipped")
├─ stage_started                    (stage="tabular", the batch, if there are tabular files)
│    for each tabular file: document_started → stage_started → stage_completed | stage_failed → document_finished
├─ stage_completed                  (stage="tabular", the batch)
├─ for each other document, in the order it starts:
│    document_started
│    ├─ stage_started → stage_completed | stage_failed   (for each stage it reaches)
│    document_finished
└─ pipeline_completed | pipeline_failed
```

---

## Event reference

Every event is a frozen `ProgressEvent` dataclass. Fields that do not apply to the event's `kind` are `None`. Every event carries `kind`, `timestamp` (timezone-aware UTC), `job_id` and `tenant_id`; document events also carry `document_index`, `doc_path`, `document_name` and `source_path`.

| `kind` | When | Specific fields |
|---|---|---|
| `pipeline_started` | `run_pipeline()` was called (before validating its parameters). | `stages`: the stages requested. |
| `documents_discovered` | The documents to process are known. Emitted once per run, before any document starts. | `total_documents`: with `input_raw`, every file found — tabular files, convertible documents and files that will be skipped (on a `stages=["tabular"]` run, only the tabular files). `stages`: the document stages that will run (after `ingestion_input_dir` / `extraction_input_dir` removed the skipped ones); includes `"tabular"` when there are tabular files. |
| `document_started` | A document got a processing slot (`parallel_docs`). | `planned_stages`: the stages this document will run. |
| `stage_started` | A document (or the tabular batch, without `document_index`) enters `stage`. | `stage` |
| `stage_completed` | A document (or the tabular batch, without `document_index`) finished `stage` and moves on. | `stage`, `duration_s`, `metrics` |
| `stage_failed` | A document stopped at `stage` because of a failure. | `stage`, `duration_s`, `metrics`, `error` |
| `document_finished` | A document is done, whatever the outcome. | `status`, `stopped_at`, `error`, `duration_s` |
| `pipeline_completed` | `run_pipeline()` is about to return its `PipelineResult`. | `success` (= `PipelineResult.success`), `duration_s` |
| `pipeline_failed` | `run_pipeline()` is about to raise (invalid parameters, a pre-flight check, cancellation, …). | `error`, `duration_s` |

Identity fields:

- `document_index` is the document's position in the discovered list. It is unique within one call and is the key to correlate the events of one document. With `input_raw`, tabular files come first (`0..T-1`), then the convertible documents, then the skipped files. It is `None` on pipeline events and on the **batch** `stage_started` / `stage_completed` of `stage="tabular"`; the per-file tabular events carry it.
- `doc_path` is the document's path, stable for the whole life of the document — use it (with `tenant_id`) as the document's key in your own database.
- `document_name` is the name predicted from the file until extraction (or ingestion) produces the real one; from that `stage_completed` on, it is the real `document_name`.
- `tenant_id` is the document's tenant on document events (for an `extract-*.json` ingested without `tenant_id`, the tenant baked into the file), otherwise the `tenant_id` passed to `run_pipeline()`.

`metrics` is a read-only mapping:

| Event | Keys |
|---|---|
| `stage_completed` / `stage_failed` of a document | `nodes_processed`, `nodes_failed` — the `DocumentResult` of that stage (`0` / `1` when an unexpected error stopped the stage). For a tabular file: tables written / sheets with errors. |
| Batch `stage_completed` with `stage="tabular"` (no `document_index`) | `total_processed`, `total_failed`, `documents` — from the tabular `StageResult` |

`document_finished.status`, checked in this order:

| Condition | `status` |
|---|---|
| An unexpected error stopped the document (`UnitResult.fatal_error`) | `"failed"` (`error` = that error) |
| The document stopped at a stage (`stopped_at` is set) | `"failed"` (`error` = the errors of that stage, joined with `"; "`) |
| A file the pipeline cannot interpret: an extension with no converter that is not tabular (`"unsupported format: .xyz"`), or a tabular file that cannot be read or has no data rows — e.g. an AppleDouble `__MACOSX/._name.csv` (see [Tabular files](#tabular-files)) | `"skipped"` (`error` = the reason) |
| Preprocess found nothing to convert | `"skipped"` |
| Some stage had `nodes_failed > 0` but the document kept going (`on_partial_failure="continue"` / `"warn"`), or a tabular file wrote some tables but some sheets failed | `"partial"` |
| Otherwise | `"success"` |

When `stage_failed` is emitted:

- Preprocess, extraction and ingestion failures always stop the document: `stage_failed` for that stage, then `document_finished(status="failed", stopped_at=<stage>)`.
- An annotation or entity-extraction stage with `nodes_failed > 0` emits `stage_failed` **only** with `on_partial_failure="abort"`. With `"continue"` / `"warn"` it emits `stage_completed` with `metrics["nodes_failed"] > 0`, and the document goes on.
- An unexpected exception during a stage emits `stage_failed` for that stage with the exception message.
- A tabular file that wrote no table (every sheet failed, or an unexpected error) emits `stage_failed` with `stage="tabular"`, then `document_finished(status="failed", stopped_at="tabular")`.
- There is no `stage_failed` for the tabular **batch**: if `run_tabular_pipeline` raises, the run fails and you get `pipeline_failed`.

Every `error` is passed through the same credential redaction as `PipelineResult` errors (connection-string passwords and configured secrets are replaced by `***`).

A document that runs every stage successfully produces:

```
document_started      planned_stages=("preprocess", "extraction", "ingestion", "annotation", "entity_extraction")
stage_started         stage="preprocess"
stage_completed       stage="preprocess"         metrics={"nodes_processed": 1, "nodes_failed": 0}
stage_started         stage="extraction"
stage_completed       stage="extraction"         document_name=<real name>
stage_started         stage="ingestion"
stage_completed       stage="ingestion"
stage_started         stage="annotation"
stage_completed       stage="annotation"         metrics={"nodes_processed": 37, "nodes_failed": 0}
stage_started         stage="entity_extraction"
stage_completed       stage="entity_extraction"  metrics={"nodes_processed": 12, "nodes_failed": 0}
document_finished     status="success"
```

To serialize an event, use `event.to_dict()`: it returns a plain `dict` with `timestamp` as an ISO 8601 string, `metrics` as a `dict` and the stage tuples as lists, ready for `json.dumps`. (`dataclasses.asdict()` cannot copy the read-only `metrics` mapping.)

### Tabular files

Tabular files (CSV/XLSX/XLS) run in a batch before the other documents, but each file is a document of its own, with `planned_stages=("tabular",)`:

```
stage_started         stage="tabular"                          # the batch, no document_index
document_started      document_index=0  planned_stages=("tabular",)
stage_started         document_index=0  stage="tabular"
stage_completed       document_index=0  stage="tabular"  metrics={"nodes_processed": 3, "nodes_failed": 0}
document_finished     document_index=0  status="success"
document_started      document_index=1  planned_stages=("tabular",)
stage_started         document_index=1  stage="tabular"
document_finished     document_index=1  status="skipped"  error="unreadable tabular file: ..."
stage_completed       stage="tabular"  metrics={"total_processed": 3, "total_failed": 0, "documents": 2}
```

- Each file is scanned first. A file that cannot be read (binary content, a corrupt workbook, `.xls`) or whose sheets have no data rows is `"skipped"`, with no stage outcome: nothing is stored and no `:Document` is created. Files are judged only by extension and content — there is no list of names or folders, so an AppleDouble `__MACOSX/._name.csv` is discovered as tabular and then skipped for its binary content.
- `"success"`: every sheet was written. `"partial"`: some tables were written but some sheets failed (`stage_completed` with `nodes_failed > 0`). `"failed"`: no table was written (`stage_failed`).
- A file with a converter but garbage content (e.g. `__MACOSX/._report.pdf`) cannot be told apart without opening it: it goes to preprocess and ends as `"failed"` there.

---

## The built-in `ProgressTracker`

`tracker.snapshot()` returns a frozen `ProgressSnapshot`:

| Field | Meaning |
|---|---|
| `status` | `"pending"` (no event yet), `"running"`, `"success"`, `"completed_with_errors"` (the run returned with `PipelineResult.success == False`), `"failed"` (the run raised) |
| `job_id`, `started_at`, `finished_at`, `elapsed_s` | The run's job id, UTC start/end times, and seconds elapsed (up to `finished_at` once finished) |
| `stages` | The document stages that will run (empty until `documents_discovered`) |
| `total_documents` | Number of documents, or `None` until discovered |
| `documents_pending` / `_running` / `_succeeded` / `_partial` / `_failed` / `_skipped` | Document counters (`documents_pending` is `0` while `total_documents` is unknown) |
| `running_by_stage` | How many documents are in each stage right now |
| `completed_by_stage` / `failed_by_stage` | How many documents completed / failed each stage |
| `tabular` | State of the tabular batch: `None`, `"running"`, `"completed"` or `"completed_with_errors"`. Each tabular file is also a document in the counters. |
| `percent` | Overall document progress, 0–100, one decimal |
| `documents` | A `DocumentProgress` for every started document, keyed by `document_index` |
| `error` | The error of `pipeline_failed`, otherwise `None` |

`DocumentProgress` holds, per document: `document_index`, `doc_path`, `document_name`, `tenant_id`, `status` (`"running"` or the final status), `current_stage`, `planned_stages`, `completed_stages`, `stopped_at`, `error`, `started_at` and `finished_at`.

How `percent` is computed:

- A finished document counts `1` (whatever its status), a running one `completed_stages / planned_stages`, a pending one `0`. The sum is divided by `total_documents`.
- Before `documents_discovered`, or with zero documents, `percent` is `0.0` while the run is going and `100.0` once it has ended.
- Every discovered document counts, tabular files and skipped files included (a skipped file counts as finished). The `tabular` field still follows the tabular batch as a whole.

Properties:

- **Thread-safe**: `snapshot()` can be called from the event loop or from any thread. It returns independent copies; later events do not change a snapshot you already hold.
- **In-process only**: the state lives in the memory of the process running the pipeline. It is not persisted and is not visible from other processes.
- **One instance per call**: a tracker follows one `run_pipeline()` call. A second `pipeline_started` resets it.

### Recipe: a script or notebook

The [Quick start](#quick-start-with-progresstracker) loop is the whole recipe. In a notebook, where an event loop is already running, `await` the coroutine directly instead of `asyncio.run(main())`.

### Recipe: a FastAPI worker

Keep the trackers **in your application**, keyed by job id, and serve their snapshot from an endpoint:

```python
import asyncio
import dataclasses
import uuid

from fastapi import FastAPI, HTTPException
from scinr.newton import ProgressTracker, run_pipeline

app = FastAPI()
trackers: dict[str, ProgressTracker] = {}
tasks: set[asyncio.Task] = set()


@app.post("/jobs")
async def start_job(input_dir: str, tenant_id: str):
    job_id = f"ingest-{uuid.uuid4()}"
    tracker = ProgressTracker()
    trackers[job_id] = tracker
    task = asyncio.create_task(
        run_pipeline(input_raw=input_dir, tenant_id=tenant_id, job_id=job_id, callbacks=[tracker])
    )
    tasks.add(task)                      # keep a reference so the task is not garbage-collected
    task.add_done_callback(tasks.discard)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str):
    tracker = trackers.get(job_id)
    if tracker is None:
        raise HTTPException(status_code=404)
    snap = tracker.snapshot()
    return {
        "status": snap.status,
        "percent": snap.percent,
        "elapsed_s": snap.elapsed_s,
        "total_documents": snap.total_documents,
        "documents_failed": snap.documents_failed,
        "running_by_stage": dict(snap.running_by_stage),
        "documents": [dataclasses.asdict(d) for d in snap.documents.values()],
    }
```

This only works while the request reaches the process that runs the job. With several workers, processes or pods, write the progress to a shared store with your own callback (next section), or dump the snapshot there periodically.

---

## Writing your own callback

Subclass `ProgressCallback` (synchronous) or `AsyncProgressCallback` (asynchronous). Every method is a no-op by default: override `on_event(event)` to receive everything, or only the `on_<kind>` methods you need (`on_pipeline_started`, `on_documents_discovered`, `on_document_started`, `on_stage_started`, `on_stage_completed`, `on_stage_failed`, `on_document_finished`, `on_pipeline_completed`, `on_pipeline_failed`). `on_event` dispatches to them, so if you override `on_event` the `on_<kind>` methods are no longer called unless you call `super().on_event(event)`.

### Asynchronous: a job document in MongoDB (motor)

One document per job in `ingestion_jobs`, one document per (job, document) in `ingestion_job_documents`:

```python
from motor.motor_asyncio import AsyncIOMotorClient
from scinr.newton import AsyncProgressCallback, ProgressEvent


class JobStatusUpdater(AsyncProgressCallback):
    def __init__(self, db, job_id: str):
        self.jobs = db["ingestion_jobs"]
        self.docs = db["ingestion_job_documents"]
        self.job_id = job_id

    async def on_pipeline_started(self, event: ProgressEvent) -> None:
        await self.jobs.update_one(
            {"_id": self.job_id},
            {"$set": {"status": "running", "started_at": event.timestamp}},
            upsert=True,
        )

    async def on_documents_discovered(self, event: ProgressEvent) -> None:
        await self.jobs.update_one(
            {"_id": self.job_id}, {"$set": {"total_documents": event.total_documents}}
        )

    async def on_stage_started(self, event: ProgressEvent) -> None:
        if event.document_index is None:          # the tabular batch (each file has its own events)
            return
        await self.docs.update_one(
            {"job_id": self.job_id, "doc_path": event.doc_path},
            {"$set": {
                "document_name": event.document_name,
                "tenant_id": event.tenant_id,
                "status": "running",
                "stage": event.stage,
                "updated_at": event.timestamp,
            }},
            upsert=True,
        )

    async def on_document_finished(self, event: ProgressEvent) -> None:
        await self.docs.update_one(
            {"job_id": self.job_id, "doc_path": event.doc_path},
            {"$set": {
                "document_name": event.document_name,
                "status": event.status,
                "stage": None,
                "stopped_at": event.stopped_at,
                "error": event.error,
                "updated_at": event.timestamp,
            }},
            upsert=True,
        )
        await self.jobs.update_one(
            {"_id": self.job_id}, {"$inc": {f"documents_{event.status}": 1}}
        )

    async def on_pipeline_completed(self, event: ProgressEvent) -> None:
        status = "success" if event.success else "completed_with_errors"
        await self.jobs.update_one(
            {"_id": self.job_id},
            {"$set": {"status": status, "finished_at": event.timestamp}},
        )

    async def on_pipeline_failed(self, event: ProgressEvent) -> None:
        await self.jobs.update_one(
            {"_id": self.job_id},
            {"$set": {"status": "failed", "error": event.error, "finished_at": event.timestamp}},
        )


db = AsyncIOMotorClient("mongodb://localhost:27017")["my_app"]
result = await run_pipeline(
    input_raw="./raw_docs",
    job_id="ingest-42",
    callbacks=[JobStatusUpdater(db, "ingest-42")],
)
```

Why a separate collection keyed by `(job_id, doc_path)` rather than one field per document inside the job (`{"documents.<doc_path>.status": ...}`)? A `doc_path` such as `reports/2024.v2.pdf` contains dots, which MongoDB reads as nested field separators, and a path may contain `$`, which is not allowed at the start of a field name. A collection with an index on `(job_id, doc_path)` also keeps the job document small however many documents the job has.

Create the motor client inside the running event loop (as above): the callback is awaited on the pipeline's loop.

### Synchronous: pymongo, SQLAlchemy, `requests`

A `ProgressCallback` runs in its own worker thread, so blocking clients are fine:

```python
from pymongo import MongoClient
from scinr.newton import ProgressCallback, ProgressEvent


class SyncJobStatusUpdater(ProgressCallback):
    def __init__(self, uri: str, job_id: str):
        self.jobs = MongoClient(uri, serverSelectionTimeoutMS=5000)["my_app"]["ingestion_jobs"]
        self.job_id = job_id

    def on_document_finished(self, event: ProgressEvent) -> None:
        self.jobs.update_one(
            {"_id": self.job_id}, {"$inc": {f"documents_{event.status}": 1}}, upsert=True
        )

    def on_pipeline_completed(self, event: ProgressEvent) -> None:
        status = "success" if event.success else "completed_with_errors"
        self.jobs.update_one({"_id": self.job_id}, {"$set": {"status": status}}, upsert=True)

    def on_pipeline_failed(self, event: ProgressEvent) -> None:
        self.jobs.update_one(
            {"_id": self.job_id}, {"$set": {"status": "failed", "error": event.error}}, upsert=True
        )
```

The same shape works with a SQLAlchemy `Session` (one short transaction per event) or an HTTP client.

### Forwarding every event

Override `on_event` to forward events as they are — to a message queue, a websocket, a log:

```python
import json
from scinr.newton import AsyncProgressCallback, ProgressEvent


class WebsocketForwarder(AsyncProgressCallback):
    def __init__(self, websocket):
        self.websocket = websocket

    async def on_event(self, event: ProgressEvent) -> None:
        await self.websocket.send_text(json.dumps(event.to_dict()))
```

### Combining callbacks

`callbacks` takes any number of callbacks of the three types. They receive the events in the order of the list:

```python
tracker = ProgressTracker()
result = await run_pipeline(
    input_raw="./raw_docs",
    job_id="ingest-42",
    callbacks=[tracker, JobStatusUpdater(db, "ingest-42")],
)
```

Anything else in the list (a plain function, a lambda, an object that does not inherit from one of the three classes) raises `TypeError` before the run starts, without emitting any event.

---

## Sync vs async callbacks

| | `AsyncProgressCallback` | `ProgressCallback` |
|---|---|---|
| Runs | awaited on the pipeline's event loop | in a dedicated worker thread (one per `run_pipeline()` call) |
| Good for | async clients (motor, httpx, async Neo4j driver, websockets) | blocking clients (pymongo, SQLAlchemy, `requests`), CPU work |
| If it blocks | **the whole event loop freezes — and with it the ingestion** | only the progress thread waits; the pipeline is not affected |
| If it is slow (but does not block) | the pipeline goes on; events queue up | the pipeline goes on; events queue up |

The rule: **if your code blocks, write it as a `ProgressCallback`.** An `async def` method that calls `requests.post()`, `pymongo` or `time.sleep()` is blocking code in disguise and stops every document of the run while it waits.

---

## Slow callbacks and shutdown

- The pipeline never waits for callbacks: the queue is unbounded. A slow callback only means that the state you write arrives later than it happens.
- When the run is over, `run_pipeline()` waits **up to 30 seconds** for the queue to drain, so the terminal event reaches your callbacks before the function returns. Past that, the remaining events are dropped and a warning is logged.
- A synchronous callback stuck in a call cannot be interrupted: its worker thread stays alive until the call returns (worker threads are not daemon threads, so this can also delay interpreter shutdown).

Recommendations:

- Put timeouts on the clients your callbacks use (database, HTTP), so a call cannot hang forever.
- Write less often: for example, update the job on `document_finished` and the terminal events only, not on every `stage_*` event, or batch writes.
- Or use a `ProgressTracker` and copy its `snapshot()` to your store from a periodic task of your own.

---

## Error handling

An exception raised by a callback is logged with its traceback on the `scinr.newton.progress` logger and ignored: the other callbacks still receive the event, the callback still receives the next events, and the ingestion is never aborted by a callback. If you need to react to a failure of your own code (retry, alert), catch it inside the callback.

The events tell you what is happening; the `PipelineResult` remains the authoritative outcome. When the run ends, reconcile your job with it if a callback may have missed writes (for example after a database outage).

---

## Limitations

- **In-process only.** Events are delivered inside the process that runs `run_pipeline()`. Nothing is persisted by the library.
- **A process that dies sends no terminal event.** If the worker is killed, the job stays `running` in your database. Detect it with your own heartbeat or timeout (for example, the time of the last update).
- **Only `run_pipeline()` emits events.** The standalone stage functions (`run_preprocess()`, `run_extraction()`, `run_annotation()`, …) do not.
- **Sharing an instance between concurrent calls.** A callback never runs concurrently with itself within one `run_pipeline()` call, but the same instance passed to two concurrent calls receives both streams and can run in parallel with itself (sync callbacks run in a different thread for each call). Use one instance per call, or make it thread-safe.

---

## See Also

- **[Running the Pipeline](running-pipeline.md)** — Full `run_pipeline()` reference, including `callbacks`.
- **[Performance Tuning](performance-tuning.md)** — Using live progress to find where documents pile up.
- **[Progress API](../api/progress.md)** — Auto-generated reference for `ProgressEvent`, the callback classes and `ProgressTracker`.
