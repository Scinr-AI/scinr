# Running the Pipeline

This is the definitive reference for `run_pipeline()` — the single entry point that orchestrates the full `scinr.newton` ingestion pipeline. Every parameter, option, and workflow pattern is documented here.

---

## Quick Start

The simplest possible pipeline run — convert raw files, extract structure, ingest to Neo4j, annotate, and extract entities:

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    result = await run_pipeline(input_raw="./raw_docs")

    print(f"Success: {result.success}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")
    print(f"Stages: {result.stages_executed}")

asyncio.run(main())
```

`configure()` automatically reads `.env` via `python-dotenv`, so if your environment variables are set, you can call `configure()` with no arguments and it works.

---

## Pipeline Stages

The pipeline consists of six named stages. The default run executes Stages 0-4 in order; Stage 5 (`"tabular"`) is auto-detected from file extensions in `input_raw` and runs alongside the main pipeline when tabular files are present.

| Stage | Name | Description |
| :--- | :--- | :--- |
| 0 | `"preprocess"` | Convert raw files (PDF, DOCX, PPTX, etc.) to intermediate JSON/Markdown |
| 1 | `"extraction"` | Parse document structure into hierarchical sections via LLM |
| 2 | `"ingestion"` | Write `:Document` and `:StructureNode` nodes into Neo4j |
| 3 | `"annotation"` | LLM agent assigns extraction models to each structure node |
| 4 | `"entity_extraction"` | Extract typed Pydantic entities and write graph subgraphs |
| 5 | `"tabular"` | Process CSV/XLSX/XLS with normalization and table understanding |

> **Note:** The `"tabular"` stage cannot be combined with other stages in a `stages=` list. When you set `stages=["tabular"]`, it runs exclusively. When you omit `"tabular"` from `stages` (the default), tabular files in `input_raw` are auto-detected and processed automatically alongside the main pipeline.

---

## Full Signature

```python
async def run_pipeline(
    # ── Raw input (Stage 0 source) ────────────────────────────────────────────
    input_raw: str | None = None,

    # ── Directory params — control data flow and stage skipping ──────────────
    converter_output_dir: str | None = None,
    extraction_input_dir: str | None = None,
    extraction_output_dir: str | None = None,
    ingestion_input_dir: str | None = None,

    # ── Stage selection ───────────────────────────────────────────────────────
    stages: list[str] | None = None,

    # ── Document identity for annotation / entity_extraction only runs ────────
    document_names: list[str] | None = None,
    document_names_dir: str | None = None,

    # ── Annotation options ────────────────────────────────────────────────────
    manual: bool = False,
    model_class: str | None = None,
    only_unannotated: bool = False,
    only_unextracted: bool = False,
    context_instructions: str | None = None,

    # ── Versioning / replacement ──────────────────────────────────────────────
    update_mode: bool = False,
    replaces: str | None = None,

    # ── Parallelism ───────────────────────────────────────────────────────────
    parallel_docs: int = 5,

    # ── Behaviour on partial failure ─────────────────────────────────────────
    on_partial_failure: Literal["abort", "continue", "warn"] = "warn",

    # ── Tabular options (auto-detected from input_raw) ────────────────────────
    tabular_extensions: set[str] | None = None,
    tabular_delimiter: str | None = None,
) -> PipelineResult
```

---

## Parameter Reference

### `input_raw` — Raw Input Directory

**Type:** `str | None`  **Default:** `None`

Path to a folder containing raw source files (PDF, DOCX, PPTX, XLSX, CSV, HTML, JSON, TXT, MD). This activates Stage 0 (`"preprocess"`), which converts all supported files into an intermediate representation.

```python
result = await run_pipeline(input_raw="./raw_docs")
```

Tabular files (`.csv`, `.xlsx`, `.xls`) found in this directory are automatically routed to the tabular pipeline in addition to the standard Stages 0-4. See [Tabular Options](#tabular_extensions-and-tabular_delimiter-tabular-options) for customizing this behavior.

> **Mutual exclusion:** `input_raw` cannot be used together with `extraction_input_dir` or `ingestion_input_dir`. These parameters represent different entry points into the pipeline.

### Directory Parameters — Intermediate Data Flow

These four parameters control how data flows between stages and allow you to skip stages by providing pre-computed intermediate files.

#### `converter_output_dir`

**Type:** `str | None`  **Default:** `None` (in-memory only)

Folder where Stage 0 (`"preprocess"`) writes intermediate JSON files to disk. When `None`, converted documents are kept in memory only and a temporary directory is used internally (cleaned up automatically).

Set this when you want to persist the Stage 0 output for reuse in a later run:

```python
# First run: convert and persist intermediate JSON
result = await run_pipeline(
    input_raw="./raw_docs",
    converter_output_dir="./data/converted/",
    stages=["preprocess"],
)

# Later run: skip Stage 0, read from persisted JSON
result = await run_pipeline(
    converter_output_dir="./data/converted/",
    stages=["extraction", "ingestion"],
)
```

#### `extraction_input_dir`

**Type:** `str | None`  **Default:** `None`

Folder where Stage 1 (`"extraction"`) reads JSON input from disk, **skipping Stage 0 entirely**. When provided, the pipeline starts directly at extraction using the files found in this directory.

> **Precedence:** `extraction_input_dir` takes absolute priority over `document_names` and `document_names_dir` for document discovery. If both are provided, `document_names` / `document_names_dir` are silently ignored. This is intentional — the directory contents define the document set.

```python
# Skip Stage 0; start extraction from pre-converted JSON
result = await run_pipeline(
    extraction_input_dir="./data/converted/",
    stages=["extraction", "ingestion", "annotation", "entity_extraction"],
)
```

#### `extraction_output_dir`

**Type:** `str | None`  **Default:** `None` (in-memory only)

Folder where Stage 1 (`"extraction"`) writes `extract-*.json` output files. When `None`, extracted documents are kept in memory only.

Useful for persisting Stage 1 output so Stage 2 can be run independently later:

```python
# First run: extract and persist
result = await run_pipeline(
    input_raw="./raw_docs",
    extraction_output_dir="./data/extracted/",
    stages=["preprocess", "extraction"],
)

# Later run: ingest from persisted extraction output
result = await run_pipeline(
    extraction_output_dir="./data/extracted/",
    stages=["ingestion"],
)
```

#### `ingestion_input_dir`

**Type:** `str | None`  **Default:** `None`

Folder where Stage 2 (`"ingestion"`) reads `extract-*.json` files from disk, **skipping both Stages 0 and 1**. When provided, the pipeline starts directly at ingestion.

> **Precedence:** Same absolute-priority rule as `extraction_input_dir`. Takes precedence over `document_names` / `document_names_dir` regardless of which stages are requested.

```python
# Skip Stages 0 and 1; start directly at ingestion
result = await run_pipeline(
    ingestion_input_dir="./data/extracted/",
    stages=["ingestion", "annotation", "entity_extraction"],
)
```

#### Directory Parameter Precedence Summary

The directory parameters are mutually exclusive with each other and with `input_raw`:

| Parameter | Skips | Input for |
| :--- | :--- | :--- |
| `input_raw` | *(none)* | Stage 0 |
| `extraction_input_dir` | Stage 0 | Stage 1 |
| `ingestion_input_dir` | Stages 0, 1 | Stage 2 |

You can combine `converter_output_dir` and `extraction_output_dir` with other parameters to control where intermediate data is written:

```python
# Full pipeline with all intermediate data persisted to disk
result = await run_pipeline(
    input_raw="./raw_docs",
    converter_output_dir="./data/converted/",
    extraction_output_dir="./data/extracted/",
)
```

### `stages` — Stage Selection

**Type:** `list[str] | None`  **Default:** `["preprocess", "extraction", "ingestion", "annotation", "entity_extraction"]`

Ordered list of stage names to execute. Omitting stages skips them entirely. The default runs Stages 0-4.

```python
# Only preprocess + extraction (Stages 0-1)
result = await run_pipeline(
    input_raw="./raw_docs",
    stages=["preprocess", "extraction"],
)

# Only annotation + entity extraction (Stages 3-4)
result = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names=["my_document.pdf"],
)

# Tabular-only (Stage 5)
result = await run_pipeline(
    input_raw="./data",
    stages=["tabular"],
)
```

> **Important:** `"tabular"` cannot be combined with other stages. Use `stages=["tabular"]` alone, or omit `"tabular"` from `stages` and let the pipeline auto-detect tabular files from `input_raw`.

When running annotation or entity extraction without ingestion, you must provide document names via `document_names` or `document_names_dir` (see [Document Selection](#document_names-and-document_names_dir-document-selection)).

### `document_names` and `document_names_dir` — Document Selection

**Type:** `list[str] | None` / `str | None`  **Default:** `None`

These parameters select which documents to process when running annotation (`"annotation"`) or entity extraction (`"entity_extraction"`) without running ingestion first.

#### `document_names`

An explicit list of Neo4j `document_name` values. The pipeline looks up each document in Neo4j and processes only those documents.

```python
# Run annotation on specific documents
result = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names=["Clinical_Trial_Report_2024", "Safety_Summary_Q3"],
)
```

#### `document_names_dir`

A directory containing `extract-*.json` files. The pipeline extracts document names from these files and processes the corresponding documents in Neo4j.

```python
# Derive document names from extraction JSON files
result = await run_pipeline(
    stages=["annotation"],
    document_names_dir="./data/extracted/",
)
```

> **Mutual exclusion:** `document_names` and `document_names_dir` cannot both be provided. Use one or the other.
>
> **Precedence:** Both are silently ignored if `extraction_input_dir` or `ingestion_input_dir` is also provided — the directory parameters take absolute priority for document discovery.

### `manual` and `model_class` — Manual Annotation Mode

**Type:** `bool` / `str | None`  **Default:** `False` / `None`

When `manual=True`, Stage 3 (`"annotation"`) assigns `model_class` to **all** structure nodes without making any LLM calls. This forces a specific extraction model on every node in the selected documents.

```python
# Force a specific model on all structure nodes
result = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names=["my_document.pdf"],
    manual=True,
    model_class="CompoundAssayResult",
)
```

> **Validation rules:**
> - `manual=True` requires `model_class` to be set (and vice versa).
> - `model_class` requires `manual=True`.
> - `manual=True` is only valid when `"annotation"` is in `stages`.

Use this when you already know which extraction model applies to a document and want to skip the LLM annotation step entirely. This is significantly faster than the default annotation mode.

### `only_unannotated` — Skip Already-Annotated Nodes

**Type:** `bool`  **Default:** `False`

When `True`, Stage 3 (`"annotation"`) skips structure nodes that already have an annotation decision (a `model_class` property set in Neo4j). This is the primary mechanism for resuming an interrupted annotation run.

```python
# First run: annotate all nodes
result = await run_pipeline(
    stages=["annotation"],
    document_names=["large_document.pdf"],
)
# ... run is interrupted after annotating 40 of 120 nodes ...

# Resume: only annotate the remaining 80 nodes
result = await run_pipeline(
    stages=["annotation"],
    document_names=["large_document.pdf"],
    only_unannotated=True,
)
```

### `only_unextracted` — Skip Already-Extracted Nodes

**Type:** `bool`  **Default:** `False`

When `True`, Stage 4 (`"entity_extraction"`) skips structure nodes that already have extracted entities connected in the graph. Use this to resume an interrupted extraction run:

```python
# Resume extraction after an interruption
result = await run_pipeline(
    stages=["entity_extraction"],
    document_names=["large_document.pdf"],
    only_unextracted=True,
)
```

### `context_instructions` — Custom LLM Instructions

**Type:** `str | None`  **Default:** `None`

Free-text instructions injected into both the converter prompts (Stage 0) and the annotation prompts (Stage 3). Use this to add domain-specific guidance for the LLM.

```python
result = await run_pipeline(
    input_raw="./raw_docs",
    context_instructions=(
        "Focus on extracting clinical trial data. "
        "Pay special attention to adverse events and dosage information. "
        "When a table contains numerical data, preserve the exact values "
        "and units of measurement."
    ),
)
```

This parameter is forwarded to every document unit processed by the pipeline. It is particularly useful when processing documents from a specific domain where the default prompts need additional context.

### `update_mode` — In-Place Document Update

**Type:** `bool`  **Default:** `False`

When `True`, Stage 2 (`"ingestion"`) replaces the latest version of an existing document in Neo4j **without incrementing the version number**. This is designed for single-document correction runs where you want to fix a document in place.

```python
# Re-ingest a single document, overwriting the existing version
result = await run_pipeline(
    input_raw="./corrected_docs/",
    update_mode=True,
    stages=["preprocess", "extraction", "ingestion"],
)
```

> **Constraints:**
> - `update_mode=True` is not allowed when ingesting multiple documents. It is designed for single-document correction.
> - `update_mode` and `replaces` are mutually exclusive — they cannot be used together.

### `replaces` — Document Replacement

**Type:** `str | None`  **Default:** `None`

The `document_name` of an existing document that is superseded by the newly ingested document. After ingestion completes, the pipeline creates a replacement relationship in Neo4j linking the new document as the successor of the old one.

```python
# Ingest a new version that replaces an old document
result = await run_pipeline(
    input_raw="./new_version/",
    replaces="Clinical_Trial_Report_2024_v1",
)
```

The pipeline performs a pre-flight check before ingestion to verify that the document named in `replaces` actually exists in Neo4j. If it does not exist, the pipeline raises an error before processing any documents.

> **Mutual exclusion:** `replaces` and `update_mode` cannot be used together. `update_mode` fixes the current version in-place; `replaces` creates a new version linked as the successor.

### `parallel_docs` — Document-Level Parallelism

**Type:** `int`  **Default:** `5`

Maximum number of documents processed concurrently across all stages. The pipeline uses an `asyncio.Semaphore` to bound concurrency at this level.

```python
# Process 10 documents concurrently
result = await run_pipeline(
    input_raw="./raw_docs",
    parallel_docs=10,
)

# Process one document at a time (sequential)
result = await run_pipeline(
    input_raw="./raw_docs",
    parallel_docs=1,
)
```

Each document unit is bounded by this semaphore for its entire duration across all stages. Within each stage, additional concurrency control is provided by `llm_concurrency` and `neo4j_concurrency` (configured via `configure()`).

> **Default is 5**, not 1. The pipeline processes up to 5 documents concurrently by default.

### `fast_extraction` — Parallel Stage 1 Extraction (opt-in)

**Type:** `bool`  **Default:** `False`

Controls whether Stage 1 extraction runs its sliding-window chunks sequentially
(default) or in parallel with a deferred, single-call consolidation step. Resolved
once per `run_pipeline()` call and passed explicitly through every layer down to
Stage 1 — never read from global config — so that concurrent `run_pipeline()` calls
with different values never interfere with each other.

```python
# Opt into parallel Stage 1 extraction
result = await run_pipeline(
    input_raw="./raw_docs",
    stages=["preprocess", "extraction", "ingestion"],
    fast_extraction=True,
)
```

Raises `ValueError` if `fast_extraction=True` is passed while `"extraction"` is not
included in `stages` — the flag has no effect without Stage 1 running.

See [Performance Tuning: Fast Extraction](performance-tuning.md#fast-extraction-fast_extraction)
for the full trade-off and risk explanation before enabling this in production.

### `on_partial_failure` — Error Handling Strategy

**Type:** `Literal["abort", "continue", "warn"]`  **Default:** `"warn"`

Controls behavior when a stage reports partial failures (`nodes_failed > 0`). The pipeline **never** stops processing other documents — every document in the batch runs independently. This parameter only affects whether a **single document** continues to its remaining stages after a partial failure.

#### `"abort"`

Stops the document from advancing to its remaining stages after a partial failure in annotation or entity extraction. This is the strictest mode.

```python
# Abort a document's remaining stages on first failure
result = await run_pipeline(
    input_raw="./raw_docs",
    on_partial_failure="abort",
)
```

#### `"continue"`

The document keeps advancing to its next requested stage even if some nodes failed in the previous one. Completely silent — no warnings are logged.

```python
# Continue processing despite failures, no warnings
result = await run_pipeline(
    input_raw="./raw_docs",
    on_partial_failure="continue",
)
```

#### `"warn"` (default)

Behaves like `"continue"` (the document keeps advancing) but additionally logs warnings at two levels:

1. **Immediately:** A per-document warning is emitted the moment a specific document decides to keep advancing despite a partial failure — naming the document, the stage, the failed-node count, and the concrete errors.
2. **At the end:** An aggregated per-stage warning fires whenever a stage reports one or more failed documents overall.

```python
# Continue with detailed logging (default behavior)
result = await run_pipeline(
    input_raw="./raw_docs",
    on_partial_failure="warn",
)
```

#### Stage-Specific Behavior

The effect of `on_partial_failure` depends on which stage failed:

| Failed Stage | Effect of `on_partial_failure` |
| :--- | :--- |
| `"preprocess"` | **Always** stops the document — no valid artifact exists for subsequent stages. |
| `"extraction"` | **Always** stops the document — no valid document object for subsequent stages. |
| `"ingestion"` | **Always** stops the document — no valid Neo4j document for subsequent stages. |
| `"annotation"` | Partial failure (`nodes_failed > 0`). `"abort"` stops the document; `"continue"` / `"warn"` let it advance. |
| `"entity_extraction"` | Partial failure (`nodes_failed > 0`). `"abort"` stops the document; `"continue"` / `"warn"` let it advance. |

Stages 0-2 (preprocess, extraction, ingestion) are **total** failures for a document — there is nothing valid to continue with. Only Stages 3-4 (annotation, entity_extraction) are **partial** failures where `on_partial_failure` has an effect.

### `tabular_extensions` and `tabular_delimiter` — Tabular Options

**Type:** `set[str] | None` / `str | None`  **Default:** `{".csv", ".xlsx", ".xls"}` / `None`

#### `tabular_extensions`

File extensions to process via the tabular pipeline. Files with these extensions found in `input_raw` are automatically routed to the tabular pipeline alongside the standard Stages 0-4.

```python
# Include .dat files as tabular data
result = await run_pipeline(
    input_raw="./data",
    tabular_extensions={".csv", ".tsv", ".dat", ".xlsx"},
)
```

#### `tabular_delimiter`

Delimiter character for CSV tabular files. When `None`, the pipeline auto-detects the delimiter (`,` , `;`, `\t`, `|`).

```python
# Force tab-delimited CSV processing
result = await run_pipeline(
    input_raw="./data",
    tabular_delimiter="\t",
)
```

Both parameters are also forwarded to the tabular pipeline when it runs as an auto-detected sidecar alongside the main pipeline.

---

## Complete Workflows

### 1. First-Time Full Ingestion

The canonical first run — convert raw files, extract structure, ingest to Neo4j, annotate, and extract entities.

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    result = await run_pipeline(input_raw="./raw_docs")

    # Inspect results
    print(f"Success: {result.success}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")
    print(f"Stages: {result.stages_executed}")

    for stage_name in result.stages_executed:
        stage = getattr(result, stage_name)
        if stage:
            print(f"  {stage_name}: "
                  f"{stage.total_processed} processed, "
                  f"{stage.total_failed} failed")

asyncio.run(main())
```

### 2. Re-Run Annotation Only

Re-run annotation and entity extraction on documents already in Neo4j, without touching Stages 0-2.

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Re-annotate and re-extract entities for specific documents
    result = await run_pipeline(
        stages=["annotation", "entity_extraction"],
        document_names=["Clinical_Trial_Report_2024"],
        only_unannotated=True,
        only_unextracted=True,
    )

    print(f"Annotation: {result.annotation.total_processed} nodes")
    print(f"Extraction: {result.entity_extraction.total_processed} nodes")

asyncio.run(main())
```

### 3. Add New Documents to Existing Graph

Ingest new documents into a Neo4j graph that already contains previously ingested documents.

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # New documents in a separate folder
    result = await run_pipeline(
        input_raw="./new_batch/",
        parallel_docs=3,
    )

    print(f"New batch ingested: {result.success}")
    if result.ingestion:
        for doc in result.ingestion.documents:
            print(f"  {doc.document_name}: "
                  f"{doc.nodes_processed} processed")

asyncio.run(main())
```

### 4. Replace a Document with Updated Version

A document has been corrected or updated. Replace it in the graph while maintaining a link to the old version.

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Ingest the new version, linking it as the replacement
    result = await run_pipeline(
        input_raw="./corrected/",
        replaces="Clinical_Trial_Report_2024_v1",
    )

    print(f"Replacement ingested: {result.success}")
    if result.ingestion:
        for doc in result.ingestion.documents:
            print(f"  New document: {doc.document_name}")

asyncio.run(main())
```

### 5. Process Tabular Data Only

Process only CSV/XLSX/XLS files without running the standard document pipeline.

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Tabular-only pipeline
    result = await run_pipeline(
        input_raw="./tabular_data/",
        stages=["tabular"],
        tabular_extensions={".csv", ".xlsx"},
        tabular_delimiter=",",
    )

    print(f"Tabular pipeline: {result.success}")
    if result.tabular:
        print(f"  {result.tabular.total_processed} files processed")

asyncio.run(main())
```

### 6. Manual Model Application

Force a specific extraction model on all nodes of a document, skipping the LLM annotation step entirely.

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Apply a known model to all nodes without LLM annotation
    result = await run_pipeline(
        stages=["annotation", "entity_extraction"],
        document_names=["Assay_Results_Q4.xlsx"],
        manual=True,
        model_class="CompoundAssayResult",
    )

    print(f"Manual extraction: {result.success}")
    if result.entity_extraction:
        print(f"  {result.entity_extraction.total_processed} nodes extracted")

asyncio.run(main())
```

---

## PipelineResult Inspection

`run_pipeline()` returns a `PipelineResult` dataclass with structured access to per-stage metrics.

### Overall Pipeline

```python
result = await run_pipeline(input_raw="./raw_docs")

# Overall success
print(f"Success: {result.success}")                          # bool

# Total wall-clock time
print(f"Duration: {result.total_duration_seconds:.2f}s")    # float

# Ordered list of stages that were actually executed
print(f"Stages: {result.stages_executed}")                   # list[str]
```

### Per-Stage Access

Each stage is accessible as an attribute on the result. If a stage was not executed, its attribute is `None`.

```python
result = await run_pipeline(input_raw="./raw_docs")

# StageResult attributes (or None if stage was skipped)
result.preprocess           # Stage 0
result.extraction           # Stage 1
result.ingestion            # Stage 2
result.annotation           # Stage 3
result.entity_extraction    # Stage 4
result.tabular              # Stage 5
```

### StageResult Details

Each `StageResult` contains:

```python
stage = result.ingestion

if stage:
    print(f"Stage: {stage.stage}")              # str — stage name
    print(f"Success: {stage.success}")           # bool
    print(f"Duration: {stage.duration_seconds:.2f}s")  # float
    print(f"Processed: {stage.total_processed}")  # int
    print(f"Failed: {stage.total_failed}")       # int
    print(f"Errors: {stage.errors}")             # list[str]
```

### Per-Document Details

Each `StageResult` has a `documents` list of `DocumentResult` entries:

```python
if result.ingestion:
    for doc in result.ingestion.documents:
        print(f"  {doc.document_name}: "
              f"{doc.nodes_processed} processed, "
              f"{doc.nodes_failed} failed")
        if doc.errors:
            for err in doc.errors:
                print(f"    ERROR: {err}")
```

### DocumentResult Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `document_name` | `str` | The Neo4j `document_name` (or filename stem) of the processed document. |
| `nodes_processed` | `int` | Nodes (or files) successfully processed. For Stages 0-2: 1 for success, 0 for failure. For Stages 3-4: number of structure nodes processed. |
| `nodes_failed` | `int` | Nodes (or files) that failed processing. |
| `errors` | `list[str]` | Error messages for this document. Empty on full success. |

---

## Multi-Step Workflows with Intermediate Directories

For production pipelines, you often want to split processing into separate runs with persisted intermediate data. Here is a complete multi-step workflow:

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # ── Step 1: Convert raw files to intermediate JSON ──────────────────
    r1 = await run_pipeline(
        input_raw="./raw_docs/",
        converter_output_dir="./data/converted/",
        stages=["preprocess"],
    )
    print(f"Step 1 (preprocess): {r1.success}")

    # ── Step 2: Extract structure from converted JSON ───────────────────
    r2 = await run_pipeline(
        extraction_input_dir="./data/converted/",
        extraction_output_dir="./data/extracted/",
        stages=["extraction"],
    )
    print(f"Step 2 (extraction): {r2.success}")

    # ── Step 3: Ingest extracted documents into Neo4j ───────────────────
    r3 = await run_pipeline(
        ingestion_input_dir="./data/extracted/",
        stages=["ingestion"],
    )
    print(f"Step 3 (ingestion): {r3.success}")

    # ── Step 4: Annotate and extract entities ───────────────────────────
    r4 = await run_pipeline(
        stages=["annotation", "entity_extraction"],
        document_names_dir="./data/extracted/",
        context_instructions="Focus on clinical trial data.",
    )
    print(f"Step 4 (annotation + extraction): {r4.success}")

asyncio.run(main())
```

This pattern is useful when:
- Different teams own different stages of the pipeline.
- You want to re-run a specific stage without re-processing earlier stages.
- You need to inspect intermediate data between stages.
- You want to distribute work across different machines or time slots.

---

## Error Handling

The pipeline can raise several exceptions before or during execution:

```python
import asyncio
from scinr.newton import (
    configure, run_pipeline,
    ConfigurationError, PreconditionError,
    ExtractionError, IngestionError,
)

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    try:
        result = await run_pipeline(input_raw="./raw_docs")
    except ConfigurationError as e:
        # Missing Neo4j or LLM configuration
        print(f"Configuration error: {e}")
    except PreconditionError as e:
        # Invalid parameter combination
        print(f"Precondition error: {e}")
    except ExtractionError as e:
        # Entity extraction failure
        print(f"Extraction error: {e}")
    except IngestionError as e:
        # Neo4j graph write failure
        print(f"Ingestion error: {e}")

asyncio.run(main())
```

### Parameter Validation Errors

The pipeline validates parameter combinations before execution. Invalid combinations raise `ValueError`:

| Invalid Combination | Error Message |
| :--- | :--- |
| `input_raw` + `extraction_input_dir` | Mutually exclusive — different entry points |
| `input_raw` + `ingestion_input_dir` | Mutually exclusive — different entry points |
| `extraction_input_dir` + `ingestion_input_dir` | Mutually exclusive — different entry points |
| `update_mode=True` + `replaces` | Mutually exclusive — different versioning strategies |
| `manual=True` without `model_class` | `model_class` required when `manual=True` |
| `model_class` without `manual=True` | `manual=True` required when `model_class` is set |
| `document_names` + `document_names_dir` | Mutually exclusive — use one or the other |
| `"tabular"` + other stages | `"tabular"` must be used alone |
| `parallel_docs < 1` | Must be >= 1 |
| `"preprocess"` without `input_raw` | Requires raw file input |
| `"annotation"` without document names | Requires `document_names` or `document_names_dir` (or `ingestion` in stages) |

---

## Common Patterns

### Running the Pipeline from a Script

```python
#!/usr/bin/env python
"""run_ingestion.py — Full pipeline run from command line."""

import asyncio
import sys
from pathlib import Path

from scinr.newton import configure, run_pipeline

async def run(input_dir: str, parallel: int = 5) -> None:
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    if not Path(input_dir).is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    result = await run_pipeline(
        input_raw=input_dir,
        parallel_docs=parallel,
        on_partial_failure="warn",
    )

    print(f"\nPipeline {'succeeded' if result.success else 'failed'} "
          f"in {result.total_duration_seconds:.1f}s")

    for stage_name in result.stages_executed:
        stage = getattr(result, stage_name)
        if stage:
            status = "OK" if stage.success else f"FAILED ({stage.total_failed})"
            print(f"  {stage_name}: {status}")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./raw_docs"
    parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(run(input_dir, parallel))
```

### Conditional Pipeline Based on File Types

```python
import asyncio
from pathlib import Path
from scinr.newton import configure, run_pipeline

async def smart_run(input_dir: str) -> None:
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    tabular_ext = {".csv", ".xlsx", ".xls"}
    files = list(Path(input_dir).rglob("*"))
    has_tabular = any(f.suffix.lower() in tabular_ext for f in files if f.is_file())
    has_docs = any(f.suffix.lower() not in tabular_ext for f in files if f.is_file())

    if has_docs:
        # Run full pipeline for documents (auto-detects tabular too)
        result = await run_pipeline(input_raw=input_dir)
    elif has_tabular:
        # Tabular-only
        result = await run_pipeline(
            input_raw=input_dir,
            stages=["tabular"],
        )
    else:
        print("No supported files found.")
        return

    print(f"Pipeline result: {result.success}")

asyncio.run(smart_run("./raw_docs"))
```

### Resuming a Failed Pipeline

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def resume_pipeline(document_name: str) -> None:
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Skip already-annotated and already-extracted nodes
    result = await run_pipeline(
        stages=["annotation", "entity_extraction"],
        document_names=[document_name],
        only_unannotated=True,
        only_unextracted=True,
        on_partial_failure="warn",
    )

    # Report what was skipped vs. what was processed
    if result.annotation:
        print(f"Annotation: {result.annotation.total_processed} nodes "
              f"({result.annotation.total_failed} failed)")
    if result.entity_extraction:
        print(f"Extraction: {result.entity_extraction.total_processed} nodes "
              f"({result.entity_extraction.total_failed} failed)")

asyncio.run(resume_pipeline("Large_Document.pdf"))
```

---

## See Also

- **[Configuration](../configuration.md)** — Complete reference for `configure()`, environment variables, and all settings.
- **[Architecture](../architecture.md)** — Detailed walkthrough of each pipeline stage and data flow.
- **[Custom Models](custom-models.md)** — Defining domain-specific Pydantic extraction models.
- **[Tabular Pipeline](tabular-pipeline.md)** — Tabular data normalization and processing.
- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding the graph model and querying results.
- **[Pipeline API](../api/pipeline.md)** — Auto-generated docstring for `run_pipeline()`.
- **[Results API](../api/results.md)** — Auto-generated documentation for `PipelineResult`, `StageResult`, and `DocumentResult`.
