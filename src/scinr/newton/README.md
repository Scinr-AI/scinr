# scinr.newton — Technical Reference

> Part of the [scinr](../../README.md) library. `scinr.newton` is the document ingestion module. See root README for installation, quick start, and platform overview.

> **Requires Python 3.11+** and a running Neo4j 5.x instance.

## Table of Contents

1. [Public API](#public-api)
2. [Pipeline Stages — Detailed Reference](#pipeline-stages--detailed-reference)
3. [Neo4j Schema](#neo4j-schema)
4. [Storage Layer](#storage-layer)
5. [Extending the Pipeline](#extending-the-pipeline)

---

## Public API

All symbols below are importable directly from `scinr.newton`:

```python
from scinr.newton import (
    configure, get_config, get_available_themes, ThemePath,
    run_pipeline,
    run_preprocess, run_extraction, run_ingestion,
    run_annotation, run_entity_extraction, run_tabular_pipeline,
    delete_document,
    DocumentResult, StageResult, PipelineResult, DeletionResult,
    ScinrError, ConfigurationError, PreconditionError,
    ExtractionError, IngestionError, ModelError, StorageError, ConversionError,
)
```

---

### `configure()`

**Module:** `scinr.newton.config`

Configure the library before using any pipeline function. Must be called once at startup. Parameter resolution order: explicit argument → environment variable → hard-coded default.

```python
from langchain_openai import ChatOpenAI
from scinr.newton import configure

configure(
    llm=ChatOpenAI(model="gpt-4o"),
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="secret",
)
```

#### Parameter Table

| Parameter | Type | Env Var | Default | Description |
|---|---|---|---|---|
| `llm` | `BaseChatModel` | `MODEL_ID` | — | LangChain chat model for all LLM calls. If `None`, falls back to `ChatBedrockConverse` using `MODEL_ID`. Required. |
| `repair_llm` | `BaseChatModel` | — | falls back to `llm` | Separate model for JSON repair loop. Recommended: a smaller/cheaper model. |
| `neo4j_uri` | `str` | `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI. |
| `neo4j_user` | `str` | `NEO4J_USER` or `NEO4J_AUTH` | — | Neo4j username. Required. Also parsed from `NEO4J_AUTH=user/password`. |
| `neo4j_password` | `str` | `NEO4J_PASSWORD` or `NEO4J_AUTH` | — | Neo4j password. Required. Also parsed from `NEO4J_AUTH=user/password`. |
| `enabled_base_themes` | `list[ThemePath \| str] \| None` | — | `None` (all) | Whitelist of built-in theme paths to activate. `None` activates all. |
| `enabled_user_themes` | `list[str] \| None` | — | `None` (all) | Whitelist of user-defined theme paths to activate. `None` activates all. |
| `extra_models_paths` | `list[str \| Path] \| None` | `SCINR_EXTRA_MODELS_PATHS` | `[]` | Filesystem directories scanned for additional user themes. Colon-separated in env var. |
| `storage_backend` | `Literal["none", "mongodb", "custom"]` | `STORAGE_BACKEND` | `"none"` | Storage backend. `"none"` silently skips all storage calls. |
| `mongodb_uri` | `str` | `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection URI. |
| `mongodb_database` | `str` | `MONGODB_DATABASE` | `"scinr"` | MongoDB database name. |
| `mongodb_raw_files_collection` | `str` | `MONGODB_RAW_FILES_COLLECTION` | `"raw_files"` | Collection for raw file metadata. |
| `mongodb_pages_collection` | `str` | `MONGODB_PAGES_COLLECTION` | `"converted_pages"` | Collection for converted page content. |
| `mongodb_gridfs_bucket` | `str` | `MONGODB_GRIDFS_BUCKET` | `"raw_binaries"` | GridFS bucket for raw binary files. |
| `custom_storage` | `tuple \| None` | — | `None` | `(RawFileRepository, PageRepository)` when `storage_backend="custom"`. |
| `extra_converters` | `dict[str, type] \| None` | — | `{}` | Maps file extensions to `BaseConverter` subclasses, overriding built-in converters. |
| `mistral_api_key` | `str \| None` | `MISTRAL_API_KEY` | `None` | Mistral API key for PDF OCR conversion. |
| `prompt_caching_enabled` | `bool \| None` | `PROMPT_CACHING_ENABLED` | `True` | Enable Bedrock Converse prompt caching (~90% token cost reduction on repeated calls). |
| `full_docstring` | `bool \| None` | `FULL_DOCSTRING` | `True` | Use the full class docstring (vs. only its first line) when building the model catalog description for LLM prompts (annotation stage) and Neo4j `CatalogModel.description`. |
| `extraction_batch_size` | `int \| None` | `EXTRACTION_BATCH_SIZE` | `1` | Pages processed per extraction chunk (sliding window step). |
| `llm_concurrency` | `int \| None` | `LLM_CONCURRENCY` | `4` | Max concurrent LLM calls (asyncio semaphore size). |
| `neo4j_concurrency` | `int \| None` | `NEO4J_CONCURRENCY` | `10` | Max concurrent Neo4j session writes during annotation and entity extraction. |
| `log_level` | `str` | — | `"INFO"` | Logging level string. |

**Returns:** `ScinrConfig` — the populated configuration object (also stored as module-level singleton).

**Raises:** `ConfigurationError` — if `llm` is not set and `MODEL_ID` is absent, if Neo4j credentials are missing, or if `storage_backend` is invalid.

---

### `run_pipeline()`

**Module:** `scinr.newton.pipeline`

Async function. Orchestrates the full scinr.newton pipeline end-to-end, chaining Stages 0–4 in sequence. Passes data between stages in memory when intermediate directory parameters are omitted. Tabular files (`.csv`, `.xlsx`, `.xls`) found in `input_raw` are automatically routed to the tabular pipeline.

```python
import asyncio
from scinr.newton import configure, run_pipeline

configure(llm=my_llm, neo4j_user="neo4j", neo4j_password="secret")
result = asyncio.run(run_pipeline(input_raw="files/"))
print(result.success, result.total_duration_seconds)
```

#### Parameter Table

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_raw` | `str \| None` | `None` | Folder of raw source files for Stage 0. Required when `stages` includes `"preprocess"`. Mutually exclusive with `extraction_input_dir` and `ingestion_input_dir`. |
| `converter_output_dir` | `str \| None` | `None` | Where Stage 0 writes intermediate JSON to disk. When `None`, Stage 0 returns documents in memory only. |
| `extraction_input_dir` | `str \| None` | `None` | Where Stage 1 reads JSON input from disk, skipping Stage 0. Mutually exclusive with `input_raw`. |
| `extraction_output_dir` | `str \| None` | `None` | Where Stage 1 writes `extract-*.json` output to disk. `None` keeps output in memory. |
| `ingestion_input_dir` | `str \| None` | `None` | Where Stage 2 reads `extract-*.json` from disk, skipping Stages 0 and 1. Mutually exclusive with `input_raw` and `extraction_input_dir`. |
| `stages` | `list[str] \| None` | `None` (all 5) | Ordered list of stages to run. Valid values: `"preprocess"`, `"extraction"`, `"ingestion"`, `"annotation"`, `"entity_extraction"`, `"tabular"`. `None` runs the full pipeline. `"tabular"` cannot be combined with other stages. |
| `document_names` | `list[str] \| None` | `None` | Explicit list of Neo4j `document_name` values for Stage 3/4 when running without Stage 2. Mutually exclusive with `document_names_dir`. |
| `document_names_dir` | `str \| None` | `None` | Folder containing `extract-*.json` files whose `document_name` fields are used for Stage 3/4. Mutually exclusive with `document_names`. |
| `manual` | `bool` | `False` | If `True`, Stage 3 assigns `model_class` to all qualifying nodes without LLM. Requires `model_class`. |
| `model_class` | `str \| None` | `None` | CamelCase model class name for manual annotation. Required when `manual=True`. |
| `only_unannotated` | `bool` | `False` | Stage 3 only: skip nodes already having a `:HAS_MODEL_DECISION` relationship. Useful for resuming. |
| `only_unextracted` | `bool` | `False` | Stage 4 only: skip nodes already having a `:HAS_EXTRACTION` relationship. Useful for resuming. |
| `context_instructions` | `str \| None` | `None` | Free-text context injected into Stage 0 (converter) and Stage 3 (annotation agent). |
| `update_mode` | `bool` | `False` | If `True`, Stage 2 replaces the latest document version without creating a new one. Only one source file allowed. Mutually exclusive with `replaces`. |
| `replaces` | `str \| None` | `None` | `document_name` of an existing document superseded by newly ingested content. Verified in Neo4j before stages run. Mutually exclusive with `update_mode`. |
| `parallel_docs` | `int` | `1` | Maximum documents processed concurrently across all stages, including tabular. Must be ≥ 1. |
| `on_partial_failure` | `Literal["abort", "continue", "warn"]` | `"abort"` | Behaviour when a stage returns `success=False`. `"abort"`: stop immediately. `"continue"`: proceed. `"warn"`: log warning and proceed. |
| `tabular_extensions` | `set[str] \| None` | `None` | File extensions treated as tabular. Defaults to `{'.csv', '.xlsx', '.xls'}`. |
| `tabular_delimiter` | `str \| None` | `None` | CSV field delimiter forwarded to the tabular agent. Uses agent default when `None`. |

**Returns:** `PipelineResult`

**Raises:** `ValueError` — on invalid parameter combinations. `FileNotFoundError` — if a required directory does not exist.

---

### Stage Functions

All stage functions are async and importable from `scinr.newton`:

#### `run_preprocess(input_raw, output_dir=None, context_instructions=None)`

**Stage 0.** Converts raw source files in `input_raw` to intermediate JSON using the registered converters. When `output_dir` is `None`, intermediate documents are returned in memory only. Returns `tuple[StageResult, list[IntermediateDocument]]`. With `parallel_docs > 1`, conversions run with real parallelism — dispatched to a worker thread for sync converters or awaited natively for async converters (e.g. `PdfConverter`) — instead of blocking the event loop.

#### `run_extraction(input_folder=None, output_folder=None, intermediate_documents=None, parallel_docs=1)`

**Stage 1.** Reads intermediate JSON (from `input_folder` on disk or from `intermediate_documents` in memory) and runs LLM extraction to produce `Document` objects. Exactly one of `input_folder` or `intermediate_documents` must be provided. Returns `tuple[StageResult, list[Document]]`.

#### `run_ingestion(output_folder=None, files=None, documents=None, update_mode=False)`

**Stage 2.** Loads extracted documents into Neo4j. Accepts documents from `output_folder` (scanned for `extract-*.json`), an explicit `files` list, or in-memory `documents`. Returns `StageResult`.

#### `run_annotation(document_name, manual=False, model_class=None, parallel_docs=1, only_unannotated=False, context_instructions_override=None)`

**Stage 3.** Runs the LangGraph annotation agent over every `(:StructureNode)` in the named document. In manual mode, assigns `model_class` without LLM. Returns `StageResult`.

#### `run_entity_extraction(document_name, parallel_docs=1, only_unextracted=False)`

**Stage 4.** Runs the LangGraph entity extraction agent over annotated `(:StructureNode)` nodes in the named document. Returns `StageResult`.

#### `run_tabular_pipeline(input_raw, update_mode=False, parallel_docs=1, tabular_extensions=None, tabular_delimiter=None)`

**Tabular bypass.** Ingests CSV/XLSX/XLS files directly into Neo4j, bypassing Stages 0–4. Returns `StageResult`.

---

### Document Deletion

**Module:** `scinr.newton.ingest.deletion`

#### `delete_document(path, version=None)`

Async function. Completely removes a document from Neo4j — unlike `delete_document_content()` (an internal helper used by the `--update` in-place re-ingestion flow, which only wipes content and keeps the `:Document` node), `delete_document()` deletes the `:Document` node(s) themselves plus their entire structure, and then cleans up orphans.

```python
import asyncio
from scinr.newton import delete_document

result = asyncio.run(delete_document("ModuloA/SubModulo/doc_a"))        # deletes every version
result = asyncio.run(delete_document("ModuloA/SubModulo/doc_a", version=2))  # deletes only version 2
print(result.found, result.documents_deleted, result.structure_nodes_deleted)

# Inside an already-running event loop, use await instead:
result = await delete_document("ModuloA/SubModulo/doc_a")
```

Behavior:

1. Opens and closes its own Neo4j driver internally (via `get_driver()`) — no driver management required by the caller.
2. Read-only check: finds every `(:Document {path: $path})` matching `version` (or all versions when `version=None`). If none match, returns immediately with `found=False` and all counters at 0 — no storage cleanup, delete, or garbage-collection queries are executed.
3. **Storage cleanup (runs before any Neo4j deletion):** collects the `raw_file_id` property of every matched `:Document` and every descendant reached via `IS_COMPOSED_OF*` (skipping empty `raw_file_id` values, e.g. folders or documents ingested with `storage_backend="none"`), then deletes the corresponding records from the configured documental storage backend (see [Storage Layer](#storage-layer) below) — the converted Markdown pages first, then the raw binary + its metadata, for each `raw_file_id`. This step is **fail-fast**: if deleting storage for any `raw_file_id` raises an unexpected exception, it propagates immediately and neither the cascade delete nor the GC passes run (the Neo4j driver is still closed via the `finally` block).
4. Cascade delete (single write transaction): deletes the matched `:Document` node(s), everything reachable via `IS_COMPOSED_OF*` (folder-parent Documents, sibling documents), and every `:StructureNode` descendant (`HAS_STRUCTURE`/`HAS_CHILD`) together with its `:InfoUnit`, `:ModelDecision`, `:ProposedModel`, `:ProposedField`, and `:ExtractionResult` children.
5. Global garbage collection, run **after** the cascade delete completes: two independent passes, each re-run up to `GC_MAX_PASSES` (7) times, stopping as soon as an iteration deletes 0 nodes:
   - **Pass 1:** deletes orphaned `:Entity`/`:ModelInstance` nodes (no `:ExtractionResult` reaches them within 7 hops).
   - **Pass 2** (runs only after Pass 1 fully finishes): deletes orphaned `:LabeledEntity` nodes (no incoming relationship at all).

> **Breaking change note:** if you configure `storage_backend="custom"`, your custom `RawFileRepository`/`PageRepository` implementations must now also implement `delete(raw_file_id)` / `delete_pages(raw_file_id)` respectively (see [Storage Layer](#storage-layer)) — these are new abstract methods on the base interfaces.

**Returns:** `DeletionResult`

---

### Result Types

**Module:** `scinr.newton.results`

#### `DocumentResult`

Result of processing a single document through a pipeline stage.

| Field | Type | Description |
|---|---|---|
| `document_name` | `str` | Neo4j `document_name` (or filename stem) of the processed document. |
| `nodes_processed` | `int` | Successfully processed nodes. For Stages 0–2: `1` on success, `0` on failure. For Stages 3–4: number of `StructureNode`s processed. |
| `nodes_failed` | `int` | Number of nodes that failed processing. |
| `errors` | `list[str]` | Error messages for this document. Empty on full success. |

#### `StageResult`

Aggregated result of running a single pipeline stage.

| Field | Type | Description |
|---|---|---|
| `stage` | `str` | Stage identifier: `"preprocess"`, `"extraction"`, `"ingestion"`, `"annotation"`, `"entity_extraction"`, or `"tabular"`. |
| `success` | `bool` | `True` if `total_failed == 0` and no global errors occurred. |
| `documents` | `list[DocumentResult]` | Per-document results, one per file or document processed. |
| `total_processed` | `int` | Sum of `nodes_processed` across all `DocumentResult` entries. |
| `total_failed` | `int` | Sum of `nodes_failed` across all `DocumentResult` entries. |
| `duration_seconds` | `float` | Wall-clock time in seconds for the entire stage. |
| `errors` | `list[str]` | Global stage-level errors not attributable to a specific document. |

#### `PipelineResult`

Aggregated result of a full `run_pipeline()` invocation.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | `True` only if every executed stage succeeded. |
| `total_duration_seconds` | `float` | Total wall-clock time for the entire pipeline run. |
| `stages_executed` | `list[str]` | Ordered list of stages actually run (skipped stages excluded). |
| `preprocess` | `StageResult \| None` | Stage 0 result, or `None` if not executed. |
| `extraction` | `StageResult \| None` | Stage 1 result, or `None` if not executed. |
| `ingestion` | `StageResult \| None` | Stage 2 result, or `None` if not executed. |
| `annotation` | `StageResult \| None` | Stage 3 result, or `None` if not executed. |
| `entity_extraction` | `StageResult \| None` | Stage 4 result, or `None` if not executed. |
| `tabular` | `StageResult \| None` | Tabular pipeline result, or `None` if not executed. |

#### `DeletionResult`

Result of a `delete_document()` call — full Document + cascade + garbage-collection deletion.

| Field | Type | Description |
|---|---|---|
| `path` | `str` | The Document `path` that was targeted for deletion. |
| `version` | `int \| None` | The specific version requested, or `None` if all versions were targeted. |
| `found` | `bool` | `True` if at least one matching Document existed before deletion. When `False`, all counters below are 0 and no delete or GC queries were executed. |
| `versions_deleted` | `list[int]` | Sorted list of integer versions that matched and were deleted. Empty when `found` is `False`. |
| `documents_deleted` | `int` | Number of `:Document` nodes deleted (the matched Document(s) plus any reached via `IS_COMPOSED_OF*`). |
| `structure_nodes_deleted` | `int` | Number of `:StructureNode` nodes deleted. |
| `info_units_deleted` | `int` | Number of `:InfoUnit` nodes deleted. |
| `model_decisions_deleted` | `int` | Number of `:ModelDecision` nodes deleted. |
| `proposed_models_deleted` | `int` | Number of `:ProposedModel` nodes deleted. |
| `proposed_fields_deleted` | `int` | Number of `:ProposedField` nodes deleted. |
| `extraction_results_deleted` | `int` | Number of `:ExtractionResult` nodes deleted. |
| `gc_entity_model_instance_deleted` | `int` | Total `:Entity`/`:ModelInstance` nodes deleted across all GC iterations. |
| `gc_entity_model_instance_passes` | `int` | Number of GC iterations actually run for the Entity/ModelInstance pass (capped at `GC_MAX_PASSES`). |
| `gc_labeled_entity_deleted` | `int` | Total `:LabeledEntity` nodes deleted across all GC iterations. |
| `gc_labeled_entity_passes` | `int` | Number of GC iterations actually run for the LabeledEntity pass (capped at `GC_MAX_PASSES`). |
| `raw_files_deleted` | `int` | Number of `RawFileRecord` (binaries) deleted from the storage layer for the `raw_file_id`s referenced by the deleted Document(s) and their descendants. |
| `converted_pages_deleted` | `int` | Number of `ConvertedPageRecord` (converted Markdown pages) deleted from the storage layer for the same `raw_file_id`s. |

---

### Exceptions

**Module:** `scinr.newton.exceptions`

All exceptions inherit from `ScinrError`, so the entire family can be caught with a single `except ScinrError` clause.

```
ScinrError (base)
├── ConfigurationError   — library misconfigured (LLM missing, bad credentials, invalid storage backend)
├── PreconditionError    — pipeline called out of order (e.g. entity extraction before annotation)
├── ExtractionError      — LLM extraction failed after all retries exhausted
├── IngestionError       — Neo4j write failed (version conflict, schema constraint violation)
├── ModelError           — Pydantic model cannot be resolved or is invalid (bad catalog.py)
├── StorageError         — MongoDB unavailable or misconfigured
└── ConversionError      — file converter failed to process a source file
```

---

## Pipeline Stages — Detailed Reference

### Stage 0 — Preprocess

**Directory:** `src/scinr/newton/converters/`  
**Input:** Raw source files  
**Output:** `data/json/{filename}.json`

Normalises every supported file format into a uniform intermediate JSON envelope consumed by Stage 1:

```json
{
  "pages": [
    { "index": 0, "markdown": "# Title\n\nContent of page 1…", "page_id": "…" },
    { "index": 1, "markdown": "Content of page 2…", "page_id": "…" }
  ],
  "folder_path": "optional/subfolder",
  "raw_file_id": "…"
}
```

| Format | Library / API |
|---|---|
| PDF | Mistral OCR API (returns markdown with tables + images) |
| DOCX | python-docx |
| XLSX / XLS | openpyxl + pandas |
| PPTX | python-pptx |
| CSV | pandas |
| HTML | BeautifulSoup4 |
| TXT / MD | plain read |
| JSON API | httpx + jsonpath-ng |
| XML / SOAP API | lxml + XPath |

Converters are registered in `src/scinr/newton/converters/registry.py` (keyed by file extension), so the pipeline auto-selects the right converter for each file. When a storage backend is configured, Stage 0 also stores the raw binary and converted pages in MongoDB.

```bash
# CLI
newton --stage preprocess --input-raw files/ --input data/json/

# Library API
result, docs = asyncio.run(run_preprocess("files/", output_dir="data/json/"))
```

See each converter's docstring in `src/scinr/newton/converters/` for converter details and how to add new formats.

---

### Stage 1 — Extract

**Directory:** `src/scinr/newton/extraction/`  
**Input:** `data/json/{filename}.json`  
**Output:** `data/output/extract-{filename}.json`  
**LLM:** Any LangChain `BaseChatModel` — temperature `0.0`, max tokens configurable

Reads the paged JSON and processes pages through a **sliding window** (default 2 pages per chunk). For each window:

1. Sends page content plus the active document hierarchy as context to the LLM.
2. Receives back a `DocumentStructure` Pydantic tree (sections, subsections, tables, info units, verbatim quotes).
3. Merges the chunk result into the growing `Document` object.
4. **Crash-safe write**: writes the intermediate JSON to disk after every successfully processed chunk.

A **2-phase extraction + repair loop** handles malformed LLM output: if Pydantic validation fails, a dedicated repair model retries up to 3 times with escalating temperatures (`0.0 → 0.3 → 0.6`). The repair logic is shared across all LLM stages via `src/scinr/newton/utils/llm_repair.py`.

```bash
# CLI
newton --stage extract --input data/json/ --output data/output/ --parallel-docs 4

# Library API
result, docs = asyncio.run(run_extraction(input_folder="data/json/", output_folder="data/output/", parallel_docs=4))
```

---

### Stage 2 — Ingest

**Directory:** `src/scinr/newton/ingest/`  
**Input:** `data/output/extract-{filename}.json`  
**Target:** Neo4j database

Reads the extracted `Document` JSON and writes it to Neo4j using **`MERGE` statements throughout** — every write is idempotent, so re-running the stage on the same document is always safe. All writes for a single document are wrapped in one transaction.

Key behaviours:

- **Deterministic UIDs** — `InfoUnit` nodes receive a SHA-256[:16] hash derived from their content, so identical text fragments always map to the same node across runs.
- **Folder hierarchy** — when a `folder_path` is present, ancestor `(:Document)` nodes are created and connected via `[:IS_COMPOSED_OF]` relationships, mirroring the source directory tree inside the graph.
- **Versioning** — each ingest increments the `version` counter and sets `latest=true` on the new node while all previous versions become `latest=false`.

```bash
# CLI — ingest from output folder
newton --stage ingest --output data/output/

# CLI — update in-place (no new version created)
newton --stage ingest --output data/output/ --update

# CLI — link as successor of another document
newton --stage all --input-raw files/new/ --input data/json/ --output data/output/ --replaces "OldDocumentName"

# Library API
result = asyncio.run(run_ingestion(output_folder="data/output/"))
result = asyncio.run(run_ingestion(output_folder="data/output/", update_mode=True))
```

---

### Stage 3 — Annotate

**Directory:** `src/scinr/newton/annotation/`  
**Requires:** Stage 2 completed for the target document

Runs a **LangGraph `StateGraph` agent** over every `(:StructureNode)` in the graph for the given document and assigns it an extraction model. The agent loop:

```
load_nodes → [check_done] → prepare_node → classify_theme
           ↑                                     ↓
      write_decision ←──────────────── decide_model
```

For each node the agent reads the theme assigned during Stage 1, then makes a structured LLM call producing an `AnnotationDecision` (primary model, complementary models, supplementary fields, and rationale). Results are written as `(:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)` nodes.

The **ThemeRegistry** auto-discovers all `src/scinr/newton/models/*/catalog.py` files at startup and presents their `THEME_DESCRIPTION` and `SELECTABLE_MODELS` to the LLM — no code changes needed when a new domain is added.

```bash
# CLI — annotate all nodes
newton --stage annotate --document "MyDocument"

# CLI — resume (skip already-annotated nodes)
newton --stage annotate --document "MyDocument" --only-unannotated

# CLI — manual override (assign a fixed model without LLM)
newton --stage annotate --document "MyDocument" --manual --model "Triple"

# Library API
result = asyncio.run(run_annotation("MyDocument"))
result = asyncio.run(run_annotation("MyDocument", only_unannotated=True))
result = asyncio.run(run_annotation("MyDocument", manual=True, model_class="Triple"))
```

---

### Stage 4 — Entity Extract

**Directory:** `src/scinr/newton/entity_extraction/`  
**Requires:** Stage 3 completed for the target document

Runs a second **LangGraph `StateGraph` agent** that reads the model decisions from Stage 3 and extracts typed entities from each structural node. The agent loop:

```
load_targets → [check_done] → prepare_target → compose_schema
             ↑                                        ↓
        mark_extracted ← write_entities ← extract_entities
```

The **`compose_schema`** step dynamically constructs a composite Pydantic model at runtime by merging the node's primary model, any complementary models, and supplementary fields declared in the `ModelDecision`. This composite schema is used as the structured output target for the extraction LLM call.

- Fields annotated with `json_schema_extra={"entity_label": "X"}` are written as **global `(:LabeledEntity)` singletons** keyed by `(label, normalized_value)`, enabling cross-document deduplication.
- Fields with `field_relationships` metadata generate typed Neo4j relationships between entity nodes.
- Nodes without a model decision fall back to the `Triple` (RDF) model.

```bash
# CLI — extract entities for all annotated nodes
newton --stage entity_extract --document "MyDocument"

# CLI — resume (skip nodes that already have an ExtractionResult)
newton --stage entity_extract --document "MyDocument" --only-unextracted --parallel-docs 4

# Library API
result = asyncio.run(run_entity_extraction("MyDocument"))
result = asyncio.run(run_entity_extraction("MyDocument", only_unextracted=True, parallel_docs=4))
```

---

### Tabular Bypass

**Directory:** `src/scinr/newton/tabular/`  
**Input:** CSV / XLSX files  
**Target:** Neo4j database  
**LLM:** 3 calls per sheet (classify theme → decide model → map columns)

Ingests tabular files directly into Neo4j, bypassing Stages 0–4. For each sheet:

1. Reads headers and a 5-row preview.
2. Selects an extraction model via LLM.
3. Maps sheet columns to model fields via LLM.
4. Writes `(:Document)-[:HAS_STRUCTURE]->(:StructureNode:Table)-[:HAS_CHILD]->(:StructureNode:Row)` subgraph.

```bash
# CLI — ingest all CSV/XLSX files in a folder
newton --stage tabular --input-raw files/data/

# CLI — update mode (wipe and re-ingest)
newton --stage tabular --input-raw files/data/ --update

# Library API
result = asyncio.run(run_tabular_pipeline("files/data/"))
result = asyncio.run(run_tabular_pipeline("files/data/", update_mode=True))
```

The tabular pipeline is also automatically invoked by `--stage all` (and `run_pipeline()` with `input_raw`) when CSV/XLSX/XLS files are present in the input directory.

> When two or more columns map to the same model field, values are combined/deduplicated (only for `str` and `list[str]` fields — other field types keep the last value and log a warning). See the [Tabular Pipeline guide, §7.4](../../../docs/user-guides/tabular-pipeline.md#74-combining-values-when-multiple-columns-map-to-the-same-field) for details.

---

### CLI Reference

The CLI entry point is `scinr.newton.cli:main_sync`, registered as `newton` via `pyproject.toml`:

```bash
newton --stage <STAGE> [options]
```

| `--stage` choice | Equivalent `run_pipeline()` stages | Description |
|---|---|---|
| `all` | `["preprocess", "extraction", "ingestion", "annotation", "entity_extraction"]` | Full pipeline |
| `preprocess` | `["preprocess"]` | Stage 0 only |
| `extract` | `["extraction"]` | Stage 1 only |
| `ingest` | `["ingestion"]` | Stage 2 only |
| `annotate` | `["annotation"]` | Stage 3 only |
| `entity_extract` | `["entity_extraction"]` | Stage 4 only |
| `tabular` | `["tabular"]` | Tabular bypass only |

Key CLI flags:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | `DIR` | `data/json/` | Input folder for Stage 1 (intermediate JSON files) |
| `--input-raw` | `DIR` | — | Raw source files folder for Stage 0 / tabular |
| `--output` | `DIR` | `data/output/` | Output folder for Stage 1/2 |
| `--document` | `NAME` | — | Document name for Stage 3/4. Required with `--stage annotate` and `--stage entity_extract` |
| `--update` | flag | off | Update mode: re-ingest into the latest version without creating a new one |
| `--replaces` | `DOC_NAME` | — | Name of existing document being superseded |
| `--parallel-docs` | `N` | `1` | Concurrent documents across stages |
| `--only-unannotated` | flag | off | Stage 3 only: skip already-annotated nodes |
| `--only-unextracted` | flag | off | Stage 4 only: skip already-extracted nodes |
| `--manual` | flag | off | Stage 3 only: assign fixed model without LLM. Requires `--model` |
| `--model` | `CLASS_NAME` | — | CamelCase model class name for `--manual` annotation |
| `--context` | `TEXT` | — | Free-text context instructions passed to Stage 0 and Stage 3 LLMs |

---

## Neo4j Schema

### Node Labels

| Label | Key Properties | Description |
|---|---|---|
| `:Document` | `name`, `path`, `version`, `latest` | Document root. `latest=true` on the current version. |
| `:StructureNode` + role label | `node_id`, `title`, `role`, `theme` | One node per structural element; also labeled `:Section`, `:Subsection`, `:Table`, `:Appendix`, `:FieldGroup`, or `:FreeformBlock`. |
| `:InfoUnit` | `uid` (SHA-256[:16]), `title`, `description` | Semantic concept unit extracted from a node's body text. |
| `:ModelDecision` | `node_id` | Stage 3 annotation result for a StructureNode. |
| `:CatalogModel` | `name` | Registered Pydantic extraction model. |
| `:Theme` | `name`, `path` | Extraction domain. |
| `:ExtractionResult` | `node_id` | Stage 4 entity extraction result for a StructureNode. |
| `:LabeledEntity` | `label`, `normalized_value` | Global entity singleton — shared across documents. |
| `:Entity` | `normalized_value` | Triple-fallback entity (RDF subject / object). |

### Key Relationships

| Relationship | From → To | Description |
|---|---|---|
| `HAS_STRUCTURE` | `:Document` → `:StructureNode` | Document root to top-level structural nodes. |
| `HAS_CHILD` | `:StructureNode` → `:StructureNode` | Parent → child in the document tree. |
| `HAS_INFO_UNIT` | `:StructureNode` → `:InfoUnit` | Node to its semantic information units. |
| `IS_COMPOSED_OF` | `:Document` → `:Document` | Folder hierarchy (parent folder → child document). |
| `HAS_NEWER_VERSION` | `:Document` → `:Document` | Version chain: old version → new version. |
| `HAS_MODEL_DECISION` | `:StructureNode` → `:ModelDecision` | Stage 3 annotation result. |
| `MATCHED_MODEL` | `:ModelDecision` → `:CatalogModel` | Primary model selected for a node. |
| `HAS_EXTRACTION` | `:StructureNode` → `:ExtractionResult` | Stage 4 result link. |
| `REFERENCES` | `:ExtractionResult` → `:LabeledEntity` | Extraction → global entity singleton. |
| `<REL_TYPE>` | `:LabeledEntity` → `:LabeledEntity` | Field relationship (defined in model metadata). |

### Cypher Examples

```cypher
-- Find all latest documents
MATCH (d:Document {latest: true})
RETURN d.name, d.path, d.version
ORDER BY d.name;

-- Traverse document structure
MATCH (d:Document {name: "MyDocument", latest: true})
      -[:HAS_STRUCTURE]->
      (s:StructureNode)
      -[:HAS_INFO_UNIT]->
      (u:InfoUnit)
RETURN s.title, u.title, u.description
ORDER BY s.appearance_order;

-- Find all global entities of a given label
MATCH (e:LabeledEntity {label: "Substance"})
RETURN e.normalized_value
ORDER BY e.normalized_value;

-- Follow a document version chain
MATCH path = (old:Document)-[:HAS_NEWER_VERSION*]->(latest:Document {latest: true})
WHERE old.name = "MyDocument"
RETURN [n IN nodes(path) | {name: n.name, version: n.version, latest: n.latest}];

-- Find all nodes annotated with a specific model
MATCH (s:StructureNode)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
      -[:MATCHED_MODEL]->(m:CatalogModel {name: "Triple"})
RETURN s.node_id, s.title, s.theme;
```

### Document Versioning

Every ingest run auto-increments the `version` counter and marks only the newest node as `latest=true`.

| Scenario | CLI flag / API param | Behaviour |
|---|---|---|
| First ingest | *(none)* | Creates version 1 with `latest=true`. |
| Re-ingest (correction) | `--update` / `update_mode=True` | Wipes and re-ingests into the existing latest version; no new version node created. |
| New version | *(none, run again)* | Creates version N+1, links via `HAS_NEWER_VERSION`, sets `latest=true`. |
| Document supersedes another | `--replaces <name>` / `replaces="name"` | Links the new document as the successor of the named existing document. The old document's `latest=True` version becomes `latest=False`. |

---

## Storage Layer

The storage layer is **fully optional**. When `STORAGE_BACKEND` is not set (or set to `"none"`), the pipeline runs without MongoDB and all storage calls are silently skipped.

When enabled (`STORAGE_BACKEND=mongodb`), the storage layer persists:

| Collection / Bucket | Contents |
|---|---|
| `raw_files` (MongoDB) | Metadata for each ingested raw file (filename, content type, folder path, upload timestamp). |
| `converted_pages` (MongoDB) | Converted page content (markdown) + source page metadata per page. |
| `raw_binaries` (GridFS) | Binary content of raw source files (PDF bytes, DOCX bytes, etc.). |

The `raw_file_id` and `page_id` fields stored in Neo4j nodes allow cross-referencing back to the original binary and page content in MongoDB.

### Enable / Disable Storage

```dotenv
# Enable MongoDB storage (with authentication)
STORAGE_BACKEND=mongodb
MONGODB_URI=mongodb://your_user:your_password@localhost:27017/your_db?authSource=your_db
MONGODB_DATABASE=your_db

# Disable storage (omit STORAGE_BACKEND or leave blank)
# STORAGE_BACKEND=
```

> **`MONGODB_URI` must include credentials** when connecting to an authenticated MongoDB instance. The `authSource` query parameter must match the database where the user was created (same value as `MONGODB_DATABASE`).
>
> MongoDB creates the database automatically on the first write — no prior `CREATE DATABASE` step is needed.

### Custom Storage Backend

```python
from scinr.newton import configure

class MyRawFileRepo:
    def save(self, filename, content_type, folder_path, binary): ...
    async def delete(self, raw_file_id): ...  # required — see delete_document()

class MyPageRepo:
    def save(self, raw_file_id, pages): ...
    async def delete_pages(self, raw_file_id): ...  # required — see delete_document()

configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="secret",
    storage_backend="custom",
    custom_storage=(MyRawFileRepo(), MyPageRepo()),
)
```

> **Breaking change:** `RawFileRepository.delete(raw_file_id)` and `PageRepository.delete_pages(raw_file_id)` are new required abstract methods, added so that `delete_document()` (see [Document Deletion](#document-deletion)) can clean up documental storage before deleting the corresponding Neo4j nodes. Any pre-existing `storage_backend="custom"` implementation must add both methods. Both must be idempotent: `delete()` must not raise if the `raw_file_id` no longer exists, and `delete_pages()` must return `0` (not raise) if no pages match.

The storage backend abstraction lives in `src/scinr/newton/storage/base.py` and `src/scinr/newton/storage/factory.py`. Additional backends (e.g. PostgreSQL, S3) can be added by implementing the base interface.

---

## Extending the Pipeline

### Adding a New File Format Converter

1. Create `src/scinr/newton/converters/<format>.py` and subclass `BaseConverter` from `src/scinr/newton/converters/base.py`.
2. Set `supported_extensions: list[str]` on the class.
3. Implement `async convert(file_path: Path) -> dict` returning the paged JSON envelope:
   ```python
   {
       "pages": [{"index": 0, "markdown": "…"}],
       "folder_path": None,  # or a relative subfolder string
   }
   ```
4. Register the converter in `src/scinr/newton/converters/registry.py` by adding it to the `CONVERTERS` dict keyed by extension.

Alternatively, pass `extra_converters` to `configure()` to override converters at runtime without modifying the package:

```python
configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="secret",
    extra_converters={".rtf": MyRtfConverter},
)
```

See each converter's docstring in `src/scinr/newton/converters/` for the full `BaseConverter` interface and working examples.

---

### Adding a New Extraction Domain

1. Create `src/scinr/newton/models/<your_theme>/` with `__init__.py` and `catalog.py`.
2. Export `THEME_DESCRIPTION: str` and `SELECTABLE_MODELS: list[type[ExtractionModel]]` from `catalog.py`:
   ```python
   # src/scinr/newton/models/my_domain/catalog.py

   THEME_DESCRIPTION: str = (
       "One-line description used by the Stage 3 classification LLM to pick this theme."
   )

   SELECTABLE_MODELS: list[type[ExtractionModel]] = [MyModelA, MyModelB]
   ```
3. Write Pydantic model classes that inherit from `ExtractionModel`.
4. No other changes required — `ThemeRegistry` (`src/scinr/newton/utils/theme_registry.py`) picks up the new domain automatically on next startup.

For user-defined themes outside the package, pass the directory path to `configure()`:

```python
configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="secret",
    extra_models_paths=["/path/to/my/themes/"],
    enabled_user_themes=["my_domain"],   # None activates all
)
```

See [model-creation/README.md](model-creation/README.md) for the full developer guide including worked examples, entity label conventions, nested model patterns, `field_relationships` syntax, and `instance_relationships` syntax. For AI agent instructions on creating models, see [model-creation/AGENTS.md](model-creation/AGENTS.md).
