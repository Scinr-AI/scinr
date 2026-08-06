# scinr.newton Architecture

`scinr.newton` is an **async-first Python library** that processes life sciences documents through a **6-stage modular pipeline**, producing a structured knowledge graph stored in Neo4j with optional binary/auxiliary storage in MongoDB.

The pipeline operates via **two parallel tracks**:

- **Unstructured pipeline** (Stages 0–4): handles PDF, DOCX, PPTX, HTML, XML, TXT files through format conversion, LLM-powered structural extraction, graph ingestion, annotation, and entity extraction.
- **Tabular pipeline** (Stage 5): handles CSV, XLSX, XLS files through a dedicated path that bypasses Stages 0–4 entirely, using LLM-driven column mapping and direct graph writes.

Both tracks converge into the **same Neo4j knowledge graph**, sharing node labels, relationship types, and schema constraints.

---

## Table of Contents

1. [Pipeline Architecture Diagram](#1-pipeline-architecture-diagram)
2. [Stage Details](#2-stage-details)
   - [Stage 0: Preprocess](#stage-0-preprocess-run_preprocess)
   - [Stage 1: Extraction](#stage-1-extraction-run_extraction)
   - [Stage 2: Ingestion](#stage-2-ingestion-run_ingestion)
   - [Stage 3: Annotation](#stage-3-annotation-run_annotation)
   - [Stage 4: Entity Extraction](#stage-4-entity-extraction-run_entity_extraction)
   - [Tabular Pipeline](#tabular-pipeline-run_tabular_pipeline)
3. [Async Architecture](#3-async-architecture)
4. [Configuration System](#4-configuration-system)
5. [Module Structure](#5-module-structure)
6. [Data Flow](#6-data-flow)
7. [Neo4j Schema](#7-neo4j-schema)
8. [Storage Backends](#8-storage-backends)
9. [Prompt System](#9-prompt-system)
10. [Error Handling](#10-error-handling)
11. [Result Types](#11-result-types)

---

## 1. Pipeline Architecture Diagram

```
                    ┌──────────────────────────────────────────────────┐
                    │                  Raw Documents                    │
                    │  .pdf .docx .pptx .xlsx .csv .json .html .xml    │
                    └───────────────┬──────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         │                         ▼
┌──────────────────┐                │              ┌──────────────────┐
│ Stage 0:         │                │              │ Tabular Files    │
│ Preprocess       │                │              │ (.csv .xlsx .xls)│
│ Converters       │                │              │                  │
└───────┬──────────┘                │              └───────┬──────────┘
        ▼                           │                      ▼
┌──────────────────┐                │              ┌──────────────────┐
│ Stage 1:         │                │              │ Tabular Pipeline │
│ Extraction       │                │              │ Stage 5          │
│ LLM Chunking     │                │              │                  │
└───────┬──────────┘                │              └───────┬──────────┘
        ▼                           │                      │
┌──────────────────┐                │              ┌───────┴──────────┐
│ Stage 2:         │                │              │ Document +       │
│ Ingestion        │                │              │ Table + Row      │
│ Neo4j write      │                │              │ nodes (direct)   │
└───────┬──────────┘                │              └──────────────────┘
        ▼                           │
┌──────────────────┐                │
│ Stage 3:         │                │
│ Annotation       │                │
│ LLM classify     │                │
└───────┬──────────┘                │
        ▼                           │
┌──────────────────┐                │
│ Stage 4:         │                │
│ Entity Extract   │                │
│ Pydantic + Neo4j │                │
└────────┬─────────┘                │
         │                          │
         └──────────┬───────────────┘
                    ▼
         ┌────────────────────────┐
         │   Neo4j Knowledge      │
         │   Graph                │
         └────────────────────────┘
```

### Pipeline Entry Points

| Function | Module | Description |
|---|---|---|
| `run_pipeline()` | `pipeline.py` | Main orchestrator; chains stages 0–4 sequentially with tabular auto-detection |
| `run_preprocess()` | `stages/preprocess.py` | Standalone Stage 0 |
| `run_extraction()` | `stages/extraction.py` | Standalone Stage 1 |
| `run_ingestion()` | `stages/ingestion.py` | Standalone Stage 2 |
| `run_annotation()` | `stages/annotation.py` | Standalone Stage 3 |
| `run_entity_extraction()` | `stages/entity_extraction.py` | Standalone Stage 4 |
| `run_tabular_pipeline()` | `stages/tabular.py` | Standalone Stage 5 (tabular only) |

All stage functions are `async def` and return typed `StageResult` dataclasses.

---

## 2. Stage Details

### Stage 0: Preprocess (`run_preprocess`)

**Purpose:** Convert raw source files into a standardized intermediate JSON format.

**Input:** Directory of raw files (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.json`, `.html`, `.xml`, `.txt`).

**Output:**
- `StageResult` with per-file success/failure counts
- List of `IntermediateDocument` objects (in-memory)
- Optional JSON files on disk (when `output_dir` is provided)

**Architecture:**

Each file format has a dedicated converter inheriting from `BaseConverter` (abstract class in `converters/base.py`):

| Converter | File | Supported Extensions | Dependencies |
|---|---|---|---|
| `PdfConverter` | `converters/pdf.py` | `.pdf` | `pdfplumber`, Mistral OCR API |
| `DocxConverter` | `converters/docx.py` | `.docx` | `python-docx` |
| `PptxConverter` | `converters/pptx.py` | `.pptx` | `python-pptx` |
| `XlsxConverter` | `converters/xlsx.py` | `.xlsx`, `.xls` | `openpyxl`, `pandas` |
| `CsvConverter` | `converters/csv.py` | `.csv` | `pandas` |
| `HtmlConverter` | `converters/html.py` | `.html`, `.htm` | `BeautifulSoup` |
| `TextConverter` | `converters/text.py` | `.txt`, `.md`, `.rst` | stdlib |
| `ApiJsonConverter` | `converters/api_json.py` | `.json` | stdlib |
| `ApiXmlConverter` | `converters/api_xml.py` | `.xml` | stdlib |

Converters are registered in `converters/registry.py` via a lazy-loaded extension-to-class map. Custom converters can be injected at runtime via `configure(extra_converters={...})` — the `apply_converter_overrides()` function handles both new extensions and built-in overrides.

**PDF Conversion Strategy:**

The `PdfConverter` uses a two-tier approach:
1. **pdfplumber** for native (text-extractable) PDFs — extracts text, tables, images, and page dimensions.
2. **Mistral OCR API** for scanned PDFs — chunks large PDFs by page count (`mistral_ocr_safe_max_pages`, default 900) and file size (`mistral_ocr_safe_max_bytes`, default 45 MiB), sends each chunk to the Mistral OCR endpoint, and reassembles the result. The `pdf_splitter.py` module handles structural PDF partitioning.

Error strategy for Mistral OCR is configurable: `fail_fast` (default, aborts the entire document on any chunk failure) or `best_effort` (skips failed chunks and continues).

**Intermediate Document Format:**

Every converter produces an `IntermediateDocument` (Pydantic model) with:
- `pages`: list of `IntermediatePage` objects, each containing `index`, `markdown` (text content), `images` (base64-encoded with MIME type), `dimensions`, `tables`, `hyperlinks`, `header`, `footer`, and `page_id`.
- `folder_path`: relative path of the source file's parent directory from the input root.
- `raw_file_id`: MongoDB ObjectId of the stored raw file (when storage backend is configured).
- `context_instructions`: free-text user context injected via CLI `--context`.
- `document_name`: stem of the original source file.

**Concurrency:** Documents are processed with bounded parallelism via `parallel_docs` parameter (default: 1 for sequential processing, matching pre-existing behavior).

**Storage Integration:**

When `storage_backend` is configured (not `"none"`), Stage 0 stores:
- Raw file binary via `RawFileRepository.store()` (GridFS for MongoDB)
- Converted pages via `PageRepository.store()` (MongoDB collection)

The storage layer is abstracted behind `storage/factory.py` and supports three backends: `"none"`, `"mongodb"`, and `"custom"`.

---

### Stage 1: Extraction (`run_extraction`)

**Purpose:** Use an LLM to parse intermediate document pages into a hierarchical tree of `StructureNode` objects, producing `Document` objects.

**Input:** Either `IntermediateDocument` objects (from Stage 0 in-memory) or JSON files on disk (from `extraction_input_dir`).

**Output:**
- `StageResult` with per-document success/failure counts
- List of `Document` objects (in-memory)
- Optional `extract-*.json` files on disk (when `output_folder` is provided)

**Architecture:**

The extraction engine (`extraction/` module) processes documents in **sliding-window chunks** of configurable size (`extraction_batch_size`, default: 3 pages per chunk). Each chunk:

1. Builds the **active hierarchy** — the current tree of `StructureNode` objects accumulated so far for this document.
2. Sends the previous page (for context), current pages, and active hierarchy to the LLM via `extract_chunk()`.
3. The LLM returns a `DocumentStructure` (Pydantic model) containing a list of `StructureNode` objects for the new content in the chunk.
4. `compact_extraction()` merges the new nodes into the existing document tree, handling:
   - Continuation of nodes that started on a previous chunk
   - Insertion of new top-level and nested nodes
   - Preservation of parent-child relationships via `parent_id` references
   - Deduplication of nodes that appear across chunk boundaries

**StructureNode Model:**

Each `StructureNode` has:
- `node_id`: unique identifier (derived from heading number or appearance order + slug)
- `title`: exact heading text from the source document (never paraphrased)
- `role`: one of `section`, `subsection`, `table`, `appendix`, `field_group`, `freeform_block`, `row`
- `appearance_order`: 1-based position among siblings
- `parent_id`: reference to parent node's `node_id` (null for top-level)
- `theme`: theme path (default: `"default"`)
- `source_page_ids`: list of MongoDB page IDs (set by pipeline, not LLM)
- `info_units`: list of `InfoUnit` objects — the semantic content extracted from this node
- `children`: nested `StructureNode` objects

**InfoUnit Model:**

The smallest semantic unit, containing:
- `title`: short label (3–8 words)
- `order`: 0-based position within parent node
- `description`: self-contained technical note preserving all quantitative values, named entities, conditions, and qualifiers

InfoUnits are the **sole content representation** available to downstream agents (Stages 3 and 4). The LLM is instructed to make each description independently interpretable.

**LLM Bounded Concurrency:**

Each `extract_chunk()` call is bounded by the global `get_llm_semaphore()` (size: `llm_concurrency`, default: 4). This ensures that Stage 1 never exceeds the configured LLM provider rate limits, regardless of how many documents are processed concurrently.

**Output Persistence:**

When `output_folder` is provided, the extracted `Document` is written as `extract-{doc_name}.json` after each chunk (crash-safe incremental writes) and after final completion (final write). The subdirectory structure mirrors the input folder hierarchy.

---

### Stage 2: Ingestion (`run_ingestion`)

**Purpose:** Write `Document` objects and their `StructureNode` hierarchies into Neo4j.

**Input:** Either `Document` objects (from Stage 1 in-memory), `extract-*.json` files on disk (`output_folder` or explicit `files` list), or `extract-*.json` files from `ingestion_input_dir`.

**Output:** `StageResult` with per-document success/failure counts. Neo4j graph state updated.

**Architecture:**

The ingestion module (`ingest/`) provides:

- **`setup_schema(driver)`** — Creates all Neo4j constraints and indexes idempotently (using `IF NOT EXISTS`). Includes:
  - 10 unique constraints (Document path+version, StructureNode id, InfoUnit uid, etc.)
  - 9 regular indexes (Document name/latest/path, StructureNode role, etc.)
  - 2 fulltext indexes (InfoUnit description and title for semantic search)
  - Best-effort Neo4j version check (requires >= 4.4)

- **`load_documents(documents, driver, update_mode)`** — In-memory ingestion of `Document` objects.
- **`load_files(files, driver, update_mode)`** — Ingestion from explicit file paths.
- **`load_folder(folder, driver, update_mode)`** — Ingestion from a directory of `extract-*.json` files.

**Neo4j Graph Structure (unstructured pipeline):**

```
(:Document) -[:HAS_STRUCTURE]-> (:StructureNode)
(:StructureNode) -[:HAS_CHILD]-> (:StructureNode)
(:StructureNode) -[:HAS_INFO_UNIT]-> (:InfoUnit)
(:Document) -[:IS_COMPOSED_OF]-> (:Document)    [folder hierarchy]
(:Document) -[:HAS_NEWER_VERSION]-> (:Document) [versioning]
```

**Node Properties:**

- `:Document`: `name`, `path`, `version`, `latest`, `raw_file_id`, `context_instructions`, `ingestion_timestamp`
- `:StructureNode`: `id` (composite key), `title`, `role`, `appearance_order`, `theme`, `source_page_ids`, `row_index` (tabular only)
- `:InfoUnit`: `uid`, `title`, `order`, `description`

**Versioning:**

The ingestion loader resolves version numbers by querying existing `:Document` nodes with the same `path`. If `update_mode=True`, the existing version is reused (in-place update without creating a new version). Otherwise, a new version is created (`max_version + 1`). The `replaces` parameter links a newly ingested document to an existing one via `HAS_NEWER_VERSION` relationships.

**Update Mode:**

When `update_mode=True`, the ingestion process:
1. Finds the latest version of the document by `path`
2. Deletes all existing `StructureNode` and `InfoUnit` descendants
3. Re-inserts the new structure with the same version number
4. Does not create a new version entry

**Concurrency:**

Stage 2 uses `get_neo4j_sync_semaphore()` (size: `neo4j_sync_concurrency`, default: 8) to bound concurrent dispatches to `asyncio.to_thread()` for the synchronous Neo4j driver operations. The sync driver is used for Stage 2 because the Neo4j Python driver's synchronous API is used for document ingestion (the async driver is reserved for Stages 3 and 4).

---

### Stage 3: Annotation (`run_annotation`)

**Purpose:** Classify each `StructureNode` against registered extraction models using an LLM, writing annotation decisions to Neo4j.

**Input:** Document name (must already exist in Neo4j from Stage 2).

**Output:** `StageResult` with per-node annotation counts and errors. Neo4j graph updated with annotation subgraph.

**Architecture:**

The annotation module (`annotation/`) provides a two-step LLM pipeline per node:

**Step 1 — Model Decision (`decide_model`):**

The LLM receives:
- The `StructureNode`'s `title` and `role`
- All `InfoUnit` descriptions within the node
- The **catalog of available extraction models** (loaded from themes)
- **Theme descriptions** (via `THEME_DESCRIPTION` in each theme's `catalog.py`)
- Optional user-provided `context_instructions`

The LLM returns a `ModelDecision` with:
- `matched_model_class`: CamelCase name of the best-matching Pydantic model (or `NULL` for no match)
- `confidence`: qualitative confidence level
- `reasoning`: brief justification

**Step 2 — Decision Formatting (`format_decision`):**

A second LLM call validates and formats the decision, ensuring the model class name exactly matches a registered class in the model catalog.

**Neo4j Annotation Subgraph:**

```
(:StructureNode) -[:HAS_MODEL_DECISION]-> (:ModelDecision)
(:ModelDecision) -[:MATCHES_MODEL]-> (:CatalogModel)
(:ModelDecision) -[:BELONGS_TO_THEME]-> (:Theme)
(:CatalogModel) -[:HAS_FIELD]-> (:ModelField)
(:ModelField) -[:HAS_ENTITY_LABEL]-> (:EntityLabel)
```

**Theme System:**

Themes organize extraction models into domain-specific groups. Each theme has:
- A path (e.g., `"pharma_operations/batch_manufacturing"`)
- A `catalog.py` file listing all model classes in the theme
- A `THEME_DESCRIPTION` string used by the annotation LLM for theme selection
- Model files defining Pydantic schemas with `json_schema_extra` annotations for entity labels, field relationships, and instance keys

Built-in themes are registered in `utils/theme_registry.py`. User themes are loaded from `extra_models_paths` at configure time. The `enabled_base_themes` and `enabled_user_themes` configuration parameters act as whitelists.

**Catalog and Theme Neo4j Setup:**

Before annotation begins, `ensure_catalog_models_once()` and `ensure_theme_structure_once()` are called (idempotent, memoized) to:
1. Create `:CatalogModel` and `:ModelField` nodes for all registered models
2. Create `:Theme` nodes and link them to their models
3. Create `:EntityLabel` nodes for all entity labels defined in model fields

These operations run once per pipeline run and are memoized to avoid redundant Neo4j queries.

**Modes:**

- **LLM Agent Mode** (default): Uses the two-step LLM pipeline for each node.
- **Manual Mode** (`manual=True`): Assigns a fixed `model_class` to all qualifying nodes without LLM calls. Requires `model_class` parameter. Useful for bulk annotation of homogeneous document sets.

**Resume Flags:**

- `only_unannotated=True`: Skips nodes that already have a `:HAS_MODEL_DECISION` relationship, enabling partial re-runs.

**Folder Documents:**

When `document_name` refers to a folder (a document with `IS_COMPOSED_OF` children), all leaf descendants are resolved and annotated. Up to `parallel_docs` leaves are processed concurrently. Failures on individual leaves are logged but do not stop processing of other leaves.

---

### Stage 4: Entity Extraction (`run_entity_extraction`)

**Purpose:** Extract typed domain entities from annotated `StructureNode` objects using Pydantic structured output, writing entity subgraphs to Neo4j.

**Input:** Document name (must have annotated nodes from Stage 3).

**Output:** `StageResult` with per-node extraction counts and errors. Neo4j graph updated with entity subgraph.

**Architecture:**

The entity extraction module (`entity_extraction/`) operates per annotated `StructureNode`:

**Step 1 — Schema Composition (`schema_composer.py`):**

For each target node, the system:
1. Reads the `ModelDecision.matched_model_class` from Neo4j
2. Resolves the Pydantic model class via `model_resolver.py` (consults the theme registry)
3. Builds a **composite schema** — a single Pydantic model that combines the primary model and any complementary models (optional nested models declared in the primary model's fields)
4. The composite schema is used for structured output via `llm.with_structured_output()`

**Step 2 — LLM Extraction:**

The LLM receives:
- The composite Pydantic schema as structured output target
- The `StructureNode`'s `InfoUnit` descriptions (the sole content representation)
- The node's `title` and `role` for context
- Model field descriptions from the Pydantic schema

The LLM returns a populated instance of the composite schema.

**Step 3 — Graph Write (`graph_mapper.py`):**

The populated Pydantic instance is converted into a Neo4j subgraph with three levels of entity representation:

**Level 1 — Entity Labeling:**
- Fields with `json_schema_extra={"entity_label": "X"}` become `MERGE`d `:LabeledEntity {label, value, normalized_value}` nodes
- Same label + same normalized value always resolves to the same node across all extractions (global deduplication)
- Connected via `[:REFERENCES]` relationships

**Level 2 — Field Relationships:**
- Fields with `json_schema_extra={"field_relationships": [{"to_field": "...", "rel_type": "..."}]}` trigger `MERGE` relationships between source and target `:LabeledEntity` nodes

**Level 3 — Instance Key Relationships:**
- Fields with `json_schema_extra={"instance_key": True}` define a composite key for `:ModelInstance` deduplication
- Fields with `json_schema_extra={"instance_relationships": [...]}` trigger `MERGE` of target `:ModelInstance` shells and typed relationships between instances
- Enables forward references across `StructureNode` boundaries

**Neo4j Entity Subgraph:**

```
(:StructureNode) -[:HAS_EXTRACTION]-> (:ExtractionResult)
(:ExtractionResult) -[:USES_PRIMARY_MODEL]-> (:CatalogModel)
(:ExtractionResult) -[:USES_COMPLEMENTARY_MODEL]-> (:CatalogModel)  [0..*]
(:ExtractionResult) -[:HAS_<FIELD>]-> (:ModelInstance)              [nested models]
(:ModelInstance | :ExtractionResult) -[:REFERENCES]-> (:LabeledEntity)
(:LabeledEntity) -[:REL_TYPE]-> (:LabeledEntity)                    [field_relationships]
(:ModelInstance) -[:REL_TYPE]-> (:ModelInstance)                    [instance_relationships]
```

**Triple (Fallback) Extraction:**

For nodes where `ModelDecision.matched_model_class` is `NULL` (no specific domain model matched), a fallback `Triple` model extracts subject-predicate-object statements:

```
(:StructureNode) -[:HAS_EXTRACTION]-> (:ExtractionResult {model_class: "Triple"})
(:ExtractionResult) -[:HAS_ENTITY {role}]-> (:Entity)
(:Entity) -[:NORMALIZED_PREDICATE {predicate_raw}]-> (:Entity)
```

Entity nodes are global singletons (MERGE by normalized value), shared across all extractions.

**JSON Repair Loop:**

When the LLM returns malformed JSON, the `utils/llm_repair.py` module attempts repair via a secondary LLM call (using `repair_llm`, which defaults to the main `llm` if not configured separately). This is transparent to the calling code.

**Resume Flags:**

- `only_unextracted=True`: Skips nodes that already have a `:HAS_EXTRACTION` relationship, enabling partial re-runs.

---

### Tabular Pipeline (`run_tabular_pipeline`)

**Purpose:** Ingest CSV, XLSX, and XLS files directly into Neo4j, bypassing Stages 0–4 entirely.

**Input:** Directory of raw tabular files (`.csv`, `.xlsx`, `.xls`).

**Output:** `StageResult` with per-file success/failure counts. Neo4j graph updated with Document, Table, and Row nodes.

**Architecture:**

The tabular pipeline (`tabular/`) uses a **LangGraph**-based state machine (`tabular/graph.py`) with the following nodes:

1. **`load_sheets`** — Reads the file (CSV via pandas, XLSX/XLS via openpyxl), extracts headers and a 5-row preview, and stores per-sheet data in the `TabularState`.

2. **`decide_model`** — Makes one LLM call per sheet to decide which extraction model to use. The LLM receives the sheet headers, preview rows, and the catalog of available models. Returns the `matched_model_class`.

3. **`map_columns`** — Makes one LLM call per sheet to map column names to model fields. The LLM receives the sheet headers, the selected model's field definitions, and returns a column-to-field mapping.

4. **`write_tabular`** — Writes the `Table` and `Row` `StructureNode` subgraph directly to Neo4j. Each row becomes a `:StructureNode {role: "row"}` with `InfoUnit` children for each mapped cell value.

**Per-File Process:**

For each tabular file:
1. Create `:Document` node and folder hierarchy in Neo4j (single transaction)
2. Run the LangGraph state machine for each sheet
3. Store raw file binary in MongoDB (if storage backend is configured)

**NormalizationEngine:**

When `normalization_enabled=True` (default: `False`), the `NormalizationEngine` (in `tabular/`) performs post-extraction normalization for nested model fields. It batches entries (configurable via `normalization_batch_size`, default: 5) and uses a dedicated LLM (`normalization_llm`, falls back to main `llm`) to normalize values into consistent formats.

**Tabular Neo4j Structure:**

```
(:Document) -[:HAS_STRUCTURE]-> (:StructureNode {role: "table"})
(:StructureNode {role: "table"}) -[:HAS_CHILD]-> (:StructureNode {role: "row"})
(:StructureNode {role: "row"}) -[:HAS_INFO_UNIT]-> (:InfoUnit)
```

Each row node has a `row_index` property (0-based position within the table).

**Auto-Detection in Mixed Folders:**

When `run_pipeline()` detects tabular files in `input_raw` alongside non-tabular files, it:
1. Runs the tabular pipeline first for all tabular files
2. Proceeds with the unstructured pipeline (Stages 0–4) for non-tabular files
3. Both tracks share the same Neo4j graph and versioning system

**Tabular-Only Mode:**

When `stages=["tabular"]` is specified, only the tabular pipeline runs. This stage cannot be combined with other stages in the `stages` parameter.

---

## 3. Async Architecture

All pipeline stages are `async def`. The async architecture is built around three layers of concurrency control:

### Document-Level Concurrency

`run_pipeline()` uses a per-document `asyncio.Semaphore(parallel_docs)` to bound how many documents are processed concurrently across all stages. Each document is dispatched as an independent task via `asyncio.gather()` and runs through its applicable stages sequentially.

```python
document_semaphore = asyncio.Semaphore(parallel_docs)
unit_results = await asyncio.gather(
    *[
        _process_document_unit(u, document_semaphore=document_semaphore, ...)
        for u in units
    ],
    return_exceptions=True,
)
```

### LLM-Level Concurrency

The global `get_llm_semaphore()` (size: `llm_concurrency`, default: 4) bounds all LLM calls across all stages. Every `extract_chunk()`, annotation decision, entity extraction, and tabular mapping call acquires this semaphore before invoking the LLM. This prevents exceeding provider rate limits.

### Neo4j-Level Concurrency

Two separate semaphores control Neo4j access:

- **`get_neo4j_semaphore()`** (size: `neo4j_concurrency`, default: 10): Bounds concurrent Neo4j async sessions during annotation (Stage 3) and entity extraction (Stage 4).
- **`get_neo4j_sync_semaphore()`** (size: `neo4j_sync_concurrency`, default: 8): Bounds concurrent dispatches to `asyncio.to_thread()` for Stage 2 (synchronous ingestion). Must be acquired/released on the event loop, never inside the worker thread.

### Per-Document Unit Processing

`_process_document_unit()` processes a single `DocumentUnit` through all applicable stages. It:
1. Acquires the document semaphore for its entire duration
2. Runs stages sequentially within the unit
3. Implements soft-abort semantics:
   - **Stage 0/1/2 failure**: Always stops the unit (no valid artifact for subsequent stages)
   - **Stage 3/4 partial failure** (`nodes_failed > 0`): Stops the unit only when `on_partial_failure="abort"` (default). With `"continue"` or `"warn"`, the unit advances to its next stage despite partial failures.
4. Never propagates exceptions — returns a `UnitResult` with `fatal_error` set for uncaught exceptions

### Pre-Warming

Before dispatching document units, `run_pipeline()` pre-warms shared resources:
1. Opens sync Neo4j driver and sets up schema
2. Resolves batch version for all documents
3. Ensures catalog models and theme structure exist in Neo4j
4. Initializes storage backends

This reduces per-document startup latency.

---

## 4. Configuration System

### Singleton Pattern

Configuration is managed by the `ScinrConfig` dataclass in `config.py` via a module-level singleton (`_config`). The public API is:

- **`configure(...)`** — Sets global configuration. Must be called before any pipeline function.
- **`get_config()`** — Returns the current configuration. Raises `ConfigurationError` if not configured.

### Triple Resolution

All parameters follow a three-tier resolution order:

1. **Explicit argument** passed to `configure()`
2. **Environment variable** (via `os.getenv`)
3. **Hard-coded default** value

Example:
```python
resolved_neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
```

### Key Configuration Parameters

| Parameter | Env Var | Default | Description |
|---|---|---|---|
| `llm` | `MODEL_ID` (Bedrock) | None | LangChain `BaseChatModel` instance |
| `repair_llm` | — | Falls back to `llm` | Secondary LLM for JSON repair |
| `neo4j_uri` | `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `neo4j_user` | `NEO4J_USER` / `NEO4J_AUTH` | — | Neo4j username (required) |
| `neo4j_password` | `NEO4J_PASSWORD` / `NEO4J_AUTH` | — | Neo4j password (required) |
| `storage_backend` | `STORAGE_BACKEND` | `"none"` | `"none"`, `"mongodb"`, or `"custom"` |
| `llm_concurrency` | `LLM_CONCURRENCY` | 4 | Max concurrent LLM calls |
| `neo4j_concurrency` | `NEO4J_CONCURRENCY` | 10 | Max concurrent Neo4j async sessions |
| `neo4j_sync_concurrency` | `NEO4J_SYNC_CONCURRENCY` | 8 | Max concurrent sync ingestion dispatches |
| `extraction_batch_size` | `EXTRACTION_BATCH_SIZE` | 3 | Pages per extraction chunk |
| `prompt_family` | `PROMPT_FAMILY` | `"generic"` | Prompt variant family |
| `prompt_caching_enabled` | `PROMPT_CACHING_ENABLED` | `True` | Bedrock prompt caching |
| `normalization_enabled` | `NORMALIZATION_ENABLED` | `False` | Enable tabular normalization |
| `normalization_batch_size` | `NORMALIZATION_BATCH_SIZE` | 5 | Max entries per normalization batch |

### LLM Configuration

The library supports any LangChain `BaseChatModel` that implements `with_structured_output()`:
- `ChatOpenAI` (OpenAI)
- `ChatBedrockConverse` (AWS Bedrock)
- `ChatAnthropic` (Anthropic/Claude)
- `ChatOllama` (Ollama)
- Any other LangChain chat model with structured output support

When `MODEL_ID` is set (and no explicit `llm` is passed), `configure()` automatically creates a `ChatBedrockConverse` instance with connection pool sizing relative to `llm_concurrency`.

### Prompt Family

Three prompt families are supported via `PromptFamily` enum:

| Family | Description | Best For |
|---|---|---|
| `GENERIC` | Simplified, model-agnostic prompts | All LLM families (default) |
| `CLAUDE` | XML-structured instructions, multi-step protocols, internal checklists | Claude/Sonnet models |
| `GPT_REASONING` | Markdown section headers, goal-based language, no CoT elicitation | OpenAI reasoning models (GPT-5.5, o3, o4-mini) |

Each family has dedicated prompt files in `annotation/` and `entity_extraction/` modules (e.g., `prompts_claude.py`, `prompts_gpt_reasoning.py`, `prompts_generic.py`).

### Post-Configure Reset

`configure()` resets dependent lazy singletons after setting the config:
- Theme registry
- Async Neo4j driver singleton
- MongoDB client
- Catalog memoization
- LLM/Neo4j semaphores (manual reset required via `reset_*_semaphore()`)

---

## 5. Module Structure

```
scinr.newton/
├── __init__.py                 # Package exports
├── config.py                   # ScinrConfig, configure(), get_config(), semaphore helpers
├── pipeline.py                 # run_pipeline() orchestrator + _process_document_unit()
├── pipeline_units.py           # DocumentUnit discovery (raw_file, extraction_json, ingestion_json, pre_ingested)
├── results.py                  # PipelineResult, StageResult, DocumentResult dataclasses
├── exceptions.py               # ScinrError hierarchy
├── cli.py                      # CLI entry point (Typer-based)
│
├── annotation/                 # Stage 3: LLM classification
│   ├── agent.py                # run_annotation_agent(), run_manual_annotation()
│   ├── models.py               # AnnotationDecision, ModelDecision Pydantic models
│   ├── neo4j_ops.py            # fetch_nodes_to_annotate(), write_annotation(), catalog/theme setup
│   ├── nodes.py                # process_single_annotation_node() — per-node LLM pipeline
│   ├── prompts.py              # Prompt family dispatcher
│   ├── prompts_generic.py      # GENERIC family prompts
│   ├── prompts_claude.py       # CLAUDE family prompts
│   ├── prompts_gpt_reasoning.py # GPT_REASONING family prompts
│   └── state.py                # AnnotationState dataclass
│
├── converters/                 # File format converters
│   ├── base.py                 # BaseConverter ABC, IntermediateDocument, IntermediatePage
│   ├── registry.py             # Extension-to-converter map, apply_converter_overrides()
│   ├── main.py                 # convert_one(), convert_folder() — parallel conversion
│   ├── pdf.py                  # PdfConverter (pdfplumber + Mistral OCR)
│   ├── pdf_splitter.py         # Structural PDF partitioning for OCR chunking
│   ├── docx.py                 # DocxConverter (python-docx)
│   ├── pptx.py                 # PptxConverter (python-pptx)
│   ├── xlsx.py                 # XlsxConverter (openpyxl + pandas)
│   ├── csv.py                  # CsvConverter (pandas)
│   ├── html.py                 # HtmlConverter (BeautifulSoup)
│   ├── text.py                 # TextConverter (stdlib)
│   ├── api_json.py             # ApiJsonConverter (stdlib)
│   ├── api_xml.py              # ApiXmlConverter (stdlib)
│   └── config.py               # Converter-specific configuration
│
├── entity_extraction/          # Stage 4: entity extraction
│   ├── agent.py                # run_entity_extraction_agent()
│   ├── graph_mapper.py         # write_extraction_subgraph(), write_triple_subgraph()
│   ├── model_resolver.py       # resolve_model_class() — theme registry lookup
│   ├── neo4j_ops.py            # fetch_extraction_targets()
│   ├── nodes.py                # process_single_extraction_target() — per-node LLM pipeline
│   ├── prompts.py              # Prompt family dispatcher
│   ├── prompts_generic.py      # GENERIC family prompts
│   ├── prompts_claude.py       # CLAUDE family prompts
│   ├── prompts_gpt_reasoning.py # GPT_REASONING family prompts
│   ├── schema_composer.py      # Composite schema construction from primary + complementary models
│   └── state.py                # EntityExtractionState dataclass
│
├── extraction/                 # Stage 1: chunking
│   ├── extraction.py           # extract_chunk() — LLM call for one chunk
│   ├── compact_extraction.py   # compact_extraction() — merge chunk results into document tree
│   └── prompts/                # Extraction-specific prompt templates
│
├── ingest/                     # Stage 2: Neo4j ingestion
│   ├── config.py               # get_driver(), get_async_driver() — Neo4j driver singletons
│   ├── loader.py               # load_documents(), load_files(), load_folder(), version resolution
│   ├── nodes.py                # insert_document(), insert_structure_node(), insert_info_unit()
│   └── schema.py               # setup_schema() — constraints and indexes
│
├── models/                     # Core Pydantic models
│   ├── document_structure.py   # Document, StructureNode, InfoUnit, DocumentStructure, NodeRole
│   └── base.py                 # StrictModel base class
│
├── prompts/                    # System prompt templates
│   ├── system_prompt.py        # Prompt family dispatcher
│   ├── system_prompt_generic.py  # GENERIC system prompts
│   ├── system_prompt_claude.py   # CLAUDE system prompts
│   └── system_prompt_gpt_reasoning.py # GPT_REASONING system prompts
│
├── stages/                     # Stage orchestrators
│   ├── __init__.py             # Re-exports all public stage functions
│   ├── preprocess.py           # run_preprocess()
│   ├── extraction.py           # run_extraction()
│   ├── ingestion.py            # run_ingestion(), apply_replacement(), preflight_check_replaces()
│   ├── annotation.py           # run_annotation()
│   ├── entity_extraction.py    # run_entity_extraction()
│   └── tabular.py              # run_tabular_pipeline()
│
├── storage/                    # Storage backends
│   ├── base.py                 # RawFileRepository, PageRepository ABCs
│   ├── factory.py              # get_storage() — backend factory
│   ├── null.py                 # NullRawFileRepository, NullPageRepository (no-op)
│   ├── config.py               # Storage configuration
│   ├── models.py               # RawFileRecord, PageRecord Pydantic models
│   └── mongodb/                # MongoDB implementation
│       ├── client.py           # Motor async client singleton
│       ├── raw_files.py        # MongoDBRawFileRepository (GridFS)
│       └── pages.py            # MongoDBPageRepository (collection)
│
├── tabular/                    # Tabular pipeline (Stage 5)
│   ├── agent.py                # run_tabular_agent(), run_tabular_agent_sync()
│   ├── graph.py                # LangGraph state machine (load_sheets → decide_model → map_columns → write)
│   ├── state.py                # TabularState — LangGraph state schema
│   ├── reader.py               # CSV/XLSX/XLS file reading
│   ├── neo4j_ops.py            # Tabular Neo4j writes
│   ├── nodes.py                # Per-sheet tabular node processing
│   ├── prompts.py              # Prompt family dispatcher
│   ├── prompts_generic.py      # GENERIC family prompts
│   ├── prompts_claude.py       # CLAUDE family prompts
│   ├── prompts_gpt_reasoning.py # GPT_REASONING family prompts
│   ├── models.py               # Tabular-specific Pydantic models
│   └── ...
│
└── utils/                      # Utilities
    ├── theme_registry.py       # Theme discovery, model loading, get_theme_registry()
    ├── llm_factory.py          # LLM instance creation helpers
    ├── llm_retry.py            # LLM call retry with exponential backoff
    ├── llm_repair.py           # JSON repair loop via secondary LLM
    ├── neo4j_retry.py          # Neo4j operation retry with exponential backoff
    ├── neo4j_concurrency.py    # Neo4j concurrency utilities
    ├── document_resolver.py    # resolve_leaf_document_names() — folder → leaf resolution
    ├── file_archiver.py        # File archiving utilities
    ├── logging_config.py       # Structured logging setup
    └── uid.py                  # Deterministic UID generation (make_uid, make_instance_uid)
```

---

## 6. Data Flow

### Between Stages

Data flows between stages through three mechanisms:

| Mechanism | Direction | Description |
|---|---|---|
| **In-memory objects** | Stage N → Stage N+1 | `IntermediateDocument` (0→1), `Document` (1→2), document names (2→3→4) |
| **Intermediate JSON files** | Stage N → disk → Stage N+1 | `*.json` (0→1), `extract-*.json` (1→2) |
| **Neo4j graph** | Stage N → graph → Stage N+1 | `:Document`/`:StructureNode` (2→3→4), annotation subgraph (3→4) |

### Pipeline Data Flow (Full Run)

```
input_raw/                          converter_output_dir/          extraction_output_dir/         Neo4j
  ├── doc1.pdf ──Stage 0──►  doc1.json ──Stage 1──►  extract-doc1.json ──Stage 2──►  (:Document)
  ├── doc2.docx ──Stage 0──► doc2.json ──Stage 1──►  extract-doc2.json ──Stage 2──►  (:Document)
  └── data.csv ──tabular──►  (bypassed)    (bypassed)       (bypassed)   ──Stage 5──►  (:Document)
```

### Intermediate Directory Structure

When intermediate directories are used:

```
converter_output_dir/
├── doc1.json              # Stage 0 output (IntermediateDocument)
├── doc2.json
└── subfolder/
    └── doc3.json

extraction_output_dir/
├── extract-doc1.json      # Stage 1 output (Document)
├── extract-doc2.json
└── subfolder/
    └── extract-doc3.json
```

The subdirectory structure mirrors the input folder hierarchy, preserving the `folder_path` metadata.

### Stage Skipping

Stages can be skipped by providing input from a later stage:

| Skip To | Parameter | Effect |
|---|---|---|
| Stage 1 | `extraction_input_dir` | Skips Stage 0; reads JSON from disk |
| Stage 2 | `ingestion_input_dir` | Skips Stages 0 and 1; reads `extract-*.json` from disk |
| Stage 3 | `document_names` | Skips Stages 0–2; uses already-ingested documents |
| Stage 3 | `document_names_dir` | Skips Stages 0–2; reads names from `extract-*.json` files |

The `stages` parameter controls which stages execute. When `stages=["annotation", "entity_extraction"]`, only Stages 3 and 4 run, and `document_names` or `document_names_dir` must be provided.

### Independent Stage Execution

Each stage function can be called independently:

```python
# Stage 0 only
result, docs = await run_preprocess(input_raw="files/", output_dir="data/json/")

# Stage 1 only (from disk)
result, docs = await run_extraction(input_folder="data/json/", output_folder="data/extract/")

# Stage 2 only (from disk)
result = await run_ingestion(output_folder="data/extract/")

# Stage 3 only (from Neo4j)
result = await run_annotation(document_name="MyDocument")

# Stage 4 only (from Neo4j)
result = await run_entity_extraction(document_name="MyDocument")

# Tabular only
result = await run_tabular_pipeline(input_raw="files/")
```

---

## 7. Neo4j Schema

### Node Labels

| Label | Description | Primary Key |
|---|---|---|
| `:Document` | Ingested document (versioned) | `(path, version)` composite |
| `:StructureNode` | Structural division (section, table, row, etc.) | `id` |
| `:InfoUnit` | Semantic information unit | `uid` |
| `:ModelDecision` | Annotation decision for a node | — |
| `:CatalogModel` | Registered Pydantic extraction model | `name` |
| `:ModelField` | Field of a CatalogModel | `(name, model)` composite |
| `:EntityLabel` | Schema-level entity label singleton | `label` |
| `:Theme` | Extraction model theme | — |
| `:ExtractionResult` | Entity extraction result | `uid` |
| `:ModelInstance` | Extracted model instance (nested) | `uid` |
| `:LabeledEntity` | Globally deduplicated entity | `(label, normalized_value)` |
| `:Entity` | Triple extraction entity (fallback) | `uid` |

### Relationship Types

| Type | Source → Target | Description |
|---|---|---|
| `HAS_STRUCTURE` | Document → StructureNode | Root structural nodes |
| `HAS_CHILD` | StructureNode → StructureNode | Hierarchical nesting |
| `HAS_INFO_UNIT` | StructureNode → InfoUnit | Semantic content |
| `IS_COMPOSED_OF` | Document → Document | Folder hierarchy |
| `HAS_NEWER_VERSION` | Document → Document | Version succession |
| `HAS_MODEL_DECISION` | StructureNode → ModelDecision | Annotation result |
| `MATCHES_MODEL` | ModelDecision → CatalogModel | Selected model |
| `BELONGS_TO_THEME` | ModelDecision → Theme | Theme assignment |
| `HAS_FIELD` | CatalogModel → ModelField | Model schema |
| `HAS_ENTITY_LABEL` | ModelField → EntityLabel | Entity label declaration |
| `HAS_EXTRACTION` | StructureNode → ExtractionResult | Extraction output |
| `USES_PRIMARY_MODEL` | ExtractionResult → CatalogModel | Primary model used |
| `USES_COMPLEMENTARY_MODEL` | ExtractionResult → CatalogModel | Complementary model |
| `HAS_<FIELD>` | ExtractionResult → ModelInstance | Nested model instance |
| `REFERENCES` | ModelInstance → LabeledEntity | Entity reference |
| `HAS_ENTITY` | ExtractionResult → Entity | Triple extraction entity |
| `NORMALIZED_PREDICATE` | Entity → Entity | Triple relationship |

### Constraints and Indexes

See `ingest/schema.py` for the complete DDL. Key constraints:
- **10 unique constraints** ensuring node identity and preventing duplicates
- **9 regular indexes** for query performance
- **2 fulltext indexes** for semantic search on InfoUnit content

---

## 8. Storage Backends

The storage layer abstracts raw file and converted page persistence behind repository interfaces:

### Backend Types

| Backend | Description | Configuration |
|---|---|---|
| `"none"` | No persistence; null repositories | Default |
| `"mongodb"` | MongoDB with GridFS for binaries | `mongodb_uri`, `mongodb_database`, etc. |
| `"custom"` | User-provided repository pair | `custom_storage=(raw_repo, page_repo)` |

### MongoDB Structure

When `storage_backend="mongodb"`:
- **GridFS bucket** (`mongodb_gridfs_bucket`, default: `"raw_binaries"`): Stores raw file binaries with metadata (filename, content_type, folder_path)
- **`raw_files` collection** (`mongodb_raw_files_collection`): Raw file metadata records
- **`converted_pages` collection** (`mongodb_pages_collection`): Converted page records with markdown content, images, and dimensions

### Repository Interfaces

- **`RawFileRepository`** (ABC): `store(filename, content, content_type, folder_path)` → ObjectId
- **`PageRepository`** (ABC): `store(document_name, pages, folder_path)` → list of ObjectIds

Null implementations (`NullRawFileRepository`, `NullPageRepository`) are used when `storage_backend="none"`, returning `None` for all operations.

---

## 9. Prompt System

The prompt system supports three families, each with dedicated prompt files per stage:

### Prompt Resolution

1. `get_prompt_family()` returns the configured `PromptFamily` enum value
2. Each stage's `prompts.py` dispatcher selects the appropriate prompt file based on the family
3. System prompts are resolved from `prompts/system_prompt_*.py`
4. Stage-specific prompts are resolved from their respective module's `prompts_*.py` files

### Prompt File Convention

```
<stage_module>/
├── prompts.py              # Dispatcher: selects file based on PromptFamily
├── prompts_generic.py      # GENERIC prompts (default, model-agnostic)
├── prompts_claude.py       # CLAUDE prompts (XML-structured, extended reasoning)
└── prompts_gpt_reasoning.py # GPT_REASONING prompts (Markdown, goal-based)
```

### Bedrock Prompt Caching

When using `ChatBedrockConverse` with `prompt_caching_enabled=True`, the `make_system_message()` function appends a `cachePoint` block to the system message, reducing token costs by ~90% on repeated calls with the same prompt.

---

## 10. Error Handling

### Exception Hierarchy

All exceptions inherit from `ScinrError`:

```
ScinrError (base)
├── ConfigurationError      # Misconfiguration (missing LLM, Neo4j credentials, etc.)
├── PreconditionError       # Pipeline called out of order
├── ExtractionError         # LLM extraction failed after retries
├── IngestionError          # Neo4j write failed
├── ModelError              # Pydantic model resolution failure
├── StorageError            # MongoDB unavailable/misconfigured
└── ConversionError         # File converter failure
```

### Retry Mechanisms

- **LLM Retry** (`utils/llm_retry.py`): Exponential backoff retry for LLM calls
- **Neo4j Retry** (`utils/neo4j_retry.py`): Exponential backoff retry for Neo4j operations
- **JSON Repair** (`utils/llm_repair.py`): Secondary LLM call to repair malformed JSON output
- **Bedrock Retry** (`utils/bedrock_retry.py`): Bedrock-specific retry with service-aware backoff

### Partial Failure Handling

The `on_partial_failure` parameter controls pipeline behavior when a stage reports failures:

| Value | Behavior |
|---|---|
| `"abort"` | Stops the failing document's remaining stages (default) |
| `"continue"` | Document advances to next stage silently |
| `"warn"` | Document advances with per-document warning logged |

Note: `on_partial_failure` only affects annotation (Stage 3) and entity extraction (Stage 4) partial failures. Stage 0/1/2 failures always stop the document (no valid artifact to continue with).

---

## 11. Result Types

### Result Hierarchy

```
PipelineResult
├── success: bool
├── total_duration_seconds: float
├── stages_executed: list[str]
├── preprocess: StageResult | None
├── extraction: StageResult | None
├── ingestion: StageResult | None
├── annotation: StageResult | None
├── entity_extraction: StageResult | None
└── tabular: StageResult | None

StageResult
├── stage: str
├── success: bool
├── documents: list[DocumentResult]
├── total_processed: int
├── total_failed: int
├── duration_seconds: float
└── errors: list[str]

DocumentResult
├── document_name: str
├── nodes_processed: int
├── nodes_failed: int
└── errors: list[str]

DeletionResult
├── path: str
├── version: int | None
├── found: bool
├── versions_deleted: list[int]
├── documents_deleted: int
├── structure_nodes_deleted: int
├── info_units_deleted: int
├── model_decisions_deleted: int
├── proposed_models_deleted: int
├── proposed_fields_deleted: int
├── extraction_results_deleted: int
├── gc_entity_model_instance_deleted: int
├── gc_entity_model_instance_passes: int
├── gc_labeled_entity_deleted: int
└── gc_labeled_entity_passes: int
```

### Result Semantics

- **Stages 0–2**: `nodes_processed` = 1 for success, 0 for failure (per document)
- **Stages 3–4**: `nodes_processed` = number of StructureNodes processed (per document)
- **Tabular**: `nodes_processed` = 1 for success, 0 for failure (per file)

All result dataclasses are defined in `results.py` and provide structured, type-safe access to pipeline outcomes.
