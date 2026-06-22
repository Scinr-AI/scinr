# scinr-ingest

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A 5-stage LLM-powered document ingestion pipeline that converts raw documents into a richly structured Neo4j knowledge graph.**

`scinr-ingest` accepts documents in a wide range of formats (PDF, DOCX, XLSX, PPTX, CSV, HTML, XML, TXT and more), converts them to a uniform paged-markdown representation, uses AWS Bedrock (Claude Sonnet) to extract a hierarchical semantic document tree, ingests that tree into Neo4j with full versioning and folder-hierarchy support, and then runs two independent LangGraph agents to annotate each structural node with an extraction model and pull typed entities out of the text — all with idempotent, crash-safe writes at every step.

Designed for scientific and pharmaceutical documents, but fully extensible to any domain via a simple model plug-in system.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Pipeline Stages](#pipeline-stages)
   - [Stage 0 — Preprocess](#stage-0--preprocess)
   - [Stage 1 — Extract](#stage-1--extract)
   - [Stage 2 — Ingest](#stage-2--ingest)
   - [Stage 3 — Annotate](#stage-3--annotate)
   - [Stage 4 — Entity Extract](#stage-4--entity-extract)
   - [Tabular Bypass](#tabular-bypass)
6. [Extraction Domain Models](#extraction-domain-models)
7. [Neo4j Schema](#neo4j-schema)
8. [Storage Layer](#storage-layer)
9. [Development](#development)
10. [Contributing](#contributing)
11. [License](#license)

---

## Features

- **Multi-format ingestion** — PDF (Mistral OCR), DOCX, XLSX, PPTX, CSV, HTML, TXT, MD, XML/SOAP APIs, JSON APIs — all normalized to the same intermediate format
- **LLM-powered extraction** — sliding-window 2-phase extraction with an automatic repair loop; crash-safe intermediate writes after every chunk
- **Knowledge graph output** — idempotent Neo4j writes (MERGE throughout); safe to re-run at any stage
- **Two independent LangGraph agents** — Annotation agent assigns an extraction model to each structural node; Entity Extraction agent pulls typed Pydantic entities from text
- **Auto-discovery of extraction domains** — `ThemeRegistry` scans `models/*/catalog.py` at startup; no registration code needed when a new domain is added
- **Dynamic schema composition** — per-node composite Pydantic schemas built at runtime from the annotation decision (primary + complementary models + supplementary fields)
- **Global entity deduplication** — `LabeledEntity` singletons keyed by `(label, normalized_value)` enable cross-document dedup in the graph
- **Versioning & folder hierarchy** — full document version chain in Neo4j; folder structure mirrored as `IS_COMPOSED_OF` relationships
- **Tabular bypass pipeline** — direct CSV/XLSX → Neo4j without LLM extraction stages; only 3 LLM calls per sheet (classify, decide model, map columns)
- **Parallel processing** — `--parallel-docs N` for concurrent document handling at every stage
- **Prompt caching** — Bedrock `cachePoint` support for ~90% reduction in repeated token costs
- **Optional storage layer** — MongoDB backend for raw file + page storage; pipeline runs without it
- **Two retry layers** — `bedrock_retry` (exponential backoff for throttling) + `neo4j_retry` (deadlock-safe writes)

---

## Architecture

```
 Raw Files
 (PDF, DOCX, XLSX, PPTX, CSV, HTML, XML, TXT …)
        │
        ▼
┌──────────────────────┐
│  Stage 0             │  converters/
│  PREPROCESS          │  Mistral OCR (PDF) · python-docx · openpyxl · …
│                      │  → data/json/{filename}.json
└─────────┬────────────┘
          │  {"pages": [{"index": N, "markdown": "…"}]}
          │
          │         CSV / XLSX ──────────────────────────────────────────────┐
          ▼                                                                  │
┌──────────────────────┐                                              ┌──────▼──────────────┐
│  Stage 1             │  extraction/                                 │  Tabular Bypass     │
│  EXTRACT             │  AWS Bedrock · Claude Sonnet                 │  tabular/           │
│                      │  Sliding window → Document tree              │  3 LLM calls / sheet│
│                      │  → data/output/extract-{filename}.json       └──────┬──────────────┘
└─────────┬────────────┘                                                    
          │  Document + StructureNode + InfoUnit                             
          ▼                                                                  
┌──────────────────────┐                                                     
│  Stage 2             │  ingest/                                            
│  INGEST              │  Neo4j MERGE writes (idempotent)    
│                      │  Versioning · folder hierarchy
└─────────┬────────────┘
          │  (:Document)-[:HAS_STRUCTURE]→(:StructureNode)
          ▼
┌──────────────────────┐
│  Stage 3             │  annotation/
│  ANNOTATE            │  LangGraph agent
│                      │  classify_theme → decide_model → format_decision
│                      │  → (:StructureNode)-[:HAS_MODEL_DECISION]→(:ModelDecision)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 4             │  entity_extraction/
│  ENTITY EXTRACT      │  LangGraph agent
│                      │  Dynamic schema composition → typed entity nodes
│                      │  → (:StructureNode)-[:HAS_EXTRACTION]→(:ExtractionResult)
└──────────────────────┘
                              → (:LabeledEntity) global cross-document singletons
```

### Stage summary

| Stage | Directory | Input | Output | LLM? |
|---|---|---|---|---|
| 0 — Preprocess | `converters/` | Raw files | Paged markdown JSON | Mistral OCR (PDF only) |
| 1 — Extract | `extraction/` | Paged markdown JSON | Document tree JSON | Yes (Claude Sonnet) |
| 2 — Ingest | `ingest/` | Document tree JSON | Neo4j nodes & rels | No |
| 3 — Annotate | `annotation/` | Neo4j StructureNodes | `ModelDecision` nodes | Yes (Claude Sonnet) |
| 4 — Entity Extract | `entity_extraction/` | Annotated StructureNodes | `ExtractionResult` + entity nodes | Yes (Claude Sonnet) |
| Tabular | `tabular/` | CSV / XLSX files | Neo4j Table + Row nodes | Yes — 3 calls/sheet |

---

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager |
| Neo4j | 5.x | Local or [AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) |
| AWS account | — | Bedrock access enabled for your chosen Claude model |
| Mistral API key | — | Required only for PDF files |

### Installation

```bash
git clone https://github.com/your-org/scinr-ingest.git
cd scinr-ingest
uv sync
```

### Environment setup

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

```dotenv
# Required — AWS Bedrock
AWS_DEFAULT_REGION=us-east-1
MODEL_ID=us.anthropic.claude-sonnet-4-6
REPAIR_MODEL_ID=us.anthropic.claude-sonnet-4-6

# Required — Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Optional — PDF OCR
MISTRAL_API_KEY=your_mistral_key

# Optional — MongoDB storage backend
STORAGE_BACKEND=mongodb
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=scinr
```

> **AWS region & model ID**: The model ID prefix must match the region group. `us.anthropic.claude-sonnet-4-6` requires `AWS_DEFAULT_REGION` in `us-east-1` or `us-west-2`. For Europe use `eu.` prefix with `eu-central-1`; for Asia Pacific use `ap.` prefix with `ap-northeast-1`.

### Run the full pipeline

```bash
# Full pipeline: raw files → knowledge graph
python main.py --stage all --input-raw files/ --input data/json/ --output data/output/

# Full pipeline with parallel processing
python main.py --stage all --input-raw files/ --input data/json/ --output data/output/ --parallel-docs 4
```

---

## Configuration

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AWS_DEFAULT_REGION` | Yes | `us-east-1` | AWS region; must match the model ID prefix |
| `MODEL_ID` | Yes | — | Bedrock model ID (e.g. `us.anthropic.claude-sonnet-4-6`) |
| `REPAIR_MODEL_ID` | Yes | — | Model used by the repair loop (can be the same as `MODEL_ID`) |
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `MISTRAL_API_KEY` | No | — | Mistral OCR API key; required only for PDF conversion |
| `STORAGE_BACKEND` | No | — | Set to `mongodb` to enable raw file + page storage |
| `MONGODB_URI` | No | `mongodb://localhost:27017` | MongoDB connection URI (requires `STORAGE_BACKEND=mongodb`) |
| `MONGODB_DATABASE` | No | `scinr` | MongoDB database name |
| `MONGODB_RAW_FILES_COLLECTION` | No | `raw_files` | Collection for raw file metadata |
| `MONGODB_PAGES_COLLECTION` | No | `converted_pages` | Collection for converted page content |
| `MONGODB_GRIDFS_BUCKET` | No | `raw_binaries` | GridFS bucket for raw binary files |
| `EXTRACTION_BATCH_SIZE` | No | `1` | Pages per extraction chunk (increase for denser documents) |
| `PROMPT_CACHING_ENABLED` | No | `true` | Enable Bedrock prompt caching (`cachePoint`) |

AWS credentials can also be supplied via `~/.aws/credentials` or an IAM role; environment variables take precedence.

### CLI reference

```bash
python main.py [options]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--stage` | `preprocess` \| `extract` \| `ingest` \| `annotate` \| `entity_extract` \| `tabular` \| `all` | `all` | Pipeline stage(s) to run |
| `--input` | `DIR` | `data/json/` | Intermediate JSON folder (Stage 1 input) |
| `--input-raw` | `DIR` | — | Raw source files folder (Stage 0 input) |
| `--output` | `DIR` | `data/output/` | Extracted JSON folder (Stage 1 output / Stage 2 input) |
| `--document` | `NAME` | — | Document name — required for `annotate` and `entity_extract` |
| `--update` | flag | off | Re-ingest into the existing latest version without creating a new one |
| `--replaces` | `NAME` | — | Link the ingested document as successor of this existing document |
| `--parallel-docs` | `N` | `1` | Concurrent documents (1 = sequential) |
| `--only-unannotated` | flag | off | `annotate`: skip nodes that already have a `ModelDecision` |
| `--only-unextracted` | flag | off | `entity_extract`: skip nodes that already have an `ExtractionResult` |
| `--manual` | flag | off | `annotate`: assign a fixed model to all nodes without LLM |
| `--model` | `CLASS_NAME` | — | CamelCase model class name for `--manual` annotation |
| `--window-size` | `N` | `2` | Sliding window size in pages (informational; currently fixed at 2) |

---

## Pipeline Stages

### Stage 0 — Preprocess

**Directory:** `converters/`  
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

Converters are registered in `converters/registry.py` (keyed by file extension), so the pipeline auto-selects the right converter for each file. When a storage backend is configured, Stage 0 also stores the raw binary and converted pages in MongoDB.

```bash
python main.py --stage preprocess --input-raw files/ --input data/json/
```

See [`converters/README.md`](converters/README.md) for converter details and how to add new formats.

---

### Stage 1 — Extract

**Directory:** `extraction/`  
**Input:** `data/json/{filename}.json`  
**Output:** `data/output/extract-{filename}.json`  
**LLM:** Claude Sonnet via AWS Bedrock — temperature `0.0`, max tokens `65 536`

Reads the paged JSON and processes pages through a **sliding window** (default 2 pages per chunk). For each window:

1. Sends page content plus the active document hierarchy as context to the LLM.
2. Receives back a `DocumentStructure` Pydantic tree (sections, subsections, tables, info units, verbatim quotes).
3. Merges the chunk result into the growing `Document` object.
4. **Crash-safe write**: writes the intermediate JSON to disk after every successfully processed chunk.

A **2-phase extraction + repair loop** handles malformed LLM output: if Pydantic validation fails, a dedicated repair model retries up to 3 times with escalating temperatures (`0.0 → 0.3 → 0.6`). The repair logic is shared across all LLM stages via `utils/llm_repair.py`.

```bash
python main.py --stage extract --input data/json/ --output data/output/ --parallel-docs 4
```

---

### Stage 2 — Ingest

**Directory:** `ingest/`  
**Input:** `data/output/extract-{filename}.json`  
**Target:** Neo4j database

Reads the extracted `Document` JSON and writes it to Neo4j using **`MERGE` statements throughout** — every write is idempotent, so re-running the stage on the same document is always safe. All writes for a single document are wrapped in one transaction.

Key behaviours:

- **Deterministic UIDs** — `InfoUnit` nodes receive a SHA-256[:16] hash derived from their content, so identical text fragments always map to the same node across runs.
- **Folder hierarchy** — when a `folder_path` is present, ancestor `(:Document)` nodes are created and connected via `[:IS_COMPOSED_OF]` relationships, mirroring the source directory tree inside the graph.
- **Versioning** — each ingest increments the `version` counter and sets `latest=true` on the new node while all previous versions become `latest=false`.

```bash
# Ingest a specific folder
python main.py --stage ingest --output data/output/

# Update in-place (no new version created)
python main.py --stage ingest --output data/output/ --update

# Link as successor of another document
python main.py --stage all --input-raw files/new/ --input data/json/ --output data/output/ --replaces "OldDocumentName"
```

---

### Stage 3 — Annotate

**Directory:** `annotation/`  
**Requires:** Stage 2 completed for the target document

Runs a **LangGraph `StateGraph` agent** over every `(:StructureNode)` in the graph for the given document and assigns it an extraction model. The agent loop:

```
load_nodes → [check_done] → prepare_node → classify_theme
           ↑                                     ↓
      write_decision ←──────────────── decide_model
```

For each node the agent reads the theme assigned during Stage 1, then makes a structured LLM call producing an `AnnotationDecision` (primary model, complementary models, supplementary fields, and rationale). Results are written as `(:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)` nodes.

The **ThemeRegistry** auto-discovers all `models/*/catalog.py` files at startup and presents their `THEME_DESCRIPTION` and `SELECTABLE_MODELS` to the LLM — no code changes needed when a new domain is added.

```bash
# Annotate all nodes
python main.py --stage annotate --document "MyDocument"

# Resume — skip already-annotated nodes
python main.py --stage annotate --document "MyDocument" --only-unannotated

# Manual override — assign a fixed model without LLM
python main.py --stage annotate --document "MyDocument" --manual --model "Triple"
```

---

### Stage 4 — Entity Extract

**Directory:** `entity_extraction/`  
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
# Extract entities for all annotated nodes
python main.py --stage entity_extract --document "MyDocument"

# Resume — skip nodes that already have an ExtractionResult
python main.py --stage entity_extract --document "MyDocument" --only-unextracted --parallel-docs 4
```

---

### Tabular Bypass

**Directory:** `tabular/`  
**Input:** CSV / XLSX files  
**Target:** Neo4j database  
**LLM:** 3 calls per sheet (classify theme → decide model → map columns)

Ingests tabular files directly into Neo4j, bypassing Stages 0–4. For each sheet:

1. Reads headers and a 5-row preview.
2. Classifies the sheet theme.
3. Selects an extraction model.
4. Maps sheet columns to model fields.
5. Writes `(:Document)-[:HAS_STRUCTURE]->(:StructureNode:Table)-[:HAS_CHILD]->(:StructureNode:Row)` subgraph.

```bash
# Ingest all CSV/XLSX files in a folder
python main.py --stage tabular --input-raw files/data/

# Update mode — wipe and re-ingest
python main.py --stage tabular --input-raw files/data/ --update
```

The tabular pipeline is also automatically invoked by `--stage all` when CSV/XLSX files are present in `--input-raw`.

---

## Extraction Domain Models

### Model system overview

Extraction models are Pydantic classes that define the typed entities the pipeline should extract from a structural node. They are organised into **themes** (extraction domains) under `models/`:

```
models/
├── document_structure.py          ← Core pipeline models (NOT extraction)
├── default/
│   └── catalog.py                 ← Triple — RDF fallback (always included)
├── structural_specs/
│   └── catalog.py                 ← DocumentStructure example domain
├── pharmaceutical_quality/        ← Proprietary reference (not open-source API)
├── equipment_qualification/       ← Proprietary reference (not open-source API)
└── pharma_operations/             ← Proprietary reference (not open-source API)
```

### Open-source models

| Theme | Model(s) | Description |
|---|---|---|
| `default` | `Triple` | RDF subject-predicate-object fallback; used when no other theme matches |
| `structural_specs` | `DocumentStructure` | Example: models for documents that prescribe how other documents must be structured |

### Proprietary reference examples

The `pharmaceutical_quality/`, `equipment_qualification/`, and `pharma_operations/` folders contain domain-specific models for pharmaceutical/scientific use cases. They are included in the repository as **reference implementations** to illustrate the model system, but they are not part of the open-source API surface. You are free to study them, but they may be removed or made private in a future release.

### How ThemeRegistry works

`ThemeRegistry` (in `utils/theme_registry.py`) scans `models/*/catalog.py` at startup using Python's `importlib`. Every folder containing a `catalog.py` is registered as a theme. **No registration code needs to be touched when a new domain is added.**

Each `catalog.py` must export exactly two names:

```python
# models/my_domain/catalog.py

THEME_DESCRIPTION: str = (
    "One-line description used by the Stage 3 classification LLM to pick this theme."
)

SELECTABLE_MODELS: list[type[ExtractionModel]] = [MyModelA, MyModelB]
```

### Adding a new extraction domain

1. Create `models/<your_theme>/` with `__init__.py` and `catalog.py`.
2. Define `THEME_DESCRIPTION` and `SELECTABLE_MODELS`.
3. Write Pydantic models that inherit from `ExtractionModel`.
4. No other changes needed — `ThemeRegistry` picks them up automatically.

#### Entity field annotations

Fields can carry `json_schema_extra` metadata to control how Neo4j graph nodes and relationships are created:

```python
from pydantic import Field
from models.base import ExtractionModel

class ActiveSubstance(ExtractionModel):
    """Active pharmaceutical ingredient with dose information."""

    # Creates a global (:LabeledEntity {label: "Substance"}) singleton
    name: str = Field(..., json_schema_extra={"entity_label": "Substance"})

    # Creates a typed Neo4j relationship between entity nodes
    dose: str = Field(
        ...,
        json_schema_extra={
            "field_relationships": [
                {"from_field": "name", "rel_type": "HAS_DOSE", "to_field": "dose"}
            ]
        }
    )
```

See [`model-creation/README.md`](model-creation/README.md) for the full guide including worked examples, entity label conventions, nested models, and `field_relationships` syntax.

---

## Neo4j Schema

### Node labels

| Label | Key Properties | Description |
|---|---|---|
| `:Document` | `name`, `path`, `version`, `latest` | Document root. `latest=true` on the current version |
| `:StructureNode` + role label | `node_id`, `title`, `role`, `theme` | One node per structural element; also labeled `:Section`, `:Subsection`, `:Table`, `:Appendix`, `:FieldGroup`, or `:FreeformBlock` |
| `:InfoUnit` | `uid` (SHA-256[:16]), `title`, `description` | Semantic concept unit extracted from a node's body text |
| `:ModelDecision` | `node_id` | Stage 3 annotation result for a StructureNode |
| `:CatalogModel` | `name` | Registered Pydantic extraction model |
| `:Theme` | `name`, `path` | Extraction domain |
| `:ExtractionResult` | `node_id` | Stage 4 entity extraction result for a StructureNode |
| `:LabeledEntity` | `label`, `normalized_value` | Global entity singleton — shared across documents |
| `:Entity` | `normalized_value` | Triple-fallback entity (RDF subject / object) |

### Key relationships

| Relationship | From → To | Description |
|---|---|---|
| `HAS_STRUCTURE` | `:Document` → `:StructureNode` | Document root to top-level nodes |
| `HAS_CHILD` | `:StructureNode` → `:StructureNode` | Parent → child in the document tree |
| `HAS_INFO_UNIT` | `:StructureNode` → `:InfoUnit` | Node to its semantic information units |
| `IS_COMPOSED_OF` | `:Document` → `:Document` | Folder hierarchy (parent folder → child document) |
| `HAS_NEWER_VERSION` | `:Document` → `:Document` | Version chain: old version → new version |
| `HAS_MODEL_DECISION` | `:StructureNode` → `:ModelDecision` | Stage 3 annotation result |
| `MATCHED_MODEL` | `:ModelDecision` → `:CatalogModel` | Primary model selected for a node |
| `HAS_EXTRACTION` | `:StructureNode` → `:ExtractionResult` | Stage 4 result link |
| `REFERENCES` | `:ExtractionResult` → `:LabeledEntity` | Extraction → global entity singleton |
| `<REL_TYPE>` | `:LabeledEntity` → `:LabeledEntity` | Field relationship (defined in model metadata) |

### Cypher examples

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

### Document versioning

Every ingest run auto-increments the `version` counter and marks only the newest node as `latest=true`.

| Scenario | CLI flag | Behaviour |
|---|---|---|
| First ingest | *(none)* | Creates version 1 with `latest=true` |
| Re-ingest (correction) | `--update` | Wipes and re-ingests into the existing latest version; no new version node |
| New version | *(none, run again)* | Creates version N+1, links via `HAS_NEWER_VERSION`, sets `latest=true` |
| Document supersedes another | `--replaces <name>` | Links the new document as the successor of the named existing document |

---

## Storage Layer

The storage layer is **fully optional**. When `STORAGE_BACKEND` is not set, the pipeline runs without MongoDB and all storage calls are silently skipped.

When enabled (`STORAGE_BACKEND=mongodb`), the storage layer persists:

| Collection / Bucket | Contents |
|---|---|
| `raw_files` (MongoDB) | Metadata for each ingested raw file (filename, content type, folder path, upload timestamp) |
| `converted_pages` (MongoDB) | Converted page content (markdown) + source page metadata per page |
| `raw_binaries` (GridFS) | Binary content of raw source files (PDF bytes, DOCX bytes, etc.) |

The `raw_file_id` and `page_id` fields stored in Neo4j nodes allow cross-referencing back to the original binary and page content in MongoDB.

### Enable / disable storage

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
> MongoDB creates the database automatically on the first write — no prior `CREATE DATABASE` step is needed. See [`storage/README.md`](storage/README.md) for the full setup guide including user creation and Atlas configuration.

The storage backend abstraction lives in `storage/base.py` and `storage/factory.py`. Additional backends (e.g. PostgreSQL, S3) can be added by implementing the base interface.

---

## Development

### Adding a new file format converter

1. Create `converters/<format>.py` and subclass `BaseConverter` from `converters/base.py`.
2. Set `supported_extensions: list[str]` on the class.
3. Implement `async convert(file_path: Path) -> dict` returning the paged JSON envelope:
   ```python
   {
       "pages": [{"index": 0, "markdown": "…"}],
       "folder_path": None,  # or a relative subfolder string
   }
   ```
4. Register the converter in `converters/registry.py` by adding it to the `CONVERTERS` dict keyed by extension.

See [`converters/README.md`](converters/README.md) for the full `BaseConverter` interface and working examples.

### Adding a new extraction domain

1. Create `models/<your_theme>/` with `__init__.py` and `catalog.py`.
2. Export `THEME_DESCRIPTION: str` and `SELECTABLE_MODELS: list[type[ExtractionModel]]` from `catalog.py`.
3. Write Pydantic model classes that inherit from `ExtractionModel`.
4. No other changes required — `ThemeRegistry` picks up the new domain automatically on next startup.

See [`model-creation/README.md`](model-creation/README.md) for worked examples, entity label conventions, nested model patterns, and `field_relationships` syntax.

### Project structure

```
scinr-ingest/
├── main.py                        ← Pipeline orchestrator & CLI entry point
├── pyproject.toml                 ← Dependencies (uv)
├── .env.example                   ← Environment variable template
│
├── converters/                    ← Stage 0: raw files → intermediate JSON
│   ├── registry.py                ←   Converter auto-selection by extension
│   ├── base.py                    ←   BaseConverter interface
│   ├── pdf.py                     ←   Mistral OCR
│   ├── docx.py / xlsx.py / …      ←   One file per format
│   └── README.md
│
├── extraction/                    ← Stage 1: paged JSON → Document tree (LLM)
├── ingest/                        ← Stage 2: Document tree → Neo4j
├── annotation/                    ← Stage 3: LangGraph annotation agent
├── entity_extraction/             ← Stage 4: LangGraph entity extraction agent
├── tabular/                       ← Tabular bypass: CSV/XLSX → Neo4j
├── storage/                       ← Optional MongoDB storage layer
│
├── models/                        ← All Pydantic extraction domain models
│   ├── document_structure.py      ←   Core pipeline schema (Document, StructureNode, …)
│   ├── default/                   ←   Triple — RDF fallback (open-source)
│   ├── structural_specs/          ←   DocumentStructure example (open-source)
│   ├── pharmaceutical_quality/    ←   Reference domain (proprietary examples)
│   ├── equipment_qualification/   ←   Reference domain (proprietary examples)
│   └── pharma_operations/         ←   Reference domain (proprietary examples)
│
├── prompts/                       ← LLM system prompt for Stage 1 extraction
├── utils/                         ← Shared utilities
│   ├── theme_registry.py          ←   ThemeRegistry: auto-discovery of extraction domains
│   ├── llm_repair.py              ←   Generic LLM repair loop (all stages)
│   ├── bedrock_retry.py           ←   Exponential backoff for Bedrock throttling
│   ├── neo4j_retry.py             ←   Retry layer for Neo4j deadlocks
│   └── llm_factory.py             ←   LLM client factory
│
└── model-creation/                ← Developer guide: adding new extraction models
    └── README.md
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch from `main`.
2. **Install dependencies** with `uv sync`.
3. **Make your changes** following the existing code style (async functions, Pydantic v2 models, explicit error handling, no magic values).
4. **Test your changes** manually against a local Neo4j instance with a small set of sample documents.
5. **Open a pull request** with a clear description of what changed and why.

### Areas where contributions are especially welcome

- New file format converters (`converters/`)
- New open-source extraction domain models (`models/`)
- Additional storage backends (`storage/`)
- Performance improvements to the extraction or entity extraction agents
- Documentation and examples

### Reporting issues

Please open a GitHub issue with:
- A minimal reproducible example (document type, CLI command used)
- The full error traceback
- Your Python version, OS, and Neo4j version

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
