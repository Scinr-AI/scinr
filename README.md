# scinr

[![PyPI version](https://img.shields.io/pypi/v/scinr.svg)](https://pypi.org/project/scinr/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-scinr.ai-5e35b1.svg)](https://scinr-ai.github.io/scinr/)
[![llms.txt](https://img.shields.io/badge/agent-llms.txt-4400ff.svg)](https://scinr-ai.github.io/scinr/llms.txt)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**scinr** is an AI-powered document knowledge library for the life sciences industry, built by [Scinr AI](https://www.scinr.com/).

📖 **[Documentation Site](https://scinr-ai.github.io/scinr/)** &nbsp;|&nbsp; 🤖 **[llms.txt for Agents](https://scinr-ai.github.io/scinr/llms.txt)** ([Full Context](https://scinr-ai.github.io/scinr/llms-full.txt))

Its **`newton` module** (`scinr.newton`) is the document ingestion engine: a complete 5-stage LLM pipeline that converts raw documents (PDF, DOCX, XLSX, PPTX, HTML, CSV, JSON, XML), images, audio and video into structured knowledge graphs (Neo4j, PostgreSQL or SQLite).

---

## About Scinr

SCINR is the first agentic memory platform purpose-built for life sciences domains. Learn more at [scinr.com](https://www.scinr.com/).

`scinr` is the open library layer of the Newton platform, providing AI-driven extraction of knowledge from complex datasets and regulatory documents. Its `newton` module is the structured and unstructured ingestion foundation.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [LLM Providers](#llm-providers)
7. [Pipeline Stages](#pipeline-stages)
8. [Extraction Domain Models](#extraction-domain-models)
9. [Project Structure](#project-structure)
10. [Further Reading](#further-reading)
11. [Contributing](#contributing)
12. [License](#license)

---

## Features

- **Multi-format ingestion** — PDF (Mistral OCR), DOCX, XLSX, PPTX, CSV, HTML, TXT, MD, XML/SOAP APIs, JSON APIs — all normalized to the same intermediate format
- **Images, audio and video** — image files transcribed and described by a configurable vision model; audio and video transcribed segment by segment with time ranges (video frames described too); images embedded in PDF, DOCX and PPTX described in place. Cloud or self-hosted providers (OpenAI-compatible endpoints, Bedrock, Ollama, faster-whisper)
- **Provider-agnostic** — OpenAI, AWS Bedrock, Ollama, Anthropic, or any LangChain-compatible model
- **LLM-powered extraction** — sliding-window 2-phase extraction with an automatic repair loop; crash-safe intermediate writes after every chunk
- **Knowledge graph output** — idempotent graph writes (upserts throughout; `MERGE` on Neo4j); safe to re-run at any stage
- **Two independent LangGraph agents** — Annotation agent assigns an extraction model to each structural node; Entity Extraction agent pulls typed Pydantic entities from text
- **Auto-discovery of extraction domains** — `ThemeRegistry` scans `models/*/catalog.py` at startup; no registration code needed when a new domain is added
- **Dynamic schema composition** — per-node composite Pydantic schemas built at runtime from the annotation decision (primary + complementary models + supplementary fields)
- **Global entity deduplication** — `LabeledEntity` singletons keyed by `(label, normalized_value)` enable cross-document dedup in the graph
- **Cross-document model linking** — `instance_relationships` create typed edges between `ModelInstance` nodes across different document sections; forward-reference shell nodes are merged when the target model is later extracted
- **Versioning & folder hierarchy** — full document version chain in the graph; folder structure mirrored as `IS_COMPOSED_OF` relationships
- **Document lifecycle** — `delete_document()` removes a document and its whole cascade; `freeze_document()` archives its subgraph to a snapshot and leaves a stub in the graph, and `restore_document()` rebuilds it with no LLM calls. All three are tenant-scoped and work in bounded transactions, so, on Neo4j, they fit in the transaction memory whatever the size of the folder. See the [Document Freezing guide](https://scinr-ai.github.io/scinr/user-guides/document-freezing/)
- **Read-back navigation API** — `scinr.newton.navigation`: a read-only, `async`, engine-abstracted layer of ~90 typed methods over the graph (documents, structure nodes, model instances, entities, schema introspection) — no Cypher required. See the [Graph Navigation guide](https://scinr-ai.github.io/scinr/user-guides/graph-navigation/)
- **Tabular bypass pipeline** — direct CSV/XLSX → graph without LLM extraction stages; only 3 LLM calls per sheet (classify, decide model, map columns)
- **Parallel processing** — `parallel_docs=N` for concurrent document handling at every stage
- **Live progress reporting** — `run_pipeline(callbacks=[...])` emits an event at every stage boundary of every document; use the built-in `ProgressTracker` (`snapshot().percent`) or your own sync/async callbacks to update a job in your database. See the [Progress Tracking guide](https://scinr-ai.github.io/scinr/user-guides/progress-tracking/)
- **Prompt caching** — Bedrock `cachePoint` support for ~90% reduction in repeated token costs
- **Optional storage layer** — MongoDB, PostgreSQL or SQLite backend for raw file + page storage; pipeline runs without it
- **Choice of graph engine** — Neo4j (default), PostgreSQL or SQLite, selected by `graph_backend`; see the [SQL Graph Backends guide](https://scinr-ai.github.io/scinr/user-guides/sql-graph/)
- **Two retry layers** — `bedrock_retry` (exponential backoff for throttling) + Neo4j retry (`graph/neo4j/retry.py`, deadlock-safe writes)

---

## Architecture

```
 Raw Files
 (PDF, DOCX, XLSX, PPTX, CSV, HTML, XML, TXT, images, audio, video …)
        │
        ▼
┌──────────────────────┐
│  Stage 0             │  converters/
│  PREPROCESS          │  Mistral OCR (PDF) · python-docx · openpyxl · vision_llm · transcriber + ffmpeg · …
│                      │  → data/json/{filename.ext}.json
└─────────┬────────────┘
          │  {"pages": [{"index": N, "markdown": "…"}]}
          │
          │         CSV / XLSX ──────────────────────────────────────────────┐
          ▼                                                                  │
┌──────────────────────┐                                              ┌──────▼──────────────┐
│  Stage 1             │  extraction/                                 │  Tabular Bypass     │
│  EXTRACT             │  any LangChain BaseChatModel                 │  tabular/           │
│                      │  Sliding window → Document tree              │  3 LLM calls / sheet│
│                      │  → data/output/extract-{filename}.json       └─────────────────────┘
└─────────┬────────────┘                                                    
          │  Document + StructureNode + InfoUnit                             
          ▼                                                                  
┌──────────────────────┐                                                     
│  Stage 2             │  ingest/                                            
│  INGEST              │  Graph writes (idempotent upserts)  
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

---

## Installation

**Requirements:** Python 3.11+, and one graph engine: a running Neo4j instance (5+), **or** PostgreSQL, **or** SQLite (no server); for audio and video, the `ffmpeg` / `ffprobe` system binaries

```bash
pip install scinr
```

Install with your preferred LLM provider:

```bash
pip install 'scinr[openai]'    # OpenAI / Azure OpenAI
pip install 'scinr[bedrock]'   # AWS Bedrock (Claude)
pip install 'scinr[ollama]'    # Ollama (local models)
pip install 'scinr[mongodb]'   # Optional MongoDB storage layer
pip install 'scinr[media]'     # Local transcription (faster-whisper) and HEIC/HEIF images
```

Install the extra for your graph engine:

```bash
pip install 'scinr[neo4j]'     # Neo4j driver (the default graph backend)
pip install 'scinr[postgres]'  # PostgreSQL (SQLAlchemy + psycopg 3)
pip install 'scinr[sqlite]'    # SQLite (SQLAlchemy + aiosqlite)
```

The default graph backend is `neo4j`, and the Neo4j driver is an optional extra: without `scinr[neo4j]` it fails at first use with `graph_backend='neo4j' requires the optional dependency: pip install 'scinr[neo4j]'`. For a no-Docker start, see the [SQLite recipe](docs/user-guides/quick-start.md#sqlite-no-docker) and the [PostgreSQL recipe](docs/user-guides/quick-start.md#postgresql-with-docker).

For development:

```bash
git clone https://github.com/Scinr-AI/scinr.git
cd scinr
uv sync --all-extras
```

---

## Quick Start

### Configuration in code

```python
from scinr.newton import configure, run_pipeline
from langchain_openai import ChatOpenAI
import asyncio

# Configure once at startup
configure(
    llm=ChatOpenAI(model="gpt-4o"),
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your-password",
)

# Run the full pipeline
result = asyncio.run(run_pipeline(input_raw="files/"))
print(f"Processed {result.total_documents} documents")
```

**AWS Bedrock variant:**

```python
from langchain_aws import ChatBedrockConverse
from scinr.newton import configure, run_pipeline

configure(
    llm=ChatBedrockConverse(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1",
    ),
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your-password",
)
```

The example above uses Neo4j, the default graph backend. To run on PostgreSQL or SQLite, pass `graph_backend`, `storage_backend` and `sql_url` instead of the `neo4j_*` settings; see the [SQL Graph Backends guide](docs/user-guides/sql-graph.md).

### Configuration from environment variables

Every `configure()` parameter you omit is read from the environment. Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

```dotenv
# LLM — configured in code via configure(llm=...), NOT here.
# Provider SDK settings still come from the environment:
AWS_DEFAULT_REGION=us-east-1

# Required when GRAPH_BACKEND=neo4j (the default) — Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Required to ingest any PDF (PDF conversion is Mistral OCR only)
MISTRAL_API_KEY=your_mistral_key

# Optional — MongoDB storage backend
STORAGE_BACKEND=mongodb
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=scinr

# Alternative to Neo4j — PostgreSQL or SQLite (instead of the NEO4J_* block)
# GRAPH_BACKEND=sqlite
# STORAGE_BACKEND=sqlite
# SQL_URL=sqlite:///newton.db
```

Then pass only the LLM in code and run the pipeline:

```python
configure(llm=ChatOpenAI(model="gpt-4o"))  # the graph, Mistral and storage settings come from the environment

result = asyncio.run(run_pipeline(input_raw="files/", parallel_docs=4))
```

---

## Configuration

### Library mode — `configure()` key parameters

| Parameter | Type | Description |
|---|---|---|
| `llm` | `BaseChatModel \| None` | LangChain model for all LLM calls. Optional at `configure()` time; needed for LLM pipeline stages |
| `repair_llm` | `BaseChatModel` | Model for the JSON repair loop; falls back to `llm` if not set |
| `graph_backend` | `str` | `'neo4j'` (default), `'postgres'` or `'sqlite'` |
| `neo4j_uri` | `str` | Neo4j connection URI. Default: `bolt://localhost:7687` |
| `neo4j_user` | `str` | Neo4j username (required when `graph_backend='neo4j'`) |
| `neo4j_password` | `str` | Neo4j password (required when `graph_backend='neo4j'`) |
| `sql_url` | `str` | SQLAlchemy URL for the SQL backends (`postgresql://…` or `sqlite:///…`); required when any backend is `postgres` or `sqlite` |
| `mistral_api_key` | `str` | Mistral OCR API key. Required to process any PDF — PDF conversion is Mistral OCR only, with no fallback |
| `storage_backend` | `str` | `'none'` (default), `'mongodb'`, `'postgres'`, `'sqlite'` or `'custom'` |
| `extraction_batch_size` | `int` | Pages per extraction chunk. Default: `1` |
| `llm_concurrency` | `int` | Max concurrent LLM calls (semaphore). Default: `4` |
| `enabled_base_themes` | `list[str]` | Whitelist of built-in theme paths to activate; `None` activates all |
| `extra_models_paths` | `list[Path]` | Filesystem paths to scan for user-defined themes |

For the full parameter reference, see the [module README](src/scinr/newton/README.md#configure).

### Environment variables

`configure()` reads these when the matching parameter is omitted. The usual way to set the LLM is to build a LangChain chat model and pass it to `configure(llm=...)`; the only environment shortcut is `MODEL_ID`, a Bedrock shortcut used when `llm` is omitted. Provider SDK settings (`AWS_DEFAULT_REGION`, `OPENAI_API_KEY`, …) come from the environment as each SDK expects.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GRAPH_BACKEND` | No | `neo4j` | Graph engine: `neo4j`, `postgres` or `sqlite` |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | If `GRAPH_BACKEND=neo4j` | — | Neo4j username (also accepts `NEO4J_AUTH=user/password`) |
| `NEO4J_PASSWORD` | If `GRAPH_BACKEND=neo4j` | — | Neo4j password (also accepts `NEO4J_AUTH=user/password`) |
| `NEO4J_DATABASE` | If `GRAPH_BACKEND=neo4j` | — | Neo4j target database name |
| `SQL_URL` | If any backend is `postgres` / `sqlite` | — | SQLAlchemy URL shared by all SQL backends |
| `SQL_SCHEMA` | No | `newton` | Postgres schema that holds the tables (ignored by SQLite) |
| `SQL_POOL_SIZE` | No | `10` | Connection pool size of each SQL engine |
| `SQL_FULLTEXT_CONFIG` | No | `simple` | Postgres text-search configuration (applied when the schema is created) |
| `SQLITE_FILES_DIR` | No | next to the `.db` | Folder for the binaries kept by the SQLite backend |
| `MISTRAL_API_KEY` | For PDFs | — | Mistral OCR API key; required to ingest any PDF (no non-Mistral path) |
| `STORAGE_BACKEND` | No | `none` | `none`, `mongodb`, `postgres`, `sqlite` or `custom`; enables raw file + page storage |
| `MONGODB_URI` | No | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DATABASE` | No | `scinr` | MongoDB database name |
| `MONGODB_RAW_FILES_COLLECTION` | No | `raw_files` | Collection for raw file metadata |
| `MONGODB_PAGES_COLLECTION` | No | `converted_pages` | Collection for converted page content |
| `MONGODB_GRIDFS_BUCKET` | No | `raw_binaries` | GridFS bucket for raw binary files |
| `FREEZE_BACKEND` | No | the storage backend | Backend for document snapshots (`freeze_document()` / `restore_document()`): `none`, `mongodb`, `postgres`, `sqlite` or `custom` |
| `MONGODB_FROZEN_COLLECTION` | No | `frozen_documents` | Collection for snapshot metadata |
| `MONGODB_FROZEN_GRIDFS_BUCKET` | No | `frozen_snapshots` | GridFS bucket for snapshot files |
| `EXTRACTION_BATCH_SIZE` | No | `1` | Pages per extraction chunk |
| `LLM_CONCURRENCY` | No | `4` | Max concurrent LLM calls |
| `PROMPT_CACHING_ENABLED` | No | `true` | Enable Bedrock prompt caching (`cachePoint`) |
| `MODEL_ID` | No | — | Shortcut for AWS Bedrock: used only when `llm` is omitted; newton then builds a `ChatBedrockConverse` (reads `AWS_DEFAULT_REGION`, default `us-east-1`, and `MAX_TOKENS`, default `65536`). Passing `llm=` is the usual way |
| `SCINR_EXTRA_MODELS_PATHS` | No | — | Colon-separated paths to user-defined theme directories |

> **AWS region & model ID** (when using `ChatBedrockConverse`): the model ID prefix must match the region group. `us.anthropic.claude-sonnet-4-6` needs `AWS_DEFAULT_REGION` in `us-east-1` / `us-west-2`; Europe uses the `eu.` prefix with `eu-central-1`; Asia Pacific uses `ap.` with `ap-northeast-1`. Set `region_name=` on the model.

---

## LLM Providers

`scinr` accepts any [LangChain `BaseChatModel`](https://python.langchain.com/docs/concepts/chat_models/) via `configure(llm=...)`.

| Provider | Extra | Example |
|---|---|---|
| OpenAI / Azure | `[openai]` | `ChatOpenAI(model="gpt-4o")` |
| AWS Bedrock | `[bedrock]` | `ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")` |
| Ollama (local) | `[ollama]` | `ChatOllama(model="llama3.1")` |
| Anthropic | — | `ChatAnthropic(model="claude-3-5-sonnet-20241022")` |
| Any LangChain model | — | Pass any `BaseChatModel` instance |

> You can set a cheaper `repair_llm` model for JSON repair retries:
>
> ```python
> configure(
>     llm=ChatOpenAI(model="gpt-4o"),
>     repair_llm=ChatOpenAI(model="gpt-4o-mini"),
> )
> ```

---

## Pipeline Stages

| Stage | Name | Input → Output | LLM? |
|---|---|---|---|
| 0 | Preprocess | Raw files → Paged markdown JSON | Mistral OCR (PDF only) |
| 1 | Extract | Paged markdown JSON → Document tree JSON | Yes |
| 2 | Ingest | Document tree JSON → graph nodes & relationships | No |
| 3 | Annotate | Graph StructureNodes → `ModelDecision` nodes | Yes |
| 4 | Entity Extract | Annotated StructureNodes → `ExtractionResult` + entity nodes | Yes |
| — | Tabular | CSV / XLSX → graph Table + Row nodes | Yes (3 calls/sheet) |

Run them all with `run_pipeline()`, a subset with `run_pipeline(stages=[...])`, or one at a time with `run_preprocess()`, `run_extraction()`, `run_ingestion()`, `run_annotation()`, `run_entity_extraction()` and `run_tabular_pipeline()`. For every parameter, see the [Running the Pipeline guide](docs/user-guides/running-pipeline.md); for per-stage examples, input/output formats, and agent diagrams, see the [module README](src/scinr/newton/README.md#pipeline-stages--detailed-reference).

---

## Extraction Domain Models

### Model system overview

Extraction models are Pydantic classes that define the typed entities the pipeline should extract from a structural node. They are organised into **themes** (extraction domains) under `models/`. `ThemeRegistry` scans `models/*/catalog.py` at startup using Python's `importlib` — no registration code is needed when a new domain is added.

### Open-source models

| Theme | Model | Description |
|---|---|---|
| `default` | `Triple` | RDF subject-predicate-object fallback; used when no other theme matches |

### How to add an extraction domain

1. Create `models/<your_theme>/` with `__init__.py` and `catalog.py`.
2. Export `THEME_DESCRIPTION: str` and `SELECTABLE_MODELS: list[type[ExtractionModel]]` from `catalog.py`.
3. Write Pydantic model classes that inherit from `ExtractionModel`.
4. No other changes required — `ThemeRegistry` picks up the new domain automatically on next startup.

See [`src/scinr/newton/model-creation/README.md`](src/scinr/newton/model-creation/README.md) for the full guide, including worked examples, entity label conventions, nested model patterns, `field_relationships` syntax (entity-level edges within a model instance), and `instance_relationships` syntax (cross-document model linking via `ModelInstance` nodes).

### Agent skill for model creation

The [`scinr-extraction-models`](skills/scinr-extraction-models/SKILL.md) skill ([Agent Skills](https://agentskills.io) format) teaches coding agents to write themes that follow these rules. It ships with a validator (`scripts/validate_theme.py`) that the agent runs until there are no errors.

```bash
# Any agent supported by the skills CLI (Claude Code, Cursor, Codex, Gemini CLI, Copilot, OpenCode, …)
npx skills add Scinr-AI/scinr --skill scinr-extraction-models

# Claude Code plugin
/plugin marketplace add Scinr-AI/scinr
/plugin install scinr@scinr
```

---

## Project Structure

```
scinr/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── CHANGELOG.md
├── tests/
└── src/
    └── scinr/
        └── newton/                 ← scinr.newton package
            ├── __init__.py         ← Public API
            ├── config.py           ← ScinrConfig + configure()
            ├── pipeline.py         ← run_pipeline()
            ├── results.py          ← Result dataclasses
            ├── exceptions.py       ← Exception hierarchy
            ├── stages/             ← Stage runner functions
            ├── converters/         ← Stage 0: file → paged JSON
            ├── extraction/         ← Stage 1: paged JSON → Document tree
            ├── graph/              ← GraphStore: every write to the graph (Neo4j in graph/neo4j/, SQL in graph/sql/)
            ├── navigation/         ← Read-only graph navigation API (neo4j/ and sql/ implementations)
            ├── sql/                ← Shared PostgreSQL / SQLite layer: engines, schema, migrations, admin
            ├── ingest/             ← Stage 2: Document tree → graph; delete / freeze / restore
            ├── annotation/         ← Stage 3: LangGraph annotation agent
            ├── entity_extraction/  ← Stage 4: LangGraph entity extraction
            ├── tabular/            ← Tabular bypass pipeline
            ├── storage/            ← Optional storage layer (MongoDB, PostgreSQL or SQLite)
            ├── freeze/             ← Snapshot backend for document freezing
            ├── models/             ← Extraction domain models
            │   └── default/        ← Built-in open-source models
            ├── utils/              ← Shared utilities
            └── model-creation/     ← Developer guide (not a Python package)
                └── README.md
```

---

## Further Reading

- **[src/scinr/newton/README.md](src/scinr/newton/README.md)** — Public API reference (`configure()`, `run_pipeline()`, result types, exceptions), detailed stage documentation, Neo4j schema, storage layer configuration, and extension guides.
- **[src/scinr/newton/model-creation/README.md](src/scinr/newton/model-creation/README.md)** — Step-by-step guide for authoring new extraction domain models.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch from `main`.
2. **Install dependencies** with `uv sync --all-extras`.
3. **Make your changes** following the existing code style (async functions, Pydantic v2 models, explicit error handling, no magic values).
4. **Test your changes** with a small set of sample documents. The SQL suites need no Docker for SQLite (`tests/sql`); the Postgres variants run against a disposable `postgres-test` container and are skipped unless `SCINR_SQL_ALLOW_WIPE=1` is set (see `tests/sql/README.md`). Neo4j-backed tests need a local Neo4j instance.
5. **Open a pull request** with a clear description of what changed and why.

### Areas where contributions are especially welcome

- New file format converters (`converters/`)
- New open-source extraction domain models (`models/`)
- Additional storage or graph backends (`storage/`, `graph/`)
- Performance improvements to the extraction or entity extraction agents
- Documentation and examples

### Reporting issues

Please open a GitHub issue with:
- A minimal reproducible example (document type, the `run_pipeline()` / `run_*()` call used)
- The full error traceback
- Your Python version, OS, and graph engine and version (Neo4j, PostgreSQL or SQLite)

---

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
