# scinr

[![PyPI version](https://img.shields.io/pypi/v/scinr.svg)](https://pypi.org/project/scinr/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**scinr** is an AI-powered document knowledge library for the life sciences industry, built by [Scinr AI](https://www.scinr.com/).

Its **`newton` module** (`scinr.newton`) is the document ingestion engine: a complete 5-stage LLM pipeline that converts raw documents (PDF, DOCX, XLSX, PPTX, HTML, CSV, JSON, XML) into structured Neo4j knowledge graphs.

---

## About Scinr

SCINR is the first AI-native supply chain autonomous orchestration platform purpose-built for the life sciences industry. It embeds AI at every step of the supply chain — enabling master data orchestration, continuous planning, and self-healing resiliency — to help pharmaceutical and biotech companies prevent medicine shortages, accelerate time-to-market, and maintain full regulatory compliance. Learn more at [scinr.com](https://www.scinr.com/).

`scinr` is the open library layer of the Newton platform, providing AI-driven extraction of structured knowledge from complex biomedical and supply chain documents. Its `newton` module is the document ingestion foundation.

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
- **Provider-agnostic** — OpenAI, AWS Bedrock, Ollama, Anthropic, or any LangChain-compatible model
- **LLM-powered extraction** — sliding-window 2-phase extraction with an automatic repair loop; crash-safe intermediate writes after every chunk
- **Knowledge graph output** — idempotent Neo4j writes (MERGE throughout); safe to re-run at any stage
- **Two independent LangGraph agents** — Annotation agent assigns an extraction model to each structural node; Entity Extraction agent pulls typed Pydantic entities from text
- **Auto-discovery of extraction domains** — `ThemeRegistry` scans `models/*/catalog.py` at startup; no registration code needed when a new domain is added
- **Dynamic schema composition** — per-node composite Pydantic schemas built at runtime from the annotation decision (primary + complementary models + supplementary fields)
- **Global entity deduplication** — `LabeledEntity` singletons keyed by `(label, normalized_value)` enable cross-document dedup in the graph
- **Cross-document model linking** — `instance_relationships` create typed edges between `ModelInstance` nodes across different document sections; forward-reference shell nodes are merged when the target model is later extracted
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
│  EXTRACT             │  any LangChain BaseChatModel                 │  tabular/           │
│                      │  Sliding window → Document tree              │  3 LLM calls / sheet│
│                      │  → data/output/extract-{filename}.json       └─────────────────────┘
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

---

## Installation

**Requirements:** Python 3.11+, a running Neo4j instance (5.x+)

```bash
pip install scinr
```

Install with your preferred LLM provider:

```bash
pip install 'scinr[openai]'    # OpenAI / Azure OpenAI
pip install 'scinr[bedrock]'   # AWS Bedrock (Claude)
pip install 'scinr[ollama]'    # Ollama (local models)
pip install 'scinr[mongodb]'   # Optional MongoDB storage layer
```

For development:

```bash
git clone https://github.com/Scinr-AI/scinr.git
cd scinr
uv sync --all-extras
```

---

## Quick Start

### Library mode (recommended)

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

### CLI mode

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

```dotenv
# Required — LLM (AWS Bedrock example)
MODEL_ID=us.anthropic.claude-sonnet-4-6
AWS_DEFAULT_REGION=us-east-1

# Required — Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Optional — PDF OCR
MISTRAL_API_KEY=your_mistral_key

# Optional — MongoDB storage backend
STORAGE_BACKEND=mongodb
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=scinr
```

Run the full pipeline:

```bash
newton --stage all --input-raw files/
```

Run with parallel processing:

```bash
newton --stage all --input-raw files/ --parallel-docs 4
```

---

## Configuration

### Library mode — `configure()` key parameters

| Parameter | Type | Description |
|---|---|---|
| `llm` | `BaseChatModel` | LangChain model for all LLM calls (required) |
| `repair_llm` | `BaseChatModel` | Model for the JSON repair loop; falls back to `llm` if not set |
| `neo4j_uri` | `str` | Neo4j connection URI. Default: `bolt://localhost:7687` |
| `neo4j_user` | `str` | Neo4j username (required) |
| `neo4j_password` | `str` | Neo4j password (required) |
| `mistral_api_key` | `str` | Mistral API key for PDF OCR; required only when processing PDF files |
| `storage_backend` | `str` | `'none'` (default), `'mongodb'`, or `'custom'` |
| `extraction_batch_size` | `int` | Pages per extraction chunk. Default: `1` |
| `llm_concurrency` | `int` | Max concurrent LLM calls (semaphore). Default: `4` |
| `enabled_base_themes` | `list[str]` | Whitelist of built-in theme paths to activate; `None` activates all |
| `extra_models_paths` | `list[Path]` | Filesystem paths to scan for user-defined themes |

For the full parameter reference, see the [module README](src/scinr/newton/README.md#configure).

### Environment variables (CLI mode)

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_ID` | Yes* | — | Bedrock model ID (e.g. `us.anthropic.claude-sonnet-4-6`) — *Bedrock only* |
| `AWS_DEFAULT_REGION` | Yes* | `us-east-1` | AWS region; must match model ID prefix — *Bedrock only* |
| `REPAIR_MODEL_ID` | No | same as `MODEL_ID` | Model for the repair loop — *Bedrock only* |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | Yes | — | Neo4j username (also accepts `NEO4J_AUTH=user/password`) |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password (also accepts `NEO4J_AUTH=user/password`) |
| `MISTRAL_API_KEY` | No | — | Mistral OCR API key; required only for PDF conversion |
| `STORAGE_BACKEND` | No | — | Set to `mongodb` to enable raw file + page storage |
| `MONGODB_URI` | No | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DATABASE` | No | `scinr` | MongoDB database name |
| `MONGODB_RAW_FILES_COLLECTION` | No | `raw_files` | Collection for raw file metadata |
| `MONGODB_PAGES_COLLECTION` | No | `converted_pages` | Collection for converted page content |
| `MONGODB_GRIDFS_BUCKET` | No | `raw_binaries` | GridFS bucket for raw binary files |
| `EXTRACTION_BATCH_SIZE` | No | `1` | Pages per extraction chunk |
| `LLM_CONCURRENCY` | No | `4` | Max concurrent LLM calls |
| `PROMPT_CACHING_ENABLED` | No | `true` | Enable Bedrock prompt caching (`cachePoint`) |
| `SCINR_EXTRA_MODELS_PATHS` | No | — | Colon-separated paths to user-defined theme directories |

> **AWS region & model ID**: The model ID prefix must match the region group. `us.anthropic.claude-sonnet-4-6` requires `AWS_DEFAULT_REGION` in `us-east-1` or `us-west-2`. For Europe use `eu.` prefix with `eu-central-1`; for Asia Pacific use `ap.` prefix with `ap-northeast-1`.

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
| 2 | Ingest | Document tree JSON → Neo4j nodes & relationships | No |
| 3 | Annotate | Neo4j StructureNodes → `ModelDecision` nodes | Yes |
| 4 | Entity Extract | Annotated StructureNodes → `ExtractionResult` + entity nodes | Yes |
| — | Tabular | CSV / XLSX → Neo4j Table + Row nodes | Yes (3 calls/sheet) |

**CLI flag reference:**

| Flag | Values | Default | Description |
|---|---|---|---|
| `--stage` | `preprocess` \| `extract` \| `ingest` \| `annotate` \| `entity_extract` \| `tabular` \| `all` | `all` | Pipeline stage(s) to run |
| `--input-raw` | `DIR` | — | Raw source files folder (Stage 0 input) |
| `--input` | `DIR` | `data/json/` | Intermediate JSON folder (Stage 1 input) |
| `--output` | `DIR` | `data/output/` | Extracted JSON folder (Stage 1 output / Stage 2 input) |
| `--document` | `NAME` | — | Document name — required for `annotate` and `entity_extract` |
| `--update` | flag | off | Re-ingest into the existing latest version without creating a new one |
| `--replaces` | `NAME` | — | Link the ingested document as successor of this existing document |
| `--parallel-docs` | `N` | `1` | Concurrent documents (1 = sequential) |
| `--only-unannotated` | flag | off | `annotate`: skip nodes that already have a `ModelDecision` |
| `--only-unextracted` | flag | off | `entity_extract`: skip nodes that already have an `ExtractionResult` |
| `--manual` | flag | off | `annotate`: assign a fixed model to all nodes without LLM |
| `--model` | `CLASS_NAME` | — | CamelCase model class name for `--manual` annotation |
| `--context` | `TEXT` | — | Free-text context passed to extraction and annotation LLMs as additional guidance |

For per-stage CLI commands, input/output formats, and agent diagrams, see the [module README](src/scinr/newton/README.md#pipeline-stages--detailed-reference).

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

---

## Project Structure

```
scinr/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── CHANGELOG.md
├── files/                          ← sample input files
├── tests/
└── src/
    └── scinr/
        └── newton/                 ← scinr.newton package
            ├── __init__.py         ← Public API
            ├── cli.py              ← CLI entry point (newton)
            ├── config.py           ← ScinrConfig + configure()
            ├── pipeline.py         ← run_pipeline()
            ├── results.py          ← Result dataclasses
            ├── exceptions.py       ← Exception hierarchy
            ├── stages/             ← Stage runner functions
            ├── converters/         ← Stage 0: file → paged JSON
            ├── extraction/         ← Stage 1: paged JSON → Document tree
            ├── ingest/             ← Stage 2: Document tree → Neo4j
            ├── annotation/         ← Stage 3: LangGraph annotation agent
            ├── entity_extraction/  ← Stage 4: LangGraph entity extraction
            ├── tabular/            ← Tabular bypass pipeline
            ├── storage/            ← Optional MongoDB storage layer
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
4. **Test your changes** against a local Neo4j instance with a small set of sample documents.
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

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.
