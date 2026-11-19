# Getting Started

This guide walks you through installing `scinr`, configuring your environment, and running your first document ingestion pipeline end-to-end.

---

## Installation

### Core Package

Install `scinr` with `pip`:

```bash
pip install scinr
```

Or with `uv`:

```bash
uv add scinr
```

### Optional Extras

`scinr` ships with optional extras for different LLM providers, storage backends, and development tooling. Install only what you need:

```bash
# AWS Bedrock (recommended — includes langchain-aws and boto3)
pip install "scinr[bedrock]"

# OpenAI (includes langchain-openai)
pip install "scinr[openai]"

# Ollama (includes langchain-ollama)
pip install "scinr[ollama]"

# Graph engine — install the one you use (see the note below)
pip install "scinr[neo4j]"      # Neo4j driver (the default graph backend)
pip install "scinr[postgres]"   # PostgreSQL (SQLAlchemy + psycopg 3)
pip install "scinr[sqlite]"     # SQLite (SQLAlchemy + aiosqlite)

# MongoDB storage (includes motor and pymongo)
pip install "scinr[mongodb]"

# Local transcription (faster-whisper) and HEIC/HEIF images
pip install "scinr[media]"

# Documentation tooling (mkdocs, mkdocstrings, griffe, ruff)
pip install "scinr[docs]"

# Development tooling (pytest, ruff, mypy)
pip install "scinr[dev]"

# Multiple extras at once
pip install "scinr[bedrock,neo4j,mongodb,dev]"
```

!!! warning "Install a graph extra"
    The default graph backend is `neo4j`, and the Neo4j driver is an optional extra. Without `scinr[neo4j]` the first use of the graph fails with `graph_backend='neo4j' requires the optional dependency: pip install 'scinr[neo4j]'`. If you use PostgreSQL or SQLite instead, install `scinr[postgres]` or `scinr[sqlite]`; see the [SQL Graph Backends guide](user-guides/sql-graph.md).

With `uv`:

```bash
uv add "scinr[bedrock]"
uv add "scinr[bedrock,neo4j,mongodb]"
```

---

## Prerequisites

Before running the pipeline, ensure you have the following:

### Required

1. **Python 3.11+** — `scinr` requires Python 3.11 or later.

2. **A graph engine** — one of the following:

   - **Neo4j 5.0+** (the default) — a running instance accessible via the Bolt protocol, plus `pip install "scinr[neo4j]"`. You can run Neo4j locally with Docker:

     ```bash
     docker run -p 7687:7687 -p 7474:7474 \
       -e NEO4J_AUTH=neo4j/your_password \
       neo4j:5
     ```

   - **PostgreSQL** — an existing database that your user can create tables in, plus `pip install "scinr[postgres]"`. newton creates the schema and tables inside the database, but not the database itself nor any role.
   - **SQLite** — no server at all, plus `pip install "scinr[sqlite]"`.

   For a first run without Neo4j, follow the [PostgreSQL with Docker](user-guides/quick-start.md#postgresql-with-docker) or [SQLite (no Docker)](user-guides/quick-start.md#sqlite-no-docker) recipe in the Quick Start. The rest of this page uses Neo4j, the default.

3. **An LLM** — build a LangChain chat model and pass it to `configure(llm=...)`. It is used by the extraction, annotation, and entity-extraction stages (and tabular normalization); `configure()` itself does not require it. Provider credentials come from the environment as each SDK expects:

   - **AWS Bedrock** (`ChatBedrockConverse`): AWS credentials via `~/.aws/credentials`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or an IAM role; region via `AWS_DEFAULT_REGION`.
   - **OpenAI** (`ChatOpenAI`): `OPENAI_API_KEY`.
   - **Ollama** (`ChatOllama`): a running Ollama instance (`ollama serve`) with the model pulled (`ollama pull llama3`).
   - **Any other LangChain `BaseChatModel`** also works.

### Optional

4. **A storage backend** — Required only if you want persistent storage of raw files and converted pages. **MongoDB** (any currently supported release, 4.4+ / 5.0+), **PostgreSQL** or **SQLite** work. Run MongoDB locally with Docker:

   ```bash
   docker run -p 27017:27017 mongo:7
   ```

   For PostgreSQL or SQLite, set `storage_backend="postgres"` / `"sqlite"` together with `sql_url`. Without a storage backend, `scinr` operates in memory-only mode (`storage_backend=none`), which is perfectly fine for most workflows.

5. **Mistral API key** — Required to process **any** PDF. PDF conversion is done entirely through the Mistral OCR API; there is no non-Mistral path. Obtain a key from [Mistral AI](https://console.mistral.ai/). Not needed if you never ingest PDFs.

6. **`ffmpeg` / `ffprobe`** — System binaries required only for audio and video files; they are not installed by `pip`. `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Debian/Ubuntu). Images, audio and video also need a vision model and/or a transcriber configured in `configure()` — see [Multimodal Ingestion](user-guides/multimodal-ingestion.md).

---

## Environment Setup

`scinr` reads configuration from environment variables. The recommended approach is to create a `.env` file from the provided template.

### Step 1: Create `.env` from the template

```bash
cp .env.example .env
```

### Step 2: Fill in your values

Open `.env` and set the values for your environment. The usual way to set the LLM is in code, as `configure(llm=...)` (the only environment shortcut is `MODEL_ID`, a Bedrock shortcut used when `llm` is omitted). Here is what the template looks like and what to fill in:

```ini
# ─── LLM provider SDK settings (the model itself is passed to configure(llm=...)) ──
AWS_DEFAULT_REGION=us-east-1
PROMPT_CACHING_ENABLED=true          # Bedrock only

# ─── Neo4j (only when GRAPH_BACKEND=neo4j, the default) ─────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# ─── PostgreSQL / SQLite (instead of Neo4j) ─────────────────────────────────
# GRAPH_BACKEND=sqlite               # or "postgres"
# SQL_URL=sqlite:///newton.db        # or postgresql://user:pass@host:5432/db

# ─── PDF Conversion (Mistral OCR) ───────────────────────────────────────────
MISTRAL_API_KEY=your_mistral_api_key   # required for any PDF

# ─── Storage (optional) ─────────────────────────────────────────────────────
STORAGE_BACKEND=none                 # or "mongodb", "postgres" or "sqlite" for persistent storage
# MONGODB_URI=mongodb://user:pass@localhost:27017

# ─── Pipeline ───────────────────────────────────────────────────────────────
EXTRACTION_BATCH_SIZE=1              # default; 2–3 trades a little quality for fewer LLM calls
LLM_CONCURRENCY=4                    # default; raise it toward your provider's rate limit
```

### Required fields for a first run (Neo4j)

These are required when `GRAPH_BACKEND=neo4j`, which is the default. With `GRAPH_BACKEND=postgres` or `sqlite`, set `SQL_URL` instead (see the [Quick Start](user-guides/quick-start.md)).

| Variable | What to set |
| :--- | :--- |
| `NEO4J_URI` | The uri from your Neo4j database. |
| `NEO4J_USER` | Your Neo4j username (usually `neo4j`). |
| `NEO4J_PASSWORD` | Your Neo4j password. |
| `NEO4J_DATABASE` | Your Neo4j database name.(usually `neo4j`). |

### Other fields

| Variable | What to set |
| :--- | :--- |
| `MISTRAL_API_KEY` | Mistral OCR API key. **Required to ingest any PDF** (there is no non-Mistral PDF path). Optional if you never process PDFs. |
| `STORAGE_BACKEND` | `none` (default), `mongodb`, `postgres`, `sqlite` or `custom` for persistent storage. |
| `EXTRACTION_BATCH_SIZE` | Number of pages processed per extraction call. Default `1` (best quality). `2`–`3` reduces call count at a small quality cost. See [Performance Tuning](user-guides/performance-tuning.md#extraction-batch-size-extraction_batch_size). |
| `LLM_CONCURRENCY` | Number of LLM calls made in parallel across all stages. Default `4`. Raise it gradually toward your provider's rate limit while watching for `429` errors. See [Performance Tuning](user-guides/performance-tuning.md#llm-concurrency-llm_concurrency). |

> **Note:** `python-dotenv` is included as a core dependency. The `.env` file is loaded when `scinr.newton` is imported (several of its modules call `load_dotenv()` at import time), not by `configure()`. You do not need to import `dotenv` manually, but the variables must be in place before the import.

---

## First Run

### Step 1: Prepare your documents

Create a directory and place some documents in it. `scinr` supports the following formats:

| Format | Extension | Notes |
| :--- | :--- | :--- |
| PDF | `.pdf` | Converted via the Mistral OCR API — requires `MISTRAL_API_KEY` |
| Word | `.docx` | Full text + structure extraction |
| Excel | `.xlsx`, `.xls` | Routed to tabular pipeline automatically |
| CSV | `.csv` | Routed to tabular pipeline automatically |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp`, `.tif`, `.tiff`, `.heic`, `.heif` | Transcribed and described — requires a `vision_llm`; HEIC/HEIF also `pip install "scinr[media]"` |
| Audio | `.mp3`, `.m4a`, `.wav`, `.ogg`, `.flac` | Transcribed in segments — requires a `transcriber` and the `ffmpeg` / `ffprobe` system binaries |
| Video | `.mp4`, `.mov`, `.mkv`, `.webm` | Transcription plus described frames — requires `ffmpeg`, a `transcriber` (if it has audio) and a `vision_llm` for the frames |

Images, audio and video are always accepted: without their requirement they show up as failed documents with an error that says what to configure. See [Multimodal Ingestion](user-guides/multimodal-ingestion.md).

```bash
mkdir -p raw_docs
# Place your .pdf, .docx, .xlsx, etc. files in raw_docs/
```

### Step 2: Run the pipeline

The following script configures `scinr` and runs the full pipeline on all documents in `raw_docs/`:

```python
import asyncio

from langchain_aws import ChatBedrockConverse

from scinr.newton import configure, run_pipeline

async def main():
    # The model is passed via llm=. Graph / storage settings resolve
    # from the arguments below, then the environment / .env, then defaults.
    # This example uses Neo4j (the default); for PostgreSQL or SQLite see the Quick Start.
    llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6", region_name="us-east-1")

    configure(
        llm=llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        mistral_api_key="your_mistral_key",  # required to ingest any PDF
    )

    result = await run_pipeline(input_raw="./raw_docs")

    # PipelineResult has structured per-stage results
    print(f"Success: {result.success}")
    print(f"Stages executed: {result.stages_executed}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")

    # Inspect individual stages
    for stage_name in result.stages_executed:
        stage = getattr(result, stage_name, None)
        if stage:
            print(f"  {stage_name}: {stage.total_processed} processed, "
                  f"{stage.total_failed} failed")

asyncio.run(main())
```

Save this as `run_ingestion.py` and execute it:

```bash
python run_ingestion.py
```

### Pipeline stages

The full pipeline runs these stages in order:

0. **Preprocess** — Converts raw files to an intermediate JSON with the content transformed into Markdown.
1. **Extraction** — Uses the LLM to parse document structure and extract hierarchical sections.
2. **Ingestion** — Writes document, structure nodes and Info Units into the graph.
3. **Annotation** — An LLM agent assigns an extraction model to each structural node.
4. **Entity Extraction** — Extracts typed Pydantic entities from annotated nodes and writes them as graph subgraphs.

> **Tabular path:** if `.csv`, `.xlsx`, or `.xls` files are present in `input_raw`, they are auto-detected and processed through a separate tabular pipeline (with LLM-powered normalization) alongside Stages 0–4.


### Running individual stages

You can run only specific stages by passing the `stages` parameter:

```python
# Run only annotation and entity extraction on a known document
result = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names=["MyDocument.pdf"],
)
```

See the [Configuration](configuration.md) documentation for all `run_pipeline()` parameters.

---

## Verifying Results

After the pipeline completes, your data is available in the graph you configured. Here are ways to verify the results. The Cypher and Neo4j driver examples apply when `graph_backend="neo4j"`; for PostgreSQL or SQLite, see [Verifying on PostgreSQL or SQLite](#verifying-on-postgresql-or-sqlite) below.

### Via Neo4j Browser (Neo4j only)

1. Open `http://localhost:7474` in your browser.
2. Log in with your Neo4j credentials.
3. Run these queries to inspect the ingested data:

```cypher
-- List all ingested documents
MATCH (d:Document)
RETURN d.name AS name, d.version AS version, d.path AS path
ORDER BY d.name;

-- Count structure nodes per document
MATCH (d:Document)-[:HAS_STRUCTURE|HAS_CHILD]->(s:StructureNode)
RETURN d.name AS document, count(s) AS nodes
ORDER BY nodes DESC;

--NOTE: The root structure nodes are connected through HAS_STRUCTURE with the Document, while structure nodes are conected to other child structure nodes through HAS_CHILD relationship.  

-- View annotated nodes with their assigned model and Info Units
MATCH (s:StructureNode)-[:HAS_MODEL_DECISION]->(m:ModelDecision)
MATCH (s)-[:HAS_INFO_UNIT]->(i:InfoUnit)
WHERE m.matched_model_class IS NOT NULL
RETURN s.id AS Structure_Node_ID, s.title AS Structure_Node_Title, m.matched_model_class AS model, collect(i.description) AS Info_Unit_Descriptions
ORDER BY s.id;

-- NOTE: You can check with this query how the matched model adapts to the information contained in the structure node title and the Info Unit Descriptions.

-- Explore extracted Instances and their relationships
MATCH (s:StructureNode)-[rsm:HAS_EXTRACTION]->(m:ExtractionResult)-[rmm2*..7]->(m2:ModelInstance)
return s,rsm,m,rmm2, m2
```

### Via Python (Neo4j only)

```python
from neo4j import AsyncGraphDatabase

async def verify():
    async with AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    ) as driver:
        async with driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (d:Document) RETURN count(d) AS doc_count"
            )
            record = await result.single()
            print(f"Documents in Neo4j: {record['doc_count']}")

asyncio.run(verify())
```

### Verifying on PostgreSQL or SQLite

With `graph_backend="postgres"` or `"sqlite"` the same data lives in SQL tables (`documents`, `structure_nodes`, `info_units`, …; see the [table summary](user-guides/sql-graph.md#table-summary)). Query them with any SQL client; on Postgres they are in the schema set by `sql_schema` (default `newton`):

```sql
SELECT name, version, path FROM documents ORDER BY name;
```

Or, on any backend, use the read-only [navigation API](user-guides/graph-navigation.md):

```python
from scinr.newton import graph_navigator

async def verify_any_backend():
    async with graph_navigator() as nav:
        print(await nav.count_root_documents())
```

### Inspecting PipelineResult

The `PipelineResult` returned by `run_pipeline()` contains detailed per-stage metrics:

```python
result = await run_pipeline(input_raw="./raw_docs")

# Overall success
print(f"Pipeline success: {result.success}")

# Per-stage details
if result.ingestion:
    for doc in result.ingestion.documents:
        print(f"  {doc.document_name}: "
              f"{doc.nodes_processed} processed, "
              f"{doc.nodes_failed} failed")
        if doc.errors:
            for err in doc.errors:
                print(f"    ERROR: {err}")
```

---

## Troubleshooting

### An LLM stage fails with "no LLM configured"

Pass an `llm=` argument to `configure()` with a LangChain `BaseChatModel` instance. The extraction / annotation / entity-extraction stages and tabular normalization require it; `configure()` itself does not.

### `ConfigurationError: Neo4j username is not configured` / `... password is not configured`

`configure()` raises these separately when `graph_backend` is `neo4j` (the default). Set `NEO4J_USER`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` in your `.env` file, or pass them as arguments to `configure()`. To run without Neo4j, use `graph_backend="postgres"` or `"sqlite"` with `sql_url`.

### `graph_backend='neo4j' requires the optional dependency`

The Neo4j driver is an optional extra. Install it with `pip install 'scinr[neo4j]'`, or switch to `graph_backend="postgres"` / `"sqlite"` and install `scinr[postgres]` / `scinr[sqlite]`.

### Neo4j connection refused

If you use Neo4j, make sure your instance is running and accessible:

```bash
# Test Bolt connectivity
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); d.verify_connectivity(); d.close(); print('OK')"
```


### PDFs fail to process

PDF conversion runs entirely on the Mistral OCR API. Set `MISTRAL_API_KEY` in your `.env` (or pass `mistral_api_key=` to `configure()`). There is no local/text-extraction fallback — without a key, PDFs cannot be ingested.

### `No documents discovered for this run`

Check that:
- The `input_raw` directory exists and contains supported file types.
- Recognized file extensions are: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.csv`, plus images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.tif`, `.tiff`, `.heic`, `.heif`), audio (`.mp3`, `.m4a`, `.wav`, `.ogg`, `.flac`) and video (`.mp4`, `.mov`, `.mkv`, `.webm`).
- The path is correct (relative paths are resolved from the current working directory).

### LLM calls are slow or rate-limited

Adjust concurrency in your `.env` or via `configure()`:

```python
configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_concurrency=2,       # Reduce concurrent LLM calls
    neo4j_concurrency=5,     # Reduce concurrent graph writes
)
```

---

## Next Steps

Now that you have a working pipeline, explore the rest of the documentation:

- **[Configuration](configuration.md)** — Complete reference for `configure()`, all environment variables, prompt families, concurrency tuning, and advanced settings.
- **[Architecture](architecture.md)** — Detailed walkthrough of each pipeline stage, data flow between stages, and system design decisions.
- **User Guides** — Domain-specific guides for working with extraction models, custom themes, and tabular data processing.
