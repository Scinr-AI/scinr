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

# MongoDB storage (includes motor and pymongo)
pip install "scinr[mongodb]"

# Documentation tooling (mkdocs, mkdocstrings, griffe, ruff)
pip install "scinr[docs]"

# Development tooling (pytest, ruff, mypy)
pip install "scinr[dev]"

# Multiple extras at once
pip install "scinr[bedrock,mongodb,dev]"
```

With `uv`:

```bash
uv add "scinr[bedrock]"
uv add "scinr[bedrock,mongodb]"
```

---

## Prerequisites

Before running the pipeline, ensure you have the following:

### Required

1. **Python 3.11+** — `scinr` requires Python 3.11 or later.

2. **Neo4j 5.0+** — A running Neo4j instance accessible via the Bolt protocol. You can run Neo4j locally with Docker:

   ```bash
   docker run -p 7687:7687 -p 7474:7474 \
     -e NEO4J_AUTH=neo4j/your_password \
     neo4j:5
   ```

3. **LLM credentials** — depending on your provider:

   - **AWS Bedrock**: AWS credentials configured via `~/.aws/credentials`, environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), or an IAM role. The model is selected via the `MODEL_ID` environment variable (e.g., `us.anthropic.claude-sonnet-4-6`).
   - **OpenAI**: An `OPENAI_API_KEY` environment variable.
   - **Ollama**: A locally running Ollama instance (`ollama serve`) with the desired model pulled (`ollama pull llama3`).
   - **Any LangChain-compatible model**: You can pass a `BaseChatModel` instance directly to `configure()`.

### Optional

4. **MongoDB 4.6+** — Required only if you want persistent storage of raw files and converted pages. Run locally with Docker:

   ```bash
   docker run -p 27017:27017 mongo:7
   ```

   Without MongoDB, `scinr` operates in memory-only mode (`storage_backend=none`), which is perfectly fine for most workflows.

5. **Mistral API key** — Required to process PDF files with OCR. Obtain a key from [Mistral AI](https://console.mistral.ai/). Without it, PDFs can still be processed with `pdfplumber` (text-based extraction, no OCR).

---

## Environment Setup

`scinr` reads configuration from environment variables. The recommended approach is to create a `.env` file from the provided template.

### Step 1: Create `.env` from the template

```bash
cp .env.example .env
```

### Step 2: Fill in your values

Open `.env` and set the values for your environment. Here is what the template looks like and what to fill in:

```ini
# ─── LLM (AWS Bedrock) ──────────────────────────────────────────────────────
AWS_DEFAULT_REGION=us-east-1
MODEL_ID=us.anthropic.claude-sonnet-4-6
REPAIR_MODEL_ID=us.anthropic.claude-haiku-3
PROMPT_CACHING_ENABLED=true

# ─── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# ─── PDF Conversion (Mistral OCR) ───────────────────────────────────────────
MISTRAL_API_KEY=your_mistral_api_key

# ─── Storage (optional) ─────────────────────────────────────────────────────
STORAGE_BACKEND=mongodb
MONGO_URI=mongodb://user:pass@localhost:27017

# ─── Pipeline ───────────────────────────────────────────────────────────────
EXTRACTION_BATCH_SIZE=3
LLM_CONCURRENCY=32
```

### Required fields for a first run

| Variable | What to set |
| :--- | :--- |
| `NEO4J_URI` | The uri from your Neo4j database. |
| `NEO4J_USER` | Your Neo4j username (usually `neo4j`). |
| `NEO4J_PASSWORD` | Your Neo4j password. |
| `NEO4J_DATABASE` | Your Neo4j database name.(usually `neo4j`). |

### Optional fields

| Variable | What to set |
| :--- | :--- |
| `MISTRAL_API_KEY` | Mistral API key for PDF OCR. |
| `STORAGE_BACKEND` | `none` (default) or `mongodb` for persistent storage. |
| `EXTRACTION_BATCH_SIZE` | It indicates the number of pages processed in the same call. The recommended values for most use cases are 2 or 3. |
| `LLM_CONCURRENCY` | It indicates the number of llm calls that are done in parallel. You can start with 32, but it depends mainly from the rate limits from your provider. |

> **Note:** `python-dotenv` is included as a core dependency. When you call `configure()`, it automatically loads variables from a `.env` file in your current working directory. You do not need to import `dotenv` manually.

---

## First Run

### Step 1: Prepare your documents

Create a directory and place some documents in it. `scinr` supports the following formats:

| Format | Extension | Notes |
| :--- | :--- | :--- |
| PDF | `.pdf` | Text-based via `pdfplumber`; OCR via Mistral API |
| Word | `.docx` | Full text + structure extraction |
| Excel | `.xlsx`, `.xls` | Routed to tabular pipeline automatically |
| CSV | `.csv` | Routed to tabular pipeline automatically |

```bash
mkdir -p raw_docs
# Place your .pdf, .docx, .xlsx, etc. files in raw_docs/
```

### Step 2: Run the pipeline

The following script configures `scinr` and runs the full pipeline on all documents in `raw_docs/`:

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    # configure() reads .env automatically via python-dotenv.
    # It resolves LLM, Neo4j, and storage settings from:
    #   1. Explicit arguments (highest priority)
    #   2. Environment variables / .env file
    #   3. Hard-coded defaults
    llm = ChatBedrockConverse(...)

    configure(
        llm=llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        mistral_api_key="", # Needed por pdfs OCR
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

1. **Preprocess** — Converts raw files to an intermediate JSON with the content transformed into Markdown.
2. **Extraction** — Uses the LLM to parse document structure and extract hierarchical sections.
3. **Ingestion** — Writes document, structure nodes and Info Units into Neo4j.
4. **Annotation** — An LLM agent assigns an extraction model to each structural node.
5. **Entity Extraction** — Extracts typed Pydantic entities from annotated nodes and writes them as graph subgraphs.

1a. **Tabular** — If `.csv`, `.xlsx`, or `.xls` files are detected in `input_raw`, they are processed through a separate tabular pipeline with LLM-powered normalization.


### Running individual stages

You can run only specific stages by passing the `stages` parameter:

```python
# Run only annotation and entity extraction on a known document
result = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names=["MyDocument"],
)
```

See the [Configuration](configuration.md) documentation for all `run_pipeline()` parameters.

---

## Verifying Results

After the pipeline completes, your data is available in Neo4j. Here are ways to verify the results:

### Via Neo4j Browser

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
RETURN s.node_id as Structure_Node_ID, s.title as Structure_Node_Title, m.matched_model_class AS model, collect(i.description) as Info_Unit_Descriptions
ORDER BY s.node_id;

-- NOTE: You can check with this query how the matched model adapts to the information contained in the structure node title and the Info Unit Descriptions.

-- Explore extracted Instances and their relationships
MATCH (s:StructureNode)-[rsm:HAS_EXTRACTION]->(m:ExtractionResult)-[rmm2*..7]->(m2:ModelInstance)
return s,rsm,m,rmm2, m2
```

### Via Python

```python
from neo4j import AsyncGraphDatabase

async def verify():
    async with AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    ) as driver:
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(
                "MATCH (d:Document) RETURN count(d) AS doc_count"
            )
            record = await result.single()
            print(f"Documents in Neo4j: {record['doc_count']}")

asyncio.run(verify())
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

### `ConfigurationError: No LLM configured`

You must:
- Pass an `llm=` argument to `configure()` with a LangChain `BaseChatModel` instance.

### `ConfigurationError: Neo4j username/password is not configured`

Set `NEO4J_USER` and `NEO4J_PASSWORD` in your `.env` file, or pass them as arguments to `configure()`.

### Neo4j connection refused

Make sure your Neo4j instance is running and accessible:

```bash
# Test Bolt connectivity
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); d.verify_connectivity(); d.close(); print('OK')"
```


### PDFs fail to process

PDF processing requires either:
- A Mistral API key (for OCR) set via `MISTRAL_API_KEY` in your `.env`, or

### `No documents discovered for this run`

Check that:
- The `input_raw` directory exists and contains supported file types.
- Recognized file extensions are: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.csv`.
- The path is correct (relative paths are resolved from the current working directory).

### LLM calls are slow or rate-limited

Adjust concurrency in your `.env` or via `configure()`:

```python
configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_concurrency=2,       # Reduce concurrent LLM calls
    neo4j_concurrency=5,     # Reduce concurrent Neo4j writes
)
```

---

## Next Steps

Now that you have a working pipeline, explore the rest of the documentation:

- **[Configuration](configuration.md)** — Complete reference for `configure()`, all environment variables, prompt families, concurrency tuning, and advanced settings.
- **[Architecture](architecture.md)** — Detailed walkthrough of each pipeline stage, data flow between stages, and system design decisions.
- **User Guides** — Domain-specific guides for working with extraction models, custom themes, and tabular data processing.
