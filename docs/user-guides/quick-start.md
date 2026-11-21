# Quick Start — Your First Knowledge Graph

Get from zero to a working knowledge graph in under 15 minutes.

This guide walks you through installing `scinr`, configuring your environment, preparing documents, running the full ingestion pipeline (Stages 0–4, plus an auto-detected tabular path), and verifying the extracted knowledge graph. The main path uses Neo4j (the default graph backend); if you prefer PostgreSQL or SQLite, jump to [Running without Neo4j](#running-without-neo4j).

---

## Prerequisites

Before you begin, make sure you have:

1. **Python 3.11+** installed on your system.
2. **A graph engine**: **Neo4j 5.0+** (the default) accessible via the Bolt protocol, **or** PostgreSQL, **or** SQLite (no server). If you use Neo4j and do not have it installed, start one locally with Docker:

   ```bash
   docker run -p 7687:7687 -p 7474:7474 \
     -e NEO4J_AUTH=neo4j/your_password \
     neo4j:5
   ```

   For PostgreSQL or SQLite, follow [Running without Neo4j](#running-without-neo4j).

3. **LLM credentials** for at least one provider (AWS Bedrock, OpenAI, or Ollama).

---

## Step 1: Install scinr

Install `scinr` with `pip`. Choose the extras that match your LLM provider:

```bash
# AWS Bedrock (recommended — includes langchain-aws and boto3)
pip install "scinr[bedrock]"

# Or with OpenAI (includes langchain-openai)
pip install "scinr[openai]"

# Or with Ollama (includes langchain-ollama)
pip install "scinr[ollama]"

# Or with all extras at once
pip install "scinr[bedrock,openai,ollama,mongodb]"
```

Then add the extra for your graph engine:

```bash
pip install "scinr[neo4j]"      # Neo4j driver (the default graph backend)
pip install "scinr[postgres]"   # PostgreSQL
pip install "scinr[sqlite]"     # SQLite
```

> **Important:** the default graph backend is `neo4j`, and the Neo4j driver is an optional extra. Without `scinr[neo4j]` the first use of the graph fails with `graph_backend='neo4j' requires the optional dependency: pip install 'scinr[neo4j]'`.

> **Tip:** To process **any** PDF you need a [Mistral AI](https://console.mistral.ai/) API key — PDF conversion is done entirely through the Mistral OCR API. There is no fallback path for PDFs without a key.

---

## Step 2: Set up your environment

`scinr` reads graph (Neo4j or SQL), storage, and OCR settings from environment variables. The usual way to set the **LLM is in code** — you pass a ready-built LangChain model as `llm=` to `configure()` (see [Step 4](#step-4-run-the-pipeline)). The recommended approach for everything else is a `.env` file in your working directory.

### Create `.env` from the template

If you cloned the `scinr` repository, copy the example template:

```bash
cp .env.example .env
```

### Fill in the required values

At a minimum, with the default Neo4j backend you need to set the Neo4j credentials (they are required only when `GRAPH_BACKEND=neo4j`, which is the default). Here is a minimal `.env` for a first run:

```env
# ─── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# ─── PDF conversion ────────────────────────────────────────────────────────
MISTRAL_API_KEY=your_mistral_key
```

### Required vs. optional fields

| Variable | Required? | What to set |
| :--- | :--- | :--- |
| `NEO4J_URI` | No | Neo4j Bolt URI. Default: `bolt://localhost:7687`. |
| `NEO4J_USER` | Yes, if `GRAPH_BACKEND=neo4j` | Neo4j username (usually `neo4j`). |
| `NEO4J_PASSWORD` | Yes, if `GRAPH_BACKEND=neo4j` | Neo4j password. |
| `NEO4J_DATABASE` | Yes, if `GRAPH_BACKEND=neo4j` | Neo4j database name (usually `neo4j`). |
| `GRAPH_BACKEND` | No | `neo4j` (default), `postgres` or `sqlite`. |
| `SQL_URL` | Yes, if any backend is `postgres` or `sqlite` | SQLAlchemy URL, e.g. `postgresql://user:pass@host:5432/db` or `sqlite:///newton.db`. |
| `MISTRAL_API_KEY` | Yes, to process any PDF | Mistral OCR API key. PDF conversion has no non-Mistral path. |
| `STORAGE_BACKEND` | No | `none` (default, in-memory), `mongodb`, `postgres`, `sqlite` or `custom`. |

> **LLM:** the usual way is to build a LangChain chat model in your script and pass it as `llm=` to `configure()`. If you omit `llm` and the environment variable `MODEL_ID` is set, newton builds an AWS Bedrock `ChatBedrockConverse` from it (it reads `AWS_DEFAULT_REGION`, default `us-east-1`, and `MAX_TOKENS`, default `65536`); this needs the `bedrock` extra.

> **Note:** `python-dotenv` is included as a core dependency. The `.env` file is loaded when `scinr.newton` is imported (several of its modules call `load_dotenv()` at import time), not by `configure()`. You do not need to import `dotenv` manually.

---

## Step 3: Prepare your documents

Create a directory and place your source documents in it:

```bash
mkdir -p raw_docs
# Place your files here
```

### Supported file formats

| Format | Extensions | Notes |
| :--- | :--- | :--- |
| PDF | `.pdf` | Converted via the Mistral OCR API — requires `MISTRAL_API_KEY` |
| Word | `.docx` | Full text + structure extraction |
| Excel | `.xlsx`, `.xls` | Auto-routed to tabular pipeline |
| CSV | `.csv` | Auto-routed to tabular pipeline |

> **Roadmap:** `.pptx`, `.html`, `.json`, `.xml`, and `.txt` converters exist in the codebase but are not yet officially supported or tested. Stick to the formats above for production use.

> **Warning:** Tabular files (`.csv`, `.xlsx`, `.xls`) are automatically routed to the tabular pipeline and bypass the standard extraction flow (Stages 0–4). This is intentional — spreadsheets have a fundamentally different structure than narrative documents.

---

## Step 4: Run the pipeline

Create a Python script to configure `scinr` and run the full pipeline:

```python
# quickstart.py
import asyncio

from langchain_aws import ChatBedrockConverse  # or any LangChain chat model

from scinr.newton import configure, run_pipeline


async def main():
    # The model is passed via llm=. Neo4j and storage settings may also come
    # from .env (loaded when scinr.newton is imported).
    configure(
        llm=ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6", region_name="us-east-1"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
    )

    # Run the default pipeline (Stages 0–4; tabular files are auto-detected)
    result = await run_pipeline(input_raw="./raw_docs")

    # Check overall result
    print(f"Pipeline success: {result.success}")
    print(f"Total duration: {result.total_duration_seconds:.1f}s")
    print(f"Stages executed: {result.stages_executed}")

    # Inspect individual stages
    for stage_name in result.stages_executed:
        stage = getattr(result, stage_name, None)
        if stage is not None:
            status = "OK" if stage.success else "FAIL"
            print(
                f"  [{status}] {stage_name}: "
                f"{stage.total_processed} processed, "
                f"{stage.total_failed} failed "
                f"({stage.duration_seconds:.1f}s)"
            )

            # Show per-document details
            for doc in stage.documents:
                if doc.errors:
                    for err in doc.errors:
                        print(f"    - {doc.document_name}: {err}")


asyncio.run(main())
```

Run the script:

```bash
python quickstart.py
```

### What happens behind the scenes

The pipeline runs these stages in order:

| Stage | Name | What it does |
| :--- | :--- | :--- |
| 0 | **Preprocess** | Converts raw files to an intermediate JSON/Markdown format |
| 1 | **Extraction** | Uses the LLM to parse document structure and extract hierarchical sections |
| 2 | **Ingestion** | Writes document and structure nodes into the graph |
| 3 | **Annotation** | An LLM agent assigns an extraction model to each structural node |
| 4 | **Entity Extraction** | Extracts typed Pydantic entities from annotated nodes and writes them as graph subgraphs |
| 5 | **Tabular** | If `.csv`, `.xlsx`, or `.xls` files are detected in `input_raw`, they are processed through a separate tabular pipeline with LLM-powered normalization |

### Expected output

A successful run produces output similar to:

```
Pipeline success: True
Total duration: 45.2s
Stages executed: ['preprocess', 'extraction', 'ingestion', 'annotation', 'entity_extraction']
  [OK] preprocess: 3 processed, 0 failed (2.1s)
  [OK] extraction: 3 processed, 0 failed (18.5s)
  [OK] ingestion: 3 processed, 0 failed (4.3s)
  [OK] annotation: 47 processed, 0 failed (12.8s)
  [OK] entity_extraction: 47 processed, 2 failed (8.5s)
```

> **Tip:** If you see failures in entity extraction, this is often expected — not every structural node contains extractable domain entities. The pipeline continues gracefully. Check the per-document errors for details.

---

## Running without Neo4j

The graph, the raw files and pages, and the document snapshots can live in PostgreSQL or SQLite instead of Neo4j. The pipeline code is the same; only `configure()` changes. See [SQL Graph Backends](sql-graph.md) for the full guide.

### PostgreSQL with Docker

newton creates the schema and tables **inside** an existing database. It does not create the database itself nor any role, so start the container with `POSTGRES_DB` (and a user) already set:

```bash
docker run -d --name newton-postgres -p 5432:5432 \
  -e POSTGRES_USER=newton \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=scinr \
  postgres:16

pip install "scinr[postgres]"
```

Then configure the three backends (the graph, the storage and, by inheritance, the snapshots) with one `sql_url`:

```python
from scinr.newton import configure

configure(
    llm=...,                      # as in Step 4
    graph_backend="postgres",
    storage_backend="postgres",   # freeze_backend inherits it
    sql_url="postgresql://newton:secret@localhost:5432/scinr",
)
```

No `neo4j_*` settings are needed. On the first use newton creates the `newton` schema (`sql_schema`) and its tables in the `scinr` database; the user must be allowed to create them. The `sql_url` scheme must match the backends (`postgresql://` with `"postgres"`), or `configure()` raises a `ConfigurationError`.

### SQLite (no Docker)

```bash
pip install "scinr[sqlite]"
```

```python
from scinr.newton import configure

configure(
    llm=...,
    graph_backend="sqlite",
    storage_backend="sqlite",
    sql_url="sqlite:///newton.db",
)
```

The database is the file `newton.db` (created on first use), plus a folder for the stored binaries next to it (`newton.files/` by default, or `sqlite_files_dir`). Use this `configure()` call in the Step 4 script in place of the Neo4j one.

### Verifying the data

With the SQL backends the data lives in tables (`documents`, `structure_nodes`, `info_units`, …; see the [table summary](sql-graph.md#table-summary)). Use any SQL client on the database; on Postgres the tables are in the `newton` schema:

```sql
SELECT name, version, path FROM documents ORDER BY name;
```

Or use the read-only [navigation API](graph-navigation.md), which works on every backend: `async with graph_navigator() as nav: await nav.list_root_documents()`.

---

## Step 5: Verify in Neo4j

This step applies to the default Neo4j backend. If you used PostgreSQL or SQLite, see [Verifying the data](#verifying-the-data) above.

After the pipeline completes, your data is available in Neo4j. Use the Neo4j Browser (`http://localhost:7474`) or any Cypher client to inspect the results.

> **Schema note:** `HAS_STRUCTURE` connects a `:Document` **only** to its root `:StructureNode`; deeper nesting is `HAS_CHILD`. Extracted entities hang off `(:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)-[:HAS_<field>]->(:ModelInstance)-[:REFERENCES]->(:LabeledEntity)`. See [Neo4j Graph Model](neo4j-graph.md) for the full reference.

### List all ingested documents

```cypher
MATCH (d:Document)
WHERE d.latest = true
RETURN d.name, d.path, d.version, d.load_date
ORDER BY d.load_date DESC;
```

### View document structure

```cypher
MATCH (d:Document)-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)
RETURN d.name AS document, s.title AS section, s.role
LIMIT 20;
```

### View the full hierarchy

```cypher
MATCH (d:Document)-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(parent:StructureNode)-[:HAS_CHILD]->(child:StructureNode)
RETURN d.name AS document, parent.title AS parent, child.title AS child, child.role AS child_role
LIMIT 30;
```

### View extracted entity types

```cypher
MATCH (m:ModelInstance)
RETURN m.model_class AS type, count(m) AS count
ORDER BY count DESC;
```

### View labeled entities (globally deduplicated)

```cypher
MATCH (:ModelInstance)-[:REFERENCES]->(e:LabeledEntity)
RETURN e.label AS entity_type, e.value, count(*) AS references
ORDER BY references DESC
LIMIT 20;
```

### View entity relationships

```cypher
MATCH (a:ModelInstance)-[r]->(b:ModelInstance)
RETURN type(r) AS relationship, count(r) AS count
ORDER BY count DESC;
```

### Explore a specific document's extraction results

```cypher
MATCH (d:Document {name: 'YourDocumentName'})-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)
MATCH (s)-[:HAS_EXTRACTION]->(:ExtractionResult)-[r]->(m:ModelInstance)
RETURN s.title AS section,
       m.model_class AS model_type,
       count(m) AS entities
ORDER BY entities DESC
LIMIT 20;
```

### Visualize the graph

In Neo4j Browser, run this query to get a visual overview:

```cypher
MATCH path = (d:Document {name: 'YourDocumentName'})-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..2]->(:StructureNode)
RETURN path
LIMIT 50;
```

---

## Step 6: Next steps

Now that you have a working pipeline, explore the rest of the documentation:

- **[Configuration](../configuration.md)** — Full configuration reference: all environment variables, `configure()` parameters, prompt families, concurrency tuning, and advanced settings.
- **[Custom Models](custom-models.md)** — Define your own Pydantic extraction schemas for domain-specific entities.
- **[Architecture](../architecture.md)** — Detailed walkthrough of each pipeline stage, data flow between stages, and system design decisions.
- **[Tabular Pipeline](tabular-pipeline.md)** — Working with CSV, XLSX, and spreadsheet data.
- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding the graph model and node/relationship types.

---

## Common variations

### Using OpenAI instead of Bedrock

```python
from langchain_openai import ChatOpenAI
from scinr.newton import configure, run_pipeline

configure(
    llm=ChatOpenAI(model="gpt-4o"),
    neo4j_user="neo4j",
    neo4j_password="your_password",
)
```

Set `OPENAI_API_KEY` in your `.env` file.

### Using Ollama (local models)

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline

configure(
    llm=ChatOllama(model="llama3"),
    neo4j_user="neo4j",
    neo4j_password="your_password",
)
```

Make sure Ollama is running locally (`ollama serve`) and the model is pulled (`ollama pull llama3`).

### Running only specific stages

You can skip stages by providing input from a later point in the pipeline:

```python
# Run only annotation and entity extraction on an already-ingested document
result = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names=["MyDocument.pdf"],
)
```

### Processing a single document

```python
# Only process files matching a specific name pattern
result = await run_pipeline(
    input_raw="./raw_docs",
    document_names=["ClinicalTrialReport.pdf"],
)
```

### Enabling MongoDB storage

Persist raw files and converted pages to MongoDB:

```python
configure(
    llm=ChatOllama(model="llama3"),
    neo4j_user="neo4j",
    neo4j_password="your_password",
    neo4j_database="neo4j",
    storage_backend="mongodb",
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="scinr",
)
```

---

## Troubleshooting

### An LLM stage fails with "no LLM configured"

The extraction, annotation, and entity-extraction stages (and tabular normalization) need a model. Pass an `llm=` argument to `configure()` with a LangChain `BaseChatModel` instance (e.g. `ChatBedrockConverse`, `ChatOpenAI`, `ChatOllama`). If `llm` is omitted and the environment variable `MODEL_ID` is set, newton builds an AWS Bedrock `ChatBedrockConverse` from it (reading `AWS_DEFAULT_REGION` and `MAX_TOKENS`); otherwise there is no model. `configure()` itself does not require it — only these stages do.

### `"Neo4j username is not configured"` / `"Neo4j password is not configured"`

`configure()` raises these separately when `graph_backend` is `neo4j` (the default). Set `NEO4J_USER` and `NEO4J_PASSWORD` in your `.env` file, or pass them as arguments to `configure()`. `NEO4J_DATABASE` is also required. To run without Neo4j, see [Running without Neo4j](#running-without-neo4j).

### `graph_backend='neo4j' requires the optional dependency`

Install the Neo4j driver with `pip install 'scinr[neo4j]'`, or use PostgreSQL or SQLite as described in [Running without Neo4j](#running-without-neo4j).

### Neo4j connection refused

If you use Neo4j, make sure your instance is running and accessible:

```bash
# Test Bolt connectivity
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your_password'))
driver.verify_connectivity()
driver.close()
print('Neo4j connection OK')
"
```

### `"No documents discovered for this run"`

Check that:

- The `input_raw` directory exists and contains supported file types.
- File extensions are recognized (`.pdf`, `.docx`, `.xlsx`, `.xls`, `.csv`).
- The path is correct — relative paths are resolved from your current working directory.

### ImportError: `langchain-aws is not installed`

If you build a `ChatBedrockConverse` model but have not installed the Bedrock extra:

```bash
pip install "scinr[bedrock]"
```

### PDFs fail to process

PDF conversion goes entirely through the Mistral OCR API. Set `MISTRAL_API_KEY` in your `.env` (get a key at [console.mistral.ai](https://console.mistral.ai/)). Without it, no PDF can be processed — there is no local/text-extraction fallback.

### LLM calls are slow or rate-limited

Adjust concurrency in your `.env` or via `configure()`:

```python
configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="your_password",
    llm_concurrency=2,       # Reduce concurrent LLM calls
    graph_concurrency=5,     # Reduce concurrent graph writes
)
```

### Annotation stage returns no model matches

This is expected for documents that do not contain content matching any registered extraction model. The pipeline falls back to generic triple extraction for unmatched nodes. To get specific entity extraction, define custom models matching your domain — see [Custom Models](custom-models.md).
