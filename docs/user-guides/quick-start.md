# Quick Start — Your First Knowledge Graph

Get from zero to a working knowledge graph in under 15 minutes.

This guide walks you through installing `scinr`, configuring your environment, preparing documents, running the full 6-stage ingestion pipeline, and verifying the extracted knowledge graph in Neo4j.

---

## Prerequisites

Before you begin, make sure you have:

1. **Python 3.11+** installed on your system.
2. **Neo4j 5.0+** running and accessible via the Bolt protocol. If you do not have Neo4j installed, start one locally with Docker:

   ```bash
   docker run -p 7687:7687 -p 7474:7474 \
     -e NEO4J_AUTH=neo4j/your_password \
     neo4j:5
   ```

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

> **Tip:** If you plan to process PDF files with OCR, you will also need a [Mistral AI](https://console.mistral.ai/) API key. Without it, text-based PDFs are still processed via `pdfplumber`.

---

## Step 2: Set up your environment

`scinr` reads configuration from environment variables. The recommended approach is to create a `.env` file in your working directory.

### Create `.env` from the template

If you cloned the `scinr` repository, copy the example template:

```bash
cp .env.example .env
```

### Fill in the required values

At a minimum, you need to set the LLM model identifier and Neo4j credentials. Here is a minimal `.env` for a first run:

```env
# ─── LLM (AWS Bedrock) ──────────────────────────────────────────────────────
MODEL_ID=us.anthropic.claude-sonnet-4-6

# ─── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Required vs. optional fields

| Variable | Required? | What to set |
| :--- | :--- | :--- |
| `MODEL_ID` | Yes (unless you pass `llm=` to `configure()`) | Your LLM model ID. For Bedrock: `us.anthropic.claude-sonnet-4-6`. |
| `NEO4J_URI` | No | Neo4j Bolt URI. Default: `bolt://localhost:7687`. |
| `NEO4J_USER` | Yes | Neo4j username (usually `neo4j`). |
| `NEO4J_PASSWORD` | Yes | Neo4j password. |
| `MISTRAL_API_KEY` | No | Mistral API key for PDF OCR. |
| `STORAGE_BACKEND` | No | `none` (default, in-memory) or `mongodb`. |

> **Note:** `python-dotenv` is included as a core dependency. When you call `configure()`, it automatically loads variables from a `.env` file in your current working directory. You do not need to import `dotenv` manually.

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
| PDF | `.pdf` | Text-based via `pdfplumber`; OCR via Mistral API |
| Word | `.docx` | Full text + structure extraction |
| Excel | `.xlsx`, `.xls` | Auto-routed to tabular pipeline |
| PowerPoint | `.pptx` | Slide text extraction |
| CSV | `.csv` | Auto-routed to tabular pipeline |
| HTML | `.html`, `.htm` | Cleaned and parsed |
| JSON | `.json` | API responses, structured data |
| Text | `.txt`, `.md`, `.rst` | Plain text |

> **Warning:** Tabular files (`.csv`, `.xlsx`, `.xls`) are automatically routed to the tabular pipeline and bypass the standard 5-stage extraction flow. This is intentional — spreadsheets have a fundamentally different structure than narrative documents.

---

## Step 4: Run the pipeline

Create a Python script to configure `scinr` and run the full pipeline:

```python
# quickstart.py
import asyncio
from scinr.newton import configure, run_pipeline


async def main():
    # configure() reads .env automatically via python-dotenv.
    # It resolves LLM, Neo4j, and storage settings from:
    #   1. Explicit arguments (highest priority)
    #   2. Environment variables / .env file
    #   3. Hard-coded defaults
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Run the full 6-stage pipeline
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
| 2 | **Ingestion** | Writes document and structure nodes into Neo4j |
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

## Step 5: Verify in Neo4j

After the pipeline completes, your data is available in Neo4j. Use the Neo4j Browser (`http://localhost:7474`) or any Cypher client to inspect the results.

### List all ingested documents

```cypher
MATCH (d:Document)
RETURN d.document_name, d.format, d.ingested_at, d.version
ORDER BY d.ingested_at DESC;
```

### View document structure

```cypher
MATCH (d:Document)-[:HAS_STRUCTURE]->(s:StructureNode)
RETURN d.document_name AS document, s.section_title AS section, s.node_type
LIMIT 20;
```

### View the full hierarchy

```cypher
MATCH (d:Document)-[:HAS_STRUCTURE]->(s:StructureNode)-[:HAS_CHILD]->(c:StructureNode)
RETURN d.document_name AS document, s.section_title AS parent, c.section_title AS child, c.node_type AS child_type
LIMIT 30;
```

### View extracted entity types

```cypher
MATCH (m:ModelInstance)
RETURN labels(m) AS type, count(m) AS count
ORDER BY count DESC;
```

### View labeled entities (globally deduplicated)

```cypher
MATCH (e:LabeledEntity)
RETURN head(labels(e)[1..]) AS entity_type, e.value, count(e) AS occurrences
ORDER BY occurrences DESC
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
MATCH (d:Document)-[:HAS_STRUCTURE*0..]->(s:StructureNode)<-[:EXTRACTED_FROM]-(m:ModelInstance)
WHERE d.document_name = 'YourDocumentName'
RETURN s.section_title AS section,
       head(labels(m)[1..]) AS model_type,
       count(m) AS entities
ORDER BY entities DESC
LIMIT 20;
```

### Visualize the graph

In Neo4j Browser, run this query to get a visual overview:

```cypher
MATCH path = (d:Document)-[:HAS_STRUCTURE*1..3]->(s:StructureNode)
WHERE d.document_name = 'YourDocumentName'
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
    document_names=["MyDocument"],
)
```

### Processing a single document

```python
# Only process files matching a specific name pattern
result = await run_pipeline(
    input_raw="./raw_docs",
    document_names=["ClinicalTrialReport"],
)
```

### Enabling MongoDB storage

Persist raw files and converted pages to MongoDB:

```python
configure(
    neo4j_user="neo4j",
    neo4j_password="your_password",
    storage_backend="mongodb",
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="scinr",
)
```

---

## Troubleshooting

### `"No LLM configured"`

You must either:

- Set `MODEL_ID` in your `.env` file (for AWS Bedrock), or
- Pass an `llm=` argument to `configure()` with a LangChain `BaseChatModel` instance.

### `"Neo4j username/password is not configured"`

Set `NEO4J_USER` and `NEO4J_PASSWORD` in your `.env` file, or pass them as arguments to `configure()`.

### Neo4j connection refused

Make sure your Neo4j instance is running and accessible:

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
- File extensions are recognized (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.pptx`, `.html`, `.json`, `.txt`, `.md`).
- The path is correct — relative paths are resolved from your current working directory.

### ImportError: `langchain-aws is not installed`

If you set `MODEL_ID` but have not installed the Bedrock extra:

```bash
pip install "scinr[bedrock]"
```

### PDFs fail to process

PDF processing requires either:

- A Mistral API key (for OCR) set via `MISTRAL_API_KEY` in your `.env`, or
- The PDF must contain extractable text (processed via `pdfplumber` without OCR).

If you see OCR-related errors and do not have a Mistral key, try text-based PDFs or set `MISTRAL_API_KEY`.

### LLM calls are slow or rate-limited

Adjust concurrency in your `.env` or via `configure()`:

```python
configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="your_password",
    llm_concurrency=2,       # Reduce concurrent LLM calls
    neo4j_concurrency=5,     # Reduce concurrent Neo4j writes
)
```

### Annotation stage returns no model matches

This is expected for documents that do not contain content matching any registered extraction model. The pipeline falls back to generic triple extraction for unmatched nodes. To get specific entity extraction, define custom models matching your domain — see [Custom Models](custom-models.md).
