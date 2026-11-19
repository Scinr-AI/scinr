# scinr

> **AI-powered knowledge extraction for life sciences**

`scinr` (`scinr.newton`) is a Python library for turning life sciences documents and tabular data into structured, queryable knowledge.

For unstructured documents, `scinr` extracts domain entities and relationships and stores them as connected knowledge graphs in **Neo4j** (the default), **PostgreSQL** or **SQLite**, with optional document storage in **MongoDB**, PostgreSQL or SQLite. See [SQL Graph Backends](user-guides/sql-graph.md).

It uses LLMs and Pydantic models to extract structured, domain-specific information from scientific content.

---

## Key Features

* **5-stage document pipeline** — Preprocess, extract structure, ingest, annotate, and extract domain entities from unstructured documents.

* **Live progress reporting through callbacks** — Follow `run_pipeline()` stage by stage and document by document with the built-in `ProgressTracker`, or your own sync/async callbacks to update a job in your database.

* **Tabular data pipeline** — Normalize scientific spreadsheets and extract structured entities from tabular data.

* **Multi-format ingestion** — Supports `.pdf`, `.docx`, `.xlsx`, `.xls`, and `.csv`. Converters for `.pptx`, `.json`, `.xml`, `.html`, and `.txt` exist in the codebase but are not yet officially supported.

* **Images, audio and video** — Image files are transcribed and described by a configurable vision model, audio and video are transcribed segment by segment with time ranges (video frames described too), and the images embedded in PDF, DOCX and PPTX are described in place. Cloud or self-hosted providers, open interfaces. See [Multimodal Ingestion](user-guides/multimodal-ingestion.md).

* **Pydantic extraction models** — Define structured schemas for domain entities such as compounds, clinical trials, and assays.

* **Knowledge graphs on Neo4j, PostgreSQL or SQLite** — Store extracted entities and relationships with document provenance in the engine you choose with `graph_backend`. See [SQL Graph Backends](user-guides/sql-graph.md).

* **Multi-tenancy** — The `tenant_id` is part of every document's identity: tenants ingesting the same path get fully independent documents, and tenant-less (public) documents are never merged with a tenant's. Stamp runs with `created_by_user_id` and `job_id`, and delete a whole ingestion run of a tenant in one call with `delete_document(job_id=..., tenant_id=...)`.

* **Document freezing** — Archive a document's subgraph to a snapshot with `freeze_document()` and keep only a stub in the graph, then rebuild it with `restore_document()`: same ids, no LLM calls. Backups and exports use the same snapshot.

* **Optional storage layer** — Store raw files, converted documents, and binary assets in MongoDB (GridFS), PostgreSQL or SQLite.

* **LLM-ready documentation** — Provides `llms.txt` and `llms-full.txt` files for AI coding agents and other LLM-based tools.

---

## Pipelines

### Unstructured Documents

Unstructured documents go through five stages:

```text
Raw Documents
(.pdf, .docx, .xlsx, .xls, .csv,
 images, audio, video)
        │
        ▼
1. Preprocess
   Convert files to a common representation
        │
        ▼
2. Extraction
   Parse sections, structure, and hierarchy
        │
        ▼
3. Ingestion
   Store document structure and provenance in the graph
        │
        ▼
4. Annotation
   Identify relevant content and prepare it for extraction
        │
        ▼
5. Entity Extraction
   Extract domain entities and relationships
        │
        ▼
Knowledge Graph
```

### Tabular Data

Tabular data follows a separate pipeline designed for scientific spreadsheets and other structured data:

```text
Tabular Data
(.xlsx, .csv, ...)
        │
        ▼
LLM Entity Extraction
        │
        ▼
Normalization
        │
        ▼
Structured Knowledge
```

---

## Quick Example

```python
import asyncio

from scinr.newton import configure, run_pipeline
from langchain_aws import ChatBedrockConverse # Or any langchain adapter. 

async def main():
    llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6", region_name="us-east-1")

    configure(
        llm=llm,  # the usual way to set the model (see Configuration for the MODEL_ID shortcut)
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        mistral_api_key="your_mistral_key",  # required to ingest any PDF
    )

    result = await run_pipeline(input_raw="./raw_documents")

    print(f"Success: {result.success}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")


asyncio.run(main())
```

This example uses Neo4j, the default graph backend (`pip install "scinr[neo4j]"`). To run on PostgreSQL or SQLite instead, see the recipes in the [Quick Start](user-guides/quick-start.md#running-without-neo4j).

---

## How It Works

1. **Configure** — Call `configure()` with graph (Neo4j or SQL) and storage settings (these may also come from the environment) and, for anything that runs an LLM stage, your model as `llm=`. If `llm` is omitted and the environment variable `MODEL_ID` is set, newton builds an AWS Bedrock `ChatBedrockConverse` from it (reading `AWS_DEFAULT_REGION`, default `us-east-1`, and `MAX_TOKENS`, default `65536`).

2. **Run the pipeline** — Call `run_pipeline()` with your input directory. `scinr` handles document conversion, structure extraction, ingestion, annotation, and entity extraction.

3. **Query the knowledge graph** — Explore extracted entities and relationships with the [navigation API](user-guides/graph-navigation.md), or directly in the engine you chose (Cypher on Neo4j, SQL on PostgreSQL or SQLite).

---

## Documentation

* **[Getting Started](getting-started.md)** — Installation, prerequisites, and your first ingestion run.

* **[Configuration](configuration.md)** — `configure()`, environment variables, LLM settings, and prompts.

* **[Architecture](architecture.md)** — Pipeline stages, data flow, and system design.

* **[SQL Graph Backends](user-guides/sql-graph.md)** — Run the graph, storage and snapshots on PostgreSQL or SQLite instead of Neo4j.
