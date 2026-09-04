# scinr

> **AI-powered knowledge extraction for life sciences**

`scinr` (`scinr.newton`) is a Python library for turning life sciences documents and tabular data into structured, queryable knowledge.

For unstructured documents, `scinr` extracts domain entities and relationships and stores them as connected knowledge graphs in **Neo4j**, with optional document storage in **MongoDB**.

It uses LLMs and Pydantic models to extract structured, domain-specific information from scientific content.

---

## Key Features

* **5-stage document pipeline** — Preprocess, extract structure, ingest, annotate, and extract domain entities from unstructured documents.

* **Tabular data pipeline** — Normalize scientific spreadsheets and extract structured entities from tabular data.

* **Multi-format ingestion** — Supports .pdf, .docx, .xlsx, and .csv. Support for .pptx, .json, .xml, .html, and .txt is planned in the roadmap.

* **Pydantic extraction models** — Define structured schemas for domain entities such as compounds, clinical trials, and assays.

* **Neo4j knowledge graphs** — Store extracted entities and relationships with document provenance.

* **Optional MongoDB storage** — Store raw files, converted documents, and binary assets using MongoDB/GridFS. Support for other database are planned in the roadmap.

* **LLM-ready documentation** — Provides `llms.txt` and `llms-full.txt` files for AI coding agents and other LLM-based tools.

---

## Pipelines

### Unstructured Documents

Unstructured documents go through five stages:

```text
Raw Documents
(.pdf, .docx, .pptx, .json, .html, .txt, ...)
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
   Store document structure and provenance in Neo4j
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
    llm = ChatBedrockConverse(...)

    configure(
        llm=llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        mistral_api_key="", # Needed por pdfs OCR
    )

    result = await run_pipeline(input_raw="./raw_documents")

    print(f"Success: {result.success}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")


asyncio.run(main())
```

---

## How It Works

1. **Configure** — Set up Neo4j, MongoDB, and LLM providers using `configure()` or environment variables.

2. **Run the pipeline** — Call `run_pipeline()` with your input directory. `scinr` handles document conversion, structure extraction, ingestion, annotation, and entity extraction.

3. **Query the knowledge graph** — Explore extracted entities and relationships in Neo4j.

---

## Documentation

* **[Getting Started](getting-started.md)** — Installation, prerequisites, and your first ingestion run.

* **[Configuration](configuration.md)** — `configure()`, environment variables, LLM settings, and prompts.

* **[Architecture](architecture.md)** — Pipeline stages, data flow, and system design.
