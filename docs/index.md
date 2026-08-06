# scinr

> **AI-Powered Document Knowledge Library for Life Sciences**

`scinr` (`scinr.newton`) is a Python library that transforms unstructured and structured life sciences documents into queryable, connected knowledge graphs in **Neo4j** and optional document stores in **MongoDB**.

It provides an async 6-stage pipeline that converts raw files into structured, annotated, and graph-connected domain entities — all driven by LLMs and Pydantic extraction models.

---

## Key Features

* **6-Stage Async Pipeline** — Preprocess, extract, ingest, annotate, entity-extract, and normalize tabular data in a single orchestrated flow.
* **Multi-Format Ingestion** — Supports `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.json`, `.html`, and `.txt`.
* **Tabular Pipeline with LLM Normalization** — Auto-detection, structural normalization, and LLM-powered entity extraction for scientific spreadsheets.
* **Pydantic Extraction Models** — Define structured schemas for scientific target entities (e.g., compound synthesis, clinical trials, assays) with automatic graph annotations.
* **Neo4j Knowledge Graph Output** — Automatically map extracted entities and triples into Neo4j subgraphs with document provenance.
* **Optional MongoDB Storage** — Persist raw files, converted pages, and binary assets via GridFS.
* **Agent-Ready Documentation** — Native support for [llms.txt](https://github.com/Scinr-AI/scinr/blob/main/llms.txt) and [llms-full.txt](https://github.com/Scinr-AI/scinr/blob/main/llms-full.txt) context windows for AI agents.

---

## Pipeline Overview

The ingestion pipeline processes documents through six stages:

```
Raw Documents (.pdf, .docx, .pptx, .xlsx, .csv, .json, .html, .txt)
                     │
                     ▼
        ┌─────────────────────────┐
        │ 1. Preprocess           │  Format converters → JSON / Markdown
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ 2. Extraction           │  Section chunking & hierarchy parsing
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ 3. Ingestion            │  Document & structure nodes → Neo4j
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ 4. Annotation           │  LLM relevance filtering & schema prep
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ 5. Entity Extraction    │  Pydantic extraction → graph subgraphs
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ 6. Tabular              │  Spreadsheet normalization & extraction
        └─────────────────────────┘
```

---

## Quick Example

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    # 1. Configure backend connections
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )

    # 2. Run the end-to-end ingestion pipeline
    result = await run_pipeline(input_raw="./raw_documents")

    print(f"Success: {result.success}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")

asyncio.run(main())
```

---

## How It Works

1. **Configure** — Set up backends (Neo4j, MongoDB, LLM) via `configure()` or environment variables.
2. **Run the Pipeline** — Call `run_pipeline()` with your input directory. The pipeline handles conversion, extraction, ingestion, annotation, and entity extraction automatically.
3. **Query the Graph** — Explore extracted entities and their relationships in Neo4j.

---

## Documentation

* **[Getting Started](getting-started.md)** — Installation, prerequisites, and your first ingestion run.
* **[Configuration](configuration.md)** — Complete reference for `configure()`, environment variables, prompt families, and all settings.
* **[Architecture](architecture.md)** — Detailed pipeline stages, data flow, and system design.
