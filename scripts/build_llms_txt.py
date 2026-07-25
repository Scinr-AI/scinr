#!/usr/bin/env python3
"""
build_llms_txt.py — Generate site/llms.txt and site/llms-full.txt post mkdocs build.
Compliant with https://llmstxt.org specification.
"""
from pathlib import Path

SITE_DIR = Path("site")
DOCS_DIR = Path("docs")


def generate_llms_txt() -> None:
    if not SITE_DIR.exists():
        raise RuntimeError("site/ directory does not exist. Run `mkdocs build` first.")

    # 1. Write llms.txt
    llms_txt_content = """# scinr

> AI-powered document knowledge library for life sciences — scinr.newton document ingestion engine.

## Overview
scinr is a Python library and CLI tool (`newton`) designed to process complex life sciences documents (PDFs, Word documents, PowerPoint presentations, Excel spreadsheets, CSVs) into structured graph knowledge in Neo4j and MongoDB.

## Key Capabilities
- **5-Stage Document Ingestion Pipeline**: Preprocess -> Extract -> Ingest -> Annotate -> Entity Extraction.
- **Tabular Data Normalization**: Automatic detection and normalization of spreadsheet headers and tabular schemas.
- **Graph Knowledge Storage**: Direct mapping of extracted entities, concepts, and relationships into Neo4j subgraphs.
- **Pydantic Extraction Catalogs**: Custom extraction models registered dynamically.

## Documentation
- [Getting Started](getting-started.md): Installation, prerequisites, and quick start example.
- [Architecture](architecture.md): Overview of stages 0-4 and internal storage model.
- [Configuration](configuration.md): Environment settings, database connections, LLM provider setup.
- [CLI Reference](cli.md): `newton` CLI command line tools and options.
- [User Guides - Custom Models](user-guides/custom-models.md): Defining Pydantic extraction models.
- [User Guides - Tabular Pipeline](user-guides/tabular-pipeline.md): Processing Excel and CSV spreadsheets.
- [User Guides - Neo4j Graph](user-guides/neo4j-graph.md): Neo4j graph schema and triple storage.
- [API Reference](api/index.md): Complete Python library API documentation.

## Key Public Python APIs
- `scinr.newton.run_pipeline`: Execute the document ingestion pipeline.
- `scinr.newton.configure`: Configure environment, database URIs, and LLM parameters.
- `scinr.newton.get_config`: Retrieve active library configuration.
- `scinr.newton.exceptions.ScinrError`: Base exception family for scinr error handling.
- `scinr.newton.results.PipelineResult`: Document execution summary and metrics.
"""
    (SITE_DIR / "llms.txt").write_text(llms_txt_content, encoding="utf-8")
    print(f"Generated {SITE_DIR / 'llms.txt'}")

    # 2. Write llms-full.txt
    full_docs_chunks = [
        "# scinr Full Documentation\n\nThis file contains the complete documentation for the scinr library for LLM agent context.\n\n---\n"
    ]

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        # Exclude superpowers plan and spec docs from public llms-full.txt if present inside docs/
        if "superpowers" in md_file.parts:
            continue
        rel_path = md_file.relative_to(DOCS_DIR)
        content = md_file.read_text(encoding="utf-8")
        full_docs_chunks.append(f"## File: {rel_path}\n\n{content}\n\n---\n")

    (SITE_DIR / "llms-full.txt").write_text("".join(full_docs_chunks), encoding="utf-8")
    print(f"Generated {SITE_DIR / 'llms-full.txt'}")


if __name__ == "__main__":
    generate_llms_txt()
