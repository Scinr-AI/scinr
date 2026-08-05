# scinr

> **Document knowledge library for life sciences**

`scinr.newton` turns life-sciences documents into connected, queryable knowledge in Neo4j. Use it as a Python library or through the `newton` command-line tool.

## What it provides

* Ingestion for common document and spreadsheet formats.
* Configurable extraction models for domain-specific data.
* Neo4j storage for documents and extracted knowledge.
* A Python API and CLI for complete runs or targeted operations.

## Quick example

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=my_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your-password",
    )

    result = await run_pipeline(input_raw="./raw_documents")
    print(result.success)

asyncio.run(main())
```

See [Getting Started](getting-started.md) for setup and [Configuration](configuration.md) for the supported options.
