# scinr

> **AI-Powered Document Knowledge Library for Life Sciences**

`scinr` (`scinr.newton`) is a Python library and command-line interface (`newton`) designed to transform unstructured and structured life sciences documents into queryable, connected knowledge graphs in **Neo4j** and document stores in **MongoDB**.

---

## Key Features

* 🚀 **5-Stage Pipeline**: Seamlessly convert, extract, ingest, annotate, and extract entities from raw files.
* 📄 **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.json`, `.html`, and `.txt`.
* 📊 **Tabular Pipeline**: Auto-detection, structural normalization, and LLM entity extraction for scientific spreadsheets.
* 🏷️ **Pydantic Extraction Models**: Define structured schemas for scientific target entities (e.g. compound synthesis, clinical trials, assays).
* 🕸️ **Neo4j Graph Integration**: Automatically map extracted entities and triples into Neo4j subgraphs.
* 🤖 **Agent Ready**: Native support for <a href="llms.txt">llms.txt</a> and <a href="llms-full.txt">llms-full.txt</a> context windows for AI agents.

---

## Quick Example

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    # 1. Configure backend connections and model defaults
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_auth=("neo4j", "password"),
        llm_provider="openai",
        llm_model="gpt-4o",
    )

    # 2. Run the end-to-end ingestion pipeline on a directory of documents
    result = await run_pipeline(
        input_raw="./raw_documents",
        model_class="CompoundAssayModel",
        parallel_docs=4,
    )

    print(f"Pipeline Succeeded: {result.success}")
    print(f"Total Duration: {result.total_duration_seconds:.2f}s")
    print(f"Executed Stages: {result.stages_executed}")

if __name__ == "__main__":
    asyncio.run(main())
```
