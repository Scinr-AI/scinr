# Tabular Pipeline

The tabular pipeline ingests CSV, XLSX, and XLS files into Neo4j as structured data.

## Usage

Run tabular ingestion from the CLI:

```bash
newton --stage tabular --input-raw files/
```

Or use the Python API:

```python
import asyncio
from scinr.newton import configure, run_pipeline

configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="...",
)

result = asyncio.run(run_pipeline(input_raw="files/", stages=["tabular"]))
```

When you run the complete pipeline with `input_raw`, supported tabular files are detected automatically.

## Supported formats

| Extension | Support |
|---|---|
| `.csv` | Delimiter detection and header handling. |
| `.xlsx` | Multi-sheet workbooks. |
| `.xls` | Legacy Excel workbooks. |
