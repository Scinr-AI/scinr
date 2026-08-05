# Architecture

`scinr.newton` turns source documents into connected, queryable knowledge in Neo4j. It preserves the connection between extracted data and the document content it came from.

```text
┌──────────────────────────────────────────────────────────────┐
│ Source files                                                 │
│ PDF, Office documents, web content, and spreadsheets         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Structured document data                                     │
│ Content is organized for reliable processing                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Domain knowledge                                             │
│ Configured models identify relevant information              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Neo4j graph                                                  │
│ Documents, context, and extracted entities stay connected    │
└──────────────────────────────────────────────────────────────┘
```

## Inputs and outputs

The library accepts PDF, DOCX, PPTX, XLSX, XLS, CSV, JSON, HTML, XML, and text files.

* **Documents** produce a graph representation that retains document context alongside extracted domain data.
* **Spreadsheets and CSV files** are treated as structured input and are detected automatically during a complete run.
* **Neo4j** is the primary query surface for the resulting connected knowledge.

## Using the library

Configure the library once at application startup, then choose the entry point that fits your application:

| Need | Public entry point |
|---|---|
| Process a directory end to end | `run_pipeline(input_raw="...")` |
| Run a targeted operation | Exported functions such as `run_preprocess()` or `run_ingestion()` |
| Run from automation or a terminal | `newton` CLI |
| Tailor extracted data to a domain | Custom extraction models |

Every pipeline call returns result objects that report overall success and document-level outcomes, so applications can surface failures or retry affected inputs.

## Data lifecycle

An ingestion creates graph data for the source document and its extracted knowledge. Subsequent runs support three common maintenance workflows:

| Workflow | Use when |
|---|---|
| New ingestion | Adding a new document or revision. |
| Update | Correcting the latest ingested document in place. |
| Replacement | Recording that a new document supersedes an existing one. |

Use `update_mode=True` or `replaces="ExistingDocument"` in Python, or `--update` and `--replaces` with the CLI.

## Extension points

Developers can extend the library without changing the ingestion workflow:

* Add custom extraction models for new domain data.
* Add custom converters for additional file types.
* Select optional storage for raw source material when required by the deployment.

See [Configuration](configuration.md), [Custom Extraction Models](user-guides/custom-models.md), and [Neo4j Graph Storage](user-guides/neo4j-graph.md) for the corresponding public interfaces.
