# Architecture

`scinr.newton` processes life sciences documents through a modular 5-stage pipeline, storing raw structure and extracted entities in Neo4j and MongoDB.

---

## 5-Stage Ingestion Pipeline

```
Raw Documents (.pdf, .docx, .pptx, .xlsx, .csv)
                    │
                    ▼
       ┌─────────────────────────┐
       │ Stage 0: Preprocess     │  Format Converters -> JSON / Markdown
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Stage 1: Extraction     │  Section Chunking & Hierarchy Parsing
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Stage 2: Ingestion      │  Document & Structure Nodes written to Neo4j
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Stage 3: Annotation     │  LLM Relevance Filtering & Target Schema Prep
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Stage 4: Entity Extract │  Structured Pydantic Extraction & Graph Subgraph
       └─────────────────────────┘
```

### Stage Details

* **Stage 0: Preprocess** — Uses custom document converters (`python-docx`, `pdfplumber`, `python-pptx`, `openpyxl`, `pandas`) to convert heterogeneous raw formats into standardized internal representations.
* **Stage 1: Extraction** — Chunks documents into structural blocks (paragraphs, tables, headers, figures) while retaining parent-child document hierarchy.
* **Stage 2: Ingestion** — Writes document metadata and structural nodes into Neo4j (`:Document`, `:StructureNode`).
* **Stage 3: Annotation** — Uses LLMs to evaluate structure nodes against target extraction models and attach domain annotations.
* **Stage 4: Entity Extraction** — Invokes target Pydantic schemas on annotated nodes to extract typed domain entities and write subgraph nodes and triples into Neo4j.

