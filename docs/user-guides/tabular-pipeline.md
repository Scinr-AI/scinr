# Tabular Pipeline

The tabular pipeline is a **complete alternative path** to the standard Stages 0-4 of `scinr.newton`. It processes `.csv`, `.xlsx`, and `.xls` files directly — reading headers, mapping columns to extraction model fields, instantiating Pydantic models from row data, and writing structured graph subgraphs to Neo4j.

This is the definitive reference for the tabular pipeline. Every aspect of file discovery, header normalization, column mapping, model instantiation, normalization integration, and Neo4j output is documented here.

---

## 1. Introduction

### What the tabular pipeline is

The tabular pipeline takes structured tabular files (CSV, XLSX) and converts each row into typed Pydantic model instances, which are then written as `:ModelInstance` and `:LabeledEntity` nodes in Neo4j. It bypasses the document-oriented Stages 0-4 entirely, using a dedicated LangGraph workflow instead:

```
load_sheets → prepare_sheet → classify_theme → decide_model → map_columns → write_tabular → (loop for next sheet)
```

### How it differs from the unstructured pipeline

The standard pipeline (Stages 0-4) is designed for unstructured documents (PDF, DOCX, PPTX). It converts files to intermediate representations, extracts document structure via LLM, ingests a hierarchical graph of `:Document` and `:StructureNode` nodes, annotates each section with a model, and extracts entities from free text.

The tabular pipeline replaces this entire flow for structured data:

| Aspect | Unstructured Pipeline (Stages 0-4) | Tabular Pipeline |
| :--- | :--- | :--- |
| **Input** | PDF, DOCX, PPTX, HTML, TXT, MD | CSV, XLSX |
| **Stages** | 0 → 1 → 2 → 3 → 4 | Direct (bypasses 0-4) |
| **Internal flow** | Sequential stages with intermediate files | LangGraph StateGraph per file |
| **Document hierarchy** | Full hierarchy (`:Document` → `:StructureNode` tree) | `:Document` + `:Table` + `:Row` (flat per sheet) |
| **Model selection** | LLM annotation per structure node (Stage 3) | LLM column mapping per sheet |
| **Entity extraction** | LLM extracts from free text per section | Column values mapped to model fields directly |
| **LLM calls** | One per section (annotation) + one per section (extraction) | Three per sheet: classify theme, decide model, map columns |
| **Normalization** | Optional (LLM hint via `description=`) | **Primary use case** for `normalization_model` fields |
| **Neo4j output** | `:Document` + `:StructureNode` tree + entities | `:Document` + `:Table` + `:Row` + `:ModelInstance` + `:LabeledEntity` |

### When to use it

Use the tabular pipeline when:

- Your source data is **already structured** in tabular format (CSV, XLSX).
- You have a known extraction model that can receive column values directly.
- You need **LLM-based normalization** of raw column values into structured nested models (the `normalization_model` mechanism).
- You want to avoid the overhead of document conversion, structure extraction, and free-text entity extraction.

---

## 2. Pipeline Architecture

### 2.1 LangGraph StateGraph

The tabular pipeline uses a LangGraph `StateGraph` that processes one file at a time, iterating over its sheets (CSV files have one sheet; XLSX files can have multiple):

```
┌─────────────┐
│ load_sheets │  Read file, build previews, store pages
└──────┬──────┘
       │
       ▼
┌───────────────┐
│ check_done? ──┼──── end ──► END
└──────┬────────┘
       │ more sheets
       ▼
┌───────────────┐
│ prepare_sheet │  Load current sheet data
└──────┬────────┘
       │
       ▼
┌────────────────┐
│ classify_theme │  LLM Call 0: detect thematic domain
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ decide_model  │  LLM Call 1: select best extraction model
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ map_columns   │  LLM Call 2: map columns → model fields
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ write_tabular │  Write Table + Row subgraph + entities to Neo4j
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ check_done? ──┼──── end ──► END
└───────────────┘
       │ more sheets
       └──────► loop back to prepare_sheet
```

### 2.2 Per-sheet processing

Each sheet goes through three LLM calls:

1. **`classify_theme`** — The LLM examines column headers and a data preview to classify the sheet's thematic domain (e.g., `"pharmaceutical_quality"`). This narrows the model catalog to the relevant theme.

2. **`decide_model`** — The LLM receives the sheet preview (headers + up to 5 representative rows as Markdown) and the catalog of models from the classified theme. It selects the best `AnnotationDecision` (primary model class, optional complementary models, supplementary fields).

3. **`map_columns`** — The LLM maps each column header to a field in the selected model, producing a `ColumnMapping` with confidence scores and notes. Unmapped columns are tracked separately.

After mapping, `write_tabular` instantiates Pydantic models from each row, runs normalization (if enabled), and writes the complete subgraph to Neo4j.

---

## 3. Running the Tabular Pipeline

### 3.1 Via `run_pipeline()` — Auto-Detection

When `run_pipeline()` receives an `input_raw` directory containing tabular files, it automatically routes them to the tabular pipeline alongside the standard Stages 0-4:

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # Auto-detection: CSV/XLSX files in input_raw are processed by tabular pipeline
    # PDF/DOCX files are processed by Stages 0-4
    result = await run_pipeline(input_raw="./mixed_data")

    # Inspect tabular results
    if result.tabular:
        print(f"Tabular: {result.tabular.total_processed} files, "
              f"{result.tabular.total_failed} failed")

asyncio.run(main())
```

### 3.2 Via `run_pipeline()` — Tabular Only

To process **only** tabular files (skipping Stages 0-4 entirely):

```python
# Tabular-only pipeline
result = await run_pipeline(
    input_raw="./tabular_data",
    stages=["tabular"],
)
```

> **Important:** `"tabular"` cannot be combined with other stages in the `stages=` list. When you set `stages=["tabular"]`, it runs exclusively. When you omit `"tabular"` from `stages` (the default), tabular files in `input_raw` are auto-detected and processed automatically alongside the main pipeline.

### 3.3 Via `run_tabular_pipeline()` — Direct Call

For full control, call the tabular pipeline directly:

```python
from scinr.newton import configure, run_tabular_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    result = await run_tabular_pipeline(
        input_raw="./tabular_data",
        parallel_docs=4,
        tabular_extensions={".csv", ".xlsx"},
        tabular_delimiter=",",
    )

    print(f"Success: {result.success}")
    print(f"Files: {result.total_processed}")

asyncio.run(main())
```

### 3.4 Full Signature

```python
async def run_tabular_pipeline(
    input_raw: str,
    update_mode: bool = False,
    parallel_docs: int = 1,
    tabular_extensions: set | None = None,
    tabular_delimiter: str | None = None,
) -> StageResult
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_raw` | `str` | *(required)* | Folder containing raw tabular files (searched recursively). |
| `update_mode` | `bool` | `False` | If `True`, wipe existing Table/Row subgraph and re-insert at the same version. |
| `parallel_docs` | `int` | `1` | Maximum number of files to process concurrently. Default is 1 (sequential). |
| `tabular_extensions` | `set[str] \| None` | `{".csv", ".xlsx", ".xls"}` | File extensions to treat as tabular. |
| `tabular_delimiter` | `str \| None` | `None` | Field delimiter for CSV files. When `None`, auto-detected. |

---

## 4. File Discovery

### 4.1 Default Extensions

The pipeline searches recursively in `input_raw` for files with these extensions (case-insensitive):

| Extension | Format | Support |
| :--- | :--- | :--- |
| `.csv` | Comma-separated values | Full (auto-delimiter detection) |
| `.xlsx` | Excel 2007+ | Full (multi-sheet support) |
| `.xls` | Excel 97-2003 | **Not supported** — raises `ConversionError` |

> **Note:** `.xls` files (Excel 97-2003 binary format) are not supported by `openpyxl`. If discovered, the pipeline raises a `ConversionError` with instructions to convert the file to `.xlsx` first.

### 4.2 Custom Extensions

Extend the set of recognized extensions:

```python
# Include .tsv and .dat files as tabular data
result = await run_pipeline(
    input_raw="./data",
    tabular_extensions={".csv", ".tsv", ".dat", ".xlsx"},
)
```

Custom extensions are passed through to the tabular agent. If the extension is not `.csv` or `.xlsx`, the reader will raise a `ValueError` for unsupported formats.

### 4.3 Custom Delimiter

Force a specific delimiter for CSV files:

```python
# Force tab-delimited CSV processing
result = await run_pipeline(
    input_raw="./data",
    tabular_delimiter="\t",
)
```

When `tabular_delimiter` is `None` (default), the pipeline uses Python's `csv.Sniffer` to auto-detect the delimiter from a 4096-byte sample. Supported delimiters: `,`, `;`, `\t`, `|`. Falls back to `,` if detection fails.

---

## 5. File Reading and Header Normalization

### 5.1 CSV Reading

CSV files are read with UTF-8-BOM awareness (`utf-8-sig` encoding). The reader:

1. Reads the entire file content.
2. Auto-detects delimiter via `csv.Sniffer` on first 4096 bytes.
3. Parses all rows, skipping empty rows.
4. Treats row 0 as headers, rows 1+ as data.
5. Converts all cell values to strings (strips whitespace).
6. **Deduplicates headers** — if duplicate column names exist, appends `_2`, `_3`, etc. to subsequent occurrences.
7. Pads or trims each data row to match the header count.

```python
# Example: CSV with duplicate headers
# Name, Code, Name, Value
# A, X, B, Y
#
# After deduplication:
# headers = ["Name", "Code", "Name_2", "Value"]
```

### 5.2 XLSX Reading

XLSX files are read via `openpyxl` in `read_only=True, data_only=True` mode. The reader:

1. Opens the workbook.
2. Iterates over all worksheets.
3. For each worksheet: converts cells to strings, strips whitespace, skips empty rows.
4. Treats row 0 as headers.
5. Deduplicates headers (same as CSV).
6. Pads/trims data rows to header count.
7. Skips empty worksheets entirely.

Each worksheet becomes a separate `TabularSheet` entry, processed independently through the LangGraph.

### 5.3 Preview Generation

For LLM calls (classify_theme, decide_model, map_columns), the pipeline generates a preview of up to 5 representative rows:

- **≤ 5 rows:** all rows included.
- **> 5 rows:** rows at indices 0, ~25%, ~50%, ~75%, and last row.

The preview is rendered as a GFM Markdown table for LLM context.

---

## 6. Column Mapping

### 6.1 Theme Classification (LLM Call 0)

Before model selection, the pipeline classifies the sheet's thematic domain. The LLM receives:

- Document name and sheet name.
- All column headers.
- Preview rows as Markdown table.

Output: a `ThemeClassification` with the detected theme path and a justification. On any failure, falls back to `"default"` (never crashes the graph).

### 6.2 Model Decision (LLM Call 1)

The LLM receives:

- The catalog of models from the classified theme (class docstrings, field descriptions).
- Sheet preview as Markdown (headers + up to 5 rows).
- Total row count.

Output: an `AnnotationDecision` containing:
- `matched_model_class` — the primary extraction model class name.
- `complementary_models` — optional additional models to extract alongside.
- `supplementary_fields` — optional additional fields to include.
- `confidence` — the LLM's confidence level.
- `justification` — reasoning for the decision.

If no model matches, `matched_model_class` is `None` and all columns are mapped to `__extra__` (stored as raw data without model instantiation).

### 6.3 Column-to-Field Mapping (LLM Call 2)

The LLM receives:

- The selected model class and its full schema.
- Sheet preview as Markdown.
- All column headers.

Output: a `ColumnMapping` containing:
- `mappings` — list of `ColumnFieldMapping` entries (column name → model field name, with confidence and notes).
- `unmapped_columns` — columns that could not be mapped to any field.

Each mapping entry:

```python
class ColumnFieldMapping:
    column_name: str          # Original column header
    model_field_name: str     # Target field in the extraction model
    confidence: str           # "high", "medium", "low"
    notes: str                # Explanation of the mapping
```

### 6.4 Mapping Fallbacks

If the LLM mapping fails entirely (parse error + repair loop exhaustion), all columns are mapped to `__extra__` with `confidence="low"`. This ensures the pipeline never crashes — data is still stored, just without model structuring.

---

## 7. Model Instantiation

### 7.1 Row-to-Model Conversion

After column mapping, each data row is converted into a Pydantic model instance:

1. The column mapping defines which column value goes to which model field.
2. Column values are assembled into a dictionary matching the model's field names.
3. The Pydantic model is instantiated from the dictionary.
4. Pydantic validation runs (including `extra="forbid"` from `ExtractionModel`).

### 7.2 Validation Behavior

Since extraction models inherit from `ExtractionModel` (which sets `extra="forbid"`), any column value that doesn't map to a declared field causes a validation error. The pipeline handles this gracefully:

- **Mapped columns:** values are set on the model instance.
- **Unmapped columns:** tracked in `ColumnMapping.unmapped_columns` and stored as `__extra__` data on the row.
- **Type mismatches:** Pydantic's `str_strip_whitespace=True` auto-trims strings. Other type coercion follows Pydantic's default behavior.

### 7.3 Complementary Models

When the `AnnotationDecision` includes complementary models, the pipeline resolves them and composes a composite schema. Each row can produce instances of the primary model and all complementary models simultaneously.

---

## 8. Normalization Integration

### 8.1 The `normalization_model` Mechanism

The `normalization_model` field annotation is the **primary feature** of the tabular pipeline. It enables LLM-based normalization of raw column values into structured nested models:

```python
from pydantic import Field
from scinr.newton.models.base import ExtractionModel


class NormalizedSubstance(ExtractionModel):
    """Structured, normalized substance data."""

    substance_name: str | None = Field(
        default=None,
        description="Canonical substance name.",
    )
    substance_type: str | None = Field(
        default=None,
        description="Type: API, excipient, preservative, etc.",
    )
    cas_number: str | None = Field(
        default=None,
        description="CAS registry number, if present.",
    )


class ProductRecord(ExtractionModel):
    """A single product record from a product catalogue CSV."""

    product_name: str = Field(
        ...,
        description="Product name from the 'Name' column.",
    )
    raw_substance: str = Field(
        ...,
        description="Active substance as written in the source column.",
    )
    raw_strength: str = Field(
        ...,
        description="Strength as written in the source column.",
    )
    manufacturer: str = Field(
        ...,
        description="Manufacturer name.",
    )

    # Normalization: raw → structured (tabular pipeline)
    normalized_substance: NormalizedSubstance | None = Field(
        default=None,
        description="Structured substance data derived from raw_substance.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_substance"],
        },
    )
    normalized_strength: NormalizedStrength | None = Field(
        default=None,
        description="Structured strength data derived from raw_strength.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_strength"],
        },
    )
```

### 8.2 How Normalization Works

The `NormalizationEngine` processes instances in batches:

1. **Detection:** For each model instance, the engine scans fields for `json_schema_extra["normalization_model"] == True`.
2. **Source extraction:** For each normalizable field, it extracts the values of the sibling fields listed in `normalization_source_fields`.
3. **Deduplication:** Instances with identical source values are grouped by a hash key — the LLM is called once per unique source combination.
4. **Batching:** Unique entries are grouped by target type and processed in batches of `normalization_batch_size` (default: 3).
5. **LLM call:** Each batch is sent to the LLM with structured output, requesting normalized instances of the target type.
6. **Application:** Results are applied back to the original instances via `setattr` (with validation bypass fallback).
7. **Caching:** Results are cached by hash key — duplicate source values reuse the cached normalization.

### 8.3 `normalization_model` is Mandatory for Tabular Pipeline

Without the `normalization_model: True` + `normalization_source_fields` annotation on a nested field, the tabular `NormalizationEngine` hook has nothing to trigger on. The nested submodel field silently stays `None` for every row, with no error raised.

| Pipeline | `normalization_model` required? |
| :--- | :--- |
| Tabular only | ✅ **Mandatory** — without it, the nested field is never populated |
| Unstructured only (Stages 3-4) | ⚪ Optional — the extraction LLM fills it from `description=` |
| Both | ✅ **Recommended** — mandatory for tabular, optional-but-useful for unstructured |

### 8.4 `normalization_source_fields`: Always Explicit

**Never omit `normalization_source_fields`.** If you omit it or leave it empty, the engine silently uses **all other scalar fields** of the parent model as source data — wasting tokens and leaking irrelevant context.

```python
# ✅ GOOD — explicit, minimal source fields
normalized_address: NormalizedAddress | None = Field(
    default=None,
    description="...",
    json_schema_extra={
        "normalization_model": True,
        "normalization_source_fields": ["raw_address"],  # only what's needed
    },
)

# ❌ BAD — implicit fallback vacuums ALL scalar fields
normalized_address: NormalizedAddress | None = Field(
    default=None,
    description="...",
    json_schema_extra={
        "normalization_model": True,
        # Missing normalization_source_fields — sends raw_name, raw_address,
        # raw_phone, internal_notes, etc. to the LLM
    },
)
```

### 8.5 Normalization Caching

The `NormalizationEngine` caches results by a hash of the source values. If two rows have identical source data (e.g., the same `"Paracetamol 500mg"` in `raw_strength`), the LLM is called only once and the result is reused. This dramatically reduces LLM calls for datasets with repeated values.

---

## 9. Neo4j Output

### 9.1 Graph Structure

The tabular pipeline writes the following nodes and relationships to Neo4j:

```
(:Document)
  └── [:HAS_STRUCTURE] ──► (:StructureNode:Table)
                                ├── [:HAS_MODEL_DECISION] ──► (:ModelDecision)
                                │                                   ├── [:MATCHES_MODEL] ──► (:Model {class: "ProductRecord"})
                                │                                   └── [:MATCHES_THEME] ──► (:Theme)
                                └── [:HAS_STRUCTURE] ──► (:StructureNode:Row)
                                                            ├── [:HAS_INFO_UNIT] ──► (:InfoUnit)
                                                            ├── [:HAS_MODEL_DECISION] ──► (:ModelDecision)
                                                            └── [:HAS_MODEL_INSTANCE] ──► (:ModelInstance:ProductRecord)
                                                                                              ├── [:HAS_LABELED_ENTITY] ──► (:LabeledEntity)
                                                                                              └── (field properties from model data)
```

### 9.2 Node Types

| Node | Labels | Created by | Description |
| :--- | :--- | :--- | :--- |
| Document | `:Document` | Tabular agent | Source file tracking node (path, version, raw_file_id). |
| Table | `:StructureNode:Table` | `write_tabular` | One per sheet. Contains sheet metadata (column/row count, theme). |
| Row | `:StructureNode:Row` | `write_tabular` | One per data row. Contains row data as InfoUnit Markdown. |
| ModelDecision | `:ModelDecision` | `write_annotation` | The LLM's model selection decision for the table. |
| ModelInstance | `:ModelInstance:{ModelName}` | `write_extraction_subgraph` | One per row. Contains all model field values as properties. |
| LabeledEntity | `:LabeledEntity:{Label}` | `write_extraction_subgraph` | One per `entity_label` field value. Globally deduplicated. |
| InfoUnit | `:InfoUnit` | `write_tabular` | Markdown table representation of a single row. |

### 9.3 Relationships

| Relationship | Source → Target | Description |
| :--- | :--- | :--- |
| `HAS_STRUCTURE` | Document → Table | Links document to its table sheets. |
| `HAS_STRUCTURE` | Table → Row | Links table to its data rows. |
| `HAS_MODEL_DECISION` | Table → ModelDecision | The model selection decision for this table. |
| `HAS_MODEL_DECISION` | Row → ModelDecision | Links row to the table's model decision. |
| `HAS_INFO_UNIT` | Row → InfoUnit | Row's data rendered as Markdown. |
| `HAS_MODEL_INSTANCE` | Row → ModelInstance | The Pydantic model instance for this row. |
| `HAS_LABELED_ENTITY` | ModelInstance → LabeledEntity | Entity fields (from `entity_label` annotation). |
| `MATCHES_MODEL` | ModelDecision → Model | The selected extraction model class. |
| `MATCHES_THEME` | ModelDecision → Theme | The classified thematic domain. |

### 9.4 Entity Relationships

Fields with `field_relationships` or `instance_relationships` in their `json_schema_extra` create additional graph edges:

- **`field_relationships`:** Connects two `:LabeledEntity` nodes within the same model instance (sibling fields with `entity_label`).
- **`instance_relationships`:** Connects `:ModelInstance` nodes across rows or documents via `join_via` key matching.

These work identically to the unstructured pipeline — the tabular pipeline uses the same entity extraction subgraph writer.

---

## 10. Configuration

### 10.1 Normalization Settings

The tabular normalization engine is configured via `configure()`:

```python
from scinr.newton import configure

configure(
    # ── Neo4j ──────────────────────────────────────────────────────
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",

    # ── LLM (used for theme classification, model decision, column mapping) ──
    llm=my_llm,

    # ── Tabular normalization ──────────────────────────────────────
    normalization_enabled=True,           # Enable the NormalizationEngine
    normalization_batch_size=10,          # Max entries per LLM batch (default: 3)
    normalization_llm=cheaper_llm,        # Optional dedicated LLM for normalization
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `normalization_enabled` | `bool` | `True` | Enable/disable the `NormalizationEngine`. When `False`, `normalization_model` fields are inert and stay `None`. |
| `normalization_batch_size` | `int` | `3` | Maximum number of unique normalization entries per LLM call. Higher values batch more entries but increase prompt size. |
| `normalization_llm` | `BaseChatModel` | `None` | Dedicated LLM for normalization calls. Falls back to the main `llm` when `None`. Use a cheaper/faster model here. |

### 10.2 Concurrency

Tabular normalization LLM calls share the global LLM semaphore configured via `llm_concurrency` in `configure()`. This prevents the normalization engine from exceeding the provider's connection pool limits. The `NormalizationEngine.concurrency` parameter is kept for API compatibility but no longer creates a local semaphore.

---

## 11. Complete Example

### 11.1 Full Pipeline Run

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        # Enable normalization for tabular data
        normalization_enabled=True,
        normalization_batch_size=10,
    )

    result = await run_pipeline(
        input_raw="./product_catalogues",
        stages=["tabular"],
        tabular_extensions={".csv", ".xlsx"},
    )

    print(f"Success: {result.success}")
    if result.tabular:
        print(f"Files processed: {result.tabular.total_processed}")
        print(f"Files failed: {result.tabular.total_failed}")
        for doc in result.tabular.documents:
            status = "OK" if doc.nodes_failed == 0 else f"FAILED ({doc.errors})"
            print(f"  {doc.document_name}: {status}")

asyncio.run(main())
```

### 11.2 Mixed Pipeline (Documents + Tabular)

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        normalization_enabled=True,
    )

    # input_raw contains both PDFs and CSVs
    # PDFs → Stages 0-4 (unstructured pipeline)
    # CSVs → tabular pipeline (auto-detected)
    result = await run_pipeline(input_raw="./mixed_data")

    # Inspect both pipeline results
    if result.ingestion:
        print(f"Documents ingested: {result.ingestion.total_processed}")
    if result.tabular:
        print(f"Tabular files processed: {result.tabular.total_processed}")

asyncio.run(main())
```

### 11.3 Direct Tabular Pipeline with Custom Settings

```python
import asyncio
from scinr.newton import configure, run_tabular_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        normalization_enabled=True,
        normalization_batch_size=5,
    )

    result = await run_tabular_pipeline(
        input_raw="./clinical_data",
        update_mode=False,
        parallel_docs=4,
        tabular_extensions={".csv", ".xlsx"},
        tabular_delimiter=";",  # Force semicolon delimiter
    )

    print(f"Tabular pipeline: {result.success}")
    print(f"Duration: {result.duration_seconds:.2f}s")

asyncio.run(main())
```

---

## 12. Troubleshooting

### 12.1 Common Issues

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| Tabular files not processed | Extension not in `tabular_extensions` | Add extension to `tabular_extensions={".csv", ".tsv", ".dat"}` |
| Normalized fields are `None` | `normalization_enabled=False` in `configure()` | Set `normalization_enabled=True` |
| Normalized fields are `None` | Missing `normalization_model: True` on field | Add `json_schema_extra={"normalization_model": True, ...}` |
| Normalized fields are `None` | Missing `normalization_source_fields` | Add explicit `normalization_source_fields` list |
| Column mapping wrong | Model fields don't match column semantics | Adjust field descriptions to be more specific |
| Column mapping wrong | Theme classification incorrect | Check `THEME_DESCRIPTION` in your `catalog.py` |
| CSV delimiter wrong | Auto-detection failed on unusual format | Set `tabular_delimiter=";"` or `tabular_delimiter="\t"` |
| `.xls` files fail | Excel 97-2003 format not supported | Convert to `.xlsx` first |
| Validation errors on row | Extra columns not mapped to model fields | Add missing fields to model, or use `default=None` |
| Duplicate headers cause issues | Source file has repeated column names | Headers are auto-deduped (`col`, `col_2`, `col_3`) — check mapping |
| XLSX sheet skipped | Worksheet is entirely empty | Empty worksheets are silently skipped (expected behavior) |
| Model decision is `None` | No model in catalog matches the sheet data | Add appropriate extraction models to your theme catalog |
| All columns mapped to `__extra__` | Model decision failed + repair loop exhausted | Check LLM connectivity and model catalog availability |

### 12.2 Debugging Column Mapping

To inspect the column mapping for a specific sheet, query Neo4j:

```cypher
MATCH (t:Table)
WHERE t.title = 'Sheet1'
MATCH (t)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
RETURN t.title AS table,
       md.matched_model_class AS model,
       md.confidence AS confidence,
       md.justification AS justification
```

### 12.3 Debugging Normalization

To check which normalizations were applied:

```cypher
MATCH (mi:ModelInstance)
WHERE mi.normalized_substance IS NOT NULL
RETURN count(mi) AS normalized_count

MATCH (mi:ModelInstance)
WHERE mi.normalized_substance IS NULL
RETURN count(mi) AS unnormalized_count
```

---

## 13. Performance Considerations

### 13.1 LLM Call Budget

Each sheet requires exactly 3 LLM calls (classify theme, decide model, map columns). Normalization adds additional calls proportional to unique source-value combinations divided by `normalization_batch_size`.

For a file with N sheets and M unique normalization entries per type:

```
Total LLM calls = N × 3 + (M / normalization_batch_size) × number_of_normalization_types
```

### 13.2 Parallelism

`parallel_docs` controls how many files are processed concurrently. Default is 1 (sequential). Increase this when:

- Processing many small files.
- LLM provider has high concurrency limits.
- Network latency is the bottleneck.

### 13.3 Normalization Caching

The `NormalizationEngine` caches results by source-value hash. Datasets with many repeated values (e.g., a product catalogue with the same manufacturer across hundreds of rows) benefit significantly from this — the LLM is called once per unique value, not once per row.

---

## See Also

- **[Running the Pipeline](running-pipeline.md)** — Full reference for `run_pipeline()`, including `stages=["tabular"]` and tabular options.
- **[Configuration](../configuration.md)** — All `configure()` parameters, including `normalization_enabled`, `normalization_batch_size`, and `normalization_llm`.
- **[Custom Models](custom-models.md)** — Defining extraction models with `normalization_model` fields for tabular use.
- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding `:ModelInstance`, `:LabeledEntity`, and relationship types in the graph.
- **[Architecture](../architecture.md)** — Detailed walkthrough of each pipeline stage, including the tabular LangGraph workflow.
- **[Pipeline API](../api/pipeline.md)** — Auto-generated docstring for `run_pipeline()`.
- **[Tabular API](../api/stages.md)** — Auto-generated docstring for `run_tabular_pipeline()`.
