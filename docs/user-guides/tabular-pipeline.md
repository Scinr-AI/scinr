# Tabular Pipeline

The tabular pipeline is a **complete alternative path** to the standard Stages 0-4 of `scinr.newton`. It processes `.csv`, `.xlsx`, and `.xls` files directly — reading headers, mapping columns to extraction model fields, instantiating Pydantic models from row data, and writing structured graph subgraphs to Neo4j.

This is the definitive reference for the tabular pipeline. Every aspect of file discovery, header normalization, column mapping, model instantiation, normalization integration, and Neo4j output is documented here.

!!! note "LLM setup"
    The tabular pipeline calls an LLM for theme classification, model decision
    and column mapping, so it needs `configure(llm=...)` — there is no `MODEL_ID`
    env var. The examples use a local open-source model via
    [Ollama](https://ollama.com/) (`pip install "scinr[ollama]"`, then
    `ollama pull llama3`):

    ```python
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3")   # or any LangChain BaseChatModel
    ```

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
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
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
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_tabular_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
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

1. Auto-detects delimiter via `csv.Sniffer` on the first 4096 characters, then rewinds.
2. Streams the file once (it is never loaded as a whole text), skipping empty rows.
3. Parses each row.
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

### 5.4 Memory: rows are streamed, not held in the graph state

The pipeline never keeps a sheet's data rows in memory. `load_sheets` only *scans* each sheet (headers, row count and the preview above — a sheet with more than 5 rows is read twice to pick the preview rows), so the LangGraph state carries no rows. The rows are re-read from the source file in batches of 500 when the sheet is written, so the memory of the write path depends on the batch, not on the number of rows. With normalization enabled the file is read in three sequential passes (scan the normalization keys → normalize the unique keys → write). Memory is O(batch + unique normalization keys); time and LLM calls still grow with the number of unique values.

> **The source file must not be modified while it is being processed:** it is read several times, and the passes must see the same rows.

The materialising helpers `read_csv()`, `read_xlsx()` and `read_tabular_file()` still exist and return `all_rows`; the streaming API is `scan_tabular_file()` and `iter_sheet_batches()`.

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

### 7.4 Combining Values When Multiple Columns Map to the Same Field

It is common for the LLM column mapping (`map_columns`) to map **two or more columns to the same model field** — for example, when a sheet does not have a clean 1:1 header-to-field correspondence and the LLM reasonably assigns overlapping columns to the same target. When this happens, the pipeline **combines** the values instead of silently letting the last-processed column overwrite the field.

**Combination logic — applies only to `str` and `list[str]` fields:**

1. Every non-empty value routed to the same field is collected, in the order the LLM emitted the corresponding entries in `ColumnMapping.mappings` — **not** the column's left-to-right position in the source file.
2. The collected values are deduplicated **by containment**, not by exact string match: if a value is fully contained in — or all of its individual words appear in — another, more complete value in the same group, the shorter/redundant value is dropped and only the more complete value survives.
3. The surviving values are combined:
   - **`str` fields:** joined with `"; "` as the separator.
   - **`list[str]` fields:** appended as separate list elements (no string concatenation).
4. Empty values (`""`, whitespace-only) never participate — they are filtered out before deduplication even starts.

**Example:**

Three columns — `"Drug"`, `"Strength (mg)"`, `"Unit"` — are all mapped to the same field (e.g. `product_description: str`). For a given row:

```
"Drug"           → "Amox 500 mg"
"Strength (mg)"  → "500 mg"
"Unit"           → "mg"
```

Both `"500 mg"` and `"mg"` are fully contained (as a substring, and word-for-word) inside `"Amox 500 mg"`, so they are discarded as redundant. The final value written to `product_description` is:

```
"Amox 500 mg"
```

If the field were `list[str]` instead of `str`, the same containment-based dedup applies, but the surviving value(s) are appended as list elements rather than joined into a single string.

**Any other field type (`int`, `float`, `bool`, `date`, `datetime`, `Enum`, nested submodels, …):**

The combine/dedup logic described above applies **only** to `str` and `list[str]` fields. For every other field type, the pipeline does **not** attempt to merge multiple column values — it keeps the **last** value it processes (the pre-existing "last write wins" behavior) and emits a **warning** in the logs. No exception is raised and the pipeline never crashes because of this, but values from earlier-processed columns mapped to that field are silently discarded.

> **⚠️ Model design recommendation:** because of this asymmetry, **extraction models intended for the tabular pipeline should declare every mappable field as `str` or `list[str]`.** If a field genuinely needs a non-string type (a parsed `int`, a `date`, a validated `Enum` member, etc.), **do not** try to solve it with a `@model_validator`/`@field_validator` on the model — every row instantiation in the tabular pipeline uses `model_construct(**kwargs)` (never a normal constructor call), and `model_construct()` unconditionally skips **all** Pydantic validators, in every mode (`"before"`, `"after"`, `"wrap"`). A validator added for this purpose will simply never execute in the tabular pipeline; the field will silently stay a raw string. The only mechanism that actually performs real Pydantic validation/coercion in this pipeline today is the existing LLM-based normalization system: wrap the value in a **nested Pydantic submodel** and mark it `json_schema_extra={"normalization_model": True, ...}` (see **[§8.1](#81-the-normalization_model-mechanism)**) — the `NormalizationEngine` fills it via `with_structured_output()` + `TypeAdapter(...).validate_python(...)`, which is real validation, applied to the nested submodel. See **[Custom Models](custom-models.md)** and the model-creation `AGENTS.md` §4.7 for the full explanation and a worked example.

**Determinism and configuration:**

- Combination order follows the order of entries for that field in `ColumnMapping.mappings` — i.e., the order the LLM emitted the mappings, not the column's position in the source file.
- This behavior is **always active**; there is no `configure()` flag to disable it. Given the same row values and the same column mapping, the combined result is always the same.

**Join-key fields are the exception you should avoid.** Combining is harmless for a descriptive field, but not for a field with identity or join semantics — an `instance_key` field, a field carrying `instance_relationships` (or used in its `join_via`), or a field another model joins on. Packing `ID Code` + `ID` into one value (`"312947; CANTABNEW-0081"`) changes the key, so a reference built from one of the columns never finds it by key. The pipeline therefore:

- marks those fields as `JOIN KEY: map exactly one column` in the column-mapping prompt, so the LLM maps each identifier column to its own field;
- logs **one warning per sheet** when the mapping still sends several columns to such a field: `tabular: sheet 'Portfolio' maps 2 columns ['ID Code', 'ID'] to MarketingAuthorisationRecord.portfolio_record_id, a join-key field …`.

Prefer one field per identifier in the model (`authorisation_code`, `portfolio_id`, …). Same-row links (§7.5) do not depend on the key, so they survive the merge; key-based links (other rows, other documents) do not.

### 7.5 Same-Row Instances: Composition vs Cross-Document References

A row with a primary model and complementary models produces **several `:ModelInstance` nodes under one `:ExtractionResult`**. When one of them declares `instance_relationships` to the class of another instance **of the same row**, the relationship is *composition*: it is linked to that sibling directly, without looking anything up by key.

```
(:ExtractionResult)
  ├── [:HAS_MARKETING_AUTHORISATION_RECORD] ──► (ma:ModelInstance)   ← key-less
  └── [:HAS_REGULATORY_PRODUCT_SPECIFICATION] ──► (spec:ModelInstance)
                                                       └── [:SPECIFICATION_FOR {resolution: "sibling"}] ──► (ma)
```

The rule is generic — any models, any number of complementary models per row, any declared relationship — and lives in `entity_extraction.instance_links.match_sibling`. For each target the relationship addresses (one per item for a fan-out list):

1. **Candidates** are the instances of `target_model` written under the same `:ExtractionResult` (nested ones included), except the source itself.
2. Each candidate is compared with the join values on every `join_via` field, after normalisation (lower-case, accents stripped, whitespace collapsed). A value is also split on `;`, so `"312947"` is *contained* in `"312947; CANTABNEW-0081"`.
3. The first non-empty tier wins: **exact** (every field equal) → **compatible** (no field contradicts, at least one equal or contained) → **unknown** (the candidate has no value for any join field).
4. **One** candidate in that tier → linked. **Several** → *ambiguous*: none is picked, a warning is logged and the shell is used (step 6).
5. No candidate in any tier, but **exactly one** candidate of the class, the relationship addresses a single target (not a fan-out), and that candidate is **key-less** → linked (*sole sibling*): a key-less instance can never be reached by key, so the sibling is the only thing the reference can mean. A *keyed* sibling whose key contradicts the reference is a different record — the reference is cross-document and keeps its shell.
6. Otherwise — no instance of that class in the row, or none matching — a key-addressed **shell** is MERGE'd exactly as before (`make_instance_uid(target_model, join keys, tenant)`) and the relationship points at it (`resolution: "key"`). When the target is extracted later — another row, another document — with `instance_key` fields equal to those join values, its MERGE lands on the same uid and **fills the shell**.

Relationships are resolved **after** every instance of the row is written, so it does not matter whether the source model comes before or after its target in the composite. Each `:ExtractionResult` write logs its outcome counts, e.g. `instance_relationships: {'compatible': 1}`.

| Situation | Result |
| :--- | :--- |
| Sibling of the target class matches | Edge to the sibling, no shell (`resolution: "sibling"`) |
| Several siblings match equally | Warning, edge to the key-addressed shell |
| No sibling of the target class (cross-row / cross-document) | Edge to the key-addressed shell, filled later by MERGE |
| Shell filled later | Same uid as the real keyed instance — the edge now reaches it |

**Self-references across rows** (e.g. `Part.parent_id → Part.part_id`) never pick the source itself: they resolve by key, so the target row's keyed instance fills the shell.

**Repairing graphs written before this behaviour.** Older ingestions linked same-row instances by key only, which left dead shells whenever the target was key-less or its join value was built from several columns. `relink_same_row_instances()` repairs one tenant: it moves every relationship that points at a shell to the same-`ExtractionResult` sibling the rule above picks, deletes the shells left without any relationship, and reports counts. It is idempotent (a second run moves nothing), tenant-scoped, and writes in bounded batches. Run it with `dry_run=True` first:

```python
from scinr.newton import relink_same_row_instances

report = await relink_same_row_instances(tenant_id="a705513c-…", dry_run=True)
print(report.relinked, report.ambiguous, report.unmatched, report.shells_deleted)
report = await relink_same_row_instances(tenant_id="a705513c-…")
```

Ambiguous and unmatched relationships are left untouched (unmatched ones are genuine cross-document references waiting for their target). Do not run it while the same tenant is being ingested.

### 7.6 Supplementary Fields

Supplementary fields (extra columns `decide_model` proposes beyond the models, e.g. `marketed`, `mah_group`, `region`) belong to the row's composite schema, so they are written as properties of the row's `:ExtractionResult`. When they describe the primary record, set `tabular_supplementary_on_primary=True` (env `TABULAR_SUPPLEMENTARY_ON_PRIMARY=true`) to **also** write them on the row's primary `:ModelInstance`:

- they stay on the `:ExtractionResult` too, so existing readers are unaffected;
- a field the primary model declares itself is never overwritten;
- if the primary model is keyed (`instance_key`) and two rows share its key, the last row written wins for these properties, as for any other field of a merged instance.

It is opt-in because it changes what a primary instance carries: properties that are not in its model's schema. The cleaner long-term option, when a column is consistently useful, is to add it as a real field of the model.

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
4. **Batching:** Unique entries are grouped by target type and processed in batches of `normalization_batch_size` (default: 5).
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

The tabular pipeline reuses the **same** subgraph writer (`write_extraction_subgraph`) as the unstructured pipeline, so the node and relationship types match the [Neo4j Graph Model](neo4j-graph.md). One sheet becomes a `:StructureNode:Table`; each data row becomes a `:StructureNode:Row`:

```
(:Document)
  └── [:HAS_STRUCTURE] ──► (:StructureNode:Table)
                                ├── [:HAS_MODEL_DECISION] ──► (:ModelDecision)
                                │                                   └── [:MATCHED_MODEL] ──► (:CatalogModel {name: "ProductRecord"})
                                │                                                                └── [:BELONGS_TO_THEME] ──► (:Theme)
                                └── [:HAS_CHILD] ──► (:StructureNode:Row)
                                                        ├── [:HAS_INFO_UNIT] ──► (:InfoUnit)
                                                        ├── [:HAS_MODEL_DECISION] ──► (:ModelDecision)
                                                        └── [:HAS_EXTRACTION] ──► (:ExtractionResult)
                                                                                    ├── [:USES_PRIMARY_MODEL] ──► (:CatalogModel)
                                                                                    └── [:HAS_<FIELD>] ──► (:ModelInstance {model_class: "ProductRecord"})
                                                                                                              ├── [:REFERENCES] ──► (:LabeledEntity)
                                                                                                              ├── (field properties from model data)
                                                                                                              └── [:HAS_<NORMALIZED_FIELD>] ──► (:ModelInstance {model_class: "NormalizedXxx", ...})
```

> A field marked `normalization_model: True` produces its own child `:ModelInstance` node (linked via `[:HAS_<FIELDNAME>]`) — never a flattened/embedded property on the parent row's node.

### 9.2 Node Types

| Node | Labels | Description |
| :--- | :--- | :--- |
| Document | `:Document` | Source file tracking node (`path`, `version`, `raw_file_id`, …). |
| Table | `:StructureNode:Table` | One per sheet. Root structure node of the sheet. |
| Row | `:StructureNode:Row` | One per data row, a child of the table. |
| InfoUnit | `:InfoUnit` | The row rendered as a Markdown table, attached to the `:Row`. |
| ModelDecision | `:ModelDecision` | The LLM's model-selection decision for the table (also linked from each row). |
| ExtractionResult | `:ExtractionResult` | One per row; anchors the row's extraction output. |
| ModelInstance | `:ModelInstance` (`model_class: "{ModelName}"`) | One per row. All model field values as properties. |
| LabeledEntity | `:LabeledEntity` (`label: "{Label}"`) | One per `entity_label` field value. Globally deduplicated. |
| CatalogModel / Theme | `:CatalogModel`, `:Theme` | The registered model and its theme. |

### 9.3 Relationships

| Relationship | Source → Target | Description |
| :--- | :--- | :--- |
| `HAS_STRUCTURE` | Document → Table | Links document to its (root) table sheet node. |
| `HAS_CHILD` | Table → Row | Links table to its data rows. |
| `HAS_INFO_UNIT` | Row → InfoUnit | Row's data rendered as Markdown. |
| `HAS_MODEL_DECISION` | Table → ModelDecision / Row → ModelDecision | The model-selection decision for this table (also linked from each row). |
| `MATCHED_MODEL` | ModelDecision → CatalogModel | The selected extraction model class. |
| `BELONGS_TO_THEME` | CatalogModel → Theme | The classified thematic domain (declared at the catalog-model level). |
| `HAS_EXTRACTION` | Row → ExtractionResult | The row's extraction output. |
| `USES_PRIMARY_MODEL` | ExtractionResult → CatalogModel | The primary model used for the row. |
| `HAS_<FIELD>` | ExtractionResult → ModelInstance | The row's model instance (relationship type derived from the field name). |
| `REFERENCES` | ModelInstance → LabeledEntity | Entity fields (from `entity_label` annotation). |

### 9.4 Entity Relationships

Fields with `field_relationships` or `instance_relationships` in their `json_schema_extra` create additional graph edges:

- **`field_relationships`:** Connects two `:LabeledEntity` nodes within the same model instance (sibling fields with `entity_label`).
- **`instance_relationships`:** Connects `:ModelInstance` nodes. Within one row the target is the matching **same-row sibling** (composition, no key lookup — see [§7.5](#75-same-row-instances-composition-vs-cross-document-references)); across rows or documents it is resolved via `join_via` key matching.

> **Note:** without an `instance_key` declared on the target model, the `:ModelInstance` nodes created by `instance_relationships` for a *cross-row or cross-document* reference are permanent "shell" nodes that never merge with the real, fully-extracted instance — they stay separate and unresolved. Same-row references are not affected (they link to the sibling). The catalog setup logs a warning at startup for every `join_via` target field that is not `instance_key`. See [Cross-Section ModelInstance Linking via instance_key](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key) for the full mechanism.

These work identically to the unstructured pipeline — the tabular pipeline uses the same entity extraction subgraph writer.

---

## 10. Configuration

### 10.1 Normalization Settings

The tabular normalization engine is configured via `configure()`:

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure

configure(
    # ── LLM (used for theme classification, model decision, column mapping) ──
    llm=ChatOllama(model="llama3"),

    # ── Neo4j ──────────────────────────────────────────────────────
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    neo4j_database="neo4j",

    # ── Tabular normalization ──────────────────────────────────────
    normalization_enabled=True,           # On by default; pass False to disable
    normalization_batch_size=10,          # Max entries per LLM batch (default: 5)
    normalization_llm=ChatOllama(model="llama3"),  # Optional dedicated LLM for normalization
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `normalization_enabled` | `bool` | `True` | Enable/disable the `NormalizationEngine`. When `False`, `normalization_model` fields are inert and stay `None`. |
| `normalization_batch_size` | `int` | `5` | Maximum number of unique normalization entries per LLM call. Higher values batch more entries but increase prompt size. |
| `normalization_llm` | `BaseChatModel` | `None` | Dedicated LLM for normalization calls. Falls back to the main `llm` when `None`. Use a cheaper/faster model here. |

### 10.2 Supplementary Fields

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tabular_supplementary_on_primary` | `bool` | `False` | Also write each row's supplementary fields on its primary `:ModelInstance` (they always stay on the `:ExtractionResult`). Env: `TABULAR_SUPPLEMENTARY_ON_PRIMARY`. See [§7.6](#76-supplementary-fields). |

### 10.3 Concurrency

Tabular normalization LLM calls share the global LLM semaphore configured via `llm_concurrency` in `configure()`. This prevents the normalization engine from exceeding the provider's connection pool limits. The `NormalizationEngine.concurrency` parameter is kept for API compatibility but no longer creates a local semaphore.

---

## 11. Complete Example

### 11.1 Full Pipeline Run

```python
import asyncio
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
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
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
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
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_tabular_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
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
| Only one of several mapped columns' data survives on a field | Field type is not `str`/`list[str]` — multi-column combination only applies to those two types; other types keep the last value processed | Change the field to `str` or `list[str]`. Do **not** add a `model_validator` — it never runs in the tabular pipeline (`model_construct()` skips it). If a real type is genuinely needed, wrap it in a nested submodel marked `normalization_model: True` (see [§7.4](#74-combining-values-when-multiple-columns-map-to-the-same-field) and [§8.1](#81-the-normalization_model-mechanism)) |
| Validation errors on row | Extra columns not mapped to model fields | Add missing fields to model, or use `default=None` |
| Duplicate headers cause issues | Source file has repeated column names | Headers are auto-deduped (`col`, `col_2`, `col_3`) — check mapping |
| XLSX sheet skipped | Worksheet is entirely empty | Empty worksheets are silently skipped (expected behavior) |
| Model decision is `None` | No model in catalog matches the sheet data | Add appropriate extraction models to your theme catalog |
| All columns mapped to `__extra__` | Model decision failed + repair loop exhausted | Check LLM connectivity and model catalog availability |
| Warning `maps N columns … a join-key field` | Several identifier columns mapped to one `instance_key` / join field; their values are joined with `"; "` | Give each identifier its own model field (see [§7.4](#74-combining-values-when-multiple-columns-map-to-the-same-field)) |
| `instance_relationships` edges end in empty shell `:ModelInstance` nodes inside one row | Graph written before same-row sibling resolution | Run `relink_same_row_instances(tenant_id=...)` (see [§7.5](#75-same-row-instances-composition-vs-cross-document-references)) |
| Warning `… match … equally; not picking one` | Several instances of the target class in the row match the reference | Add a distinguishing field to `join_via`, or reference the target by a value that is unique within the row |
| Startup warning `join_via remote field X.y is not instance_key` | Target of an `instance_relationships` declaration has no `instance_key` on its join field | Mark the field `instance_key: True` if cross-row/cross-document references must resolve; same-row links work without it |

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

A normalized field is a **separate child `:ModelInstance` node** (linked via `[:HAS_<FIELDNAME>]`), not a property on the parent row's node. To count how many parent instances got a normalized child:

```cypher
// Rows that produced a normalized_substance child
MATCH (parent:ModelInstance)-[:HAS_NORMALIZED_SUBSTANCE]->(:ModelInstance)
RETURN count(DISTINCT parent) AS normalized_count;

// Rows of a given model class that did NOT
MATCH (parent:ModelInstance {model_class: "ProductRecord"})
WHERE NOT (parent)-[:HAS_NORMALIZED_SUBSTANCE]->(:ModelInstance)
RETURN count(parent) AS unnormalized_count;
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
- **[Neo4j Graph Storage — Normalized Models](neo4j-graph.md#normalized-models-tabular-pipeline)** — How a normalized field is written to the graph as a separate, linked `:ModelInstance` node (never an embedded property).
- **[Neo4j Graph Storage — instance_key linking](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key)** — Why `instance_relationships` targets need `instance_key` on the target model to actually merge into one node instead of staying permanent unresolved shells.
- **[Architecture](../architecture.md)** — Detailed walkthrough of each pipeline stage, including the tabular LangGraph workflow.
- **[Pipeline API](../api/pipeline.md)** — Auto-generated docstring for `run_pipeline()`.
- **[Tabular API](../api/stages.md)** — Auto-generated docstring for `run_tabular_pipeline()`.
