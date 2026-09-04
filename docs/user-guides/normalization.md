# LLM Normalization System

The normalization system turns free-text fields into structured nested Pydantic models via LLM calls. It is wired exclusively into the **tabular pipeline** (CSV/XLSX/XLS), running between column mapping and Neo4j write. The unstructured pipeline (PDF/DOCX) does not use the normalization engine — the LLM fills nested fields directly during entity extraction.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [How Normalization Works (Mechanical Flow)](#2-how-normalization-works-mechanical-flow)
3. [Declaring Normalization Fields](#3-declaring-normalization-fields)
4. [Configuration](#4-configuration)
5. [Mandatory vs. Optional Decision Matrix](#5-mandatory-vs-optional-decision-matrix)
6. [The Implicit Fallback (Footgun)](#6-the-implicit-fallback-footgun)
7. [NormalizationEngine Architecture](#7-normalizationengine-architecture)
8. [Performance Considerations](#8-performance-considerations)
9. [Multiple Normalization Fields](#9-multiple-normalization-fields)
10. [Troubleshooting](#10-troubleshooting)
11. [Complete Example](#11-complete-example)

---

## 1. Introduction

### What normalization is

Normalization is the process of taking a raw, free-text field value and transforming it into a structured nested Pydantic model via an LLM call with structured output. For example, a CSV column containing `"123 Main St, Springfield, IL 62704"` becomes a `NormalizedAddress` instance with separate `street`, `city`, `postal_code`, and `country_code` fields.

### Where it fits in the pipeline

```
CSV Row → Column Mapping → Model Instance
                                  │
                         NormalizationEngine
                                  │
                          LLM structured output
                                  │
                       Apply to Instance (setattr)
                                  │
                          Write to Neo4j
```

The normalization step sits between Pydantic model instantiation and Neo4j graph write. It is a **tabular-only hook**: it never runs during the unstructured pipeline (Stages 3-4).

### Why it matters

Tabular data often contains messy, inconsistent, or composite values in a single column. A "Manufacturer Address" column might contain street, city, state, and country all jammed together. Without normalization, you get one unstructured string property on your Neo4j node. With normalization, you get a properly structured `:ModelInstance` node with queryable, comparable, and deduplicatable fields.

### Key characteristics

| Characteristic | Detail |
|---|---|
| **Opt-in** | Only fields with `normalization_model: True` in `json_schema_extra` are processed |
| **Off by default** | The engine is disabled unless `normalization_enabled=True` |
| **Tabular-only** | Wired into the tabular pipeline; ignored by the unstructured pipeline |
| **Additive** | A normalization field is still an ordinary nested model for all other purposes |
| **Deduplicated** | Identical source values across rows trigger only one LLM call |
| **Batched** | Multiple unique entries of the same target type share a single LLM call |

> **Note:** normalization alone does not create graph links or deduplication. It only produces a clean, structured value. Cross-document/cross-row linking still requires the normalized sub-model to declare its own `entity_label` and/or `instance_key` fields — exactly as any other `:ModelInstance`. See [Normalized Models (Tabular Pipeline)](neo4j-graph.md#normalized-models-tabular-pipeline) in the Neo4j Graph Model guide for the full mechanism and why this distinction matters for linking across datasets.

---

## 2. How Normalization Works (Mechanical Flow)

### End-to-end diagram

```
CSV Row → Column Mapping → Model Instance
                                  │
                         NormalizationEngine
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
               Detection      Batching       LLM Call
               (scan schema)  (group + dedup) (structured output)
                    │             │             │
                    ▼             ▼             ▼
               Source vals   Unique keys   Normalized model
               collected     cached        instances
                    │                         │
                    └─────────┬───────────────┘
                              ▼
                       Apply to Instance
                       (setattr + fallback)
                              ▼
                       Write to Neo4j
```

### Step-by-step

#### Phase 1: Detection (schema inspection)

The `NormalizationEngine` calls `get_normalization_specs(model_class)` for each model class in the pipeline. This function inspects the Pydantic schema and extracts all fields that have `json_schema_extra={"normalization_model": True}`. For each such field, it records:

- **`field_name`** — the name of the normalization target field on the parent model
- **`target_type`** — the Pydantic class of the normalized model (extracted from the field annotation, handling `X | None`, `list[X]`, `Annotated[X, ...]`)
- **`source_fields`** — the list of source field names declared in `normalization_source_fields`, or `None` if omitted (implicit fallback)

```python
# detector.py — simplified
def get_normalization_specs(model_class: type[BaseModel]) -> list[NormalizationSpec]:
    specs = []
    for field_name, field_info in model_class.model_fields.items():
        extra = _get_json_schema_extra(field_info)
        if not extra.get("normalization_model", False):
            continue
        target_type = _extract_target_type(field_info.annotation)
        source_fields = extra.get("normalization_source_fields", []) or []
        specs.append(NormalizationSpec(
            field_name=field_name,
            target_type=target_type,
            source_fields=source_fields if source_fields else None,
        ))
    return specs
```

#### Phase 2: Entry creation (source value collection)

For each model instance, the engine calls `extract_source_values(spec, instance)` to collect the raw values from the declared source fields. If `spec.source_fields` is `None`, it falls back to **all scalar fields** on the model (the implicit fallback — see Section 6).

Each entry is wrapped in a `NormalizationEntry` dataclass:

```python
@dataclass
class NormalizationEntry:
    instance_id: int            # id() of the Pydantic instance
    model_class_name: str       # e.g. "ContactRecord"
    field_name: str             # e.g. "normalized_address"
    target_type: type[BaseModel]  # e.g. NormalizedAddress
    source_values: dict[str, object]  # e.g. {"raw_address": "123 Main St..."}
    unique_key: str             # "{target_type_name}:{md5_hash}"
    row_indices: list[int]      # row indices from pre-scan
```

The **unique key** is constructed as `{target_type.__name__}:{md5_hash}` where the MD5 hash is computed from the sorted, lowercased string representation of the source values. This ensures identical source values across different rows produce the same key.

#### Phase 3: Batching (group by target type)

Entries are grouped by `target_type.__name__` because the LLM structured output call requires a homogeneous target type. Within each group, entries are batched by `normalization_batch_size` (default: 3). Duplicate entries (same unique key from different rows) are deduplicated — only unique keys proceed to the LLM.

```python
# engine.py — simplified
unique_entries: dict[str, list[NormalizationEntry]] = {}
for entry in entries:
    if entry.unique_key in seen_keys:
        continue  # dedup
    seen_keys.add(entry.unique_key)
    type_key = entry.target_type.__name__
    unique_entries.setdefault(type_key, []).append(entry)
```

#### Phase 4: LLM call (structured output)

For each batch, the engine:

1. Builds dynamic Pydantic output schemas (`BatchOutput` and `BatchResponse`) wrapping the target type
2. Calls `self.llm.with_structured_output(BatchResponse)` to create a structured-output LLM
3. Builds a prompt with the system message and all entries' source data
4. Calls `ainvoke()` with retry via `with_llm_retry()`
5. Coerces the result to `BatchResponse` (handles dict returns from some providers)

The dynamic schemas look like:

```python
BatchOutput = type(
    f"BatchOutput_{target_type.__name__}",
    (BaseModel,),
    {
        "__annotations__": {"key": str, "result": target_type},
        "model_config": ConfigDict(extra="forbid"),
    },
)

BatchResponse = type(
    f"BatchResponse_{target_type.__name__}",
    (BaseModel,),
    {
        "__annotations__": {"results": list[BatchOutput]},
        "model_config": ConfigDict(extra="forbid"),
    },
)
```

The system prompt is:

```
You are a data normalization assistant. You receive raw extracted data
and must normalize it into a structured format. Fill in all fields you can
confidently identify from the source data. Leave uncertain fields as null.
```

#### Phase 5: Result caching and application

Results from the LLM are stored in `self.result_cache` keyed by unique key. The engine then applies each result back to the original model instances via `setattr`, with `object.__setattr__` as a fallback for models with `validate_assignment=True` or class identity mismatches:

```python
def _apply_to_instance(self, instance_id, field_name, normalized, all_instances):
    for _, instance in all_instances:
        if id(instance) == instance_id:
            try:
                setattr(instance, field_name, normalized)
            except Exception:
                # Fallback: bypass Pydantic validation
                object.__setattr__(instance, field_name, normalized)
            return
```

#### Phase 6: Write to Neo4j

Once all normalization fields are populated, the instances proceed to the Neo4j write phase. The graph mapper treats the normalized nested models as regular `:ModelInstance` nodes, creating them with their properties and any `entity_label` or `instance_key` relationships they declare.

### Missing results retry

If the LLM returns fewer results than entries in the batch (e.g., it skips one entry), the engine automatically retries the missing keys **once**. The retry is done with the same batch mechanism but only for the missing entries.

---

## 3. Declaring Normalization Fields

### The dual-field pattern

Normalization uses the **dual-field pattern**: a raw free-text field paired with a normalized nested model. The raw field preserves the original text; the normalized field provides structured, queryable components.

```python
from pydantic import Field
from scinr.newton.models.base import ExtractionModel


class NormalizedAddress(ExtractionModel):
    """Structured, normalized postal address."""

    street: str | None = Field(
        default=None,
        description="Street address line, without city or postal code.",
    )
    city: str | None = Field(
        default=None,
        description="City name.",
    )
    postal_code: str | None = Field(
        default=None,
        description="Postal or ZIP code.",
    )
    country_code: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code (e.g. 'US', 'DE', 'JP').",
        json_schema_extra={"entity_label": "Country"},
    )


class ContactRecord(ExtractionModel):
    """A single contact record from a CSV file."""

    # ─── Tier 1: Free-text raw field ───
    raw_name: str = Field(
        ...,
        description="Full name as written in source column.",
    )
    raw_address: str = Field(
        ...,
        description="Free-text address from source column.",
    )
    raw_phone: str | None = Field(
        default=None,
        description="Phone number if present.",
    )

    # ─── Tier 2: Normalization target ───
    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address derived from raw_address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],
        },
    )
```

### Key components

| Component | Location | Purpose |
|---|---|---|
| `normalization_model: True` | `json_schema_extra` on the **target field** | Marks the field for the `NormalizationEngine` |
| `normalization_source_fields` | `json_schema_extra` on the **target field** | Declares which parent fields feed the normalization |
| Target type | Field annotation (e.g., `NormalizedAddress \| None`) | The Pydantic model to populate via LLM |
| Source fields | Regular fields on the **parent model** | The raw data used as LLM input |

### Rules for the target field

| Rule | Details |
|---|---|
| **Must be nullable** | Use `TargetType \| None` with `default=None` — the field starts as `None` until normalization runs |
| **Must be a Pydantic model** | The annotation must resolve to a `BaseModel` subclass (through `X`, `X \| None`, `list[X]`, etc.) |
| **Must have `normalization_model: True`** | Without this flag, the engine skips the field entirely |
| **Should have `normalization_source_fields`** | Without it, the engine falls back to all scalar fields (the implicit footgun — see Section 6) |

### Rules for the target model

| Rule | Details |
|---|---|
| **Inherit from `ExtractionModel`** | Ensures compatibility with the graph mapper |
| **Fields should be nullable** | Use `str \| None` with `default=None` — the LLM may not be able to extract all fields |
| **Use `entity_label` for dedup fields** | Fields like `country_code` or `substance_name` benefit from cross-document deduplication |
| **Use `instance_key` for composite identity** | If the normalized model represents a globally unique entity, declare instance keys |

### Rules for source fields

| Rule | Details |
|---|---|
| **Must exist on the parent model** | The source field names must match actual field names on the parent |
| **Should be scalar** | Source fields are typically `str` or `str \| None` — the engine reads their values with `getattr()` |
| **Must have data** | If all source fields are `None` or empty string, the entry is skipped entirely |
| **Can be multiple** | List multiple source fields if the normalization needs context from several columns |

---

## 4. Configuration

### Via `configure()`

```python
from langchain_aws import ChatBedrockConverse
from scinr.newton import configure

# Option 1: Use the same LLM as the main pipeline
configure(
    normalization_enabled=True,
    normalization_batch_size=10,
)

# Option 2: Use a dedicated (cheaper) LLM for normalization
normalize_llm = ChatBedrockConverse(
    model="us.anthropic.claude-haiku-3",
    region_name="us-east-1",
)
configure(
    normalization_enabled=True,
    normalization_batch_size=10,
    normalization_llm=normalize_llm,
)
```

### Via environment variables

| Parameter | Env Var | Default | Description |
|---|---|---|---|
| `normalization_enabled` | `NORMALIZATION_ENABLED` | `false` | Enable/disable the normalization engine |
| `normalization_batch_size` | `NORMALIZATION_BATCH_SIZE` | `3` | Max entries per LLM batch call |
| `normalization_llm` | — | Falls back to main `llm` | Dedicated LLM instance for normalization calls |

### Parameter resolution order

For each parameter: **explicit argument** > **environment variable** > **default value**.

```bash
# Enable normalization via env var
export NORMALIZATION_ENABLED=true
export NORMALIZATION_BATCH_SIZE=10

# Run pipeline — env vars are picked up automatically
scinr-ingest --input ./data/contacts.csv --theme contacts
```

### When to use a dedicated normalization LLM

| Factor | Main LLM | Dedicated LLM |
|---|---|---|
| **Cost** | Higher per call | Lower (use a cheaper model like Haiku) |
| **Quality** | Best for complex extraction | Sufficient for straightforward normalization |
| **Latency** | Shared queue with extraction | Can run in parallel with extraction |
| **Recommendation** | Use for small datasets | Use for large tabular datasets (100+ rows) |

The normalization task is relatively simple: parse structured text into a known schema. A cheaper, faster model like Claude Haiku or a small GPT variant handles this well.

---

## 5. Mandatory vs. Optional Decision Matrix

### Decision table

| Model used with... | Add `normalization_model` keys? | Why |
|---|---|---|
| **Tabular only** (CSV/XLSX/XLS) | ✅ **Mandatory** | The `NormalizationEngine` requires `normalization_model: True` to trigger. Without it, the field stays `None`. |
| **Unstructured only** (PDF/DOCX) | ⚪ **Optional** | The LLM fills the nested field directly from the `description=` during entity extraction. No separate normalization step is needed. |
| **Both pipelines** | ✅ **Recommended** | Mandatory for the tabular half; the keys serve as a useful hint for the unstructured LLM. |

### Tabular-only model

```python
# This model is used exclusively with CSV files.
# Normalization keys are MANDATORY — without them, normalized_address is always None.

class SupplierRecord(ExtractionModel):
    """A supplier record from a CSV spreadsheet."""

    supplier_name: str = Field(..., description="Supplier name.")
    raw_address: str = Field(..., description="Free-text address.")
    raw_phone: str | None = Field(default=None, description="Phone number.")

    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],
        },
    )
```

### Unstructured-only model

```python
# This model is used exclusively with PDF/DOCX documents.
# Normalization keys are OPTIONAL — the LLM fills normalized_address directly.

class PatentRecord(ExtractionModel):
    """A patent record extracted from a PDF document."""

    patent_number: str = Field(..., description="Patent number.")
    title: str = Field(..., description="Patent title.")
    applicant_name: str = Field(..., description="Applicant name.")

    # No normalization_model needed — the LLM fills this during extraction
    applicant_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address of the applicant.",
    )
```

### Dual-pipeline model

```python
# This model is used with BOTH CSV and PDF.
# Normalization keys are RECOMMENDED — mandatory for tabular, helpful for unstructured.

class ManufacturerRecord(ExtractionModel):
    """A manufacturer record usable in both tabular and unstructured pipelines."""

    manufacturer_name: str = Field(..., description="Manufacturer name.")
    raw_address: str = Field(..., description="Free-text address from source.")
    country: str | None = Field(default=None, description="Country name or code.")

    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address derived from raw_address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],
        },
    )
```

---

## 6. The Implicit Fallback (Footgun)

### What it is

When `normalization_source_fields` is omitted or empty in the `json_schema_extra`, the `NormalizationEngine` falls back to using **ALL scalar fields** on the parent model as source data for the normalization LLM call. This is almost never what you want.

### The problem

```python
# ❌ BAD — no normalization_source_fields declared
class WideRecord(ExtractionModel):
    """A wide record with many fields."""

    raw_name: str = Field(..., description="Full name.")
    raw_address: str = Field(..., description="Free-text address.")
    raw_phone: str | None = Field(default=None, description="Phone number.")
    internal_notes: str | None = Field(default=None, description="Internal notes.")
    department: str | None = Field(default=None, description="Department name.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")

    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address.",
        json_schema_extra={
            "normalization_model": True,
            # Missing! Engine sends ALL scalar fields to LLM:
            # raw_name, raw_address, raw_phone, internal_notes, department, created_at
        },
    )
```

In this case, the normalization LLM receives **six fields** as source data for an address normalization task. Only `raw_address` is relevant. The other five fields are noise that:

- **Waste tokens** — every irrelevant field adds to the prompt size
- **Leak context** — internal notes or department names may confuse the LLM
- **Degrade accuracy** — the LLM may try to extract city from a department name
- **Break dedup** — the unique key hash includes all fields, so two rows with the same address but different departments get separate LLM calls

### How the fallback works (internally)

```python
# detector.py — extract_source_values()
if spec.source_fields:
    # Explicit: only declared fields
    for src_field in spec.source_fields:
        val = getattr(instance, src_field, None)
        if val is not None and val != "":
            values[src_field] = val
else:
    # Implicit fallback: ALL scalar fields
    for field_name, field_info in instance.model_fields.items():
        if field_name == spec.field_name:
            continue  # skip the target field itself
        ann_type = _extract_target_type(field_info.annotation)
        if ann_type is not None:
            continue  # skip nested Pydantic models
        val = getattr(instance, field_name, None)
        if val is not None and val != "":
            values[field_name] = val
```

### The fix

```python
# ✅ GOOD — explicit source fields
class NarrowRecord(ExtractionModel):
    """A record with explicit normalization source."""

    raw_name: str = Field(..., description="Full name.")
    raw_address: str = Field(..., description="Free-text address.")
    raw_phone: str | None = Field(default=None, description="Phone number.")
    internal_notes: str | None = Field(default=None, description="Internal notes.")
    department: str | None = Field(default=None, description="Department name.")

    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],  # Only what's needed
        },
    )
```

### Comparison

| Aspect | Implicit (bad) | Explicit (good) |
|---|---|---|
| **Source data** | All scalar fields on parent | Only declared fields |
| **LLM prompt size** | Larger, noisy | Minimal, focused |
| **LLM accuracy** | Lower (distracted by irrelevant fields) | Higher (focused on relevant data) |
| **Dedup hash** | Includes irrelevant fields (fewer cache hits) | Based on relevant data only (more cache hits) |
| **Token cost** | Higher | Lower |
| **Predictability** | Changes if parent model gains fields | Stable and explicit |

**Always declare `normalization_source_fields` explicitly.** The implicit fallback exists only for backward compatibility and should be treated as a bug if encountered in new code.

---

## 7. NormalizationEngine Architecture

### Class overview

```python
class NormalizationEngine:
    def __init__(
        self,
        llm: BaseLanguageModel,
        batch_size: int = 5,
        concurrency: int = 5,  # kept for API compat, not used
    ) -> None:

    async def normalize_instances(
        self,
        instances: list[tuple[type[BaseModel], BaseModel]],
    ) -> list[tuple[type[BaseModel], BaseModel]]:

    async def process_key_batch(
        self,
        entries: list[NormalizationEntry],
        retry_count: int = 0,
    ) -> dict[str, BaseModel]:

    def apply_cached_to_instance(
        self,
        instance: BaseModel,
        field_name: str,
        unique_key: str,
    ) -> bool:
```

### `__init__`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | `BaseLanguageModel` | — | LangChain LLM for normalization calls |
| `batch_size` | `int` | `5` | Maximum entries per LLM batch call |
| `concurrency` | `int` | `5` | **Deprecated** — kept for API compatibility. Real concurrency is governed by `config.get_llm_semaphore()` |

The `concurrency` parameter is no longer used to create a local semaphore. All LLM calls (extraction, entity extraction, annotation, normalization) share a single global semaphore from `config.get_llm_semaphore()` to avoid exceeding the Bedrock botocore connection pool.

### `normalize_instances()`

The main entry point. Takes a list of `(model_class, instance)` tuples and returns the same list with normalization fields populated.

**Internal phases:**

1. **Collect entries** — scan each instance's model class for normalization specs, extract source values, build unique keys
2. **Build key-to-targets map** — maps each unique key to the list of `(instance_id, field_name)` pairs that need the result
3. **Group by target type** — separate entries by `target_type.__name__` (LLM needs homogeneous batches)
4. **Deduplicate** — skip entries with duplicate unique keys (same source values from different rows)
5. **Batch and dispatch** — split each type group into batches of `batch_size`, create async tasks with `get_llm_semaphore()`
6. **Await all tasks** — `asyncio.gather(*tasks)` runs all batches concurrently (bounded by global semaphore)

### `process_key_batch()`

Public method for processing a batch of unique normalization keys. Used by the normalization-first write path in `neo4j_ops.py`.

Returns `{unique_key: normalized_result}` for successfully processed keys. Results are cached in `self.result_cache` for reuse.

**Internal flow:**

1. Validate all entries share the same `target_type`
2. Build dynamic `BatchOutput` and `BatchResponse` schemas
3. Create structured-output LLM: `self.llm.with_structured_output(BatchResponse)`
4. Build prompt messages via `_build_batch_messages()`
5. Call LLM with retry: `await with_llm_retry(lambda: structured_llm.ainvoke(messages))`
6. Coerce result to `BatchResponse` (handles dict returns)
7. Cache results in `self.result_cache`
8. Retry missing keys once (if `retry_count < 1`)

### `apply_cached_to_instance()`

Applies a cached normalization result to a specific instance field. Returns `True` if the key was found and applied, `False` otherwise.

Uses `setattr()` with `object.__setattr__()` fallback to handle models with `validate_assignment=True`.

### `_build_batch_messages()`

Constructs the prompt for a batch of entries:

```python
# System message
"You are a data normalization assistant. You receive raw extracted data "
"and must normalize it into a structured format. Fill in all fields you can "
"confidently identify from the source data. Leave uncertain fields as null."

# Human message (simplified)
"Normalize the following extracted data entries into structured format.

For each entry, return a result with:
- key: the exact unique key from the entry (must match exactly)
- result: the normalized structured output

Entries:
--- Entry NormalizedAddress:abc123 ---
Source data:
  raw_address: 123 Main St, Springfield, IL 62704

--- Entry NormalizedAddress:def456 ---
Source data:
  raw_address: 456 Oak Ave, Portland, OR 97201

Return a list of results, one per entry."
```

### `_hash_source_values()`

Generates a deterministic MD5 hash from source values:

```python
@staticmethod
def _hash_source_values(source_values: dict[str, Any]) -> str:
    normalized = str(sorted(source_values.items())).lower()
    return hashlib.md5(normalized.encode()).hexdigest()
```

Sorting ensures the hash is independent of dict ordering. Lowercasing ensures case-insensitive dedup (though the values themselves are not lowercased — only the hash input).

---

## 8. Performance Considerations

### Batch size tuning

| Batch size | LLM calls (100 unique entries) | Tokens per call | Total latency |
|---|---|---|---|
| 1 | 100 | Low | High (100 sequential calls) |
| 5 (default) | 20 | Moderate | Balanced |
| 10 | 10 | Higher | Lower (fewer calls) |
| 20 | 5 | High | Lowest (but risk of context overflow) |

**Guidance:**
- **Small datasets** (< 50 rows): batch size 5-10 is fine
- **Medium datasets** (50-500 rows): batch size 10-15
- **Large datasets** (500+ rows): batch size 15-20, monitor for context overflow
- **Very wide source fields** (many source fields per entry): keep batch size lower (5-10) to avoid context overflow

### Caching (deduplication)

The engine caches results by unique key (`{target_type}:{md5_hash}`). This means:

- **Identical source values** across rows trigger only one LLM call
- **Different source values** for the same target type still batch together
- **Cache is per-engine-instance** — not persisted across pipeline runs

Example: a CSV with 1000 rows where 200 have the same address value:

```
Without caching: 1000 LLM calls (worst case)
With caching:    801 unique calls (200 rows share one result)
With batching (size 10): ~81 LLM calls
```

### Dedicated LLM

Using a separate, cheaper LLM for normalization:

```python
configure(
    llm=ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-20250514"),  # Main LLM
    normalization_llm=ChatBedrockConverse(model="us.anthropic.claude-haiku-3"),  # Normalization LLM
)
```

Benefits:
- **Cost reduction** — normalization is a simpler task than full extraction
- **Parallel execution** — normalization LLM calls share the global semaphore but use a separate model endpoint
- **Quality isolation** — a normalization failure doesn't block the main extraction pipeline

### Concurrency

Normalization LLM calls share the global `llm_concurrency` semaphore (default: 4). This means:

- Maximum 4 concurrent LLM calls across **all** pipeline stages (extraction, entity extraction, annotation, normalization)
- The semaphore is acquired per batch, not per entry
- Increasing `llm_concurrency` allows more parallel normalization calls but may exceed Bedrock rate limits

```python
configure(
    llm_concurrency=8,           # More parallel LLM calls
    normalization_batch_size=10,  # Larger batches
)
```

### Pre-scan dedup map (neo4j_ops.py)

The normalization-first write path in `neo4j_ops.py` performs a **pre-scan** of all rows before any LLM calls:

1. Scans all rows to build a global dedup map of unique normalization keys
2. Groups unique keys by target type
3. Dispatches all key batches concurrently via `asyncio.gather()`
4. Instantiates composites with cached normalization results
5. Writes to Neo4j in row batches

This approach ensures:
- **Single-pass dedup** — all rows scanned once before any LLM call
- **Maximal batching** — all unique keys of the same type batch together
- **Concurrent LLM calls** — all type batches run in parallel
- **Atomic writes** — rows are written only after their normalization results are available

---

## 9. Multiple Normalization Fields

A single model can declare multiple normalization target fields, each with its own target type and source fields. The engine processes each independently.

```python
class ComplexRecord(ExtractionModel):
    """Record with multiple normalization targets."""

    raw_address: str = Field(..., description="Free-text address.")
    raw_substance: str = Field(..., description="Free-text substance description.")
    raw_strength: str = Field(..., description="Free-text strength information.")
    raw_date: str | None = Field(default=None, description="Free-text date.")

    # ─── Normalization target 1: Address ───
    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],
        },
    )

    # ─── Normalization target 2: Substance ───
    normalized_substance: NormalizedSubstance | None = Field(
        default=None,
        description="Structured substance data.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_substance"],
        },
    )

    # ─── Normalization target 3: Strength ───
    normalized_strength: NormalizedStrength | None = Field(
        default=None,
        description="Structured strength data.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_strength"],
        },
    )

    # ─── Normalization target 4: Date ───
    normalized_date: NormalizedDate | None = Field(
        default=None,
        description="Structured date.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_date"],
        },
    )
```

### How the engine handles multiple fields

1. **Detection**: `get_normalization_specs()` returns a list of all normalization specs for the model class (one per `normalization_model: True` field)
2. **Entry creation**: Each spec generates its own `NormalizationEntry` with its own unique key
3. **Grouping**: Entries are grouped by `target_type.__name__` — different target types get separate LLM calls
4. **Batching**: Entries of the same target type batch together regardless of which parent field they came from
5. **Application**: Each result is applied to the correct field on the correct instance

### Performance implications

| Scenario | Effect |
|---|---|
| Multiple fields, same target type | Entries batch together — efficient |
| Multiple fields, different target types | Separate LLM calls per type — more calls |
| Multiple fields, same source fields | Each field still gets its own LLM call — consider if you really need separate models |

### Shared source fields

Multiple normalization fields can reference the same source field:

```python
class SubstanceRecord(ExtractionModel):
    raw_description: str = Field(..., description="Full substance description.")

    normalized_substance: NormalizedSubstance | None = Field(
        default=None,
        description="Substance identity.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_description"],
        },
    )

    normalized_strength: NormalizedStrength | None = Field(
        default=None,
        description="Strength and form.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_description"],  # Same source!
        },
    )
```

This triggers two separate LLM calls (different target types) with the same source data. The LLM extracts different aspects in each call. This is intentional when you want to decompose a complex field into multiple structured models.

---

## 10. Troubleshooting

### Common problems

| Problem | Cause | Fix |
|---|---|---|
| Normalized field stays `None` | `normalization_enabled=False` | Set to `True` in `configure()` or env var |
| Normalized field stays `None` | No `normalization_model: True` on the field | Add the flag to `json_schema_extra` |
| Wrong data in normalized field | Implicit fallback (no `normalization_source_fields`) | Add explicit `normalization_source_fields` |
| Wrong data in normalized field | Source field name mismatch | Verify source field names match actual model fields |
| Slow normalization | Batch size too small | Increase `normalization_batch_size` |
| Slow normalization | Too many unique entries | Check for dedup opportunities |
| LLM errors | `normalization_llm` not configured and main LLM unavailable | Pass `normalization_llm` or verify main `llm` |
| LLM errors | Context overflow (too many entries per batch) | Decrease `normalization_batch_size` |
| Missing results | LLM skipped an entry in the batch | Engine retries once; check logs for warnings |
| Type errors | Target type not a Pydantic model | Ensure annotation resolves to `BaseModel` subclass |
| Validation errors | `validate_assignment=True` rejecting normalized value | Engine uses `object.__setattr__` fallback automatically |

### Debug logging

Enable debug logging to see normalization internals:

```python
import logging
logging.getLogger("scinr.newton.tabular.normalization").setLevel(logging.DEBUG)
```

Key log messages to watch for:

```
# Entry collection
"Normalization: collected {N} entries from {M} instances"

# Dedup
"Normalization: {N} unique keys (from {M} total entries)"

# Batching
"Normalization: processing batch of {N} entries for {TargetType}"

# LLM call
"Normalization batch returned {N} results"

# Missing results retry
"Normalization: {N}/{M} results missing, retrying: {keys}"

# Batch failure
"Normalization batch failed for {TargetType} ({N} entries): {error}"

# Instance not found
"Normalization: instance id {id} not found for {field}, skipping"
```

### Verifying normalization is active

```python
from scinr.newton.config import get_config

cfg = get_config()
print(f"Normalization enabled: {cfg.normalization_enabled}")
print(f"Batch size: {cfg.normalization_batch_size}")
print(f"Dedicated LLM: {cfg.normalization_llm is not None}")
```

### Checking model specs

```python
from scinr.newton.tabular.normalization.detector import get_normalization_specs

specs = get_normalization_specs(ContactRecord)
for spec in specs:
    print(f"Field: {spec.field_name}")
    print(f"  Target: {spec.target_type.__name__}")
    print(f"  Source fields: {spec.source_fields}")
```

---

## 11. Complete Example

### Scenario

A pharmaceutical company has a CSV file of manufacturer contact information. Each row has a free-text address column that needs to be normalized into structured components (street, city, postal code, country). The company also wants to normalize the substance information from a separate column.

### Step 1: Define the models

```python
"""
models/pharma_contacts.py — Pharmaceutical manufacturer contact models.

Demonstrates the complete normalization flow:
- Dual-field pattern (raw + normalized)
- Multiple normalization targets per model
- Entity labels for cross-document deduplication
- Instance keys for model instance deduplication
"""

from __future__ import annotations

from pydantic import Field

from scinr.newton.models.base import ExtractionModel


# ─── Normalization targets ───────────────────────────────────────────────────


class NormalizedAddress(ExtractionModel):
    """Structured, normalized postal address."""

    street: str | None = Field(
        default=None,
        description="Street address line, without city or postal code.",
    )
    city: str | None = Field(
        default=None,
        description="City name.",
    )
    postal_code: str | None = Field(
        default=None,
        description="Postal or ZIP code.",
    )
    country_code: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code (e.g. 'US', 'DE', 'JP').",
        json_schema_extra={"entity_label": "Country"},
    )


class NormalizedSubstance(ExtractionModel):
    """Structured, normalized substance information."""

    inn_name: str | None = Field(
        default=None,
        description="International Nonproprietary Name (INN).",
        json_schema_extra={"entity_label": "ActiveSubstance"},
    )
    cas_number: str | None = Field(
        default=None,
        description="CAS Registry Number (e.g. '1105-50-9').",
        json_schema_extra={"entity_label": "CasNumber"},
    )
    strength: str | None = Field(
        default=None,
        description="Strength with units (e.g. '500 mg', '10 mg/mL').",
    )
    pharmaceutical_form: str | None = Field(
        default=None,
        description="Pharmaceutical form (e.g. 'tablet', 'solution', 'powder').",
        json_schema_extra={"entity_label": "PharmaceuticalForm"},
    )


# ─── Parent model (tabular record) ───────────────────────────────────────────


class ManufacturerContact(ExtractionModel):
    """A manufacturer contact record from a CSV file.

    Uses the dual-field pattern with explicit normalization source fields.
    The NormalizationEngine processes raw_address and raw_substance columns
    to populate the normalized nested models.
    """

    # ─── Raw scalar fields ────────────────────────────────────────────────
    manufacturer_name: str = Field(
        ...,
        description="Legal name of the manufacturing company.",
        json_schema_extra={"entity_label": "Manufacturer"},
    )
    raw_address: str = Field(
        ...,
        description="Free-text address as it appears in the source column.",
    )
    raw_substance: str | None = Field(
        default=None,
        description="Free-text description of the manufactured substance.",
    )
    contact_email: str | None = Field(
        default=None,
        description="Contact email address.",
    )
    raw_phone: str | None = Field(
        default=None,
        description="Contact phone number.",
    )

    # ─── Normalization targets ────────────────────────────────────────────
    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address derived from raw_address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],
        },
    )

    normalized_substance: NormalizedSubstance | None = Field(
        default=None,
        description="Structured substance data derived from raw_substance.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_substance"],
        },
    )
```

### Step 2: Configure the pipeline

```python
from langchain_aws import ChatBedrockConverse
from scinr.newton import configure

# Main LLM for extraction and annotation
main_llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-20250514",
    region_name="us-east-1",
)

# Dedicated (cheaper) LLM for normalization
normalize_llm = ChatBedrockConverse(
    model="us.anthropic.claude-haiku-3",
    region_name="us-east-1",
)

configure(
    llm=main_llm,
    normalization_enabled=True,
    normalization_batch_size=10,
    normalization_llm=normalize_llm,
    llm_concurrency=8,
)
```

### Step 3: Sample CSV input

```csv
manufacturer_name,raw_address,raw_substance,contact_email,raw_phone
PharmaCorp Inc,"123 Main Street, Springfield, IL 62704, USA","Metformin Hydrochloride 500mg tablets, CAS 1105-50-9",info@pharmacorp.com,+1-555-0100
BioMed Labs,"456 Oak Avenue, Portland, OR 97201, USA","Atorvastatin Calcium 20mg tablets, CAS 134523-03-8",contact@biomedlabs.com,+1-555-0200
EuroPharm GmbH,"Hauptstraße 7, 10115 Berlin, Germany","Ibuprofen 400mg capsules, CAS 15357-78-8",info@europharm.de,+49-30-123456
PharmaCorp Inc,"123 Main Street, Springfield, IL 62704, USA","Lisinopril 10mg tablets, CAS 83915-66-8",info@pharmacorp.com,+1-555-0100
```

Note that row 4 has the same `raw_address` as row 1 — the normalization engine will deduplicate this and use the cached result.

### Step 4: Pipeline execution

```python
import asyncio
from scinr.newton.pipeline import run_pipeline

async def main():
    result = await run_pipeline(
        input_path="./data/manufacturers.csv",
        theme="pharma_contacts",
    )
    print(f"Processed {result.total_rows} rows")
    print(f"Normalized {result.normalization_count} fields")

asyncio.run(main())
```

### Step 5: What happens internally

```
1. Column Mapping (LLM)
   CSV columns → ManufacturerContact fields

2. Model Instantiation
   4 rows → 4 ManufacturerContact instances
   normalized_address = None (all 4)
   normalized_substance = None (all 4)

3. Normalization Detection
   get_normalization_specs(ManufacturerContact) → [
       NormalizationSpec(field_name="normalized_address", target_type=NormalizedAddress, source_fields=["raw_address"]),
       NormalizationSpec(field_name="normalized_substance", target_type=NormalizedSubstance, source_fields=["raw_substance"]),
   ]

4. Entry Collection
   Row 1: NormalizedAddress entry (key: "NormalizedAddress:abc123")
   Row 2: NormalizedAddress entry (key: "NormalizedAddress:def456")
   Row 3: NormalizedAddress entry (key: "NormalizedAddress:ghi789")
   Row 4: NormalizedAddress entry (key: "NormalizedAddress:abc123") ← DUPLICATE
   Row 1: NormalizedSubstance entry (key: "NormalizedSubstance:jkl012")
   Row 2: NormalizedSubstance entry (key: "NormalizedSubstance:mno345")
   Row 3: NormalizedSubstance entry (key: "NormalizedSubstance:pqr678")
   Row 4: NormalizedSubstance entry (key: "NormalizedSubstance:stu901")

5. Deduplication
   NormalizedAddress: 3 unique keys (row 4 shares with row 1)
   NormalizedSubstance: 4 unique keys

6. Batching (batch_size=10)
   NormalizedAddress batch: [abc123, def456, ghi789] → 1 LLM call
   NormalizedSubstance batch: [jkl012, mno345, pqr678, stu901] → 1 LLM call

7. LLM Calls (concurrent, bounded by llm_concurrency=8)
   Call 1: NormalizedAddress batch → 3 results cached
   Call 2: NormalizedSubstance batch → 4 results cached

8. Result Application
   Row 1: normalized_address ← cached[abc123], normalized_substance ← cached[jkl012]
   Row 2: normalized_address ← cached[def456], normalized_substance ← cached[mno345]
   Row 3: normalized_address ← cached[ghi789], normalized_substance ← cached[pqr678]
   Row 4: normalized_address ← cached[abc123], normalized_substance ← cached[stu901]

9. Neo4j Write
   4 ManufacturerContact :ModelInstance nodes
   4 NormalizedAddress :ModelInstance nodes (3 unique, row 4 shares with row 1)
   4 NormalizedSubstance :ModelInstance nodes
   Country :LabeledEntity nodes (deduplicated by entity_label)
   ActiveSubstance :LabeledEntity nodes (deduplicated by entity_label)
```

### Step 6: Neo4j graph result

```
(:StructureNode {name: "manufacturers.csv"})
  -[:HAS_EXTRACTION]->
  (:ExtractionResult)
    -[:HAS_CONDITION_IDS]->
    (:ModelInstance {model_class: "ManufacturerContact"})
      -[:HAS_PROPERTIES {manufacturer_name: "PharmaCorp Inc", ...}]
      -[:REFERENCES]->(:LabeledEntity {label: "Manufacturer", value: "PharmaCorp Inc"})
      -[:HAS_NORMALIZED_ADDRESS]->
      (:ModelInstance {model_class: "NormalizedAddress"})
        -[:HAS_PROPERTIES {street: "123 Main Street", city: "Springfield", ...}]
        -[:REFERENCES]->(:LabeledEntity {label: "Country", value: "US"})
      -[:HAS_NORMALIZED_SUBSTANCE]->
      (:ModelInstance {model_class: "NormalizedSubstance"})
        -[:HAS_PROPERTIES {inn_name: "Metformin", strength: "500 mg", ...}]
        -[:REFERENCES]->(:LabeledEntity {label: "ActiveSubstance", value: "Metformin"})
        -[:REFERENCES]->(:LabeledEntity {label: "CasNumber", value: "1105-50-9"})
```

Row 4 (PharmaCorp Inc, same address) shares the same `NormalizedAddress` node as row 1 via the `country_code` entity_label dedup. The `NormalizedSubstance` nodes are different because the substances differ.

---

## Next steps

- **[Advanced Model Design Patterns](model-patterns.md)** — The dual-field pattern, entity labels, instance keys, and more.
- **[Custom Models](custom-models.md)** — Basic model definition and field descriptions.
- **[Tabular Pipeline](tabular-pipeline.md)** — How the tabular pipeline processes CSV/XLSX files.
- **[Neo4j Graph Storage](neo4j-graph.md#normalized-models-tabular-pipeline)** — See how a normalized sub-model is written to the graph as a real `:ModelInstance` node, and why normalization by itself doesn't create cross-dataset links.
- **[Architecture](../architecture.md)** — Detailed pipeline stage walkthrough.
