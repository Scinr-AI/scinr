# Advanced Model Design Patterns

Production-grade extraction models in `scinr.newton` use a set of interconnected design patterns to produce clean, deduplicated knowledge graphs. This guide covers each pattern in detail, when to use it, and how the patterns compose together.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Dual-Field Pattern](#2-the-dual-field-pattern)
3. [Entity Labels (`entity_label`)](#3-entity-labels-entity_label)
4. [Instance Keys (`instance_key`)](#4-instance-keys-instance_key)
5. [Domain Validators](#5-domain-validators)
6. [OCR Fix Validators](#6-ocr-fix-validators)
7. [The Normalization Model Mechanism](#7-the-normalization-model-mechanism)
8. [The Implicit Fallback (Footgun)](#8-the-implicit-fallback-footgun)
9. [Instance Relationships](#9-instance-relationships)
10. [Field Relationships](#10-field-relationships)
11. [Pattern Summary Table](#11-pattern-summary-table)
12. [Complete Example](#12-complete-example)

---

## 1. Introduction

Advanced patterns for robust extraction models that produce clean, deduplicated knowledge graphs.

A well-designed extraction model does more than define fields and descriptions. It declares *how* those fields interact with the pipeline: which values should be globally deduplicated, which fields form a unique identity, which raw fields feed a normalization step, and which values need domain-specific cleaning before they reach Neo4j.

These patterns are expressed entirely through Pydantic `Field()` annotations and `json_schema_extra` metadata. The pipeline reads this metadata at runtime to wire up entity labeling, instance deduplication, normalization batching, and graph relationships — all without any changes to pipeline code.

**The patterns covered in this guide:**

| Pattern | Declared Via | Pipeline Effect |
|---|---|---|
| Dual-Field | Two fields (raw + nested) | Raw preserves source; nested enables structured matching |
| `entity_label` | `json_schema_extra` | Creates `:LabeledEntity` nodes for cross-document deduplication |
| `instance_key` | `json_schema_extra` | Deterministic UID for `:ModelInstance` deduplication |
| `normalization_model` | `json_schema_extra` | Triggers `NormalizationEngine` in tabular pipeline |
| `normalization_source_fields` | `json_schema_extra` | Declares which raw fields feed the normalization |
| `instance_relationships` | `json_schema_extra` | Cross-instance graph relationships (Level 3) |
| `field_relationships` | `json_schema_extra` | Cross-entity graph relationships (Level 2) |
| Domain validators | `@field_validator` | Pre-validation normalization and OCR correction |

---

## 2. The Dual-Field Pattern

The core pattern: a **raw free-text field** paired with a **normalized nested model**.

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
    """A single contact record from a regulatory document."""

    # ─── Tier 1: Free-text raw field (high tolerance) ───
    raw_address: str = Field(
        ...,
        description=(
            "Free-text address exactly as it appears in the source document. "
            "Preserve original formatting, line breaks, and abbreviations."
        ),
    )

    # ─── Tier 2: Normalized nested model (structured) ───
    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description=(
            "Structured address derived from raw_address: street, city, "
            "postal code, and country code parsed into separate fields."
        ),
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],
        },
    )
```

### Why dual-field?

| Aspect | Raw Field | Normalized Field |
|---|---|---|
| **Purpose** | Preserves original text verbatim | Enables structured querying and matching |
| **LLM tolerance** | High — free-text, no structure required | Lower — must conform to nested schema |
| **Neo4j representation** | Scalar property on parent node | Separate `:ModelInstance` child node |
| **Cross-document matching** | None (unstructured) | Via `entity_label` fields within |

The raw field captures the source text with maximum fidelity. The normalized field breaks that text into queryable, comparable components. Together they give you both provenance and utility.

### How the pipeline uses it

- **`normalization_model: True`** — marks the field for the `NormalizationEngine` in the tabular pipeline. The engine scans for this flag and schedules the field for LLM-driven normalization.
- **`normalization_source_fields`** — declares which raw fields on the *parent* model feed the normalization. The engine collects values from these fields, builds a dedup hash, and batches them for a single LLM call.

### When mandatory vs. optional

| Pipeline | Requirement | Reason |
|---|---|---|
| **Tabular** (CSV/XLSX/XLS) | **Mandatory** | The `NormalizationEngine` hook requires `normalization_model: True` and `normalization_source_fields` to know which raw columns to use as input. Without them, the normalization step is a no-op. |
| **Unstructured** (PDF/DOCX) | **Optional** | The LLM fills the nested field directly from the field description in a single extraction call. No separate normalization step is needed. |
| **Both pipelines** | **Recommended** | Keeps the model consistent across pipeline types. A model that works in both pipelines should always declare both keys. |

---

## 3. Entity Labels (`entity_label`)

Entity labels create `:LabeledEntity` nodes in Neo4j that are globally deduplicated by `(label, normalized_value)`. This is the primary mechanism for cross-document entity matching.

### Good usage

```python
# ✅ GOOD — stable, identifier-like value
substance_name: str | None = Field(
    default=None,
    description="INN name of the active substance (e.g. 'Metformin').",
    json_schema_extra={"entity_label": "ActiveSubstance"},
)

# ✅ GOOD — code or standard identifier
country_code: str | None = Field(
    default=None,
    description="ISO 3166-1 alpha-2 country code (e.g. 'US', 'DE', 'JP').",
    json_schema_extra={"entity_label": "Country"},
)

# ✅ GOOD — procedure type with stable set of values
procedure_type: str | None = Field(
    default=None,
    description="Procedure type code: IA, IB, II, IAIN, A, or BA.",
    json_schema_extra={"entity_label": "ProcedureType"},
)
```

### Bad usage

```python
# ❌ BAD — free-text narrative (every value is unique, no dedup benefit)
full_description: str | None = Field(
    default=None,
    description="Detailed paragraph describing the manufacturing process.",
    json_schema_extra={"entity_label": "ProcessDescription"},  # WRONG
)

# ❌ BAD — measurement in context (dedup is meaningless)
temperature: float | None = Field(
    default=None,
    description="Processing temperature in degrees Celsius.",
    json_schema_extra={"entity_label": "Temperature"},  # WRONG
)

# ❌ BAD — boolean (only two values, no dedup value)
is_active: bool | None = Field(
    default=None,
    description="Whether the product is currently on the market.",
    json_schema_extra={"entity_label": "IsActive"},  # WRONG
)
```

### Rules

| Rule | Details |
|---|---|
| **Use for** | Codes, names, identifiers, stable categorical values |
| **Don't use for** | Long descriptions, narratives, free-text paragraphs |
| **Don't use for** | Measurements in context, timestamps, booleans |
| **Label naming** | Use `CamelCase` (e.g. `ActiveSubstance`, `Country`, `ProcedureType`) |
| **Neo4j effect** | Creates `:LabeledEntity {label, value, normalized_value}` nodes merged by `(label, normalized_value)` |

### How deduplication works

When the graph mapper encounters a field with `entity_label`, it:

1. Normalizes the value: lowercase, strips accents, collapses whitespace
2. Computes a deterministic UID from the label and normalized value
3. MERGEs a `:LabeledEntity` node with that UID

This means the same substance name extracted from ten different documents resolves to a single `:LabeledEntity` node. The `REFERENCES` relationship from each extraction points to that shared node.

---

## 4. Instance Keys (`instance_key`)

Instance keys define a composite unique identifier for a `:ModelInstance` node. When a nested model has one or more fields marked with `instance_key: True`, the graph mapper computes a deterministic UID and uses MERGE instead of CREATE, enabling global deduplication of model instances.

### Basic usage

```python
class Fee(ExtractionModel):
    """A single fee entry from a fee schedule table."""

    country_code: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code.",
        json_schema_extra={"entity_label": "Country", "instance_key": True},
    )
    procedure_type: str = Field(
        ...,
        description="Procedure type code: IA, IB, II, IAIN, A, or BA.",
        json_schema_extra={"entity_label": "ProcedureType", "instance_key": True},
    )
    role: str = Field(
        ...,
        description="Fee role: applicant, holder, or third party.",
        json_schema_extra={"entity_label": "FeeRole", "instance_key": True},
    )
    rate: str = Field(
        ...,
        description="Fee amount without currency symbol (e.g. '1234.56').",
    )
```

Here, `(country_code, procedure_type, role)` forms the composite key. A `Fee` for `("DE", "IA", "applicant")` extracted from any document will always resolve to the same `:ModelInstance` node.

### Rules

| Rule | Details |
|---|---|
| **Mark ALL key fields** | Every field that forms the unique identity must have `instance_key: True` |
| **Composite keys** | Mark ALL constituent fields; the UID is computed from all of them |
| **Required for `instance_relationships`** | A model referenced in another model's `instance_relationships` must declare `instance_key` fields |
| **UID stability** | The UID is `make_instance_uid(model_class, sorted_key_fields)` — field order does not matter |

### Without instance key

A nested model without instance keys gets a random UUID (`uuid.uuid4().hex[:16]`) and is always created as a new node. This is fine for truly unique entities (e.g., a specific manufacturing batch) but prevents deduplication across extractions.

### With instance key

A nested model with instance keys gets a deterministic UID and is MERGE'd. If the same instance is extracted from multiple documents, the MERGE ensures a single node with accumulated properties.

---

## 5. Domain Validators

Domain validators apply business-specific normalization to field values *before* Pydantic validation. They live in a shared base class and use `check_fields=False` so they are optional per subclass.

### Pattern

```python
import re
from pydantic import BaseModel, field_validator, Field


def normalize_code(v: str) -> str:
    """Apply domain-specific normalization to a code string."""
    v = v.strip().upper()
    v = re.sub(r"[\s_/-]", "", v)
    return v


class NormalizedBaseModel(BaseModel):
    """Base class applying field normalization before Pydantic validation."""

    @field_validator(
        "procedure_type", "procedure_types_referenced",
        mode="before", check_fields=False
    )
    @classmethod
    def normalize_procedure_types(cls, v):
        if isinstance(v, str):
            return normalize_code(v)
        if isinstance(v, list):
            return [
                normalize_code(item) if isinstance(item, str) else item
                for item in v
            ]
        return v


class VariationModel(NormalizedBaseModel):
    """A variation entry with auto-normalized procedure type."""

    procedure_type: str = Field(
        ...,
        description="Procedure type code: IA, IB, II, IAIN, A, or BA.",
        json_schema_extra={"entity_label": "ProcedureType"},
    )
    procedure_types_referenced: list[str] | None = Field(
        default=None,
        description="Other procedure type codes referenced in this variation.",
        json_schema_extra={"entity_label": "ProcedureType"},
    )
```

### Key details

| Detail | Explanation |
|---|---|
| **`check_fields=False`** | The validator is registered on the base class but only fires for subclasses that actually declare the named fields. This makes the validator optional per subclass. |
| **`mode="before"`** | Applies *before* Pydantic type validation, so normalization happens on the raw LLM output before any type coercion. |
| **List handling** | Always check `isinstance(v, list)` and map over items. The LLM may return a single string or a list depending on the field type. |
| **Inheritance** | Subclasses of `NormalizedBaseModel` automatically get the validators for any fields they declare that match the validator's field names. |

### Common normalization functions

```python
def normalize_country_code(v: str) -> str:
    """Normalize ISO 3166-1 alpha-2 country code."""
    return v.strip().upper()[:2]


def normalize_substance_name(v: str) -> str:
    """Normalize substance name to INN-style format."""
    v = v.strip().lower()
    # Remove common suffixes
    for suffix in (" hydrochloride", " sulfate", " phosphate", " tablet", " capsule"):
        v = v.replace(suffix, "")
    return v.strip()


def normalize_date(v: str) -> str:
    """Normalize date string to YYYY-MM-DD if possible."""
    v = v.strip()
    # Attempt common formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            from datetime import datetime
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return v  # Return as-is if no format matched
```

---

## 6. OCR Fix Validators

OCR errors in scanned PDFs produce systematic misrecognitions. OCR fix validators correct these at the validation layer so the rest of the pipeline always sees clean data.

### Pattern

```python
import re
from pydantic import BaseModel, field_validator, Field


def normalize_variation_code(v: str) -> str:
    """Fix common OCR mis-recognition in variation codes."""
    # OCR fix: Q.1.a.1 → Q.I.a.1 (digit 1 → roman numeral I)
    v = re.sub(r"(?<=[A-Za-z])\.(?:1|l)\.", ".I.", v)
    # OCR fix: Q.I.a.1.a → Q.I.a.1(a) (trailing letter in parentheses)
    v = re.sub(r"(?<=\d)\.([a-zA-Z])$", r"(\1)", v)
    return v


class VariationCodeModel(BaseModel):
    """A variation code entry with OCR-corrected code."""

    variation_code: str = Field(
        ...,
        description="Variation code (e.g. 'Q.I.a.1', 'II.A.1(a)').",
        json_schema_extra={"entity_label": "VariationCode"},
    )

    @field_validator("variation_code", mode="before")
    @classmethod
    def fix_ocr_errors(cls, v):
        if isinstance(v, str):
            return normalize_variation_code(v)
        return v
```

### Common OCR patterns to fix

| Pattern | Fix | Regex |
|---|---|---|
| `O` → `0` (letter O to zero) | Context-dependent | N/A (requires domain logic) |
| `l` → `1` (lowercase L to one) | In code positions | `r"(?<=[A-Za-z])\.l\."` → `.1.` |
| `1` → `I` (one to roman I) | In variation codes | `r"(?<=[A-Za-z])\.1\."` → `.I.` |
| Missing parentheses | `Q.I.a.1.a` → `Q.I.a.1(a)` | `r"(?<=\d)\.([a-zA-Z])$"` → `r"(\1)"` |
| Hyphen vs. en-dash | `–` → `-` | `re.sub(r"[\u2013\u2014]", "-", v)` |
| Extra spaces | Multiple spaces → single | `re.sub(r"\s+", " ", v)` |

---

## 7. The Normalization Model Mechanism

Deep dive into how `normalization_model` and `normalization_source_fields` work together in the tabular pipeline.

### The mechanical flow

```
Tabular file (CSV/XLSX)
        │
        ▼
  ┌─────────────┐
  │ Column Map   │  LLM maps columns → model fields
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Instantiate  │  Pydantic model created from mapped columns
  │ Model        │  Raw fields populated, nested fields = None
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ NormalizationEngine scans for normalization_model: True │
  └──────┬──────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Collects source field values from source_fields list    │
  └──────┬──────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Dedup hash: MD5 of sorted source values                 │
  │ → Identical source values share one LLM call            │
  └──────┬──────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Batch entries by target_type (max batch_size per call)  │
  └──────┬──────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ LLM call with structured output → normalized objects    │
  └──────┬──────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Results written back via setattr (or object.__setattr__) │
  │ to the original model instances                          │
  └──────┬──────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │ Write to     │  Neo4j with populated nested fields
  │ Neo4j        │
  └─────────────┘
```

### When mandatory

**Tabular pipeline (CSV/XLSX/XLS)** — the `NormalizationEngine` hook requires these keys. Without `normalization_model: True`, the engine skips the field entirely. Without `normalization_source_fields`, the engine falls back to *all* scalar fields (the implicit fallback — see Section 8).

### When optional

**Unstructured pipeline (PDF/DOCX)** — the LLM fills the nested field directly from the field description during entity extraction. No separate normalization step is needed. The `normalization_model` flag is ignored in this pipeline.

### When recommended

**Models used with both pipelines** — always declare both `normalization_model: True` and `normalization_source_fields`. This keeps the model consistent and prevents the implicit fallback in the tabular pipeline.

### Configuration

| Parameter | Env Var | Default | Description |
|---|---|---|---|
| `normalization_enabled` | `NORMALIZATION_ENABLED` | `False` | Enable/disable the normalization engine |
| `normalization_batch_size` | `NORMALIZATION_BATCH_SIZE` | `3` | Max entries per LLM batch call |
| `normalization_llm` | — | Falls back to main `llm` | Dedicated LLM for normalization calls |

---

## 8. The Implicit Fallback (Footgun)

When `normalization_source_fields` is omitted or empty, the `NormalizationEngine` falls back to using **ALL scalar fields** on the model as source data. This is almost never what you want.

### The problem

```python
# ❌ BAD — no normalization_source_fields declared
class BadContact(ExtractionModel):
    """Contact record with implicit source fallback."""

    raw_address: str = Field(...)
    phone: str = Field(...)
    email: str = Field(...)

    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address.",
        json_schema_extra={
            "normalization_model": True,
            # Missing! Falls back to ALL scalar fields:
            # raw_address, phone, email — all sent to the LLM
        },
    )
```

In this case, the normalization LLM receives `raw_address`, `phone`, AND `email` as source data for address normalization. The phone and email fields are noise — they dilute the prompt and may confuse the LLM.

### The fix

```python
# ✅ GOOD — explicit source fields
class GoodContact(ExtractionModel):
    """Contact record with explicit source fields."""

    raw_address: str = Field(...)
    phone: str = Field(...)
    email: str = Field(...)

    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],  # exact fields
        },
    )
```

### Why it matters

| Aspect | Implicit (bad) | Explicit (good) |
|---|---|---|
| **Source data** | All scalar fields | Only declared fields |
| **LLM prompt size** | Larger, noisy | Minimal, focused |
| **LLM accuracy** | Lower (distracted) | Higher (focused) |
| **Dedup hash** | Includes irrelevant fields | Based on relevant data only |
| **Cost** | Higher (more tokens) | Lower |

**Always declare `normalization_source_fields` explicitly.** The implicit fallback exists only for backward compatibility and should be treated as a bug if encountered in new code.

---

## 9. Instance Relationships

Instance relationships (Level 3 in the graph mapper) create typed relationships between `:ModelInstance` nodes. They enable forward references across `StructureNode` boundaries: a model can reference another model instance that has not yet been extracted.

### Pattern

```python
class VariationModel(ExtractionModel):
    """A variation entry that references conditions."""

    variation_code: str = Field(
        ...,
        description="Variation code (e.g. 'Q.I.a.1').",
        json_schema_extra={
            "entity_label": "VariationCode",
            "instance_key": True,
        },
    )
    procedure_type: str = Field(
        ...,
        description="Procedure type code.",
        json_schema_extra={
            "entity_label": "ProcedureType",
            "instance_key": True,
        },
    )
    condition_ids: list[str] | None = Field(
        default=None,
        description="IDs of applicable conditions (e.g. ['1', '2', '3']).",
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "ConditionModel",
                    "rel_type": "APPLIES_TO",
                    "join_via": {
                        "condition_ids": "condition_id",
                        "variation_code": "variation_code",
                    },
                }
            ],
        },
    )
```

### How it works

1. The `condition_ids` field is a `list[str]` — each item triggers a relationship
2. `join_via` maps local fields to remote fields on the target model
3. The local field itself (`condition_ids`) is the **fan-out** field — it provides the list of values
4. Other fields in `join_via` (`variation_code`) are **fixed** fields — they come from the same instance
5. For each item in `condition_ids`, a `:ModelInstance` shell for `ConditionModel` is MERGE'd with the composite key `(condition_id, variation_code)`
6. A typed relationship `[:APPLIES_TO]` is created from the source to the target

### Rules

| Rule | Details |
|---|---|
| **Target model must have `instance_key`** | The target model class must declare `instance_key: True` on all fields referenced in `join_via` |
| **Fan-out field** | The annotated field itself (the one with `instance_relationships`) must be in `join_via` as the fan-out key |
| **Fixed fields** | Other fields in `join_via` are read from the same instance and must be non-empty |
| **Empty fixed fields** | If a fixed field is `None` or empty string, no relationships are created for this instance (logged as warning) |

### Graph result

```
(:ModelInstance {model_class: "VariationModel"})
  -[:APPLIES_TO]->
(:ModelInstance {model_class: "ConditionModel", condition_id: "1", variation_code: "Q.I.a.1"})
  -[:APPLIES_TO]->
(:ModelInstance {model_class: "ConditionModel", condition_id: "2", variation_code: "Q.I.a.1"})
```

The target `ConditionModel` nodes are "shells" — they exist with only the key fields populated. When `ConditionModel` is later extracted in a child section, the MERGE on the same UID populates the remaining fields (`description`, etc.).

---

## 10. Field Relationships

Field relationships (Level 2 in the graph mapper) create typed relationships between `:LabeledEntity` nodes. They express domain relationships between entities within the same extraction.

### Pattern

```python
class IngredientRelationship(ExtractionModel):
    """Relationship between an active substance and its excipient."""

    active_substance: str = Field(
        ...,
        description="INN name of the active substance.",
        json_schema_extra={
            "entity_label": "ActiveSubstance",
            "field_relationships": [
                {
                    "to_field": "excipient",
                    "rel_type": "CONTAINS_EXCIPIENT",
                }
            ],
        },
    )
    excipient: str = Field(
        ...,
        description="Name of the excipient.",
        json_schema_extra={"entity_label": "Excipient"},
    )
```

### How it works

1. Both `active_substance` and `excipient` have `entity_label` — they become `:LabeledEntity` nodes
2. `field_relationships` on `active_substance` declares a relationship to the `excipient` field
3. The graph mapper MERGEs a `[:CONTAINS_EXCIPIENT]` relationship between the two entity nodes

### Rules

| Rule | Details |
|---|---|
| **Both fields need `entity_label`** | Source and target fields must both have `entity_label` to create entity nodes |
| **Sibling fields** | `to_field` references a field at the same nesting level (or within the same model instance) |
| **Target must exist** | If the target field is `None`, the relationship is skipped (logged as debug) |

### Graph result

```
(:LabeledEntity {label: "ActiveSubstance", value: "Metformin"})
  -[:CONTAINS_EXCIPIENT]->
(:LabeledEntity {label: "Excipient", value: "Microcrystalline Cellulose"})
```

---

## 11. Pattern Summary Table

| Pattern | Declared Via | Purpose | When to Use |
|---|---|---|---|
| **Dual-Field** | Two fields (raw + nested) | Raw preserves source; nested enables structured querying | When you need both original text and structured data |
| **`entity_label`** | `json_schema_extra` | Creates `:LabeledEntity` nodes for cross-document deduplication | Stable, identifier-like fields (codes, names, IDs) |
| **`instance_key`** | `json_schema_extra` | Deterministic UID for `:ModelInstance` deduplication | Fields forming the unique identity of a model instance |
| **`normalization_model`** | `json_schema_extra` | Triggers `NormalizationEngine` in tabular pipeline | CSV/XLSX pipeline with nested models needing LLM normalization |
| **`normalization_source_fields`** | `json_schema_extra` | Declares which raw fields feed normalization | Always with `normalization_model: True` — never omit |
| **`instance_relationships`** | `json_schema_extra` | Cross-instance graph relationships (Level 3) | When one model references instances of another model |
| **`field_relationships`** | `json_schema_extra` | Cross-entity graph relationships (Level 2) | When two entity-labeled fields have a domain relationship |
| **Domain validators** | `@field_validator` | Pre-validation normalization and cleaning | Shared normalization across models via base class |
| **OCR fix validators** | `@field_validator` | Corrects systematic OCR misrecognitions | Models processing data from scanned PDFs |

---

## 12. Complete Example

A complete model file using all patterns together. This example models pharmaceutical variation data — a domain with codes, nested structures, cross-references, and OCR-prone scanned source documents.

```python
"""
models/variations.py — Pharmaceutical variation extraction models.

Demonstrates all advanced model design patterns:
- Dual-field pattern (raw + normalized)
- Entity labels for cross-document deduplication
- Instance keys for model instance deduplication
- Domain validators for code normalization
- OCR fix validators for scanned document correction
- Normalization model hooks for tabular pipeline
- Instance relationships for cross-model references
- Field relationships for entity-to-entity links
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from scinr.newton.models.base import ExtractionModel


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────


def normalize_code(v: str) -> str:
    """Apply domain-specific normalization to a code string."""
    v = v.strip().upper()
    v = re.sub(r"[\s_/-]", "", v)
    return v


def normalize_variation_code(v: str) -> str:
    """Fix common OCR mis-recognition in variation codes."""
    # OCR fix: Q.1.a.1 → Q.I.a.1 (digit 1 → roman numeral I in code position)
    v = re.sub(r"(?<=[A-Za-z])\.(?:1|l)\.", ".I.", v)
    # OCR fix: Q.I.a.1.a → Q.I.a.1(a) (trailing letter in parentheses)
    v = re.sub(r"(?<=\d)\.([a-zA-Z])$", r"(\1)", v)
    return v


def normalize_country_code(v: str) -> str:
    """Normalize ISO 3166-1 alpha-2 country code."""
    return v.strip().upper()[:2]


# ─────────────────────────────────────────────────────────────────────────────
# Base class with shared validators
# ─────────────────────────────────────────────────────────────────────────────


class VariationBaseModel(BaseModel):
    """Base class applying field normalization before Pydantic validation."""

    @field_validator(
        "procedure_type",
        "procedure_types_referenced",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def normalize_procedure_types(cls, v: Any) -> Any:
        """Normalize procedure type codes: strip whitespace, uppercase, remove separators."""
        if isinstance(v, str):
            return normalize_code(v)
        if isinstance(v, list):
            return [
                normalize_code(item) if isinstance(item, str) else item
                for item in v
            ]
        return v

    @field_validator(
        "variation_code",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def fix_variation_code_ocr(cls, v: Any) -> Any:
        """Fix OCR errors in variation codes."""
        if isinstance(v, str):
            return normalize_variation_code(v)
        return v

    @field_validator(
        "country_code",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def fix_country_code(cls, v: Any) -> Any:
        """Normalize country codes to ISO 3166-1 alpha-2."""
        if isinstance(v, str):
            return normalize_country_code(v)
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Normalized nested models (Tier 2)
# ─────────────────────────────────────────────────────────────────────────────


class NormalizedDate(ExtractionModel):
    """Structured date normalized to ISO 8601."""

    year: int | None = Field(default=None, description="Year (e.g. 2024).")
    month: int | None = Field(default=None, description="Month (1-12).")
    day: int | None = Field(default=None, description="Day (1-31).")
    iso_string: str | None = Field(
        default=None,
        description="Full ISO 8601 date string (e.g. '2024-03-15').",
    )


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


# ─────────────────────────────────────────────────────────────────────────────
# Condition model (referenced by VariationModel via instance_relationships)
# ─────────────────────────────────────────────────────────────────────────────


class ConditionModel(VariationBaseModel, ExtractionModel):
    """A regulatory condition applicable to a variation.

    Has instance_key so that VariationModel can reference this model via
    instance_relationships and the graph mapper can MERGE the same node
    across multiple extractions.
    """

    condition_id: str = Field(
        ...,
        description="Numeric condition identifier (e.g. '1', '2', '3').",
        json_schema_extra={
            "entity_label": "ConditionId",
            "instance_key": True,
        },
    )
    variation_code: str = Field(
        ...,
        description="Parent variation code (e.g. 'Q.I.a.1').",
        json_schema_extra={
            "entity_label": "VariationCode",
            "instance_key": True,
        },
    )
    description: str | None = Field(
        default=None,
        description="Free-text description of the condition requirement.",
    )
    is_mandatory: bool | None = Field(
        default=None,
        description="Whether this condition is mandatory (True) or optional (False).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Substance model (dual-field pattern)
# ─────────────────────────────────────────────────────────────────────────────


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


class SubstanceEntry(VariationBaseModel, ExtractionModel):
    """A substance entry with dual-field pattern: raw text + normalized structure.

    The raw_substance field captures the original text with maximum fidelity.
    The normalized_substance field breaks it into structured, queryable components.
    In the tabular pipeline, the NormalizationEngine uses raw_substance as input
    to populate normalized_substance via LLM.
    In the unstructured pipeline, the LLM fills both fields directly.
    """

    raw_substance: str = Field(
        ...,
        description=(
            "Free-text substance description exactly as it appears in the source. "
            "Preserve original formatting, trade names, and abbreviations."
        ),
    )

    normalized_substance: NormalizedSubstance | None = Field(
        default=None,
        description=(
            "Structured substance data derived from raw_substance: INN name, "
            "CAS number, strength, and pharmaceutical form parsed into separate fields."
        ),
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_substance"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fee model (instance_key for deduplication)
# ─────────────────────────────────────────────────────────────────────────────


class Fee(VariationBaseModel, ExtractionModel):
    """A fee entry from a fee schedule.

    Uses a composite instance_key (country_code, procedure_type, role) so that
    the same fee from different documents resolves to the same ModelInstance node.
    """

    country_code: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code.",
        json_schema_extra={
            "entity_label": "Country",
            "instance_key": True,
        },
    )
    procedure_type: str = Field(
        ...,
        description="Procedure type code: IA, IB, II, IAIN, A, or BA.",
        json_schema_extra={
            "entity_label": "ProcedureType",
            "instance_key": True,
        },
    )
    role: str = Field(
        ...,
        description="Fee role: applicant, holder, or third party.",
        json_schema_extra={
            "entity_label": "FeeRole",
            "instance_key": True,
        },
    )
    rate: str = Field(
        ...,
        description="Fee amount without currency symbol (e.g. '1234.56').",
    )
    currency: str = Field(
        ...,
        description="ISO 4217 currency code (e.g. 'EUR', 'USD').",
        json_schema_extra={"entity_label": "Currency"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ingredient relationship model (field_relationships)
# ─────────────────────────────────────────────────────────────────────────────


class IngredientPair(VariationBaseModel, ExtractionModel):
    """Relationship between an active substance and its excipient.

    Uses field_relationships to create a typed relationship between the
    ActiveSubstance and Excipient LabeledEntity nodes.
    """

    active_substance: str = Field(
        ...,
        description="INN name of the active substance.",
        json_schema_extra={
            "entity_label": "ActiveSubstance",
            "field_relationships": [
                {
                    "to_field": "excipient",
                    "rel_type": "CONTAINS_EXCIPIENT",
                }
            ],
        },
    )
    excipient: str = Field(
        ...,
        description="Name of the excipient.",
        json_schema_extra={"entity_label": "Excipient"},
    )
    ratio: str | None = Field(
        default=None,
        description="Mixing ratio or percentage (e.g. '95:5', '10%').",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main variation model (all patterns combined)
# ─────────────────────────────────────────────────────────────────────────────


class VariationModel(VariationBaseModel, ExtractionModel):
    """A pharmaceutical variation entry.

    Combines all advanced model design patterns:
    - Dual-field: raw_date + normalized_date
    - Entity labels: variation_code, procedure_type, country_code
    - Instance keys: variation_code, procedure_type
    - Domain validators: procedure_type normalization, variation_code OCR fix
    - Instance relationships: condition_ids → ConditionModel
    - Normalization model: normalized_date from raw_date
    """

    # ── Entity labels + instance keys ────────────────────────────────────
    variation_code: str = Field(
        ...,
        description="Variation code (e.g. 'Q.I.a.1', 'II.A.1(a)').",
        json_schema_extra={
            "entity_label": "VariationCode",
            "instance_key": True,
        },
    )
    procedure_type: str = Field(
        ...,
        description="Procedure type code: IA, IB, II, IAIN, A, or BA.",
        json_schema_extra={
            "entity_label": "ProcedureType",
            "instance_key": True,
        },
    )
    country_code: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code.",
        json_schema_extra={"entity_label": "Country"},
    )

    # ── Entity labels (no instance key) ──────────────────────────────────
    product_name: str | None = Field(
        default=None,
        description="Trade name of the medicinal product.",
        json_schema_extra={"entity_label": "ProductName"},
    )
    marketing_authorization_holder: str | None = Field(
        default=None,
        description="Full legal name of the MAH.",
        json_schema_extra={"entity_label": "MAH"},
    )

    # ── Dual-field: raw + normalized ─────────────────────────────────────
    raw_date: str | None = Field(
        default=None,
        description="Free-text date as it appears in the source (e.g. '15 March 2024').",
    )

    normalized_date: NormalizedDate | None = Field(
        default=None,
        description="Structured date derived from raw_date.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_date"],
        },
    )

    # ── Instance relationships (Level 3) ─────────────────────────────────
    condition_ids: list[str] | None = Field(
        default=None,
        description="IDs of applicable conditions (e.g. ['1', '2', '3']).",
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "ConditionModel",
                    "rel_type": "APPLIES_TO",
                    "join_via": {
                        "condition_ids": "condition_id",
                        "variation_code": "variation_code",
                    },
                }
            ],
        },
    )

    # ── Regular scalar fields ────────────────────────────────────────────
    procedure_types_referenced: list[str] | None = Field(
        default=None,
        description="Other procedure type codes referenced in this variation.",
        json_schema_extra={"entity_label": "ProcedureType"},
    )
    summary: str | None = Field(
        default=None,
        description="Brief summary of the variation purpose.",
    )
    regulatory_pathway: str | None = Field(
        default=None,
        description="Regulatory pathway: centralized, mutual recognition, or decentralized.",
        json_schema_extra={"entity_label": "RegulatoryPathway"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# List wrapper for multi-instance sections
# ─────────────────────────────────────────────────────────────────────────────


class VariationList(ExtractionModel):
    """A section containing multiple variation entries.

    Use this model when a document section contains a table or list of
    variations. The LLM extracts all entries into the variations list.
    """

    variations: list[VariationModel] = Field(
        ...,
        description="All variation entries found in this section.",
    )
    section_title: str | None = Field(
        default=None,
        description="Title or heading of the section containing these variations.",
    )
```

### What this example demonstrates

| Model | Patterns Used |
|---|---|
| `VariationBaseModel` | Domain validators (`normalize_procedure_types`, `fix_variation_code_ocr`, `fix_country_code`) with `check_fields=False` |
| `NormalizedDate` | Simple nested model for normalization target |
| `NormalizedAddress` | Nested model with `entity_label` on `country_code` |
| `NormalizedSubstance` | Nested model with multiple `entity_label` fields |
| `ConditionModel` | `instance_key` (composite), `entity_label`, inherits validators |
| `SubstanceEntry` | **Dual-field** pattern (`raw_substance` + `normalized_substance`), `normalization_model`, `normalization_source_fields` |
| `Fee` | Composite `instance_key` (3 fields), `entity_label` on multiple fields |
| `IngredientPair` | `field_relationships` creating `[:CONTAINS_EXCIPIENT]` between entities |
| `VariationModel` | **All patterns**: dual-field, entity labels, instance keys, validators, instance relationships, normalization model |
| `VariationList` | List wrapper for multi-instance sections |

### Neo4j graph produced

Processing a document with `VariationModel` produces:

```
(:StructureNode)
  -[:HAS_EXTRACTION]->
  (:ExtractionResult)
    -[:USES_PRIMARY_MODEL]->(:CatalogModel {name: "VariationModel"})
    -[:HAS_CONDITION_IDS {index}]->(:ModelInstance {model_class: "VariationModel"})
    -[:REFERENCES]->(:LabeledEntity {label: "VariationCode", value: "Q.I.a.1"})
    -[:REFERENCES]->(:LabeledEntity {label: "ProcedureType", value: "II"})
    -[:REFERENCES]->(:LabeledEntity {label: "Country", value: "DE"})
    -[:REFERENCES]->(:LabeledEntity {label: "ProductName", value: "Metformin 500mg"})

  (:ModelInstance {model_class: "VariationModel"})
    -[:APPLIES_TO]->
    (:ModelInstance {model_class: "ConditionModel", condition_id: "1"})
    -[:APPLIES_TO]->
    (:ModelInstance {model_class: "ConditionModel", condition_id: "2"})

  (:LabeledEntity {label: "VariationCode", value: "Q.I.a.1"})
    ← shared across all extractions with this code
```

---

## Next steps

- **[Custom Models](custom-models.md)** — Basic model definition and field descriptions.
- **[Tabular Pipeline](tabular-pipeline.md)** — How the tabular pipeline uses normalization models.
- **[Neo4j Graph Storage — instance_key linking](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key)** — §4 "Instance Keys" and §9 "Instance Relationships" of this guide map directly onto this section's real-graph validation (UID hashing, shell-node lifecycle, live `CTDSectionSpec` example).
- **[Neo4j Graph Storage — Normalized Models](neo4j-graph.md#normalized-models-tabular-pipeline)** — §7 "The Normalization Model Mechanism" of this guide maps directly onto this section's explanation of how normalized fields are written to the graph.
- **[Architecture](../architecture.md)** — Detailed pipeline stage walkthrough, including entity extraction graph mapping.
