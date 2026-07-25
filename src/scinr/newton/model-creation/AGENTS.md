# AGENTS.md — Model Creation Guide for AI Agents

> **Audience:** AI coding agents (Claude, GPT, Cursor, etc.) creating or modifying extraction models for `scinr.newton`.
>
> **Stack:** Python 3.11+ · Pydantic v2 · Neo4j 5.x · `scinr.newton`

---

## 0. Non-Negotiable Rules

Before writing any code, internalise these rules. Violating them causes silent failures.

| # | Rule | Why |
|---|---|---|
| 1 | Every directory that contains `.py` files **must** have an `__init__.py` | Python won't treat it as a package; relative imports will raise `ImportError` |
| 2 | All imports between files in the same external package **must** be relative (`.`, `..`, `...`) | Bare absolute imports depend on `sys.path` order and break unpredictably |
| 3 | `default=[]` on list fields is **forbidden** | Mutable default is shared across instances; always use `default_factory=list` |
| 4 | All model classes **must** inherit from `ExtractionModel` | `ConfigDict(extra="forbid")` is required to prevent silent LLM hallucinations |
| 5 | Every field **must** have `description=` with ≥ 15 words | The LLM uses the description as its primary extraction signal |
| 6 | `entity_label` **must not** be added to free-text narrative fields | Creates meaningless graph singletons and degrades deduplication quality |
| 7 | Every model referenced as `target_model` in `instance_relationships` **must** mark its key fields `instance_key: True` | Without this the shell `ModelInstance` node never merges with the real one |

---

## 1. Directory Structure and Python Modules

### 1.1 The `__init__.py` rule is absolute

Every folder that contains Python files must be a Python package. This means it needs an `__init__.py` file (which may be empty). **No exceptions.**

```
my_package/                        ← add __init__.py HERE
├── __init__.py                    ← REQUIRED (may be empty)
├── base.py                        ← shared ExtractionModel or custom base
└── pharma_regulatory/             ← add __init__.py HERE
    ├── __init__.py                ← REQUIRED
    ├── baseModels.py              ← domain-specific base with validators
    ├── structuralSignalModel.py   ← cross-cutting classification model
    ├── variation_guidelines/      ← add __init__.py HERE
    │   ├── __init__.py            ← REQUIRED
    │   ├── catalog.py
    │   └── models.py
    └── bpg/                       ← add __init__.py HERE
        ├── __init__.py            ← REQUIRED
        ├── catalog.py
        └── models.py
```

**When creating new sub-themes or helper folders, the first file you create must always be `__init__.py`.**

### 1.2 Theme vs. Sub-theme

A **theme** is a top-level extraction domain registered in `ThemeRegistry`. It requires a `catalog.py` with `THEME_DESCRIPTION` and `SELECTABLE_MODELS`.

A **sub-theme** is a nested folder with its own `catalog.py`, representing a specialised subset of a parent theme.

```
models/
└── pharma_regulatory/            ← theme (has catalog.py)
    ├── __init__.py
    ├── catalog.py                ← aggregates all sub-theme models
    ├── variation_guidelines/     ← sub-theme (has catalog.py)
    │   ├── __init__.py
    │   ├── catalog.py
    │   └── models.py
    └── bpg/                      ← sub-theme (has catalog.py)
        ├── __init__.py
        ├── catalog.py
        └── models.py
```

**Create a sub-theme when:** the domain has clearly differentiated document types with incompatible model sets (e.g. variation guidelines vs. best practice guidelines vs. Q&A documents).

**Keep in the same theme when:** models are complementary and often used together on the same document type.

### 1.3 Shared base files

Place shared validators, normalization functions, and base classes in a `baseModels.py` (or `base.py`) at the appropriate level of the hierarchy:

```
my_package/
├── base.py                  ← ExtractionModel (if needed as a local copy)
└── pharma_regulatory/
    ├── baseModels.py        ← NormalizedBaseModel shared across ALL sub-themes
    ├── variation_guidelines/
    │   └── models.py        ← imports from ..baseModels
    └── bpg/
        └── models.py        ← imports from ..baseModels
```

---

## 2. Imports: The Golden Rule

### 2.1 Always use relative imports inside your package

```python
# ✅ CORRECT — import from the same directory
from .models import MyModel
from .base import MyBaseClass

# ✅ CORRECT — import from the parent directory
from ..baseModels import NormalizedBaseModel

# ✅ CORRECT — import from two levels up
from ...base import ExtractionModel

# ✅ CORRECT — import from the installed scinr library (absolute is correct here)
from scinr.newton.models.base import ExtractionModel

# ❌ INCORRECT — bare import (works only if the directory happens to be in sys.path)
from models import MyModel

# ❌ INCORRECT — absolute path using your own package name
from my_package.pharma_regulatory.models import MyModel

# ❌ INCORRECT — absolute path using a sibling in your own package
from pharma_regulatory.baseModels import NormalizedBaseModel
```

### 2.2 Counting the dots: depth chart

The number of leading dots equals the number of directory levels to go up, **not including the current file's directory** (which is always `.`):

```
own_models/
├── base.py                        ← 3 dots from bpg/models.py: from ...base
└── pharma_regulatory/
    ├── baseModels.py              ← 2 dots from bpg/models.py: from ..baseModels
    └── bpg/
        └── models.py              ← I am HERE
```

```python
# In own_models/pharma_regulatory/bpg/models.py:
from ...base import ExtractionModel          # 3 dots → own_models/base.py
from ..baseModels import NormalizedBaseModel # 2 dots → own_models/pharma_regulatory/baseModels.py
from .catalog import THEME_DESCRIPTION       # 1 dot  → own_models/pharma_regulatory/bpg/catalog.py
```

### 2.3 The one absolute-import exception

`scinr` is an installed package. You may (and should) import `ExtractionModel` from it using an absolute path when you do not maintain your own `base.py`:

```python
# This is always correct regardless of where your package lives:
from scinr.newton.models.base import ExtractionModel
```

---

## 3. File Structure and Ordering

### 3.1 Canonical order within a `models.py`

```
1. Module docstring
2. from __future__ import annotations
3. Standard library imports (re, enum, typing)
4. Third-party imports (pydantic)
5. Local relative imports (base classes, shared models)
6. ── ENUMS ──────────────────────────────── (controlled vocabulary)
7. ── BASE / SHARED SUBMODELS ────────────── (reused across main models)
8. ── TARGET MODELS ──────────────────────── (models that are referenced via instance_relationships)
9. ── MAIN MODELS ─────────────────────────── (one per document section type)
10. ── LIST WRAPPERS ──────────────────────── (XxxModelList for multi-instance sections)
```

Declare models before they are referenced. If model A has a field of type B, declare B first.

### 3.2 Minimal well-formed `models.py`

```python
"""Short description of what this module covers."""
from __future__ import annotations

from enum import Enum
from pydantic import Field
from scinr.newton.models.base import ExtractionModel


class StatusEnum(str, Enum):
    """
    Status of the item.

    Values:
      active   — Item is currently in use.
      inactive — Item has been retired or replaced.
      unknown  — Status not stated in the document.
    """
    ACTIVE   = "active"
    INACTIVE = "inactive"
    UNKNOWN  = "unknown"


class ItemModel(ExtractionModel):
    """A single item entry from a regulatory catalogue document."""

    item_code: str = Field(
        ...,
        description=(
            "Unique alphanumeric code as written in the document (e.g. 'A-001', 'Q.I.a.1'). "
            "Never omit the prefix. None not applicable — this field is required."
        ),
        json_schema_extra={"entity_label": "ItemCode", "instance_key": True},
    )
    description: str = Field(
        ...,
        description="Verbatim description of the item as stated in the document.",
    )
    status: StatusEnum | None = Field(
        default=None,
        description=(
            "Lifecycle status of the item per StatusEnum. "
            "'active' if currently valid, 'inactive' if retired. None if not stated."
        ),
    )
    related_codes: list[str] = Field(
        default_factory=list,
        description=(
            "Other item codes explicitly cross-referenced in this entry. "
            "Each value is a raw code string. Empty list if none are mentioned."
        ),
    )


class ItemModelList(ExtractionModel):
    """Use when the section defines TWO OR MORE item entries in a list or table."""

    items: list[ItemModel] = Field(
        default_factory=list,
        description=(
            "List of item entries. Each element represents one distinct item. "
            "Use this model instead of ItemModel when the section is a table or list "
            "covering two or more items."
        ),
    )
```

---

## 4. Field Design Best Practices

### 4.1 Scalar fields

```python
# Optional — field may or may not be present
name: str | None = Field(default=None, description="...")

# Required — always present in this document type
code: str = Field(..., description="...")

# Required but can be empty — prefer empty string over None for key fields
# used in join_via or instance_key context
root_code: str = Field(default="", description="...")
```

### 4.2 List fields

```python
# ALWAYS use default_factory — NEVER default=[]
items: list[str]      = Field(default_factory=list, description="...")
sub_models: list[Sub] = Field(default_factory=list, description="...")
```

### 4.3 Enums

```python
class ProcedureType(str, Enum):
    """
    Normalized procedure type code.

    Use str as the base class — guarantees JSON-serializable values and
    correct Neo4j storage without extra serialization steps.

    Values:
      IA   — Type IA notification (immediate effect, notify within 12 months).
      IB   — Type IB notification (implement after 30-day review window).
      II   — Type II prior-approval variation.
    """
    IA  = "IA"
    IB  = "IB"
    II  = "II"
```

**Rule:** Always use `str` as the base class for enums used in extraction models. The `use_enum_values=True` in `ExtractionModel.model_config` will store the plain string value, but `str` base is still required for safe serialization elsewhere.

### 4.4 Numeric values with units

```python
# Preserve original format — do NOT convert to float
batch_size: str | None = Field(
    default=None,
    description=(
        "Batch size as stated in the document, including units "
        "(e.g. '100 kg', '500 L', '1×10⁶ cells'). "
        "Preserve original formatting and units. None if not stated."
    ),
)
```

### 4.5 `entity_label`: when to use and when NOT to use

**Use `entity_label` for:**
- Named real-world entities that may recur across documents (codes, substances, facilities, procedure types, country codes)
- Values stable enough for normalisation (identifiers, proper nouns)
- Fields where cross-document deduplication is meaningful

**Do NOT use `entity_label` for:**
- Long free-text descriptions unique to each document instance
- Narrative summary paragraphs
- Numeric measurements in context (`"25°C/60% RH"`)
- Boolean flags or status strings
- Fields where the value is essentially a sentence or paragraph

### 4.6 `instance_key: True`

Mark a field `instance_key: True` when:
1. The model is (or may be) referenced as a `target_model` in another model's `instance_relationships`.
2. The field forms part of the unique key that identifies one instance of this model.

When multiple fields together form the key (composite key), mark ALL of them:

```python
class Fee(ExtractionModel):
    country_code:   str = Field(..., json_schema_extra={"entity_label": "Country",       "instance_key": True})
    procedure_type: str = Field(..., json_schema_extra={"entity_label": "ProcedureType", "instance_key": True})
    role:           str = Field(..., json_schema_extra={"entity_label": "FeeRole",       "instance_key": True})
    rate:           str = Field(..., description="Fee amount without currency symbol.")
```

---

## 5. Graph Relationships

### 5.1 Choosing between `field_relationships` and `instance_relationships`

| Question | Answer | Use |
|---|---|---|
| Are both fields extracted from the **same** structural section (same `StructureNode`)? | Yes | `field_relationships` |
| Do the fields come from **different** document sections or documents? | Yes | `instance_relationships` |
| Is the relationship between two **named entities** (`:LabeledEntity` nodes)? | Yes | `field_relationships` |
| Is the relationship between two **model records** (`:ModelInstance` nodes)? | Yes | `instance_relationships` |
| Do you need a **forward reference** (target may not exist yet)? | Yes | `instance_relationships` |

### 5.2 `field_relationships` syntax

Connects two `:LabeledEntity` nodes within the same extracted model instance. Both fields must be siblings and both must have `entity_label`.

```python
root_code: str | None = Field(
    default=None,
    json_schema_extra={
        "entity_label": "VariationCode",
        "field_relationships": [
            {"to_field": "child_code", "rel_type": "HAS_CHILD_VARIATION_CODE"},
        ],
    },
)
child_code: str = Field(
    ...,
    json_schema_extra={"entity_label": "VariationCode", "instance_key": True},
)
```

Produces: `(:LabeledEntity:VariationCode {value:"Q.I.a.1"}) -[:HAS_CHILD_VARIATION_CODE]-> (:LabeledEntity:VariationCode {value:"Q.I.a.1(a)"})`

**Rules:**
- Both source and target fields must have `entity_label`
- `to_field` must be the **name** of a sibling field in the same model
- Relationship is only created when **both** fields are non-null
- `rel_type` must be `UPPER_SNAKE_CASE`

### 5.3 `instance_relationships` syntax

Connects `:ModelInstance` nodes across different sections or documents. Creates shell nodes for targets that have not yet been extracted.

```python
json_schema_extra={
    "instance_relationships": [
        {
            "target_model": "TargetModelClassName",  # string — PascalCase class name
            "join_via": {
                "local_field": "remote_key_field",   # scalar sibling → target instance_key field
                "list_field":  "remote_key_field_2", # list field (fan-out) → target instance_key field
            },
            "rel_type": "RELATIONSHIP_TYPE",         # UPPER_SNAKE_CASE
        }
    ]
}
```

### 5.4 Fan-out pattern (one-to-many via list field)

When the source field is a **list**, one target `ModelInstance` is created per list item:

```python
condition_ids: list[str] = Field(
    default_factory=list,
    description="IDs of associated conditions (e.g. ['1', '2', 'A']).",
    json_schema_extra={
        "instance_relationships": [
            {
                "target_model": "ConditionModel",
                "join_via": {
                    "root_variation_code": "variation_code",  # fixed anchor key
                    "condition_ids":       "condition_id",    # fan-out: one target per item
                },
                "rel_type": "HAS_CONDITION",
            }
        ]
    },
)
```

- **Fixed keys** (`root_variation_code → variation_code`): scalar fields that scope the target to the correct parent.
- **Fan-out key** (`condition_ids → condition_id`): the list field itself; `join_via` entry maps the list field name to the corresponding `instance_key` field on the target.

### 5.5 Dual pattern: `entity_label` + `instance_relationships`

A field can simultaneously create a `:LabeledEntity` global singleton AND a `:ModelInstance` cross-model edge:

```python
procedure_type: str = Field(
    default="",
    description="Procedure type code: IA, IB, II, IAIN, A, or BA.",
    json_schema_extra={
        "entity_label": "ProcedureType",       # → creates :LabeledEntity node
        "instance_relationships": [
            {
                "target_model": "ProcedureTypeModel",
                "join_via": {"procedure_type": "procedure_type"},
                "rel_type": "HAS_PROCEDURE_TYPE",
            }
        ],                                      # → creates :ModelInstance edge
    },
)
```

Use this dual pattern when the value is both a named entity (needs global dedup) AND points to a structured model instance (needs cross-document linking).

---

## 6. Validators and Normalization

### 6.1 The `NormalizedBaseModel` pattern

When a domain requires consistent normalization of specific field values across all models (OCR correction, code normalization, case normalization), create a shared base class with `field_validator`:

```python
# pharma_regulatory/baseModels.py
import re
from pydantic import BaseModel, field_validator


def normalize_code(v: str) -> str:
    """Apply domain-specific normalization to a code string."""
    # Example: OCR fix → uppercase → strip whitespace
    v = v.strip().upper()
    v = re.sub(r"[\s_/-]", "", v)  # remove separators
    return v


class NormalizedBaseModel(BaseModel):
    """Base class that applies field normalization before Pydantic validation."""

    @field_validator("procedure_type", "procedure_types_referenced", mode="before", check_fields=False)
    @classmethod
    def normalize_procedure_types(cls, v):
        if isinstance(v, str):
            return normalize_code(v)
        if isinstance(v, list):
            return [normalize_code(item) if isinstance(item, str) else item for item in v]
        return v
```

**Key details:**
- `check_fields=False` makes the validator **optional**: it runs only if the subclass actually has that field. Without this flag, Pydantic raises an error when a subclass inherits the validator but doesn't declare the field.
- `mode="before"` applies normalization before Pydantic's own type validation.
- List handling: always check `isinstance(v, list)` and map over items.

### 6.2 When to create a domain-specific base class

Create a `NormalizedBaseModel` (or equivalent) when:
- Multiple models in the same domain share the same normalization logic.
- OCR corrections are needed (e.g. `Q.1.a.1` → `Q.I.a.1` for variation codes).
- Code values need to be consistently uppercased and stripped (e.g. `"i a"` → `"IA"` for procedure types).

Do NOT create a domain-specific base just for convenience — it adds indirection. Use it only when sharing validation is meaningful.

### 6.3 OCR fix validators

A common pattern for regulatory codes that suffer from OCR mis-recognition:

```python
def normalize_variation_code(v: str) -> str:
    # OCR fix: Q.1.a.1 → Q.I.a.1 (digit 1 misread as letter l or numeral)
    v = re.sub(r"(?<=[A-Za-z])\.(?:1|l)\.", ".I.", v)
    # OCR fix: Q.I.a.1.a → Q.I.a.1(a) (trailing sub-code format)
    v = re.sub(r"(?<=\d)\.([a-zA-Z])$", r"(\1)", v)
    return v
```

Apply this in a `NormalizedBaseModel` validator, not inline in the field description, so it applies consistently without relying on the LLM.

---

## 7. Docstrings for the Annotation LLM

The annotation agent reads **class docstrings** to decide which model to apply to each document section. Poorly written docstrings cause misclassification.

### 7.1 First line rules

- **≤ 15 words**, in English.
- Start with the entity type: `"A single..."`, `"Full definition of..."`, `"Use when..."`.
- Must be informative enough to distinguish from similar models.

```python
# ✅ GOOD
class VariationCodeModel(ExtractionModel):
    """A single variation code entry from the EU variation guidelines (Official Journal)."""

# ✅ GOOD — explicit USE/DO NOT USE
class VariationCodeWithDocsAndConditionModelList(ExtractionModel):
    """Use when TWO OR MORE variation codes and their conditions are defined inline."""

# ❌ BAD — too vague
class VariationCodeModel(ExtractionModel):
    """Variation code model."""

# ❌ BAD — too long for first line
class VariationCodeModel(ExtractionModel):
    """This model captures variation codes from EU regulatory documents including all procedure types and conditions."""
```

### 7.2 When to use / when NOT to use

For models where misclassification is likely, add explicit conditions:

```python
class VariationCodeModel(ExtractionModel):
    """
    A single variation code entry from EU variation guidelines.

    USE THIS MODEL when the section defines exactly ONE variation code, OR when
    conditions and documentation are listed in separate sibling/child sections.

    DO NOT use this model when the section defines TWO OR MORE variation codes
    with their conditions and documentation all inline — use
    VariationCodeWithDocsAndConditionModelList instead.
    """
```

### 7.3 Complementary model hints

When a model is typically used together with other models, declare this explicitly in the docstring:

```python
class DocumentationModel(ExtractionModel):
    """
    A documentation requirement for a variation.
    They are perfect candidates as ComplementaryModels for VariationCodeModel.
    """
```

The annotation agent reads `ComplementaryModels` hints and may suggest them as secondary models to apply alongside the primary model.

### 7.4 Field descriptions

A good field description answers three questions:
1. **What** is this data point exactly?
2. **What format** is expected (with concrete examples)?
3. **When is `None` (or empty string/empty list) correct?**

```python
# ✅ GOOD
variation_code: str = Field(
    ...,
    description=(
        "Full, absolute variation code identifier as written in the document — "
        "always include the top-level prefix (e.g. 'Q.III.1(b)4', not '(b)4'; "
        "'Q.I.a.1(a)', not '(a)'; 'B.II.b.1'). "
        "Never extract a partial or relative sub-code without its parent prefix."
    ),
)

# ❌ BAD — too short, no examples, no None condition
variation_code: str = Field(..., description="The variation code.")
```

---

## 8. `catalog.py` and `THEME_DESCRIPTION`

### 8.1 Minimal correct `catalog.py`

```python
"""Catalog for the variation_guidelines sub-theme."""
from __future__ import annotations

# ✅ Relative import — references a sibling in the same package
from .models import (
    VariationCodeModel,
    VariationCodeWithDocsAndConditionModelList,
    DocumentationModel,
    DocumentationModelList,
    ConditionModel,
    ConditionModelList,
    ProcedureTypeModel,
    ProcedureTypeModelList,
)

THEME_DESCRIPTION: str = (
    "EU pharmaceutical variation guidelines (Official Journal, EC Regulation 1234/2008). "
    "Covers variation type classification (IA, IAIN, IB, II, A, BA), variation codes "
    "(e.g. Q.I.a.1, B.II.b.1), conditions, documentation requirements, and procedural rules. "
    "Use for sections defining or explaining variation codes, their procedure types, "
    "conditions, and supporting documentation. "
    "Distinct from BPG (best practice guidelines) and Q&A documents."
)

SELECTABLE_MODELS: list[type] = [
    VariationCodeWithDocsAndConditionModelList,   # multi-code sections with inline data
    VariationCodeModel,                           # single-code sections or separate docs/conditions
    DocumentationModelList,                       # sections listing ≥2 documentation requirements
    DocumentationModel,                           # single documentation requirement
    ConditionModelList,                           # sections listing ≥2 conditions
    ConditionModel,                               # single condition
    ProcedureTypeModelList,                       # sections defining ≥2 procedure types
    ProcedureTypeModel,                           # single procedure type definition
]
```

### 8.2 How to write an effective `THEME_DESCRIPTION`

The annotation LLM reads `THEME_DESCRIPTION` to decide whether a document section belongs to this theme. A good description:

- **Names the document types** it covers (`"EU Official Journal"`, `"EMA Best Practice Guidelines"`)
- **Names regulatory standards** when applicable (`"EC Regulation 1234/2008"`, `"ICH CTD Module 3"`)
- **Distinguishes** from adjacent themes that could be confused (`"Distinct from BPG..."`)
- **Gives examples** of the entities it captures (`"variation codes (e.g. Q.I.a.1)"`)

```python
# ✅ GOOD
THEME_DESCRIPTION: str = (
    "EU pharmaceutical variation guidelines (Official Journal, EC Regulation 1234/2008). "
    "Covers variation codes (IA, IB, II), conditions, and documentation requirements. "
    "Distinct from BPG and Q&A documents."
)

# ❌ BAD — too vague, LLM will classify everything here
THEME_DESCRIPTION: str = "Pharmaceutical regulatory documents."

# ❌ BAD — only one sentence, no distinguishing information
THEME_DESCRIPTION: str = "Documents about variation codes."
```

### 8.3 `SELECTABLE_MODELS` ordering

Order from most to least specific (the LLM tends to select models appearing earlier when confidence is similar):
1. List wrapper models for multi-instance sections (most specific)
2. Main models for single-instance sections
3. Supporting/complementary models

### 8.4 Parent catalog aggregating sub-themes

When a parent folder has its own `catalog.py` that aggregates sub-theme models, use explicit relative imports from sub-packages:

```python
# pharma_regulatory/catalog.py
from .variation_guidelines.models import VariationCodeModel, ProcedureTypeModel
from .bpg.models import BPGRecommendationModel
from .qa.models import QAEntryModel

THEME_DESCRIPTION: str = "..."
SELECTABLE_MODELS: list[type] = [
    VariationCodeModel,
    BPGRecommendationModel,
    QAEntryModel,
]
```

---

## 9. The List Wrapper Pattern

### 9.1 When to create a `XxxModelList` wrapper

Create a list wrapper model alongside the main model when:
- A document section **regularly contains a table or list** of the same entity (e.g. a fee schedule table, a variation code table)
- A section can contain **zero, one, or many** instances of the entity
- The annotation agent needs a way to extract multiple entities in a single extraction call

**Do NOT create a list wrapper when:**
- Every section of this type **always** has exactly one instance
- The entities are already captured as a `list[SubModel]` field inside a parent model

### 9.2 List wrapper structure

```python
class XxxModelList(ExtractionModel):
    """
    Use when the section defines TWO OR MORE Xxx entries in a list or table.
    Each item captures: [key fields of XxxModel].
    DO NOT use this model for a single Xxx — use XxxModel instead.
    """

    xxx_models: list[XxxModel] = Field(
        default_factory=list,
        description=(
            "List of Xxx models. Use XxxModel when there is only one entry. "
            "Use this model when the section is a table or list with two or more entries."
        ),
    )
```

**Rules:**
- The wrapper inherits from `ExtractionModel`, not from the main model's base class.
- The wrapper has **one field**: the list.
- Both `XxxModel` AND `XxxModelList` must appear in `SELECTABLE_MODELS`.
- The docstring first line should start with `"Use when the section defines TWO OR MORE..."`.

---

## 10. Anti-Patterns to Avoid

| Anti-pattern | Why it fails | Correct approach |
|---|---|---|
| `default=[]` on a list field | Mutable default shared across instances — Pydantic rejects it or causes subtle bugs | `default_factory=list` |
| Bare import: `from models import X` | Fragile; depends on `sys.path` at runtime | Relative: `from .models import X` |
| Absolute own-package import: `from my_pkg.theme.models import X` | Breaks when `sys.path` changes | Relative: `from .models import X` |
| Missing `__init__.py` in a subfolder | Python won't treat it as a package; relative imports fail with `ImportError` | Add empty `__init__.py` to every folder with `.py` files |
| `entity_label` on a free-text description field | Creates meaningless `:LabeledEntity` singletons; degrades graph quality | Only add `entity_label` to short, stable, identifier-like values |
| `field_relationships` pointing to a field without `entity_label` | The target node doesn't exist; Neo4j write silently ignored | Ensure `to_field` also has `entity_label` |
| Missing `instance_key: True` on target model key fields | Shell nodes created by `instance_relationships` never merge with real nodes | Mark ALL key fields on the target model with `instance_key: True` |
| Fan-out `join_via` key name mismatch | Zero target nodes created — field name in `join_via` must exactly match Python field name | Double-check that `join_via` keys use the exact Python field names |
| `Optional[str]` instead of `str \| None` | Verbose; inconsistent with the codebase style | Use `str \| None` (PEP 604 union syntax) |
| Vague `THEME_DESCRIPTION` | LLM misclassifies sections; wrong model applied | Be specific: name document types, regulatory standards, distinguish from adjacent themes |
| Not adding `XxxModelList` to `SELECTABLE_MODELS` | Agent can never select it directly | Add both `XxxModel` and `XxxModelList` to `SELECTABLE_MODELS` |
| Inheriting from `BaseModel` instead of `ExtractionModel` | `extra="forbid"` is missing; LLM hallucinations not caught | Inherit from `ExtractionModel` (except `Triple` fallback — historical exception) |
| Class docstring longer than 15 words on the first line | Annotation agent truncates; key info may not be read | First line ≤ 15 words; put details on subsequent lines |
| Validator without `check_fields=False` on inherited base | Pydantic raises `PydanticUserError` when a subclass doesn't declare the validated field | Always use `check_fields=False` on validators in shared base classes |

---

## 11. Pre-merge Checklist

### Structure
- [ ] Every directory with `.py` files contains `__init__.py` (including new sub-theme folders)
- [ ] `catalog.py` uses relative imports (`from .models import ...`)
- [ ] `models.py` uses relative imports for all internal modules

### Models
- [ ] All classes inherit from `ExtractionModel` (or a subclass of it)
- [ ] Every class has a docstring (first line ≤ 15 words, in English)
- [ ] Every field has `description=` with ≥ 15 words (what, format, when None)
- [ ] Every optional scalar uses `default=None`
- [ ] Every list field uses `default_factory=list`
- [ ] Enums use `str` as base class

### Graph annotations
- [ ] `entity_label` only on short, stable, identifier-like fields (not free-text)
- [ ] `field_relationships` declared where a directed edge between sibling entities is needed
- [ ] `instance_relationships` declared where cross-section or cross-document linking is needed
- [ ] Every `target_model` in `instance_relationships` has `instance_key: True` on its key fields
- [ ] Fan-out `join_via` field names exactly match Python field names

### Theme registration
- [ ] `THEME_DESCRIPTION` is specific, technical, and distinguishable
- [ ] `SELECTABLE_MODELS` lists all top-level models (including list wrappers)
- [ ] If list wrapper `XxxModelList` exists, both `XxxModel` and `XxxModelList` are in `SELECTABLE_MODELS`

### Validation
- [ ] Auto-discovery verified: `python -c "from scinr.newton.utils.theme_registry import ThemeRegistry; print(ThemeRegistry().list_themes())"`
- [ ] No import errors: `python -c "import my_package.my_theme.catalog"`
- [ ] At least one real document processed end-to-end (Stage 3 + Stage 4)

---

## 12. Reference Implementations

| File | What to learn |
|---|---|
| `own_models/pharma_regulatory/variation_guidelines/models.py` | `field_relationships` + `instance_relationships` + `instance_key` on same field; fan-out with fixed anchor |
| `own_models/pharma_regulatory/bpg/models.py` | Dual `entity_label` + `instance_relationships`; multiple relationships on one list field |
| `own_models/pharma_regulatory/fees/models.py` | Composite `instance_key` (3 fields); scalar `instance_relationships` |
| `own_models/pharma_regulatory/qa/models.py` | Cross-document linking from Q&A to variation guidelines models |
| `own_models/pharma_regulatory/baseModels.py` | `NormalizedBaseModel` with `check_fields=False`; OCR-fix validators |
| `own_models/pharma_regulatory/structuralSignalModel.py` | 6 enum fields all marked `instance_key: True`; cross-cutting classification model |
| `src/scinr/newton/model-creation/templates/models.py` | Copy-paste template with all patterns annotated in Spanish |
| `src/scinr/newton/model-creation/templates/catalog.py` | Copy-paste template for `catalog.py` |
