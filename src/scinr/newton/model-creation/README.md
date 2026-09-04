# Adding New Extraction Domains

> **Audience:** developers and data engineers adding new extraction capabilities to the `scinr.newton` pipeline.
>
> **Stack:** AWS Bedrock (Claude Sonnet) · LangGraph · Pydantic v2 · Neo4j · Python 3.11+

> For AI agent instructions and best practices, see [`model-creation/AGENTS.md`](AGENTS.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model System Concepts](#2-model-system-concepts)
3. [Directory Structure](#3-directory-structure)
4. [Step-by-Step: Adding a New Domain](#4-step-by-step-adding-a-new-domain)
5. [catalog.py Reference](#5-catalogpy-reference)
6. [Model Class Conventions](#6-model-class-conventions)
7. [field_relationships Syntax](#7-field_relationships-syntax)
8. [instance_relationships Syntax](#8-instance_relationships-syntax)
9. [Entity Label Convention](#9-entity-label-convention)
10. [Sub-Theme Catalogs](#10-sub-theme-catalogs)
11. [The Triple Fallback Model](#11-the-triple-fallback-model)
12. [Reference Implementations](#12-reference-implementations)
13. [Integration Checklist](#13-integration-checklist)
14. [User Theme Structure (external packages)](#14-user-theme-structure-external-packages)

---

## 1. Overview

The pipeline processes documents through five stages. Extraction models are used in the final two:

```
Document (PDF / Word)
      │
      ▼
[Stage 0] Preprocessor  →  JSON page representation
      │
      ▼
[Stage 1] Extractor     →  DocumentStructure (tree of StructureNodes)
      │
      ▼
[Stage 2] Ingestion     →  StructureNode graph in Neo4j
      │
      ▼
[Stage 3] Annotation    →  Assigns the best extraction model to each StructureNode
      │                     ← MODELS ARE SELECTED HERE
      ▼
[Stage 4] Entity Extraction  →  Named-entity subgraph in Neo4j
                                 ← MODELS ARE EXECUTED HERE
```

**Adding a new extraction domain requires only:**
1. A folder under `models/` with a `catalog.py` and one or more model files.
2. No manual registration — `ThemeRegistry` auto-discovers every folder that contains `catalog.py` at startup.

---

## 2. Model System Concepts

### ExtractionModel

All extraction model classes inherit from `ExtractionModel`, defined in
`models/pharmaceutical_quality/base.py`:

```python
from pydantic import BaseModel, ConfigDict

class ExtractionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",           # rejects unknown fields produced by the LLM
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,     # stores the enum value, not the enum object
    )
```

> **Exception:** `models/default/` (`Triple`, `TripleItem`) and `models/structural_specs/`
> (`DocumentStructure`) inherit directly from `BaseModel` for historical reasons.
> All new domains must use `ExtractionModel`.

### SELECTABLE_MODELS

Each `catalog.py` must export a list named `SELECTABLE_MODELS`. The annotation agent
presents these classes to the LLM so it can pick the best fit for each document section.
Sub-models that are only used as embedded fields inside other models do not need to appear
here — they are discovered transitively at extraction time.

### THEME_DESCRIPTION

A string in `catalog.py` that the annotation LLM reads to decide whether a document section
belongs to this domain. It must be **specific, technical, and distinguishable** from other
themes. A vague description leads to misclassification.

### field_relationships

A class-level annotation on individual model fields (placed inside `json_schema_extra`)
that tells the graph mapper to create a directed edge in Neo4j between two
`:LabeledEntity` nodes. See [§7](#7-field_relationships-syntax) for full syntax. For cross-model linking between
different document sections, use `instance_relationships` instead — see [§8](#8-instance_relationships-syntax).

### instance_relationships

A `json_schema_extra` key on a model field that instructs the graph mapper to create a
**directed edge between two `:ModelInstance` nodes** — potentially in completely different
documents or document sections. Unlike `field_relationships` (which connects `:LabeledEntity`
nodes within the same extracted record), `instance_relationships` connects the structured
extraction records themselves and supports **forward references**: the target model may not
yet exist when the relationship is written. The engine creates a "shell" `ModelInstance`
node on first encounter and populates it when the target model is later extracted.

See [§8](#8-instance_relationships-syntax) for the full syntax.

### instance_key

A `json_schema_extra` key (`"instance_key": True`) that marks which fields of a model form
its **composite key** for `ModelInstance` deduplication. The engine combines all
`instance_key` fields into a deterministic UID (`make_instance_uid`) so that:

1. Shell nodes created by an `instance_relationships` source and real nodes created by
   the target model extraction merge into the same Neo4j node.
2. Multiple documents referencing the same logical model instance (e.g. the same
   variation code `"Q.I.a.1(a)"`) share one node in the graph.

Every model that is referenced as a `target_model` in an `instance_relationships`
declaration **must** mark its key fields with `instance_key: True`.

> **See it validated against a live graph:** [`docs/user-guides/neo4j-graph.md` — Cross-Section `:ModelInstance` Linking via `instance_key`](../../../../docs/user-guides/neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key) shows this exact mechanism (UID hashing, shell-node lifecycle) against a real, populated graph example (`CTDSectionSpec`).

### LabeledEntity

Any field annotated with `json_schema_extra={"entity_label": "SomeLabel"}` becomes a
**global singleton node** in Neo4j with the base label `:LabeledEntity` and a `label`
property set to `SomeLabel`. The graph engine
deduplicates by normalised value, so the same entity mentioned in different documents
resolves to the same node and can be traversed across the entire graph.

---

## 3. Directory Structure

```
models/
├── __init__.py
├── document_structure.py          # Stage 1–2 tree models (not extraction models)
│
├── default/                       # Fallback theme — always present
│   ├── __init__.py
│   ├── catalog.py
│   └── triple.py                  # Triple / TripleItem  (RDF-style fallback)
│
├── structural_specs/              # Theme: documents that define document structure
│   ├── __init__.py
│   ├── catalog.py
│   └── document_structure.py
│
├── pharmaceutical_quality/        # Theme: ICH CTD Module 3 (CMC)
│   ├── __init__.py
│   ├── base.py                    # ExtractionModel base class
│   ├── catalog.py
│   ├── drug_substance_identity.py
│   └── ...
│
└── pharma_operations/             # Theme: supply chain & operations
    ├── __init__.py
    ├── catalog.py                 # Aggregates all 5 sub-theme models
    ├── product_master/
    │   ├── __init__.py
    │   ├── catalog.py
    │   └── models.py
    └── ...
```

A folder is treated as a **theme** if and only if it contains a `catalog.py` file.
Folders without `catalog.py` are silently skipped by `ThemeRegistry` (but their
sub-directories are still scanned for nested themes).

---

## 4. Step-by-Step: Adding a New Domain

### Step 1 — Create the directory structure

```
models/
└── my_domain/
    ├── __init__.py      # can be empty
    ├── catalog.py       # THEME_DESCRIPTION + SELECTABLE_MODELS
    └── models.py        # Pydantic model classes
```

For a domain with multiple model files, split by sub-topic and keep `catalog.py` as the
single assembly point:

```
models/
└── my_domain/
    ├── __init__.py
    ├── catalog.py
    ├── identity.py
    ├── manufacturing.py
    └── stability.py
```

### Step 2 — Define models in `models.py`

Write one `ExtractionModel` subclass per logical section type. See
[§6](#6-model-class-conventions) for detailed conventions.

```python
# models/my_domain/models.py
from __future__ import annotations
from typing import ClassVar
from pydantic import Field
from models.pharmaceutical_quality.base import ExtractionModel

class MyEntity(ExtractionModel):
    """Short docstring used in the LLM annotation prompt (≤ 15 words)."""

    name: str | None = Field(
        default=None,
        description="Full name of the entity as it appears in the document. None if absent.",
        json_schema_extra={"entity_label": "MyEntity"},
    )
    related_site: str | None = Field(
        default=None,
        description="Name of the associated site. None if not stated.",
        json_schema_extra={
            "entity_label": "Site",
            "field_relationships": [
                {"to_field": "name", "rel_type": "LOCATED_AT"}
            ],
        },
    )
```

### Step 3 — Create `catalog.py`

```python
# models/my_domain/catalog.py
from __future__ import annotations

from models.my_domain.models import MyEntity, AnotherModel

THEME_DESCRIPTION: str = (
    "Brief, specific, technical description of the domain. "
    "Mention the document types it covers, any regulatory standards (e.g. ICH Q10, GS1 EPCIS), "
    "and what distinguishes it from other themes already in the system."
)

SELECTABLE_MODELS: list[type] = [
    MyEntity,       # higher-level / most-frequently-selected models first
    AnotherModel,
]
```

### Step 4 — Verify auto-discovery (no registration needed)

`ThemeRegistry` discovers the new theme automatically. Verify with:

```bash
python -c "
from utils.theme_registry import ThemeRegistry
r = ThemeRegistry()
print(r.list_themes())
"
```

The new theme path (`my_domain`) must appear in the output. If it does not, check for
import errors in `catalog.py`.

---

## 5. `catalog.py` Reference

A minimal, complete `catalog.py`:

```python
from __future__ import annotations

from models.my_domain.models import MyEntity, AnotherModel

THEME_DESCRIPTION: str = (
    "One to three sentences. Technical and specific. "
    "Mention document types, regulatory standards, and what makes this domain distinct. "
    "The LLM annotation agent reads this verbatim to classify document sections."
)

SELECTABLE_MODELS: list[type] = [MyEntity, AnotherModel]
```

**Rules:**
- `THEME_DESCRIPTION` and `SELECTABLE_MODELS` are both **required**.
- Any import error in `catalog.py` silently drops the entire theme from `ThemeRegistry`.
  Always run the verification command after adding new imports.
- Models listed here are those the agent can select as the **primary** extraction target
  for a node. Embedded sub-models do not need to be listed (they are reached transitively).
- Order matters: place more general, frequently-matched models first.

---

## 6. Model Class Conventions

### Minimal well-formed model

```python
from __future__ import annotations
from pydantic import Field
from models.pharmaceutical_quality.base import ExtractionModel

class DrugSubstanceManufacture(ExtractionModel):
    """
    Drug substance manufacturing: process, batch formula, and manufacturer info.
    CTD sections 3.2.S.2.1–3.2.S.2.6.
    """

    substance_name: str | None = Field(
        default=None,
        description=(
            "INN or working name of the drug substance as stated in the document "
            "(e.g. 'ibuprofen', 'adalimumab'). None if not explicitly mentioned."
        ),
        json_schema_extra={"entity_label": "Substance"},
    )
    manufacturer_name: str | None = Field(
        default=None,
        description=(
            "Full legal name of the drug substance manufacturer "
            "(e.g. 'Almirall S.A.', 'Pfizer Manufacturing Belgium NV'). "
            "None if not stated."
        ),
        json_schema_extra={"entity_label": "Facility"},
    )
    batch_sizes: list[str] = Field(
        default_factory=list,
        description=(
            "All batch sizes mentioned, each with units "
            "(e.g. '100 kg', '500 L'). Empty list if none stated."
        ),
    )
```

### Naming conventions

| Element | Convention | Example |
|---|---|---|
| Theme folder | `snake_case` | `pharma_operations/` |
| Model file | `snake_case.py` | `drug_substance_identity.py` |
| Model class | `PascalCase` | `DrugSubstanceIdentity` |
| Model field | `snake_case` | `substance_name` |
| `entity_label` value | `PascalCase` | `"Substance"`, `"Facility"` |
| Neo4j relationship type | `UPPER_SNAKE_CASE` | `"MANUFACTURED_BY"` |

### Docstrings

The class docstring is included verbatim in the annotation agent's model catalogue. Rules:

- **First line** ≤ 15 words, in English — this is what the agent reads when scanning candidates.
- Include a regulatory reference if applicable (e.g. `CTD 3.2.S.2`).
- Describe *what information is captured*, not *how the model works*.

### Field descriptions

The `description` string is the primary signal the LLM uses to know what to extract.
A well-formed description answers three questions:

1. **What** is this data point exactly?
2. **What format** is expected (with concrete examples)?
3. **When is `None` correct?**

```python
cas_registry_number: str | None = Field(
    default=None,
    description=(
        "Chemical Abstracts Service (CAS) Registry Number for the drug substance, "
        "formatted as the dash-separated string (e.g. '15687-27-1', '100-51-6'). "
        "None if not stated."
    ),
    json_schema_extra={"entity_label": "CASNumber"},
)
```

### Defaults

```python
# Optional scalar field
field: str | None = Field(default=None, description="...")

# List field — always use default_factory, never default=[]
items: list[str] = Field(default_factory=list, description="...")

# Optional embedded sub-model
detail: MySubModel | None = Field(default=None, description="...")
```

### Enums

Use `str` as the base class so `use_enum_values=True` returns the plain string for JSON
serialisation and Neo4j storage:

```python
from enum import Enum

class StudyPhase(str, Enum):
    """
    Clinical trial phase. Values:
      phase_1 — First-in-human, safety and tolerability.
      phase_2 — Dose-finding and initial efficacy.
      phase_3 — Confirmatory pivotal trials.
      phase_4 — Post-marketing studies.
    """
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
```

### Field types when the model targets the tabular pipeline

If a model is (or may be) used with the **tabular ingestion pipeline** (direct CSV/XLSX
row mapping), every mappable field must be `str` or `list[str]` — not `int`, `float`,
`date`, `Enum`, or a nested submodel. The tabular pipeline combines/deduplicates values
when multiple columns map to the same field, but only for those two types; other types
silently keep the last value processed. **Do not reach for `@model_validator` to fix
this** — all tabular row instantiation goes through `model_construct()`, which skips
every Pydantic validator unconditionally, so it would never run. If a field truly needs a
real type, wrap it in a nested submodel marked `normalization_model: True` instead. See
[`model-creation/AGENTS.md` §4.7](AGENTS.md#47-models-used-with-the-tabular-pipeline-fields-must-be-str-or-liststr)
for the full rationale and a worked example.

---

## 7. `field_relationships` Syntax

`field_relationships` is declared inside `json_schema_extra` on a field that already has
an `entity_label`. It instructs the graph mapper to create a directed Neo4j edge between
the `:LabeledEntity` node of that field and the `:LabeledEntity` node of a sibling field.

### Format

```python
json_schema_extra={
    "entity_label": "SourceLabel",
    "field_relationships": [
        {"to_field": "target_field_name", "rel_type": "RELATIONSHIP_TYPE"}
    ],
}
```

- `to_field` — name of the **sibling field** in the same model that is the edge target. That field must also have an `entity_label`.
- `rel_type` — Neo4j relationship type in `UPPER_SNAKE_CASE`.
- A field may declare multiple relationships by adding more entries to the list.

### Example

```python
class MaterialLifecycleStatus(ExtractionModel):
    """Lifecycle and commercial status of a pharmaceutical material in an ERP system."""

    material_code: str | None = Field(
        default=None,
        description="Current catalogue code. None if absent.",
        json_schema_extra={"entity_label": "ProductCatalogueCode"},
    )
    predecessor_material_code: str | None = Field(
        default=None,
        description="Catalogue code of the material this one replaces. None if no predecessor.",
        json_schema_extra={
            "entity_label": "ProductCatalogueCode",
            "field_relationships": [
                {"to_field": "material_code", "rel_type": "REPLACED_BY"}
            ],
        },
    )
```

Produces the following Cypher edge when both fields are non-null:

```cypher
(:LabeledEntity {label: "ProductCatalogueCode", value: "310001"})
  -[:REPLACED_BY]->
(:LabeledEntity {label: "ProductCatalogueCode", value: "310002"})
```

### Rules

- The relationship is only created when **both** source and target fields are non-null in
  the extraction result.
- Both fields must be at the **same nesting level** (siblings in the same model class).
- `to_field` must refer to a field that exists in the same model and carries an
  `entity_label`.
- `rel_type` must be `UPPER_SNAKE_CASE`.

---

## 8. `instance_relationships` Syntax

`instance_relationships` is declared inside `json_schema_extra` on a model field. It
instructs the graph mapper to create a directed Neo4j edge between the `:ModelInstance`
node of the current model and a `:ModelInstance` node of a **target model**, which may
live in a completely different document section or even a different document.

### When to use

Use `instance_relationships` when:
- Two extraction models reference the same real-world entity but are extracted from
  **different structural sections** (different `StructureNode` nodes in Neo4j).
- You need a **forward reference** — model A mentions IDs that will be fully defined
  by model B in a sibling or parent section extracted later.
- You want to build a **cross-document knowledge graph** by linking the same logical
  entity as it appears across multiple ingested files.

Use `field_relationships` (§7) instead when both entities are extracted from the same
model instance (same structural section).

### Format

```python
json_schema_extra={
    "instance_relationships": [
        {
            "target_model": "TargetModelClassName",   # string — PascalCase class name
            "join_via": {
                "local_field_name": "remote_field_name",   # scalar sibling → target key field
                "list_field_name":  "remote_key_field",    # list field (fan-out) → target key field
            },
            "rel_type": "RELATIONSHIP_TYPE",              # UPPER_SNAKE_CASE
        }
    ]
}
```

**`join_via` key rules:**
- Each entry maps a **local field name** (from the same model) to a **remote field name** (from the target model that must be marked `instance_key: True`).
- If a local field is the **same field** that carries `instance_relationships` AND it is a `list[str]`, it acts as the **fan-out key**: one target `ModelInstance` is created per list item.
- All other entries in `join_via` are **fixed keys**: scalar values from sibling fields of the same model, narrowing the scope of the target instance.

### `instance_key: True` — mandatory on the target model

Every model that appears as a `target_model` in an `instance_relationships` declaration
**must** mark its identifier fields with `instance_key: True` in `json_schema_extra`.
Without this, the engine cannot compute a deterministic UID and the shell node will not
merge with the target when it is later extracted.

```python
class TargetModel(ExtractionModel):
    """..."""

    target_key_field: str = Field(
        ...,
        json_schema_extra={
            "entity_label": "TargetLabel",
            "instance_key": True,          # ← required for cross-model resolution
        },
    )
    anchor_field: str = Field(
        ...,
        json_schema_extra={"instance_key": True},   # ← part of composite key
    )
```

When multiple fields are marked `instance_key: True`, the UID is the combination of all
of them (e.g. `Fee` is uniquely identified by `country_code + procedure_type + role`).

### Example 1 — Fan-out with a fixed anchor (one-to-many)

A `VariationCodeModel` lists the IDs of related conditions (`condition_ids`). Each ID
references a separate `ConditionModel` instance that will be extracted from a sibling
section. The `root_variation_code` scopes the condition to the correct parent code.

```python
# Source model
condition_ids: list[str] = Field(
    default_factory=list,
    description=(
        "Identifiers of conditions for this code (e.g. ['1', '2', 'A']). "
        "Each ID references a ConditionModel in a sibling section."
    ),
    json_schema_extra={
        "instance_relationships": [
            {
                "target_model": "ConditionModel",
                "join_via": {
                    "root_variation_code": "variation_code",  # fixed: scopes to parent code
                    "condition_ids": "condition_id",          # fan-out: one target per list item
                },
                "rel_type": "HAS_CONDITION",
            }
        ]
    },
)
```

For a `VariationCodeModel` with `root_variation_code="Q.I.a.1"` and
`condition_ids=["1", "2"]`, the engine produces:

```cypher
// Shell nodes (created immediately, populated when ConditionModel is extracted)
MERGE (:ModelInstance {model_class: "ConditionModel", variation_code: "q.i.a.1", condition_id: "1"})
MERGE (:ModelInstance {model_class: "ConditionModel", variation_code: "q.i.a.1", condition_id: "2"})

// Typed edges
(variationCodeMI)-[:HAS_CONDITION]->(conditionMI_1)
(variationCodeMI)-[:HAS_CONDITION]->(conditionMI_2)
```

```python
# Target model — must mark its key fields with instance_key: True
class ConditionModel(ExtractionModel):
    """A specific condition associated with a variation."""

    variation_code: str = Field(
        ...,
        json_schema_extra={"entity_label": "VariationCode", "instance_key": True},
    )
    condition_id: str = Field(
        ...,
        json_schema_extra={"instance_key": True},
    )
    description: str = Field(..., description="Verbatim text of the condition.")
```

### Example 2 — Simple scalar join (one-to-one)

A `VariationCodeModel` links to one `ProcedureTypeModel` via a single scalar field.
The field simultaneously creates a `:LabeledEntity` node (via `entity_label`) **and**
a cross-model `ModelInstance` link (via `instance_relationships`).

```python
procedure_type: str = Field(
    default="",
    description="Procedure type code: IA, IAIN, IB, II, A, or BA.",
    json_schema_extra={
        "entity_label": "ProcedureType",        # creates a :LabeledEntity node
        "instance_relationships": [
            {
                "target_model": "ProcedureTypeModel",
                "join_via": {
                    "procedure_type": "procedure_type",  # local → remote key field
                },
                "rel_type": "HAS_PROCEDURE_TYPE",
            }
        ],
    },
)
```

### Example 3 — Multiple relationships on the same field

A `BPGRecommendationModel` links to variation code models via two relationship types
simultaneously — one by exact code (`variation_code`) and one by root code
(`root_variation_code`). Both relationships target the same `VariationCodeModel` class.

```python
variation_codes_referenced: list[str] = Field(
    default_factory=list,
    description="Variation codes this recommendation applies to (e.g. ['Q.I.a.1(a)', 'B.II.b.1']).",
    json_schema_extra={
        "entity_label": "VariationCode",
        "instance_relationships": [
            {
                "target_model": "VariationCodeModel",
                "join_via": {
                    "variation_codes_referenced": "variation_code",
                },
                "rel_type": "BPG_MENTIONS_VARIATION_CODE",
            },
            {
                "target_model": "VariationCodeModel",
                "join_via": {
                    "variation_codes_referenced": "root_variation_code",
                },
                "rel_type": "BPG_MENTIONS_ROOT_VARIATION_CODE",
            },
        ],
    },
)
```

### Example 4 — Composite key with multiple `instance_key` fields

When a model is uniquely identified by a combination of fields (e.g. `Fee` is identified
by `country_code + procedure_type + role`), mark ALL key fields with `instance_key: True`.

```python
class Fee(ExtractionModel):
    """Variation fee for one country + procedure_type + role combination."""

    country_code: str = Field(
        ...,
        json_schema_extra={
            "entity_label": "Country",
            "instance_key": True,   # part 1 of composite key
            "instance_relationships": [
                {"target_model": "Country", "join_via": {"country_code": "country_code"}, "rel_type": "APPLIES_TO_COUNTRY"}
            ],
        },
    )
    procedure_type: str = Field(
        ...,
        json_schema_extra={
            "instance_key": True,   # part 2 of composite key
            "entity_label": "ProcedureType",
            "instance_relationships": [
                {"target_model": "ProcedureTypeModel", "join_via": {"procedure_type": "procedure_type"}, "rel_type": "FEE_APPLIES_TO"}
            ],
        },
    )
    role: str = Field(
        ...,
        json_schema_extra={"entity_label": "FeeRole", "instance_key": True},  # part 3 of composite key
    )
    rate: str = Field(..., description="Fee amount without currency symbol (e.g. '511.29').")
```

### Rules

- `target_model` must be a **string** containing the PascalCase class name of the target model (not the class itself).
- Every field listed as a remote key in `join_via` **must** be marked `instance_key: True` on the target model.
- A **list field** acting as the fan-out key must be the same field that declares `instance_relationships`. One `:ModelInstance` edge is created per list item.
- **Scalar fields** in `join_via` (other than the fan-out field) act as fixed scope keys — they narrow the target to a specific parent context.
- A field may declare **multiple** `instance_relationships` entries (list); each creates a different typed edge.
- A field may simultaneously declare `entity_label` AND `instance_relationships` — this creates both a `:LabeledEntity` singleton (for global dedup) and a `:ModelInstance` relationship edge.
- `rel_type` must be `UPPER_SNAKE_CASE`.
- `instance_relationships` can be placed on both scalar and list fields. On scalar fields, exactly one target instance is created (or zero if the value is empty). On list fields, one target instance is created per non-empty list item.

---

## 9. Entity Label Convention

The `entity_label` value in `json_schema_extra` becomes the Neo4j node label:

```
(:LabeledEntity {label: "Substance", value: "ibuprofen"})
(:LabeledEntity {label: "Facility",  value: "Almirall S.A."})
(:LabeledEntity {label: "CASNumber", value: "15687-27-1"})
```

The graph engine deduplicates by `(label, normalised_value)`. Two extraction results from
different documents that mention the same substance will share one `:LabeledEntity` node
(with `label: "Substance"`), enabling cross-document graph queries.

**Use `entity_label` for:**
- Named real-world entities that may recur across documents (substances, facilities, codes).
- Values stable enough for normalisation (proper nouns, identifiers, country codes).

**Do not use `entity_label` for:**
- Long free-text descriptions unique to each document instance.
- Numeric measurements in context (`"25°C/60% RH"`).
- Boolean flags or status strings.
- Narrative summary paragraphs.

---

## 10. Sub-Theme Catalogs

Nest sub-folders inside a parent theme folder when a domain has clearly differentiated
sub-domains that need distinct model sets. Each sub-folder that contains its own `catalog.py`
is an independent theme that the annotation agent can target directly.

```
models/
└── pharma_operations/
    ├── __init__.py
    ├── catalog.py                  # parent: aggregates all sub-theme models
    ├── product_master/
    │   ├── __init__.py
    │   ├── catalog.py              # sub-theme: product_master
    │   └── models.py
    ├── commercial_sales/
    │   ├── __init__.py
    │   ├── catalog.py              # sub-theme: commercial_sales
    │   └── models.py
    └── batch_manufacturing/
        ├── __init__.py
        ├── catalog.py              # sub-theme: batch_manufacturing
        └── models.py
```

The LLM may classify a node as `pharma_operations/product_master` or simply
`pharma_operations`. If the exact path is not found, `ThemeRegistry` degrades
automatically to the nearest ancestor that does exist:

```
pharma_operations/product_master/legacy  →  not found
pharma_operations/product_master         →  ✓  found — use this
```

**Sub-themes do not inherit from the parent.** If a model defined in the parent is also
relevant in a sub-theme, include it explicitly in the sub-theme's `SELECTABLE_MODELS`.

A parent catalog can aggregate all sub-theme models for cross-domain documents:

```python
# models/pharma_operations/catalog.py
from models.pharma_operations.product_master.models import PharmaceuticalPresentation
from models.pharma_operations.commercial_sales.models import MarketSalesPerformance
from models.pharma_operations.batch_manufacturing.models import ProductionBatchRecord

THEME_DESCRIPTION: str = "..."

SELECTABLE_MODELS: list[type] = [
    PharmaceuticalPresentation,
    MarketSalesPerformance,
    ProductionBatchRecord,
    # ... all sub-theme models
]
```

---

## 11. The `Triple` Fallback Model

`models/default/triple.py` defines a generic RDF-style extractor used when no specific
domain model matches a document section:

```python
class TripleItem(BaseModel):
    """A single subject-predicate-object statement."""
    subject: str
    predicate: str
    object: str

class Triple(BaseModel):
    """Generic RDF-style extraction for content that does not fit a specific domain model."""
    triples: list[TripleItem] = Field(
        ..., description="All subject-predicate-object statements extracted from the content."
    )
```

`Triple` is always available as a safety net. The annotation agent selects it when no
theme-specific model reaches sufficient confidence. It produces a navigable graph even
for sections that lack a dedicated model, ensuring no document content is silently dropped.

---

## 12. Reference Implementations

| Path | What to learn from it |
|---|---|
| `models/default/triple.py` | Minimal two-class model; `BaseModel` instead of `ExtractionModel` |
| `models/default/catalog.py` | Simplest possible `catalog.py` |
| `models/structural_specs/catalog.py` | Single-model theme |
| `models/pharma_operations/product_master/models.py` | Production-quality model with `entity_label`, `field_relationships`, and enums |
| `models/pharma_operations/catalog.py` | Parent catalog aggregating five sub-theme model sets |
| `models/pharmaceutical_quality/base.py` | `ExtractionModel` definition |
| `model-creation/templates/models.py` | Copy-paste template for a new model file |
| `model-creation/templates/catalog.py` | Copy-paste template for a new `catalog.py` |
| `own_models/pharma_regulatory/variation_guidelines/models.py` | `field_relationships` + `instance_relationships` + `instance_key` on the same field; fan-out with fixed anchor (`condition_ids`, `document_ids`) |
| `own_models/pharma_regulatory/bpg/models.py` | Multiple `instance_relationships` on a single list field; dual `entity_label` + `instance_relationships` pattern |
| `own_models/pharma_regulatory/fees/models.py` | Composite `instance_key` across three fields (`country_code`, `procedure_type`, `role`) |

---

## 13. Integration Checklist

Use this checklist before merging a new extraction domain.

### Files

- [ ] `models/my_domain/__init__.py` exists (may be empty)
- [ ] `models/my_domain/catalog.py` exists with both `THEME_DESCRIPTION` and `SELECTABLE_MODELS`
- [ ] All model classes are in files under `models/my_domain/`

### Model quality

- [ ] All model classes inherit from `ExtractionModel`
- [ ] Every class has a docstring (first line ≤ 15 words, in English)
- [ ] Every field has `Field(description=...)` with a description ≥ 15 words
- [ ] Every optional field uses `default=None` explicitly
- [ ] Every list field uses `default_factory=list` (never `default=[]`)
- [ ] `entity_label` added to all fields representing real-world named entities
- [ ] `field_relationships` defined wherever a directed edge between two entities is needed
- [ ] `instance_key: True` set on all fields that form the composite key of any model that is a `target_model` in another model's `instance_relationships`
- [ ] `instance_relationships` defined on any field that references an entity extracted from a **different** structural section or document
- [ ] For list fields with `instance_relationships`, the fan-out field name appears as a key in `join_via` mapping to the correct remote key field of the target model

### Theme registration

- [ ] `THEME_DESCRIPTION` is specific and technically distinguishable from existing themes
- [ ] `SELECTABLE_MODELS` includes all top-level models (sub-models that are only embedded fields do not need to be listed)
- [ ] Auto-discovery verified:
  ```bash
  python -c "from utils.theme_registry import ThemeRegistry; r = ThemeRegistry(); print(r.list_themes())"
  ```
- [ ] No import errors when loading `catalog.py`

### Validation

- [ ] At least one real document processed end-to-end (Stage 3 + Stage 4)
- [ ] Annotation agent selects the new models with `confidence >= "medium"` for target sections
- [ ] Extracted fields are coherent with document content
- [ ] Neo4j entity nodes are correct; no unexpected duplicates
- [ ] If `coverage_gaps` appear frequently, consider adding fields to cover missing information

---

## 14. User Theme Structure (external packages)

This section is for **user-defined models loaded from outside the `scinr` package**
via `extra_models_paths`. The rules here differ from §3–§4 because your code lives in a
separate directory that is dynamically added to Python's import machinery.

### 14.1 Required directory layout

```
my_models/                        ← package root (pass this path to extra_models_paths)
  __init__.py                     ← REQUIRED: marks this as a Python package
  my_theme/
    __init__.py                   ← REQUIRED: marks this as a sub-package
    catalog.py                    ← REQUIRED: defines THEME_DESCRIPTION and SELECTABLE_MODELS
    models.py                     ← your Pydantic models
  my_group/                       ← group without its own catalog.py (optional)
    __init__.py
    my_sub_theme/
      __init__.py
      catalog.py
      models.py
      base.py                     ← shared base models within the sub-theme
```

Every directory that contains Python files **must** have an `__init__.py`, even if it is
empty. Without it, Python does not treat the directory as a package and relative imports
will fail with `ImportError: attempted relative import with no known parent package`.
This applies without exception to every level of nesting — theme folders, sub-theme folders, shared helper folders, and the package root. The first file to create in any new folder is always `__init__.py`.

### 14.2 The golden rule: always use relative imports

When your code lives in an external package, you **must** use relative imports for any
module that belongs to that same package. Using bare (absolute) imports is fragile because
it depends on `sys.path` ordering and can collide with other packages.

```python
# ✅ CORRECT — import a sibling (same directory)
from .models import MyModel
from .base import BaseClass

# ✅ CORRECT — import from the parent directory
from ..shared_base import SharedModel

# ✅ CORRECT — import from two levels up
from ...root_base import RootModel

# ❌ INCORRECT — bare import (only works if the directory is already in sys.path)
from models import MyModel

# ❌ INCORRECT — absolute path using your own package name
from my_models.my_theme.models import MyModel

# ✅ CORRECT — scinr is an installed package; this absolute import always resolves correctly
from scinr.newton.models.base import ExtractionModel
```

**Why does this matter?**

> In Java, `package com.empresa.modelos;` tells the compiler where a class lives. The
> Python equivalent is the leading `.` in `from .models import`: it means "look for
> `models.py` in the same package as the current file". Without the dot, Python searches
> the entire `sys.path` — fragile and prone to name collisions when two packages define a
> module with the same name.

The only exception is the `scinr` library itself: because it is installed as a
proper package, `from scinr.newton.models.base import ExtractionModel` always resolves
correctly regardless of where your code lives.

### 14.3 Correct `catalog.py` example

```python
"""My custom theme."""
from __future__ import annotations

# ✅ Relative imports — references siblings inside the same package
from .models import MyTopLevelModel, MyDetailModel

THEME_DESCRIPTION: str = (
    "One-line description of what this theme covers. "
    "The LLM uses this to decide which theme to apply to each document section."
)

SELECTABLE_MODELS: list[type] = [
    MyTopLevelModel,
    MyDetailModel,
]
```

### 14.4 Correct `models.py` example

```python
"""Pydantic models for my theme."""
from __future__ import annotations

from pydantic import Field

# ✅ scinr is an installed package — absolute import is correct here
from scinr.newton.models.base import ExtractionModel


class MyDetailModel(ExtractionModel):
    """Short description — appears in the LLM catalogue."""

    field_one: str | None = Field(default=None, description="Description of the field.")
    field_two: int | None = Field(default=None, description="Description of the field.")


class MyTopLevelModel(ExtractionModel):
    """Main model for the theme."""

    name: str | None = Field(default=None, description="Name.")
    details: list[MyDetailModel] = Field(default_factory=list, description="Details.")
```

### 14.5 Sub-theme with a shared base

When a sub-theme defines base classes that other files in the same sub-theme inherit from,
use relative imports throughout:

```python
# my_models/my_group/my_sub_theme/base.py
from __future__ import annotations
from scinr.newton.models.base import ExtractionModel

class MySubThemeBase(ExtractionModel):
    """Shared base for all models in my_sub_theme."""
    source_document: str | None = Field(default=None, description="Source document name.")
```

```python
# my_models/my_group/my_sub_theme/models.py
from __future__ import annotations
from pydantic import Field

from .base import MySubThemeBase   # ✅ relative — same directory


class ConcreteModel(MySubThemeBase):
    """Concrete extraction model."""
    detail: str | None = Field(default=None, description="Detail field.")
```

```python
# my_models/my_group/my_sub_theme/catalog.py
from __future__ import annotations

from .models import ConcreteModel   # ✅ relative — same directory

THEME_DESCRIPTION: str = "Description of my_sub_theme."
SELECTABLE_MODELS: list[type] = [ConcreteModel]
```

### 14.6 How to register user themes

Pass the **root directory** of your package (the folder that contains the top-level
`__init__.py`) to `extra_models_paths`. The discovery engine walks that directory tree and
registers every sub-folder that contains a `catalog.py`.

```python
from scinr import configure

configure(
    llm=my_llm,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    extra_models_paths=["./my_models"],   # root directory of your package
    enabled_user_themes=["my_theme"],     # optional: activate only specific user themes
)
```

`enabled_user_themes` accepts theme paths relative to the package root
(e.g. `"my_theme"`, `"my_group/my_sub_theme"`). Omit it to activate all discovered themes.

### 14.7 Checklist for user themes

- [ ] Package root directory contains `__init__.py`
- [ ] Every sub-directory that contains `.py` files also contains `__init__.py`
- [ ] All internal imports use relative syntax (`from .module import ...`)
- [ ] `catalog.py` uses `from .models import ...` (not bare or absolute imports)
- [ ] `models.py` imports `ExtractionModel` from `scinr.newton.models.base` (absolute — installed package)
- [ ] `extra_models_paths` points to the package **root** directory (not a sub-theme directory)
- [ ] No import errors: `python -c "import my_models.my_theme.catalog"` runs without error

---

## Appendix: Common Errors

| Error | Cause | Fix |
|---|---|---|
| Theme missing from `list_themes()` | No `catalog.py` in the folder, or it has an import error | Create/fix `catalog.py`; check all imports resolve |
| `KeyError: No model registered for 'X'` | Class not reachable by BFS from `SELECTABLE_MODELS` | Add the class directly to `SELECTABLE_MODELS`, or make sure it is a field type of a model that is already listed |
| `extra inputs are not permitted` | LLM produced a field not in the model schema | If the field should exist, add it; otherwise this is correct behavior (`extra="forbid"`) |
| Sub-model never selected by agent | Not in `SELECTABLE_MODELS` | Add it if it should be a primary extraction target |
| `entity_label` node not created | Field value is `None` in the extraction result | Improve the `description` so the LLM knows when and how to populate it |
| `ThemeRegistry: failed to import catalog` | Syntax or import error in `catalog.py` | Fix the error; run `python -c "import models.my_domain.catalog"` to surface it |
| `instance_relationships` target node never merged (shell stays empty) | Target model does not declare `instance_key: True` on its key fields; UID mismatch between shell and real node | Add `instance_key: True` to the correct key fields of the target model |
| Fan-out creates zero target nodes | The list field is empty OR the fan-out field name in `join_via` does not match the actual field name | Check `join_via` key matches the exact Python field name |
| Two models create duplicate `ModelInstance` nodes | `instance_key` fields include a non-normalized value (e.g. mixed case); or a required key field is missing from `join_via` | Normalize values in a `field_validator`; verify all remote key fields are in `join_via` |
