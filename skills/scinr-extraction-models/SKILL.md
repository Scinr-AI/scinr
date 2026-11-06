---
name: scinr-extraction-models
description: Creates, reviews and debugs custom extraction models and themes for the scinr Python SDK (scinr.newton) — Pydantic ExtractionModel classes plus a catalog.py with THEME_DESCRIPTION and SELECTABLE_MODELS, loaded via configure(extra_models_paths=...). Use when the user wants to define what scinr extracts from documents (PDF, DOCX) or spreadsheets (CSV, XLSX), add a domain theme, set entity_label / instance_key / instance_relationships / field_relationships / normalization_model, or fix a model that is never selected, returns empty fields, or whose graph nodes don't link.
license: Apache-2.0
compatibility: Requires Python 3.11+ with the scinr package installed (pip install scinr).
metadata:
  author: scinr-ai
  version: "0.1.0"
  docs: https://scinr-ai.github.io/scinr/user-guides/custom-models/
---

# scinr extraction models

A **theme** is a folder with `catalog.py` (`THEME_DESCRIPTION` + `SELECTABLE_MODELS`) and
`models.py` (Pydantic classes). The pipeline reads these as LLM prompts:

- Stage 1 picks a theme per document section from `THEME_DESCRIPTION`.
- Stage 3 picks a model from that theme's `SELECTABLE_MODELS` using the **class docstrings**.
- Stage 4 fills the model from field `description=` text and writes it to Neo4j as a
  `:ModelInstance`, plus `:LabeledEntity` nodes and edges from `json_schema_extra`.

So docstrings and descriptions are the extraction prompts. Write them for an LLM reader.

## What a model becomes in the graph

Each extracted model is one `(:ModelInstance {model_class, uid, …})` node. Each field becomes one of:

| Field declaration | Graph result |
|---|---|
| plain scalar or `list[str]` | property on the instance node |
| `entity_label: "X"` | `(instance)-[:REFERENCES {field_name}]->(:LabeledEntity {label:"X", value})`. One node per tenant + label + normalized value, shared by every instance that mentions that value. |
| nested submodel or `list[Submodel]` | child `:ModelInstance` linked by `-[:HAS_<FIELD_NAME>]->`. Created automatically, never an embedded map. |
| `instance_key: True` | instance uid = hash(model_class + **all** key values), so the same logical record from any section or document collapses into one node. With no key, every extraction makes a new node. |

### Which relationship to declare

| Scope | Need | Use |
|---|---|---|
| **Within one instance** | Two identifiers in the same record relate (drug *interacts with* drug, code *has child* code) | `field_relationships`: an edge between the two `:LabeledEntity` nodes of sibling fields |
| | The record has a part with its own structure (address, dosage, list of conditions extracted together) | Nested submodel field. The `HAS_<FIELD>` edge is automatic. |
| **Between instances of the same model** | Same record seen in several sections or documents | `instance_key` on its identifying fields (dedup, no edge needed) |
| | Hierarchy or sequence of records of one class (parent/child section, supersedes, previous version) | `instance_relationships` with `target_model` = **its own class**, joining a reference field such as `parent_code` to the key `code` |
| **Between different models** | A record references a record defined elsewhere (a variation → its conditions or procedure type) | `instance_relationships` with `join_via` → target `instance_key` fields. Creates a shell when the target isn't extracted yet. |
| | Entity-level link only | Give both fields the same `entity_label`. Instances then connect through the shared `:LabeledEntity`, with no declaration needed. |

Edges between instances need `instance_key` on the target. Edges between entities need
`entity_label` on both ends. Read [references/graph-annotations.md](references/graph-annotations.md)
before declaring either kind.

### Normalized models

A **normalized model** is a nested submodel that turns a raw source string into structured,
typed fields. Example: `raw_address: str` plus
`normalized_address: NormalizedAddress | None` (`street`, `city`, `country_code`, …).

- Mark the submodel field `normalization_model: True` and set `normalization_source_fields: ["raw_address"]`.
- In the **tabular** pipeline, an LLM step fills it after column mapping, then real Pydantic
  validation runs on the result. That is the only way to get typed or validated values there.
- In **documents** (Stages 3–4), the extraction LLM fills it directly from `description=`.
- The raw field is kept unchanged as provenance.
- It is written as a child `:ModelInstance` (`HAS_NORMALIZED_ADDRESS`). Normalization creates
  no links by itself. Its value is that the clean parts can carry `entity_label` / `instance_key`,
  so rows that wrote the same thing differently (`"123 Main St"` vs `"123 main street"`) dedupe
  and link. Put those keys on the normalized submodel, not on the raw string.

## Workflow

Copy this checklist and tick it off:

- [ ] 1. Clarify scope (see below). Read a real sample document or sheet if one exists.
- [ ] 2. Create the folders, each one starting with an empty `__init__.py`.
- [ ] 3. Write `models.py` from [assets/models.py](assets/models.py).
- [ ] 4. Write `catalog.py` from [assets/catalog.py](assets/catalog.py).
- [ ] 5. Validate, fix and re-run until there are 0 errors:
      `python scripts/validate_theme.py <models_root> [--tabular] --registry`
- [ ] 6. Show the user how to wire it in (see "Wiring") and suggest one end-to-end run on a real file.

### 1. Clarify before writing

Ask about anything you can't infer from the request:

- **Which pipeline**: unstructured (PDF/DOCX, Stages 1–4), tabular (CSV/XLSX), or both.
  This changes the typing rules. If the answer is tabular or both, read
  [references/tabular-and-normalization.md](references/tabular-and-normalization.md) before step 3.
- **Entities**: which values are identifiers that recur across documents (codes, substances,
  countries)? Those become `entity_label`.
- **Links**: does one section refer to things defined in another section or document?
  If so, read [references/graph-annotations.md](references/graph-annotations.md) before step 3.
- **Shape**: does a section hold one instance, or a table/list of many? If many, add a list wrapper.

### 2. Layout

```
<models_root>/                 ← the path you pass to extra_models_paths
└── my_theme/                  ← theme = any folder with catalog.py
    ├── __init__.py            ← required in EVERY folder with .py files
    ├── catalog.py
    ├── models.py
    └── sub_theme/             ← optional: only for document types with incompatible model sets
        ├── __init__.py
        ├── catalog.py
        └── models.py
```

Inside the package, use relative imports: `from .models import X`, `from ..baseModels import Y`.
One dot is the current folder, and each extra dot goes up one level. Never a bare
`from models import X`, and never an absolute import of your own package. The one absolute
import allowed is `from scinr.newton.models.base import ExtractionModel`.
Put shared validators and helpers in a `baseModels.py` at the lowest folder that all the
models using it share. Create one only when several models really share the logic.

### 3. Model rules

- Inherit from `ExtractionModel`, never plain `BaseModel`. Its config matters:
  - `extra="forbid"` rejects hallucinated fields.
  - `str_strip_whitespace=True` strips strings during validation, not in `mode="before"` validators.
  - `use_enum_values=True` stores enums as plain strings, so compare `self.status == "active"`.
  - `validate_assignment=True` re-validates every assignment (see Gotchas).
- **Class docstring**: first line ≤ 15 words, starting with the entity type ("A single…",
  "Use when…"). Where two models could be confused, add `USE THIS MODEL when…` /
  `DO NOT use… — use OtherModel instead` lines below it.
- **Every field** has `description=` of ≥ 15 words answering three questions: what it is, its
  format (with literal examples from the domain), and when it is `None` or empty.
- If a model is usually extracted together with another one, say so in plain prose in its
  docstring ("Typically extracted alongside VariationCodeModel as a complementary model.").
  It's a convention, not a parsed keyword, but it steers the annotation agent's choice of
  `complementary_models`.
- Optional scalar: `str | None = Field(default=None, …)`. Write `X | None`, never `Optional[X]`. Key fields used in
  `instance_key`/`join_via`: `str = Field(default="", …)`. Lists: `default_factory=list`.
- **Keys are only as good as their values.** The uid hashes the validated values after just
  lowercasing, dropping accents and collapsing whitespace, so `"Q.I.a.1"` ≠ `"QIa1"`. When a key
  needs canonicalizing, deriving from another field (`parent_code` from `code`) or splitting
  into composite parts, add a `@model_validator(mode="before")` that runs **one shared helper on
  both the key and every field that references it**. See "Computing keys with `model_validator`"
  in [references/graph-annotations.md](references/graph-annotations.md). This applies to documents only, because validators never run in tabular.
- Enums: `class X(str, Enum)`, with a docstring `Values:` section explaining each member.
- Keep quantities as strings with units (`"100 kg"`). Never convert them to `float`.
- Put `entity_label` only on short, stable identifiers. Never on free text, sentences,
  measurements, booleans or statuses.
- **List wrapper** `XxxModelList`: exactly one field, `list[XxxModel]`, with a docstring
  starting "Use when the section defines TWO OR MORE…" and a line "DO NOT use this model for a
  single Xxx — use XxxModel instead." It inherits `ExtractionModel`, not the
  main model's base. Put both `XxxModelList` and `XxxModel` in `SELECTABLE_MODELS`. Skip the
  wrapper when a section always holds exactly one instance, or when the items are already a
  `list[Sub]` field of a parent model.
- Order in `models.py`: enums → shared submodels → link-target models → main models →
  list wrappers. Declare a class before any class that uses it.

### 4. Catalog rules

- `THEME_DESCRIPTION`: 2–3 sentences. Name the document types and regulatory standards,
  give example entities (`"variation codes (e.g. Q.I.a.1)"`), and say what it is
  *distinct from*. A vague description pulls every section into this theme.
- `SELECTABLE_MODELS`: most specific first. List wrappers, then main models, then supporting models.
- A parent theme folder can have its own `catalog.py` that collects sub-theme models with
  relative imports (`from .variation_guidelines.models import VariationCodeModel`).

## Wiring

```python
from scinr.newton import configure
configure(llm=..., neo4j_user=..., neo4j_password=..., neo4j_database=...,
          extra_models_paths=["/abs/path/to/models_root"])   # or env SCINR_EXTRA_MODELS_PATHS (":"-separated)
```

To test one model on every section without LLM annotation, use
`run_pipeline(..., stages=[..., "annotation", ...], manual=True, model_class="MyModel")`.
To ship a theme as a pip package, use an entry point:
`[project.entry-points."scinr.newton.models"] my_theme = "my_pkg.catalog"`.

## Gotchas

- **Broken import, theme still listed.** If the catalog can't be imported, ThemeRegistry only
  logs a warning. The theme still appears in `get_all_theme_paths()`, with **0 models**. In a
  package layout (folders have `__init__.py`), the folder *above* the top `__init__.py` must
  be on `sys.path`. Run from that folder or set `PYTHONPATH`. `validate_theme.py --registry`
  catches this. Listing the theme paths does not.
- **Spurious override warning.** `ThemeRegistry: user theme 'x' ... overrides a built-in theme`
  is logged for every user theme, even when nothing is overridden. Ignore it unless the name
  really matches a built-in theme (`default`).
- **Unkeyed join target.** If an `instance_relationships` target lacks `instance_key: True` on
  each `join_via` target field, its real instances get random uids and never merge with the
  shell nodes the references created. Nothing errors. The links just never appear.
- **One value per key field.** Never design a key field to hold `"A; B"`. A packed key matches
  no reference. In tabular use, make sure exactly one column maps to each key field. Several
  columns mapped to a `str` field are joined with `"; "`, which breaks the key.
- **Blank keys collapse.** A `None` key value is left out of the hash, but `""` is hashed. So
  every record whose key is `""` merges into **one** node. Make the main key required or derive it.
- **`mode="after"` validators can recurse.** `ExtractionModel` has `validate_assignment=True`, so
  an unconditional `self.x = …` in an `after` validator raises `RecursionError`. Use
  `mode="before"` and edit the input dict instead.
- **Validators don't run in tabular.** Tabular rows are built with `model_construct()`, so
  `field_validator`/`model_validator` never run there, and typed fields (`int`, `Enum`,
  `date`) lose data. See the tabular reference.
- **`join_via` names must match exactly.** Keys and values must be exact Python field names. A
  typo creates zero edges and raises no error.
- **`join_via` must cover the whole key.** Its target fields must be *all* of the target's
  `instance_key` fields. A shell's uid is hashed from the `join_via` values only, while a real
  instance's uid is hashed from all its key fields. Cover only part of the key and the two never merge.
- **The declaring field must be in its own `join_via`.** The field that carries
  `instance_relationships` must appear in `join_via` as the fan-out or single key. If it
  doesn't, the relationship is skipped with only a log warning. When an anchor field in
  `join_via` is empty or `None`, no edge is created.
- **Pin the source fields.** `normalization_model: True` without `normalization_source_fields`
  sends *every* other scalar field to the normalization LLM.

## When to load more

- Fields that link sections or documents, composite keys, fan-out lists, or entity-to-entity
  edges → [references/graph-annotations.md](references/graph-annotations.md)
- Tabular use, typed values from strings, OCR fixes, or shared normalizing validators →
  [references/tabular-and-normalization.md](references/tabular-and-normalization.md)
- What the written graph really looks like (instance_key merges, normalized nodes) →
  https://scinr-ai.github.io/scinr/user-guides/neo4j-graph.md
- Anything not covered here → the full guide at
  https://scinr-ai.github.io/scinr/user-guides/custom-models.md and
  https://scinr-ai.github.io/scinr/user-guides/model-patterns.md
