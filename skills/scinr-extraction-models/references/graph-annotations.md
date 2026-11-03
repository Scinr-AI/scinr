# Graph annotations (`json_schema_extra`)

All graph behaviour is declared per field in `json_schema_extra`. Pick by intent:

| Intent | Key |
|---|---|
| Dedupe a value across all documents as a global node | `entity_label` |
| Give a model instance a stable identity, so references can find it | `instance_key: True` |
| Edge between two entities in the SAME instance | `field_relationships` |
| Edge from this instance to another instance, of a different model or the same one, in any section or document | `instance_relationships` |
| A structured part of this record | nested submodel field (no key needed) |

## What gets written, per extraction

```
(:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)-[:HAS_<FIELD>]->(:ModelInstance)  # primary + complementary models
(:ModelInstance)-[:HAS_<FIELD> {index}]->(:ModelInstance)       # nested submodels / list items
(:ModelInstance)-[:REFERENCES {field_name}]->(:LabeledEntity)   # entity_label fields
(:LabeledEntity)-[:REL_TYPE]->(:LabeledEntity)                  # field_relationships
(:ModelInstance)-[:REL_TYPE]->(:ModelInstance)                  # instance_relationships
```

## Nested submodels (composition)

A field typed `Submodel | None` or `list[Submodel]` is written as a child `:ModelInstance`
with `-[:HAS_<FIELD_NAME>]->` (list items get an `index` property). Use this for parts that are
extracted **together with** the parent from the same text. Without `instance_key`, each child is
a new node owned by its parent. With `instance_key`, identical children from different parents
merge into one shared node.

Rule of thumb: if the part is defined *in the same section*, nest it. If it is defined
*elsewhere* and only referenced here, use `instance_relationships`.

## `entity_label`

Creates or merges `(:LabeledEntity {label, value})`, keyed by `(label, normalized value)`.
Reuse the same label string across models for the same kind of thing (`"Country"` everywhere).

```python
substance: str = Field(..., description="...", json_schema_extra={"entity_label": "Substance"})
```

## `instance_key`

Fields marked `instance_key` determine the instance uid:
`hash(model_class, {key: normalized value}, tenant)`. The same values give the same node,
whatever the load order. For a composite key, mark **every** part:

```python
class Fee(ExtractionModel):
    country_code:   str = Field(default="", description="...", json_schema_extra={"entity_label": "Country", "instance_key": True})
    procedure_type: str = Field(default="", description="...", json_schema_extra={"entity_label": "ProcedureType", "instance_key": True})
    rate:           str = Field(default="", description="Fee amount without currency symbol ...")
```

Key fields hold exactly one scalar value: never a list, never `"A; B"`.

## `field_relationships`

Both fields need `entity_label`. `to_field` names a sibling field. The edge is created only
when both values are non-null. `rel_type` is UPPER_SNAKE_CASE.

```python
root_code: str | None = Field(default=None, description="...", json_schema_extra={
    "entity_label": "VariationCode",
    "field_relationships": [{"to_field": "child_code", "rel_type": "HAS_CHILD_VARIATION_CODE"}],
})
child_code: str = Field(..., description="...", json_schema_extra={"entity_label": "VariationCode", "instance_key": True})
```

## `instance_relationships`

Declared on the source field. `join_via` maps **local field name → target field name**, and
every target field must be `instance_key` on the target model. If the target hasn't been
extracted yet, a shell `:ModelInstance` is created with the same uid. It merges when the real
instance arrives.

```python
procedure_type: str = Field(default="", description="...", json_schema_extra={
    "instance_relationships": [{
        "target_model": "ProcedureTypeModel",          # class name string
        "join_via": {"procedure_type": "procedure_type"},
        "rel_type": "HAS_PROCEDURE_TYPE",
    }],
})
```

### Fan-out (one edge per list item)

Put the list field itself in `join_via`, and add any scalar "anchor" fields that scope the
target to its parent:

```python
root_variation_code: str = Field(default="", description="...")
condition_ids: list[str] = Field(default_factory=list, description="...", json_schema_extra={
    "instance_relationships": [{
        "target_model": "ConditionModel",
        "join_via": {
            "root_variation_code": "variation_code",   # fixed anchor
            "condition_ids":       "condition_id",     # fan-out: one target per item
        },
        "rel_type": "HAS_CONDITION",
    }],
})

class ConditionModel(ExtractionModel):
    variation_code: str = Field(default="", description="...", json_schema_extra={"instance_key": True})
    condition_id:   str = Field(default="", description="...", json_schema_extra={"instance_key": True})
```

### How the uids line up (why the rules exist)

- A real instance's uid is `hash(model_class, {every instance_key field: normalized value}, tenant)`.
  Normalization lowercases, strips accents and collapses whitespace.
- A shell's uid is `hash(target_model, {join_via target field: normalized local value}, tenant)`.
- These are equal only when the `join_via` target fields are **exactly** the target's
  `instance_key` set. If one of the target's key fields is missing from `join_via`, the shell
  and the real instance stay as two separate nodes.
- The declaring field must itself be a `join_via` key. With a list it fans out, one target per
  item. With a scalar it creates one target. Otherwise the entry is skipped and a warning is logged.
- An empty or `None` anchor field skips the edge for that instance.
- Shells that are never filled in stay in the graph as key-only nodes. They are removed only
  by deletion, freezing, or `collect_orphans()`.

### Computing keys with `model_validator`

The uid is hashed from the instance's attribute values **after** Pydantic validation. The only
normalization the pipeline applies on top is `normalize_key`: NFKD, drop accents, collapse
whitespace, lowercase. Punctuation, prefixes and separators are kept, so `"Q.I.a.1"`,
`"QIa1"` and `"Q.1.a.1"` (OCR) are three different nodes. Anything else that has to match is
the model's job, and a `model_validator` is the place for it when the logic needs more than one
field:

| Use | Example |
|---|---|
| **Canonicalize** a key the same way on both sides of a join | `ItemModel.item_code` and `ConditionModel.item_code` both pass through one `canonical_code()` |
| **Derive** a key or anchor from another field, instead of trusting the LLM to fill it | `parent_code` from `code` (`"3.2.S.2"` → `"3.2.S"`), `root_variation_code` from `variation_code` |
| **Split** one value into the parts of a composite key | `"A-001/2"` → `item_code="A-001"`, `condition_id="2"` |

```python
import re
from typing import Any
from pydantic import Field, model_validator

def canonical_code(v: str) -> str:
    return re.sub(r"\s+", "", v).upper()          # use the SAME helper on every model that joins on it

class SectionSpec(ExtractionModel):
    """A single CTD section specification with its code and title."""

    code: str = Field(default="", description="...", json_schema_extra={"entity_label": "CTDSection", "instance_key": True})
    parent_code: str = Field(default="", description="...", json_schema_extra={
        "instance_relationships": [{"target_model": "SectionSpec", "join_via": {"parent_code": "code"}, "rel_type": "HAS_PARENT_SECTION"}],
    })

    @model_validator(mode="before")
    @classmethod
    def _derive_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        code = canonical_code(data.get("code") or "")
        data["code"] = code
        if code and not data.get("parent_code") and "." in code:
            data["parent_code"] = code.rsplit(".", 1)[0]
        else:
            data["parent_code"] = canonical_code(data.get("parent_code") or "")
        return data
```

Rules:

- **Prefer `mode="before"`.** It edits the input dict, so it runs once. `ExtractionModel` sets
  `validate_assignment=True`, so in `mode="after"` every `self.x = …` re-runs validation,
  including the `after` validator itself. An unconditional assignment there raises
  `RecursionError`. If you must use `after`, assign only when the value actually changes.
- **Normalize the reference too.** A shell's uid is hashed from the *local* field values in
  `join_via`, so the source model (`parent_code`, `condition_ids` items) must go through the same
  helper as the target's key. Fixing only the target leaves the two sides different.
- **Never derive a key from a list or a free-text field**, and never join several values into one
  key (`"A; B"`).
- **Blank keys collapse.** Only `None` key values are left out of the hash. `""` is hashed, so
  every record whose only key is `""` becomes **one** node per tenant. Make the main key required
  (`Field(...)`) or derive it, so a key is never blank.
- **Documents pipeline only.** Validators run in Stage 4 (`with_structured_output` validates
  fully) and on manual `Model(...)` calls. Tabular rows are built with `model_construct()`, so
  no validator runs there. A tabular key has to come out of the columns as-is, or from a
  `normalization_model` submodel (see the tabular reference).
- For a plain single-field cleanup shared across models, a `field_validator(..., check_fields=False)`
  on a shared base is simpler. Use `model_validator` when the value depends on other fields.

### Same model: hierarchy and sequence

`target_model` can be the model's own class. Use this for parent/child records, supersession or
versions where each record is a full instance of one class:

```python
class SectionSpec(ExtractionModel):
    """A single CTD section specification with its code and title."""

    code: str = Field(default="", description="...", json_schema_extra={"entity_label": "CTDSection", "instance_key": True})
    parent_code: str = Field(default="", description="Code of the enclosing section, e.g. '3.2.S' for '3.2.S.2'; empty for top-level sections.", json_schema_extra={
        "instance_relationships": [{
            "target_model": "SectionSpec",
            "join_via": {"parent_code": "code"},
            "rel_type": "HAS_PARENT_SECTION",
        }],
    })
```

If only the *identifiers* relate, and there is no record behind the parent, use
`field_relationships` between two `entity_label` fields instead (`HAS_CHILD_VARIATION_CODE` above).

### Dual pattern

One field can carry both `entity_label` (global dedup node) and `instance_relationships`
(a link to the structured definition). Use it when a code is both a named entity and points
to a model that defines it.

## Implicit links through shared entities

Two instances that reference the same `:LabeledEntity` are already connected:
`(a)-[:REFERENCES]->(le)<-[:REFERENCES]-(b)`. This happens even across models and documents,
as long as both fields use the **same label string** (e.g. `"Substance"` everywhere). Rely on it
for "mentions the same thing" questions. Declare `instance_relationships` only when the
relationship itself has meaning (`HAS_CONDITION`, `SUPERSEDES`).

## Checks that `validate_theme.py` runs

- `join_via` local fields exist on the source model, and target fields exist on the target.
- Every target field is `instance_key`, and together they cover the target's whole key.
- The declaring field appears in its own `join_via`.
- `target_model` names a model in the models root.
- `field_relationships` targets exist and both ends have `entity_label`.
- Every `rel_type` is UPPER_SNAKE_CASE.
- No `instance_key` on list fields.
