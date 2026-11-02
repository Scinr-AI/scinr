# Graph annotations (`json_schema_extra`)

All graph behaviour is declared per field in `json_schema_extra`. Pick by intent:

| Intent | Key |
|---|---|
| Dedupe a value across all documents as a global node | `entity_label` |
| Give a model instance a stable identity, so references can find it | `instance_key: True` |
| Edge between two entities in the SAME instance | `field_relationships` |
| Edge from this instance to ANOTHER model's instance (any section or document) | `instance_relationships` |

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

### Dual pattern

One field can carry both `entity_label` (global dedup node) and `instance_relationships`
(a link to the structured definition). Use it when a code is both a named entity and points
to a model that defines it.

## Checks that `validate_theme.py` runs

- `join_via` local fields exist on the source model, and target fields exist on the target.
- Every target field is `instance_key`.
- `target_model` names a model in the models root.
- `field_relationships` targets exist and both ends have `entity_label`.
- Every `rel_type` is UPPER_SNAKE_CASE.
- No `instance_key` on list fields.
