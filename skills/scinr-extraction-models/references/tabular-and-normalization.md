# Tabular pipeline, normalization and validators

## Tabular rules (models used with CSV/XLSX, alone or together with documents)

The tabular pipeline makes 3 LLM calls per sheet (classify, decide model, map columns). It then
builds each row with `model_construct(**kwargs)`, which **skips all Pydantic validation**.
That has these consequences:

1. **Every mappable field must be `str` or `list[str]`** (or `| None`). When several columns
   map to one field, only those types are combined: deduplicated, then `"; "`-joined for `str`
   or appended for `list[str]`. With `int`, `float`, `bool`, `date`, `Enum` or a nested model,
   the last column wins, a warning is logged, and the other columns' data is lost.
2. **`field_validator` / `model_validator` never run** on tabular rows, in any mode. A
   validator passes unit tests (`Model(year="2023")`) but does nothing in production tabular
   ingestion. Don't use one for coercion here.
3. **Key fields need one value.** `instance_key` and `join_via` fields must receive exactly one
   source column. A `"; "`-joined key matches no reference, so links silently fail.

## Normalized models

A normalized model follows the **dual-field pattern**: a raw field holds the value exactly as
the source wrote it, and a nested submodel holds the same information cleaned up and split into
typed parts.

**How it works in the tabular pipeline:** CSV row → column mapping → instance →
`NormalizationEngine` → Neo4j write. The engine works in five steps:

1. It finds fields marked `normalization_model: True`.
2. It reads the `normalization_source_fields` values from the row. If they are all empty, it skips the row.
3. It batches rows by target type (`normalization_batch_size`, default 5).
4. It calls `llm.with_structured_output(...)` and validates the result with `TypeAdapter`.
5. It sets the submodel on the instance.

Identical source values are cached, so 200 rows with the same address cost one LLM call.

**How it works in the documents pipeline:** the engine never runs. The Stage 4 LLM fills the
submodel directly from its `description=`.

**What it adds to the graph:** a child `(:ModelInstance {model_class:"NormalizedAddress"})`
reached through `-[:HAS_NORMALIZED_ADDRESS]->`. It adds structure, not links. To link rows that
spell the same thing differently, put `entity_label` (e.g. `country_code` → `"Country"`) and/or
`instance_key` on the **normalized submodel's** fields.

Rules:

- **Target field:** nullable (`X | None = None`), typed as a Pydantic model (or `list[X]`), with
  `normalization_model: True` and an explicit `normalization_source_fields`.
- **Target model:** inherits `ExtractionModel`, and every field is nullable (the LLM may not
  find every part). Add `entity_label` / `instance_key` on the parts that identify things.
- **Source fields:** existing scalar sibling fields, usually `str`. List several when the
  normalization needs context from more than one column.
- Keep the raw field. It is the provenance, and it is what the column-mapper writes to.

### Getting a typed value anyway: `normalization_model`

Wrap the typed value in a nested submodel and mark the field. The `NormalizationEngine` (tabular
only) calls an LLM with the listed source fields, then runs real Pydantic validation on the result:

```python
class ParsedYear(ExtractionModel):
    """Normalized batch year derived from the raw year string."""
    value: int | None = Field(default=None, description="Parsed four-digit calendar year, e.g. 2023; None when the source has no recognisable year.")

class BatchRecord(ExtractionModel):
    """A single manufacturing batch row from a batch register sheet."""
    year: str | None = Field(default=None, description="Batch year exactly as it appears in the source column, for example '2023' or 'FY23'. None if not stated.")
    year_normalized: ParsedYear | None = Field(
        default=None,
        description="Structured year derived from the raw year column through the tabular LLM normalization step; None until normalized.",
        json_schema_extra={"normalization_model": True, "normalization_source_fields": ["year"]},
    )
```

- **Always** set `normalization_source_fields` explicitly. Leaving it out sends every other
  scalar field of the row to the LLM.
- On by default. Tune with `configure(normalization_enabled=..., normalization_llm=..., normalization_batch_size=...)`.
- The nested submodel can still carry `entity_label` / `instance_key` / relationships.

| Model used with | `normalization_model` keys |
|---|---|
| Tabular only | **Mandatory** on every field that needs normalizing |
| Documents only (Stages 3–4) | Optional: the extraction LLM fills the nested field from `description=` |
| Both | Recommended: mandatory for the tabular half, harmless for documents |

Run `validate_theme.py --tabular` to enforce rule 1 on every selectable non-wrapper model.

## Shared validators (documents pipeline)

Validators **do** run in Stage 4 (`with_structured_output` validates fully). Use a shared base
only when several models need the same normalization:

```python
# my_theme/baseModels.py
import re
from pydantic import field_validator
from scinr.newton.models.base import ExtractionModel

def normalize_code(v: str) -> str:
    v = v.strip().upper()
    return re.sub(r"[\s_/-]", "", v)                    # "i a" → "IA"

class NormalizedBaseModel(ExtractionModel):
    @field_validator("procedure_type", "procedure_types_referenced", mode="before", check_fields=False)
    @classmethod
    def _norm(cls, v):
        if isinstance(v, str):
            return normalize_code(v)
        if isinstance(v, list):
            return [normalize_code(i) if isinstance(i, str) else i for i in v]
        return v
```

- `check_fields=False` is required on validators in a shared base. Without it, any subclass
  that doesn't declare the field raises `PydanticUserError`.
- `mode="before"` normalizes the input before type validation.
- Handle both `str` and `list` inputs.

### OCR fixes

Put deterministic OCR fixes in validators, not in descriptions. Don't rely on the LLM for them:

```python
def normalize_variation_code(v: str) -> str:
    v = re.sub(r"(?<=[A-Za-z])\.(?:1|l)\.", ".I.", v)    # Q.1.a.1 → Q.I.a.1
    v = re.sub(r"(?<=\d)\.([a-zA-Z])$", r"(\1)", v)      # Q.I.a.1.a → Q.I.a.1(a)
    return v
```

For tabular use, these validators won't run. Put them behind a `normalization_model` submodel instead.
