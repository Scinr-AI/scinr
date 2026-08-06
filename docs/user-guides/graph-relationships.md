# Graph Relationships

Define edges between extracted entities in your Pydantic extraction models. Relationships declared in `json_schema_extra` become Neo4j graph edges during Stage 4 (entity extraction).

---

## Overview

`scinr.newton` builds a Neo4j knowledge graph from extracted entities. Relationships between entities are declared in model field definitions using `json_schema_extra`. Two relationship types exist:

| Type | Connects | Scope | Node Type |
| :--- | :--- | :--- | :--- |
| **`field_relationships`** | Two fields within the same model instance | Intra-model | `:LabeledEntity` — `[:REL_TYPE]`→ `:LabeledEntity` |
| **`instance_relationships`** | Model instances across sections or documents | Cross-model | `:ModelInstance` — `[:REL_TYPE]`→ `:ModelInstance` |

Both are declared inside `json_schema_extra` on the **source field** of the relationship.

```python
from pydantic import Field
from scinr.newton.models.base import ExtractionModel


class Example(ExtractionModel):
    """Example model showing both relationship types."""

    source_field: str = Field(
        ...,
        description="...",
        json_schema_extra={
            "entity_label": "SourceLabel",
            # Level 2: connects two :LabeledEntity nodes in this model
            "field_relationships": [
                {"to_field": "target_field", "rel_type": "RELATED_TO"},
            ],
            # Level 3: connects this :ModelInstance to another :ModelInstance
            "instance_relationships": [
                {
                    "target_model": "OtherModel",
                    "join_via": {"source_field": "key_field"},
                    "rel_type": "LINKS_TO",
                }
            ],
        },
    )
    target_field: str = Field(
        ...,
        description="...",
        json_schema_extra={"entity_label": "TargetLabel"},
    )
```

---

## Decision Matrix

| Question | Answer | Use |
| :--- | :--- | :--- |
| Both fields from the same structural section? | Yes | `field_relationships` |
| Fields from different sections or documents? | Yes | `instance_relationships` |
| Relationship between named entities (`:LabeledEntity`)? | Yes | `field_relationships` |
| Relationship between model records (`:ModelInstance`)? | Yes | `instance_relationships` |
| Need forward reference (target may not exist yet)? | Yes | `instance_relationships` |

---

## Three Levels of Graph Construction

Entity extraction builds the graph in three passes:

| Level | Mechanism | What it creates |
| :--- | :--- | :--- |
| **Level 1** | `entity_label` | `:LabeledEntity` nodes (globally deduplicated by label + normalized value) |
| **Level 2** | `field_relationships` | Edges between `:LabeledEntity` nodes within the same model instance |
| **Level 3** | `instance_relationships` | Edges between `:ModelInstance` nodes across sections/documents |

All three levels work together. A single field can participate in all three simultaneously.

---

## `field_relationships` — Intra-Model Entity Edges

Connects two `:LabeledEntity` nodes within the same extracted model instance. Both fields must be siblings in the same model and both must have `entity_label`.

### Syntax

```python
class VariationLink(ExtractionModel):
    """A parent-child variation code relationship."""

    root_code: str | None = Field(
        default=None,
        description=(
            "Parent variation code (e.g. 'Q.I.a.1'). "
            "None if this is a top-level code with no parent."
        ),
        json_schema_extra={
            "entity_label": "VariationCode",
            "field_relationships": [
                {"to_field": "child_code", "rel_type": "HAS_CHILD_VARIATION"},
            ],
        },
    )
    child_code: str = Field(
        ...,
        description="Child variation code (e.g. 'Q.I.a.1(a)').",
        json_schema_extra={"entity_label": "VariationCode", "instance_key": True},
    )
```

### Resulting Neo4j Graph

```
(:LabeledEntity:VariationCode {value:"Q.I.a.1"})
  -[:HAS_CHILD_VARIATION]->
(:LabeledEntity:VariationCode {value:"Q.I.a.1(a)"})
```

Both entities share the same label (`VariationCode`) but have different values. The relationship connects them directionally.

### Rules

- Both source and target fields **must** have `entity_label` set
- `to_field` must be the **exact Python name** of a sibling field in the same model
- Relationship is only created when **both** fields are non-null
- `rel_type` must be `UPPER_SNAKE_CASE`
- Multiple relationships can be declared on the same field (list of dicts)

### Multiple Relationships from One Field

A single field can declare relationships to multiple sibling fields:

```python
class DrugInteraction(ExtractionModel):
    """A drug-drug interaction record."""

    source_drug: str = Field(
        ...,
        description="The initiating drug in the interaction.",
        json_schema_extra={
            "entity_label": "Drug",
            "field_relationships": [
                {"to_field": "target_drug", "rel_type": "INTERACTS_WITH"},
                {"to_field": "mechanism", "rel_type": "HAS_MECHANISM"},
            ],
        },
    )
    target_drug: str = Field(
        ...,
        description="The affected drug in the interaction.",
        json_schema_extra={"entity_label": "Drug"},
    )
    mechanism: str = Field(
        ...,
        description="Biological mechanism of the interaction.",
        json_schema_extra={"entity_label": "Mechanism"},
    )
    severity: str | None = Field(
        default=None,
        description="Severity level of the interaction.",
    )
```

### Resulting Neo4j Graph

```
(:LabeledEntity:Drug {value:"Warfarin"})
  -[:INTERACTS_WITH]->
(:LabeledEntity:Drug {value:"Amiodarone"})

(:LabeledEntity:Drug {value:"Warfarin"})
  -[:HAS_MECHANISM]->
(:LabeledEntity:Mechanism {value:"CYP2C9 inhibition"})
```

Note: `severity` has no `entity_label`, so it becomes a scalar property on the `:ModelInstance` node — no entity node is created for it.

---

## `instance_relationships` — Cross-Model Record Edges

Connects `:ModelInstance` nodes across different sections or documents. Creates **shell nodes** for targets that have not yet been extracted, which automatically merge with the real nodes when the target model is later extracted.

### Syntax — Simple 1:1

```python
class DocumentReference(ExtractionModel):
    """A reference to a supporting document."""

    reference_id: str = Field(
        ...,
        description="Unique reference identifier for this document.",
        json_schema_extra={"entity_label": "ReferenceID", "instance_key": True},
    )
    target_variation_code: str = Field(
        ...,
        description="The variation code this document supports.",
        json_schema_extra={
            "entity_label": "VariationCode",
            "instance_relationships": [
                {
                    "target_model": "VariationModel",
                    "join_via": {
                        "target_variation_code": "variation_code",
                    },
                    "rel_type": "SUPPORTS_VARIATION",
                }
            ],
        },
    )
```

### Resulting Neo4j Graph

```
(:ModelInstance:DocumentReference {reference_id:"REF-001"})
  -[:SUPPORTS_VARIATION]->
(:ModelInstance:VariationModel {variation_code:"Q.I.a.1"})
```

The source is the `DocumentReference` model instance. The target is a `VariationModel` model instance, identified by its `instance_key` field (`variation_code`).

### Shell Node Mechanics

When the source model is extracted before the target model exists, `scinr` creates a **shell node**: a `:ModelInstance` with only the key fields populated. When the target model is later extracted from its own section, the shell node is found via `MERGE` and enriched with the remaining fields.

```
Step 1 (DocumentReference extracted first):
  (:ModelInstance {uid:"...", model_class:"VariationModel", variation_code:"Q.I.a.1"})
  └─ shell node — only key fields set

Step 2 (VariationModel extracted from its own section):
  (:ModelInstance {uid:"...", model_class:"VariationModel",
                   variation_code:"Q.I.a.1",
                   description:"...", procedure_type:"IA"})
  └─ shell merged with real data — same uid, all fields populated
```

### Rules

- `target_model` is the **class name** as a string (PascalCase, no module path)
- `join_via` maps local field names → target model's `instance_key` field names
- All fields referenced in `join_via` on the target model **must** have `instance_key: True`
- Creates shell nodes for targets not yet extracted
- Shell nodes merge with real nodes when the target model is later extracted
- Field names in `join_via` must **exactly** match the Python field names

### Target Model Requirements

The target model must mark its key fields with `instance_key: True`:

```python
class VariationModel(ExtractionModel):
    """A single pharmaceutical variation entry."""

    variation_code: str = Field(
        ...,
        description="The variation code identifier (e.g. 'Q.I.a.1').",
        json_schema_extra={
            "entity_label": "VariationCode",
            "instance_key": True,  # ← REQUIRED for instance_relationships resolution
        },
    )
    description: str = Field(
        ...,
        description="Full description of the variation as stated in the document.",
    )
    procedure_type: str | None = Field(
        default=None,
        description="Procedure type code: IA, IAIN, IB, II, A, or BA.",
    )
```

Without `instance_key: True`, shell nodes created by `instance_relationships` will never merge with the real nodes when the target model is extracted.

---

## Fan-Out Pattern (One-to-Many via List)

When the source field is a `list[str]`, one target `:ModelInstance` is created per list item. This is the primary pattern for one-to-many relationships.

### Syntax

```python
class ConditionGroup(ExtractionModel):
    """A group of conditions for a variation."""

    variation_code: str = Field(
        ...,
        description="Parent variation code that scopes these conditions.",
        json_schema_extra={"entity_label": "VariationCode", "instance_key": True},
    )
    condition_ids: list[str] = Field(
        default_factory=list,
        description="IDs of associated conditions (e.g. ['1', '2', 'A']).",
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "ConditionModel",
                    "join_via": {
                        "variation_code": "variation_code",   # fixed anchor key
                        "condition_ids": "condition_id",      # fan-out key
                    },
                    "rel_type": "HAS_CONDITION",
                }
            ],
        },
    )
```

### Resulting Neo4j Graph

```
(:ModelInstance:ConditionGroup {variation_code:"Q.I.a.1"})
  -[:HAS_CONDITION]->
(:ModelInstance:ConditionModel {variation_code:"Q.I.a.1", condition_id:"1"})

(:ModelInstance:ConditionGroup {variation_code:"Q.I.a.1"})
  -[:HAS_CONDITION]->
(:ModelInstance:ConditionModel {variation_code:"Q.I.a.1", condition_id:"2"})

(:ModelInstance:ConditionGroup {variation_code:"Q.I.a.1"})
  -[:HAS_CONDITION]->
(:ModelInstance:ConditionModel {variation_code:"Q.I.a.1", condition_id:"A"})
```

One `ConditionGroup` produces three `:HAS_CONDITION` edges to three different `ConditionModel` shell nodes.

### Rules

- **Fixed keys**: scalar fields in `join_via` that are NOT the annotated field itself. These scope the target to the correct parent context.
- **Fan-out key**: the annotated list field itself. One target `:ModelInstance` is created per list item.
- Field names in `join_via` must **exactly** match Python field names
- If a fixed key field is `None` or empty string, **no relationships** are created for that instance (logged as warning)

### Composite Fan-Out with Multiple Fixed Keys

```python
class FeeSchedule(ExtractionModel):
    """A fee schedule for regulatory procedures."""

    country_code: str = Field(
        ...,
        description="ISO country code (e.g. 'BE', 'DE', 'FR').",
        json_schema_extra={"entity_label": "Country", "instance_key": True},
    )
    procedure_type: str = Field(
        ...,
        description="Procedure type: IA, IB, II, IAIN, A, or BA.",
        json_schema_extra={"entity_label": "ProcedureType", "instance_key": True},
    )
    fee_roles: list[str] = Field(
        default_factory=list,
        description="Fee roles applicable for this country/procedure combination.",
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "FeeModel",
                    "join_via": {
                        "country_code": "country_code",       # fixed key 1
                        "procedure_type": "procedure_type",   # fixed key 2
                        "fee_roles": "role",                  # fan-out key
                    },
                    "rel_type": "HAS_FEE",
                }
            ],
        },
    )
```

Each `FeeModel` target is identified by a composite key: `(country_code, procedure_type, role)`. The fan-out creates one edge per role.

---

## Dual Pattern: `entity_label` + `instance_relationships`

A field can simultaneously create a `:LabeledEntity` global singleton **and** a `:ModelInstance` cross-model edge. Use this when the value is both a named entity (needs global dedup) AND points to a structured model instance (needs cross-document linking).

### Syntax

```python
class VariationRecord(ExtractionModel):
    """A variation with a linked procedure type."""

    variation_code: str = Field(
        ...,
        description="Variation code identifier.",
        json_schema_extra={"entity_label": "VariationCode", "instance_key": True},
    )
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

### Resulting Neo4j Graph

```
Level 1 (entity_label):
  (:ModelInstance:VariationRecord)-[:REFERENCES]->(:LabeledEntity:ProcedureType {value:"IA"})

Level 3 (instance_relationships):
  (:ModelInstance:VariationRecord)-[:HAS_PROCEDURE_TYPE]->(:ModelInstance:ProcedureTypeModel {procedure_type:"IA"})
```

The same field value ("IA") produces:
1. A `:LabeledEntity:ProcedureType` node (globally deduplicated — all "IA" values across all documents point to the same node)
2. A `:HAS_PROCEDURE_TYPE` edge from the `VariationRecord` instance to a `ProcedureTypeModel` instance

### When to Use

Use the dual pattern when:
- The value is a stable identifier that appears across many documents (justifies `entity_label`)
- The value also represents a structured concept with its own model (justifies `instance_relationships`)
- You want both cross-document entity deduplication AND cross-model structural linking

---

## Relationship Type Naming

### Conventions

- Always `UPPER_SNAKE_CASE`
- Descriptive direction: the relationship type reads naturally from source to target
- Consistent across models: the same relationship type always has the same semantic meaning

### Good Examples

| Relationship Type | Reads as |
| :--- | :--- |
| `HAS_CHILD_VARIATION` | parent HAS_CHILD_VARIATION child |
| `SUPPORTS_VARIATION` | document SUPPORTS_VARIATION variation |
| `HAS_CONDITION` | variation HAS_CONDITION condition |
| `BELONGS_TO` | child BELONGS_TO parent |
| `NORMALIZES` | raw value NORMALIZES canonical value |
| `INTERACTS_WITH` | drug A INTERACTS_WITH drug B |
| `HAS_FEE` | schedule HAS_FEE fee entry |
| `HAS_PROCEDURE_TYPE` | record HAS_PROCEDURE_TYPE procedure type |

### Bad Examples

| Relationship Type | Problem |
| :--- | :--- |
| `rel1` | Not descriptive |
| `has_child` | Not UPPER_SNAKE_CASE |
| `variation_to_condition` | Not UPPER_SNAKE_CASE |
| `LINKS` | Too generic — what kind of link? |

---

## Common Patterns — Complete Examples

### Pattern 1: Simple `field_relationships` (Two Entities in Same Model)

Two named entities in the same model, connected by a directed edge:

```python
class IngredientPair(ExtractionModel):
    """An active ingredient and its excipient."""

    active_ingredient: str = Field(
        ...,
        description=(
            "Name of the active pharmaceutical ingredient. "
            "Extract the INN or official name without modifications."
        ),
        json_schema_extra={
            "entity_label": "Substance",
            "field_relationships": [
                {"to_field": "excipient", "rel_type": "HAS_EXCIPIENT"},
            ],
        },
    )
    excipient: str = Field(
        ...,
        description=(
            "Name of the excipient used with the active ingredient. "
            "Extract the full chemical or trade name."
        ),
        json_schema_extra={"entity_label": "Substance"},
    )
    concentration: str | None = Field(
        default=None,
        description=(
            "Concentration of the active ingredient in the formulation "
            "(e.g. '50 mg/g', '10% w/w'). None if not stated."
        ),
    )
```

**Graph:**

```
(:LabeledEntity:Substance {value:"Metformin"})
  -[:HAS_EXCIPIENT]->
(:LabeledEntity:Substance {value:"Microcrystalline cellulose"})
```

---

### Pattern 2: Simple `instance_relationships` (Cross-Model Join)

A document reference linking to a variation defined in another section:

```python
class SupportingDocument(ExtractionModel):
    """A reference to a document supporting a variation."""

    doc_title: str = Field(
        ...,
        description="Title of the supporting document as stated in the text.",
        json_schema_extra={"entity_label": "DocumentTitle"},
    )
    linked_variation: str = Field(
        ...,
        description=(
            "Variation code this document supports (e.g. 'B.II.b.1'). "
            "This code links to a VariationModel in another section."
        ),
        json_schema_extra={
            "entity_label": "VariationCode",
            "instance_relationships": [
                {
                    "target_model": "VariationModel",
                    "join_via": {
                        "linked_variation": "variation_code",
                    },
                    "rel_type": "SUPPORTS_VARIATION",
                }
            ],
        },
    )
```

**Graph:**

```
(:ModelInstance:SupportingDocument {doc_title:"Stability Study"})
  -[:SUPPORTS_VARIATION]->
(:ModelInstance:VariationModel {variation_code:"B.II.b.1"})
```

---

### Pattern 3: Fan-Out (One-to-Many via List)

A variation code with multiple associated conditions:

```python
class VariationWithConditions(ExtractionModel):
    """A variation code with its applicable conditions."""

    variation_code: str = Field(
        ...,
        description=(
            "Full variation code (e.g. 'Q.I.a.1'). "
            "Always include the top-level prefix."
        ),
        json_schema_extra={"entity_label": "VariationCode", "instance_key": True},
    )
    condition_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Condition numbers applicable to this variation "
            "(e.g. ['1', '2', '3']). Empty list if none."
        ),
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "ConditionModel",
                    "join_via": {
                        "variation_code": "variation_code",
                        "condition_ids": "condition_number",
                    },
                    "rel_type": "HAS_CONDITION",
                }
            ],
        },
    )
```

**Graph:**

```
(:ModelInstance:VariationWithConditions {variation_code:"Q.I.a.1"})
  -[:HAS_CONDITION]->
(:ModelInstance:ConditionModel {variation_code:"Q.I.a.1", condition_number:"1"})

(:ModelInstance:VariationWithConditions {variation_code:"Q.I.a.1"})
  -[:HAS_CONDITION]->
(:ModelInstance:ConditionModel {variation_code:"Q.I.a.1", condition_number:"2"})

(:ModelInstance:VariationWithConditions {variation_code:"Q.I.a.1"})
  -[:HAS_CONDITION]->
(:ModelInstance:ConditionModel {variation_code:"Q.I.a.1", condition_number:"3"})
```

---

### Pattern 4: Dual Pattern (Entity + Instance)

A procedure type that is both a named entity and a model instance:

```python
class ProcedureRecord(ExtractionModel):
    """A regulatory procedure with its type classification."""

    procedure_name: str = Field(
        ...,
        description="Name of the procedure as it appears in the document.",
    )
    procedure_type: str = Field(
        ...,
        description="Procedure type code: IA, IAIN, IB, II, A, or BA.",
        json_schema_extra={
            "entity_label": "ProcedureType",
            "instance_relationships": [
                {
                    "target_model": "ProcedureTypeModel",
                    "join_via": {"procedure_type": "procedure_type"},
                    "rel_type": "HAS_PROCEDURE_TYPE",
                }
            ],
        },
    )
```

**Graph:**

```
Level 1: (:ModelInstance:ProcedureRecord)-[:REFERENCES]->(:LabeledEntity:ProcedureType {value:"IA"})
Level 3: (:ModelInstance:ProcedureRecord)-[:HAS_PROCEDURE_TYPE]->(:ModelInstance:ProcedureTypeModel {procedure_type:"IA"})
```

---

### Pattern 5: Multi-Hop Chain (A → B → C)

Build multi-hop graph chains by combining `field_relationships` and `instance_relationships` across models:

```python
# Step 1: field_relationships creates entity-to-entity edges
class SubstanceRoute(ExtractionModel):
    """A substance and its route of administration."""

    substance_name: str = Field(
        ...,
        description="Official name of the substance.",
        json_schema_extra={
            "entity_label": "Substance",
            "field_relationships": [
                {"to_field": "route", "rel_type": "ADMINISTERED_VIA"},
            ],
        },
    )
    route: str = Field(
        ...,
        description="Route of administration (e.g. 'oral', 'intravenous', 'topical').",
        json_schema_extra={"entity_label": "Route"},
    )


# Step 2: instance_relationships links SubstanceRoute to a DosageModel
class DosageModel(ExtractionModel):
    """A dosage specification for a substance."""

    substance_name: str = Field(
        ...,
        description="Substance this dosage applies to.",
        json_schema_extra={"entity_label": "Substance", "instance_key": True},
    )
    dosage: str = Field(
        ...,
        description="Dosage instruction (e.g. '500 mg twice daily').",
    )


# Step 3: SubstanceRoute links to DosageModel via instance_relationships
class SubstanceRoute(ExtractionModel):
    """A substance and its route of administration."""

    substance_name: str = Field(
        ...,
        description="Official name of the substance.",
        json_schema_extra={
            "entity_label": "Substance",
            "field_relationships": [
                {"to_field": "route", "rel_type": "ADMINISTERED_VIA"},
            ],
            "instance_relationships": [
                {
                    "target_model": "DosageModel",
                    "join_via": {"substance_name": "substance_name"},
                    "rel_type": "HAS_DOSAGE",
                }
            ],
        },
    )
    route: str = Field(
        ...,
        description="Route of administration (e.g. 'oral', 'intravenous', 'topical').",
        json_schema_extra={"entity_label": "Route"},
    )
```

**Graph:**

```
Entity level (field_relationships):
(:LabeledEntity:Substance {value:"Metformin"})
  -[:ADMINISTERED_VIA]->
(:LabeledEntity:Route {value:"oral"})

Instance level (instance_relationships):
(:ModelInstance:SubstanceRoute)-[:HAS_DOSAGE]->(:ModelInstance:DosageModel {substance_name:"Metformin"})

Multi-hop query:
(:LabeledEntity:Substance)-[:ADMINISTERED_VIA]->(:LabeledEntity:Route)
(:ModelInstance:SubstanceRoute)-[:HAS_DOSAGE]->(:ModelInstance:DosageModel)
```

---

## Anti-Patterns

| Anti-Pattern | Why It Fails | Fix |
| :--- | :--- | :--- |
| `field_relationships` pointing to a field without `entity_label` | The target `:LabeledEntity` node does not exist; Neo4j write is silently skipped | Add `entity_label` to the target field |
| Missing `instance_key: True` on target model key fields | Shell nodes created by `instance_relationships` never merge with real nodes when the target is extracted | Mark ALL key fields on the target model with `instance_key: True` |
| `join_via` field name mismatch | Zero target nodes created — the field name in `join_via` must exactly match the Python field name | Double-check that `join_via` keys use the exact Python field names |
| `entity_label` on free-text narrative fields | Creates meaningless `:LabeledEntity` singletons; degrades cross-document deduplication quality | Only add `entity_label` to short, stable, identifier-like values |
| `target_model` using wrong class name | Shell nodes created with wrong `model_class`; real nodes never merge | Use the exact PascalCase class name as a string |
| `field_relationships` on a field with `None` value | Relationship silently skipped — both source and target must be non-null | Make the field required or handle the `None` case in the description |
| `instance_relationships` on a scalar field without including it in `join_via` | The engine requires the annotated field itself to be in `join_via` as the fan-out key | Always include the annotated field name in `join_via` |

---

## Verification

### Verify Entity Relationships in Neo4j

```cypher
-- List all field_relationships (LabeledEntity → LabeledEntity)
MATCH (a:LabeledEntity)-[r]->(b:LabeledEntity)
RETURN labels(a) AS source,
       type(r) AS relationship,
       labels(b) AS target,
       count(r) AS count
ORDER BY count DESC;
```

### Verify Instance Relationships in Neo4j

```cypher
-- List all instance_relationships (ModelInstance → ModelInstance)
MATCH (a:ModelInstance)-[r]->(b:ModelInstance)
RETURN a.model_class AS source_model,
       type(r) AS relationship,
       b.model_class AS target_model,
       count(r) AS count
ORDER BY count DESC;
```

### Verify a Specific Model's Relationships

```cypher
-- All outgoing relationships from VariationModel instances
MATCH (a:ModelInstance)-[r]->(b)
WHERE a.model_class = 'VariationModel'
RETURN a.model_class AS source,
       a.variation_code AS code,
       type(r) AS relationship,
       labels(b) AS target_labels,
       b.model_class AS target_model
LIMIT 20;
```

### Verify Shell Nodes (Unresolved Targets)

```cypher
-- Find shell nodes that have not been merged with real data
MATCH (m:ModelInstance)
WHERE m.description IS NULL
  AND m.model_class IS NOT NULL
RETURN m.model_class AS model,
       count(m) AS shell_count;
```

### Visualize a Relationship Chain

```cypher
-- Visualize the full chain from entity to instance to instance
MATCH path = (e:LabeledEntity)-[*1..3]->(m:ModelInstance)
WHERE e.label = 'VariationCode'
RETURN path
LIMIT 20;
```

---

## See Also

- **[Custom Models](custom-models.md)** — Defining Pydantic extraction schemas for domain-specific entities.
- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding the overall graph model and node types.
- **[Running the Pipeline](running-pipeline.md)** — Orchestrating the full ingestion pipeline.
- **[Architecture](../architecture.md)** — Detailed pipeline stages and data flow.
