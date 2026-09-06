# Neo4j Graph Model

`scinr` builds a connected graph representation of input documents and extracted domain concepts inside Neo4j. This guide covers every node type, relationship, and query pattern produced by the `scinr.newton` extraction engine.

The graph serves as the primary output store. After a document passes through the Newton pipeline, all structural elements and extracted entities are persisted as interconnected nodes and relationships, enabling complex cross-document queries that would be difficult or impossible with a flat relational schema.

Two distinct pipelines produce different graph structures:

- **Unstructured pipeline** — processes PDFs, Word documents, and other narrative sources. Produces a hierarchy of headings, paragraphs, tables, and figures, with entities extracted from specific sections.
- **Tabular pipeline** — processes CSV, Excel, and other structured tables. Produces one model instance per row, with labeled entities for stable, deduplicated fields.

---

## Node Types

### `:Document`

Represents the original input file or folder that was ingested into the system. Every document in the graph is a `:Document` node.

```
(:Document {
  name: "clinical_trial_report",
  path: "/path/to/clinical_trial_report",
  version: 1,
  load_date: 2024-01-15T10:30:00,
  latest: true,
  is_folder: false,
  tenant_id: "acme-corp",
  created_by_user_id: "user-42",
  job_id: "ingest-2026-09-06-a"
})
```

| Property | Type | Description |
|---|---|---|
| `name` | String | Name of the file or folder as provided at ingestion time. Without extension. |
| `path` | String | Full path identifier for this document within the ingestion hierarchy. Without extension. |
| `version` | Integer | Version number. Starts at 1; increments on replacement or update. |
| `load_date` | DateTime | Timestamp when the document was first ingested. |
| `latest` | Boolean | Indicates if this is the latest version of the document. |
| `is_folder` | Boolean | Indicates if this node represents a folder (true) vs a file (false). |
| `raw_file_id` | String | Storage-backend id of the stored raw file (empty for folders / when no storage backend is configured). |
| `context_instructions` | String \| null | Free-text ingestion context passed via `run_pipeline(context_instructions=...)`; `null` when not supplied. |
| `tenant_id` | String \| null | Caller-supplied multi-tenant owner id from `run_pipeline(tenant_id=...)`. Always set (`null` when not supplied). Written on leaf **and** folder-parent nodes. |
| `created_by_user_id` | String \| null | Caller-supplied id of the user that launched the ingestion, from `run_pipeline(created_by_user_id=...)`. Always set (`null` when not supplied). |
| `job_id` | String \| null | Caller-supplied ingestion job/run id from `run_pipeline(job_id=...)`. Always set (`null` when not supplied). Usable as a bulk-delete selector — see [Document Deletion](document-deletion.md). |

---

### `:StructureNode`

Represents a structural element within a document — a section, subsection, freeform block, table, field group, appendix, or row. Structure nodes form a tree rooted at the document.

```
(:StructureNode:Section {
  id: "M3- Notice to applicant::1::1-foreword",
  role: "section",
  title: "Foreword",
  text: "The introduction text...",
  page_number: 1,
  hierarchy_level: 1,
  appearance_order: 1,
  latest: true
})
```

| Property | Type | Description |
|---|---|---|
| `id` | String | Unique hierarchical identifier for this structure node. |
| `role` | String | One of: `section`, `subsection`, `freeform_block`, `table`, `field_group`, `appendix`, `row`. |
| `title` | String | Heading text, if this node represents a section heading. Null for leaf nodes like paragraphs. |
| `text` | String | The raw text content of this structural element. |
| `page_number` | Integer | Source page number, when available (PDF, paginated documents). Null for non-paginated sources. |
| `hierarchy_level` | Integer | Nesting depth in the document outline. Root headings are level 1, sub-sections increment from there. |
| `appearance_order` | Integer | Order of appearance within the sibling nodes at the same hierarchy level. |
| `latest` | Boolean | Indicates if this is the latest version of the structure node. |

**Key characteristics:** Unlike `:ModelInstance` and `:LabeledEntity` (which use a single label plus a type property), `:StructureNode` genuinely carries a **second dynamic label** derived from its `role` property, applied via `SET n:{RoleLabel}` at ingestion time (see `ingest/nodes.py`). The mapping is deterministic PascalCase-of-role:

| `role` (property) | Second label |
|---|---|
| `section` | `:Section` |
| `subsection` | `:Subsection` |
| `freeform_block` | `:FreeformBlock` |
| `table` | `:Table` |
| `field_group` | `:FieldGroup` |
| `appendix` | `:Appendix` |
| `row` | `:Row` |

So a real node looks like `(:StructureNode:Section {role: "section", ...})` — both the base label and the role-specific label coexist. The `role` property remains the recommended way to filter in Cypher (`WHERE s.role = "table"`), since it is indexed (`idx_structure_node_role`) and avoids needing to know the exact label spelling, but `MATCH (s:Table)` also works and is sometimes faster for label-scan-heavy queries.

---

### `:LabeledEntity`

Represents a stable, deduplicated entity value extracted from a field marked with an `entity_label` in the schema. Labeled entities are created for fields that act as identifiers or cross-references — drug names, product codes, anatomical terms, etc.

```
(:LabeledEntity {
  label: "ActiveSubstance",
  value: "Metformin",
  uid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  normalized_value: "metformin"
})
```

| Property | Type | Description |
|---|---|---|
| `label` | String | The entity label name (e.g., `ActiveSubstance`, `AnatomicalStructure`, `ProductCode`). |
| `value` | String | The canonical entity value. Used for deduplication across documents. |
| `uid` | String | Unique identifier for this specific entity node. |
| `normalized_value` | String | Normalized version of the value for case-insensitive matching. |

Key characteristics:

- **Single label**: The node has a single `:LabeledEntity` label and uses a `label` property to store the entity category.
- **Deduplication**: If the same `value` is extracted from multiple documents, a single `:LabeledEntity` node is reused. This enables cross-document entity matching and aggregation.
- **Stable identity**: The `uid` remains constant for a given value, allowing reliable joins and traces.

---

### `:ModelInstance`

Represents a single extracted entity record — one instance of a model class defined in the extraction schema. Each model instance stores all field values as node properties.

```
(:ModelInstance {
  model_class: "AdverseEventModel",
  uid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  event_type: "Headache",
  severity: "Mild",
  occurrence_count: 3
})
```

| Property | Type | Description |
|---|---|---|
| `model_class` | String | The model class name (e.g., `AdverseEventModel`, `ProductRecord`, `LaboratoryResult`). |
| `uid` | String | Unique identifier for this model instance. |
| *(dynamic)* | Varies | All model field values are stored as additional properties on the node. Property names match the field names defined in the schema. |

Key characteristics:

- **Single label**: The node has a single `:ModelInstance` label and uses a `model_class` property to store the model type.
- **One node per record**: Each extracted entity produces exactly one model instance node. In the tabular pipeline, this corresponds to one row per node.
- **Field values as properties**: Every field defined in the model schema becomes a property on the node, enabling direct Cypher filtering without additional joins.

---

## Relationship Types

### Document Structure

These relationships form the hierarchical tree of a document.

| Relationship | From | To | Description |
|---|---|---|---|
| `[:HAS_STRUCTURE]` | `:Document` | `:StructureNode` | A document contains this top-level structure node. |
| `[:HAS_CHILD]` | `:StructureNode` | `:StructureNode` | Parent-child relationship in the document hierarchy. Headings point to their child headings, paragraphs, tables, etc. |

### Extraction

These relationships link extracted entities back to the document sections they came from, enabling provenance tracing.

| Relationship | From | To | Description |
|---|---|---|---|
| `[:HAS_EXTRACTION]` | `:StructureNode` | `:ExtractionResult` | This section contains an extraction result. |
| `[:HAS_<FIELD>]` | `:ExtractionResult` | `:ModelInstance` | The extraction result contains this model instance (field name determines relationship type). |

### Entity Relationships (from `field_relationships`)

When a schema defines `field_relationships` between labeled entity fields, the graph creates domain-specific edges between `:LabeledEntity` nodes. The relationship type is the field relationship name in `UPPER_SNAKE_CASE`.

| Relationship | From | To | Description |
|---|---|---|---|
| Custom `UPPER_SNAKE_CASE` | `:LabeledEntity` | `:LabeledEntity` | Domain-specific semantic edge between two entity values (e.g., `:TREATS`, `:CONTRAINDICATED_WITH`, `:PART_OF`). |

### Instance Relationships (from `instance_relationships`)

When a schema defines `instance_relationships` between model instances, the graph creates edges between `:ModelInstance` nodes. The relationship type is the instance relationship name in `UPPER_SNAKE_CASE`.

| Relationship | From | To | Description |
|---|---|---|---|
| Custom `UPPER_SNAKE_CASE` | `:ModelInstance` | `:ModelInstance` | Cross-model record edge (e.g., `:ASSOCIATED_WITH`, `:PRECEDES`, `:BELONGS_TO`). |

### Cross-Section `:ModelInstance` Linking via `instance_key`

The `instance_relationships` mechanism described above creates a *relationship*, but the real power of the graph comes from `instance_key`: a `json_schema_extra` flag on one or more fields of a model that makes the **same logical entity, extracted from different sections or documents, collapse into a single Neo4j node** instead of creating a duplicate.

**How the UID is computed:** when a model declares one or more fields with `"instance_key": True`, the graph mapper computes a deterministic 16-character UID (`make_instance_uid`) from the model's class name plus every `instance_key` field, sorted alphabetically by field name, and each value normalized (Unicode NFKD, accents stripped, lowercased, whitespace collapsed) before hashing with SHA-256. This means:

- The UID does **not** depend on which document or section the value came from.
- The UID **is** effectively case-insensitive and accent-insensitive (because inputs are normalized before hashing), even though the hash function itself is not.
- A model **without** any `instance_key` field gets a random UID (`uuid4`) on every extraction — no deduplication is possible, and each extraction always creates a brand-new node.

**Real example from this graph** — `CTDSectionSpec` uses `ctd_code` as its `instance_key`. The same section code referenced from a `HAS_REQUIREMENTS` edge (written first, as a shell) and later extracted directly from its own document section (written second, with full detail) resolve to the exact same node:

```cypher
MATCH (a:ModelInstance {model_class: "CTDSectionSpec"})-[:HAS_REQUIREMENTS]->(:ModelInstance)-[:REQUIREMENT_APPLIES_TO]->(a2:ModelInstance {model_class: "CTDSectionSpec"})
RETURN a.ctd_code, a.uid = a2.uid AS same_node
LIMIT 3;
-- {a.ctd_code: "3.2.S.2.2", same_node: true}
-- {a.ctd_code: "3.2.S.6",   same_node: true}
```

**Shell nodes:** when a source model references a target model via `instance_relationships` before the target has been extracted from its own section, the graph mapper writes a `:ModelInstance` node containing **only** its `instance_key` field(s) plus `model_class` and `uid` (a "shell"). The exact same `MERGE ... ON CREATE SET ... ON MATCH SET ...` Cypher pattern is used whether the node is being created as a shell or enriched later — there is no separate "shell-writing" code path. When the target model is later extracted from its own section, the graph mapper recomputes the same `instance_key`-derived UID, the `MERGE` finds the existing shell, and the remaining (non-key) fields are added to that same node via a plain property `SET` — with no guard against overwriting, but since only the shell's key fields were ever set, nothing is lost.

**Real example — a `CTDSectionSpec` shell (3 properties) next to a fully-extracted one (11 properties), same model_class, same graph:**

```cypher
MATCH (m:ModelInstance {model_class: "CTDSectionSpec"})
RETURN m.ctd_code, size(keys(m)) AS property_count
ORDER BY property_count ASC
LIMIT 1;
-- {m.ctd_code: "3.2.p.6", property_count: 3}   ← shell: only ctd_code + model_class + uid

MATCH (m:ModelInstance {model_class: "CTDSectionSpec"})
RETURN m.ctd_code, size(keys(m)) AS property_count
ORDER BY property_count DESC
LIMIT 1;
-- {m.ctd_code: "3.2.S.2.5", property_count: 11}  ← fully extracted: title, nta_summary, cross_references, ...
```

**Important caveats:**
- The relationship to a shell target is created **unconditionally** as soon as the fixed `join_via` key fields are non-empty — there is no check that the target model will ever actually be extracted. If it never is, the shell simply stays a shell (3 properties) forever.
- Orphaned shells are **not** garbage-collected automatically after ingestion. They are only cleaned up as a side effect of `delete_document()`, which runs a multi-pass query removing any `ModelInstance`/`Entity` no longer reachable from any `ExtractionResult` within 7 hops.
- A list field driving `instance_relationships` (fan-out) creates **one shell per list item**, each with its own UID derived from the fixed anchor key(s) plus that one item's value.

**Query: find shell nodes for any model class (generic property-count heuristic)**

```cypher
-- A genuine shell only has its instance_key fields populated, so it will have
-- far fewer properties than the average fully-extracted instance of the same model_class.
-- This is domain-agnostic: it does not assume any specific field like `description` exists.
MATCH (m:ModelInstance)
WITH m.model_class AS model, avg(size(keys(m))) AS avgProps
MATCH (m2:ModelInstance {model_class: model})
WITH model, avgProps, m2, size(keys(m2)) AS propCount
WHERE propCount < avgProps * 0.5
RETURN model, count(*) AS likely_shell_count
ORDER BY likely_shell_count DESC;
```

### Normalized Models (Tabular Pipeline)

The tabular pipeline (CSV/XLSX/XLS ingestion) supports an optional **normalization** step for fields whose raw source value needs to be turned into a structured sub-model before it becomes useful for cross-dataset linking. This is triggered by two `json_schema_extra` keys on a nested-submodel field: `normalization_model: True` and `normalization_source_fields: [...]`.

**Non-normalized vs. normalized fields — the core distinction:**

- A **non-normalized** field (e.g. `raw_address: str`) holds the value **exactly as it came from the source dataset** — whatever string was in that CSV/Excel column, untouched. It is populated purely by column-mapping, with no LLM call involved.
- A **normalized** field (e.g. `normalized_address: NormalizedAddress | None`) is a nested sub-model that the `NormalizationEngine` fills in by sending the listed `normalization_source_fields` to an LLM (`with_structured_output()`), validating the result with `TypeAdapter.validate_python()`, and writing it back onto the instance. This runs **in memory, before the row is written to Neo4j** — it never touches Cypher directly; the same graph writer (`write_extraction_subgraph`) used by the unstructured (PDF/DOCX) pipeline is reused afterward.

**Why this matters for the graph — linking across datasets:** the raw field on its own is just a string property; it does not participate in any deduplication unless it separately carries its own `entity_label`. The **normalized** sub-model, by contrast, is written to the graph as an ordinary `:ModelInstance` node (linked from its parent via `[:HAS_<FIELDNAME>]`) — and if *that* sub-model declares its own `instance_key` and/or `entity_label` fields (exactly as described in the section above), it deduplicates and links globally just like any other keyed `ModelInstance`. In other words: **normalization does not create graph links by itself** — it produces a clean, structured value that is *much easier* to key or label consistently than free-text, which is what actually enables reliable joins between rows/documents that describe the same real-world entity but wrote it differently (e.g. `"123 Main St, Springfield"` vs `"123 main street springfield"`).

**Example pattern** (illustrative — this exact mechanism is documented and implemented, but no domain model in this repository currently uses it in production; see `docs/user-guides/normalization.md` for the full guide):

```python
class NormalizedAddress(ExtractionModel):
    street: str | None = Field(default=None, description="...")
    city: str | None = Field(default=None, description="...")
    postal_code: str | None = Field(default=None, description="...")
    country_code: str | None = Field(default=None, description="...")

class ContactRecord(ExtractionModel):
    raw_address: str = Field(..., description="Free-text address exactly as it appears in the source column.")
    normalized_address: NormalizedAddress | None = Field(
        default=None,
        description="Structured address derived from raw_address via LLM normalization.",
        json_schema_extra={
            "normalization_model": True,
            "normalization_source_fields": ["raw_address"],  # explicit — see caveat below
        },
    )
```

Resulting graph shape — note `normalized_address` is a **separate node**, never an embedded property map (Neo4j cannot store nested maps as a property value):

```
(:ModelInstance {model_class: "ContactRecord", raw_address: "123 Main St, Springfield"})
  └─[:HAS_NORMALIZED_ADDRESS]→ (:ModelInstance {model_class: "NormalizedAddress", street: "123 Main St", city: "Springfield", ...})
```

**Caveat — `normalization_source_fields` fallback:** if this list is omitted or empty, the engine silently falls back to sending **every other scalar field** of the parent model to the LLM as context. Always set it explicitly to avoid leaking unrelated columns into the normalization prompt.

### Versioning

| Relationship | From | To | Description |
|---|---|---|---|
| `[:HAS_NEWER_VERSION]` | `:Document` | `:Document` | The target document is a newer version that replaces the source document. |

The `latest` property on `:Document` nodes provides a direct indicator of the current version, while the `[:HAS_NEWER_VERSION]` relationship enables version chain traversal.

---

## Graph Structure Diagrams

### Unstructured Pipeline Output

Narrative documents (PDF, Word) produce a heading hierarchy with extracted entities anchored to specific sections:

```
(:Document {name: "clinical_trial.pdf", latest: true})
  └─[:HAS_STRUCTURE]→ (:StructureNode {role: "section", title: "3. Results", latest: true})
                          └─[:HAS_CHILD]→ (:StructureNode {role: "section", title: "3.2 Adverse Events", latest: true})
                                              └─[:HAS_CHILD]→ (:StructureNode {role: "freeform_block", latest: true})
                                                                  └─[:HAS_EXTRACTION]→ (:ExtractionResult {model_class: "AdverseEventModel"})
                                                                                      └─[:HAS_EVENTS]→ (:ModelInstance {model_class: "AdverseEventModel", uid: "..."})
                                                                                                          └─[:REFERENCES]→ (:LabeledEntity {label: "ActiveSubstance", value: "Metformin"})
```

In this pattern:

- The document contains a top-level section (`3. Results`).
- That section contains a sub-section (`3.2 Adverse Events`).
- The sub-section contains a freeform block with text.
- A model instance (`AdverseEventModel`) was extracted from that block.
- The `:LabeledEntity` (`ActiveSubstance`) is reached through the `:ModelInstance` via `[:REFERENCES]`, not directly from `:ExtractionResult`.
- The `:LabeledEntity` node is shared across documents if the same entity value appears elsewhere.

### Tabular Pipeline Output

Structured documents (CSV, Excel) produce a flat table structure with one model instance per row:

```
(:Document {name: "products.csv", latest: true})
  └─[:HAS_STRUCTURE]→ (:StructureNode {role: "table", latest: true})
                          └─[:HAS_CHILD]→ (:StructureNode {role: "row", latest: true})
                                              └─[:HAS_EXTRACTION]→ (:ExtractionResult {model_class: "ProductRecord"})
                                                                  └─[:HAS_FIELDS]→ (:ModelInstance {model_class: "ProductRecord", uid: "..."})
                                                                                      └─[:REFERENCES]→ (:LabeledEntity {label: "ProductCode", value: "PROD-001"})
                          └─[:HAS_CHILD]→ (:StructureNode {role: "row", latest: true})
                                              └─[:HAS_EXTRACTION]→ (:ExtractionResult {model_class: "ProductRecord"})
                                                                  └─[:HAS_FIELDS]→ (:ModelInstance {model_class: "ProductRecord", uid: "..."})
```

In this pattern:

- The document contains a single table structure node.
- Each table row is a child of the table.
- Each row produces one model instance (`ProductRecord`) via an `:ExtractionResult`.
- Labeled entity fields within the model instance are connected via `[:REFERENCES]` from the `:ModelInstance` node, not directly from `:ExtractionResult`.

---

## Versioning

`scinr` supports two document update strategies, both reflected in the graph:

### In-place Update (`update_mode=True`)

When `update_mode=True`, the existing document and all its downstream nodes are replaced in-place. The `version` property on the `:Document` node increments, and the `path` remains the same.

```
(:Document {
  name: "clinical_trial_report.pdf",
  path: "/path/to/clinical_trial_report.pdf",
  version: 2,
  load_date: 2024-06-01T09:00:00,
  latest: true,
  is_folder: false
})
```

### Replacement (`replaces="old_name"`)

When `replaces` is set, a new `:Document` node is created with the same path and version, and a `[:HAS_NEWER_VERSION]` relationship links the old document to the new one.

```
(:Document {name: "clinical_trial_report.pdf", path: "/path/to/clinical_trial_report.pdf", version: 1, latest: false})
  └─[:HAS_NEWER_VERSION]→ (:Document {name: "clinical_trial_report.pdf", path: "/path/to/clinical_trial_report.pdf", version: 2, latest: true})
```

This preserves the full history of document versions in the graph, allowing queries to trace which version produced which extraction results. The `latest` property indicates the current version, while `[:HAS_NEWER_VERSION]` enables version chain traversal.

### Document Deletion

When a document needs to be permanently removed from the graph (rather than updated in-place), use `delete_document()`. This function:

- Removes the `:Document` node(s) matching the given `path` (and optionally `version`), **or** every `:Document` carrying a given `job_id` (`delete_document(job_id=...)`), optionally narrowed further by `tenant_id` / `created_by_user_id`.
- Cascade-deletes all connected structure, annotation, and extraction nodes.
- Runs garbage collection on orphaned `:Entity`, `:ModelInstance`, and `:LabeledEntity` nodes.

Unlike `--update` re-ingestion, deletion is **irreversible** — there is no undo. See the [Document Deletion](document-deletion.md) guide for details.

---

## Query Patterns

### Find All Documents

List all ingested documents, ordered by most recent first:

```cypher
MATCH (d:Document)
WHERE d.latest = true
RETURN d.name, d.path, d.version, d.load_date
ORDER BY d.load_date DESC;
```

### Find All Entities of a Type

Aggregate extracted model instances by their field values:

```cypher
MATCH (m:ModelInstance {model_class: "AdverseEventModel"})
RETURN m.event_type, m.severity, count(m) AS occurrences
ORDER BY occurrences DESC;
```

### Find Entities Extracted from a Specific Document

Trace all model instances back through their source document:

```cypher
MATCH (d:Document {latest: true})-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)-[r]->(m:ModelInstance)
WHERE d.name = "clinical_trial_report.pdf"
RETURN m.model_class AS entity_type, count(m) AS count
ORDER BY count DESC;
```

The correct pattern to descend the hierarchy is `[:HAS_STRUCTURE]` (once, from the `:Document` to its root structure node) followed by `[:HAS_CHILD*0..]` (recursive between `:StructureNode`s) — not `[:HAS_STRUCTURE*0..]`, since `HAS_STRUCTURE` only connects a `:Document` to its root structure node and never repeats between `:StructureNode`s. Also, the relationship between `:ExtractionResult` and `:ModelInstance` is `[:HAS_<FIELD>]` with a dynamic name based on the schema field, so the query uses a generic `-[r]->` instead of a fixed relationship type.

### Find Entity Relationships

List all domain-specific edges between labeled entities:

```cypher
MATCH (a:LabeledEntity)-[r]->(b:LabeledEntity)
RETURN a.label AS source_type, type(r) AS relationship, b.label AS target_type, count(r) AS count
ORDER BY count DESC;
```

### Find Cross-Model Connections

List all relationships between model instances:

```cypher
MATCH (a:ModelInstance)-[r]->(b:ModelInstance)
RETURN a.model_class AS source_type, type(r) AS relationship, b.model_class AS target_type, count(r) AS count
ORDER BY count DESC;
```

### Full Extraction Trace

Reconstruct the full provenance chain from document to extracted entity:

```cypher
MATCH (d:Document {latest: true})-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)-[r]->(m:ModelInstance)
RETURN d.name, s.title, s.role, m.model_class, m;
```

This query is useful for auditing: it shows exactly which section of which document produced each extracted entity.

### Find Entity Usage Across Documents

Identify which documents share a particular entity value. Since labeled entities are deduplicated (shared across documents), find where the entity is referenced via model instances:

```cypher
// Find which sections reference this entity via ModelInstance-[:REFERENCES]->LabeledEntity
MATCH (e:LabeledEntity {label: "ActiveSubstance", value: "Metformin"})
MATCH (m:ModelInstance)-[:REFERENCES]->(e)
MATCH (er:ExtractionResult)-[r]->(m)
MATCH (s:StructureNode)-[:HAS_EXTRACTION]->(er)
MATCH (d:Document)-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s)
WHERE d.latest = true
RETURN DISTINCT d.name, s.title
ORDER BY d.name;
```

Note: `[:HAS_EXTRACTION]` connects exclusively `:StructureNode`→`:ExtractionResult`. The `:ExtractionResult`→`:ModelInstance` segment is always `[:HAS_<FIELD>]` (dynamic), hence the generic `-[r]->`.

### Find Document Version History

Trace the lineage of a document through its versions:

```cypher
MATCH (old:Document)-[:HAS_NEWER_VERSION*0..]->(new:Document)
WHERE new.path = "/path/to/clinical_trial_report.pdf" AND new.latest = true
RETURN old.name AS document_name, old.path AS path, old.version AS version, old.load_date AS load_date
ORDER BY old.version;
```

---

## Indexes and Constraints

For production workloads, create the following indexes and constraints to ensure performant lookups:

```cypher
-- Unique constraint on document path + version combination (enforces deduplication at ingestion)
CREATE CONSTRAINT constraint_document_path_version
FOR (d:Document) REQUIRE (d.path, d.version) IS UNIQUE;

-- Unique constraint on labeled entity UID
CREATE CONSTRAINT constraint_labeled_entity_key
FOR (e:LabeledEntity) REQUIRE e.uid IS UNIQUE;

-- Index on labeled entity label (accelerates label-based queries)
CREATE INDEX idx_labeled_entity_label
FOR (e:LabeledEntity) ON (e.label);

-- Unique constraint on model instance UID
CREATE CONSTRAINT constraint_model_instance_uid
FOR (m:ModelInstance) REQUIRE m.uid IS UNIQUE;

-- Index on model instance model_class (accelerates model class filtering)
CREATE INDEX idx_model_instance_model_class
FOR (m:ModelInstance) ON (m.model_class);

-- Unique constraint on structure node id
CREATE CONSTRAINT constraint_structure_node_id
FOR (s:StructureNode) REQUIRE s.id IS UNIQUE;

-- Index on structure node role (accelerates role-based queries)
CREATE INDEX idx_structure_node_role
FOR (s:StructureNode) ON (s.role);

-- Index on document latest property (accelerates latest version filtering)
CREATE INDEX idx_document_latest
FOR (d:Document) ON (d.latest);
```

---

## Graph Size Estimation

The following table provides rough estimates for planning database capacity:

| Input | Approximate Nodes | Approximate Relationships |
|---|---|---|
| 1 PDF (50 pages) | ~500 StructureNodes + extracted entities | ~600 (structure + extraction) |
| 10 PDFs (50 pages each) | ~5,000 StructureNodes + extracted entities | ~6,000 (structure + extraction) |
| 1 CSV (1,000 rows) | ~1,000 ModelInstances + labeled entities | ~1,000 (extraction) + entity/instance relationships |
| 1 Excel (5 sheets, 500 rows each) | ~2,500 ModelInstances + labeled entities | ~2,500 (extraction) + entity/instance relationships |
| 100 PDFs (mixed size) | ~50,000 StructureNodes + extracted entities | ~60,000 (structure + extraction) |

Notes:

- Actual counts depend on document complexity, schema definition, and extraction yield.
- Labeled entity deduplication reduces total node count when the same entity appears across multiple documents.
- Entity and instance relationships add proportionally to the relationship count based on schema configuration.
- For large-scale deployments, consider partitioning strategies and Neo4j clustering.

## Shell Nodes — Two Distinct Concepts

"Shell node" can refer to two unrelated things in this graph — do not confuse them:

1. **`:ModelInstance` shells** (the mechanism documented in detail above, under "Cross-Section `:ModelInstance` Linking via `instance_key`"): a `ModelInstance` node created ahead of time by an `instance_relationships` reference, containing only its `instance_key` fields until the target model is extracted from its own section. Detect these with the property-count heuristic query shown above.
2. **`:Document` folder nodes** (`is_folder: true`): these are **not** a deduplication artifact at all — they are legitimate, permanent nodes representing a directory in the ingestion hierarchy (as opposed to a leaf file). They are not "incomplete" or waiting to be enriched; `is_folder: true` is simply their normal, final state. Do not apply the `ModelInstance` shell-detection heuristic to `:Document` nodes — check `is_folder` directly instead:

```cypher
MATCH (d:Document {latest: true, is_folder: true})
RETURN d.name, d.path;
```
