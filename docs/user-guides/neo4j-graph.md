# Neo4j Graph Model

`scinr` builds a connected graph representation of input documents and extracted domain concepts inside Neo4j. This guide covers every node type, relationship, and query pattern produced by the `scinr.newton` extraction engine.

The graph serves as the primary output store. After a document passes through the Newton pipeline, all structural elements and extracted entities are persisted as interconnected nodes and relationships, enabling complex cross-document queries that would be difficult or impossible with a flat relational schema.

Two distinct pipelines produce different graph structures:

- **Unstructured pipeline** — processes PDFs, Word documents, and other narrative sources. Produces a hierarchy of headings, paragraphs, tables, and figures, with entities extracted from specific sections.
- **Tabular pipeline** — processes CSV, Excel, and other structured tables. Produces one model instance per row, with labeled entities for stable, deduplicated fields.

---

## Node Types

### `:Document`

Represents the original input file that was ingested into the system. Every document in the graph is a `:Document` node.

```
(:Document {
  document_name: "clinical_trial_report.pdf",
  format: "pdf",
  ingested_at: 2024-01-15T10:30:00,
  version: 1
})
```

| Property | Type | Description |
|---|---|---|
| `document_name` | String | Original file name as provided at ingestion time. |
| `format` | String | File extension or MIME type (e.g., `pdf`, `docx`, `xlsx`, `csv`). |
| `ingested_at` | DateTime | Timestamp when the document was first ingested. |
| `version` | Integer | Version number. Starts at 1; increments on replacement or update. |

---

### `:StructureNode`

Represents a structural element within a document — a heading, paragraph, table, figure, or list item. Structure nodes form a tree rooted at the document.

```
(:StructureNode {
  uid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  section_title: "3.2 Adverse Events",
  node_type: "paragraph",
  text: "The most common adverse events were headache and nausea...",
  page_number: 15,
  hierarchy_level: 2
})
```

| Property | Type | Description |
|---|---|---|
| `uid` | String | Unique identifier (UUID) for this structure node. |
| `section_title` | String | Heading text, if this node represents a section heading. Null for leaf nodes like paragraphs. |
| `node_type` | String | One of: `heading`, `paragraph`, `table`, `figure`, `list`, `row`. |
| `text` | String | The raw text content of this structural element. |
| `page_number` | Integer | Source page number, when available (PDF, paginated documents). Null for non-paginated sources. |
| `hierarchy_level` | Integer | Nesting depth in the document outline. Root headings are level 1, sub-sections increment from there. |

---

### `:LabeledEntity`

Represents a stable, deduplicated entity value extracted from a field marked with an `entity_label` in the schema. Labeled entities are created for fields that act as identifiers or cross-references — drug names, product codes, anatomical terms, etc.

```
(:LabeledEntity:ActiveSubstance {
  value: "Metformin",
  uid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
})
```

| Property | Type | Description |
|---|---|---|
| `value` | String | The canonical entity value. Used for deduplication across documents. |
| `uid` | String | Unique identifier for this specific entity node. |

Key characteristics:

- **Dynamic label**: In addition to the base `:LabeledEntity` label, the node carries a second label derived from the `entity_label` field metadata (e.g., `:ActiveSubstance`, `:AnatomicalStructure`, `:ProductCode`).
- **Deduplication**: If the same `value` is extracted from multiple documents, a single `:LabeledEntity` node is reused. This enables cross-document entity matching and aggregation.
- **Stable identity**: The `uid` remains constant for a given value, allowing reliable joins and traces.

---

### `:ModelInstance`

Represents a single extracted entity record — one instance of a model class defined in the extraction schema. Each model instance stores all field values as node properties.

```
(:ModelInstance:AdverseEventModel {
  uid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  event_type: "Headache",
  severity: "Mild",
  occurrence_count: 3
})
```

| Property | Type | Description |
|---|---|---|
| `uid` | String | Unique identifier for this model instance. |
| *(dynamic)* | Varies | All model field values are stored as additional properties on the node. Property names match the field names defined in the schema. |

Key characteristics:

- **Dynamic label**: In addition to the base `:ModelInstance` label, the node carries a second label matching the model class name (e.g., `:AdverseEventModel`, `:ProductRecord`, `:LaboratoryResult`).
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
| `[:EXTRACTED_FROM]` | `:ModelInstance` | `:StructureNode` | This entity was extracted from the text in this section. |
| `[:EXTRACTED_FROM]` | `:LabeledEntity` | `:StructureNode` | This entity value was extracted from this section. |

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

### Versioning

| Relationship | From | To | Description |
|---|---|---|---|
| `[:REPLACES]` | `:Document` | `:Document` | The source document is a newer version that replaces the target document. |

---

## Graph Structure Diagrams

### Unstructured Pipeline Output

Narrative documents (PDF, Word) produce a heading hierarchy with extracted entities anchored to specific sections:

```
(:Document {document_name: "clinical_trial.pdf"})
  └─[:HAS_STRUCTURE]→ (:StructureNode {node_type: "heading", section_title: "3. Results"})
                          └─[:HAS_CHILD]→ (:StructureNode {node_type: "heading", section_title: "3.2 Adverse Events"})
                                              └─[:HAS_CHILD]→ (:StructureNode {node_type: "paragraph"})
                                                                  └─[:EXTRACTED_FROM]← (:ModelInstance:AdverseEventModel)
                                                                  └─[:EXTRACTED_FROM]← (:LabeledEntity:ActiveSubstance)
```

In this pattern:

- The document contains a top-level heading (`3. Results`).
- That heading contains a sub-heading (`3.2 Adverse Events`).
- The sub-heading contains a paragraph with text.
- Both a model instance (`AdverseEventModel`) and a labeled entity (`ActiveSubstance`) were extracted from that paragraph.
- The `:LabeledEntity:ActiveSubstance` node is shared across documents if the same substance appears elsewhere.

### Tabular Pipeline Output

Structured documents (CSV, Excel) produce a flat table structure with one model instance per row:

```
(:Document {document_name: "products.csv"})
  └─[:HAS_STRUCTURE]→ (:StructureNode {node_type: "table"})
                          └─[:HAS_CHILD]→ (:StructureNode {node_type: "row"})
                                              └─[:EXTRACTED_FROM]← (:ModelInstance:ProductRecord)
                                                                                  └─[:EXTRACTED_FROM]← (:LabeledEntity:ProductCode)
                          └─[:HAS_CHILD]→ (:StructureNode {node_type: "row"})
                                              └─[:EXTRACTED_FROM]← (:ModelInstance:ProductRecord)
```

In this pattern:

- The document contains a single table structure node.
- Each table row is a child of the table.
- Each row produces one model instance (`ProductRecord`).
- Labeled entity fields within the model instance share the same extraction provenance, pointing to the same `:StructureNode` via `[:EXTRACTED_FROM]` for deduplication.
- `[:HAS_FIELD]` is NOT a real relationship — the labeled entities and model instances are both connected to the same StructureNode via `[:EXTRACTED_FROM]`. The connection between a ModelInstance and its LabeledEntity is implicit through shared field values, not an explicit graph relationship.

---

## Versioning

`scinr` supports two document update strategies, both reflected in the graph:

### In-place Update (`update_mode=True`)

When `update_mode=True`, the existing document and all its downstream nodes are replaced in-place. The `version` property on the `:Document` node increments, but the `document_name` remains the same.

```
(:Document {
  document_name: "clinical_trial_report.pdf",
  version: 2,
  ingested_at: 2024-06-01T09:00:00
})
```

### Replacement (`replaces="old_name"`)

When `replaces` is set, a new `:Document` node is created with the new name and version, and a `[:REPLACES]` relationship links it to the old document.

```
(:Document {document_name: "clinical_trial_report_v1.pdf", version: 1})
  └─[:REPLACES]← (:Document {document_name: "clinical_trial_report_v2.pdf", version: 2})
```

This preserves the full history of document versions in the graph, allowing queries to trace which version produced which extraction results.

### Document Deletion

When a document needs to be permanently removed from the graph (rather than updated in-place), use `delete_document()`. This function:

- Removes the `:Document` node(s) matching the given `path` (and optionally `version`).
- Cascade-deletes all connected structure, annotation, and extraction nodes.
- Runs garbage collection on orphaned `:Entity`, `:ModelInstance`, and `:LabeledEntity` nodes.

Unlike `--update` re-ingestion, deletion is **irreversible** — there is no undo. See the [Document Deletion](document-deletion.md) guide for details.

---

## Query Patterns

### Find All Documents

List all ingested documents, ordered by most recent first:

```cypher
MATCH (d:Document)
RETURN d.document_name, d.format, d.ingested_at, d.version
ORDER BY d.ingested_at DESC;
```

### Find All Entities of a Type

Aggregate extracted model instances by their field values:

```cypher
MATCH (m:ModelInstance:AdverseEventModel)
RETURN m.event_type, m.severity, count(m) AS occurrences
ORDER BY occurrences DESC;
```

### Find Entities Extracted from a Specific Document

Trace all model instances back through their source document:

```cypher
MATCH (d:Document)-[:HAS_STRUCTURE*0..]->(s:StructureNode)<-[:EXTRACTED_FROM]-(m:ModelInstance)
WHERE d.document_name = "clinical_trial_report.pdf"
RETURN labels(m) AS entity_type, count(m) AS count
ORDER BY count DESC;
```

The `*0..` variable-length relationship allows matching both direct children and deeply nested structure nodes.

### Find Entity Relationships

List all domain-specific edges between labeled entities:

```cypher
MATCH (a:LabeledEntity)-[r]->(b:LabeledEntity)
RETURN labels(a) AS source_type, type(r) AS relationship, labels(b) AS target_type, count(r) AS count
ORDER BY count DESC;
```

### Find Cross-Model Connections

List all relationships between model instances:

```cypher
MATCH (a:ModelInstance)-[r]->(b:ModelInstance)
RETURN labels(a) AS source_type, type(r) AS relationship, labels(b) AS target_type, count(r) AS count
ORDER BY count DESC;
```

### Full Extraction Trace

Reconstruct the full provenance chain from document to extracted entity:

```cypher
MATCH (d:Document)-[:HAS_STRUCTURE*0..]->(s:StructureNode)<-[:EXTRACTED_FROM]-(m:ModelInstance)
RETURN d.document_name, s.section_title, s.node_type, labels(m), m;
```

This query is useful for auditing: it shows exactly which section of which document produced each extracted entity.

### Find Entity Usage Across Documents

Identify which documents share a particular entity value:

```cypher
MATCH (e:LabeledEntity:ActiveSubstance)-[:EXTRACTED_FROM]->(s:StructureNode)<-[:HAS_STRUCTURE*0..]-(d:Document)
WHERE e.value = "Metformin"
RETURN d.document_name, s.section_title
ORDER BY d.document_name;
```

### Find Document Version History

Trace the lineage of a document through its versions:

```cypher
MATCH (old:Document)-[:REPLACES*0..]->(new:Document)
WHERE new.document_name = "clinical_trial_report_v2.pdf"
RETURN old.document_name AS version_name, old.version, old.ingested_at
ORDER BY old.version;
```

---

## Indexes and Constraints

For production workloads, create the following indexes and constraints to ensure performant lookups:

```cypher
-- Unique constraint on document name (enforces deduplication at ingestion)
CREATE CONSTRAINT document_name_unique
FOR (d:Document) REQUIRE d.document_name IS UNIQUE;

-- Index on labeled entity value (accelerates deduplication and cross-document lookups)
CREATE INDEX labeled_entity_value
FOR (n:LabeledEntity) ON (n.value);

-- Index on model instance UID (accelerates direct entity lookups)
CREATE INDEX model_instance_uid
FOR (m:ModelInstance) ON (m.uid);

-- Index on structure node UID (accelerates provenance tracing)
CREATE INDEX structure_node_uid
FOR (s:StructureNode) ON (s.uid);

-- Index on document ingestion date (accelerates temporal queries)
CREATE INDEX document_ingested_at
FOR (d:Document) ON (d.ingested_at);
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
