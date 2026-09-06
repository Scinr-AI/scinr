# Document Deletion

`delete_document()` permanently removes a `:Document` node and its entire subgraph from Neo4j, then runs garbage collection on orphaned nodes. This is the definitive way to remove a document from your knowledge graph — it is irreversible and cannot be undone.

---

## Introduction

`delete_document()` is exported from the package root:

```python
from scinr.newton import delete_document, DeletionResult
```

It is an `async` function — `await` it (or wrap it with `asyncio.run()`). It:

1. **Locates** the target `:Document` node(s) by either `path` (optionally narrowed by `version`) **or** `job_id` — exactly one of the two must be given. `tenant_id` and `created_by_user_id` are optional extra filters on top of either selector.
2. **Cascade-deletes** the document and every node reachable from it:
   - Folder-parent documents and siblings via `IS_COMPOSED_OF*`
   - All `:StructureNode` descendants via `HAS_STRUCTURE*` / `HAS_CHILD*`
   - All `:InfoUnit`, `:ModelDecision`, `:ProposedModel`, `:ProposedField`, and `:ExtractionResult` children
3. **Garbage-collects** orphaned `:Entity`, `:ModelInstance`, and `:LabeledEntity` nodes in two independent passes.

The function opens and closes its own Neo4j driver — you do not need to manage connections manually.

---

## When to Use Deletion vs. Update

`scinr` provides two mechanisms for replacing document content:

| Operation | What it does | Use when |
|---|---|---|
| `delete_document()` | Permanently removes the `:Document` node and all descendants. No undo. | The document should no longer exist in the graph at all. |
| `--update` re-ingestion | Keeps the `:Document` node, wipes its content, and re-ingests new data. | You want to refresh the content of an existing document while preserving its identity and version history. |

If you simply need to update content, use the `--update` flag with `run_pipeline()`. Use `delete_document()` only when you want complete, permanent removal.

---

## Basic Usage

```python
import asyncio
from scinr.newton import delete_document, configure, DeletionResult

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )

    result = await delete_document("/path/to/document.pdf")
    print(f"Found: {result.found}")
    print(f"Documents deleted: {result.documents_deleted}")

asyncio.run(main())
```

`delete_document()` is `async` — `await` it from a coroutine, or drive it with `asyncio.run()` as above. The remaining snippets on this page show just the `await delete_document(...)` call for brevity; each assumes it runs inside an `async` function after `configure()`.

The `path` parameter matches the `path` property on `:Document` nodes in Neo4j. This is the file path (relative or absolute) as it was recorded at ingestion time.

---

## Version-Targeted Deletion

By default, `delete_document()` deletes **all versions** of a document matching the given `path`:

```python
# Delete ALL versions of a document
result = await delete_document("/path/to/document.pdf")
```

To delete a **specific version**, pass the `version` parameter:

```python
# Delete only version 2
result = await delete_document("/path/to/document.pdf", version=2)
```

When `version` is specified, only that version's `:Document` node and its cascade are removed. Other versions of the same document remain untouched.

---

## Selecting by `job_id`

Instead of a `path`, you can delete **every** document produced by a single ingestion run by passing its `job_id` (the value given to `run_pipeline(job_id=...)`):

```python
# Delete every :Document whose job_id property equals "job-2026-09-06-a",
# across all paths and versions of that run — plus each one's full cascade.
result = await delete_document(job_id="job-2026-09-06-a")
```

Exactly one of `path` or `job_id` must be provided — passing neither, or both, raises `ValueError`. `version` is still accepted alongside `job_id` as an additional filter.

---

## Extra filters: `tenant_id` and `created_by_user_id`

`tenant_id` and `created_by_user_id` are optional keyword filters applied **on top of** either selector (`path` or `job_id`):

```python
# Only delete this path if it also belongs to tenant "acme"
result = await delete_document("/path/to/document.pdf", tenant_id="acme")

# Delete a whole job, but only the documents created by one user
result = await delete_document(job_id="job-123", created_by_user_id="user-42")
```

A filter left unset (`None`) means **"do not filter on this property"** — it does *not* mean "the property must be null". A document with no `tenant_id` is still matched by `delete_document("/path", tenant_id=None)`.

These values are populated by `run_pipeline(tenant_id=..., created_by_user_id=..., job_id=...)` at ingestion time. `DeletionResult` echoes back whichever selector and filters were used (`result.path`, `result.job_id`, `result.tenant_id`, `result.created_by_user_id`); `result.path` is `None` for a `job_id`-selected deletion.

---

## Understanding the Cascade

When you call `delete_document()`, the following nodes are deleted in a single transaction:

### Target Document(s)

The `:Document` node(s) matching the `path` (and `version`, if specified). If the document is part of a folder hierarchy, every `:Document` reachable via `IS_COMPOSED_OF*` is also deleted — this includes folder-parent documents and their sibling documents.

### Structure Tree

For each deleted document, all descendants are removed:

- `:StructureNode` nodes reached via `HAS_STRUCTURE*` and `HAS_CHILD*`
- `:InfoUnit` nodes attached to those structure nodes
- `:ModelDecision` nodes (annotation results)
- `:ProposedModel` and `:ProposedField` nodes (annotation details)
- `:ExtractionResult` nodes (entity extraction results)

### Visual Representation

```
(:Document {path: "/path/to/document.pdf"})
  │
  ├─[:IS_COMPOSED_OF]→ (:Document)  [folder parent — also deleted]
  │
  └─[:HAS_STRUCTURE]→ (:StructureNode)
                         ├─[:HAS_CHILD]→ (:StructureNode)
                         │                    ├─[:HAS_INFO_UNIT]→ (:InfoUnit)
                         │                    ├─[:HAS_MODEL_DECISION]→ (:ModelDecision)
                         │                    │                              ├─[:HAS_PROPOSED_MODEL]→ (:ProposedModel)
                         │                    │                              │                              └─[:HAS_PROPOSED_FIELD]→ (:ProposedField)
                         │                    └─[:HAS_EXTRACTION]→ (:ExtractionResult)
                         └─[:HAS_CHILD]→ (:StructureNode)
```

All of the above are `DETACH DELETE`d in a single query, meaning all their relationships are severed before the nodes are removed.

---

## Garbage Collection

After the cascade delete, two independent garbage-collection passes run to clean up orphaned nodes that were not directly connected to the deleted documents.

### Pass 1: Entity / ModelInstance

Finds `:Entity` and `:ModelInstance` nodes that are no longer reachable from any `:ExtractionResult` within 7 hops:

```cypher
MATCH (mi:Entity|ModelInstance)
WHERE NOT EXISTS {
  MATCH (e:ExtractionResult)-[*1..7]->(mi)
}
DETACH DELETE mi
```

> This pass is precisely what reclaims orphaned `:ModelInstance` **shell nodes** — targets created by an `instance_relationships` reference whose actual model was never extracted from any document section. See [Cross-Section `:ModelInstance` Linking via `instance_key`](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key) for how shells are created and why they are never garbage-collected automatically outside of `delete_document()`.

### Pass 2: LabeledEntity

Finds `:LabeledEntity` nodes with no incoming relationships at all:

```cypher
MATCH (mi:LabeledEntity)
WHERE NOT EXISTS { (mi)<--() }
DETACH DELETE mi
```

### Iteration Behavior

Each pass runs up to **7 iterations** (`GC_MAX_PASSES = 7`). A pass stops early as soon as an iteration deletes zero nodes. This handles cascading orphans — deleting a batch of `:Entity` nodes might reveal new orphaned `:LabeledEntity` nodes that were only reachable through the deleted entities.

---

## Inspecting DeletionResult

`delete_document()` returns a `DeletionResult` dataclass with detailed counters:

| Field | Type | Description |
|---|---|---|
| `path` | `str \| None` | The document `path` that was targeted, or `None` when the deletion was selected by `job_id`. |
| `version` | `int \| None` | The specific version requested, or `None` if all versions were targeted. |
| `job_id` | `str \| None` | The `job_id` selector that was targeted, or `None` when selected by `path`. |
| `tenant_id` | `str \| None` | The `tenant_id` filter applied to the match, or `None` if none was requested. |
| `created_by_user_id` | `str \| None` | The `created_by_user_id` filter applied to the match, or `None` if none was requested. |
| `found` | `bool` | `True` if at least one matching `:Document` existed before deletion. When `False`, all counters are `0` and no queries were executed. |
| `versions_deleted` | `list[int]` | Sorted list of integer versions that matched and were deleted. Empty when `found` is `False`. |
| `documents_deleted` | `int` | Number of `:Document` nodes deleted (matched documents plus any reached via `IS_COMPOSED_OF*`). |
| `structure_nodes_deleted` | `int` | Number of `:StructureNode` nodes deleted. |
| `info_units_deleted` | `int` | Number of `:InfoUnit` nodes deleted. |
| `model_decisions_deleted` | `int` | Number of `:ModelDecision` nodes deleted. |
| `proposed_models_deleted` | `int` | Number of `:ProposedModel` nodes deleted. |
| `proposed_fields_deleted` | `int` | Number of `:ProposedField` nodes deleted. |
| `extraction_results_deleted` | `int` | Number of `:ExtractionResult` nodes deleted. |
| `gc_entity_model_instance_deleted` | `int` | Total `:Entity`/`:ModelInstance` nodes deleted across all GC iterations. |
| `gc_entity_model_instance_passes` | `int` | Number of GC iterations actually run for the Entity/ModelInstance pass (capped at 7). |
| `gc_labeled_entity_deleted` | `int` | Total `:LabeledEntity` nodes deleted across all GC iterations. |
| `gc_labeled_entity_passes` | `int` | Number of GC iterations actually run for the LabeledEntity pass (capped at 7). |

### Example Output

```python
result = await delete_document("/path/to/document.pdf")

if result.found:
    print(f"Deleted {result.documents_deleted} document(s), "
          f"{result.structure_nodes_deleted} structure node(s)")
    print(f"GC cleaned up {result.gc_entity_model_instance_deleted} entity/model instance(s) "
          f"and {result.gc_labeled_entity_deleted} labeled entity(s)")
else:
    print("No document found at that path.")
```

Bulk-deleting an entire ingestion run and reading back which selector was used:

```python
result = await delete_document(job_id="ingest-2026-09-06-a")

print(result.path)        # None  — this was a job_id-selected deletion
print(result.job_id)      # "ingest-2026-09-06-a"
print(result.versions_deleted)     # e.g. [1, 1, 2] across the matched documents
print(result.documents_deleted)    # total :Document nodes removed
```

---

## Important Caveats

### Irreversible Operation

`delete_document()` uses `DETACH DELETE` — once nodes are removed, they cannot be recovered. There is no undo mechanism. Always verify the target `path` and `version` before calling.

### No Undo

Unlike `--update` re-ingestion (which preserves the `:Document` node and allows you to re-run the pipeline), `delete_document()` removes the document entirely. If you need the document back, you must re-ingest it from the original source file.

### Shared LabeledEntity Deduplication

`:LabeledEntity` nodes are globally deduplicated — the same entity value from multiple documents shares a single node. The garbage collection pass only removes `:LabeledEntity` nodes that have **no incoming relationships at all**. If the same labeled entity appears in other documents that remain in the graph, it will **not** be deleted. This is intentional and preserves cross-document entity integrity.

### IS_COMPOSED_OF Cascade Scope

If the target document is part of a folder hierarchy (connected via `IS_COMPOSED_OF`), the cascade delete reaches **all** documents connected through that relationship — including folder-parent documents and their siblings. This means deleting a leaf document in a folder hierarchy may also delete the parent folder document and its other children.

If you need to delete only a single document without affecting its folder hierarchy, consider using `--update` re-ingestion instead, or manually manage the folder structure before deletion.

The same cascade applies in `job_id` mode: `delete_document(job_id=...)` seeds the cascade with every `:Document` carrying that `job_id`, then follows `IS_COMPOSED_OF*` to their descendants. In the normal case every document produced by one `run_pipeline()` call shares the `job_id`, so this simply deletes the whole run. The edge case to be aware of is a folder-parent node that was first created by job A and later reused (via `MERGE`) by a document ingested under job B — deleting job A will also remove that job-B leaf through the cascade.

### Version Isolation

When `version` is specified, only that version's cascade is deleted. However, shared `:LabeledEntity` nodes connected to other versions are preserved by the GC pass (they still have incoming relationships from the remaining versions).

### Driver Management

`delete_document()` opens its own Neo4j driver via `get_driver()` and closes it in a `finally` block. You do not need to manage driver lifecycle manually. However, if you are calling `delete_document()` in a tight loop, consider the connection overhead — each call creates and closes a driver.

---

## See Also

- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding the graph model, node types, and relationships affected by deletion.
- **[Neo4j Graph Storage — instance_key shells](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key)** — Understand what a `:ModelInstance` "shell" node is and why unreferenced ones only get cleaned up via the garbage-collection pass described below.
- **[Running the Pipeline](running-pipeline.md)** — Pipeline entry points, including the `--update` flag for in-place document updates.
- **[Deletion API](../api/deletion.md)** — Auto-generated reference for `delete_document()`.
- **[Results API](../api/results.md)** — `DeletionResult` dataclass reference.
- **[Architecture](../architecture.md)** — Pipeline stages and Neo4j schema details.
