# Document Deletion

`delete_document()` permanently removes a `:Document` node and its entire subgraph from Neo4j, then runs garbage collection on orphaned nodes. This is the definitive way to remove a document from your knowledge graph — it is irreversible and cannot be undone.

---

## Introduction

`delete_document()` is exported from the package root:

```python
from scinr.newton import delete_document, DeletionResult
```

It is an `async` function — `await` it (or wrap it with `asyncio.run()`). It:

1. **Locates** the target `:Document` node(s) **within one tenant** — `tenant_id` is mandatory (`None` = public documents) — by either `path` (optionally narrowed by `version`) **or** `job_id`; exactly one of the two must be given. `created_by_user_id` is an optional extra filter on top of either selector.
2. **Cascade-deletes** the document and every node reachable from it:
   - The descendants of a matched folder document via `IS_COMPOSED_OF*` (downwards only — a leaf's parent folder and siblings are never touched)
   - All `:StructureNode` descendants via `HAS_STRUCTURE*` / `HAS_CHILD*`
   - All `:InfoUnit`, `:ModelDecision`, `:ProposedModel`, `:ProposedField`, and `:ExtractionResult` children
3. **Garbage-collects** the `:Entity`, `:ModelInstance`, and `:LabeledEntity` nodes of the tenant that the deletion left orphaned.

The function opens and closes its own Neo4j driver — you do not need to manage connections manually.

---

## When to Use Deletion vs. Update

`scinr` provides two mechanisms for replacing document content:

| Operation | What it does | Use when |
|---|---|---|
| `delete_document()` | Permanently removes the `:Document` node and all descendants. No undo. | The document should no longer exist in the graph at all. |
| `update_mode=True` re-ingestion | Keeps the `:Document` node, wipes its content, and re-ingests new data at the same version. | You want to refresh the content of an existing document while preserving its identity and version history. |

If you simply need to update content, pass `update_mode=True` to `run_pipeline()`. Use `delete_document()` only when you want complete, permanent removal.

---

## Basic Usage

```python
import asyncio
from scinr.newton import delete_document, configure, DeletionResult

async def main():
    # delete_document() makes no LLM calls, so no llm= is needed here.
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
    )

    result = await delete_document("/path/to/document.pdf", tenant_id="acme")
    print(f"Found: {result.found}")
    print(f"Documents deleted: {result.documents_deleted}")

asyncio.run(main())
```

`delete_document()` is `async` — `await` it from a coroutine, or drive it with `asyncio.run()` as above. The remaining snippets on this page show just the `await delete_document(...)` call for brevity; each assumes it runs inside an `async` function after `configure()`.

The `path` parameter matches the `path` property on `:Document` nodes in Neo4j. This is the file path (relative or absolute) as it was recorded at ingestion time. A path is only unique **within a tenant** (see [Tenant scope](#tenant-scope-mandatory) below), which is why `tenant_id` must always be given.

---

## Version-Targeted Deletion

By default, `delete_document()` deletes **all versions** of a document matching the given `path`:

```python
# Delete ALL versions of tenant acme's document
result = await delete_document("/path/to/document.pdf", tenant_id="acme")
```

To delete a **specific version**, pass the `version` parameter:

```python
# Delete only version 2
result = await delete_document("/path/to/document.pdf", version=2, tenant_id="acme")
```

When `version` is specified, only that version's `:Document` node and its cascade are removed. Other versions of the same document remain untouched.

---

## Selecting by `job_id`

Instead of a `path`, you can delete **every** document produced by a single ingestion run by passing its `job_id` (the value given to `run_pipeline(job_id=...)`):

```python
# Delete every :Document of tenant acme whose job_id property equals
# "job-2026-09-06-a", across all paths and versions of that run — plus each
# one's full cascade.
result = await delete_document(job_id="job-2026-09-06-a", tenant_id="acme")
```

Exactly one of `path` or `job_id` must be provided — passing neither, or both, raises `ValueError`. `version` is still accepted alongside `job_id` as an additional filter.

---

## Tenant scope (mandatory)

The tenant is part of every document's identity: two tenants that ingest the same path own two independent documents (see [Neo4j Graph Storage — Multi-tenancy](neo4j-graph.md#multi-tenancy-the-tenant-is-part-of-the-document-identity)). A deletion therefore always targets exactly one tenant:

- `tenant_id` is a **keyword-only argument with no default**. Omitting it raises `TypeError` — there is no way to delete "in any tenant".
- `tenant_id="acme"` only ever matches tenant `acme`'s documents.
- `tenant_id=None` — or, equivalently, `tenant_id="__public__"` — explicitly means **public documents** (those ingested without a tenant, stored with the reserved value `"__public__"`). It never matches a tenant's documents. (Unlike navigation, where `tenant_id=None` means "all tenants", a deletion never spans tenants: it is destructive.)

```python
# Tenant acme's copy only — tenant globex's document at the same path is untouched
result = await delete_document("/path/to/document.pdf", tenant_id="acme")

# The public document at that path
result = await delete_document("/path/to/document.pdf", tenant_id=None)
```

Because each tenant has its own folder `:Document` nodes, the `IS_COMPOSED_OF*` cascade stays inside the tenant as well. Who may delete a public document is an authorization decision for your API layer; `delete_document()` only enforces the scope.

## Several jobs or users at once

`job_id` and `created_by_user_id` accept **one value or a list** (matched with `IN`; OR inside each filter, AND with the tenant and with each other). An empty list raises `ValueError`.

```python
# Delete two ingestion runs of tenant acme in one call
result = await delete_document(job_id=["job-1", "job-2"], tenant_id="acme")
```

## Extra filter: `created_by_user_id`

`created_by_user_id` is an optional keyword filter applied **on top of** either selector (`path` or `job_id`) and the tenant. Like `job_id`, it takes one value or a list:

```python
# Delete a whole job, but only the documents created by one user
result = await delete_document(job_id="job-123", tenant_id="acme", created_by_user_id="user-42")

# ... or by any of several users
result = await delete_document(job_id="job-123", tenant_id="acme", created_by_user_id=["user-42", "user-43"])
```

Left unset (`None`) it means **"do not filter on this property"** — it does *not* mean "the property must be null".

These values are populated by `run_pipeline(tenant_id=..., created_by_user_id=..., job_id=...)` at ingestion time. `DeletionResult` echoes back whichever selector and filters were used (`result.path`, `result.job_id`, `result.tenant_id`, `result.created_by_user_id`, exactly as passed — a string or a list); `result.path` is `None` for a `job_id`-selected deletion, and `result.tenant_id` echoes the value you passed (`None` for a deletion of public documents).

---

## Understanding the Cascade

When you call `delete_document()`, the following nodes are deleted:

### Target Document(s)

The tenant's `:Document` node(s) matching the `path` (and `version`, if specified). If a matched document is a folder, every `:Document` below it via `IS_COMPOSED_OF*` (subfolders and leaves) is also deleted. The cascade only goes **downwards**: deleting a leaf never deletes its parent folder or its siblings.

### Structure Tree

For each deleted document, all descendants are removed:

- `:StructureNode` nodes reached via `HAS_STRUCTURE*` and `HAS_CHILD*`
- `:InfoUnit` nodes attached to those structure nodes
- `:ModelDecision` nodes (annotation results)
- `:ProposedModel`, `:ProposedField`, `:ComplementaryMatch` and `:SupplementaryField` nodes (annotation details)
- `:ExtractionResult` nodes (entity extraction results)

### Visual Representation

```
(:Document {tenant_id: "acme", path: "/path/to/document.pdf"})
  │
  ├─[:IS_COMPOSED_OF]→ (:Document)  [child of a folder — also deleted; never the parent]
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

All of the above are `DETACH DELETE`d, meaning all their relationships are severed before the nodes are removed.

### Bounded transactions

The cascade does **not** run in one transaction. A transaction keeps everything it deletes in memory until it commits, and Neo4j caps the memory of all running transactions together (`dbms.memory.transaction.total.max`, a fixed size on Aura). Deleting a folder of a few hundred documents at once goes over that cap and fails with `MemoryPoolOutOfMemoryError`. So:

- the `:Document` nodes are marked `deletion_pending = true` before anything is deleted;
- the documents are processed 50 at a time;
- each delete commits every 1,000 nodes (`CALL { ... } IN TRANSACTIONS`), children before parents;
- the `:Document` nodes go last — after the garbage collection — descendants before the matched documents;
- each query is retried on transient errors (deadlocks, a full memory pool, a lost connection or a leader change in a cluster).

The cascade is therefore not atomic. If it fails half-way, part of the subtree is gone but the `:Document` nodes are still there, so **calling `delete_document()` again with the same selector deletes what is left**. The storage records were already deleted by then (that step comes first).

When a delete query had to be retried, the batches it committed before failing are not counted again, so the `*_deleted` counters of the result are then a lower bound.

---

## Garbage Collection

Deleting an `:ExtractionResult` can leave what hung from it with no owner. Two rules say what an orphan is:

- an `:Entity` or `:ModelInstance` is an orphan when **no `:ExtractionResult` reaches it** within 7 hops;
- a `:LabeledEntity` is an orphan when **nothing points at it**.

`delete_document()` deletes the orphans **it causes**, and only looks where it can have caused one:

1. Before the `:ExtractionResult` nodes of a chunk of documents are deleted, it reads what they point at. Those nodes are the candidates.
2. Once the subtree is gone, each candidate is checked against the two rules, and the orphans are deleted 1,000 per transaction.
3. What a deleted orphan pointed at becomes a candidate in turn (a nested `:ModelInstance`, the `:LabeledEntity` values it referenced), and so on until a round deletes nothing.

A node still used by another document is checked and left alone: the same `:LabeledEntity` or keyed `:ModelInstance` shared with a document that stays in the graph is never deleted.

The cost follows the size of the deletion, not of the tenant. Deleting one document from a tenant with millions of entities checks the few hundred nodes that document pointed at.

Everything is **scoped to the deletion's tenant** (`tenant_id = $tenant_id`, the stored key — `"__public__"` for public documents): the deleted subtree belongs to one tenant and no data relationship crosses tenants. Nodes without `tenant_id` (graphs written before multi-tenancy) are never collected. `freeze_document()` collects its orphans the same way (see [Document Freezing](document-freezing.md)).

> This is what reclaims `:ModelInstance` **shell nodes** — targets created by an `instance_relationships` reference whose actual model was never extracted from any document section — once the instance that referenced them is gone. See [Cross-Section `:ModelInstance` Linking via `instance_key`](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key) for how shells are created.

### What it does not collect

Orphans that this deletion did not cause stay in the graph. Two things leave them:

- **Other write paths.** Re-ingesting with `update_mode=True` and re-running the extraction on a node delete the old `:ExtractionResult` nodes without collecting what hung from them.
- **An interrupted delete or freeze.** The candidates are held in memory by the process that read them. If it dies between deleting the `:ExtractionResult` nodes and collecting, they are lost.

The second case repairs itself: a `delete_document()` or `freeze_document()` that finds the mark of an interrupted run (`deletion_pending`, or `frozen_cleanup_pending` on a stub) checks **every** `:Entity`, `:ModelInstance` and `:LabeledEntity` of the tenant instead of its own candidates, and logs a warning saying so. For the first case, run `collect_orphans()`.

### collect_orphans(): sweeping a whole tenant

```python
from scinr.newton import collect_orphans

result = await collect_orphans(tenant_id="acme")
print(result.gc_entity_model_instance_deleted, result.gc_labeled_entity_deleted)
```

`collect_orphans()` applies the same two rules to every node of the tenant, in two passes:

```cypher
MATCH (mi:Entity|ModelInstance)
WHERE mi.tenant_id = $tenant_id AND NOT EXISTS {
  MATCH (:ExtractionResult)-[*1..7]->(mi)
}
CALL (mi) {
  DETACH DELETE mi
} IN TRANSACTIONS OF 1000 ROWS
```

```cypher
MATCH (mi:LabeledEntity)
WHERE mi.tenant_id = $tenant_id AND NOT EXISTS { (mi)<--() }
CALL (mi) {
  DETACH DELETE mi
} IN TRANSACTIONS OF 1000 ROWS
```

Each pass is repeated until an iteration deletes nothing, up to **7 iterations** (`GC_MAX_PASSES = 7`): deleting a `:LabeledEntity` can leave the one it pointed at with nothing pointing at it.

Things to know before scheduling it:

- `tenant_id` is mandatory, as in `delete_document()` (`None` = public documents). There is no sweep across tenants.
- Its cost grows with the tenant, whatever there is to collect: it starts from the tenant's indexes and checks every node. Run it from time to time (after a batch of re-ingestions, from a scheduled job), not after every operation.
- Do not run it while the same tenant is being ingested. The pipeline writes some nodes just before the `:ExtractionResult` link that will reach them; for that instant they look like orphans.

It returns an `OrphanCollectionResult` with `tenant_id` and the four `gc_*` counters (here `*_passes` counts iterations over the whole tenant).

---

## Inspecting DeletionResult

`delete_document()` returns a `DeletionResult` dataclass with detailed counters:

| Field | Type | Description |
|---|---|---|
| `path` | `str \| None` | The document `path` that was targeted, or `None` when the deletion was selected by `job_id`. |
| `version` | `int \| None` | The specific version requested, or `None` if all versions were targeted. |
| `job_id` | `str \| None` | The `job_id` selector that was targeted, or `None` when selected by `path`. |
| `tenant_id` | `str \| None` | The tenant the deletion was scoped to (always applied), or `None` when it targeted public documents. |
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
| `gc_entity_model_instance_deleted` | `int` | `:Entity`/`:ModelInstance` nodes this deletion left orphaned, and deleted. |
| `gc_entity_model_instance_passes` | `int` | GC rounds run over `:Entity`/`:ModelInstance` candidates: `0` when there was none, one more each time a deleted node turned what it pointed at into candidates. |
| `gc_labeled_entity_deleted` | `int` | `:LabeledEntity` nodes this deletion left orphaned, and deleted. |
| `gc_labeled_entity_passes` | `int` | GC rounds run over `:LabeledEntity` candidates. |

When the call finished an interrupted delete or freeze, the `gc_*` counters are those of a sweep of the whole tenant (see [What it does not collect](#what-it-does-not-collect)).

### Example Output

```python
result = await delete_document("/path/to/document.pdf", tenant_id="acme")

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
result = await delete_document(job_id="ingest-2026-09-06-a", tenant_id="acme")

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

Unlike `update_mode=True` re-ingestion (which preserves the `:Document` node and allows you to re-run the pipeline), `delete_document()` removes the document entirely. If you need the document back, you must re-ingest it from the original source file.

### Shared LabeledEntity Deduplication

`:LabeledEntity` nodes are deduplicated **within a tenant** — the same entity value from multiple documents of one tenant shares a single node (the tenant is hashed into its `uid`, so tenants — and public documents — never share one). The garbage collection pass only removes `:LabeledEntity` nodes that have **no incoming relationships at all**. If the same labeled entity appears in other documents that remain in the graph, it will **not** be deleted. This is intentional and preserves cross-document entity integrity.

### IS_COMPOSED_OF Cascade Scope

`IS_COMPOSED_OF` goes from a folder to its children, and the cascade (`(d)-[:IS_COMPOSED_OF*]->(cd)`) only follows it **downwards** from the documents the selector matched:

- Deleting a **folder** document deletes its whole subtree (subfolders and leaves).
- Deleting a **leaf** by `path` deletes only that leaf: its parent folder and its siblings are untouched (the parent keeps an `IS_COMPOSED_OF` fewer).

The same cascade applies in `job_id` mode: `delete_document(job_id=...)` seeds the cascade with every `:Document` carrying that `job_id`, then follows `IS_COMPOSED_OF*` to their descendants. If the job created both a folder and its children, they are all matched directly as independent seeds — the cascade does not go up, several documents simply start matched. In the normal case every document produced by one `run_pipeline()` call shares the `job_id`, so this simply deletes the whole run. The edge case to be aware of is a folder-parent node that was first created by job A and later reused (via `MERGE`) by a document of the same tenant ingested under job B — deleting job A will also remove that job-B leaf through the cascade. Folder nodes are never shared across tenants, so this cannot reach another tenant's documents.

### Frozen Documents

`delete_document()` does not know about frozen documents (see [Document Freezing](document-freezing.md)). It deletes a frozen stub and whatever of its subtree is still hanging from the structure tree, but:

- the snapshot in the freeze backend is **not** deleted (its id is the stub's `frozen_blob_id`);
- with `keep_structure_nodes=False`, the `:ModelDecision` / `:ExtractionResult` nodes kept by `keep_annotations` / `keep_extraction_results` hang directly from the `:Document` and are **not** reached by the cascade.

To delete a frozen document completely, `restore_document()` it first and then delete it.

### Version Isolation

When `version` is specified, only that version's cascade is deleted. However, shared `:LabeledEntity` nodes connected to other versions are preserved by the GC pass (they still have incoming relationships from the remaining versions).

### Driver Management

`delete_document()` opens its own Neo4j driver via `get_driver()` and closes it in a `finally` block. You do not need to manage driver lifecycle manually. However, if you are calling `delete_document()` in a tight loop, consider the connection overhead — each call creates and closes a driver.

---

## See Also

- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding the graph model, node types, and relationships affected by deletion.
- **[Neo4j Graph Storage — instance_key shells](neo4j-graph.md#cross-section-modelinstance-linking-via-instance_key)** — Understand what a `:ModelInstance` "shell" node is and why unreferenced ones only get cleaned up via the garbage-collection pass described below.
- **[Running the Pipeline](running-pipeline.md)** — Pipeline entry points, including `update_mode=True` for in-place document updates.
- **[Deletion API](../api/deletion.md)** — Auto-generated reference for `delete_document()`.
- **[Results API](../api/results.md)** — `DeletionResult` dataclass reference.
- **[Architecture](../architecture.md)** — Pipeline stages and Neo4j schema details.
