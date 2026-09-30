# Document Freezing

`freeze_document()` archives the subgraph of a document to a **snapshot** in a document store and reduces the document in Neo4j to a **stub**: the `:Document` node stays (so the document is still listed and its version history is kept), but its structure tree, InfoUnits, annotations and extraction results can leave the graph. `restore_document()` rebuilds the subgraph from the snapshot. `export_document_snapshot()` produces the same snapshot without touching the graph.

Use it to keep the graph small without losing documents you may need again. Unlike `delete_document()`, freezing is reversible. Unlike re-ingesting, restoring gives you back exactly what was frozen, with no LLM calls.

```python
from scinr.newton import (
    FreezeError,
    FreezeResult,
    RestoreResult,
    export_document_snapshot,
    freeze_document,
    restore_document,
)
```

All three are `async` functions. Each opens and closes its own Neo4j driver.

---

## Configuration: the freeze backend

Snapshots live in a **freeze backend**, configured next to the storage backend:

| Setting | `configure()` param | Environment Variable | Default |
|---|---|---|---|
| Backend type | `freeze_backend` | `FREEZE_BACKEND` | the resolved `storage_backend` |
| Metadata collection | `mongodb_frozen_collection` | `MONGODB_FROZEN_COLLECTION` | `"frozen_documents"` |
| GridFS bucket | `mongodb_frozen_gridfs_bucket` | `MONGODB_FROZEN_GRIDFS_BUCKET` | `"frozen_snapshots"` |
| Custom repository | `custom_freeze_storage` | *(none)* | `None` |

`freeze_backend` has no default of its own: when neither the argument nor `FREEZE_BACKEND` is set, it **inherits the resolved `storage_backend`**. With `storage_backend="mongodb"` you therefore get snapshots in the same MongoDB database with no extra configuration. The MongoDB backend reuses `mongodb_uri` / `mongodb_database` and the `mongodb_ensure_indexes` setting of the storage backend.

- `"none"` — `freeze_document()`, `restore_document()` and `export_document_snapshot(destination="storage")` raise `ConfigurationError`. The `"dict"` and `"file"` destinations of the export still work.
- `"mongodb"` — the snapshot JSON goes to GridFS (`frozen_snapshots`), and a metadata document goes to `frozen_documents`. The metadata document holds the tenant, `created_by_user_id`, `job_id`, the size, the SHA-256 checksum, `frozen_at` and the keep flags. Its `_id` is the snapshot's id (`frozen_blob_id`).
- `"custom"` — pass an instance of `scinr.newton.freeze.base.FreezeRepository` as `custom_freeze_storage`. It has three mandatory methods, `store_snapshot`, `read_snapshot_to_file` and `delete_snapshot`, and an optional one, `find_snapshots`, which `restore_document()` needs to find the snapshot of a document that is no longer in the graph. All of them are tenant-scoped.

```python
configure(..., storage_backend="mongodb")                          # snapshots in MongoDB too
configure(..., storage_backend="none", freeze_backend="mongodb")   # snapshots only
```

---

## Selecting documents

The three functions take **the same selector as `delete_document()`**:

- `tenant_id` is keyword-only and mandatory, with no default. `None` or `"__public__"` means public documents. An empty string raises `ValueError`.
- Pass exactly one of `path` (all versions, or only `version=`) or `job_id` (one value or a list).
- `created_by_user_id` is an optional extra AND filter (one value or a list).
- The **`IS_COMPOSED_OF*` cascade goes downwards** from the matched documents:
  - Freezing a folder document freezes its whole subtree.
  - Freezing a leaf by `path` leaves its parent folder and its siblings alone.

```python
await freeze_document("reports/2024.pdf", tenant_id="acme")              # every version
await freeze_document("reports/2024.pdf", version=2, tenant_id="acme")   # one version
await freeze_document(job_id=["job-1", "job-2"], tenant_id="acme")       # two ingestion runs
await freeze_document("reports", tenant_id="acme")                       # a folder and everything below it
```

One call produces **one snapshot** holding one entry per document: the matched documents and their descendants.

---

## Freezing

```python
result = await freeze_document(
    "reports/2024.pdf",
    tenant_id="acme",
    keep_structure_nodes=False,
    keep_annotations=False,
    keep_extraction_results=False,
)
print(result.frozen_blob_id, result.structure_nodes_deleted, result.info_units_deleted)
```

Steps:

1. **Resolve** the documents.
   - If nothing matches, the result has `found=False` and nothing is touched.
   - If any of them is already frozen, `FreezeError` is raised. The one exception is a document whose freeze was interrupted (see [Memory and interruptions](#memory-and-interruptions)).
2. **Export** the complete snapshot to the freeze backend. The snapshot always holds the whole subtree, whatever the `keep_*` flags. This step is fail-fast: if it fails, the graph is not touched.
3. **Mark the stubs**, in one write transaction: each `:Document` gets `frozen=true`, `frozen_blob_id`, `frozen_at`, `frozen_keep_structure_nodes` / `frozen_keep_annotations` / `frozen_keep_extraction_results`, and `frozen_cleanup_pending=true`. The transaction first checks that the documents still exist and are not frozen, which catches a concurrent freeze. From this point every document points at the snapshot that holds its whole subtree.
4. **Delete the subtree**, in bounded transactions:
   - Re-link kept `:ModelDecision` / `:ExtractionResult` nodes to their `:Document` when their `:StructureNode` goes. These are the temporary `(:Document)-[:HAS_MODEL_DECISION|HAS_EXTRACTION]->()` links.
   - Delete what is not kept.
5. **Garbage-collect** the `:Entity` / `:ModelInstance` / `:LabeledEntity` nodes of the tenant that step 4 left orphaned, as `delete_document()` does (see [Garbage Collection](document-deletion.md#garbage-collection)): only what the deleted `:ExtractionResult` nodes pointed at is checked, not the whole tenant. With `keep_extraction_results=True` no `:ExtractionResult` is deleted, so there is nothing to collect.
6. **Remove `frozen_cleanup_pending`** from the stubs.

The duration of each phase is logged at `INFO` level at the end of the freeze (`timings: export=… upload=… mark=… delete=… gc=… clear_pending=…`): `export` is reading the graph into the snapshot file and `upload` storing it in the freeze backend (step 2), the rest are steps 3 to 6. Backup mode logs `export`, `upload` and `mark`; a call that only finishes an interrupted freeze has no `export` / `upload` / `mark`. If the freeze fails in step 4 or 5, the phases that finished are in the error log.

### Memory and interruptions

Steps 4 and 5 are **not atomic**. A transaction keeps everything it deletes in memory until it commits, and Neo4j caps the memory of all running transactions together (`dbms.memory.transaction.total.max`, a fixed size on Aura). Deleting the subtree of a few hundred documents in one transaction goes over that cap and fails with `MemoryPoolOutOfMemoryError`. So the subtree is deleted in pieces:

- the documents are processed 50 at a time;
- each delete commits every 1,000 nodes (`CALL { ... } IN TRANSACTIONS`);
- each query is retried on transient errors (deadlocks, a full memory pool, a lost connection or a leader change in a cluster).

If the freeze still fails in step 4 or 5, the documents are left as frozen stubs with part of their subtree in the graph and `frozen_cleanup_pending=true`. Nothing is lost, because the snapshot was stored before anything was deleted. There are two ways out:

- **Call `freeze_document()` again with the same selector.** It finishes the deletion without exporting again. The `keep_*` flags of the first call apply to those documents; the ones passed now are ignored for them. Documents of the selection that were not frozen yet are frozen as usual, with the flags passed now. The interrupted call took with it the list of nodes it had to check for orphans, so this one checks every `:Entity` / `:ModelInstance` / `:LabeledEntity` of the tenant instead, which takes longer on a large tenant.
- **Call `restore_document()`.** It rebuilds the documents from the snapshot, whatever is left in the graph.

`export_document_snapshot()` and backup mode refuse a document in this state, like any frozen document.

When a delete query had to be retried, the batches it committed before failing are not counted again, so the `*_deleted` counters of the result are then a lower bound.

### What the `keep_*` flags keep

| Node | Default | `keep_structure_nodes` | `keep_annotations` | `keep_extraction_results` |
|---|---|---|---|---|
| `:Document` | kept | kept | kept | kept |
| `:StructureNode` tree | deleted | **kept** | — | — |
| `:InfoUnit` | deleted | deleted | deleted | deleted |
| `:ModelDecision` + `:ProposedModel`, `:ProposedField`, `:ComplementaryMatch`, `:SupplementaryField` | deleted | — | **kept** | — |
| `:ExtractionResult` | deleted | — | — | **kept** |
| `:ModelInstance` / `:Entity` / `:LabeledEntity` | GC | — | — | kept while reachable from a kept `:ExtractionResult` |

The flags combine freely. InfoUnits always go: they are the bulk of a document's graph.

### Backup mode

`delete_after_export=False` only exports. The graph is not changed apart from `last_backup_blob_id` and `last_backup_at` on the `:Document` nodes, and `result.mode` is `"backup"`. The `keep_*` flags make no sense here, and passing any of them raises `ValueError`. A backed-up document is not frozen, so `restore_document()` refuses it while it exists. Once it has been deleted, the backup can be restored (see [Restoring a deleted document or a backup](#restoring-a-deleted-document-or-a-backup)).

### What stays outside the graph

Documentary storage is **not touched**: the raw files and converted pages stay in the storage backend, and a frozen `:Document` keeps its `raw_file_id`. Global catalog nodes (`:CatalogModel`, `:ModelField`, `:EntityLabel`, `:Theme`) are never deleted or exported as nodes; the snapshot only references them.

---

## Restoring

```python
result = await restore_document("reports/2024.pdf", tenant_id="acme")
print(result.documents_restored, result.nodes_created, result.snapshots_deleted)
```

The selector picks the frozen documents to restore. If the matched documents exist but are not frozen, `FreezeError("nothing to restore")` is raised. **Folder restore is symmetric to freeze**: restoring a folder also restores the descendants that were frozen in the same snapshot. A snapshot holding several documents can also be restored one document at a time, for example `version=1` only.

If the selector matches **no** `:Document` in the graph, the restore does not stop there: it looks for a snapshot in the freeze backend (see [Restoring a deleted document or a backup](#restoring-a-deleted-document-or-a-backup)).

Steps:

1. **Ensure the lookup indexes** the rebuild relies on: `:ModelDecision(uid)` and its children, and `:CatalogModel(name)`. A graph ingested before document freezing may lack them. They are created with `IF NOT EXISTS`, so this costs nothing when they already exist. A newly created index is waited for until it is online. If the database user lacks schema privileges, a warning is logged and the restore goes on, only slower.
2. **Download** every snapshot involved. The download is tenant-scoped: a snapshot of another tenant is "not found".
3. **Validate** each snapshot in a first streaming pass, without writing anything. The pass checks:
   - the schema version;
   - that the header, every document entry, every node and every linked `:Document` carry the tenant;
   - that labels and relationship types are valid identifiers;
   - that there is an entry for every document to restore;
   - that no selected `:Document` exists unfrozen, or frozen with another snapshot.

   Any problem raises `FreezeError` before any write.
4. **Rebuild** in a second streaming pass, in batches (see [Batch size and concurrency](#batch-size-and-concurrency)). The file is read incrementally with `ijson` and never loaded whole:
   - `:StructureNode` / `:InfoUnit` / `:ModelDecision` / `:ExtractionResult` and their children are `MERGE`d by business key (`id` / `uid`). A node that was kept is reused as is.
   - `:ModelInstance` / `:Entity` / `:LabeledEntity` are `MERGE`d by `uid`. A node still alive, because it is shared with another document of the tenant, keeps its properties and only gains the snapshot's `created_by_user_ids` / `job_ids` (set union).
   - Catalog nodes are `MERGE`d by their key, as the pipeline does.
   - Relationships are created only when an identical one (same type, endpoints and properties) does not already exist.
5. **Finish**, 50 documents at a time: remove the temporary `(:Document)-[:HAS_MODEL_DECISION|HAS_EXTRACTION]->()` links (in batches of 1,000, since there is one per kept node), then set `frozen=false` and remove `frozen_blob_id`, `frozen_at`, the `frozen_keep_*` properties and `frozen_cleanup_pending`. A document is unmarked only once its links are gone.
6. **Garbage-collect**, but only if the restore found `:ExtractionResult` nodes to delete on kept structure nodes — something was extracted onto the document while it was frozen. A restore otherwise only creates nodes, so nothing can have been orphaned, and the `gc_*` counters are 0. When it does run, it is a sweep of every `:Entity` / `:ModelInstance` / `:LabeledEntity` of the tenant, as [`collect_orphans()`](document-deletion.md#collect_orphans-sweeping-a-whole-tenant) does.
7. **Delete the snapshot** from the freeze backend once no `:Document` of the tenant references it any more. This applies only to the snapshots of a freeze; backups and exports are kept.

The duration of each phase is logged at `INFO` level at the end of the restore (`timings: indexes=… download=… validate=… stale_delete=… rebuild=… finish=… gc=…`).

### Restoring a deleted document or a backup

`restore_document()` also rebuilds a document that is no longer in the graph: one deleted with `delete_document()`, whether it was frozen or backed up before. It can also restore a specific snapshot.

```python
await restore_document("reports/2024.pdf", version=3, tenant_id="acme")                 # newest snapshot of v3
await restore_document(job_id="job-1", tenant_id="acme")                                # every document of the run
await restore_document("reports/2024.pdf", tenant_id="acme", frozen_blob_id=blob_id)   # this snapshot
```

- **Which snapshot.** Without `frozen_blob_id`, the freeze backend is searched (`FreezeRepository.find_snapshots`) for the snapshots of the tenant holding a matching document. **Each document is restored from its newest snapshot**, whatever its mode (freeze, backup or `destination="storage"` export). With `frozen_blob_id`, that snapshot is used, and its entries matching the selector are restored, plus their descendants in the snapshot.
- **The `:Document` is recreated** from the properties it had at export time (the `frozen*` ones excepted). Its links to other Documents are recreated where the other Document still exists: the folder's `IS_COMPOSED_OF` and the version chain's `HAS_NEWER_VERSION`. `latest` is then recomputed for its path, so it is `true` only on the newest version.
- **A document that exists unfrozen is refused** with `FreezeError` before any write, and so is one frozen with another snapshot. Restoring a backup never merges into a live document: delete it first.
- **The snapshot is kept** when it is a backup or an export: it is history. A freeze snapshot is deleted once no `:Document` references it.
- **Raw files are not in the snapshot.** `delete_document()` removed them from storage, so the recreated `:Document` keeps a `raw_file_id` that points nowhere, and a warning is logged. The graph is complete; only the original file is missing.

Limits:

- **Lookup by `job_id` / `created_by_user_id`** only finds snapshots stored since each document's `job_id` and `created_by_user_id` are recorded in the snapshot metadata. For older ones, select by `path`, or pass `frozen_blob_id`.
- **Folders and version links** are only in snapshots exported since they are recorded. An older snapshot restores a deleted document without re-linking it to its folder or to its other versions. It also cannot bring a folder's children along from a snapshot unless the selector matches them.
- **A custom backend** that does not implement `find_snapshots` raises `FreezeError`: pass `frozen_blob_id`.

### Batch size and concurrency

```python
await restore_document("reports/2024.pdf", tenant_id="acme", batch_size=1000, concurrency=4)  # defaults
await restore_document("reports/2024.pdf", tenant_id="acme", concurrency=1)  # while an ingestion is running
```

- **`batch_size`** is the number of rows per write transaction, 1000 by default. A larger batch means fewer round trips, but each transaction lives longer. It keeps its locks longer, so other writers that touch the same shared nodes wait longer: catalog nodes and the tenant's `:ModelInstance` / `:LabeledEntity`, such as a running ingestion. Lock contention is usually the limit, not memory.
- **`concurrency`** is the number of rebuild transactions in flight at once, 4 by default. Each one uses its own pooled connection. The reader waits for a free slot before preparing the next batch, so the Python side never holds more than `concurrency` batches in memory.

Neo4j's transaction memory (`dbms.memory.transaction.total.max`) does not grow with the size of the restore, only with what is in flight at once. Measured on Neo4j 2026.08 with synthetic folders of 150 and 732 documents (the latter 475,000 nodes and 520,000 relationships):

| | Peak transaction memory |
|---|---|
| Rebuild, defaults (1,000 × 4) | 8 MiB: 2 MiB per transaction in flight |
| Rebuild, `batch_size=4000`, `concurrency=8` | 16 MiB: still 2 MiB per transaction |
| Rebuild with 32 KB of text per `:InfoUnit` | 8 MiB for one `:InfoUnit` transaction |

2 MiB is the smallest amount Neo4j reserves for a transaction, so with nodes of ordinary size the peak is simply `2 MiB × concurrency`. Only large properties make a transaction grow, and then lowering `batch_size` brings it back down. The whole restore of the 732 documents, the finish step included, ran with the pool capped at 24 MiB.

How the work is spread:

- Node batches are grouped by label set. Each group is written by one transaction at a time, and different groups run in parallel.
- Every node of a document is written before its relationships, since relationships look up both endpoints.
- Relationship batches are grouped by (type, source label, target label). The groups that point at catalog nodes share a single lane, because every document points at the same few catalog nodes.
- Inside a batch, rows are sorted by key, so concurrent transactions take their locks in the same order. That makes deadlocks unlikely, and `execute_write` retries the ones that do happen.

### Changes made while frozen: the snapshot wins

Annotation and entity extraction only pick structure nodes that have InfoUnits, and a freeze always deletes the InfoUnits, so the pipeline does not touch a frozen document on its own. It can still change through `run_pipeline(update_mode=True)` on the frozen version, or through hand-written Cypher. On restore, for each family the freeze **removed** (annotations when `keep_annotations=False`, extraction results when `keep_extraction_results=False`), whatever hangs from the kept structure nodes is deleted before the snapshot is rebuilt. This uses the pipeline's own idempotency helpers, so the restored document is exactly the snapshot.

The families the freeze **kept** were never removed and are not replaced: they are live graph data, and changes to them while the document was frozen stay.

### Idempotent and retryable

Every write of the rebuild is idempotent (`MERGE`, relationship creation guarded by `NOT EXISTS`). If a restore fails half-way, the documents stay frozen and the snapshot stays stored. Call `restore_document()` again and it completes the restore.

---

## Exporting without freezing

```python
snapshot = await export_document_snapshot("reports/2024.pdf", tenant_id="acme")                 # dict
path = await export_document_snapshot("reports/2024.pdf", tenant_id="acme",
                                      destination="file", file_path="/tmp/2024.json")            # Path
blob_id = await export_document_snapshot("reports/2024.pdf", tenant_id="acme",
                                         destination="storage")                                  # frozen_blob_id
```

The export is read-only.

- `"dict"` builds the snapshot in memory. Use it for small documents and debugging.
- `"file"` and `"storage"` stream it element by element.
- Only `"storage"` needs a freeze backend.

A frozen document cannot be exported (`FreezeError`), because its subtree is no longer in the graph. Its snapshot is already in the freeze backend. `keep_flags` is `null` in the header of a standalone export.

---

## Snapshot format

```json
{
  "schema_version": 1,
  "tenant_id": "acme",
  "frozen_at": "2026-09-29T10:00:00.000000+00:00",
  "mode": "freeze",
  "keep_flags": {"structure_nodes": false, "annotations": false, "extraction_results": false},
  "documents": [
    {
      "document": {"path": "reports/2024.pdf", "version": 1, "tenant_id": "acme", "...": "..."},
      "nodes": [
        {"labels": ["StructureNode", "Section"], "key": {"id": "..."}, "properties": {"tenant_id": "acme", "...": "..."}}
      ],
      "relationships": [
        {"type": "HAS_STRUCTURE",
         "from": {"labels": ["Document"], "key": {"tenant_id": "acme", "path": "reports/2024.pdf", "version": 1}},
         "to":   {"labels": ["StructureNode", "Section"], "key": {"id": "..."}},
         "properties": {}}
      ]
    }
  ]
}
```

- **Business keys, not element ids.** Each node carries its primary label first and a `key`:
  - `:Document` — `(tenant_id, path, version)`
  - `:StructureNode` — `id`
  - catalog nodes — `name`, `(name, model)`, `label` or `path`
  - everything else — `uid`

  `properties` holds the remaining properties. Relationships reference their endpoints by labels and key.
- **`mode`** is the call that wrote the snapshot: `"freeze"`, `"backup"` or `"export"`. Snapshots written before it was recorded have no `mode`.
- **One entry per document, deduplicated within the entry.** A node reachable along two paths appears once per entry. A shared `:ModelInstance` referenced by two documents of the same snapshot appears in both entries.
- **What an entry contains:**
  - the whole `HAS_STRUCTURE` / `HAS_CHILD` tree;
  - what hangs from each structure node: `:InfoUnit`, `:ModelDecision` and its children, and `:ExtractionResult`;
  - everything reachable from each `:ExtractionResult` through outgoing relationships within **7 hops**, catalog nodes excluded. This is the same bound the GC uses to decide whether a `:ModelInstance` / `:Entity` is still reachable, so export, GC and `keep_extraction_results` agree on what hangs from an extraction result.

  - the `:Document`'s own links to other Documents of the tenant, in both directions: `IS_COMPOSED_OF` (folder) and `HAS_NEWER_VERSION` (version chain). They let a deleted document be re-linked on restore.

  Other relationships whose target lies outside the entry (beyond the bound, or in another document) are dropped. Relationships to catalog nodes are kept as references.
- **Timestamps are ISO-8601 strings**, as the pipeline stores them. A Neo4j temporal value set by hand is also written as an ISO string, and is restored as a string.

---

## Tenant scope

Freezing never crosses tenants:

- Every Cypher query filters on the tenant, the GC included.
- The snapshot is stored with the tenant.
- Reads and deletes in the freeze backend only find snapshots of the caller's tenant.
- The export checks that every node of the subtree carries the document's tenant, and raises `FreezeError` otherwise. A node **without** `tenant_id`, for example from a graph written before multi-tenancy, is also an error: such documents cannot be frozen until their nodes carry the tenant.
- The restore checks the tenant of the snapshot header, of every entry and of every node before writing anything.

As with `delete_document()`, who may freeze or restore public documents (`tenant_id=None`) is an authorization decision for your API layer.

---

## Results

`FreezeResult`:

| Field | Description |
|---|---|
| `path`, `version`, `job_id`, `tenant_id`, `created_by_user_id` | The selector, echoed back as passed. |
| `found` | `False` when nothing matched (nothing touched). |
| `mode` | `"freeze"` or `"backup"`. |
| `frozen_blob_id` | Id of the snapshot in the freeze backend. When the call only finished an interrupted freeze, the snapshot of the first matched document. |
| `versions_frozen` | Sorted versions of the matched documents. |
| `documents_frozen` | Matched documents plus their `IS_COMPOSED_OF*` descendants. |
| `structure_nodes_deleted` / `_kept`, `info_units_deleted`, `model_decisions_deleted` / `_kept`, `proposed_models_deleted`, `proposed_fields_deleted`, `extraction_results_deleted` / `_kept` | What the mutation deleted or kept (0 in backup mode). A lower bound if a delete had to be retried, and only what was left when the call finished an interrupted freeze. |
| `gc_*` | Same meaning as in `DeletionResult`: the orphans this freeze caused (0 with `keep_extraction_results=True`). |

`RestoreResult`:

| Field | Description |
|---|---|
| `path`, `version`, `job_id`, `tenant_id`, `created_by_user_id` | The selector, echoed back as passed. |
| `found` | `False` when nothing matched. |
| `versions_restored` | Sorted versions of the frozen matched documents. |
| `documents_restored` | Documents restored, descendants included (stubs and recreated ones). |
| `documents_recreated` | How many of them no longer existed and were recreated. |
| `frozen_blob_ids` | Snapshots the documents were restored from. |
| `snapshots_deleted` | Snapshots deleted because no document references them any more. |
| `nodes_created` / `nodes_reused` | Snapshot nodes created, or found and reused (kept nodes, shared entities). Catalog nodes are not counted. |
| `relationships_created` | Relationships recreated (existing identical ones are not duplicated). |
| `gc_*` | Counters of the tenant sweep of step 6; 0 when it was skipped. |

---

## Caveats

- **Ingestion does not check `frozen`.** `run_pipeline(update_mode=True)` re-ingests the `latest` version in place: if that version is frozen, it recreates the structure tree under the stub, which stays marked `frozen`, and a later restore merges the snapshot on top of it. Guard against it in your API layer by checking the `frozen` property of the `:Document`. Annotation and entity extraction skip frozen documents naturally, because they have no InfoUnits. See also [Changes made while frozen](#changes-made-while-frozen-the-snapshot-wins).
- **`delete_document()` on a frozen document** leaves the snapshot and, with `keep_structure_nodes=False`, the kept annotations and extraction results behind. Restore first, then delete (see [Document Deletion — Frozen Documents](document-deletion.md#frozen-documents)).
- **Indexes.** The restore looks nodes up by `uid` and creates the indexes it needs if they are missing (step 1). Creating an index on a large graph takes a while the first time, and needs schema privileges; without them the restore still works, but each lookup scans the label.
- Until the first freeze in a database, the Neo4j driver may log a notification that the property `frozen` does not exist. It is harmless.

---

## See Also

- **[Document Deletion](document-deletion.md)** — The irreversible counterpart, with the same selector and GC.
- **[Storage Backends](storage-backends.md)** — The MongoDB backend the freeze backend inherits from.
- **[Neo4j Graph Storage](neo4j-graph.md)** — The node types a snapshot contains.
- **[Freezing API](../api/freezing.md)** — Auto-generated reference.
