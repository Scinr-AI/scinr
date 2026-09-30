# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-09-30

Multi-tenancy and document freezing, both implemented and pushed.

The tenant is now part of every document's identity (see
`plans/multitenancy-document-identity-plan.md`), and every navigation function
and `delete_document()` filters on tenant, user and job (see
`plans/multitenancy-navigation-deletion-plan.md`). The document storage
(MongoDB / GridFS or a custom backend) now stores the tenant too and filters by
it (see `plans/multitenancy-document-storage-plan.md`). Both databases now have
the indexes those tenant-scoped reads need (see
`plans/multitenancy-indexes-plan.md`). The command-line interface is removed:
`scinr` is used as a library, through `configure()` and `run_pipeline()` /
`run_*()` (see `plans/remove-cli-plan.md`). A document's subgraph can now be
frozen to a snapshot in the document store and restored later (see
`plans/document-freezing.md` and `plans/document-freezing-implementation-plan.md`).

### Added
- **Document freezing.** `freeze_document()` exports the subgraph of a document
  to a snapshot in the freeze backend and reduces the document in Neo4j to a
  stub: the `:Document` stays (`frozen=true`, `frozen_blob_id`, `frozen_at`),
  its structure tree, InfoUnits, annotations and extraction results go.
  `restore_document()` rebuilds it from the snapshot, with no LLM calls. Both
  take the same tenant-scoped selector as `delete_document()` (`path` /
  `version` or `job_id`, mandatory `tenant_id`, optional `created_by_user_id`)
  and cascade downwards through `IS_COMPOSED_OF*`. See the new "Document
  Freezing" user guide.
  - `keep_structure_nodes`, `keep_annotations` and `keep_extraction_results`
    leave those node families in the graph; InfoUnits always go.
  - `delete_after_export=False` is a **backup**: only the snapshot is stored
    (`last_backup_blob_id` / `last_backup_at` on the `:Document`).
  - The snapshot is stored before anything is deleted, and the deletion runs in
    bounded transactions. An interrupted freeze leaves
    `frozen_cleanup_pending=true`; calling `freeze_document()` again finishes it
    without exporting again.
  - The restore validates the whole snapshot (schema version, tenant of every
    entry and node) before any write, reads it incrementally, and writes
    idempotent batches (`batch_size`, `concurrency`), so a failed restore can be
    called again. Changes made to a frozen document are discarded for the node
    families the freeze removed: the snapshot wins.
  - `restore_document()` also recreates a document that is no longer in the
    graph (deleted after a freeze or a backup) from its newest snapshot, or from
    the one passed as `frozen_blob_id`, re-linking it to its folder and version
    chain. A freeze snapshot is deleted once no `:Document` references it;
    backups and exports are kept. A document that exists unfrozen is refused.
    Raw files are not part of a snapshot: a recreated `:Document` keeps its
    `raw_file_id` and a warning is logged when the stored file is gone.
  - Garbage collection: the freeze collects the orphans it causes (see
    "Garbage collection is scoped to the operation" below). The restore only
    creates nodes, so it collects nothing, unless something was extracted onto
    the kept structure nodes of the frozen document; it then sweeps the tenant.
  - Both log the duration of each phase at `INFO` level (`timings: export=…
    upload=… mark=… delete=… gc=… clear_pending=…` for the freeze, `indexes=…
    download=… validate=… stale_delete=… rebuild=… finish=… gc=…` for the
    restore).
  - Transaction memory stays bounded whatever the size of the operation:
    measured on a folder of 732 documents (475,000 nodes), the peak is 13 MiB
    for the freeze and 8 MiB for the restore (2 MiB per rebuild transaction in
    flight).
  - Known limits: ingestion does not check `frozen` (`update_mode=True` on a
    frozen version rebuilds its structure under the stub), and
    `delete_document()` on a frozen document leaves its snapshot in the freeze
    backend. See "Caveats" in the guide.
- `export_document_snapshot()`: the same snapshot without touching the graph,
  as a `dict`, a file or a stored snapshot (`destination="dict" | "file" |
  "storage"`).
- **Freeze backend.** `configure(freeze_backend=...)` / `FREEZE_BACKEND`
  (`none`, `mongodb`, `custom`); unset, it inherits the resolved
  `storage_backend`. MongoDB stores the snapshot in GridFS
  (`mongodb_frozen_gridfs_bucket` / `MONGODB_FROZEN_GRIDFS_BUCKET`, default
  `frozen_snapshots`) and its metadata in `mongodb_frozen_collection` /
  `MONGODB_FROZEN_COLLECTION` (default `frozen_documents`), with the indexes
  `frozen_by_tenant_path_version`, `frozen_by_tenant_job`,
  `frozen_by_tenant_document` and `frozen_by_tenant_document_job`.
  `custom_freeze_storage` takes a `scinr.newton.freeze.base.FreezeRepository`
  (`store_snapshot`, `read_snapshot_to_file`, `delete_snapshot`, and optionally
  `find_snapshots`), every method tenant-scoped.
- `FreezeError`, `FreezeResult` and `RestoreResult`, exported from
  `scinr.newton`.
- `collect_orphans(tenant_id=...)`: sweeps every orphaned `:Entity` /
  `:ModelInstance` / `:LabeledEntity` of one tenant and returns an
  `OrphanCollectionResult`. For the orphans left by other write paths
  (`update_mode=True` re-ingestion, re-running the extraction on a node). Its
  cost grows with the tenant; do not run it while that tenant is being
  ingested.
- `:Document` properties written by these operations: `frozen`,
  `frozen_blob_id`, `frozen_at`, `frozen_keep_structure_nodes`,
  `frozen_keep_annotations`, `frozen_keep_extraction_results`,
  `frozen_cleanup_pending`, `last_backup_blob_id`, `last_backup_at` and
  `deletion_pending`.
- Documentation: new "Document Freezing" user guide and "Freezing" API page;
  the deletion guide documents the bounded transactions, the scoped garbage
  collection and `collect_orphans()`.
- Neo4j indexes on `uid` for `:ModelDecision`, `:ProposedModel`,
  `:ProposedField`, `:ComplementaryMatch` and `:SupplementaryField`, and on
  `:CatalogModel(name)`. `restore_document()` creates them if they are missing
  (`ingest.schema.ensure_indexes()`).
- `ijson>=3.2` as a base dependency (incremental parsing of snapshots).
- **Scope filters on every navigation method.** All `GraphNavigator` methods
  except the global-catalogue ones (`list_catalog_models`, `get_catalog_graph`,
  `list_themes`, `list_relationship_types`, `list_node_labels`) and `execute_raw`
  take keyword-only `tenant_id`, `include_public`, `created_by_user_id` and
  `job_id`. `tenant_id=None` means **no tenant filter**, `"__public__"` only
  public documents, `"acme"` only that tenant, and `include_public=True` adds the
  public ones to it. `created_by_user_id` / `job_id` accept one value or a list
  (any-of); an empty list raises `NavigationError`. See `navigation/scope.py`.
- `nav.scoped(tenant_id=..., include_public=..., created_by_user_id=..., job_id=...)`
  returns a navigator with the scope fixed once; it rejects any call that would
  widen it and refuses `execute_raw`. Recommended for multi-user APIs, since the
  unscoped default is "all tenants".
- Single-document methods (`get_one_document`, `get_latest_version`,
  `get_document_tree`, `get_document_parent`, `get_document_ancestors`,
  `get_document_stats`, `get_document_model_profile`, `get_annotation_coverage`)
  raise `NavigationError` when the path exists in several tenants of the scope;
  with `include_public=True` a tenant's document shadows the public one.
- `delete_document()` accepts one value or a list for `job_id` and
  `created_by_user_id` (`IN`); an empty list raises `ValueError`.
- Indexes `idx_info_unit_tenant_id` and `idx_model_decision_tenant_id`.
- **Storage is tenant-aware.** Every `raw_files` record, its GridFS metadata and
  every `converted_pages` record store `tenant_id` (`"__public__"` for public
  uploads, never `null`), `created_by_user_id` and `job_id`. `RawFileRecord` /
  `ConvertedPageRecord` expose them.
- `RawFileRepository.get()`, `open()` (async stream of the binary),
  `open_with_record()` (record + stream from a single lookup) and
  `list_raw_files()` (inventory), `PageRepository.get_pages_by_ids()` (only the
  given pages, e.g. a structure node's `source_page_ids`), and
  `get_document_original()` returning an
  `OriginalFile` (record + stream), so consumers no longer need to go to GridFS
  directly.
- **Source text on the navigator.** `GraphNavigator` gains concrete methods
  `get_structure_nodes_source_pages`, `get_info_unit_source_text`,
  `get_document_source_text` and `get_document_original` (delegating to
  `navigation.pages`, so graph backends need not implement them). They take the
  four scope filters and a `ScopedNavigator` applies its scope to them. The
  `navigation.pages` functions now accept the scope filters too.
- **Source pages of several structure nodes in one call.**
  `get_structure_nodes_source_pages(node_ids)` (navigator method and
  `navigation.pages` function) looks all the nodes up in one graph query and
  reads their pages with one `get_pages_by_ids()` per stored tenant of the
  nodes. It returns a `StructureNodesSourcePages`: each page once in `pages`
  (`page_id → PageText`), each node's `page_ids` (by page index) in a
  `StructureNodeSourcePages`, and status groups instead of errors —
  `not_found_structure_nodes` (missing or outside the scope),
  `structure_nodes_without_pages` and, per node, `not_found_page_ids` (missing
  or of another tenant).
- `GraphNavigator.get_structure_nodes_by_ids(node_ids)` (abstract): the
  structure nodes with these ids in one `n.id IN $ids` lookup, in request order,
  found only.
- `PageText` gains `raw_file_id`, `filename` and `folder_path`.
- Storage reads / deletes take the navigation scope (`tenant_id`,
  `include_public`, `created_by_user_id`, `job_id`) with identical semantics:
  `None` = all tenants (legacy records included). A record outside the scope
  behaves as missing. `storage.filters.mongo_scope_filter()` renders it for
  MongoDB.
- `ScopeError` (subclass of both `NavigationError` and `StorageError`) for an
  invalid scope. The scope moved to `utils/scope.py`; `navigation.scope`
  re-exports it.
- `run_preprocess()`, `convert_one()`, `convert_folder()` and
  `convert_single_file()` take keyword-only `tenant_id` / `created_by_user_id` /
  `job_id`, written on the stored raw file and pages and stamped on the
  `IntermediateDocument` (new fields, serialized in the intermediate JSON).
  `run_pipeline()` and `run_tabular_pipeline()` pass theirs.
- Extraction inherits the owner stamped on the intermediate document when the
  run passes none.
- MongoDB indexes are tenant-prefixed: `pages_by_tenant_raw_file_and_index`,
  `pages_by_tenant_filename_folder`, `raw_files_by_tenant_checksum`, plus
  `raw_files_by_tenant_user` / `raw_files_by_tenant_job`; `ensure_indexes()`
  drops the previous ones.
- **Composite tenant indexes in Neo4j:** `ModelInstance(tenant_id, model_class)`,
  `Document(tenant_id, latest)`, `Document(tenant_id, name)`,
  `StructureNode(tenant_id, role)` and `LabeledEntity(tenant_id, label)`. Neo4j
  uses one index per node, so "class X of tenant T" used to seek on one single
  index and filter every node of the tenant for the other property.
- **`instance_key` fields are indexed per tenant.** `ensure_catalog_models()`
  creates `idx_mi_key_<field>` on `ModelInstance(tenant_id, <field>)` for every
  distinct `instance_key` field name, so a search on part of a composite key is
  an index seek.
- `configure(mongodb_ensure_indexes=...)` / `MONGODB_ENSURE_INDEXES` (default
  `True`): set it to `False` when the MongoDB user lacks the `createIndex`
  privilege and operations manage the indexes.
- `scinr.newton.storage.mongodb.client.ensure_indexes_sync(cfg)`: the pymongo
  counterpart of `ensure_indexes()`, sharing one index definition.
- `tests/integration/test_index_usage.py`: checks with `EXPLAIN … USING INDEX`
  (Neo4j) and `explain()` (MongoDB) that the representative tenant-scoped
  queries can use their index; and a unit test that fails on any unlabelled
  `(var {prop: …})` node pattern in the library.

### Changed
- **`delete_document()` deletes in bounded transactions.** The cascade used to
  run in one transaction, which failed with `MemoryPoolOutOfMemoryError` on a
  folder of a few hundred documents (Neo4j caps the memory of all running
  transactions, a fixed size on Aura). The `:Document` nodes are now marked
  `deletion_pending`, processed 50 at a time, each delete commits every 1,000
  nodes, and the `:Document` nodes go last. The cascade is no longer atomic:
  if it fails half-way, calling `delete_document()` again with the same
  selector deletes what is left. When a query had to be retried, the
  `*_deleted` counters are a lower bound.
- **Garbage collection is scoped to the operation.** `delete_document()` (and
  `freeze_document()`) only check the `:Entity` / `:ModelInstance` /
  `:LabeledEntity` nodes the deleted `:ExtractionResult` nodes pointed at,
  within the deletion's tenant, instead of every such node of the graph (all
  tenants included), so the cost follows the size of the deletion. The
  `gc_*_deleted` counters of `DeletionResult` are the orphans
  this deletion caused, and `gc_*_passes` the rounds over its candidates (`0`
  when there was none). A call that finishes an interrupted delete or freeze
  sweeps the whole tenant instead. Orphans from other write paths are left to
  `collect_orphans()`. What makes a node an orphan has not changed (no
  `:ExtractionResult` within 7 hops; no incoming relationship for a
  `:LabeledEntity`), and the collection now follows a chain of orphans to its
  end instead of stopping after 7 passes.
- The idempotency deletes of the annotation and extraction writers are shared
  helpers: `annotation.neo4j_ops.delete_stale_model_decision()` and
  `entity_extraction.graph_mapper.delete_stale_extraction_result()`, used by
  `write_annotation()`, `write_manual_annotation()`,
  `write_extraction_subgraph()`, `delete_tabular_subgraph()` and
  `restore_document()`. They take one `StructureNode.id` or several. No change
  of behaviour.
- Neo4j write retries also cover `SessionExpired` and every error the driver
  declares retryable (`NotALeader`, `ForbiddenOnReadOnlyDatabase`,
  `AuthorizationExpired`), i.e. a leader change in a cluster.
- `get_gridfs_bucket()` takes an optional `bucket_name` (default: the raw-files
  bucket).
- **The library creates the MongoDB indexes itself.** The first `get_storage()`
  of the process (per MongoDB target) creates them; there is no longer anything
  to call at application startup. A failure (typically a missing `createIndex`
  privilege) is logged as a warning and does not block storage access.
- **The MongoDB connection is checked once per process.** `get_storage()` used to
  open a new `MongoClient` and ping the server on every call — i.e. on every
  source-text read. The ping (and the index bootstrap, on the same client) now
  runs once per MongoDB URI / database / collections, and again after
  `configure()`.
- **Faster source text for structure nodes and info units.**
  `get_structure_nodes_source_pages` / `get_info_unit_source_text` (and
  `describe_node(include_source_text=True)`) look the node up once and read only
  their `source_page_ids` with `get_pages_by_ids()`, filtered by the node's stored
  tenant. They no longer resolve the owning `:Document` (a variable-length
  traversal) nor load every page of the document to filter them in Python.
  - An info unit now costs one graph query instead of three, and
    `describe_node` reuses the node it has already resolved.
  - `get_document_original` reads the raw-file record once (`open_with_record`)
    instead of twice (`get` + `open`).
- **Breaking — storage repository contract.** `RawFileRepository.get`, `open`,
  `open_with_record` and `list_raw_files`, and `PageRepository.get_pages_by_ids`,
  are new abstract methods, and every method takes keyword-only
  owner arguments (writes: `tenant_id`, `created_by_user_id`, `job_id`) or scope
  arguments (reads / deletes). `custom` storage backends must be updated.
- **Breaking — ingestion verifies `raw_file_id`.** A document with a non-empty
  `raw_file_id` is only ingested if the stored raw file exists and belongs to the
  document's effective tenant (a tenant never claims a public file); otherwise
  the unit fails with `IngestionError` before any graph write. With
  `storage_backend="none"` a non-empty `raw_file_id` is rejected. Checked by
  `run_pipeline`, `run_ingestion`, `ingest_one` and `ingest_one_from_path` (the
  synchronous `load_*` primitives do not check).
- `delete_document()` deletes each stored original scoped to the tenant of the
  graph node carrying its `raw_file_id`, so a forged id no longer deletes another
  tenant's files. `navigation.pages.*` read pages filtered by the document's
  tenant.
- Storage records written before this release have no `tenant_id` and are only
  reachable with `tenant_id=None` (no backfill).
- Documented the tenant-isolation invariant of the navigation API: no data
  relationship crosses tenants (public included), so the tenant is checked on
  the starting node only. Trade-off: a tenant's content is never linked to
  public content (see "Why a traversal cannot leave the tenant" in the graph
  navigation guide).
- **Breaking — `"__public__"` is accepted as an input tenant** and means the same
  as `None` (public). Only `""` is rejected. `make_instance_uid()`,
  `delete_document()` and ingestion treat both identically. `tenant_from_key()`
  is removed.
- **Breaking — `*Ref.tenant_id` exposes the stored value**: a public document
  reports `"__public__"` instead of `None` (`None` is now reserved for "no
  filter" on reads).
- **Breaking — `get_model_instance_by_key(tenant_id=None)` raises
  `NavigationError`**: the instance `uid` embeds the tenant, so the lookup needs a
  concrete tenant or `"__public__"` (before, `None` rebuilt the public uid).
- `get_model_instances_by_class`, `get_document_model_instances`, … now accept
  lists for `created_by_user_id` / `job_id`; filtering on a real tenant no longer
  hides public data when `include_public=True`.
- `DeletionResult.job_id` / `created_by_user_id` echo what was passed (a string
  or a list).

- **Breaking — `:Document` key is `(tenant_id, path, version)`.** Two tenants
  ingesting the same path get fully independent documents, folders, versions,
  structure, annotations and extractions. `setup_schema()` drops
  `constraint_document_path_version` and creates
  `constraint_document_tenant_path_version`. Versions are numbered per tenant.
- **Breaking — public documents are stored with `tenant_id = "__public__"`**
  (`utils.tenancy.PUBLIC_TENANT`), never `null`. The public API still uses
  `None` (and `"__public__"`, see above); `""` as a tenant raises `ValueError`.
- **Breaking — `StructureNode.id` is `{tenant}::{doc_path}::{version}::{node_path}`**,
  so every uid derived from it (InfoUnit, ModelDecision, ExtractionResult, tabular
  rows) is tenant-scoped. `make_instance_uid()` and the `LabeledEntity` / `Entity`
  uids hash the stored tenant key (values differ from the previous formula).
- **Breaking — `delete_document()` requires `tenant_id`** (keyword-only, no
  default; `None` = public documents) and always filters on it.
- Annotation and entity extraction select documents by `(tenant_id, doc_path)`
  instead of by name: `run_annotation()`, `run_entity_extraction()` and their
  agents accept keyword-only `tenant_id` / `doc_path`. `resolve_leaf_document_names*`
  is replaced by `resolve_leaf_documents*`, returning `(name, path)` leaves of one
  tenant. `run_pipeline(document_names=...)`, `replaces`, `update_mode` and batch
  version resolution are tenant-scoped too.
- The tabular write path `MATCH`es its `:Document` instead of `MERGE`-ing it, so a
  failed document creation can no longer leave an empty `:Document` behind.
- **Breaking — `preflight_check_replaces()` raises `PreconditionError`** instead of
  `SystemExit` when the document to replace is missing or ambiguous, so
  `run_pipeline(replaces=...)` no longer exits the host process.

### Removed
- **Breaking — the command-line interface.** The `newton` command
  (`scinr.newton.cli` and its `[project.scripts]` entry) is gone, and so are the
  module-level entry points (`python converters/main.py`, `python -m
  ingest.loader`, and running `annotation/agent.py` / `entity_extraction/agent.py`
  as scripts). Use `run_pipeline()` or the `run_*()` stage functions: `stages=`,
  `input_raw=`, `update_mode=True`, `replaces=`, `parallel_docs=`,
  `context_instructions=`, plus `tenant_id` on every stage. The library functions
  of those modules (`convert_one`, `convert_folder`, `convert_single_file`,
  `convert_api`, `load_*`, `ingest_one*`) stay. The environment variables are
  still read by `configure()`.
- `scinr.newton.utils.file_archiver` and `scinr.newton.converters.config`
  (default folders and the `--dev` output directory), used only by the CLIs.
- `navigation.pages.get_node_source_text` and `get_node_source_page_ids`
  (without a deprecation period; the library has no external users yet).
  Structure nodes are read in batches only: use
  `get_structure_nodes_source_pages([node_id])` for the text, and
  `StructureNodeRef.source_page_ids` from `get_structure_nodes_by_ids()` /
  `get_structure_node()` for the page ids without storage. The interim names
  `get_structure_node_source_text` / `get_structure_node_source_page_ids` are
  not shipped either.

### Fixed
- **`graph_mapper` writes scanned the whole graph.** Eight statements in
  `entity_extraction/graph_mapper.py` located the parent node (or the source of an
  instance relationship) with `MATCH (parent {uid: $parent_uid})` — no label, so no
  index: every field and ModelInstance written scanned every node of every tenant.
  They now carry the label (`ExtractionResult` or `ModelInstance`) and seek on its
  uid constraint.
- `find_shell_model_instances` filtered the tenant and the model class after a
  `WITH`, where no index can apply; both `MATCH`es now filter in their own `WHERE`
  and seek on `(tenant_id, model_class)`. With `model_class` the average is also
  computed over that class only instead of every class in scope.
- `setup_schema()`'s Neo4j version check read `dbms.components()` with
  `.single()`; since Neo4j 2025.x the procedure also returns a `Cypher` row, which
  raised a "found multiple" warning. It now selects the `Neo4j Kernel` row.
- **Security: MongoDB credentials leaked through error messages.** When MongoDB
  was unreachable, `get_storage()` raised `StorageError("Cannot connect to
  MongoDB at '<MONGODB_URI>' …")` with the full URI, password included. That
  message reached the logs and, through `StageResult.errors` /
  `DocumentResult.errors`, any caller returning pipeline errors to a frontend
  (ingestion via `preprocess`, `tabular`, `delete_document()`, and
  `navigation.pages.*`). The Motor client also logged the full URI at DEBUG
  level. Now:
  - the URI is shown with its password masked (`mongodb://user:***@host`);
  - `DocumentResult`, `StageResult` and `UnitResult` scrub every error string
    they hold (URI passwords plus the configured Neo4j password, MongoDB
    password and Mistral API key), whatever stage produced it;
  - `ScinrConfig.__repr__` masks `neo4j_password`, `mistral_api_key` and the
    URI passwords (it ended up in tracebacks that capture locals);
  - `GraphConnectionError` messages, the API converters' `ConversionError`
    messages and their log lines are scrubbed too;
  - the log handlers installed by `setup_logging()` and `configure()` (only the
    one its `basicConfig` creates) use the new
    `RedactingFormatter`, which also masks exception tracebacks. Applications
    with their own logging setup should call
    `scinr.newton.utils.logging_config.redact_handlers()` once after it.
  New helpers: `scinr.newton.utils.redaction.redact_uri` / `redact_secrets`.
- `navigation.pages.get_document_source_text()` imported a helper removed with
  the navigation scope work and failed on every call; it now accepts a path or a
  `DocumentRef` (whose tenant pins the lookup).

## [0.3.10] - 2026-09-21

Memory optimization: the peak memory of each stage now depends on the size of one
unit of work (a PDF chunk of at most ~45 MB, a batch of 500 tabular rows, one node
in flight per LLM slot) rather than on the size of the file or the number of nodes
in the document.

### Changed
- **Large PDFs are converted one chunk at a time.** `PdfConverter` no longer reads
  the whole file nor materialises every chunk: it probes the size/page count, then
  generates chunks lazily from the file (`pdf_splitter.iter_pdf_chunks`), using a
  fresh `PdfReader` per window of pages. Live memory for a 938 MB PDF drops from
  ~4.3 GB to ~330 MB and no longer grows with the file. `split_pdf()` /
  `needs_splitting()` are unchanged. **Behavior change:** because chunks are now
  produced on demand, a `PdfSplitError` (a single page heavier than
  `mistral_ocr_safe_max_bytes`) can be detected *after* earlier chunks were already
  sent to Mistral OCR (previously it failed before sending anything). The document
  still always aborts, regardless of `mistral_ocr_error_strategy`. Chunk labels in
  log/error messages are now `chunk N` (the total is not known up front); the
  `[start, end)` page range and the `best_effort` hints are unchanged.
- **Raw files are streamed to GridFS** instead of being read fully into memory.
  `RawFileRepository` gains a non-abstract `store_file(path, filename, content_type,
  folder_path)`; the default implementation reads the file and delegates to
  `store()`, so custom backends keep working unchanged, and may override it to
  stream. The MongoDB backend hashes (SHA-256) and counts bytes on the fly.
- **`convert_one(output_dir=None)` writes nothing to disk** and returns
  `(entry, None, doc)`. The pipeline now uses it when `converter_output_dir` is not
  set, instead of serialising the whole document to a temporary directory that was
  deleted immediately. `_process_document_unit` also releases the intermediate and
  extracted documents as soon as their stage is done.
- **Tabular files are streamed.** Sheets are scanned once for headers, row count and
  a 5-row preview (`scan_tabular_file`); rows are then re-read in batches of 500
  (`iter_sheet_batches`) when written, so the LangGraph state (`TabularFileData`)
  no longer carries `all_rows` and the write path uses O(batch) memory. The
  normalization path is three streaming passes (scan → LLM on unique keys → write),
  with the row's normalization keys computed by a single function
  (`compute_row_normalization_keys`) in both scan and write passes, and the
  normalization LLM tasks are created lazily (at most `llm_concurrency` alive)
  instead of one per key batch up front.
  `NormalizationEntry.row_indices` is deprecated and no longer populated.
  The source file must not be modified while a sheet is being processed.
  `write_tabular_subgraph()` now takes a `row_batches` factory instead of reading
  `sheet["all_rows"]`.

### Added
- `pdf_splitter.probe_pdf(path)` and `pdf_splitter.iter_pdf_chunks(path, ...)`:
  path-based, lazy counterparts of `count_pdf_pages` / `split_pdf`.
- `RawFileRepository.store_file()` (optional override, see above).
- `tabular.reader.scan_tabular_file()` and `tabular.reader.iter_sheet_batches()`:
  streaming counterparts of `read_tabular_file()`, which is kept and still returns
  `all_rows`.
- `tabular.neo4j_ops.compute_row_normalization_keys()`: the single source of a row's
  normalization keys, shared by the scan and write passes.

### Fixed
- **Annotation (Stage 3) and entity extraction (Stage 4) no longer accumulate the
  context, prompt and schema of every node before the LLM semaphore.** The LLM slot
  is now taken first, so only `llm_concurrency` nodes hold that memory at once
  (10 000 nodes: +701 MB → +15 MB in the fan-out simulation).
- `read_csv` uses ~2.4× less memory (single streaming pass, no `StringIO` copy).
- `read_xlsx` closes the workbook even when reading a sheet fails.
- Raw files are no longer read into memory when `storage_backend="none"`.

## [0.3.9] - 2026-09-16

### Fixed
- **`delete_document()` no longer leaves folder-parent (and other
  structure-less) `:Document` nodes behind.** The cascade-delete Cypher query
  chained two `UNWIND` clauses, and `UNWIND` on an empty list silently drops
  the row — so any `:Document` with no `:StructureNode` of its own (every
  folder-parent Document created purely to model the path hierarchy, plus any
  leaf never fully processed) never reached the final `DETACH DELETE` and
  survived. Deleting by a folder `path` (or a `job_id`/selector whose cascade
  included folders) removed the real content documents underneath but left
  the folder nodes themselves — and any intermediate subfolders — orphaned
  in Neo4j. The query now guards the structure-node `UNWIND` with
  `CASE WHEN ... THEN [NULL] ELSE ...` so every matched `:Document` reaches
  `DETACH DELETE` regardless of whether it has structure of its own.

## [0.3.8] - 2026-09-08

### Changed
- **`llm` is now optional** in `configure()` / `ScinrConfig`. When no LLM is
  supplied (no `llm=` argument and no `MODEL_ID` env var), `configure()` succeeds
  with `cfg.llm = None` instead of raising `ConfigurationError: No LLM configured`,
  enabling **navigation-only** use of `scinr.newton.navigation` without any LLM
  setup — only Neo4j credentials (`NEO4J_USER`, `NEO4J_PASSWORD`,
  `NEO4J_DATABASE`) are required.
- The LLM requirement is now enforced **lazily**: `get_llm()` / `get_repair_llm()`
  raise a descriptive `ConfigurationError` only when an LLM-dependent stage
  (extraction, annotation, entity extraction, or tabular mapping/normalization)
  actually runs without a configured LLM, instead of failing at configure time.

### Added
- Navigation-only configuration examples in the **Configuration** reference and
  the **Graph Navigation** guide, showing `configure(neo4j_user=…, 
  neo4j_password=…, neo4j_database=…)` with no LLM.

### Fixed
- Inconsistency where `scinr.newton.navigation.__init__` advertised an LLM-less
  `configure()` call that `configure()` would reject at runtime.

## [0.3.7] - 2026-09-06

### Added
- **`scinr.newton.navigation` — read-only graph navigation API.** A new, fully
  `async`, engine-abstracted module for exploring the knowledge graph without
  writing Cypher by hand. `get_graph_navigator()` / the `graph_navigator()`
  async context manager return a `GraphNavigator` with ~90 typed methods:
  list root documents (no incoming `IS_COMPOSED_OF`), walk folder / structure
  trees to a given `depth`, pull the `StructureNode`s / `InfoUnit`s /
  `ModelInstance`s of a document or node, filter model instances by
  `model_class` and properties with classic operators (`Eq`, `Ne`, `Gt`, `Gte`,
  `Lt`, `Lte`, `In`, `NotIn`, `Contains`, `StartsWith`, `EndsWith`, `Regex`,
  `IsNull`, `IsNotNull`), jump from a model instance back to its owning
  `StructureNode`(s) / `Document`(s) / `ExtractionResult`(s), traverse annotation
  decisions (incl. a `get_document_model_profile` roll-up), entities and triples,
  introspect the catalogue (`get_catalog_graph`) and schema, and run generic
  `neighbors` / `shortest_path` / `subgraph` queries. Return types are
  engine-neutral Pydantic models. Nothing in the module mutates the graph.
- **Pluggable graph backend.** New `graph_backend` config field (env
  `GRAPH_BACKEND`, default `"neo4j"`), validated like `storage_backend`. The
  navigation layer is a `GraphNavigator` ABC plus a concrete
  `Neo4jGraphNavigator`; other engines can be added without touching call sites.
- **`GraphNavigator.execute_raw()` / `execute_raw_one()`** — an optional,
  non-portable escape hatch for engine-native read queries, with a `dialect=`
  fail-fast guard and a write-keyword rejection guard (`CREATE`, `MERGE`, `SET`,
  `DELETE`, `REMOVE`, `DROP`, `FOREACH`, `LOAD CSV`,
  `CALL { … } IN TRANSACTIONS`); the statement runs in a READ transaction. The
  base ABC raises `UnsupportedOperationError`.
- **`scinr.newton.navigation.pages`** — source-text bridge: resolves the verbatim
  converted markdown pages behind a structure node / info unit / document via the
  storage abstraction.
- New exceptions `NavigationError`, `GraphConnectionError`,
  `UnsupportedOperationError` (all under `ScinrError`), exported from
  `scinr.newton`.
- New `scinr.newton.utils.uid.normalize_key()` — the exact normalisation
  (`NFKD` → strip accents → lower-case → collapse whitespace) that ingestion
  applies to `instance_key` / entity values before hashing them into a UID;
  `entity_extraction.graph_mapper` now reuses it. `get_model_instance_by_key()`
  applies it so a raw key value resolves straight to the deterministic node UID.
- New docs: **Graph Navigation** user guide and **Navigation API** reference.

- **Provenance metadata on `:Document` nodes.** `run_pipeline()` accepts three new optional string parameters — `tenant_id`, `created_by_user_id`, and `job_id` — that are written verbatim onto every `:Document` node the run creates: leaf documents, ancestor folder-parent nodes, and tabular documents alike. Each property is always `SET` (stored as `null` when omitted), mirroring `context_instructions`. The values are also stamped onto the `Document` model, so they are serialized into any `extract-*.json` produced by the extraction stage; a value passed to a later ingestion-only `run_pipeline()` call overrides the baked-in one, while omitting it leaves the baked-in value untouched. Threaded through `run_extraction`-adjacent helpers (`extract_one_file`/`extract_one_intermediate`), the loader (`load_file`/`load_files`/`load_folder`/`load_documents`/`ingest_one`/`ingest_one_from_path`), `run_ingestion()`, and `run_tabular_pipeline()`/`run_tabular_agent()`. Not threaded through the standalone `preprocess` stage — supply the values on the `run_pipeline()` call that performs extraction and/or ingestion.
- New `:Document` indexes on `tenant_id`, `created_by_user_id`, and `job_id` (created by `setup_schema()`).
- `delete_document(job_id=...)` — bulk deletion selector. Exactly one of `path` or `job_id` must now be provided (`ValueError` otherwise). `job_id` deletes every `:Document` whose `job_id` matches, across all paths and versions of that ingestion run.
- `delete_document(..., tenant_id=..., created_by_user_id=...)` — optional extra AND-filters applied on top of either selector. A filter left as `None` means "do not filter on this property" (not "the property must be null"). `version` is still accepted as an extra filter in `job_id` mode. The selection `WHERE` clause is assembled at call time from only the filters actually supplied — a plain equality conjunction, so Neo4j uses the per-property `:Document` indexes (notably `idx_document_job_id`) instead of a full label scan.

### Changed
- `configure()` accepts a new `graph_backend=` parameter (default `"neo4j"`); the startup debug log line now reports it alongside `storage`.
- `DeletionResult` gained `job_id`, `tenant_id`, and `created_by_user_id` echo fields; `path` is now `str | None` (it is `None` for a `job_id`-selected deletion).

## [0.3.6] - 2026-09-04

### Fixed
- Removed redundant function-local `get_config` imports in `newton.tabular.neo4j_ops` that shadowed the module-level import and caused `UnboundLocalError` on every tabular sheet write (fixes CSV/XLSX ingestion writing zero rows).

## [0.3.5] - 2026-09-04

### Added
- **`fast_extraction` mode** (opt-in) in `run_pipeline()`: pass `fast_extraction=True` to run Stage 1 (extraction) chunks in parallel and defer cross-chunk hierarchy resolution to a single post-extraction consolidation LLM call instead of incremental per-chunk prefix matching. This can substantially reduce Stage 1 wall-clock time for multi-chunk documents. The flag is resolved once per call and passed explicitly down to Stage 1 — never read from global config — so concurrent `run_pipeline()` calls with different values never interfere. Default remains `False` (unchanged legacy behavior); raises `ValueError` if `True` while `"extraction"` is not in `stages`. (#14)
- Structure-consolidation machinery backing the fast mode: new `extraction/structure_consolidation.py` (`consolidate_structure()`), `models/consolidation.py`, and `prompts/consolidation_prompt.py`. In fast mode the structural tree is not built incrementally; it is recreated via LLM calls after all pages of a document are processed, using a sliding window (default batch ceiling 64k tokens). (#14)
- Consolidation configuration options on `ScinrConfig` / `configure()`, each with an env var:
  - `consolidation_token_safety_margin` (`CONSOLIDATION_TOKEN_SAFETY_MARGIN`, default `0.75`) — fraction of `max_tokens` used as the output-token ceiling for the consolidation LLM call when no explicit ceiling is set.
  - `consolidation_max_output_tokens` (`CONSOLIDATION_MAX_OUTPUT_TOKENS`, default `None` → derived as `max_tokens × safety margin`).
  - `consolidation_max_input_tokens` (`CONSOLIDATION_MAX_INPUT_TOKENS`, default `65536` / 64k) — governs the sliding-window batch size. (#14)
- `tiktoken>=0.13` dependency for token-count estimation in consolidation batching (o200k_base approximation). (#14)
- `neo4j_database` configuration option (`NEO4J_DATABASE`) to target a specific Neo4j database instead of the server default; all ingestion, document-resolution, annotation, and entity-extraction sessions now honor it. (#15)

### Changed
- Converter dispatch is now async-aware: sync (blocking) converters run in a worker thread via `asyncio.to_thread()`, while async converters are awaited directly on the event loop. This makes `parallel_docs > 1` produce real concurrent progress — previously a blocking in-coroutine call monopolised the event loop and silently negated the scheduled parallelism. (#12)
- New class attribute `BaseConverter.is_async: bool = False`; `PdfConverter` declares `is_async = True` (its `convert()` is a coroutine performing genuine network I/O against the Mistral OCR API). `convert_and_write()` now raises `ConversionError` for async converters. (#12)
- `mistral_ocr_max_retries` default raised from `3` to `15` (`MISTRAL_OCR_MAX_RETRIES`), retrying on HTTP 429/500/502/503/504 with exponential backoff (`mistral_ocr_retry_backoff_seconds`, default `2.0`s) to ride through API rate limits. (#12)
- Post-extraction normalization for tabular data is now **enabled by default** (`normalization_enabled` default flipped from `false` to `true`; env `NORMALIZATION_ENABLED`). (#15)

### Fixed
- `convert_single_file()` no-storage path now works for both sync and async converters, and it also injects `context_instructions` on that path (previously only the storage-aware branch did). (#12)
- The Neo4j minimum-version compatibility check in `setup_schema()` is now best-effort: it no longer breaks ingestion when the server reports a non-standard / unparseable version string (e.g. Neo4j Aura's `27-aura`) — such cases log a warning and skip the check. (#13, #15)
- Documentation and docstring corrections across the docs site and public API. (#15)

## [0.3.3] - 2026-08-12

### Added
- `RawFileRepository.delete(raw_file_id)` and `PageRepository.delete_pages(raw_file_id)` abstract methods on the storage interfaces (`storage/base.py`), implemented for `NullRawFileRepository`/`NullPageRepository` (no-op) and `MongoDBRawFileRepository`/`MongoDBPageRepository` (GridFS + `raw_files`/`converted_pages` collection cleanup). Both are idempotent — safe to call for an already-deleted or never-existing `raw_file_id`. (#9)
- Per-document pipeline orchestration: each document unit now runs all of its stages independently and concurrently, instead of processing an entire stage across every document before moving on. This prevents a single slow, large document from blocking the whole pipeline. New `neo4j_sync_concurrency` option (env `NEO4J_SYNC_CONCURRENCY`, default `8`) caps concurrent Stage 2 (sync-ingestion) dispatches to worker threads. (#8)
- Automatic PDF splitting for Mistral OCR: PDFs that exceed the safe size limits are now split into chunks before being sent to the API, with per-chunk retries and a configurable error strategy. New options — `mistral_ocr_safe_max_pages` (`MISTRAL_OCR_SAFE_MAX_PAGES`, default `900`), `mistral_ocr_safe_max_bytes` (`MISTRAL_OCR_SAFE_MAX_BYTES`, default `45 MiB`), `mistral_ocr_retry_backoff_seconds` (`MISTRAL_OCR_RETRY_BACKOFF_SECONDS`, default `2.0`s), `mistral_ocr_chunk_concurrency` (reserved for future chunk parallelism, default `1`), and `mistral_ocr_error_strategy` (`MISTRAL_OCR_ERROR_STRATEGY`, `'fail_fast'` | `'best_effort'`) — plus a new `pypdf>=5.0` dependency. (#8)
- `delete_document(path, version=None)` public API: completely removes a `:Document` node (a specific *version*, or all versions when `None`) together with its full cascade — structure nodes, info units, model decisions, proposed models/fields, and extraction results — then runs two garbage-collection passes to drop any orphaned entities it leaves behind. Returns detailed per-category counts in `DeletionResult`. (#8)
- `full_docstring` option (env `FULL_DOCSTRING`, default `True`): when `True` the LLM-facing model-catalog description (and the stored `CatalogModel.description`) uses the full class docstring; when `False` only its first non-empty line is used. (#11)

### Changed
- `delete_document()` (`scinr.newton.ingest.deletion`) is now `async` (previously sync) and now also deletes the corresponding documental storage records (raw binary + converted Markdown pages, keyed by each affected `:Document`'s `raw_file_id`) *before* running the Neo4j cascade delete. Storage cleanup is fail-fast: an unexpected exception there aborts the whole deletion before any Neo4j write happens. `DeletionResult` gained two new fields: `raw_files_deleted` and `converted_pages_deleted`. (#9)
- **Breaking change:** any `storage_backend="custom"` implementation must now also implement `delete()` on its `RawFileRepository` and `delete_pages()` on its `PageRepository`. (#9)
- Supplementary fields recommended by the LLM during annotation are now silently coerced from `dict` / `list[dict]` to `str` / `list[str]`, since richer structures are not fully supported downstream and would otherwise clutter the prompt. (#8)

### Fixed
- The pipeline can now progress from `annotation` to `entity_extraction` when some nodes fail: `on_partial_failure` is taken into account (default `warn` — it logs a warning and continues instead of aborting). (#8)
- Added a retry wrapper around the tabular normalization pipeline, with regression tests. (#8)
- Fixed value concatenation and deduplication when multiple tabular columns are mapped to the same model property. (#10)

## [0.1.0] - 2024-01-01

### Added
- `configure()` API for provider-agnostic LLM configuration (T-01)
- Exception hierarchy: `ScinrError`, `ConfigurationError`, `PreconditionError`, `ExtractionError`, `IngestionError`, `ModelError`, `StorageError`, `ConversionError` (T-02)
- `ThemeRegistry` with lazy loading, `enabled_base_themes` filtering, and external entry-point discovery (T-03)
- LLM decoupling: `make_llm()` abstraction replaces direct Bedrock coupling (T-04)
- `llm_retry` generalized for Bedrock, OpenAI, and Anthropic (T-05)
- Storage Null Object pattern: `NullRawFileRepository`, `NullPageRepository` (T-06)
- Stage preconditions with actionable error messages (T-08)
- CSV auto-detect separator, UTF-8-BOM support, duplicate header deduplication (T-14)
- MongoDB connection health check at startup (T-13)
- Custom storage backend registration via `configure(custom_storage=...)` (T-13)
- Custom converter registration via `configure(extra_converters=...)` (T-12)
- `ModelField` MERGE key fixed to composite `{name, model}` (T-18)
- `src/` package layout with `scinr.newton` namespace (T-11)

### Fixed
- Silent errors in storage initialization (T-09)
- PDF converter now shows actionable error when `MISTRAL_API_KEY` is missing (T-10)
- `.env.example` corrected with all required variables (T-07)
