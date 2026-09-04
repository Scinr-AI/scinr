# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
