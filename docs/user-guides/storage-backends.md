# Storage Backends

`scinr.newton` provides an optional persistent storage layer that runs alongside the Neo4j graph pipeline. Storage backends archive raw source files and their converted pages, giving you a durable record of every document that passes through the pipeline.

Neo4j remains the primary output store. Storage is supplementary — it exists for raw file archival, audit trails, and compliance requirements. You can run the full pipeline with Neo4j alone and never touch storage.

Three backends are available:

- **`none`** (default) — no persistent storage; all data stays in-memory during pipeline execution.
- **`mongodb`** — MongoDB with GridFS for raw files and a document collection for converted pages.
- **`custom`** — user-defined repositories implementing the `RawFileRepository` and `PageRepository` interfaces.

---

## Backend Comparison

| Feature | `none` | `mongodb` | `custom` |
|---|---|---|---|
| Raw file storage | No | Yes (GridFS) | User-defined |
| Page content | No | Yes (document collection) | User-defined |
| Document metadata | No | Yes (raw_files collection) | User-defined |
| Dependencies | None | `motor`, `pymongo` | User-defined |
| Use case | Dev/testing, Neo4j-only workflows | Production with audit trail | Custom infrastructure needs |

---

## Architecture

The storage layer is composed of two abstract repository interfaces:

```
RawFileRepository             PageRepository
┌────────────────────┐       ┌──────────────────┐
│ .store()           │       │ .store_page()    │
│ .store_file()      │       │   → page_id      │
│   → raw_file_id    │       │ .get_pages()     │
│ .get()   → record  │       │   → list[pages]  │
│ .open()  → stream  │       │ .get_pages_by_ids│
│ .open_with_record()│       │   → list[pages]  │
│ .list_raw_files()  │       │ .delete_pages()  │
│ .delete()          │       │   → int          │
│ (binary files)     │       │ (markdown pages) │
└────────────────────┘       └──────────────────┘
```

- **`RawFileRepository`** — stores the original binary file and returns a `raw_file_id`; `get()` / `open()` read its metadata / binary (`open_with_record()` both, with one lookup), `list_raw_files()` is the inventory, `delete()` removes it.
- **`PageRepository`** — stores converted page content (Markdown) linked to a `raw_file_id`; `get_pages()` reads every page of a file, `get_pages_by_ids()` only the given pages (a structure node's `source_page_ids`), and `delete_pages()` removes a file's pages.

Every record carries its owner (`tenant_id`, `created_by_user_id`, `job_id`) and every read / delete takes the same scope filters as the graph navigation — see [Multi-tenancy](#multi-tenancy).

The pipeline calls `get_storage()` to obtain the configured pair of repositories. All downstream code interacts with the abstract interfaces, keeping the pipeline backend-agnostic.

---

## The "none" Backend (Default)

When `storage_backend="none"` (the default), scinr uses no-op repository implementations that silently discard all writes. This is the recommended setting for development, testing, or when Neo4j alone is sufficient.

### Configuration

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure

# Explicit — same as omitting storage_backend entirely
configure(
    llm=ChatOllama(model="llama3"),
    storage_backend="none",
)

# storage_backend also resolves from STORAGE_BACKEND in the environment.
# Pass llm= too if this run will execute any LLM stage.
```

### Behavior

- Raw files are **not** archived to any persistent store.
- Converted pages are **not** persisted.
- All data lives in-memory during pipeline execution.
- Intermediate JSON files are written to disk only if you set `converter_output_dir` or `extraction_output_dir` on `run_pipeline()`.
- No additional dependencies are required.

### When to Use

- **Development and testing** — fastest setup, no infrastructure needed.
- **Neo4j-only workflows** — when the graph is the sole source of truth.
- **CI/CD pipelines** — avoids requiring a MongoDB instance in test environments.
- **Quick prototyping** — focus on extraction models without storage concerns.

### Complete Example

```python
import asyncio
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
        storage_backend="none",  # explicit, but this is the default
    )

    result = await run_pipeline(input_raw="./raw_docs")

    print(f"Pipeline: {'success' if result.success else 'failed'}")
    if result.preprocess:
        print(f"  Converted: {result.preprocess.total_processed} files")

asyncio.run(main())
```

---

## MongoDB Backend

The MongoDB backend stores raw files in GridFS (for arbitrary file sizes) and converted pages in a standard document collection. It provides full durability, queryability, and audit capability.

### Installation

```bash
pip install "scinr[mongodb]"
```

This installs `motor` (async MongoDB driver) and `pymongo` (sync driver, used for connection validation).

### Configuration

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure

configure(
    llm=ChatOllama(model="llama3"),
    storage_backend="mongodb",
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="scinr",
    mongodb_raw_files_collection="raw_files",
    mongodb_pages_collection="converted_pages",
    mongodb_gridfs_bucket="raw_binaries",
)
```

Or via environment variables:

```bash
# .env
STORAGE_BACKEND=mongodb
MONGODB_URI=mongodb://user:pass@mongo.internal:27017
MONGODB_DATABASE=scinr_production
MONGODB_RAW_FILES_COLLECTION=raw_files
MONGODB_PAGES_COLLECTION=converted_pages
MONGODB_GRIDFS_BUCKET=raw_binaries
```

### Collections

MongoDB creates three storage areas automatically on first use:

#### `raw_files` — Raw File Metadata

Lightweight metadata documents for each ingested file. The binary content itself lives in GridFS.

```json
{
  "_id": "ObjectId('67a3b2c1d4e5f6a7b8c9d0e1')",
  "filename": "clinical_trial_report.pdf",
  "folder_path": "ModuleA/Section3",
  "content_type": "application/pdf",
  "size_bytes": 2458624,
  "checksum_sha256": "a1b2c3d4e5f6789012345678abcdef01234567890abcdef012345678901234567",
  "stored_at": "2025-01-15T10:30:00Z",
  "gridfs_id": "ObjectId('67a3b2c1d4e5f6a7b8c9d0e2')",
  "tenant_id": "acme",
  "created_by_user_id": "user-42",
  "job_id": "job-2025-01-15-001"
}
```

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Unique identifier. Used as `raw_file_id` by pages. |
| `filename` | String | Original filename including extension. |
| `folder_path` | String or null | Relative path from the ingestion root, or `null` for root-level files. |
| `content_type` | String | MIME type of the original file (e.g., `application/pdf`). |
| `size_bytes` | Integer | Size of the binary content in bytes. |
| `checksum_sha256` | String | SHA-256 hex digest of the original binary content. Used for deduplication and integrity verification. |
| `stored_at` | DateTime | UTC timestamp when the record was persisted. |
| `gridfs_id` | ObjectId | Reference to the file stored in GridFS. |
| `tenant_id` | String | Owning tenant, **stored** form: the tenant, or `"__public__"` for a public upload — never `null`. Absent only on records written before multi-tenancy. |
| `created_by_user_id` | String or null | Provenance: user who uploaded the file. |
| `job_id` | String or null | Provenance: job the upload belongs to. |

#### `converted_pages` — Converted Page Content

One document per converted page, linked to its parent raw file.

```json
{
  "_id": "ObjectId('67a3b2c1d4e5f6a7b8c9d0e3')",
  "raw_file_id": "67a3b2c1d4e5f6a7b8c9d0e1",
  "filename": "clinical_trial_report",
  "folder_path": "ModuleA/Section3",
  "page_index": 0,
  "markdown": "# 3. Clinical Trial Results\n\nThe primary endpoint was...",
  "converted_at": "2025-01-15T10:30:05Z",
  "tenant_id": "acme",
  "created_by_user_id": "user-42",
  "job_id": "job-2025-01-15-001"
}
```

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Unique identifier for the page record. |
| `raw_file_id` | String | Reference to the parent `raw_files._id`. |
| `filename` | String | Stem of the source file without extension. |
| `folder_path` | String or null | Relative path from the ingestion root, or `null`. |
| `page_index` | Integer | Zero-based page index. Matches the converter's page ordering. |
| `markdown` | String | Full Markdown text of the page as produced by the converter. |
| `converted_at` | DateTime | UTC timestamp when the page was persisted. |
| `tenant_id`, `created_by_user_id`, `job_id` | String | Same values as the parent `raw_files` record. |

#### `raw_binaries` — GridFS Bucket

GridFS automatically creates two internal collections:

- `raw_binaries.files` — file metadata (filename, length, chunk size, upload date, GridFS metadata).
- `raw_binaries.chunks` — binary data chunks (255 kB each by default).

GridFS handles files of arbitrary size, removing the 16 MB BSON document limit. The `gridfs_id` in `raw_files` points to the corresponding GridFS file document. The GridFS `metadata` holds `content_type`, `folder_path`, `tenant_id`, `created_by_user_id` and `job_id`, so the owner is also visible from the bucket alone.

### Indexes

The library creates these indexes by itself: the first `get_storage()` call of the process (for each MongoDB URI / database / collections) pings the server and then creates them with `createIndex`, which is a no-op when an index already exists. There is nothing to call at application startup. Every index is prefixed by `tenant_id`, so a tenant-scoped read is an index seek:

```python
# converted_pages: primary lookup by (tenant, raw_file_id) + page ordering
db.converted_pages.create_index(
    [("tenant_id", 1), ("raw_file_id", 1), ("page_index", 1)],
    name="pages_by_tenant_raw_file_and_index",
)

# converted_pages: secondary lookup by filename + folder within a tenant
db.converted_pages.create_index(
    [("tenant_id", 1), ("filename", 1), ("folder_path", 1)],
    name="pages_by_tenant_filename_folder",
)

# raw_files: integrity / duplicate checks by SHA-256 within a tenant
db.raw_files.create_index(
    [("tenant_id", 1), ("checksum_sha256", 1)],
    name="raw_files_by_tenant_checksum",
)

# raw_files: inventory / audit by provenance (list_raw_files)
db.raw_files.create_index([("tenant_id", 1), ("created_by_user_id", 1)], name="raw_files_by_tenant_user")
db.raw_files.create_index([("tenant_id", 1), ("job_id", 1)], name="raw_files_by_tenant_job")
```

The pre-multitenancy indexes `pages_by_raw_file_and_index`, `pages_by_filename_folder` and `raw_files_by_checksum` are dropped if present.

Reads by `_id` (`get`, `open`, `open_with_record`, `get_pages_by_ids`, the `raw_file_id` ownership check) use the default `_id` index. GridFS creates its own indexes on `raw_binaries.chunks` on the first upload.

#### When the database user cannot create indexes

Index creation needs the `createIndex` privilege (and `dropIndex` for the obsolete ones). If it fails, scinr logs a **warning** and keeps working — storage reads and writes are not blocked, but reads may scan whole collections until the indexes exist. Two options:

- Have operations create the indexes above out of band (same keys and names), and set `mongodb_ensure_indexes=False` (env `MONGODB_ENSURE_INDEXES=false`) so scinr does not try.
- Grant the privilege to the application user.

`scinr.newton.storage.mongodb.client.ensure_indexes()` (async, Motor) and `ensure_indexes_sync(cfg)` (pymongo) create the same indexes on demand, e.g. from a migration script. To verify that the queries really use them, see [Verifying index usage](neo4j-graph.md#verifying-index-usage).

### MongoDB Queries

#### List All Stored Documents

```javascript
db.raw_files.find().pretty();                        // every tenant
db.raw_files.find({ tenant_id: "acme" }).pretty();   // one tenant
db.raw_files.find({ tenant_id: { $exists: false } }) // legacy records (no tenant)
```

#### Get All Pages for a Document

```javascript
// Find the raw_file_id first
db.raw_files.findOne({ tenant_id: "acme", filename: "clinical_trial_report.pdf" });

// Then get all pages, ordered by page index
db.converted_pages
  .find({ tenant_id: "acme", raw_file_id: "67a3b2c1d4e5f6a7b8c9d0e1" })
  .sort({ page_index: 1 });

// Or only the pages of one structure node (its source_page_ids)
db.converted_pages
  .find({ tenant_id: "acme", _id: { $in: [ObjectId("..."), ObjectId("...")] } })
  .sort({ page_index: 1 });
```

#### Get Pages by Filename

```javascript
db.converted_pages
  .find({ filename: "clinical_trial_report" })
  .sort({ page_index: 1 });
```

#### File Size Statistics by Format

```javascript
db.raw_files.aggregate([
  {
    $group: {
      _id: "$content_type",
      count: { $sum: 1 },
      total_size: { $sum: "$size_bytes" },
      avg_size: { $avg: "$size_bytes" }
    }
  },
  { $sort: { total_size: -1 } }
]);
```

#### Find Duplicate Files by Checksum

```javascript
db.raw_files.aggregate([
  { $group: { _id: { tenant: "$tenant_id", checksum: "$checksum_sha256" }, count: { $sum: 1 }, filenames: { $push: "$filename" } } },
  { $match: { count: { $gt: 1 } } }
]);
```

#### Storage Usage Over Time

```javascript
db.raw_files.aggregate([
  {
    $group: {
      _id: {
        year: { $year: "$stored_at" },
        month: { $month: "$stored_at" }
      },
      count: { $sum: 1 },
      total_bytes: { $sum: "$size_bytes" }
    }
  },
  { $sort: { "_id.year": 1, "_id.month": 1 } }
]);
```

#### Retrieve Raw File from GridFS

From Python, prefer the repository (it applies the tenant scope) or `nav.get_document_original()` — see [Multi-tenancy](#multi-tenancy). The raw `mongosh` equivalent:

```javascript
// Using the gridfs_id from a raw_files document
var gridfsId = ObjectId("67a3b2c1d4e5f6a7b8c9d0e2");
var bucket = new GridFSBucket(db, { bucketName: "raw_binaries" });
var stream = bucket.openDownloadStream(gridfsId);
stream.on("data", function(chunk) { /* process chunk */ });
```

### Connection Validation

When `storage_backend="mongodb"`, the factory validates the MongoDB connection with a synchronous ping (5-second timeout) the first time `get_storage()` is called in the process, and again after every `configure()`. Later calls skip the ping — `get_storage()` runs on every source-text read. If the server is unreachable, a `StorageError` is raised immediately:

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline
from scinr.newton.exceptions import StorageError

try:
    configure(
        llm=ChatOllama(model="llama3"),
        storage_backend="mongodb",
        mongodb_uri="mongodb://wrong-host:27017",
    )
    await run_pipeline(input_raw="./raw_docs")
except StorageError as e:
    print(f"Storage unavailable: {e}")
```

### Complete Example

```python
import asyncio
from langchain_ollama import ChatOllama
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=ChatOllama(model="llama3"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        neo4j_database="neo4j",
        storage_backend="mongodb",
        mongodb_uri="mongodb://user:pass@mongo.internal:27017",
        mongodb_database="scinr_production",
    )

    result = await run_pipeline(
        input_raw="./raw_docs",
        converter_output_dir="./data/converted/",
    )

    print(f"Pipeline: {'success' if result.success else 'failed'}")
    print(f"Raw files and pages stored in MongoDB.")

asyncio.run(main())
```

---

## Multi-tenancy

The storage layer follows the same tenant model as the graph: the tenant is part of every record, "public" is stored as `"__public__"` (never `null`), and reads take the same filters as the navigation API.

### What is stored

Every `raw_files` record, its GridFS `metadata` and every `converted_pages` record carry the **stored** `tenant_id` plus `created_by_user_id` and `job_id` of the upload (scalars — a raw file belongs to exactly one upload). On write, `tenant_id=None` and `tenant_id="__public__"` both mean public; `""` raises `ValueError`.

The owner comes from the conversion call: `run_pipeline(tenant_id=..., created_by_user_id=..., job_id=...)`, `run_preprocess(...)`, `convert_one` / `convert_folder` / `convert_single_file(...)` and `run_tabular_pipeline(...)`. It is also stamped on the intermediate JSON (`IntermediateDocument.tenant_id`, …), so a later extraction / ingestion of that JSON keeps the tenant unless the run overrides it.

### Reading and deleting: the scope

Every read and delete method takes the four keyword-only filters of the navigation API (`scinr.newton.utils.scope`), with identical semantics and errors:

| `tenant_id` | `include_public` | Records returned |
|---|---|---|
| `None` (default) | any | **all tenants**, legacy records without tenant included |
| `"__public__"` | any | public only |
| `"acme"` | `False` | `acme` only |
| `"acme"` | `True` | `acme` + public |
| `""` | — | `ScopeError` |

`created_by_user_id` / `job_id` take one value or a list (`IN`); `None` = no filter, an empty list raises `ScopeError` (a subclass of both `StorageError` and `NavigationError`). A record outside the scope behaves exactly like a missing one: `get()` / `open()` / `open_with_record()` return `None`, `get_pages()` / `get_pages_by_ids()` return `[]`, `delete()` / `delete_pages()` delete nothing.

The library does **not** impose a restriction — the default is "all tenants", which suits administrative tools. A multi-tenant API layer should **always** pass its tenant:

```python
from scinr.newton.storage.factory import get_storage

raw_repo, page_repo = get_storage()

record = await raw_repo.get(raw_file_id, tenant_id="acme", include_public=True)
stream = await raw_repo.open(raw_file_id, tenant_id="acme")          # None if not acme's
if stream is not None:
    async for chunk in stream:                                       # constant memory
        response.write(chunk)

inventory = await raw_repo.list_raw_files(tenant_id="acme", job_id=["job-1", "job-2"])
pages = await page_repo.get_pages(raw_file_id, tenant_id="acme")
```

From the graph side, the navigator methods resolve the node or document through the navigator (use `nav.scoped(tenant_id=...)`) and then read storage filtered by that node's / document's own tenant — see [Graph Navigation](graph-navigation.md):

- `get_structure_nodes_source_pages()` / `get_info_unit_source_text()` read only the nodes' `source_page_ids` with `get_pages_by_ids()` (one call per stored tenant of the nodes) — the documents and their other pages are never loaded.
- `get_document_source_text()` reads every page of the document with `get_pages()`.
- `get_document_original(document)` reads the record and the binary with `open_with_record()`.

### Ingestion checks the owner of `raw_file_id`

A `:Document` links to its upload through `raw_file_id`, which comes from an editable JSON. Before a document with a non-empty `raw_file_id` is written to Neo4j (`run_pipeline`, `run_ingestion`, `ingest_one`, `ingest_one_from_path`), the stored raw file must exist **and** belong to the document's effective tenant; otherwise the document fails with `IngestionError` and nothing is written. Only the tenant must match — the upload's `created_by_user_id` / `job_id` may differ from the ingestion run's. A tenant never "claims" a public file. With `storage_backend="none"` a non-empty `raw_file_id` cannot be verified and is rejected: configure the storage backend the file was converted with, or clear `raw_file_id` in the JSON.

`delete_document()` deletes each `raw_file_id` scoped to the tenant of the graph node that carries it, so a forged id never deletes another tenant's original.

### Records written before multi-tenancy

There is no backfill. Records without `tenant_id` are only visible with `tenant_id=None` (no tenant filter), like legacy nodes in the navigation; any tenant filter excludes them. A legacy `:Document` (no `tenant_id`) reads its pages unfiltered.

---

## Custom Backend

The `custom` backend lets you provide your own storage implementation. You implement two abstract base classes — `RawFileRepository` and `PageRepository` — and pass them as a tuple to `configure()`.

### Repository Interfaces

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

# The owner on writes, and the scope on reads / deletes (abbreviated below):
#   OWNER = *, tenant_id: str | None = None, created_by_user_id: str | None = None,
#              job_id: str | None = None
#   SCOPE = *, tenant_id: str | None = None, include_public: bool = False,
#              created_by_user_id: str | Sequence[str] | None = None,
#              job_id: str | Sequence[str] | None = None

class RawFileRepository(ABC):
    @abstractmethod
    async def store(self, filename, content: bytes, content_type, folder_path, OWNER) -> str:
        """Store a raw binary file with its owner and return its ID."""

    async def store_file(self, path: Path, filename, content_type, folder_path, OWNER) -> str:
        """Optional. Store the file at `path`. Default: read it fully and call `store()`."""

    @abstractmethod
    async def get(self, raw_file_id, SCOPE) -> RawFileRecord | None:
        """Metadata, or None if missing / outside the scope."""

    @abstractmethod
    async def open(self, raw_file_id, SCOPE) -> AsyncIterator[bytes] | None:
        """Stream of the binary, or None if missing / outside the scope."""

    @abstractmethod
    async def open_with_record(self, raw_file_id, SCOPE) -> tuple[RawFileRecord, AsyncIterator[bytes]] | None:
        """(metadata, stream) from a single lookup, or None if missing / outside the scope."""

    @abstractmethod
    async def list_raw_files(self, SCOPE, folder_path=None, filename=None) -> list[RawFileRecord]:
        """Inventory inside the scope, ordered by stored_at."""

    @abstractmethod
    async def delete(self, raw_file_id, SCOPE) -> None:
        """Delete binary + metadata if inside the scope. Idempotent (no error if missing)."""

class PageRepository(ABC):
    @abstractmethod
    async def store_page(self, raw_file_id, filename, folder_path, page_index, markdown, OWNER) -> str:
        """Store a converted page with its owner and return its ID."""

    @abstractmethod
    async def get_pages(self, raw_file_id, SCOPE) -> list[ConvertedPageRecord]:
        """Pages of a raw file inside the scope, ordered by page_index."""

    @abstractmethod
    async def get_pages_by_ids(self, page_ids, SCOPE) -> list[ConvertedPageRecord]:
        """The given pages inside the scope, ordered by page_index; unknown ids are skipped."""

    @abstractmethod
    async def delete_pages(self, raw_file_id, SCOPE) -> int:
        """Delete the pages inside the scope; return how many were removed (0 if none)."""
```

> **Breaking (multi-tenancy):** `get`, `open`, `open_with_record`, `list_raw_files` and `get_pages_by_ids` are new `@abstractmethod`s, and every method takes the owner (writes) or the scope (reads / deletes) keyword-only arguments. A custom backend written before must add them, or it cannot be instantiated / will fail with `TypeError` on the new keyword arguments.

> Validate and resolve the scope with `scinr.newton.utils.scope.make_scope(tenant_id, include_public, created_by_user_id, job_id)` — it raises the same `ScopeError` as the built-in backends — and store the owner's tenant as `scinr.newton.utils.tenancy.tenant_key(tenant_id)`. For MongoDB-like filters, `scinr.newton.storage.filters.mongo_scope_filter(...)` renders the scope directly. A record outside the scope must behave exactly like a missing one (never reveal it exists). `open()` must locate the binary from the already-filtered record, never from an id supplied by the caller.

> `store_file` is **optional** (not abstract). The pipeline calls `store_file(path=...)`, whose default implementation reads the whole file and delegates to `store()`, so a backend that only implements `store()` keeps working. Override `store_file` if your backend can upload from a file handle (S3 multipart, Azure Blob, etc.): the built-in `mongodb` backend does this to stream files to GridFS without loading them into memory, which matters for large PDFs. With `storage_backend="none"` the file is never opened.

### Implementing a Custom Backend

Here is a complete example using S3 for raw files and DynamoDB for pages:

```python
import hashlib
from datetime import UTC, datetime

from scinr.newton.storage.base import PageRepository, RawFileRepository
from scinr.newton.storage.models import ConvertedPageRecord
from scinr.newton.utils.tenancy import tenant_key


class S3RawFileRepository(RawFileRepository):
    """Stores raw files in Amazon S3."""

    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.region = region
        # Initialize boto3 client
        from boto3 import client
        self.s3 = client("s3", region_name=region)

    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        tenant = tenant_key(tenant_id)  # "__public__" for public uploads
        # Build the S3 key under the tenant's prefix
        rel = f"{folder_path}/{filename}" if folder_path else filename
        key = f"{tenant}/{rel}"

        # Compute checksum for metadata
        checksum = hashlib.sha256(content).hexdigest()

        # Upload to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
            Metadata={
                "checksum_sha256": checksum,
                "stored_at": datetime.now(UTC).isoformat(),
                "tenant_id": tenant,
                "created_by_user_id": created_by_user_id or "",
                "job_id": job_id or "",
            },
        )

        # Return an identifier (S3 key as string)
        return key


class DynamoDBPageRepository(PageRepository):
    """Stores converted pages in Amazon DynamoDB."""

    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.table_name = table_name
        self.region = region
        from boto3 import client
        self.dynamodb = client("dynamodb", region_name=region)

    async def store_page(
        self,
        raw_file_id: str,
        filename: str,
        folder_path: str | None,
        page_index: int,
        markdown: str,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        import uuid
        page_id = str(uuid.uuid4())

        self.dynamodb.put_item(
            TableName=self.table_name,
            Item={
                "page_id": {"S": page_id},
                "raw_file_id": {"S": raw_file_id},
                "filename": {"S": filename},
                "folder_path": {"S": folder_path or ""},
                "page_index": {"N": str(page_index)},
                "markdown": {"S": markdown},
                "converted_at": {"S": datetime.now(UTC).isoformat()},
                "tenant_id": {"S": tenant_key(tenant_id)},
            },
        )

        return page_id

    async def get_pages(
        self, raw_file_id: str, *, tenant_id=None, include_public=False,
        created_by_user_id=None, job_id=None,
    ) -> list[ConvertedPageRecord]:
        from boto3.dynamodb.types import TypeDeserializer
        from scinr.newton.utils.scope import make_scope

        scope = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        deserializer = TypeDeserializer()

        response = self.dynamodb.query(
            TableName=self.table_name,
            KeyConditionExpression="raw_file_id = :rfid",
            ExpressionAttributeValues={":rfid": {"S": raw_file_id}},
            ScanIndexForward=True,
        )

        pages = []
        for item in response.get("Items", []):
            item_tenant = deserializer.deserialize(item["tenant_id"])
            if scope.tenants is not None and item_tenant not in scope.tenants:
                continue  # outside the scope: behave as if it did not exist
            # (user / job filters omitted for brevity — apply scope.user_ids / scope.job_ids)
            pages.append(ConvertedPageRecord(
                id=deserializer.deserialize(item["page_id"]),
                raw_file_id=deserializer.deserialize(item["raw_file_id"]),
                filename=deserializer.deserialize(item["filename"]),
                folder_path=deserializer.deserialize(item["folder_path"]) or None,
                page_index=int(deserializer.deserialize(item["page_index"])),
                markdown=deserializer.deserialize(item["markdown"]),
                converted_at=datetime.fromisoformat(
                    deserializer.deserialize(item["converted_at"])
                ),
                tenant_id=item_tenant,
            ))

        return pages
```

### Registering the Custom Backend

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure

# Instantiate your custom repositories
raw_repo = S3RawFileRepository(bucket="scinr-raw-files", region="us-east-1")
page_repo = DynamoDBPageRepository(table_name="scinr-pages", region="us-east-1")

# Register them as a tuple
configure(
    llm=ChatOllama(model="llama3"),
    storage_backend="custom",
    custom_storage=(raw_repo, page_repo),
)
```

### Key Points

- **`custom_storage` expects a tuple of instances**, not a class and kwargs. The tuple is `(RawFileRepository, PageRepository)`.
- Both repositories must be **async** — all methods use `async def`.
- Implement **every** abstract method: `store` + `get` + `open` + `open_with_record` + `list_raw_files` + `delete` (raw files) and `store_page` + `get_pages` + `get_pages_by_ids` + `delete_pages` (pages). The S3/DynamoDB sketch above omits several of them for brevity — a real implementation must add them, all honouring the scope filters.
- Store the owner on every record and filter every read / delete by the scope (see [Multi-tenancy](#multi-tenancy)).
- The `store()` and `store_page()` methods return a string identifier. The pipeline uses these IDs to link pages to their parent raw file.
- `get_pages()` and `get_pages_by_ids()` return `ConvertedPageRecord` Pydantic models ordered by `page_index` ascending. `get_pages_by_ids()` must fetch only the requested pages (it backs the source text of a single structure node) and skip ids that do not exist, lie outside the scope or are invalid for the backend.
- If `storage_backend="custom"` but `custom_storage` is not provided, the pipeline raises a `ConfigurationError` at `get_storage()` time.

### Minimal Custom Backend (In-Memory)

For testing or lightweight scenarios, an in-memory implementation is straightforward. It shows the whole contract, owner and scope included:

```python
import hashlib
import uuid
from datetime import UTC, datetime

from scinr.newton.storage.base import PageRepository, RawFileRepository
from scinr.newton.storage.models import ConvertedPageRecord, RawFileRecord
from scinr.newton.utils.scope import make_scope
from scinr.newton.utils.tenancy import tenant_key


def _in_scope(rec, scope) -> bool:
    return (
        (scope.tenants is None or rec.tenant_id in scope.tenants)
        and (scope.user_ids is None or rec.created_by_user_id in scope.user_ids)
        and (scope.job_ids is None or rec.job_id in scope.job_ids)
    )


class InMemoryRawFileRepository(RawFileRepository):
    def __init__(self):
        self._files: dict[str, tuple[RawFileRecord, bytes]] = {}

    async def store(self, filename, content, content_type, folder_path, *,
                    tenant_id=None, created_by_user_id=None, job_id=None) -> str:
        record = RawFileRecord(
            id=str(uuid.uuid4()), filename=filename, folder_path=folder_path,
            content_type=content_type, size_bytes=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
            stored_at=datetime.now(UTC), tenant_id=tenant_key(tenant_id),
            created_by_user_id=created_by_user_id, job_id=job_id,
        )
        self._files[record.id] = (record, content)
        return record.id

    def _find(self, raw_file_id, **filters):
        scope = make_scope(**filters)
        entry = self._files.get(raw_file_id)
        return entry if entry is not None and _in_scope(entry[0], scope) else None

    async def get(self, raw_file_id, **filters):
        entry = self._find(raw_file_id, **filters)
        return entry[0] if entry else None

    async def open(self, raw_file_id, **filters):
        entry = self._find(raw_file_id, **filters)
        if entry is None:
            return None

        async def _chunks():
            yield entry[1]

        return _chunks()

    async def open_with_record(self, raw_file_id, **filters):
        entry = self._find(raw_file_id, **filters)
        if entry is None:
            return None
        return entry[0], await self.open(raw_file_id, **filters)

    async def list_raw_files(self, *, folder_path=None, filename=None, **filters):
        scope = make_scope(**filters)
        return sorted(
            (r for r, _ in self._files.values()
             if _in_scope(r, scope)
             and (folder_path is None or r.folder_path == folder_path)
             and (filename is None or r.filename == filename)),
            key=lambda r: r.stored_at,
        )

    async def delete(self, raw_file_id, **filters) -> None:
        if self._find(raw_file_id, **filters) is not None:
            del self._files[raw_file_id]


class InMemoryPageRepository(PageRepository):
    def __init__(self):
        self._pages: dict[str, list[ConvertedPageRecord]] = {}

    async def store_page(self, raw_file_id, filename, folder_path, page_index, markdown, *,
                         tenant_id=None, created_by_user_id=None, job_id=None) -> str:
        record = ConvertedPageRecord(
            id=str(uuid.uuid4()), raw_file_id=raw_file_id, filename=filename,
            folder_path=folder_path, page_index=page_index, markdown=markdown,
            converted_at=datetime.now(UTC), tenant_id=tenant_key(tenant_id),
            created_by_user_id=created_by_user_id, job_id=job_id,
        )
        self._pages.setdefault(raw_file_id, []).append(record)
        return record.id

    async def get_pages(self, raw_file_id, **filters) -> list[ConvertedPageRecord]:
        scope = make_scope(**filters)
        pages = [p for p in self._pages.get(raw_file_id, []) if _in_scope(p, scope)]
        return sorted(pages, key=lambda p: p.page_index)

    async def get_pages_by_ids(self, page_ids, **filters) -> list[ConvertedPageRecord]:
        scope = make_scope(**filters)
        wanted = set(page_ids)
        pages = [p for ps in self._pages.values() for p in ps
                 if p.id in wanted and _in_scope(p, scope)]
        return sorted(pages, key=lambda p: p.page_index)

    async def delete_pages(self, raw_file_id, **filters) -> int:
        scope = make_scope(**filters)
        pages = self._pages.get(raw_file_id, [])
        keep = [p for p in pages if not _in_scope(p, scope)]
        self._pages[raw_file_id] = keep
        return len(pages) - len(keep)
```

---

## When to Use Each Backend

| Scenario | Recommended Backend | Rationale |
|---|---|---|
| Development / Testing | `none` | Zero infrastructure, fastest iteration. |
| Production with audit trail | `mongodb` | Full durability, queryable, GridFS for large files. |
| Production with existing cloud infrastructure | `custom` | Reuse S3, Azure Blob, or other storage you already manage. |
| Neo4j-only workflow | `none` | Storage is optional; Neo4j is the primary output. |
| Compliance (raw file retention) | `mongodb` or `custom` | Persistent archive of every ingested file. |
| CI/CD pipeline | `none` | Avoids external dependencies in test environments. |
| Multi-region deployment | `custom` | Route storage to region-appropriate infrastructure. |

---

## Storage and Pipeline Integration

### Where Storage Is Used

Storage is called during **Stage 0 (preprocess)** and the **tabular pipeline**:

1. **Raw file storage** — immediately after reading a file from disk, before conversion. The binary content is stored and a `raw_file_id` is returned.
2. **Page storage** — after each page is converted to Markdown, the page content is stored and linked to the `raw_file_id`.

```
Pipeline Flow (with storage enabled):

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  Read File   │ ──→ │  Store Raw File  │ ──→ │  Convert to  │
  │  (binary)    │     │  (raw_file_id)   │     │  Markdown    │
  └──────────────┘     └──────────────────┘     └──────┬───────┘
                                                       │
  ┌──────────────┐     ┌──────────────────┐     ┌──────▼───────┐
  │  Write JSON  │ ←── │  Store Page      │ ←── │  Page N      │
  │  (intermed.) │     │  (page_id)       │     │  (markdown)  │
  └──────────────┘     └──────────────────┘     └──────────────┘
```

### Independence from Neo4j

Storage operates independently of Neo4j:

- You can configure storage independently of the Neo4j pipeline stages — storage is called during Stage 0 (preprocess) and the tabular pipeline, while Neo4j is used in Stages 2-4. Both are optional components that can be tuned independently.
- You can have Neo4j without storage (the default `none` backend).
- Storage does not affect Stages 1-4 (extraction, ingestion, annotation, entity extraction).
- If storage fails, the pipeline continues — storage errors are caught and reported without aborting the pipeline.

### Storage in the Tabular Pipeline

The tabular pipeline (Stage 5) also uses storage when available:

- Raw tabular files (CSV, XLSX) are stored via `RawFileRepository`.
- Converted tabular pages are stored via `PageRepository`.
- If no storage backend is configured, the tabular pipeline uses null repositories automatically.

---

## Configuration Resolution

Storage settings follow the standard triple-resolution pattern:

1. **Explicit argument** to `configure()` (highest priority)
2. **Environment variable** (medium priority)
3. **Hard-coded default** (lowest priority)

```python
# Example: env var sets backend to "mongodb", configure() overrides to "none"
# $ export STORAGE_BACKEND=mongodb
configure(llm=ChatOllama(model="llama3"), storage_backend="none")  # final value: "none"
```

### All Storage Settings

| Setting | `configure()` param | Environment Variable | Default |
|---|---|---|---|
| Backend type | `storage_backend` | `STORAGE_BACKEND` | `"none"` |
| MongoDB URI | `mongodb_uri` | `MONGODB_URI` | `"mongodb://localhost:27017"` |
| MongoDB database | `mongodb_database` | `MONGODB_DATABASE` | `"scinr"` |
| Raw files collection | `mongodb_raw_files_collection` | `MONGODB_RAW_FILES_COLLECTION` | `"raw_files"` |
| Pages collection | `mongodb_pages_collection` | `MONGODB_PAGES_COLLECTION` | `"converted_pages"` |
| GridFS bucket | `mongodb_gridfs_bucket` | `MONGODB_GRIDFS_BUCKET` | `"raw_binaries"` |
| Create indexes automatically | `mongodb_ensure_indexes` | `MONGODB_ENSURE_INDEXES` | `True` |
| Custom storage | `custom_storage` | *(none)* | `None` |
| Snapshot backend ([Document Freezing](document-freezing.md)) | `freeze_backend` | `FREEZE_BACKEND` | resolved `storage_backend` |
| Snapshot metadata collection | `mongodb_frozen_collection` | `MONGODB_FROZEN_COLLECTION` | `"frozen_documents"` |
| Snapshot GridFS bucket | `mongodb_frozen_gridfs_bucket` | `MONGODB_FROZEN_GRIDFS_BUCKET` | `"frozen_snapshots"` |
| Custom snapshot repository | `custom_freeze_storage` | *(none)* | `None` |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `StorageError: Cannot connect to MongoDB` | Wrong URI or MongoDB not running | Verify `mongodb_uri`; check MongoDB is accessible. Use `storage_backend="none"` to bypass. |
| Warning `Could not create MongoDB indexes` | The MongoDB user lacks the `createIndex` privilege | Grant it, or create the indexes out of band and set `mongodb_ensure_indexes=False`. See [Indexes](#indexes). |
| `ConfigurationError: storage_backend='custom' requires passing custom_storage` | Missing `custom_storage` tuple | Pass `custom_storage=(raw_repo, page_repo)` to `configure()`. |
| `ConfigurationError: Unknown storage_backend` | Invalid backend name | Use one of: `"none"`, `"mongodb"`, `"custom"`. |
| Pages not found after ingestion | Storage backend was `none` during pipeline run | Re-run with `storage_backend="mongodb"` or `custom`. |
| GridFS errors on large files | Very old MongoDB, or a driver/server mismatch | Use a currently supported MongoDB (4.4+ / 5.0+) or a managed MongoDB service. GridFS itself is available in every modern MongoDB release. |
| `ImportError: No module named 'motor'` | MongoDB extras not installed | Run `pip install "scinr[mongodb]"`. |
| Custom backend methods not called | Passed class instead of instance | `custom_storage` expects instantiated objects: `(MyRawRepo(), MyPageRepo())`. |
| Duplicate files ingested | No deduplication check | The `(tenant_id, checksum_sha256)` index on `raw_files` enables per-tenant dedup queries. Implement pre-ingest checks using this field. |
| `IngestionError: raw_file_id=... does not exist or does not belong to the document's tenant` | The JSON's `raw_file_id` was converted under another tenant (or public), or deleted | Ingest with the tenant used at conversion, re-run the preprocess for this tenant, or clear `raw_file_id`. |
| `IngestionError: ... storage_backend='none' cannot verify it` | A JSON converted with storage is ingested without storage | Configure the same storage backend, or clear `raw_file_id` in the JSON. |
| Pages / original not found for a document | The read is scoped to the document's tenant; legacy records have no tenant | Check `raw_files.tenant_id` matches the `:Document`'s `tenant_id`. |

### Debugging Storage

Enable debug logging to see storage operations:

```python
import logging
from langchain_ollama import ChatOllama
from scinr.newton import configure

logging.basicConfig(level=logging.DEBUG)

configure(
    llm=ChatOllama(model="llama3"),
    storage_backend="mongodb",
    mongodb_uri="mongodb://localhost:27017",
    log_level="DEBUG",
)
```

Debug output includes:

```
DEBUG:scinr.newton.storage.mongodb.raw_files:Stored raw file 'report.pdf' → raw_file_id=67a3..., gridfs_id=67a4... (2458624 bytes, tenant=acme)
DEBUG:scinr.newton.storage.mongodb.pages:Stored page 0 of 'report' → page_id=67a5...
DEBUG:scinr.newton.storage.mongodb.pages:Stored page 1 of 'report' → page_id=67a6...
DEBUG:scinr.newton.storage.mongodb.client:MongoDB indexes ensured.
```

---

## See Also

- **[Configuration](../configuration.md)** — Complete reference for `configure()`, environment variables, and all settings.
- **[Running the Pipeline](running-pipeline.md)** — Pipeline entry points, stage selection, and workflow patterns.
- **[Neo4j Graph Storage](neo4j-graph.md)** — Understanding the graph model and querying results.
- **[Architecture](../architecture.md)** — Detailed walkthrough of each pipeline stage and data flow.
- **[Pipeline API](../api/pipeline.md)** — Auto-generated docstring for `run_pipeline()`.
