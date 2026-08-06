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
RawFileRepository          PageRepository
┌─────────────────┐       ┌──────────────────┐
│ .store()        │       │ .store_page()    │
│   → raw_file_id │       │   → page_id      │
│                 │       │                  │
│ (binary files)  │       │ .get_pages()     │
│                 │       │   → list[pages]  │
│                 │       │                  │
│                 │       │ (markdown pages)  │
└─────────────────┘       └──────────────────┘
```

- **`RawFileRepository`** — stores the original binary file and returns a `raw_file_id`.
- **`PageRepository`** — stores converted page content (Markdown) linked to a `raw_file_id`, and supports retrieval.

The pipeline calls `get_storage()` to obtain the configured pair of repositories. All downstream code interacts with the abstract interfaces, keeping the pipeline backend-agnostic.

---

## The "none" Backend (Default)

When `storage_backend="none"` (the default), scinr uses no-op repository implementations that silently discard all writes. This is the recommended setting for development, testing, or when Neo4j alone is sufficient.

### Configuration

```python
from scinr.newton import configure

# Explicit — same as omitting the parameter entirely
configure(
    storage_backend="none",
)

# Or via environment variable
# $ export STORAGE_BACKEND=none
configure()  # picks up STORAGE_BACKEND=none from environment
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
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
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
from scinr.newton import configure

configure(
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
  "gridfs_id": "ObjectId('67a3b2c1d4e5f6a7b8c9d0e2')"
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
  "converted_at": "2025-01-15T10:30:05Z"
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

#### `raw_binaries` — GridFS Bucket

GridFS automatically creates two internal collections:

- `raw_binaries.files` — file metadata (filename, length, chunk size, upload date, GridFS metadata).
- `raw_binaries.chunks` — binary data chunks (255 kB each by default).

GridFS handles files of arbitrary size, removing the 16 MB BSON document limit. The `gridfs_id` in `raw_files` points to the corresponding GridFS file document.

### Indexes

The MongoDB backend creates the following indexes on first use (via `ensure_indexes()`):

```python
# Indexes created automatically by ensure_indexes():
#
# converted_pages: primary lookup by raw_file_id + page ordering
db.converted_pages.create_index(
    [("raw_file_id", 1), ("page_index", 1)],
    name="pages_by_raw_file_and_index",
)

# converted_pages: secondary lookup by filename + folder
db.converted_pages.create_index(
    [("filename", 1), ("folder_path", 1)],
    name="pages_by_filename_folder",
)

# raw_files: deduplication by SHA-256 checksum
db.raw_files.create_index(
    [("checksum_sha256", 1)],
    name="raw_files_by_checksum",
)
```

### MongoDB Queries

#### List All Stored Documents

```javascript
db.raw_files.find().pretty();
```

#### Get All Pages for a Document

```javascript
// Find the raw_file_id first
db.raw_files.findOne({ filename: "clinical_trial_report.pdf" });

// Then get all pages, ordered by page index
db.converted_pages
  .find({ raw_file_id: "67a3b2c1d4e5f6a7b8c9d0e1" })
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
  { $group: { _id: "$checksum_sha256", count: { $sum: 1 }, filenames: { $push: "$filename" } } },
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

```javascript
// Using the gridfs_id from a raw_files document
var gridfsId = ObjectId("67a3b2c1d4e5f6a7b8c9d0e2");
var bucket = new GridFSBucket(db, { bucketName: "raw_binaries" });
var stream = bucket.openDownloadStream(gridfsId);
stream.on("data", function(chunk) { /* process chunk */ });
```

### Connection Validation

When `storage_backend="mongodb"`, the factory validates the MongoDB connection at startup using a synchronous ping with a 5-second timeout. If the server is unreachable, a `StorageError` is raised immediately:

```python
from scinr.newton import configure, run_pipeline
from scinr.newton.exceptions import StorageError

try:
    configure(
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
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
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

## Custom Backend

The `custom` backend lets you provide your own storage implementation. You implement two abstract base classes — `RawFileRepository` and `PageRepository` — and pass them as a tuple to `configure()`.

### Repository Interfaces

```python
from abc import ABC, abstractmethod

class RawFileRepository(ABC):
    @abstractmethod
    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None,
    ) -> str:
        """Store a raw binary file and return its ID."""
        ...

class PageRepository(ABC):
    @abstractmethod
    async def store_page(
        self,
        raw_file_id: str,
        filename: str,
        folder_path: str | None,
        page_index: int,
        markdown: str,
    ) -> str:
        """Store a converted page and return its ID."""
        ...

    @abstractmethod
    async def get_pages(self, raw_file_id: str) -> list[ConvertedPageRecord]:
        """Retrieve all pages for a raw file, ordered by page_index."""
        ...
```

### Implementing a Custom Backend

Here is a complete example using S3 for raw files and DynamoDB for pages:

```python
import hashlib
from datetime import UTC, datetime

from scinr.newton.storage.base import PageRepository, RawFileRepository
from scinr.newton.storage.models import ConvertedPageRecord


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
    ) -> str:
        # Build S3 key from folder path and filename
        key = f"{folder_path}/{filename}" if folder_path else filename

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
            },
        )

        return page_id

    async def get_pages(self, raw_file_id: str) -> list[ConvertedPageRecord]:
        from boto3.dynamodb.types import TypeDeserializer
        deserializer = TypeDeserializer()

        response = self.dynamodb.query(
            TableName=self.table_name,
            KeyConditionExpression="raw_file_id = :rfid",
            ExpressionAttributeValues={":rfid": {"S": raw_file_id}},
            ScanIndexForward=True,
        )

        pages = []
        for item in response.get("Items", []):
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
            ))

        return pages
```

### Registering the Custom Backend

```python
from scinr.newton import configure

# Instantiate your custom repositories
raw_repo = S3RawFileRepository(bucket="scinr-raw-files", region="us-east-1")
page_repo = DynamoDBPageRepository(table_name="scinr-pages", region="us-east-1")

# Register them as a tuple
configure(
    storage_backend="custom",
    custom_storage=(raw_repo, page_repo),
)
```

### Key Points

- **`custom_storage` expects a tuple of instances**, not a class and kwargs. The tuple is `(RawFileRepository, PageRepository)`.
- Both repositories must be **async** — all methods use `async def`.
- The `store()` and `store_page()` methods return a string identifier. The pipeline uses these IDs to link pages to their parent raw file.
- `get_pages()` returns `ConvertedPageRecord` Pydantic models ordered by `page_index` ascending.
- If `storage_backend="custom"` but `custom_storage` is not provided, the pipeline raises a `ConfigurationError` at `get_storage()` time.

### Minimal Custom Backend (In-Memory)

For testing or lightweight scenarios, an in-memory implementation is straightforward:

```python
from scinr.newton.storage.base import PageRepository, RawFileRepository
from scinr.newton.storage.models import ConvertedPageRecord


class InMemoryRawFileRepository(RawFileRepository):
    def __init__(self):
        self._files: dict[str, bytes] = {}

    async def store(self, filename, content, content_type, folder_path) -> str:
        import uuid
        file_id = str(uuid.uuid4())
        self._files[file_id] = content
        return file_id


class InMemoryPageRepository(PageRepository):
    def __init__(self):
        self._pages: dict[str, list[ConvertedPageRecord]] = {}

    async def store_page(
        self, raw_file_id, filename, folder_path, page_index, markdown
    ) -> str:
        import uuid
        from datetime import UTC, datetime
        page_id = str(uuid.uuid4())
        record = ConvertedPageRecord(
            id=page_id,
            raw_file_id=raw_file_id,
            filename=filename,
            folder_path=folder_path,
            page_index=page_index,
            markdown=markdown,
            converted_at=datetime.now(UTC),
        )
        self._pages.setdefault(raw_file_id, []).append(record)
        return page_id

    async def get_pages(self, raw_file_id) -> list[ConvertedPageRecord]:
        return sorted(
            self._pages.get(raw_file_id, []),
            key=lambda p: p.page_index,
        )
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
configure(storage_backend="none")  # final value: "none"
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
| Custom storage | `custom_storage` | *(none)* | `None` |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `StorageError: Cannot connect to MongoDB` | Wrong URI or MongoDB not running | Verify `mongodb_uri`; check MongoDB is accessible. Use `storage_backend="none"` to bypass. |
| `ConfigurationError: storage_backend='custom' requires passing custom_storage` | Missing `custom_storage` tuple | Pass `custom_storage=(raw_repo, page_repo)` to `configure()`. |
| `ConfigurationError: Unknown storage_backend` | Invalid backend name | Use one of: `"none"`, `"mongodb"`, `"custom"`. |
| Pages not found after ingestion | Storage backend was `none` during pipeline run | Re-run with `storage_backend="mongodb"` or `custom`. |
| GridFS errors on large files | MongoDB version < 4.6 or missing GridFS support | Upgrade MongoDB to 4.6+ or use a managed MongoDB service. |
| `ImportError: No module named 'motor'` | MongoDB extras not installed | Run `pip install "scinr[mongodb]"`. |
| Custom backend methods not called | Passed class instead of instance | `custom_storage` expects instantiated objects: `(MyRawRepo(), MyPageRepo())`. |
| Duplicate files ingested | No deduplication check | The `checksum_sha256` index on `raw_files` enables dedup queries. Implement pre-ingest checks using this field. |

### Debugging Storage

Enable debug logging to see storage operations:

```python
import logging
from scinr.newton import configure

logging.basicConfig(level=logging.DEBUG)

configure(
    storage_backend="mongodb",
    mongodb_uri="mongodb://localhost:27017",
    log_level="DEBUG",
)
```

Debug output includes:

```
DEBUG:scinr.newton.storage.mongodb.raw_files:Stored raw file 'report.pdf' → raw_file_id=67a3..., gridfs_id=67a4... (2458624 bytes)
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
