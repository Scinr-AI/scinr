# Storage API

Document storage backends for raw file and page content archival.

## Backend Factory

::: scinr.newton.storage.factory

## Base Classes

::: scinr.newton.storage.base

## Records

`RawFileRecord`, `ConvertedPageRecord` (with `start_ms` / `end_ms` and the
`transcript` cues for audio and video segments since 0.5.0) and `PageImageRecord`
(a stored described image or video frame, 0.5.0). The image methods of
`RawFileRepository` (`stores_images`, `store_image`, `store_image_file`,
`list_images`, `open_image`, `delete_images`) are optional for custom backends.

::: scinr.newton.storage.models

## Scope Filters

The read / delete scope is shared with the navigation API
(`scinr.newton.utils.scope`); `scinr.newton.storage.filters` renders it for MongoDB.

::: scinr.newton.storage.filters

::: scinr.newton.utils.scope

## MongoDB Backend

### Client & GridFS

::: scinr.newton.storage.mongodb.client

### Page Repository

::: scinr.newton.storage.mongodb.pages

### Raw File Repository

::: scinr.newton.storage.mongodb.raw_files

## SQL Backend

The PostgreSQL and SQLite repositories (`storage_backend="postgres"` or `"sqlite"`,
configured with `sql_url`). Where the bytes live depends on the engine: PostgreSQL
stores them in chunk tables in the same transaction as the owner row; SQLite writes
one file per binary in `sqlite_files_dir` (by default a folder next to the database
file). `find_inconsistencies()` reports files without a row and rows without a file
after a crash; the library never calls it. On PostgreSQL the result is always clean.

### Raw File Repository

::: scinr.newton.storage.sql.raw_files

### Page Repository

::: scinr.newton.storage.sql.pages

### Blob Storage

::: scinr.newton.storage.sql.blobs
    options:
      members:
        - find_inconsistencies
        - Inconsistencies
        - get_blob_store
        - BlobStore

## Null Backend

::: scinr.newton.storage.null
