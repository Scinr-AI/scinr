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
(`scinr.newton.utils.scope`); this module renders it for MongoDB.

::: scinr.newton.storage.filters

::: scinr.newton.utils.scope

## MongoDB Backend

### Client & GridFS

::: scinr.newton.storage.mongodb.client

### Page Repository

::: scinr.newton.storage.mongodb.pages

### Raw File Repository

::: scinr.newton.storage.mongodb.raw_files

## Null Backend

::: scinr.newton.storage.null
