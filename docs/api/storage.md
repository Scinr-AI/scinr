# Storage API

Document storage backends for raw file and page content archival.

## Backend Factory

::: scinr.newton.storage.factory

## Base Classes

::: scinr.newton.storage.base

## Records

`RawFileRecord` and `ConvertedPageRecord` (with `start_ms` / `end_ms` for audio and
video segments since 0.5.0).

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
