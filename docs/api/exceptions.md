# Exceptions API

Every exception raised by the library inherits from `ScinrError`, so the whole
family can be caught with one clause.

## Hierarchy

```
ScinrError
├── ConfigurationError
│   ├── GraphConfigurationError        the graph backend is not configured
│   └── SchemaVersionError             (also a GraphConnectionError) see below
├── PreconditionError
├── ExtractionError
├── IngestionError
├── ModelError
├── StorageError
│   └── ImageStorageError
├── FreezeError
├── ConversionError
├── NavigationError                    invalid navigation input or misuse
│   ├── UnsupportedOperationError
│   └── ScopeError                     (also a StorageError)
└── GraphError                         anything that comes from the graph engine
    ├── GraphConnectionError           the engine is unreachable
    │   ├── GraphAuthenticationError   the engine rejected the credentials
    │   └── SchemaVersionError         (also a ConfigurationError) see below
    ├── TransientGraphError            transient error, retries exhausted
    ├── GraphResourceError             the engine ran out of resources
    ├── GraphConstraintError           a write violated a constraint
    │   └── DocumentVersionConflict    the document version was already ingested
    ├── GraphQueryError                the engine rejected the query
    └── TenantAccessError              (also a StorageError) the database refused the
                                       operation for the tenant (Row-Level Security)
```

Three classes have two parents and appear twice in the tree:

| Class | Parents |
|---|---|
| `ScopeError` | `NavigationError`, `StorageError` |
| `TenantAccessError` | `GraphError`, `StorageError` |
| `SchemaVersionError` | `GraphConnectionError`, `ConfigurationError` |

`SchemaVersionError` (a row-level-security schema that is not at the version this
release needs, and the connected user cannot migrate it) is not exported from
`scinr.newton`; import it from `scinr.newton.sql.schema`.

Every other class has a single parent: there are no compatibility bases. In
particular `GraphConnectionError` is a `GraphError`, **not** a `NavigationError`.

## What to catch

| You want to handle | Catch |
|---|---|
| The graph engine is down or unreachable (read or write path) | `GraphConnectionError` |
| Wrong graph credentials | `GraphAuthenticationError` (a `GraphConnectionError`) |
| A deadlock or lost connection that persisted after all retries | `TransientGraphError` |
| Out of memory in the engine: reduce the batch or transaction size, do not retry | `GraphResourceError` |
| Two runs ingested the same document version: retry the ingestion | `DocumentVersionConflict` |
| Any other write that violates a graph constraint | `GraphConstraintError` |
| A query the engine rejects (a bug, not a condition to retry) | `GraphQueryError` |
| Any failure of the graph engine | `GraphError` |
| The database refused the operation for the tenant (row-level security) | `TenantAccessError` (on the storage path, also a `StorageError`) |
| A row-level-security schema older than this release needs, and the user cannot migrate it | `SchemaVersionError` (a `GraphConnectionError` and a `ConfigurationError`) |
| Graph backend not configured | `ConfigurationError` (`GraphConfigurationError` is its subclass) |
| Bad input to the navigation API | `NavigationError` |

The graph store translates the engine's own exceptions (for Neo4j,
`neo4j.exceptions.*`) at its boundary, **after** the retries: the message is
the engine's and the original exception is chained in `__cause__`. The SQL
backends do the same for their drivers in `scinr.newton.sql.errors`: the message
is the driver's, scrubbed of secrets and never containing the SQL statement or
its parameters. On the storage path the same driver errors become `StorageError`
(or `TenantAccessError` for a refusal).

## Breaking changes (0.6.0)

| Before | Now | Write this instead |
|---|---|---|
| A version conflict raised `IngestionError` | `DocumentVersionConflict` (no longer an `IngestionError`) | `except DocumentVersionConflict` |
| Missing Neo4j credentials raised `ValueError` | `ConfigurationError` from `configure()`; `GraphConfigurationError` (also a `ConfigurationError`) from the driver (neither is a `ValueError`) | `except ConfigurationError` |
| `GraphConnectionError` was a `NavigationError` | It is a `GraphError` | `except (NavigationError, GraphConnectionError)` |
| Errors of `neo4j.exceptions.*` escaped from the write path | The classes above | `except GraphError` |

## Reference

::: scinr.newton.exceptions
    options:
      members_order: source
