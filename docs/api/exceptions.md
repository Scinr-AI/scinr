# Exceptions API

Every exception raised by the library inherits from `ScinrError`, so the whole
family can be caught with one clause.

## Hierarchy

```
ScinrError
├── ConfigurationError
│   └── GraphConfigurationError        the graph backend is not configured
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
    │   └── GraphAuthenticationError   the engine rejected the credentials
    ├── TransientGraphError            transient error, retries exhausted
    ├── GraphResourceError             the engine ran out of resources
    ├── GraphConstraintError           a write violated a constraint
    │   └── DocumentVersionConflict    the document version was already ingested
    └── GraphQueryError                the engine rejected the query
```

Each class has a single parent: there are no compatibility bases. In particular
`GraphConnectionError` is a `GraphError`, **not** a `NavigationError`.

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
| Graph credentials not configured | `GraphConfigurationError` (or `ConfigurationError`) |
| Bad input to the navigation API | `NavigationError` |

The graph store translates the engine's own exceptions (for Neo4j,
`neo4j.exceptions.*`) at its boundary, **after** the retries: the message is
the engine's and the original exception is chained in `__cause__`.

## Breaking changes (0.6.0)

| Before | Now | Write this instead |
|---|---|---|
| A version conflict raised `IngestionError` | `DocumentVersionConflict` (no longer an `IngestionError`) | `except DocumentVersionConflict` |
| Missing Neo4j credentials raised `ValueError` | `GraphConfigurationError` (no longer a `ValueError`) | `except GraphConfigurationError` or `except ConfigurationError` |
| `GraphConnectionError` was a `NavigationError` | It is a `GraphError` | `except (NavigationError, GraphConnectionError)` |
| Errors of `neo4j.exceptions.*` escaped from the write path | The classes above | `except GraphError` |

!!! note
    The version conflict still surfaces as `IngestionError` and the driver
    exceptions still escape from the write path until the graph store takes over
    those operations; the classes above are already importable from
    `scinr.newton`.

## Reference

::: scinr.newton.exceptions
    options:
      members_order: source
