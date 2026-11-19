# Graph Store API

`GraphStore` is the interface the pipeline uses to write to the graph (and to
make the reads that go with the writes). It is the interface to implement for a
new graph backend. The library ships two implementations: `Neo4jGraphStore`
(`scinr.newton.graph.neo4j.store`), which runs the same Cypher the pipeline always
had, and `SqlGraphStore` (`scinr.newton.graph.sql.store`), which serves PostgreSQL
and SQLite through SQLAlchemy Core. Read-only exploration of
the graph goes through [Navigation](navigation.md), not through the store.

Obtain the process-wide store with `get_graph_store()`. It is a singleton chosen
by `graph_backend`; `configure()` discards it, and `set_graph_store()` /
`reset_graph_store()` replace or discard it (for tests and custom backends).

Rules every backend follows:

- No engine objects (driver, session, transaction, query text) cross the interface.
- Each method owns its sessions and transactions.
- Retries live inside the implementation.
- Only the exceptions of the [graph hierarchy](exceptions.md) leave a method when
  the engine fails, with the engine's message and the original in `__cause__`.
- `sync_scope()` shares one synchronous connection among the calls made inside
  the block, including the worker threads started with `asyncio.to_thread`.

The contract test suite that every backend must pass lives in `tests/contract/`.
It uses only `GraphStore` (never query text or engine objects), runs every test
once per registered backend and observes the graph through a neutral dump;
`tests/contract/conftest.py` explains how to register a new backend. It wipes the
database it runs against, so each backend refuses to run unless that is safe:

- `neo4j` (with MongoDB for storage) runs only against the disposable containers of
  the golden harness, with `SCINR_GOLDEN_ALLOW_WIPE=1`, and is skipped otherwise.
- `sqlite` always runs: every test uses a new database file in a temporary directory.
- `postgres` runs only against the disposable `postgres-test` container with
  `SCINR_SQL_ALLOW_WIPE=1` (it drops its own test schema before each test) and is
  skipped otherwise.

## Obtaining the store

::: scinr.newton.graph
    options:
      members:
        - get_graph_store
        - set_graph_store
        - reset_graph_store

## Interface

::: scinr.newton.graph.base

## Sinks

A sink is the write side of a traversal that lives in the domain code: the
domain walks the Pydantic models and calls one sink method per statement. A
sink is one engine session, opened as an asynchronous context manager by the
store. `store.annotation.catalog_sink()` returns the `CatalogSink` that
`ensure_catalog_models` and `ensure_theme_structure`
(`scinr.newton.annotation.catalog`) write through; a backend implements those
eleven methods and does not need to reimplement the model traversal.
`store.extraction.sink()` returns the `ExtractionSink` that
`write_extraction_subgraph` and `write_triple_subgraph`
(`scinr.newton.entity_extraction.graph_mapper`) write through: the traversal of the
extracted instance (uids, timestamp, entity registry, which fields are written) is
domain code, and a backend implements the seventeen statement-level methods.
`store.tabular.sink()` returns the `TabularSink` that `write_tabular_subgraph`
(`scinr.newton.tabular.graph_writer`) writes through: the sheet traversal (batches of
rows, normalization passes, per-row extraction) is domain code, and a backend
implements four methods (`create_table`, `write_row_batch`,
`replace_raw_extraction_result`, `link_raw_column_value`). This sink opens its own
session per call instead of sharing one.

::: scinr.newton.graph.sinks

## SQL implementation

`SqlGraphStore` has the same nine operation groups as `Neo4jGraphStore`, written
in SQLAlchemy Core against the tables created by the migrations in
`scinr.newton.sql`. It uses the process-wide engines of `scinr.newton.sql.engine`
and is selected with `graph_backend="postgres"` or `"sqlite"` plus `sql_url`.
Driver errors are translated to the same graph exceptions by
`scinr.newton.sql.errors`. See the
[SQL graph backends guide](../user-guides/sql-graph.md).

::: scinr.newton.graph.sql.store

## Types

::: scinr.newton.graph.types
