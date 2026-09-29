# Navigation API

Read-only, engine-abstracted traversal of the knowledge graph. For tutorials and
recipes see the [Graph Navigation user guide](../user-guides/graph-navigation.md).

The graph store is pluggable, exactly like the [storage layer](storage.md): an
engine-agnostic `GraphNavigator` ABC plus a concrete `Neo4jGraphNavigator`,
selected by the `graph_backend` config field (env `GRAPH_BACKEND`, default
`"neo4j"`).

## Factory

::: scinr.newton.navigation.factory

## Base Interface

::: scinr.newton.navigation.base

## Scope

::: scinr.newton.navigation.scope

::: scinr.newton.navigation.scoped

## Return Types

::: scinr.newton.navigation.models

## Filter Operators

::: scinr.newton.navigation.filters

## Neo4j Backend

::: scinr.newton.navigation.neo4j.navigator

## Source-Text Bridge

Converted pages and the original uploaded file behind structure nodes / an
info unit / a document. Exposed as `GraphNavigator` methods
(`nav.get_structure_nodes_source_pages()`, `nav.get_document_original()`, … —
see the base interface, section I) and as the functions below, which take the
navigator as first argument. The graph lookup takes the scope filters (or the
`nav.scoped(tenant_id=...)` view); the storage read is then filtered by the
stored tenant of the resolved node / `:Document`, so a page id or `raw_file_id`
pointing at another tenant's upload returns nothing.

Structure nodes are read in batches only: `get_structure_nodes_source_pages()`
returns a `StructureNodesSourcePages` envelope — each page once in `pages`, each
node with its `page_ids`, and the status groups `not_found_structure_nodes`
(missing or outside the scope), `structure_nodes_without_pages` and, per node,
`not_found_page_ids` (missing or of another tenant). `get_document_original()` returns
an `OriginalFile` (the `RawFileRecord` plus a single-use byte stream).

::: scinr.newton.navigation.pages
