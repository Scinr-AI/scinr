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

## Return Types

::: scinr.newton.navigation.models

## Filter Operators

::: scinr.newton.navigation.filters

## Neo4j Backend

::: scinr.newton.navigation.neo4j.navigator

## Source-Text Bridge

::: scinr.newton.navigation.pages
