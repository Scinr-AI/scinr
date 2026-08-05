# Neo4j Graph Storage

`scinr` stores ingested documents and extracted domain data in Neo4j, so applications can query both document context and connected entities.

## What is stored

Each ingestion records the source document, its document structure, and the domain data extracted from that content. Related data is connected in the graph, making it possible to traverse from source material to extracted information and between related entities.

## Re-ingestion and versioning

You can ingest a new version of a document, update the latest version in place, or record a replacement document. Use `update_mode=True` or `replaces="ExistingDocument"` in the Python API; the CLI offers the equivalent `--update` and `--replaces` options.

## Querying

Use Neo4j's normal tools and query language to explore the graph. Query against the labels and properties present in your deployed graph rather than relying on undocumented internal storage details.
