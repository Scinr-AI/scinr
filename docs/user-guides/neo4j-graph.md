# Neo4j Graph Storage

`scinr` builds a connected graph representation of input documents and extracted domain concepts inside Neo4j.

---

## Graph Model

* `:Document` node: Represents the input file (`document_name`, format, ingestion timestamp).
* `:StructureNode`: Paragraphs, headings, tables, or figures linked via `[:HAS_CHILD]` relationships.
* Extracted Entity Nodes: Domain entities extracted during Stage 4, connected to their source `:StructureNode` via `[:EXTRACTED_FROM]`.
