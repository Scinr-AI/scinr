"""
entity_extraction — Stage 4 of the scinr-ingest pipeline.

Traverses StructureNodes that have an AnnotationDecision with a matched model,
runs LLM-based entity extraction over their InfoUnits using a composite Pydantic
schema (primary + complementary + supplementary fields), and writes the result
as a typed subgraph to Neo4j.
"""
