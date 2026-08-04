"""
utils/document_resolver.py — Resolves document hierarchies via IS_COMPOSED_OF.

Given a document name, returns all leaf documents reachable through the
IS_COMPOSED_OF relationship in Neo4j. Leaf documents are those that have no
outgoing IS_COMPOSED_OF relationships.
"""
from __future__ import annotations

import logging

from neo4j import AsyncDriver, Driver

logger = logging.getLogger(__name__)


def resolve_leaf_document_names(driver: Driver, document_name: str) -> list[str]:
    """
    Given a document name, return all leaf document names reachable via IS_COMPOSED_OF.

    Leaf documents are those without outgoing IS_COMPOSED_OF relationships.
    If the document has no children (is itself a leaf), returns [document_name].

    The traversal is performed entirely in Cypher (variable-length path match),
    so it handles arbitrarily deep hierarchies without Python-level recursion.

    Args:
        driver: An open Neo4j driver instance.
        document_name: The exact Document.name as stored in Neo4j.

    Returns:
        Ordered list of leaf document names to process.
        Always contains at least [document_name] even if the document is not found.
    """
    query = """
    MATCH (root:Document {name: $name, latest: true})
    OPTIONAL MATCH (root)-[:IS_COMPOSED_OF*1..]->(leaf:Document {latest: true})
    WHERE NOT (leaf)-[:IS_COMPOSED_OF]->(:Document)
    WITH root, collect(leaf.name) AS leaf_names
    RETURN CASE
      WHEN size(leaf_names) > 0 THEN leaf_names
      ELSE [root.name]
    END AS names_to_process
    """
    with driver.session() as session:
        result = session.run(query, name=document_name)
        record = result.single()

    if record is None:
        logger.warning(
            "Document %r not found in Neo4j. Proceeding with original name.",
            document_name,
        )
        return [document_name]

    names: list[str] = record["names_to_process"]
    if len(names) > 1:
        logger.info(
            "Document %r resolved to %d leaf documents: %s",
            document_name,
            len(names),
            names,
        )
    return names


async def resolve_leaf_document_names_async(
    driver: AsyncDriver,
    document_name: str,
) -> list[str]:
    """
    Given a document name, return all leaf document names reachable via IS_COMPOSED_OF.

    Async version using the singleton async driver. Leaf documents are those
    without outgoing IS_COMPOSED_OF relationships. If the document has no
    children (is itself a leaf), returns [document_name].

    The traversal is performed entirely in Cypher (variable-length path match),
    so it handles arbitrarily deep hierarchies without Python-level recursion.

    Args:
        driver: An open Neo4j async driver instance.
        document_name: The exact Document.name as stored in Neo4j.

    Returns:
        Ordered list of leaf document names to process.
        Always contains at least [document_name] even if the document is not found.
    """
    query = """
    MATCH (root:Document {name: $name, latest: true})
    OPTIONAL MATCH (root)-[:IS_COMPOSED_OF*1..]->(leaf:Document {latest: true})
    WHERE NOT (leaf)-[:IS_COMPOSED_OF]->(:Document)
    WITH root, collect(leaf.name) AS leaf_names
    RETURN CASE
      WHEN size(leaf_names) > 0 THEN leaf_names
      ELSE [root.name]
    END AS names_to_process
    """
    async with driver.session() as session:
        result = await session.run(query, name=document_name)
        record = await result.single()

    if record is None:
        logger.warning(
            "Document %r not found in Neo4j. Proceeding with original name.",
            document_name,
        )
        return [document_name]

    names: list[str] = record["names_to_process"]
    if len(names) > 1:
        logger.info(
            "Document %r resolved to %d leaf documents: %s",
            document_name,
            len(names),
            names,
        )
    return names
