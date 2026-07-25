"""
entity_extraction/neo4j_ops.py — Neo4j read/write operations for Stage 4.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from neo4j import AsyncDriver, Driver

from scinr.newton.entity_extraction.state import ExtractionTarget

log = logging.getLogger(__name__)


def fetch_extraction_targets(
    driver: Driver,
    document_name: str,
    only_unextracted: bool = False,
) -> list[ExtractionTarget]:
    """
    Fetch all StructureNodes that:
      - belong to *document_name*
      - have a ModelDecision (matched or unmatched)
      - have at least one InfoUnit

    When *only_unextracted* is True, nodes that already have a
    :HAS_EXTRACTION->(:ExtractionResult) relationship are excluded.

    Nodes with matched_model_class IS NULL will have model_class=None in the
    returned target, signalling the pipeline to use the default Triple extraction.

    Returns targets ordered by StructureNode.appearance_order with their
    InfoUnits ordered by iu.order (ascending). Re-running always re-fetches
    all nodes regardless of prior extraction status.
    """
    extra_filter = "AND NOT EXISTS { MATCH (n)-[:HAS_EXTRACTION]->(:ExtractionResult {source: 'tabular_raw'}) }"
    if only_unextracted:
        extra_filter += "\nAND NOT EXISTS { MATCH (n)-[:HAS_EXTRACTION]->(:ExtractionResult) }"

    query = f"""
    MATCH (d:Document {{name: $doc_name, latest: true}})-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
    MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
    WHERE EXISTS {{
      MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
    }}
    {extra_filter}
    OPTIONAL MATCH (md)-[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
    OPTIONAL MATCH (md)-[:HAS_SUPPLEMENTARY_FIELD]->(sf:SupplementaryField)
    WITH n, md,
         collect(DISTINCT CASE WHEN cm IS NOT NULL
                 THEN {{model_class: cm.model_class, coverage_note: cm.coverage_note}}
                 ELSE null END) AS complementary_raw,
         collect(DISTINCT CASE WHEN sf IS NOT NULL
                 THEN {{field_name: sf.field_name, field_type: sf.field_type,
                       description: sf.description, required: sf.required}}
                 ELSE null END) AS supplementary_raw
    MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
    WITH n, md,
         [c IN complementary_raw WHERE c IS NOT NULL] AS complementary,
         [s IN supplementary_raw WHERE s IS NOT NULL] AS supplementary,
         collect(iu {{.uid, .title, .description, .order}}) AS info_units_unsorted
    WITH n, md, complementary, supplementary,
         apoc.coll.sortMaps(info_units_unsorted, '^order') AS info_units_sorted
    RETURN n.id          AS node_full_id,
           n.node_id     AS node_id,
           n.title       AS node_title,
           md.matched_model_class AS model_class,
           complementary,
           supplementary,
           info_units_sorted AS info_units
    ORDER BY n.appearance_order
    """
    # Fallback query without apoc if needed
    fallback_query = f"""
    MATCH (d:Document {{name: $doc_name, latest: true}})-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
    MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
    WHERE EXISTS {{
      MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
    }}
    {extra_filter}
    OPTIONAL MATCH (md)-[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
    OPTIONAL MATCH (md)-[:HAS_SUPPLEMENTARY_FIELD]->(sf:SupplementaryField)
    WITH n, md,
         collect(DISTINCT CASE WHEN cm IS NOT NULL
                 THEN {{model_class: cm.model_class, coverage_note: cm.coverage_note}}
                 ELSE null END) AS complementary_raw,
         collect(DISTINCT CASE WHEN sf IS NOT NULL
                 THEN {{field_name: sf.field_name, field_type: sf.field_type,
                       description: sf.description, required: sf.required}}
                 ELSE null END) AS supplementary_raw
    MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
    WITH n, md,
         [c IN complementary_raw WHERE c IS NOT NULL] AS complementary,
         [s IN supplementary_raw WHERE s IS NOT NULL] AS supplementary,
         collect(iu {{.uid, .title, .description, .order}}) AS info_units
    RETURN n.id          AS node_full_id,
           n.node_id     AS node_id,
           n.title       AS node_title,
           md.matched_model_class AS model_class,
           complementary,
           supplementary,
           info_units
    ORDER BY n.appearance_order
    """

    with driver.session() as session:
        try:
            result = session.run(query, doc_name=document_name)
            rows = [dict(r) for r in result]
        except Exception:
            log.warning(
                "fetch_extraction_targets: apoc.coll.sortMaps unavailable, "
                "falling back to unsorted query"
            )
            result = session.run(fallback_query, doc_name=document_name)
            rows = [dict(r) for r in result]

        targets: list[ExtractionTarget] = []
        for row in rows:
            # Sort info_units by order in Python as a reliable fallback
            ius = list(row.get("info_units") or [])
            ius.sort(key=lambda x: (x.get("order") or 0))
            targets.append(ExtractionTarget(
                node_full_id=row["node_full_id"],
                node_id=row.get("node_id"),
                node_title=row.get("node_title"),
                model_class=row["model_class"],
                complementary_models=list(row.get("complementary") or []),
                supplementary_fields=list(row.get("supplementary") or []),
                info_units=ius,
            ))

    log.info(
        "fetch_extraction_targets: found %d targets for document %r",
        len(targets),
        document_name,
    )
    return targets


def mark_info_units_extracted(driver: Driver, node_full_id: str) -> None:
    """
    Set extracted = ISO timestamp on all InfoUnits of the given StructureNode.
    Only updates InfoUnits where extracted IS NULL (idempotent).
    """
    timestamp = datetime.now(UTC).isoformat()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:StructureNode {id: $nid})-[:HAS_INFO_UNIT]->(iu:InfoUnit)
            WHERE iu.extracted IS NULL
            SET iu.extracted = $timestamp
            RETURN count(iu) AS updated
            """,
            nid=node_full_id,
            timestamp=timestamp,
        )
        updated = result.single()["updated"]
    log.info(
        "mark_info_units_extracted: marked %d InfoUnits as extracted for node %r",
        updated,
        node_full_id,
    )


async def mark_info_units_extracted_async(driver: AsyncDriver, node_full_id: str) -> None:
    """Set ``extracted`` timestamp on all unextracted InfoUnits of the given StructureNode.

    Async version using the singleton async driver. Idempotent: only updates
    InfoUnits where ``extracted IS NULL``.
    """
    from datetime import datetime
    timestamp = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (n:StructureNode {id: $nid})-[:HAS_INFO_UNIT]->(iu:InfoUnit)
            WHERE iu.extracted IS NULL
            SET iu.extracted = $timestamp
            RETURN count(iu) AS updated
            """,
            nid=node_full_id,
            timestamp=timestamp,
        )
        record = await result.single()
        updated = record["updated"] if record else 0
    log.info(
        "mark_info_units_extracted_async: marked %d InfoUnits as extracted for node %r",
        updated, node_full_id,
    )
