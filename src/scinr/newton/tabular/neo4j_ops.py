"""tabular/neo4j_ops.py — Neo4j write operations for the tabular ingestion pipeline."""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from neo4j import AsyncDriver
from pydantic import BaseModel

from scinr.newton.entity_extraction.schema_composer import _to_snake_case
from scinr.newton.tabular.reader import row_to_markdown
from scinr.newton.utils.neo4j_retry import with_neo4j_retry
from scinr.newton.utils.uid import make_uid

if TYPE_CHECKING:
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.tabular.models import ColumnMapping
    from scinr.newton.tabular.state import TabularFileData

logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 500
_PARALLEL_ROW_WRITES = 10

# ── Public entry point ────────────────────────────────────────────────────────


async def write_tabular_subgraph(
    driver: AsyncDriver,
    doc_path: str,
    document_name: str,
    resolved_version: int,
    sheet: TabularFileData,
    sheet_index: int,
    decision: AnnotationDecision,
    mapping: ColumnMapping,
    update_mode: bool = False,
    theme: str = "default",
    sheet_page_id: str = "",
) -> str:
    """Write the complete Table + Row subgraph for one sheet to Neo4j.

    Parameters
    ----------
    theme : The detected thematic domain path for this sheet (e.g.
        "pharmaceutical_quality"). Written to the Table node and all Row nodes
        as the ``theme`` property. Defaults to "default".
    sheet_page_id : MongoDB page_id for this sheet, or "" when no storage backend.

    Steps:
    1. Compute composite IDs for Table and Rows.
    2. (update_mode) delete existing Table + Row subgraph.
    3. MERGE Table StructureNode + link to Document.
    4. write_annotation() for the Table (creates ModelDecision + full subgraph).
    5. Compute ModelDecision UID (for linking Row nodes to same MD node).
    6. Resolve model class for extraction (if decision.matched_model_class is not None).
    7. Process rows in batches of _ROW_BATCH_SIZE:
       a. UNWIND-based batch: create Row nodes + InfoUnits + HAS_MODEL_DECISION links.
       b. write_extraction_subgraph() per row (or _write_raw_row_extraction if no model).

    Returns table_composite_id.
    """
    from scinr.newton.annotation.neo4j_ops import write_annotation
    from scinr.newton.entity_extraction.model_resolver import resolve_model_class
    from scinr.newton.entity_extraction.schema_composer import (
        compose_extraction_schema,
    )

    table_node_id = f"table_{sheet_index + 1}"
    table_composite_id = f"{doc_path}::{resolved_version}::{table_node_id}"
    headers = sheet["headers"]
    all_rows = sheet["all_rows"]

    # Step 2: delete existing subgraph if update_mode
    if update_mode:
        await delete_tabular_subgraph(driver, doc_path, resolved_version, sheet_index)

    # Step 3: create Table StructureNode and link to Document
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            await tx.run(
                """
                MERGE (n:StructureNode {id: $id})
                SET n:Table,
                    n.node_id = $node_id,
                    n.role = 'table',
                    n.title = $title,
                    n.appearance_order = $order,
                    n.theme = $theme,
                    n.column_count = $col_count,
                    n.row_count = $row_count,
                    n.source_page_ids = CASE WHEN $page_id <> '' THEN [$page_id] ELSE [] END
                WITH n
                MERGE (d:Document {path: $doc_path, version: $version})
                MERGE (d)-[:HAS_STRUCTURE]->(n)
                """,
                id=table_composite_id,
                node_id=table_node_id,
                title=sheet["sheet_name"],
                order=sheet_index + 1,
                theme=theme,
                col_count=len(headers),
                row_count=sheet["total_rows"],
                doc_path=doc_path,
                version=resolved_version,
                page_id=sheet_page_id,
            )
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise
    logger.info(
        "write_tabular_subgraph: Table node created: %s", table_composite_id
    )

    # Step 4: write ModelDecision on the Table (full subgraph via write_annotation)
    await write_annotation(driver, table_composite_id, decision, document_name)

    # Step 5: compute ModelDecision UID for linking Row nodes
    decision_uid = make_uid(
        "model_decision",
        table_composite_id,
        decision.matched_model_class
        if decision.matched_model_class is not None
        else "null",
    )

    # Step 6: resolve model class and build composite schema (once, reused for all rows)
    primary_cls = None
    composite_cls = None
    primary_field_name = None
    comp_class_names: list[str] = []
    comp_cls_map: dict[str, type] = {}

    if decision.matched_model_class is not None:
        try:
            primary_cls = resolve_model_class(decision.matched_model_class)
            comp_cls_list = []
            for cm in decision.complementary_models:
                try:
                    resolved = resolve_model_class(cm.model_class)
                    comp_cls_list.append(resolved)
                    comp_class_names.append(cm.model_class)
                    comp_cls_map[cm.model_class] = resolved
                except KeyError:
                    logger.warning(
                        "tabular: complementary model '%s' not in registry",
                        cm.model_class,
                    )

            # Build composite schema (primary + complementary + supplementary)
            composite_cls = compose_extraction_schema(
                primary_class=primary_cls,
                complementary_classes=comp_cls_list,
                supplementary_fields=[
                    sf.model_dump() for sf in decision.supplementary_fields
                ],
            )
            primary_field_name = _to_snake_case(primary_cls.__name__)
            logger.info(
                "tabular: using model '%s' for extraction (%d comp, %d supp)",
                decision.matched_model_class,
                len(comp_cls_list),
                len(decision.supplementary_fields),
            )
        except KeyError:
            logger.warning(
                "tabular: model '%s' not found in registry, falling back to raw storage",
                decision.matched_model_class,
            )
            primary_cls = None

    # Step 7: process rows in batches
    total_rows = len(all_rows)
    for batch_start in range(0, total_rows, _ROW_BATCH_SIZE):
        batch = all_rows[batch_start : batch_start + _ROW_BATCH_SIZE]
        await _write_row_batch(
            driver=driver,
            table_composite_id=table_composite_id,
            headers=headers,
            rows_batch=batch,
            batch_start_index=batch_start,
            decision_uid=decision_uid,
            decision=decision,
            mapping=mapping,
            primary_cls=primary_cls,
            composite_cls=composite_cls,
            primary_field_name=primary_field_name,
            comp_class_names=comp_class_names,
            comp_cls_map=comp_cls_map,
            document_name=document_name,
            theme=theme,
            sheet_page_id=sheet_page_id,
        )
        logger.debug(
            "tabular: wrote batch rows %d-%d of %d",
            batch_start + 1,
            min(batch_start + _ROW_BATCH_SIZE, total_rows),
            total_rows,
        )

    logger.info(
        "write_tabular_subgraph: complete for '%s' sheet '%s' (%d rows)",
        doc_path,
        sheet["sheet_name"],
        total_rows,
    )
    return table_composite_id


# ── Batch row writer ──────────────────────────────────────────────────────────


async def _write_row_batch(
    driver: AsyncDriver,
    table_composite_id: str,
    headers: list[str],
    rows_batch: list[list[str]],
    batch_start_index: int,
    decision_uid: str,
    decision: AnnotationDecision,
    mapping: ColumnMapping,
    primary_cls,
    composite_cls,
    primary_field_name: str | None,
    comp_class_names: list[str],
    comp_cls_map: dict[str, type],
    document_name: str,
    theme: str = "default",
    sheet_page_id: str = "",
) -> None:
    """Write a batch of Row nodes + InfoUnits + HAS_MODEL_DECISION links in one
    UNWIND transaction, then write ExtractionResult per row.
    """
    from scinr.newton.entity_extraction.graph_mapper import write_extraction_subgraph

    # Prepare row data for UNWIND
    rows_data = []
    for i, row_values in enumerate(rows_batch):
        row_index = batch_start_index + i
        row_node_id = f"row_{row_index + 1}"
        row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
        row_markdown = row_to_markdown(headers, row_values)
        info_uid = make_uid(row_composite_id, "row_data")
        rows_data.append({
            "id": row_composite_id,
            "node_id": row_node_id,
            "row_index": row_index,
            "appearance_order": row_index + 1,
            "info_uid": info_uid,
            "description": row_markdown,
        })

    # One transaction: create Row nodes + InfoUnits + HAS_MODEL_DECISION links (UNWIND)
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            await tx.run(
                """
                UNWIND $rows AS row_data
                MATCH (t:StructureNode {id: $table_id})
                MERGE (r:StructureNode {id: row_data.id})
                SET r:Row,
                    r.node_id = row_data.node_id,
                    r.role = 'row',
                    r.appearance_order = row_data.appearance_order,
                    r.theme = $theme,
                    r.row_index = row_data.row_index,
                    r.source_page_ids = CASE WHEN $page_id <> '' THEN [$page_id] ELSE [] END
                MERGE (t)-[:HAS_CHILD]->(r)
                WITH r, row_data
                MERGE (u:InfoUnit {uid: row_data.info_uid})
                SET u.title = 'Row data',
                    u.description = row_data.description,
                    u.order = 0
                MERGE (r)-[:HAS_INFO_UNIT]->(u)
                WITH r
                MATCH (md:ModelDecision {uid: $md_uid})
                MERGE (r)-[:HAS_MODEL_DECISION]->(md)
                """,
                rows=rows_data,
                table_id=table_composite_id,
                md_uid=decision_uid,
                theme=theme,
                page_id=sheet_page_id,
            )
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise

    # ── Phase 1: Instantiate composite models for all rows ────────────────────
    composite_results: list[tuple[int, list[str], BaseModel | None, str]] = []

    async def _instantiate_row(i: int, row_values: list[str]) -> tuple[int, list[str], BaseModel | None, str]:
        row_index = batch_start_index + i
        row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
        row_dict = _build_row_dict(headers, row_values)
        extraction_uid = make_uid(
            "tabular_extraction",
            row_composite_id,
            decision.matched_model_class or "raw",
        )

        if (
            composite_cls is not None
            and primary_cls is not None
            and primary_field_name is not None
        ):
            composite_instance = _instantiate_composite_from_row(
                primary_cls=primary_cls,
                composite_cls=composite_cls,
                primary_field_name=primary_field_name,
                mapping=mapping,
                row_dict=row_dict,
                comp_cls_map=comp_cls_map,
            )
            return (i, row_values, composite_instance, extraction_uid)
        else:
            return (i, row_values, None, extraction_uid)

    instantiate_tasks = [
        _instantiate_row(i, row_values)
        for i, row_values in enumerate(rows_batch)
    ]
    composite_results = await asyncio.gather(*instantiate_tasks)

    # ── Phase 2: Normalization hook (batch) ──────────────────────────────────
    if primary_cls is not None:
        from scinr.newton.config import get_config
        from scinr.newton.tabular.normalization.engine import NormalizationEngine

        cfg = get_config()
        normalization_instances: list[tuple[type[BaseModel], BaseModel]] = []

        for _i, _row_values, composite_instance, _extraction_uid in composite_results:
            if composite_instance is not None:
                # Extract primary instance from composite for normalization
                primary_instance = getattr(composite_instance, primary_field_name, None)
                if primary_instance is not None:
                    normalization_instances.append((primary_cls, primary_instance))

                # Extract complementary instances for normalization too
                for class_name in comp_class_names:
                    comp_cls = comp_cls_map.get(class_name)
                    if comp_cls is None:
                        continue
                    snake_name = _to_snake_case(class_name)
                    comp_instance = getattr(composite_instance, snake_name, None)
                    if comp_instance is not None:
                        normalization_instances.append((comp_cls, comp_instance))

        if normalization_instances and cfg.normalization_enabled:
            # Build normalization engine
            norm_llm = cfg.normalization_llm or cfg.llm
            engine = NormalizationEngine(
                llm=norm_llm,
                batch_size=cfg.normalization_batch_size,
            )
            try:
                normalized = await engine.normalize_instances(normalization_instances)
                logger.debug(
                    "tabular: normalized %d instances", len(normalized)
                )
            except Exception as _exc:
                logger.error(
                    "tabular: normalization batch failed: %s", _exc, exc_info=True
                )

    # ── Phase 3: Write to Neo4j (parallel) ────────────────────────────────────
    sem = asyncio.Semaphore(_PARALLEL_ROW_WRITES)

    async def _write_single_row(
        i: int,
        row_values: list[str],
        composite_instance: BaseModel | None,
        extraction_uid: str,
    ) -> None:
        async with sem:
            row_index = batch_start_index + i
            row_composite_id = f"{table_composite_id}/row_{row_index + 1}"

            if composite_instance is not None:
                try:
                    await write_extraction_subgraph(
                        driver=driver,
                        node_full_id=row_composite_id,
                        composite_instance=composite_instance,
                        primary_model_class=decision.matched_model_class,
                        complementary_model_classes=comp_class_names,
                        document_name=document_name,
                        extraction_uid=extraction_uid,
                    )
                except Exception as exc:
                    logger.warning(
                        "tabular: write_extraction_subgraph failed for %s: %s",
                        row_composite_id,
                        exc,
                    )
            else:
                # No-model fallback: store raw row data as properties
                row_dict = _build_row_dict(headers, row_values)
                try:
                    await _write_raw_row_extraction(
                        driver, row_composite_id, row_dict, document_name, extraction_uid
                    )
                except Exception as exc:
                    logger.warning(
                        "tabular: _write_raw_row_extraction failed for %s: %s",
                        row_composite_id,
                        exc,
                    )

    await asyncio.gather(*[
        _write_single_row(i, row_values, composite_instance, extraction_uid)
        for i, row_values, composite_instance, extraction_uid in composite_results
    ])


# ── Helper functions ──────────────────────────────────────────────────────────


def _build_row_dict(headers: list[str], row_values: list[str]) -> dict[str, str]:
    """Build {header: value} dict for a single row."""
    return {h: v for h, v in zip(headers, row_values) if h}


def _instantiate_composite_from_row(
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    mapping: ColumnMapping,
    row_dict: dict[str, str],
    comp_cls_map: dict[str, type] | None = None,
) -> BaseModel | None:
    """Build a composite model instance from row data using the column mapping.

    Routing (per ColumnFieldMapping entry):
    - confidence == 'low' OR model_field_name == '__extra__'  → skip
    - target_model == 'primary'       → primary_kwargs[field_name] = value
    - target_model == 'supplementary' → supplementary_kwargs[field_name] = value
    - any other str                   → comp_kwargs[target_model][field_name] = value

    After building primary_instance, for each (class_name, kwargs) in comp_kwargs:
    - Look up cls from comp_cls_map; if found, instantiate with model_construct
      and set on composite via the snake_case field name.

    Empty values are skipped (model uses its own defaults).
    """
    comp_cls_map = comp_cls_map or {}
    primary_field_set = set(primary_cls.model_fields.keys())
    primary_kwargs: dict[str, str] = {}
    supplementary_kwargs: dict[str, str] = {}
    comp_kwargs: dict[str, dict[str, str]] = {}

    for col_mapping in mapping.mappings:
        # Skip low-confidence and unmapped entries
        if col_mapping.confidence == "low":
            continue
        field_name = col_mapping.model_field_name
        if field_name == "__extra__":
            continue

        value = row_dict.get(col_mapping.column_name, "")
        if not value:
            continue  # skip empty values; model uses its own default

        target = col_mapping.target_model

        if target == "primary":
            if field_name in primary_field_set:
                primary_kwargs[field_name] = value
            else:
                logger.warning(
                    "_instantiate_composite_from_row: field %r not found in primary model %r "
                    "(target_model='primary' but field is not a primary field) — skipping",
                    field_name, primary_cls.__name__,
                )
        elif target == "supplementary":
            supplementary_kwargs[field_name] = value
        else:
            # Complementary model — keyed by CamelCase class name
            if target not in comp_kwargs:
                comp_kwargs[target] = {}
            comp_kwargs[target][field_name] = value

    # ── Instantiate primary model ──────────────────────────────────────────────
    try:
        primary_instance = primary_cls(**primary_kwargs)
    except Exception:
        try:
            primary_instance = primary_cls.model_construct(**primary_kwargs)
        except Exception:
            logger.warning(
                "tabular: could not instantiate '%s' from row data (kwargs: %s)",
                primary_cls.__name__, list(primary_kwargs.keys()),
            )
            return None

    # ── Build composite with primary + supplementary + complementary fields ────
    composite_init = {primary_field_name: primary_instance}
    composite_field_names = set(composite_cls.model_fields.keys())

    # Supplementary fields
    for k, v in supplementary_kwargs.items():
        if k in composite_field_names:
            composite_init[k] = v
        else:
            logger.debug(
                "tabular: supplementary field '%s' not in composite schema, skipping", k
            )

    # Complementary model instances
    for class_name, kwargs in comp_kwargs.items():
        comp_cls = comp_cls_map.get(class_name)
        if comp_cls is None:
            logger.debug(
                "tabular: complementary class '%s' not in comp_cls_map, skipping", class_name
            )
            continue
        try:
            comp_instance = comp_cls.model_construct(**kwargs)
            snake_name = _to_snake_case(class_name)
            if snake_name in composite_field_names:
                composite_init[snake_name] = comp_instance
            else:
                logger.debug(
                    "tabular: complementary field '%s' not in composite schema, skipping",
                    snake_name,
                )
        except Exception as exc:
            logger.warning(
                "tabular: could not build complementary instance '%s': %s", class_name, exc
            )

    try:
        composite_instance = composite_cls.model_construct(**composite_init)
        return composite_instance
    except Exception as exc:
        logger.warning("tabular: could not build composite instance: %s", exc)
        return None


def _sanitize_rel_type(col_name: str) -> str:
    """Convert a column name to a valid Neo4j relationship type."""
    sanitized = re.sub(r"[^A-Z0-9_]", "_", col_name.upper().strip())
    if not sanitized or sanitized[0].isdigit():
        sanitized = "COL_" + sanitized
    return f"HAS_{sanitized}"


async def _write_raw_row_extraction(
    driver: AsyncDriver,
    row_composite_id: str,
    row_dict: dict[str, str],
    document_name: str,
    extraction_uid: str,
) -> None:
    """Write a raw ExtractionResult for rows where no model matched.

    Each column value is written as a global :Entity singleton (MERGEd by
    normalized_value), linked via a dynamic HAS_{COLUMN_NAME} relationship —
    mirroring the triple extraction pipeline so that tabular values can
    cross-reference with Entity nodes from other documents.

    model_class = 'GenericTabularRow', source = 'tabular_raw'.
    """
    timestamp = datetime.now(UTC).isoformat()

    # Filter empty/None values and truncate long strings
    valid_props = {
        k: (v[:500] if isinstance(v, str) and len(v) > 500 else v)
        for k, v in row_dict.items()
        if k and v
    }

    async with driver.session() as session:
        # Idempotency: delete stale ExtractionResult only.
        # :Entity nodes are global singletons — never deleted here.
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
            DETACH DELETE er
            """,
            nid=row_composite_id,
        )
        # Create fresh ExtractionResult node
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})
            CREATE (er:ExtractionResult {
                uid:           $uid,
                node_full_id:  $nid,
                document_name: $doc_name,
                model_class:   'GenericTabularRow',
                source:        'tabular_raw',
                timestamp:     $timestamp
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=row_composite_id,
            uid=extraction_uid,
            doc_name=document_name,
            timestamp=timestamp,
        )

    # MERGE one global :Entity per column value, linked via dynamic HAS_{COL} rel
    for col_name, col_value in valid_props.items():
        rel_type = _sanitize_rel_type(col_name)
        normalized = col_value.strip().lower()
        # UID mirrors triple pipeline: keyed only on normalized value → global singleton
        entity_uid = make_uid("entity", normalized)
        try:
            async with driver.session() as session:
                # Capture loop variables for the lambda closure
                _query = f"""
                    MERGE (e:Entity {{uid: $uid}})
                    ON CREATE SET e.normalized_value = $normalized_value,
                              e.value = $value
                    WITH e
                    MATCH (er:ExtractionResult {{uid: $er_uid}})
                    MERGE (er)-[:{rel_type}]->(e)
                    """
                _params = dict(
                    normalized_value=normalized,
                    uid=entity_uid,
                    value=col_value,
                    er_uid=extraction_uid,
                )
                await with_neo4j_retry(lambda: session.run(_query, **_params))
        except Exception as exc:
            logger.warning(
                "_write_raw_row_extraction: failed to write column '%s' for %s after retries: %s",
                col_name, row_composite_id, exc,
            )


# ── Subgraph deletion ─────────────────────────────────────────────────────────


async def delete_tabular_subgraph(
    driver: AsyncDriver,
    doc_path: str,
    resolved_version: int,
    sheet_index: int,
) -> None:
    """Delete an existing Table node and all its Row descendants before re-insertion.

    Used in update_mode. Deletes in leaf-first order:
    1. ExtractionResult ModelInstance children on Row nodes
    2. ExtractionResults on Row nodes
    3. InfoUnits on Row nodes
    4. Row nodes (DETACH DELETE removes HAS_CHILD, HAS_MODEL_DECISION, HAS_INFO_UNIT)
    5. ModelDecision subgraph on Table (leaf-first)
    6. Table node (DETACH DELETE removes HAS_STRUCTURE from Document)

    NOTE: ModelDecision deletion is handled by write_annotation()'s own idempotency
    guard, so we only need to detach-delete the Table itself.
    LabeledEntity singletons are NOT deleted (global across the graph).
    """
    table_node_id = f"table_{sheet_index + 1}"
    table_composite_id = f"{doc_path}::{resolved_version}::{table_node_id}"

    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            # 1. ModelInstance children under ExtractionResults on Rows
            # Skip tabular_raw rows: their ExtractionResult links to :Entity singletons
            # which must NOT be deleted here.
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                      -[:HAS_EXTRACTION]->(er:ExtractionResult)
                WHERE er.source <> 'tabular_raw'
                WITH er
                MATCH (er)-[*1..10]->(mi:ModelInstance)
                DETACH DELETE mi
                """,
                table_id=table_composite_id,
            )
            # 2. ExtractionResults on Rows
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                      -[:HAS_EXTRACTION]->(er:ExtractionResult)
                DETACH DELETE er
                """,
                table_id=table_composite_id,
            )
            # 3. InfoUnits on Rows
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                      -[:HAS_INFO_UNIT]->(iu:InfoUnit)
                DETACH DELETE iu
                """,
                table_id=table_composite_id,
            )
            # 4. Row nodes
            await tx.run(
                """
                MATCH (t:StructureNode {id: $table_id})-[:HAS_CHILD]->(r:StructureNode:Row)
                DETACH DELETE r
                """,
                table_id=table_composite_id,
            )
            # 5. ModelDecision subgraph on Table (leaf-first)
            await tx.run(
                """
                MATCH (:StructureNode {id: $id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_PROPOSED_MODEL]->(:ProposedModel)-[:HAS_PROPOSED_FIELD]->(pf)
                DETACH DELETE pf
                """,
                id=table_composite_id,
            )
            await tx.run(
                """
                MATCH (:StructureNode {id: $id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_PROPOSED_MODEL]->(pm)
                DETACH DELETE pm
                """,
                id=table_composite_id,
            )
            await tx.run(
                """
                MATCH (:StructureNode {id: $id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_SUPPLEMENTARY_FIELD]->(sf)
                DETACH DELETE sf
                """,
                id=table_composite_id,
            )
            await tx.run(
                """
                MATCH (:StructureNode {id: $id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_COMPLEMENTARY_MATCH]->(cm)
                DETACH DELETE cm
                """,
                id=table_composite_id,
            )
            await tx.run(
                """
                MATCH (:StructureNode {id: $id})-[:HAS_MODEL_DECISION]->(md)
                DETACH DELETE md
                """,
                id=table_composite_id,
            )
            # 6. Table node
            await tx.run(
                "MATCH (n:StructureNode {id: $id}) DETACH DELETE n",
                id=table_composite_id,
            )
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise

    logger.info(
        "delete_tabular_subgraph: deleted table at %s", table_composite_id
    )
