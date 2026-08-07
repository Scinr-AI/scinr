"""tabular/neo4j_ops.py — Neo4j write operations for the tabular ingestion pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import types as _builtin_types
import typing
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from neo4j import AsyncDriver
from pydantic import BaseModel

from scinr.newton.entity_extraction.schema_composer import _to_snake_case
from scinr.newton.tabular.normalization.detector import (
    extract_source_values_from_dict,
    get_normalization_specs,
    instance_has_normalizable_fields,
)
from scinr.newton.tabular.normalization.models import NormalizationEntry
from scinr.newton.tabular.reader import row_to_markdown
from scinr.newton.utils.neo4j_retry import with_neo4j_retry
from scinr.newton.utils.uid import make_uid
from scinr.newton.tabular.normalization.engine import NormalizationEngine

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
    logger.info("write_tabular_subgraph: Table node created: %s", table_composite_id)

    # Step 4: write ModelDecision on the Table (full subgraph via write_annotation)
    await write_annotation(driver, table_composite_id, decision, document_name)

    # Step 5: compute ModelDecision UID for linking Row nodes
    decision_uid = make_uid(
        "model_decision",
        table_composite_id,
        decision.matched_model_class if decision.matched_model_class is not None else "null",
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
                supplementary_fields=[sf.model_dump() for sf in decision.supplementary_fields],
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

    # Step 7: process rows — bifurcate based on normalization
    from scinr.newton.config import get_config

    cfg = get_config()
    has_normalization = (
        primary_cls is not None
        and cfg.normalization_enabled
        and (
            instance_has_normalizable_fields(primary_cls)
            or any(
                instance_has_normalizable_fields(comp_cls_map[cn])
                for cn in comp_class_names
                if comp_cls_map.get(cn) is not None
            )
        )
    )

    total_rows = len(all_rows)

    if has_normalization:
        logger.info(
            "tabular: using normalization-first write path for '%s' sheet '%s' (%d rows)",
            doc_path,
            sheet["sheet_name"],
            total_rows,
        )
        await _write_tabular_with_normalization(
            driver=driver,
            table_composite_id=table_composite_id,
            headers=headers,
            all_rows=all_rows,
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
    else:
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


async def _write_tabular_with_normalization(
    driver: AsyncDriver,
    table_composite_id: str,
    headers: list[str],
    all_rows: list[list[str]],
    decision_uid: str,
    decision: AnnotationDecision,
    mapping: ColumnMapping,
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    comp_class_names: list[str],
    comp_cls_map: dict[str, type],
    document_name: str,
    theme: str,
    sheet_page_id: str,
) -> None:
    """Write all rows using normalization-first batching.

    Flow:
    1. Pre-scan all rows → global dedup map
    2. Group unique keys by target_type
    3. For each type → for each batch of keys:
       a. LLM extraction via engine.process_key_batch()
       b. Collect affected row indices
       c. Instantiate composites with cached normalization
       d. Write to Neo4j in batches
    4. Write any remaining rows (normalization failures) with None
    """
    from scinr.newton.config import get_config
    from scinr.newton.tabular.normalization.engine import NormalizationEngine

    cfg = get_config()
    norm_llm = cfg.normalization_llm or cfg.llm

    # Step 1: Pre-scan all rows → dedup map
    logger.info("tabular: building normalization dedup map for %d rows", len(all_rows))
    # Cache normalization specs by class (called O(rows) times otherwise)
    _specs_cache: dict[type, list] = {}

    def _get_specs_cached(cls: type) -> list:
        if cls not in _specs_cache:
            _specs_cache[cls] = get_normalization_specs(cls)
        return _specs_cache[cls]

    dedup_map = _build_normalization_dedup_map(
        headers=headers,
        all_rows=all_rows,
        primary_cls=primary_cls,
        comp_cls_map=comp_cls_map,
        comp_class_names=comp_class_names,
        mapping=mapping,
        get_specs_fn=_get_specs_cached,
    )
    logger.info(
        "tabular: dedup map has %d unique keys across %d rows",
        len(dedup_map),
        len(all_rows),
    )

    if not dedup_map:
        # No normalizable fields found — fall back to standard path
        logger.info("tabular: no normalizable fields, using standard write path")
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
        return

    # Step 2: Create engine (persists across all key batches)
    engine = NormalizationEngine(
        llm=norm_llm,
        batch_size=cfg.normalization_batch_size,
    )

    # Step 3: Group unique keys by target_type (LLM needs homogeneous batches)
    keys_by_type: dict[str, list[NormalizationEntry]] = {}
    for entry in dedup_map.values():
        type_key = entry.target_type.__name__
        if type_key not in keys_by_type:
            keys_by_type[type_key] = []
        keys_by_type[type_key].append(entry)

    written_row_indices: set[int] = set()
    total_rows = len(all_rows)
    write_batch_size = _ROW_BATCH_SIZE

    # Step 4a: Collect all key batches
    all_key_batches: list[tuple[list[NormalizationEntry], str]] = []
    for type_name, type_entries in keys_by_type.items():
        for key_batch_start in range(0, len(type_entries), cfg.normalization_batch_size):
            key_batch = type_entries[
                key_batch_start : key_batch_start + cfg.normalization_batch_size
            ]
            all_key_batches.append((key_batch, type_name))

    # Step 4b: LLM extraction for ALL key batches concurrently.
    # Uses the global get_llm_semaphore() (not a local semaphore) so that
    # normalization LLM calls share the same bounded concurrency pool as
    # every other Bedrock caller in the pipeline (extraction, entity
    # extraction, annotation), avoiding overshooting the botocore connection
    # pool when multiple documents/tables are processed in parallel.
    from scinr.newton.config import get_llm_semaphore

    async def _extract_key_batch(
        key_batch: list[NormalizationEntry],
        type_name: str,
    ) -> None:
        async with get_llm_semaphore():
            logger.info(
                "tabular: normalizing batch of %d keys (type: %s)",
                len(key_batch),
                type_name,
            )
            results = await engine.process_key_batch(key_batch)
            logger.info("tabular: normalization batch returned %d results", len(results))

    extraction_tasks = [
        asyncio.create_task(_extract_key_batch(kb, tn)) for kb, tn in all_key_batches
    ]
    await asyncio.gather(*extraction_tasks)

    # Step 4c-d: Instantiate and write ALL rows with cached normalization
    # Iterate by row (not by key_batch) so that rows with multiple
    # normalizable fields of different target types get ALL normalizations applied.
    # Rows in dedup_map: have at least one normalizable field
    # Rows NOT in dedup_map: no normalizable fields (or all empty) → plain path

    # Build set of row indices that appear in dedup_map
    rows_with_normalization: set[int] = set()
    for entry in dedup_map.values():
        for ri in entry.row_indices:
            rows_with_normalization.add(ri)

    # Write rows WITH normalization
    normalized_indices = sorted(rows_with_normalization)
    for chunk_start in range(0, len(normalized_indices), write_batch_size):
        chunk_indices = normalized_indices[chunk_start : chunk_start + write_batch_size]

        # Instantiate composites with cached normalization
        composites = [
            _instantiate_composite_with_normalization(
                table_composite_id=table_composite_id,
                row_index=ri,
                headers=headers,
                row_values=all_rows[ri],
                mapping=mapping,
                decision=decision,
                primary_cls=primary_cls,
                composite_cls=composite_cls,
                primary_field_name=primary_field_name,
                comp_cls_map=comp_cls_map,
                comp_class_names=comp_class_names,
                engine=engine,
                get_specs_fn=_get_specs_cached,
            )
            for ri in chunk_indices
        ]

        # Transform absolute row indices to offsets from min_idx
        min_idx = min(chunk_indices)
        composites = [
            (ri - min_idx, row_values, composite_instance, extraction_uid)
            for ri, row_values, composite_instance, extraction_uid in composites
        ]

        # Write to Neo4j
        rows_batch = [all_rows[ri] for ri in chunk_indices]
        await _write_row_batch(
            driver=driver,
            table_composite_id=table_composite_id,
            headers=headers,
            rows_batch=rows_batch,
            batch_start_index=min_idx,
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
            composite_results=list(composites),
            row_indices=chunk_indices,
        )

        for ri in chunk_indices:
            written_row_indices.add(ri)

        logger.debug(
            "tabular: wrote %d rows with normalization (indices %s)",
            len(chunk_indices), chunk_indices,
        )

    # Step 5: Write any remaining rows (normalization failures)
    remaining_indices = [i for i in range(total_rows) if i not in written_row_indices]
    if remaining_indices:
        logger.warning(
            "tabular: %d rows not written during normalization batches. "
            "Writing with best-effort (normalization fields may be None).",
            len(remaining_indices),
        )

        # Log which rows and why (first 20)
        for ri in remaining_indices[:20]:
            row_dict = _build_row_dict(headers, all_rows[ri])
            primary_kwargs, _, _ = _route_row_values(row_dict, mapping, primary_cls)
            for spec in _get_specs_cached(primary_cls):
                source_values = extract_source_values_from_dict(spec, primary_kwargs, primary_cls)
                if source_values:
                    unique_key = (
                        f"{spec.target_type.__name__}:"
                        f"{NormalizationEngine._hash_source_values(source_values)}"
                    )
                    if unique_key not in engine.result_cache:
                        logger.warning(
                            "tabular: row %d missing normalization for key %s (source: %s)",
                            ri,
                            unique_key,
                            source_values,
                        )

        # Write remaining in batches — instantiate without normalization
        for chunk_start in range(0, len(remaining_indices), write_batch_size):
            chunk_indices = remaining_indices[chunk_start : chunk_start + write_batch_size]

            # Instantiate composites WITHOUT normalization (fields stay as-is)
            composites = [
                _instantiate_composite_plain(
                    table_composite_id=table_composite_id,
                    row_index=ri,
                    headers=headers,
                    row_values=all_rows[ri],
                    mapping=mapping,
                    decision=decision,
                    primary_cls=primary_cls,
                    composite_cls=composite_cls,
                    primary_field_name=primary_field_name,
                    comp_cls_map=comp_cls_map,
                )
                for ri in chunk_indices
            ]

            # Transform indices
            min_idx = min(chunk_indices)
            composites = [
                (ri - min_idx, row_values, composite_instance, extraction_uid)
                for ri, row_values, composite_instance, extraction_uid in composites
            ]

            rows_batch = [all_rows[ri] for ri in chunk_indices]
            await _write_row_batch(
                driver=driver,
                table_composite_id=table_composite_id,
                headers=headers,
                rows_batch=rows_batch,
                batch_start_index=min_idx,
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
                composite_results=list(composites),
                row_indices=chunk_indices,
            )


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
    composite_results: list[tuple[int, list[str], BaseModel | None, str]] | None = None,
    row_indices: list[int] | None = None,
) -> None:
    """Write a batch of Row nodes + InfoUnits + HAS_MODEL_DECISION links in one
    UNWIND transaction, then write ExtractionResult per row.
    """
    from scinr.newton.entity_extraction.graph_mapper import write_extraction_subgraph

    # Prepare row data for UNWIND
    rows_data = []
    if row_indices is not None:
        # Pre-built composites: use actual row indices
        for idx, row_values in zip(row_indices, rows_batch):
            row_node_id = f"row_{idx + 1}"
            row_composite_id = f"{table_composite_id}/row_{idx + 1}"
            row_markdown = row_to_markdown(headers, row_values)
            info_uid = make_uid(row_composite_id, "row_data")
            rows_data.append(
                {
                    "id": row_composite_id,
                    "node_id": row_node_id,
                    "row_index": idx,
                    "appearance_order": idx + 1,
                    "info_uid": info_uid,
                    "description": row_markdown,
                }
            )
    else:
        for i, row_values in enumerate(rows_batch):
            row_index = batch_start_index + i
            row_node_id = f"row_{row_index + 1}"
            row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
            row_markdown = row_to_markdown(headers, row_values)
            info_uid = make_uid(row_composite_id, "row_data")
            rows_data.append(
                {
                    "id": row_composite_id,
                    "node_id": row_node_id,
                    "row_index": row_index,
                    "appearance_order": row_index + 1,
                    "info_uid": info_uid,
                    "description": row_markdown,
                }
            )

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

    # ── Phase 1 + 2: Use pre-built composites or instantiate inline ────────────
    if composite_results is None:
        # Standard path: instantiate + normalize inline (same as before)
        composite_results: list[tuple[int, list[str], BaseModel | None, str]] = []

        async def _instantiate_row(
            i: int, row_values: list[str]
        ) -> tuple[int, list[str], BaseModel | None, str]:
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
            _instantiate_row(i, row_values) for i, row_values in enumerate(rows_batch)
        ]
        composite_results = await asyncio.gather(*instantiate_tasks)

        # ── Phase 2: Normalization hook (inline, for standard path) ────────────
        if primary_cls is not None:
            from scinr.newton.config import get_config
            from scinr.newton.tabular.normalization.engine import NormalizationEngine

            cfg = get_config()
            normalization_instances: list[tuple[type[BaseModel], BaseModel]] = []

            for _i, _row_values, composite_instance, _extraction_uid in composite_results:
                if composite_instance is not None:
                    primary_instance = getattr(composite_instance, primary_field_name, None)
                    if primary_instance is not None:
                        normalization_instances.append((primary_cls, primary_instance))
                    for class_name in comp_class_names:
                        comp_cls = comp_cls_map.get(class_name)
                        if comp_cls is None:
                            continue
                        snake_name = _to_snake_case(class_name)
                        comp_instance = getattr(composite_instance, snake_name, None)
                        if comp_instance is not None:
                            normalization_instances.append((comp_cls, comp_instance))

            if normalization_instances and cfg.normalization_enabled:
                norm_llm = cfg.normalization_llm or cfg.llm
                engine = NormalizationEngine(
                    llm=norm_llm,
                    batch_size=cfg.normalization_batch_size,
                )
                try:
                    normalized = await engine.normalize_instances(normalization_instances)
                    logger.debug("tabular: normalized %d instances", len(normalized))
                except Exception as _exc:
                    logger.error("tabular: normalization batch failed: %s", _exc, exc_info=True)

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

    await asyncio.gather(
        *[
            _write_single_row(i, row_values, composite_instance, extraction_uid)
            for i, row_values, composite_instance, extraction_uid in composite_results
        ]
    )


# ── Helper functions ──────────────────────────────────────────────────────────


def _build_row_dict(headers: list[str], row_values: list[str]) -> dict[str, str]:
    """Build {header: value} dict for a single row."""
    return {h: v for h, v in zip(headers, row_values) if h}


def _merge_values(raw_values: list[str]) -> list[str]:
    """Merge raw column values mapped to the same target field, deduplicating
    by containment (substring OR word-token subset), processed sequentially in
    the given order. Preserves original formatting (trim only) of surviving values.

    Sequential semantics:
    - If a bigger/superset value was already accepted, a later smaller/subset value
      is skipped entirely (not inserted).
    - If a smaller/subset value was already accepted and a later bigger/superset
      value arrives, the smaller survivor(s) are removed and replaced by the new one.
    - Values with no containment relation to any existing survivor are kept as
      distinct entries, in order of first appearance.

    100% deterministic: same input list always produces the same output list.
    """

    def _normalize(s: str) -> str:
        return " ".join(s.split()).lower()

    survivors: list[str] = []  # original strings (trimmed only), in order
    survivors_norm: list[str] = []  # parallel normalized forms, for comparison only

    for raw in raw_values:
        original = raw.strip()
        if not original:
            continue
        norm = _normalize(original)
        if not norm:
            continue
        norm_tokens = set(norm.split())

        # 1. Is this new value redundant vs. an existing survivor?
        redundant = False
        for s_norm in survivors_norm:
            s_tokens = set(s_norm.split())
            if norm in s_norm or norm_tokens <= s_tokens:
                redundant = True
                break
        if redundant:
            continue

        # 2. Does this new value make existing survivors redundant? (bigger arrives later)
        kept_pairs = []
        for orig_s, s_norm in zip(survivors, survivors_norm, strict=True):
            s_tokens = set(s_norm.split())
            if s_norm in norm or s_tokens <= norm_tokens:
                continue  # drop the smaller/subset survivor
            kept_pairs.append((orig_s, s_norm))
        survivors = [p[0] for p in kept_pairs]
        survivors_norm = [p[1] for p in kept_pairs]

        # 3. Insert the new value
        survivors.append(original)
        survivors_norm.append(norm)

    return survivors


def _merge_and_join(raw_values: list[str], field_name: str, context: str) -> str:
    """Merge raw_values via _merge_values() and join the survivors with "; ".

    Shared by the str-typed assembly paths in _route_row_values() (primary
    str fields, supplementary fields, complementary fields) to avoid
    repeating the merge -> join -> debug-log pattern. Logs a debug message
    when more than one raw value was received, describing the merge outcome.

    `context` is a short human-readable description of where the field lives
    (e.g. "primary", "supplementary", or "complementary 'ClassName'") and is
    interpolated directly before " field %r received ..." in the log message.
    """
    merged = _merge_values(raw_values)
    final_str = "; ".join(merged)
    if len(raw_values) > 1:
        logger.debug(
            "_route_row_values: %s field %r received %d value(s), "
            "%d survived merge -> %r",
            context,
            field_name,
            len(raw_values),
            len(merged),
            final_str,
        )
    return final_str


def _classify_merge_type(annotation: object) -> str:
    """Classify a Pydantic field annotation for _route_row_values() merging.

    Unwraps Optional[X]/Union[X, None] to find the real inner type, then
    classifies it as one of:
    - "str": the inner type is `str`, or the type cannot be determined
      confidently (no annotation, `Any`, complex multi-member Union). Fail-safe
      default.
    - "list_str": the inner type is `list[str]`, or a bare unparametrized
      `list`. Fail-safe default for list-like fields.
    - "other": any other type (int, float, bool, date, datetime, Enum, a
      nested BaseModel, etc.) — not eligible for containment-based merging.
    """
    if annotation is None or annotation is Any:
        return "str"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Union[...] / Optional[...] (incl. PEP 604 `X | None`) → unwrap
    is_union = origin is typing.Union
    if not is_union and hasattr(_builtin_types, "UnionType"):
        is_union = isinstance(annotation, _builtin_types.UnionType)
    if is_union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _classify_merge_type(non_none_args[0])
        # Complex multi-member Union — can't determine confidently.
        return "str"

    if annotation is str:
        return "str"

    if origin is list:
        if not args or args[0] is str:
            return "list_str"
        return "other"

    if annotation is list:
        # Bare `list` with no subscript at all.
        return "list_str"

    return "other"


def _route_row_values(
    row_dict: dict[str, str],
    mapping: ColumnMapping,
    primary_cls: type,
) -> tuple[dict[str, str | list[str]], dict[str, str], dict[str, dict[str, str]]]:
    """Route row dict values to primary, supplementary, and complementary kwargs.

    Returns (primary_kwargs, supplementary_kwargs, comp_kwargs) where comp_kwargs
    is {CamelCaseClassName: {field_name: value}}.

    When multiple columns map to the same (target, field_name), their raw
    values are combined via containment-based deduplication (see
    _merge_values()) instead of the last column silently overwriting the
    others. Columns are processed in mapping.mappings order (their order of
    appearance), which keeps the whole pipeline deterministic: the same
    (row_dict, mapping, primary_cls) always yields the same output.

    Type-aware assembly (primary target only, since only there do we have the
    real Pydantic class to introspect via primary_cls.model_fields):
    - str fields               → "; ".join(merged_values)   (str)
    - list[str] fields         → merged_values               (list[str])
    - any other type           → last raw value wins, unmerged (str), with a
      warning suggesting str/list[str] or a model_validator for type coercion.

    supplementary_kwargs and comp_kwargs entries have no accessible Pydantic
    class at this point, so they are always treated as str: merged via
    _merge_values() and joined with "; ".

    Same routing logic as _instantiate_composite_from_row but returns raw dicts
    instead of instantiating models. Reusable for both pre-scan and instantiation.
    """
    primary_field_set = set(primary_cls.model_fields.keys())

    # Pass 1: collect candidate raw values per (target, field_name), preserving
    # mapping.mappings order (insertion order of dict keys == first-seen order).
    primary_raw: dict[str, list[str]] = {}
    supplementary_raw: dict[str, list[str]] = {}
    comp_raw: dict[str, dict[str, list[str]]] = {}

    for col_mapping in mapping.mappings:
        if col_mapping.confidence == "low":
            continue
        field_name = col_mapping.model_field_name
        if field_name == "__extra__":
            continue

        value = row_dict.get(col_mapping.column_name, "")
        if not value:
            continue

        target = col_mapping.target_model

        if target == "primary":
            if field_name in primary_field_set:
                primary_raw.setdefault(field_name, []).append(value)
            else:
                logger.warning(
                    "_route_row_values: field %r not found in primary model %r — skipping",
                    field_name,
                    primary_cls.__name__,
                )
        elif target == "supplementary":
            supplementary_raw.setdefault(field_name, []).append(value)
        else:
            comp_raw.setdefault(target, {}).setdefault(field_name, []).append(value)

    # Pass 2: assemble final kwargs, merging/deduping and applying type-aware
    # join semantics for primary; str-only merge for supplementary/complementary.
    primary_kwargs: dict[str, str | list[str]] = {}
    supplementary_kwargs: dict[str, str] = {}
    comp_kwargs: dict[str, dict[str, str]] = {}

    for field_name, raw_values in primary_raw.items():
        field_info = primary_cls.model_fields.get(field_name)
        annotation = field_info.annotation if field_info is not None else None
        category = _classify_merge_type(annotation)

        if category == "other":
            final_value: str | list[str] = raw_values[-1]
            if len(raw_values) > 1:
                logger.warning(
                    "_route_row_values: field %r of model %r has an unsupported "
                    "type for value merging (%r) — %d values received, using "
                    "the last one (%r) without combining. Consider using str "
                    "or list[str] for this field if it will receive multiple "
                    "mapped columns in the tabular pipeline, or add a Pydantic "
                    "model_validator to coerce/merge the value after construction.",
                    field_name,
                    primary_cls.__name__,
                    annotation,
                    len(raw_values),
                    final_value,
                )
            primary_kwargs[field_name] = final_value
            continue

        if category == "str":
            primary_kwargs[field_name] = _merge_and_join(raw_values, field_name, "primary")
        else:
            merged = _merge_values(raw_values)
            if len(raw_values) > 1:
                logger.debug(
                    "_route_row_values: primary field %r received %d value(s), "
                    "%d survived merge -> %r",
                    field_name,
                    len(raw_values),
                    len(merged),
                    merged,
                )
            primary_kwargs[field_name] = merged

    for field_name, raw_values in supplementary_raw.items():
        supplementary_kwargs[field_name] = _merge_and_join(raw_values, field_name, "supplementary")

    for target, field_map in comp_raw.items():
        comp_kwargs[target] = {}
        for field_name, raw_values in field_map.items():
            comp_kwargs[target][field_name] = _merge_and_join(
                raw_values, field_name, f"complementary {target!r}"
            )

    return primary_kwargs, supplementary_kwargs, comp_kwargs


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

    primary_kwargs, supplementary_kwargs, comp_kwargs = _route_row_values(
        row_dict, mapping, primary_cls
    )

    # ── Instantiate primary model ──────────────────────────────────────────────
    try:
        primary_instance = primary_cls.model_construct(**primary_kwargs)
    except Exception:
        logger.warning(
            "tabular: could not instantiate '%s' from row data (kwargs: %s)",
            primary_cls.__name__,
            list(primary_kwargs.keys()),
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
            logger.debug("tabular: supplementary field '%s' not in composite schema, skipping", k)

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


def _build_normalization_dedup_map(
    headers: list[str],
    all_rows: list[list[str]],
    primary_cls: type,
    comp_cls_map: dict[str, type],
    comp_class_names: list[str],
    mapping: ColumnMapping,
    get_specs_fn: callable | None = None,
) -> dict[str, NormalizationEntry]:
    """Pre-scan all rows to build a global dedup map of unique normalization keys.

    For each row, extracts source values for all normalizable fields (primary +
    complementary), computes a unique key, and tracks which row indices have
    that key. No LLM calls, no composite instantiation.

    Returns {unique_key: NormalizationEntry} where each entry's row_indices
    contains all row indices that share that key.
    """
    from scinr.newton.tabular.normalization.engine import NormalizationEngine

    fn = get_specs_fn or get_normalization_specs

    dedup_map: dict[str, NormalizationEntry] = {}

    # Collect specs for primary and all complementary models
    primary_specs = fn(primary_cls)

    for row_index, row_values in enumerate(all_rows):
        row_dict = _build_row_dict(headers, row_values)
        primary_kwargs, _supp_kwargs, comp_kwargs = _route_row_values(
            row_dict, mapping, primary_cls
        )

        # Primary model specs
        for spec in primary_specs:
            source_values = extract_source_values_from_dict(spec, primary_kwargs, primary_cls)
            if not source_values:
                continue
            unique_key = (
                f"{spec.target_type.__name__}:"
                f"{NormalizationEngine._hash_source_values(source_values)}"
            )
            if unique_key not in dedup_map:
                dedup_map[unique_key] = NormalizationEntry(
                    instance_id=0,
                    model_class_name=primary_cls.__name__,
                    field_name=spec.field_name,
                    target_type=spec.target_type,
                    source_values=source_values,
                    unique_key=unique_key,
                    row_indices=[],
                )
            dedup_map[unique_key].row_indices.append(row_index)

        # Complementary model specs
        for class_name in comp_class_names:
            comp_cls = comp_cls_map.get(class_name)
            if comp_cls is None:
                continue
            comp_model_kwargs = comp_kwargs.get(class_name, {})
            for spec in fn(comp_cls):
                source_values = extract_source_values_from_dict(spec, comp_model_kwargs, comp_cls)
                if not source_values:
                    continue
                unique_key = (
                    f"{spec.target_type.__name__}:"
                    f"{NormalizationEngine._hash_source_values(source_values)}"
                )
                if unique_key not in dedup_map:
                    dedup_map[unique_key] = NormalizationEntry(
                        instance_id=0,
                        model_class_name=comp_cls.__name__,
                        field_name=spec.field_name,
                        target_type=spec.target_type,
                        source_values=source_values,
                        unique_key=unique_key,
                        row_indices=[],
                    )
                dedup_map[unique_key].row_indices.append(row_index)

    # Validate comp_kwargs coverage: warn if any comp_class_names have no rows
    # with mapped values (possible mismatch between col_mapping.target_model
    # and decision.complementary_models.model_class)
    for class_name in comp_class_names:
        comp_cls = comp_cls_map.get(class_name)
        if comp_cls is None:
            continue
        if not instance_has_normalizable_fields(comp_cls):
            continue
        # Check if any entry in dedup_map references this class
        has_entries = any(e.model_class_name == class_name for e in dedup_map.values())
        if not has_entries:
            logger.warning(
                "tabular: complementary model '%s' has normalizable fields "
                "but no rows were mapped to it (possible target_model mismatch "
                "between column mapping and decision)",
                class_name,
            )

    return dedup_map


def _instantiate_composite_plain(
    table_composite_id: str,
    row_index: int,
    headers: list[str],
    row_values: list[str],
    mapping: ColumnMapping,
    decision: AnnotationDecision,
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    comp_cls_map: dict[str, type],
) -> tuple[int, list[str], BaseModel | None, str]:
    """Instantiate a composite model WITHOUT normalization.

    Used for remaining rows where normalization failed or is not applicable.
    """
    row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
    row_dict = _build_row_dict(headers, row_values)
    extraction_uid = make_uid(
        "tabular_extraction",
        row_composite_id,
        decision.matched_model_class or "raw",
    )

    composite_instance = _instantiate_composite_from_row(
        primary_cls=primary_cls,
        composite_cls=composite_cls,
        primary_field_name=primary_field_name,
        mapping=mapping,
        row_dict=row_dict,
        comp_cls_map=comp_cls_map,
    )

    return (row_index, row_values, composite_instance, extraction_uid)


def _instantiate_composite_with_normalization(
    table_composite_id: str,
    row_index: int,
    headers: list[str],
    row_values: list[str],
    mapping: ColumnMapping,
    decision: AnnotationDecision,
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    comp_cls_map: dict[str, type],
    comp_class_names: list[str],
    engine: NormalizationEngine,
    get_specs_fn: callable | None = None,
) -> tuple[int, list[str], BaseModel | None, str]:
    """Instantiate a composite model with cached normalization applied.

    Builds the composite from raw row values, then applies cached normalization
    results from the engine for all normalizable fields.

    Uses dict-based hash (same as pre-scan) to ensure cache lookup consistency.
    """
    fn = get_specs_fn or get_normalization_specs

    row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
    row_dict = _build_row_dict(headers, row_values)
    extraction_uid = make_uid(
        "tabular_extraction",
        row_composite_id,
        decision.matched_model_class or "raw",
    )

    # Route values FIRST (needed for dict-based hash, same as pre-scan)
    primary_kwargs, _supp_kwargs, comp_kwargs = _route_row_values(row_dict, mapping, primary_cls)

    composite_instance = _instantiate_composite_from_row(
        primary_cls=primary_cls,
        composite_cls=composite_cls,
        primary_field_name=primary_field_name,
        mapping=mapping,
        row_dict=row_dict,
        comp_cls_map=comp_cls_map,
    )

    if composite_instance is None:
        return (row_index, row_values, None, extraction_uid)

    # Apply cached normalization to primary instance — dict-based hash
    primary_instance = getattr(composite_instance, primary_field_name, None)
    if primary_instance is not None:
        for spec in fn(primary_cls):
            source_values = extract_source_values_from_dict(spec, primary_kwargs, primary_cls)
            if not source_values:
                continue
            unique_key = (
                f"{spec.target_type.__name__}:"
                f"{NormalizationEngine._hash_source_values(source_values)}"
            )
            applied = engine.apply_cached_to_instance(
                primary_instance, spec.field_name, unique_key
            )
            if not applied:
                logger.warning(
                    "tabular: cache miss for row %d, field '%s', key '%s' "
                    "(normalization LLM may have failed)",
                    row_index, spec.field_name, unique_key,
                )

    # Apply cached normalization to complementary instances — dict-based hash
    for class_name in comp_class_names:
        comp_cls = comp_cls_map.get(class_name)
        if comp_cls is None:
            continue
        comp_model_kwargs = comp_kwargs.get(class_name, {})
        snake_name = _to_snake_case(class_name)
        comp_instance = getattr(composite_instance, snake_name, None)
        if comp_instance is None:
            continue
        for spec in fn(comp_cls):
            source_values = extract_source_values_from_dict(spec, comp_model_kwargs, comp_cls)
            if not source_values:
                continue
            unique_key = (
                f"{spec.target_type.__name__}:"
                f"{NormalizationEngine._hash_source_values(source_values)}"
            )
            applied = engine.apply_cached_to_instance(
                comp_instance, spec.field_name, unique_key
            )
            if not applied:
                logger.warning(
                    "tabular: cache miss for row %d, comp '%s' field '%s', key '%s'",
                    row_index, class_name, spec.field_name, unique_key,
                )

    return (row_index, row_values, composite_instance, extraction_uid)


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
                col_name,
                row_composite_id,
                exc,
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

    logger.info("delete_tabular_subgraph: deleted table at %s", table_composite_id)
