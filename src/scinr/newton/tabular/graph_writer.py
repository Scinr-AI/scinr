"""
tabular/graph_writer.py — domain side of the tabular subgraph: no Cypher.

This module holds the traversal of a sheet (:func:`write_tabular_subgraph`): ids,
batches of rows, the normalization passes, the composite instantiation and the
per-row extraction. It emits one :class:`~scinr.newton.graph.sinks.TabularSink`
call per statement, in the order the statements are executed; the Cypher is in
``graph/neo4j/tabular.py`` (``store.tabular.sink()``). The Cypher-free row logic
(routing, normalization keys, composite instantiation) is in
:mod:`scinr.newton.tabular.rows`.

Memory: at most one batch of ``_ROW_BATCH_SIZE`` rows is alive at a time (plus
O(unique normalization keys) on the normalization path); the sink never
accumulates rows. The row writes run with at most ``_PARALLEL_ROW_WRITES``
concurrent tasks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from scinr.newton.config import get_config
from scinr.newton.entity_extraction.graph_mapper import write_extraction_subgraph
from scinr.newton.entity_extraction.schema_composer import _to_snake_case
from scinr.newton.exceptions import ConfigurationError
from scinr.newton.graph import get_graph_store
from scinr.newton.graph.types import RowData, TableNode
from scinr.newton.ingest.nodes import make_structure_node_id
from scinr.newton.tabular.normalization.detector import (
    get_normalization_specs,
    instance_has_normalizable_fields,
)
from scinr.newton.tabular.normalization.models import NormalizationEntry
from scinr.newton.tabular.reader import row_to_markdown
from scinr.newton.tabular.rows import (
    _PARALLEL_ROW_WRITES,
    _ROW_BATCH_SIZE,
    RowBatchFactory,
    _build_normalization_dedup_map,
    _build_row_dict,
    _instantiate_composite_from_row,
    _instantiate_composite_plain,
    _instantiate_composite_with_normalization,
    _sanitize_rel_type,
    compute_row_normalization_keys,
)
from scinr.newton.utils.uid import make_uid

if TYPE_CHECKING:
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.graph.sinks import TabularSink
    from scinr.newton.tabular.models import ColumnMapping
    from scinr.newton.tabular.state import TabularFileData

logger = logging.getLogger(__name__)

_RAW_VALUE_MAX_CHARS = 500


async def write_tabular_subgraph(
    doc_path: str,
    document_name: str,
    resolved_version: int,
    sheet: TabularFileData,
    row_batches: RowBatchFactory,
    sheet_index: int,
    decision: AnnotationDecision,
    mapping: ColumnMapping,
    update_mode: bool = False,
    theme: str = "default",
    sheet_page_id: str = "",
    *,
    tenant_id: str,
) -> str:
    """Write the complete Table + Row subgraph for one sheet.

    Parameters
    ----------
    tenant_id : Stored tenant key of the owning :Document (``utils.tenancy``
        normalization already applied). Selects the :Document and prefixes the
        Table id, so every Row / InfoUnit / ExtractionResult id derived from it
        is tenant-scoped.
    row_batches : Factory of streaming passes over the sheet's data rows (see
        :data:`RowBatchFactory`). The rows are never held all at once: memory
        is O(one batch of ``_ROW_BATCH_SIZE`` rows), plus O(unique normalization
        keys) on the normalization path. The underlying file must not change
        while this runs.
    theme : The detected thematic domain path for this sheet (e.g.
        "pharmaceutical_quality"). Written to the Table node and all Row nodes
        as the ``theme`` property. Defaults to "default".
    sheet_page_id : Document-store page_id for this sheet, or "" when no storage backend.

    Steps:
    1. Compute composite IDs for Table and Rows.
    2. (update_mode) delete existing Table + Row subgraph.
    3. Create the Table StructureNode under the tenant's Document (``sink.create_table``).
    4. write_annotation() for the Table (creates ModelDecision + full subgraph).
    5. Compute ModelDecision UID (for linking Row nodes to same MD node).
    6. Resolve model class for extraction (if decision.matched_model_class is not None).
    7. Process rows in batches of _ROW_BATCH_SIZE:
       a. ``sink.write_row_batch``: create Row nodes + InfoUnits + HAS_MODEL_DECISION links.
       b. write_extraction_subgraph() per row (or the raw row extraction if no model).

    Returns table_composite_id.

    Raises:
        RuntimeError: If the owning Document does not exist.
        GraphError: If the engine fails.
    """
    from scinr.newton.entity_extraction.model_resolver import resolve_model_class
    from scinr.newton.entity_extraction.schema_composer import (
        compose_extraction_schema,
    )

    store = get_graph_store()
    table_node_id = f"table_{sheet_index + 1}"
    table_composite_id = make_structure_node_id(tenant_id, doc_path, resolved_version, table_node_id)
    headers = sheet["headers"]

    # Step 2: delete existing subgraph if update_mode
    if update_mode:
        await store.tabular.delete_tabular_subgraph(
            doc_path, resolved_version, sheet_index, tenant_id=tenant_id
        )

    async with store.tabular.sink() as sink:
        # Step 3: create Table StructureNode and link to Document.
        await sink.create_table(
            TableNode(
                id=table_composite_id,
                node_id=table_node_id,
                title=sheet["sheet_name"],
                order=sheet_index + 1,
                theme=theme,
                col_count=len(headers),
                row_count=sheet["total_rows"],
                doc_key=(tenant_id, doc_path, resolved_version),
                page_id=sheet_page_id,
            )
        )
        logger.info("write_tabular_subgraph: Table node created: %s", table_composite_id)

        # Step 4: write ModelDecision on the Table (full subgraph via write_annotation)
        await store.annotation.write_annotation(table_composite_id, decision, document_name)

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

        write_kwargs = dict(
            sink=sink,
            table_composite_id=table_composite_id,
            headers=headers,
            row_batches=row_batches,
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
        if has_normalization:
            logger.info(
                "tabular: using normalization-first write path for '%s' sheet '%s' (%d rows)",
                doc_path,
                sheet["sheet_name"],
                sheet["total_rows"],
            )
            total_rows = await _write_tabular_with_normalization(**write_kwargs)
        else:
            total_rows = await _write_rows_in_batches(**write_kwargs)

    logger.info(
        "write_tabular_subgraph: complete for '%s' sheet '%s' (%d rows)",
        doc_path,
        sheet["sheet_name"],
        total_rows,
    )
    return table_composite_id


# ── Batch row writer ──────────────────────────────────────────────────────────


async def _write_rows_in_batches(
    sink: TabularSink,
    table_composite_id: str,
    headers: list[str],
    row_batches: RowBatchFactory,
    decision_uid: str,
    decision: AnnotationDecision,
    mapping: ColumnMapping,
    primary_cls,
    composite_cls,
    primary_field_name: str | None,
    comp_class_names: list[str],
    comp_cls_map: dict[str, type],
    document_name: str,
    theme: str,
    sheet_page_id: str,
) -> int:
    """Standard write path: stream the rows in batches of ``_ROW_BATCH_SIZE``.

    Returns the number of rows written.
    """
    batch_start = 0
    for batch in row_batches(_ROW_BATCH_SIZE):
        await _write_row_batch(
            sink=sink,
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
            "tabular: wrote batch rows %d-%d",
            batch_start + 1,
            batch_start + len(batch),
        )
        batch_start += len(batch)
    return batch_start


async def _run_bounded(
    items: Iterable[Any],
    worker: Callable[[Any], Awaitable[None]],
    limit: int,
) -> None:
    """Run ``worker(item)`` for every item with at most *limit* tasks alive at once.

    Unlike ``asyncio.gather(*[create_task(...) for item in items])`` the tasks are
    created lazily, so the number of live task objects (and of coroutine frames
    holding their arguments) does not grow with the number of items. The first
    failure is raised and the tasks still running are cancelled.
    """

    def _raise_first_error(done: set[asyncio.Task]) -> None:
        # Retrieve every exception (avoids "never retrieved" warnings), raise the first.
        errors = [t.exception() for t in done if not t.cancelled() and t.exception() is not None]
        if errors:
            raise errors[0]

    pending: set[asyncio.Task] = set()
    try:
        for item in items:
            if len(pending) >= limit:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                _raise_first_error(done)
            pending.add(asyncio.create_task(worker(item)))
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            _raise_first_error(done)
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise


async def _write_tabular_with_normalization(
    sink: TabularSink,
    table_composite_id: str,
    headers: list[str],
    row_batches: RowBatchFactory,
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
) -> int:
    """Write all rows using normalization-first batching, streaming the rows.

    Three sequential passes over the sheet, none of which keeps more than one
    batch of rows (memory is O(batch + unique normalization keys)):

    B. Scan: for every row compute its normalization keys
       (:func:`compute_row_normalization_keys`) and fill the global dedup map
       ``unique_key -> NormalizationEntry`` (no row indices are kept).
    -. Normalize: the LLM normalizes the unique keys, grouped by target type and
       batched; the results land in ``engine.result_cache``.
    C. Write: for every batch, recompute each row's keys (same function as in B,
       so the cache lookups match), apply the cached results and write the batch.
       Rows with no key are instantiated plain.

    Returns the number of rows written.
    """
    from scinr.newton.tabular.normalization.engine import NormalizationEngine

    cfg = get_config()
    norm_llm = cfg.normalization_llm or cfg.llm
    if norm_llm is None:
        raise ConfigurationError(
            "Tabular normalization requires an LLM, but none is configured. "
            "Pass llm=... to configure() or set MODEL_ID."
        )

    standard_write_kwargs = dict(
        sink=sink,
        table_composite_id=table_composite_id,
        headers=headers,
        row_batches=row_batches,
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

    # Pass B: scan all rows -> dedup map
    logger.info("tabular: building normalization dedup map (streaming)")
    # Cache normalization specs by class (called O(rows) times otherwise)
    _specs_cache: dict[type, list] = {}

    def _get_specs_cached(cls: type) -> list:
        if cls not in _specs_cache:
            _specs_cache[cls] = get_normalization_specs(cls)
        return _specs_cache[cls]

    dedup_map, scanned_rows = _build_normalization_dedup_map(
        headers=headers,
        row_batches=row_batches(_ROW_BATCH_SIZE),
        primary_cls=primary_cls,
        comp_cls_map=comp_cls_map,
        comp_class_names=comp_class_names,
        mapping=mapping,
        get_specs_fn=_get_specs_cached,
    )
    logger.info(
        "tabular: dedup map has %d unique keys across %d rows",
        len(dedup_map),
        scanned_rows,
    )

    if not dedup_map:
        # No normalizable fields found — fall back to standard path
        logger.info("tabular: no normalizable fields, using standard write path")
        return await _write_rows_in_batches(**standard_write_kwargs)

    # Create engine (persists across all key batches)
    engine = NormalizationEngine(
        llm=norm_llm,
        batch_size=cfg.normalization_batch_size,
    )

    # Group unique keys by target_type (LLM needs homogeneous batches)
    keys_by_type: dict[str, list[NormalizationEntry]] = {}
    for entry in dedup_map.values():
        type_key = entry.target_type.__name__
        if type_key not in keys_by_type:
            keys_by_type[type_key] = []
        keys_by_type[type_key].append(entry)

    # Collect all key batches (lists of references to entries already in dedup_map)
    all_key_batches: list[tuple[list[NormalizationEntry], str]] = []
    for type_name, type_entries in keys_by_type.items():
        for key_batch_start in range(0, len(type_entries), cfg.normalization_batch_size):
            key_batch = type_entries[
                key_batch_start : key_batch_start + cfg.normalization_batch_size
            ]
            all_key_batches.append((key_batch, type_name))

    # LLM normalization of the unique keys. Each batch takes the global
    # get_llm_semaphore() (not a local one) so that normalization calls share the
    # same bounded concurrency pool as every other Bedrock caller in the pipeline
    # (extraction, entity extraction, annotation), avoiding overshooting the botocore
    # connection pool when multiple documents/tables are processed in parallel.
    # The tasks themselves are created lazily (at most `llm_concurrency` alive: more
    # would only sit blocked on the semaphore), instead of one task per key batch up front.
    from scinr.newton.config import get_llm_semaphore

    async def _extract_key_batch(item: tuple[list[NormalizationEntry], str]) -> None:
        key_batch, type_name = item
        async with get_llm_semaphore():
            logger.info(
                "tabular: normalizing batch of %d keys (type: %s)",
                len(key_batch),
                type_name,
            )
            results = await engine.process_key_batch(key_batch)
            logger.info("tabular: normalization batch returned %d results", len(results))

    await _run_bounded(all_key_batches, _extract_key_batch, max(1, cfg.llm_concurrency))

    # Pass C: stream the rows again, apply the cached normalization and write.
    # Each row's keys are recomputed with the SAME function used in pass B, so
    # the unique keys match those stored in engine.result_cache. Rows without
    # normalizable values are instantiated plain; a key without a cached result
    # (normalization LLM failed) leaves that field as-is (logged in
    # _instantiate_composite_with_normalization).
    rows_written = 0
    for batch in row_batches(_ROW_BATCH_SIZE):
        batch_start = rows_written
        composites = []
        for offset, row_values in enumerate(batch):
            row_index = batch_start + offset
            row_keys = compute_row_normalization_keys(
                headers=headers,
                row_values=row_values,
                mapping=mapping,
                primary_cls=primary_cls,
                comp_cls_map=comp_cls_map,
                comp_class_names=comp_class_names,
                get_specs_fn=_get_specs_cached,
            )
            if row_keys:
                _, _, composite_instance, extraction_uid = (
                    _instantiate_composite_with_normalization(
                        table_composite_id=table_composite_id,
                        row_index=row_index,
                        headers=headers,
                        row_values=row_values,
                        mapping=mapping,
                        decision=decision,
                        primary_cls=primary_cls,
                        composite_cls=composite_cls,
                        primary_field_name=primary_field_name,
                        comp_cls_map=comp_cls_map,
                        comp_class_names=comp_class_names,
                        engine=engine,
                        row_keys=row_keys,
                    )
                )
            else:
                _, _, composite_instance, extraction_uid = _instantiate_composite_plain(
                    table_composite_id=table_composite_id,
                    row_index=row_index,
                    headers=headers,
                    row_values=row_values,
                    mapping=mapping,
                    decision=decision,
                    primary_cls=primary_cls,
                    composite_cls=composite_cls,
                    primary_field_name=primary_field_name,
                    comp_cls_map=comp_cls_map,
                )
            # Offsets from the batch start: _write_row_batch adds batch_start_index back.
            composites.append((offset, row_values, composite_instance, extraction_uid))

        await _write_row_batch(
            sink=sink,
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
            composite_results=composites,
            row_indices=list(range(batch_start, batch_start + len(batch))),
        )
        rows_written += len(batch)
        logger.debug(
            "tabular: wrote rows %d-%d with normalization", batch_start + 1, rows_written
        )

    return rows_written


async def _write_row_batch(
    sink: TabularSink,
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
    """Write a batch of Row nodes + InfoUnits + HAS_MODEL_DECISION links through
    ``sink.write_row_batch``, then write the ExtractionResult per row.
    """
    # Prepare row data for the sink
    rows_data: list[RowData] = []
    indices = (
        row_indices
        if row_indices is not None
        # Standard path: indices follow from the batch start
        else range(batch_start_index, batch_start_index + len(rows_batch))
    )
    for idx, row_values in zip(indices, rows_batch):  # noqa: B905
        row_composite_id = f"{table_composite_id}/row_{idx + 1}"
        rows_data.append(
            RowData(
                id=row_composite_id,
                node_id=f"row_{idx + 1}",
                row_index=idx,
                appearance_order=idx + 1,
                info_uid=make_uid(row_composite_id, "row_data"),
                description=row_to_markdown(headers, row_values),
            )
        )

    # One transaction: create Row nodes + InfoUnits + HAS_MODEL_DECISION links
    await sink.write_row_batch(table_composite_id, rows_data, decision_uid, theme, sheet_page_id)

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
                if norm_llm is None:
                    raise ConfigurationError(
                        "Tabular normalization requires an LLM, but none is configured. "
                        "Pass llm=... to configure() or set MODEL_ID."
                    )
                engine = NormalizationEngine(
                    llm=norm_llm,
                    batch_size=cfg.normalization_batch_size,
                )
                try:
                    normalized = await engine.normalize_instances(normalization_instances)
                    logger.debug("tabular: normalized %d instances", len(normalized))
                except Exception as _exc:
                    logger.error("tabular: normalization batch failed: %s", _exc, exc_info=True)

    # ── Phase 3: Write to the graph (parallel) ────────────────────────────────
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
                        sink, row_composite_id, row_dict, document_name, extraction_uid
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


async def _write_raw_row_extraction(
    sink: TabularSink,
    row_composite_id: str,
    row_dict: dict[str, str],
    document_name: str,
    extraction_uid: str,
) -> None:
    """Write a raw ExtractionResult for rows where no model matched.

    Each column value is written as a per-tenant :Entity singleton (MERGEd by
    uid = hash of tenant_id + normalized_value), linked via a dynamic
    HAS_{COLUMN_NAME} relationship — mirroring the triple extraction pipeline
    so that tabular values can cross-reference with Entity nodes from other
    documents of the same tenant (never across tenants).

    model_class = 'GenericTabularRow', source = 'tabular_raw'.
    """
    timestamp = datetime.now(UTC).isoformat()

    # Filter empty/None values and truncate long strings
    valid_props = {
        k: (v[:_RAW_VALUE_MAX_CHARS] if isinstance(v, str) and len(v) > _RAW_VALUE_MAX_CHARS else v)
        for k, v in row_dict.items()
        if k and v
    }
    provenance = await sink.replace_raw_extraction_result(
        row_composite_id, extraction_uid, document_name, timestamp
    )
    if provenance is None:
        logger.warning(
            "_write_raw_row_extraction: Row StructureNode not found: %r", row_composite_id
        )
        return

    # One :Entity per column value (deduplicated within a tenant), linked via
    # dynamic HAS_{COL} rel
    for col_name, col_value in valid_props.items():
        rel_type = _sanitize_rel_type(col_name)
        normalized = col_value.strip().lower()
        # UID mirrors triple pipeline, with tenant folded in (see make_instance_uid)
        entity_uid = make_uid("entity", provenance.tenant_id, normalized)
        try:
            await sink.link_raw_column_value(
                extraction_uid, rel_type, entity_uid, col_value, normalized, provenance
            )
        except Exception as exc:
            logger.warning(
                "_write_raw_row_extraction: failed to write column '%s' for %s after retries: %s",
                col_name,
                row_composite_id,
                exc,
            )
