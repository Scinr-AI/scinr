"""tabular/nodes.py — LangGraph nodes for the tabular ingestion pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from scinr.newton.annotation.models import AnnotationDecision
from scinr.newton.config import get_llm, make_system_message
from scinr.newton.tabular.models import ColumnFieldMapping, ColumnMapping
from scinr.newton.tabular.prompts import build_tabular_decision_prompt, build_tabular_mapping_prompt
from scinr.newton.tabular.reader import preview_to_markdown, read_tabular_file, select_preview_rows
from scinr.newton.tabular.state import TabularFileData, TabularState
from scinr.newton.utils.llm_repair import extract_raw_payload, run_repair_loop
from scinr.newton.utils.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)


# ── Node: load_sheets ─────────────────────────────────────────────────────────


async def load_sheets(state: TabularState) -> dict:
    """Read the tabular file and populate state.sheets with TabularFileData.

    Also ensures catalog models and theme structure exist in Neo4j (idempotent),
    mirroring the annotation pipeline's load_nodes() behaviour.
    """
    from scinr.newton.annotation.neo4j_ops import (
        ensure_catalog_models_once,
        ensure_theme_structure_once,
    )
    from scinr.newton.ingest.config import get_async_driver
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    driver = get_async_driver()           # singleton — NO cerrar
    await ensure_catalog_models_once(driver)
    await ensure_theme_structure_once(driver, theme_registry)

    file_path = Path(state["file_path"])
    logger.info("load_sheets: reading %s", file_path)
    try:
        raw_sheets = read_tabular_file(file_path)
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(f"load_sheets: could not read {file_path}: {exc}")
        return {"sheets": [], "current_sheet_index": 0, "errors": errors}

    sheets: list[TabularFileData] = []
    for sheet in raw_sheets:
        preview = select_preview_rows(sheet)
        sheets.append({
            "sheet_name": sheet["sheet_name"],
            "headers": sheet["headers"],
            "all_rows": sheet["all_rows"],
            "total_rows": sheet["total_rows"],
            "preview": preview,
            "preview_markdown": preview_to_markdown(preview),
        })

    logger.info(
        "load_sheets: found %d sheet(s) in %s", len(sheets), file_path.name
    )

    # Store each sheet as a converted_page in MongoDB (graceful fallback)
    sheet_page_ids: list[str] = []
    raw_file_id = state.get("raw_file_id", "")
    if raw_file_id:
        try:
            from scinr.newton.storage.factory import get_storage
            _, page_repo = get_storage()
            file_path_obj = Path(state["file_path"])
            # Derive folder_path from doc_path (part before the last /)
            doc_path: str = state.get("doc_path", "")
            folder_path = doc_path.rsplit("/", 1)[0] if "/" in doc_path else None
            for i, sheet_data in enumerate(sheets):
                try:
                    page_id = await page_repo.store_page(
                        raw_file_id=raw_file_id,
                        filename=f"{state['document_name']}_{sheet_data['sheet_name']}",
                        folder_path=folder_path,
                        page_index=i,
                        markdown="",
                    )
                    sheet_page_ids.append(page_id)
                    logger.debug(
                        "load_sheets: stored page for sheet '%s' → page_id=%s",
                        sheet_data["sheet_name"], page_id,
                    )
                except Exception as page_exc:
                    logger.warning(
                        "load_sheets: failed to store page for sheet '%s': %s",
                        sheet_data["sheet_name"], page_exc,
                    )
                    sheet_page_ids.append("")
        except Exception as storage_exc:
            logger.warning("load_sheets: storage unavailable, skipping page storage: %s", storage_exc)
            sheet_page_ids = [""] * len(sheets)
    else:
        sheet_page_ids = [""] * len(sheets)

    return {
        "sheets": sheets,
        "current_sheet_index": 0,
        "sheet_page_ids": sheet_page_ids,
        "ingested_table_node_ids": [],
        "errors": list(state.get("errors", [])),
    }


# ── Node: prepare_sheet ───────────────────────────────────────────────────────


async def prepare_sheet(state: TabularState) -> dict:
    """Load current_sheet from sheets[current_sheet_index] and reset per-sheet state."""
    idx = state["current_sheet_index"]
    sheet = state["sheets"][idx]
    logger.info(
        "prepare_sheet: sheet %d/%d — '%s' (%d rows, %d cols)",
        idx + 1,
        len(state["sheets"]),
        sheet["sheet_name"],
        sheet["total_rows"],
        len(sheet["headers"]),
    )
    return {
        "current_sheet": sheet,
        "current_decision": None,
        "current_mapping": None,
    }


# ── Node: classify_theme ──────────────────────────────────────────────────────


async def classify_theme(state: TabularState) -> dict:
    """
    Classify the thematic domain of the current sheet using headers + preview.

    LLM Call 0 (before decide_model): structured output → ThemeClassification.
    Input context: sheet name, column headers, preview markdown.
    On any failure: falls back to 'default' (never crashes the graph).

    Returns: {"current_theme": detected_theme_path}
    """
    from scinr.newton.annotation.models import ThemeClassification
    from scinr.newton.tabular.prompts import build_tabular_theme_prompt

    sheet = state["current_sheet"]
    if sheet is None or not sheet["headers"]:
        logger.warning("classify_theme: no sheet or headers, defaulting to 'default'")
        return {"current_theme": "default"}

    sheet_name = sheet["sheet_name"]

    system_prompt = build_tabular_theme_prompt(
        document_name=state["document_name"],
        sheet_name=sheet_name,
        headers=sheet["headers"],
        preview_markdown=sheet["preview_markdown"],
    )
    human_content = (
        f"Classify the thematic domain of sheet '{sheet_name}' "
        f"from file '{state['document_name']}'. "
        f"Base your decision on the column headers and the preview rows shown above."
    )

    _msgs = [
        make_system_message(system_prompt),
        HumanMessage(content=human_content),
    ]
    llm_structured = get_llm(temperature=0.0).with_structured_output(
        ThemeClassification, include_raw=True
    )
    result = await with_llm_retry(lambda: llm_structured.ainvoke(_msgs))

    parsed: ThemeClassification | None = result["parsed"]
    if parsed is not None:
        logger.info(
            "classify_theme: '%s/%s' → theme='%s' (justification: %s)",
            state["document_name"], sheet_name, parsed.theme,
            parsed.justification[:80] if parsed.justification else "",
        )
        return {"current_theme": parsed.theme}

    # Parse failed — enter repair loop
    current_raw = extract_raw_payload(result["raw"])
    current_error = str(result.get("parsing_error") or "Unknown parsing error")
    logger.warning(
        "classify_theme: structured output parse failed for '%s/%s', entering repair loop",
        state["document_name"], sheet_name,
    )
    repaired = await run_repair_loop(
        schema=ThemeClassification,
        initial_raw=current_raw,
        initial_error=current_error,
        context_label=f"{state['document_name']}/{sheet_name}/classify_theme",
    )
    if repaired is not None:
        logger.info(
            "classify_theme: repair successful for '%s/%s' → theme='%s'",
            state["document_name"], sheet_name, repaired.theme,
        )
        return {"current_theme": repaired.theme}

    logger.error(
        "classify_theme: all attempts exhausted for '%s/%s', defaulting to 'default'",
        state["document_name"], sheet_name,
    )
    errors = list(state.get("errors", []))
    errors.append(f"classify_theme failed for sheet '{sheet_name}'")
    return {"current_theme": "default", "errors": errors}


# ── Node: decide_model ────────────────────────────────────────────────────────


async def decide_model(state: TabularState) -> dict:
    """LLM Call 1: Determine which Pydantic model best describes one row of this sheet.

    System message: full tabular decision prompt with catalog block.
    Human message: table context (preview markdown + headers).
    Structured output: AnnotationDecision.
    """
    from scinr.newton.utils.theme_registry import get_theme_registry
    theme_registry = get_theme_registry()

    sheet = state["current_sheet"]
    if sheet is None:
        return {"current_decision": None}

    sheet_name = sheet["sheet_name"]
    preview_markdown = sheet["preview_markdown"]
    headers = sheet["headers"]

    if not headers:
        logger.warning(
            "decide_model: sheet '%s' has no headers, skipping", sheet_name
        )
        return {"current_decision": None}

    theme_node = theme_registry.find_best_theme(state.get("current_theme", "default"))
    logger.info(
        "decide_model: using theme='%s' for sheet '%s'",
        state.get("current_theme", "default"), sheet["sheet_name"],
    )
    decision_prompt = build_tabular_decision_prompt(
        theme_node, preview_markdown, headers
    )

    human_content = (
        f"<table_context>\n"
        f"File: {state['document_name']}\n"
        f"Sheet: {sheet_name}\n"
        f"Total rows: {sheet['total_rows']}\n\n"
        f"Preview:\n{preview_markdown}\n"
        f"</table_context>"
    )

    _msgs = [
        make_system_message(decision_prompt),
        HumanMessage(content=human_content),
    ]
    llm_structured = get_llm(temperature=0.0).with_structured_output(
        AnnotationDecision, include_raw=True
    )
    result = await with_llm_retry(lambda: llm_structured.ainvoke(_msgs))

    parsed: AnnotationDecision | None = result["parsed"]
    if parsed is not None:
        logger.info(
            "decide_model: sheet '%s' → class=%s, confidence=%s",
            sheet_name,
            parsed.matched_model_class,
            parsed.confidence,
        )
        return {"current_decision": parsed}

    # Repair loop
    current_raw = extract_raw_payload(result["raw"])
    current_error = str(result.get("parsing_error") or "Unknown parsing error")
    logger.warning(
        "decide_model: parse failed for sheet '%s', entering repair loop", sheet_name
    )
    repaired = await run_repair_loop(
        schema=AnnotationDecision,
        initial_raw=current_raw,
        initial_error=current_error,
        context_label=f"{state['document_name']}/{sheet_name}",
    )
    if repaired is not None:
        return {"current_decision": repaired}

    logger.error(
        "decide_model: all repair attempts exhausted for sheet '%s'", sheet_name
    )
    errors = list(state.get("errors", []))
    errors.append(f"decide_model failed for sheet '{sheet_name}'")
    return {"current_decision": None, "errors": errors}


# ── Node: map_columns ─────────────────────────────────────────────────────────


async def map_columns(state: TabularState) -> dict:
    """LLM Call 2: Map each column to a model field.

    If decision.matched_model_class is None, skip LLM and map all to '__extra__'.
    """
    sheet = state["current_sheet"]
    decision = state["current_decision"]

    if sheet is None:
        return {"current_mapping": None}

    headers = sheet["headers"]
    sheet_name = sheet["sheet_name"]

    # No model matched: produce synthetic mapping without LLM call
    if decision is None or decision.matched_model_class is None:
        logger.info(
            "map_columns: no model for sheet '%s', mapping all columns to __extra__",
            sheet_name,
        )
        mappings = [
            ColumnFieldMapping(
                column_name=h,
                model_field_name="__extra__",
                confidence="low",
                notes="No model matched",
            )
            for h in headers
        ]
        return {
            "current_mapping": ColumnMapping(
                mappings=mappings, unmapped_columns=headers
            )
        }

    # Build mapping prompt
    mapping_prompt = build_tabular_mapping_prompt(
        matched_model_class=decision.matched_model_class,
        preview_markdown=sheet["preview_markdown"],
        headers=headers,
        supplementary_fields=[sf.model_dump() for sf in decision.supplementary_fields]
        if decision.supplementary_fields else None,
        complementary_model_names=[cm.model_class for cm in decision.complementary_models]
        if decision.complementary_models else None,
    )
    human_content = (
        f"Map the columns of sheet '{sheet_name}' "
        f"(model: {decision.matched_model_class}) to the model fields. "
        f"Headers: {', '.join(repr(h) for h in headers)}"
    )

    _msgs = [
        make_system_message(mapping_prompt),
        HumanMessage(content=human_content),
    ]
    llm_structured = get_llm(temperature=0.0).with_structured_output(
        ColumnMapping, include_raw=True
    )
    result = await with_llm_retry(lambda: llm_structured.ainvoke(_msgs))

    parsed: ColumnMapping | None = result["parsed"]
    if parsed is not None:
        logger.info(
            "map_columns: sheet '%s' — %d columns mapped (%d extra)",
            sheet_name,
            len(parsed.mappings),
            len(parsed.unmapped_columns),
        )
        return {"current_mapping": parsed}

    # Repair loop
    current_raw = extract_raw_payload(result["raw"])
    current_error = str(result.get("parsing_error") or "Unknown parsing error")
    logger.warning(
        "map_columns: parse failed for sheet '%s', entering repair loop", sheet_name
    )
    repaired = await run_repair_loop(
        schema=ColumnMapping,
        initial_raw=current_raw,
        initial_error=current_error,
        context_label=f"{state['document_name']}/{sheet_name}",
    )
    if repaired is not None:
        return {"current_mapping": repaired}

    # Fallback: all columns to __extra__
    logger.error(
        "map_columns: all repair attempts failed for '%s', mapping all to __extra__",
        sheet_name,
    )
    mappings = [
        ColumnFieldMapping(
            column_name=h,
            model_field_name="__extra__",
            confidence="low",
            notes="Repair failed",
        )
        for h in headers
    ]
    return {
        "current_mapping": ColumnMapping(
            mappings=mappings, unmapped_columns=headers
        )
    }


# ── Node: write_tabular ───────────────────────────────────────────────────────


async def write_tabular(state: TabularState) -> dict:
    """Write Table + Row subgraph to Neo4j for the current sheet. Advances sheet index."""
    from scinr.newton.ingest.config import get_async_driver  # ← async driver singleton
    from scinr.newton.tabular.neo4j_ops import write_tabular_subgraph

    sheet = state["current_sheet"]
    decision = state["current_decision"]
    mapping = state["current_mapping"]
    idx = state["current_sheet_index"]

    new_index = idx + 1

    if sheet is None or decision is None or mapping is None:
        logger.warning(
            "write_tabular: missing state for sheet %d, skipping", idx
        )
        return {"current_sheet_index": new_index}

    driver = get_async_driver()                         # ← singleton, NO cerrar
    sheet_page_ids = state.get("sheet_page_ids", [])
    sheet_page_id = sheet_page_ids[idx] if idx < len(sheet_page_ids) else ""
    try:
        table_id = await write_tabular_subgraph(        # ← await
            driver=driver,
            doc_path=state["doc_path"],
            document_name=state["document_name"],
            resolved_version=state["resolved_version"],
            sheet=sheet,
            sheet_index=idx,
            decision=decision,
            mapping=mapping,
            update_mode=state.get("update_mode", False),
            theme=state.get("current_theme", "default"),
            sheet_page_id=sheet_page_id,
        )
        node_ids = list(state.get("ingested_table_node_ids", []))
        node_ids.append(table_id)
        return {"current_sheet_index": new_index, "ingested_table_node_ids": node_ids}
    except Exception as exc:
        logger.error(
            "write_tabular: failed for sheet %d '%s': %s",
            idx,
            sheet["sheet_name"],
            exc,
        )
        errors = list(state.get("errors", []))
        errors.append(f"write_tabular failed for sheet {idx}: {exc}")
        return {"current_sheet_index": new_index, "errors": errors}
    # ← NO finally driver.close() — es singleton


# ── Conditional edge router ───────────────────────────────────────────────────


def check_done_router(state: TabularState) -> str:
    """Returns 'prepare_sheet' if more sheets remain, 'end' otherwise."""
    if state["current_sheet_index"] < len(state["sheets"]):
        return "prepare_sheet"
    return "end"
