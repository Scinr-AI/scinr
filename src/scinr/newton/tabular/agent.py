"""tabular/agent.py — Public API for the tabular ingestion pipeline."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from neo4j import Driver

from scinr.newton.tabular.graph import tabular_graph
from scinr.newton.tabular.state import TabularState

logger = logging.getLogger(__name__)


async def run_tabular_agent(
    file_path: Path,
    document_name: str,
    doc_path: str,
    driver: Driver,
    resolved_version: int,
    update_mode: bool = False,
    raw_file_id: str = "",
) -> dict:
    """Run the full tabular pipeline for one CSV or XLSX file.

    1. Creates the Document node and folder hierarchy in Neo4j (one transaction).
    2. Runs the LangGraph: load_sheets → decide_model → map_columns → write_tabular
       (per sheet).
    3. Returns the final TabularState dict.

    Parameters
    ----------
    file_path : Path to the source file (absolute).
    document_name : Display name (file stem).
    doc_path : Relative path used as Neo4j Document key.
    driver : Open Neo4j driver (caller-owned).
    resolved_version : Pre-computed batch version.
    update_mode : If True, existing Table/Row subgraph is wiped before re-inserting.
    raw_file_id : MongoDB ObjectId string for the stored raw file, or "" when no storage backend is configured.
    """
    from scinr.newton.ingest.nodes import (
        handle_versioning,
        insert_document,
        insert_folder_document_hierarchy,
        link_leaf_to_folder,
    )

    # Step 1: create Document node + folder hierarchy
    with driver.session() as session:
        with session.begin_transaction() as tx:
            try:
                if "/" in doc_path:
                    folder_path = doc_path.rsplit("/", 1)[0]
                    insert_folder_document_hierarchy(tx, folder_path, resolved_version)
                insert_document(
                    tx,
                    document_name,
                    doc_path,
                    resolved_version,
                    raw_file_id,
                    is_folder=False,
                    context_instructions=None,
                )
                handle_versioning(tx, doc_path, resolved_version)
                link_leaf_to_folder(tx, doc_path, resolved_version)
                tx.commit()
                logger.info(
                    "tabular agent: Document node created for '%s' (v%d)",
                    doc_path,
                    resolved_version,
                )
            except Exception as exc:
                tx.rollback()
                logger.error(
                    "tabular agent: Document creation failed for '%s': %s",
                    doc_path,
                    exc,
                )
                raise

    # Step 2: run LangGraph
    initial_state: TabularState = {
        "file_path": str(file_path),
        "document_name": document_name,
        "doc_path": doc_path,
        "update_mode": update_mode,
        "resolved_version": resolved_version,
        "raw_file_id": raw_file_id,
        "sheets": [],
        "current_sheet_index": 0,
        "current_sheet": None,
        "current_decision": None,
        "current_mapping": None,
        "current_theme": "default",      # reset before classify_theme runs
        "ingested_table_node_ids": [],
        "errors": [],
    }

    final_state = await tabular_graph.ainvoke(initial_state)

    errors = final_state.get("errors", [])
    if errors:
        logger.warning(
            "tabular agent: completed with %d error(s): %s", len(errors), errors
        )
    else:
        logger.info(
            "tabular agent: '%s' complete — %d table(s) written",
            document_name,
            len(final_state.get("ingested_table_node_ids", [])),
        )

    return final_state


def run_tabular_agent_sync(
    file_path: Path,
    document_name: str,
    doc_path: str,
    driver: Driver,
    resolved_version: int,
    update_mode: bool = False,
    raw_file_id: str = "",
) -> dict:
    """Synchronous wrapper around run_tabular_agent."""
    return asyncio.run(
        run_tabular_agent(
            file_path, document_name, doc_path, driver, resolved_version, update_mode, raw_file_id
        )
    )

def decide_content_type(file_suffix: str):
    match file_suffix.lower():
        case ".csv":
            return "text/csv"
        case ".xlsx":
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        case ".xls":
            return "application/vnd.ms-excel"
        case ".xls":
            return "application/vnd.ms-excel"
        case _:
            return "text/plain"
