"""tabular/state.py — LangGraph state TypedDicts for the tabular pipeline.

NOTE: No 'from __future__ import annotations' here — LangGraph requires
runtime evaluation of TypedDict field annotations.
"""
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    pass


class TabularFileData(TypedDict):
    """All data read from one sheet of a tabular file."""

    sheet_name: str
    headers: list
    all_rows: list
    total_rows: int
    preview: dict                # TabularPreview at runtime
    preview_markdown: str        # GFM markdown of the preview, ready for LLM prompts


class TabularState(TypedDict):
    """LangGraph state for the tabular pipeline. One state per file (N sheets)."""

    # Input (set once before graph runs)
    file_path: str                  # absolute path to the source CSV/XLSX file
    document_name: str              # display name (file stem)
    doc_path: str                   # relative path for Neo4j Document key
    update_mode: bool               # mirrors --update flag
    resolved_version: int           # pre-computed batch version
    raw_file_id: str                # MongoDB ObjectId str, or "" when no storage backend is configured
    sheet_page_ids: list            # list[str] — one page_id per sheet (from storage/mongodb/pages.py), or [] when no storage backend

    # Sheet data (set by load_sheets, one entry per sheet)
    sheets: list                    # list[TabularFileData]
    current_sheet_index: int        # which sheet is currently being processed

    # Per-sheet transient state (reset each iteration)
    current_sheet: dict             # TabularFileData | None at runtime
    current_decision: object        # AnnotationDecision | None at runtime
    current_mapping: object         # ColumnMapping | None at runtime
    current_theme: str               # detected theme path, e.g. "pharmaceutical_quality"

    # Accumulation
    ingested_table_node_ids: list   # list[str] of composite IDs of Table nodes written
    errors: list                    # list[str]
