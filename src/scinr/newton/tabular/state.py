"""tabular/state.py — LangGraph state TypedDicts for the tabular pipeline.

NOTE: No 'from __future__ import annotations' here — LangGraph requires
runtime evaluation of TypedDict field annotations.
"""
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    pass


class TabularFileData(TypedDict):
    """Metadata of one sheet of a tabular file.

    Deliberately carries **no data rows**: the rows are streamed from
    ``file_path`` in batches when the sheet is written (see
    ``tabular.reader.iter_sheet_batches``), so the LangGraph state stays small
    and independent of the number of rows.
    """

    file_path: str               # source file the batches are re-read from
    sheet_name: str
    headers: list
    total_rows: int
    preview: dict                # TabularPreview at runtime
    preview_markdown: str        # GFM markdown of the preview, ready for LLM prompts


class TabularState(TypedDict):
    """LangGraph state for the tabular pipeline. One state per file (N sheets)."""

    # Input (set once before graph runs)
    file_path: str                  # absolute path to the source CSV/XLSX file
    document_name: str              # display name (file stem)
    doc_path: str                   # relative path for Neo4j Document key
    tenant_id: str                  # stored tenant key (utils.tenancy.tenant_key applied) — part of the Document key
    created_by_user_id: str | None  # provenance of the upload (stored raw file / pages); None when omitted
    job_id: str | None              # provenance of the upload (stored raw file / pages); None when omitted
    update_mode: bool               # run_tabular_pipeline(update_mode=...)
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
