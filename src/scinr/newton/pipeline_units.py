"""
pipeline_units.py — Document unit discovery for the future per-document
orchestration engine (does not exist yet).

Enumerates "document units" to be processed later, WITHOUT executing any
real conversion / extraction / ingestion. Only lightweight I/O is performed:
listing files and reading small metadata fields (e.g. the `folder_path`
field of an intermediate JSON file, or the `document_name` field of an
`extract-*.json` file).

This module has no dependency on any orchestration engine — it is a pure,
self-contained discovery layer, testable in isolation. It is consumed later
by code that does not exist yet; do not add imports pointing "forward" to
that future code here.

Source-of-truth for each discovery rule (keep these in sync; do not let the
logic below drift from the real stage without updating both)::

    raw_file        -> scinr.newton.converters.main.convert_one() /
                        convert_folder(): folder_path_str derived from
                        _relative_prefix (relative to input_dir).
    extraction_json -> scinr.newton.stages.extraction.run_extraction()
                        ._process_file(): doc_path =
                        f"{folder_path}/{nombre}" if folder_path else nombre,
                        where folder_path is read from the JSON's own
                        "folder_path" field (NOT from directory structure).
    ingestion_json   -> scinr.newton.ingest.loader._read_doc_path() /
                        _FILE_GLOB — reused directly, not duplicated.
    pre_ingested     -> scinr.newton.utils.document_resolver
                        .resolve_leaf_document_names() — reused directly,
                        a sync Neo4j query run via asyncio.to_thread().

Known design discrepancy (document_names vs document_names_dir)
-----------------------------------------------------------------
The `document_names` branch below eagerly resolves every name to its leaf
documents via Neo4j (`resolve_leaf_document_names`), matching the explicit
spec for this module. The `document_names_dir` branch, by contrast, does
**not** touch Neo4j at all: it merely replicates the existing
`scinr.newton.pipeline.run_pipeline()` block (~lines 502-518) that reads the
raw `document_name` field out of each `extract-*.json` file. This mirrors
current production behavior faithfully (that block never calls
`resolve_leaf_document_names` either), but it means the two branches are
*not* symmetric: `document_names` units carry Neo4j-confirmed leaf names,
while `document_names_dir` units carry unresolved names that may still be
folder-level (leaf resolution for those currently only happens later, one
name at a time, inside `run_annotation()` / `run_entity_extraction()` via
`scinr.newton.annotation.agent` / `scinr.newton.entity_extraction.agent`).
Flagged here for whoever designs the future orchestration engine: either
also resolve `document_names_dir` eagerly for consistency, or defer
resolution for both branches uniformly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scinr.newton.results import DocumentResult

logger = logging.getLogger(__name__)


@dataclass
class DocumentUnit:
    """A single document-shaped unit of work discovered ahead of processing.

    No real conversion/extraction/ingestion has happened yet for units of
    kind ``"raw_file"``, ``"extraction_json"``, or ``"ingestion_json"`` —
    they only describe *where* the eventual document comes from and what its
    ``doc_path`` will resolve to once the real stage processes it. Units of
    kind ``"pre_ingested"`` describe a document name taken as-is (for
    ``document_names``, Neo4j-confirmed leaf names; for
    ``document_names_dir``, unresolved names read from disk — see module
    docstring "Known design discrepancy" note).

    Attributes:
        kind: One of ``"raw_file"``, ``"extraction_json"``, ``"ingestion_json"``,
            ``"pre_ingested"``.
        source_path: The file this unit was discovered from. ``None`` for
            ``"pre_ingested"`` units (no filesystem source).
        doc_path: For ``raw_file`` / ``extraction_json`` / ``ingestion_json``: the
            document path derived exactly as the real stage would derive it
            (see module docstring). For ``pre_ingested``: the document name
            itself (leaf name, Neo4j-confirmed for ``document_names``;
            unresolved for ``document_names_dir``).
        relative_dir: Directory of *source_path* relative to the input root. ``Path(".")``
            for ``pre_ingested`` units (no filesystem source).
        document_name_hint: Human-readable document name (typically a filename stem, without any
            directory prefix).
    """

    kind: Literal["raw_file", "extraction_json", "ingestion_json", "pre_ingested"]
    source_path: Path | None
    doc_path: str
    relative_dir: Path
    document_name_hint: str


@dataclass
class UnitResult:
    """Result of processing one ``DocumentUnit`` end-to-end through whichever
    stages applied to it.

    Produced by the future ``_process_document_unit()`` (Paso 2,
    ``pipeline.py``, does not exist yet).

    Attributes:
        unit_id: Resolved ``doc_path`` / final document name for this unit.
        stage_results: Only the stages actually reached by this unit (keyed by stage name,
            e.g. ``"preprocess"``, ``"extraction"``, ``"ingestion"``,
            ``"annotation"``, ``"entity_extraction"``).
        stopped_at: Name of the stage where processing stopped due to failure, or
            ``None`` if the unit completed every stage it was asked to run.
        fatal_error: Exception message not attributable to any single concrete stage
            (defense-in-depth catch-all), or ``None`` if no such error occurred.
    """

    unit_id: str
    stage_results: dict[str, DocumentResult]
    stopped_at: str | None
    fatal_error: str | None


# ---------------------------------------------------------------------------
# Mutual-exclusion validation
# ---------------------------------------------------------------------------


def _validate_exclusive(**kwargs: object) -> str:
    """Ensure exactly one of the given keyword values is not ``None``.

    Defensive re-validation: the mutual-exclusion checks that
    ``run_pipeline()`` performs upstream may not exist yet (or may not cover
    every combination), so this module must not assume it has already been
    validated by a caller.

    Returns:
        The name of the single provided (non-``None``) keyword argument.

    Raises:
        ValueError: If zero, or more than one, of the values is not ``None``.
    """
    provided = [name for name, value in kwargs.items() if value is not None]
    if len(provided) != 1:
        raise ValueError(
            "Exactly one of "
            f"{sorted(kwargs)!r} must be provided (non-None) to _discover_units(); "
            f"got {len(provided)} provided: {provided!r}."
        )
    return provided[0]


# ---------------------------------------------------------------------------
# Per-kind discovery helpers
# ---------------------------------------------------------------------------


def _discover_raw_file_units(
    input_raw: str,
    tabular_extensions: set[str] | None,
) -> list[DocumentUnit]:
    """Enumerate raw source files under *input_raw*.

    Mirrors the file discovery + folder_path derivation of
    ``scinr.newton.converters.main.convert_one()`` /
    ``convert_folder()`` (see module docstring for the exact source). Does
    **not** convert anything — only lists files, skips unsupported formats
    (and, if requested, tabular extensions), and predicts the ``doc_path``
    each file will get once actually converted and extracted (per
    ``scinr.newton.stages.extraction.run_extraction()``'s
    ``folder_path``/``doc_path`` combination rule).

    Args:
        input_raw: Root folder to walk recursively.
        tabular_extensions: File extensions (e.g. ``{".csv", ".xlsx"}``) to exclude from the
            result. When ``None``, no extension is excluded on this basis alone
            (unsupported-format files are still skipped via the converter
            registry, exactly as ``convert_one()`` does).
    """
    from scinr.newton.converters.base import UnsupportedFormatError
    from scinr.newton.converters.registry import get_converter

    root = Path(input_raw)
    if not root.exists():
        raise FileNotFoundError(f"input_raw folder not found: '{input_raw}'.")

    units: list[DocumentUnit] = []
    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue
        if tabular_extensions and entry.suffix.lower() in tabular_extensions:
            continue
        try:
            get_converter(entry)
        except UnsupportedFormatError:
            logger.warning("Skipping %s: unsupported format", entry.name)
            continue

        # See convert_one(): folder_path_str = str(_relative_prefix) if
        # _relative_prefix else None, where _relative_prefix accumulates the
        # entry's parent directory chain relative to the original input_dir.
        relative_dir = entry.parent.relative_to(root)
        folder_path_str = str(relative_dir) if relative_dir != Path(".") else None
        # See stages/extraction.py::_process_file(): doc_path is folder_path
        # joined with the document's name once it reaches the extraction
        # stage — predicted here ahead of time for a not-yet-converted file.
        doc_path = f"{folder_path_str}/{entry.stem}" if folder_path_str else entry.stem

        units.append(
            DocumentUnit(
                kind="raw_file",
                source_path=entry,
                doc_path=doc_path,
                relative_dir=relative_dir,
                document_name_hint=entry.stem,
            )
        )
    return units


def _discover_extraction_json_units(extraction_input_dir: str) -> list[DocumentUnit]:
    """Enumerate intermediate JSON files under *extraction_input_dir*.

    Mirrors ``scinr.newton.stages.extraction.run_extraction()
    ._process_file()``: files are discovered via
    ``sorted(input_path.rglob("*.json"))``, and ``doc_path`` is built from
    each file's own ``"folder_path"`` field (read from the JSON content
    itself, NOT derived from the file's actual directory location on disk).
    """
    input_path = Path(extraction_input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: '{extraction_input_dir}'.")

    units: list[DocumentUnit] = []
    for json_file in sorted(input_path.rglob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read '%s': %s", json_file, exc)
            continue

        folder_path: str | None = raw.get("folder_path")
        # Matches _process_file(): name, ext = os.path.splitext(json_file);
        # nombre = Path(name).name  (== json_file.stem, spelled out here to
        # mirror the original code exactly).
        name, _ext = os.path.splitext(json_file)
        nombre = Path(name).name
        doc_path = f"{folder_path}/{nombre}" if folder_path else nombre

        try:
            relative_dir = json_file.relative_to(input_path).parent
        except ValueError:
            relative_dir = Path(".")

        units.append(
            DocumentUnit(
                kind="extraction_json",
                source_path=json_file,
                doc_path=doc_path,
                relative_dir=relative_dir,
                document_name_hint=nombre,
            )
        )
    return units


def _discover_ingestion_json_units(ingestion_input_dir: str) -> list[DocumentUnit]:
    """Enumerate ``extract-*.json`` files under *ingestion_input_dir*.

    Mirrors ``scinr.newton.ingest.loader.load_folder()``'s glob pattern
    (``_FILE_GLOB = "extract-*.json"``) and reuses
    ``ingest.loader._read_doc_path()`` directly for the ``doc_path``
    derivation, rather than duplicating its fallback logic.
    """
    from scinr.newton.ingest.loader import _FILE_GLOB, _read_doc_path

    input_path = Path(ingestion_input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Folder not found: '{ingestion_input_dir}'.")

    units: list[DocumentUnit] = []
    for json_file in sorted(input_path.rglob(_FILE_GLOB)):
        doc_path = _read_doc_path(json_file)
        if not doc_path:
            logger.warning("Could not derive doc_path for '%s' — skipping.", json_file)
            continue

        try:
            relative_dir = json_file.relative_to(input_path).parent
        except ValueError:
            relative_dir = Path(".")

        document_name_hint = json_file.stem.removeprefix("extract-")

        units.append(
            DocumentUnit(
                kind="ingestion_json",
                source_path=json_file,
                doc_path=doc_path,
                relative_dir=relative_dir,
                document_name_hint=document_name_hint,
            )
        )
    return units


async def _discover_pre_ingested_units(document_names: list[str]) -> list[DocumentUnit]:
    """Resolve each name in *document_names* to its leaf documents in Neo4j.

    Uses ``scinr.newton.utils.document_resolver.resolve_leaf_document_names()``
    (a sync Neo4j query) run off the event loop via ``asyncio.to_thread()``.
    A single sync ``Driver`` is opened for the whole call and closed at the
    end (never left open). Leaf names are deduplicated (exact case, no
    normalization) preserving first-occurrence order across all input names.
    """
    from scinr.newton.ingest.config import get_driver
    from scinr.newton.utils.document_resolver import resolve_leaf_document_names

    driver = get_driver()
    try:
        seen: set[str] = set()
        units: list[DocumentUnit] = []
        for name in document_names:
            leaves = await asyncio.to_thread(resolve_leaf_document_names, driver, name)
            for leaf in leaves:
                if leaf in seen:
                    continue
                seen.add(leaf)
                units.append(
                    DocumentUnit(
                        kind="pre_ingested",
                        source_path=None,
                        doc_path=leaf,
                        relative_dir=Path("."),
                        document_name_hint=leaf,
                    )
                )
        return units
    finally:
        driver.close()


def _discover_document_names_dir_units(document_names_dir: str) -> list[DocumentUnit]:
    """Read the ``document_name`` field of every ``extract-*.json`` file
    under *document_names_dir*.

    Mirrors exactly the branch in ``scinr.newton.pipeline.run_pipeline()``
    (~lines 502-518) that resolves ``doc_names_for_ann_ee`` from
    ``document_names_dir`` — same glob, same per-file error handling, same
    "no readable document_name fields" guard. Does **not** touch Neo4j (see
    module docstring "Known design discrepancy" note): unlike
    ``document_names``, names returned here are NOT resolved to leaves.
    """
    dir_path = Path(document_names_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Folder not found: '{document_names_dir}'.")

    extract_files = sorted(dir_path.rglob("extract-*.json"))
    seen: set[str] = set()
    units: list[DocumentUnit] = []
    for f in extract_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read document_name from '%s': %s", f, exc)
            continue
        name = data.get("document_name")
        if not name or name in seen:
            continue
        seen.add(name)
        units.append(
            DocumentUnit(
                kind="pre_ingested",
                source_path=None,
                doc_path=name,
                relative_dir=Path("."),
                document_name_hint=name,
            )
        )

    if not units:
        raise ValueError(
            f"document_names_dir='{document_names_dir}' contains no readable "
            "'document_name' fields in extract-*.json files."
        )
    return units


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def _discover_units(
    *,
    input_raw: str | None = None,
    extraction_input_dir: str | None = None,
    ingestion_input_dir: str | None = None,
    document_names: list[str] | None = None,
    document_names_dir: str | None = None,
    tabular_extensions: set[str] | None = None,
) -> list[DocumentUnit]:
    """Discover the list of ``DocumentUnit``s for exactly one input source.

    Exactly one of *input_raw*, *extraction_input_dir*, *ingestion_input_dir*,
    *document_names*, or *document_names_dir* must be provided (non-``None``);
    this is validated defensively here since the equivalent mutual-exclusion
    validation upstream in ``run_pipeline()`` may not exist yet (or may not
    cover this exact combination).

    This function never executes any real conversion, extraction, or
    ingestion — it only enumerates files and reads lightweight metadata
    (e.g. the ``folder_path`` field of an intermediate JSON file, or the
    ``document_name`` field of an ``extract-*.json`` file), or resolves leaf
    document names already confirmed in Neo4j.

    Args:
        input_raw: Folder of raw source files (PDF, DOCX, CSV, XLSX, …) to enumerate as
            ``"raw_file"`` units.
        extraction_input_dir: Folder of intermediate JSON files (Stage 0 output) to enumerate as
            ``"extraction_json"`` units.
        ingestion_input_dir: Folder of ``extract-*.json`` files (Stage 1 output) to enumerate as
            ``"ingestion_json"`` units.
        document_names: Explicit list of Neo4j document names to resolve to leaf documents
            (``"pre_ingested"`` units, Neo4j-confirmed).
        document_names_dir: Folder of ``extract-*.json`` files to read raw ``document_name``
            values from (``"pre_ingested"`` units, unresolved — see module
            docstring).
        tabular_extensions: Only used with *input_raw*: file extensions to exclude from the
            result (e.g. tabular files routed to a separate pipeline).

    Returns:
        The discovered units, in the order produced by each branch (see the
        corresponding ``_discover_*_units()`` helper's docstring).

    Raises:
        ValueError: If zero, or more than one, of the five mutually-exclusive parameters
            is provided, or (for ``document_names_dir``) if no readable
            ``document_name`` field is found in any file.
        FileNotFoundError: If the given directory parameter does not point to an existing
            directory.
    """
    which = _validate_exclusive(
        input_raw=input_raw,
        extraction_input_dir=extraction_input_dir,
        ingestion_input_dir=ingestion_input_dir,
        document_names=document_names,
        document_names_dir=document_names_dir,
    )

    if which == "input_raw":
        return _discover_raw_file_units(input_raw, tabular_extensions)  # type: ignore[arg-type]
    if which == "extraction_input_dir":
        return _discover_extraction_json_units(extraction_input_dir)  # type: ignore[arg-type]
    if which == "ingestion_input_dir":
        return _discover_ingestion_json_units(ingestion_input_dir)  # type: ignore[arg-type]
    if which == "document_names":
        return await _discover_pre_ingested_units(document_names)  # type: ignore[arg-type]
    return _discover_document_names_dir_units(document_names_dir)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Batch-versioning helper
# ---------------------------------------------------------------------------


def build_all_paths_for_versioning(units: Sequence[DocumentUnit]) -> list[str]:
    """Flatten discovered *units* into the ``all_paths`` list (leaves +
    ancestor folders) expected by
    ``scinr.newton.ingest.loader.resolve_batch_version_sync()``.

    Delegates to ``ingest.loader._extract_all_paths()`` to avoid duplicating
    its ancestor-folder-expansion logic.

    Args:
        units: Discovered ``DocumentUnit``s (any kind). Units with an empty
            ``doc_path`` are ignored.
    """
    from scinr.newton.ingest.loader import _extract_all_paths

    leaf_paths = [u.doc_path for u in units if u.doc_path]
    return _extract_all_paths(leaf_paths)
