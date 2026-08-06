"""
ingest/loader.py — Entry point for loading JSON extractions into Neo4j.

Usage (CLI):
    python -m ingest.loader --file data/output/extract-my_doc.json
    python -m ingest.loader --folder data/output/
    python -m ingest.loader          # defaults to data/output/

    Add --update to overwrite the latest version instead of creating a new one.

Usage (library):
    from ingest.loader import load_file, load_folder, load_documents

    driver = get_driver()
    load_file(Path("data/output/extract-my_doc.json"), driver)
    load_folder(Path("data/output/"), driver)
    load_documents([doc1, doc2], driver)  # in-memory mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from neo4j.exceptions import ConstraintError

from scinr.newton.exceptions import IngestionError
from scinr.newton.ingest.config import get_driver
from scinr.newton.ingest.nodes import (
    get_current_latest_version,
    get_next_version,
    insert_document_graph,
)
from scinr.newton.ingest.schema import setup_schema
from scinr.newton.models.document_structure import Document
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync

logger = logging.getLogger(__name__)

_DEFAULT_FOLDER = Path("data/output")
_FILE_GLOB = "extract-*.json"


# ---------------------------------------------------------------------------
# Private version-resolution helpers
# ---------------------------------------------------------------------------


def _extract_all_paths(leaf_doc_paths: list[str]) -> list[str]:
    """Given a list of leaf doc_paths, return all paths including ancestor folders.

    Example:
        ["ModuloA/SubModulo/doc_a", "ModuloA/SubModulo/doc_b"]
        → ["ModuloA/SubModulo/doc_a", "ModuloA/SubModulo/doc_b",
           "ModuloA/SubModulo", "ModuloA"]
    """
    all_paths: set[str] = set()
    for path in leaf_doc_paths:
        all_paths.add(path)
        parts = path.split("/")
        for i in range(1, len(parts)):
            all_paths.add("/".join(parts[:i]))
    return list(all_paths)


def _resolve_batch_version(
    session,
    all_paths: list[str],
    update_mode: bool,
) -> int:
    """Compute a single shared integer version for a batch of documents.

    In normal mode:  max existing version across all paths + 1  (or 1 if none).
    In update mode:  max current latest version across all paths (or 1 if none).

    Parameters
    ----------
    session:
        An open Neo4j session.
    all_paths:
        All document paths in the batch (leaves + ancestor folders).
    update_mode:
        True → return the current latest version (no increment).
        False → return the next version (increment).
    """
    if not all_paths:
        return 1

    if update_mode:
        result = session.run(
            "MATCH (d:Document {latest: true}) WHERE d.path IN $paths "
            "RETURN max(d.version) AS max_version",
            paths=all_paths,
        )
    else:
        result = session.run(
            "MATCH (d:Document) WHERE d.path IN $paths RETURN max(d.version) AS max_version",
            paths=all_paths,
        )

    record = result.single()
    max_version = record["max_version"] if record else None

    if update_mode:
        return max_version if max_version is not None else 1
    else:
        return (max_version + 1) if max_version is not None else 1


def resolve_batch_version_sync(driver, all_paths: list[str], update_mode: bool) -> int:
    """Public wrapper around _resolve_batch_version(): opens its own read
    session and delegates to the existing private function, without
    duplicating any logic. Intended to be called via asyncio.to_thread()
    from the future per-document orchestration engine (does not exist yet).

    Parameters
    ----------
    driver:
        An open, authenticated Neo4j driver instance.
    all_paths:
        All document paths in the batch (leaves + ancestor folders) —
        see _extract_all_paths() for how this list is computed today.
    update_mode:
        True → return the current latest version (no increment).
        False → return the next version (increment).
    """
    with driver.session() as session:
        return _resolve_batch_version(session, all_paths, update_mode)


def _read_doc_path(path: Path) -> str | None:
    """Read *only* the doc_path field from an extract-*.json file without full validation."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        doc_path = raw.get("doc_path")
        if not doc_path:
            # Fall back to deriving from filename
            name = path.stem.removeprefix("extract-")
            folder_path = raw.get("folder_path")
            doc_path = f"{folder_path}/{name}" if folder_path else name
        return doc_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_file(
    path: Path,
    driver,
    update_mode: bool = False,
    shared_version: int | None = None,
) -> str:
    """Load a single extracted JSON file into Neo4j.

    Version resolution:
    - If *shared_version* is provided (batch context), use it directly.
    - Otherwise, resolve from Neo4j: next version (normal) or current latest (update).

    Parameters
    ----------
    path:
        Path to a JSON extraction file produced by the pipeline.
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert with the same version.
        If False (default), create a new version and link via HAS_NEWER_VERSION.
    shared_version:
        Pre-computed version for batch ingestion. When provided, skips
        the per-document version query.

    Returns
    -------
    str
        The ``document_name`` of the successfully ingested document.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    logger.info("Loading file: %s", path)
    doc = Document.model_validate_json(path.read_text(encoding="utf-8"))
    doc_path = doc.doc_path if doc.doc_path else doc.document_name

    logger.info(
        "Validated document '%s' (path=%s, %d root nodes)",
        doc.document_name,
        doc_path,
        len(doc.document_structure),
    )

    # Resolve version (use shared_version if provided by batch caller)
    if shared_version is not None:
        resolved_version = shared_version
    else:
        with driver.session() as session:
            if update_mode:
                resolved_version = get_current_latest_version(session, doc_path) or 1
            else:
                resolved_version = get_next_version(session, doc_path)

    logger.info(
        "Resolved version for '%s': %d (update_mode=%s)",
        doc_path,
        resolved_version,
        update_mode,
    )

    def _do_insert() -> None:
        with driver.session() as session:
            with session.begin_transaction() as tx:
                try:
                    insert_document_graph(tx, doc, resolved_version, update_mode=update_mode)
                    tx.commit()
                    logger.info(
                        "Transaction committed for document: %s (v%d)",
                        doc.document_name,
                        resolved_version,
                    )
                except ConstraintError as exc:
                    tx.rollback()
                    if (
                        "constraint_document_path_version" in str(exc).lower()
                        or "version" in str(exc).lower()
                    ):
                        raise IngestionError(
                            f"Version conflict: version {resolved_version} was already "
                            f"ingested by a concurrent process. Retry the ingestion."
                        ) from exc
                    raise
                except Exception:
                    tx.rollback()
                    logger.exception("Transaction rolled back for document: %s", doc.document_name)
                    raise

    with_neo4j_retry_sync(_do_insert)

    return doc.document_name


def load_files(
    files: list[Path],
    driver,
    update_mode: bool = False,
) -> list[str]:
    """Load a specific list of extracted JSON files into Neo4j.

    All files in the list share a single resolved version (batch versioning):
    the version is computed once from Neo4j before any writes, ensuring
    consistent versioning across all documents loaded together.

    Parameters
    ----------
    files:
        Explicit list of JSON extraction file paths to ingest.
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert without creating new versions.

    Returns
    -------
    list[str]
        The ``document_name`` values of every successfully ingested document.
    """
    if not files:
        logger.info("No files to ingest.")
        return []

    logger.info("Ingesting %d specific file(s) (update_mode=%s).", len(files), update_mode)

    # Collect all doc_paths (leaves + ancestor folders) for batch version resolution
    leaf_doc_paths = [p for p in (_read_doc_path(f) for f in files) if p]
    all_paths = _extract_all_paths(leaf_doc_paths)

    with driver.session() as session:
        shared_version = _resolve_batch_version(session, all_paths, update_mode)

    logger.info("Batch version resolved: %d for %d path(s)", shared_version, len(all_paths))

    errors: dict[Path, Exception] = {}
    doc_names: list[str] = []
    for path in files:
        try:
            doc_name = load_file(
                path, driver, update_mode=update_mode, shared_version=shared_version
            )
            doc_names.append(doc_name)
        except Exception as exc:
            logger.exception("Failed to load file: %s", path)
            errors[path] = exc

    success_count = len(files) - len(errors)
    logger.info(
        "Files ingestion complete. Success: %d / %d. Errors: %d.",
        success_count,
        len(files),
        len(errors),
    )
    if errors:
        logger.warning(
            "Files that failed:\n%s",
            "\n".join(f"  {p}: {e}" for p, e in errors.items()),
        )
    return doc_names


def _load_document_object(
    doc: Document,
    driver,
    update_mode: bool = False,
    shared_version: int | None = None,
) -> str:
    """Load a single in-memory Document object into Neo4j.

    Analogous to load_file() but receives a validated Document instance
    instead of a file path. Skips all disk I/O.

    Parameters
    ----------
    doc:
        A fully validated Document object (produced by Stage 1 / run_extraction()).
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert with the same version.
    shared_version:
        Pre-computed version for batch ingestion. When provided, skips
        the per-document version query.

    Returns
    -------
    str
        The document_name of the successfully ingested document.
    """
    doc_path = doc.doc_path if doc.doc_path else doc.document_name

    logger.info(
        "Loading in-memory document '%s' (path=%s, %d root node(s))",
        doc.document_name,
        doc_path,
        len(doc.document_structure),
    )

    if shared_version is not None:
        resolved_version = shared_version
    else:
        with driver.session() as session:
            if update_mode:
                resolved_version = get_current_latest_version(session, doc_path) or 1
            else:
                resolved_version = get_next_version(session, doc_path)

    logger.info(
        "Resolved version for '%s': %d (update_mode=%s)",
        doc_path,
        resolved_version,
        update_mode,
    )

    def _do_insert() -> None:
        with driver.session() as session:
            with session.begin_transaction() as tx:
                try:
                    insert_document_graph(tx, doc, resolved_version, update_mode=update_mode)
                    tx.commit()
                    logger.info(
                        "Transaction committed for document: %s (v%d)",
                        doc.document_name,
                        resolved_version,
                    )
                except ConstraintError as exc:
                    tx.rollback()
                    if (
                        "constraint_document_path_version" in str(exc).lower()
                        or "version" in str(exc).lower()
                    ):
                        raise IngestionError(
                            f"Version conflict: version {resolved_version} was already "
                            f"ingested by a concurrent process. Retry the ingestion."
                        ) from exc
                    raise
                except Exception:
                    tx.rollback()
                    logger.exception("Transaction rolled back for document: %s", doc.document_name)
                    raise

    with_neo4j_retry_sync(_do_insert)

    return doc.document_name


# ---------------------------------------------------------------------------
# Async per-document ingestion wrappers (Bloque B — orchestration engine
# does not exist yet, these are the building blocks it will call).
# ---------------------------------------------------------------------------
#
# Design decision: two explicit functions (ingest_one for an in-memory
# Document, ingest_one_from_path for a Path) rather than a single function
# that dispatches internally on `Document | Path`. This mirrors the existing
# convention in this module — load_file()/_load_document_object() and
# load_files()/load_documents() are already split by input type instead of
# accepting a union — so Bloque B call sites stay type-safe and unambiguous
# without needing an isinstance() check here.


async def ingest_one(
    doc: Document,
    driver,
    update_mode: bool = False,
    shared_version: int | None = None,
) -> str:
    """Async wrapper around _load_document_object() for the future
    per-document orchestration engine (Bloque B, does not exist yet).

    Acquires get_neo4j_sync_semaphore() in the event loop BEFORE dispatching
    to asyncio.to_thread() — never acquire an asyncio.Semaphore inside the
    worker thread it wraps (asyncio.Semaphore requires an active event loop
    and is not usable from a plain worker thread). The semaphore is released
    only after the synchronous work — including its own internal
    with_neo4j_retry_sync retries — has fully completed, not before.

    Parameters
    ----------
    doc:
        A fully validated in-memory Document object.
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert with the same version.
    shared_version:
        Pre-computed version for batch ingestion. When provided, skips
        the per-document version query.

    Returns
    -------
    str
        The document_name of the successfully ingested document.
    """
    from scinr.newton.config import get_neo4j_sync_semaphore

    semaphore = get_neo4j_sync_semaphore()
    async with semaphore:
        return await asyncio.to_thread(
            _load_document_object, doc, driver, update_mode, shared_version
        )


async def ingest_one_from_path(
    path: Path,
    driver,
    update_mode: bool = False,
    shared_version: int | None = None,
) -> str:
    """Async wrapper around load_file() for the future per-document
    orchestration engine (Bloque B, does not exist yet).

    Analogous to ingest_one() but for a not-yet-loaded JSON extraction file
    on disk, mirroring the existing load_file() vs _load_document_object()
    split in this module. Same semaphore-before-thread contract as
    ingest_one(): get_neo4j_sync_semaphore() is acquired in the event loop
    before dispatching to asyncio.to_thread(), and released only after the
    synchronous work (disk read, validation, insert, and its own
    with_neo4j_retry_sync retries) has fully completed.

    Parameters
    ----------
    path:
        Path to a JSON extraction file produced by the pipeline.
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert with the same version.
    shared_version:
        Pre-computed version for batch ingestion. When provided, skips
        the per-document version query.

    Returns
    -------
    str
        The document_name of the successfully ingested document.
    """
    from scinr.newton.config import get_neo4j_sync_semaphore

    semaphore = get_neo4j_sync_semaphore()
    async with semaphore:
        return await asyncio.to_thread(load_file, path, driver, update_mode, shared_version)


def load_documents(
    documents: list[Document],
    driver,
    update_mode: bool = False,
) -> list[str]:
    """Load a list of in-memory Document objects into Neo4j.

    Analogous to load_files() but operates entirely in memory — no disk I/O.
    All documents in the list share a single resolved version (batch versioning),
    ensuring consistent versioning across the entire batch.

    Parameters
    ----------
    documents:
        List of fully validated Document objects produced by Stage 1 (run_extraction()).
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert without creating new versions.

    Returns
    -------
    list[str]
        The document_name values of every successfully ingested document.
    """
    if not documents:
        logger.info("No in-memory documents to ingest.")
        return []

    logger.info("Ingesting %d in-memory document(s) (update_mode=%s).", len(documents), update_mode)

    # Collect all doc_paths (leaves + ancestor folders) for batch version resolution
    leaf_doc_paths = [doc.doc_path if doc.doc_path else doc.document_name for doc in documents]
    all_paths = _extract_all_paths(leaf_doc_paths)

    with driver.session() as session:
        shared_version = _resolve_batch_version(session, all_paths, update_mode)

    logger.info("Batch version resolved: %d for %d path(s)", shared_version, len(all_paths))

    errors: dict[str, Exception] = {}
    doc_names: list[str] = []
    for doc in documents:
        try:
            doc_name = _load_document_object(
                doc, driver, update_mode=update_mode, shared_version=shared_version
            )
            doc_names.append(doc_name)
        except Exception as exc:
            logger.exception("Failed to load in-memory document: %s", doc.document_name)
            errors[doc.document_name] = exc

    success_count = len(documents) - len(errors)
    logger.info(
        "In-memory ingestion complete. Success: %d / %d. Errors: %d.",
        success_count,
        len(documents),
        len(errors),
    )
    if errors:
        logger.warning(
            "Documents that failed:\n%s",
            "\n".join(f"  {name}: {e}" for name, e in errors.items()),
        )
    return doc_names


def load_folder(
    folder: Path,
    driver,
    update_mode: bool = False,
) -> list[str]:
    """Load all extracted JSON files in a folder (recursively) into Neo4j.

    All files in the folder share a single resolved version (batch versioning).

    Parameters
    ----------
    folder:
        Path to a directory containing JSON extraction files (searched recursively).
    driver:
        An open, authenticated Neo4j driver instance.
    update_mode:
        If True, wipe existing structure and re-insert without creating new versions.

    Returns
    -------
    list[str]
        The ``document_name`` values of every successfully ingested document.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    json_files = sorted(folder.rglob(_FILE_GLOB))
    if not json_files:
        logger.warning("No files matching '%s' found in '%s' (recursive).", _FILE_GLOB, folder)
        return []

    logger.info(
        "Found %d file(s) matching '%s' in '%s' (recursive, update_mode=%s).",
        len(json_files),
        _FILE_GLOB,
        folder,
        update_mode,
    )

    # Collect all doc_paths (leaves + ancestor folders) for batch version resolution
    leaf_doc_paths = [p for p in (_read_doc_path(f) for f in json_files) if p]
    all_paths = _extract_all_paths(leaf_doc_paths)

    with driver.session() as session:
        shared_version = _resolve_batch_version(session, all_paths, update_mode)

    logger.info("Batch version resolved: %d for %d path(s)", shared_version, len(all_paths))

    errors: dict[Path, Exception] = {}
    doc_names: list[str] = []
    for json_file in json_files:
        try:
            doc_name = load_file(
                json_file, driver, update_mode=update_mode, shared_version=shared_version
            )
            doc_names.append(doc_name)
        except Exception as exc:
            logger.exception("Failed to load file: %s", json_file)
            errors[json_file] = exc

    success_count = len(json_files) - len(errors)
    logger.info(
        "Folder ingestion complete. Success: %d / %d. Errors: %d.",
        success_count,
        len(json_files),
        len(errors),
    )
    if errors:
        logger.warning(
            "Files that failed:\n%s",
            "\n".join(f"  {p}: {e}" for p, e in errors.items()),
        )
    return doc_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for the scinr-ingest ingestion module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="ingest.loader",
        description="Ingest scinr JSON extraction files into Neo4j.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a single JSON extraction file to ingest.",
    )
    group.add_argument(
        "--folder",
        metavar="DIR",
        help=(
            f"Path to a folder of JSON extraction files to ingest (recursive). "
            f"Defaults to '{_DEFAULT_FOLDER}' when neither --file nor --folder is given."
        ),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help=(
            "Update mode: wipe existing structure of the latest version and re-insert. "
            "Does not create a new version. Use to fix errors in the last ingest."
        ),
    )

    args = parser.parse_args()

    driver = get_driver()
    try:
        setup_schema(driver)

        if args.file:
            load_file(Path(args.file), driver, update_mode=args.update)
        else:
            folder = Path(args.folder) if args.folder else _DEFAULT_FOLDER
            load_folder(folder, driver, update_mode=args.update)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
