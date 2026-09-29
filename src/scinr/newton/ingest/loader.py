"""
ingest/loader.py — Loading JSON extractions into Neo4j.

Usage:
    from scinr.newton.ingest.config import get_driver
    from scinr.newton.ingest.loader import load_file, load_folder, load_documents

    driver = get_driver()
    load_file(Path("data/output/extract-my_doc.json"), driver)
    load_folder(Path("data/output/"), driver)
    load_documents([doc1, doc2], driver)  # in-memory mode

The synchronous ``load_*`` functions are the graph-writing primitives: they do
**not** verify that a document's ``raw_file_id`` belongs to its tenant (the
storage repositories are async). The public entry points — ``ingest_one``,
``ingest_one_from_path`` and ``stages.run_ingestion`` — run that check (``ingest.raw_file_check``) first; call those, or the check itself,
rather than the ``load_*`` functions on untrusted input.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from neo4j.exceptions import ConstraintError

from scinr.newton.config import get_config
from scinr.newton.exceptions import IngestionError
from scinr.newton.ingest.nodes import (
    get_current_latest_version,
    get_next_version,
    insert_document_graph,
)
from scinr.newton.ingest.schema import DOCUMENT_KEY_CONSTRAINT
from scinr.newton.models.document_structure import Document
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

_FILE_GLOB = "extract-*.json"


def _apply_metadata_overrides(
    doc: Document,
    tenant_id: str | None,
    created_by_user_id: str | None,
    job_id: str | None,
) -> None:
    """Overlay caller-supplied provenance metadata onto *doc* in place.

    Each field is overridden only when a non-None value is supplied; an
    omitted (None) argument leaves whatever the Document already carries
    (e.g. a value baked into an ``extract-*.json`` at extraction time)
    untouched. The resulting ``doc.tenant_id`` / ``doc.created_by_user_id`` /
    ``doc.job_id`` are then always written to Neo4j by
    ``insert_document_graph`` (null when still None). ``doc.tenant_id`` keeps
    its public-API form (``None`` = public); ``insert_document_graph``
    normalizes it to the stored key.
    """
    if tenant_id is not None:
        doc.tenant_id = tenant_id
    if created_by_user_id is not None:
        doc.created_by_user_id = created_by_user_id
    if job_id is not None:
        doc.job_id = job_id


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


def _batch_tenants(tenant_id: str | None, baked: Iterable[str | None]) -> list[str | None]:
    """Effective tenant(s) of a batch, in public-API form: the caller-supplied
    *tenant_id* when given (it overrides every document's own value — see
    ``_apply_metadata_overrides``), otherwise each document's *baked* value."""
    if tenant_id is not None:
        return [tenant_id]
    return list(dict.fromkeys(baked))


def _resolve_batch_version(
    session,
    all_paths: list[str],
    update_mode: bool,
    tenant_ids: Iterable[str | None],
) -> int:
    """Compute a single shared integer version for a batch of documents.

    Versions are numbered per tenant, so only the batch's own tenants'
    documents are considered — another tenant's documents at the same paths
    never bump (or, in update mode, select) this batch's version.

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
    tenant_ids:
        Tenant(s) the batch's documents will be written under, in public-API
        form (``None`` = public). Usually one; several only when documents
        carry different tenants baked into their ``extract-*.json``.
    """
    if not all_paths:
        return 1

    stored_tenants = sorted({tenant_key(t) for t in tenant_ids})
    if update_mode:
        result = session.run(
            "MATCH (d:Document {latest: true}) "
            "WHERE d.tenant_id IN $tenant_ids AND d.path IN $paths "
            "RETURN max(d.version) AS max_version",
            tenant_ids=stored_tenants,
            paths=all_paths,
        )
    else:
        result = session.run(
            "MATCH (d:Document) WHERE d.tenant_id IN $tenant_ids AND d.path IN $paths "
            "RETURN max(d.version) AS max_version",
            tenant_ids=stored_tenants,
            paths=all_paths,
        )

    record = result.single()
    max_version = record["max_version"] if record else None

    if update_mode:
        return max_version if max_version is not None else 1
    else:
        return (max_version + 1) if max_version is not None else 1


def resolve_batch_version_sync(
    driver,
    all_paths: list[str],
    update_mode: bool,
    tenant_ids: Iterable[str | None],
) -> int:
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
    tenant_ids:
        Tenant(s) the batch will be written under (public-API form, ``None``
        = public) — see ``_resolve_batch_version``.
    """
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        return _resolve_batch_version(session, all_paths, update_mode, tenant_ids)


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


def _read_tenant_id(path: Path) -> str | None:
    """Read *only* the tenant_id baked into an extract-*.json file (``None``
    when absent or unreadable — the file then fails later, on full validation)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("tenant_id")
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
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata. When not None, each
        overrides the corresponding field already present in the JSON before
        the :Document node is written. See ``_apply_metadata_overrides``.

    Returns
    -------
    str
        The ``document_name`` of the successfully ingested document.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    logger.info("Loading file: %s", path)
    doc = Document.model_validate_json(path.read_text(encoding="utf-8"))
    _apply_metadata_overrides(doc, tenant_id, created_by_user_id, job_id)
    doc_path = doc.doc_path if doc.doc_path else doc.document_name
    tenant = tenant_key(doc.tenant_id)

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
        cfg = get_config()
        with driver.session(database=cfg.neo4j_database) as session:
            if update_mode:
                resolved_version = get_current_latest_version(session, doc_path, tenant) or 1
            else:
                resolved_version = get_next_version(session, doc_path, tenant)

    logger.info(
        "Resolved version for '%s': %d (update_mode=%s)",
        doc_path,
        resolved_version,
        update_mode,
    )

    def _do_insert() -> None:
        cfg = get_config()
        with driver.session(database=cfg.neo4j_database) as session:
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
                        DOCUMENT_KEY_CONSTRAINT in str(exc).lower()
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
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata, applied to every document
        in the batch (see ``load_file``).

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
    tenants = _batch_tenants(tenant_id, (_read_tenant_id(f) for f in files))
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        shared_version = _resolve_batch_version(session, all_paths, update_mode, tenants)

    logger.info("Batch version resolved: %d for %d path(s)", shared_version, len(all_paths))

    errors: dict[Path, Exception] = {}
    doc_names: list[str] = []
    for path in files:
        try:
            doc_name = load_file(
                path,
                driver,
                update_mode=update_mode,
                shared_version=shared_version,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
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
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata. When not None, each
        overrides the corresponding field already on *doc* before the
        :Document node is written (see ``_apply_metadata_overrides``).

    Returns
    -------
    str
        The document_name of the successfully ingested document.
    """
    _apply_metadata_overrides(doc, tenant_id, created_by_user_id, job_id)
    doc_path = doc.doc_path if doc.doc_path else doc.document_name
    tenant = tenant_key(doc.tenant_id)

    logger.info(
        "Loading in-memory document '%s' (path=%s, %d root node(s))",
        doc.document_name,
        doc_path,
        len(doc.document_structure),
    )

    if shared_version is not None:
        resolved_version = shared_version
    else:
        cfg = get_config()
        with driver.session(database=cfg.neo4j_database) as session:
            if update_mode:
                resolved_version = get_current_latest_version(session, doc_path, tenant) or 1
            else:
                resolved_version = get_next_version(session, doc_path, tenant)

    logger.info(
        "Resolved version for '%s': %d (update_mode=%s)",
        doc_path,
        resolved_version,
        update_mode,
    )

    def _do_insert() -> None:
        cfg = get_config()
        with driver.session(database=cfg.neo4j_database) as session:
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
                        DOCUMENT_KEY_CONSTRAINT in str(exc).lower()
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
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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

    Raises
    ------
    IngestionError
        If ``doc.raw_file_id`` is set but is not a stored raw file of the
        document's effective tenant (see ``ingest.raw_file_check``). Nothing is
        written to Neo4j in that case.
    """
    from scinr.newton.config import get_neo4j_sync_semaphore
    from scinr.newton.ingest.raw_file_check import verify_document

    await verify_document(doc, tenant_id)

    semaphore = get_neo4j_sync_semaphore()
    async with semaphore:
        return await asyncio.to_thread(
            _load_document_object,
            doc,
            driver,
            update_mode,
            shared_version,
            tenant_id,
            created_by_user_id,
            job_id,
        )


async def ingest_one_from_path(
    path: Path,
    driver,
    update_mode: bool = False,
    shared_version: int | None = None,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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

    Raises
    ------
    IngestionError
        If the file's ``raw_file_id`` is set but is not a stored raw file of the
        document's effective tenant (see ``ingest.raw_file_check``). Nothing is
        written to Neo4j in that case.
    """
    from scinr.newton.config import get_neo4j_sync_semaphore
    from scinr.newton.ingest.raw_file_check import verify_json_file

    await verify_json_file(path, tenant_id)

    semaphore = get_neo4j_sync_semaphore()
    async with semaphore:
        return await asyncio.to_thread(
            load_file,
            path,
            driver,
            update_mode,
            shared_version,
            tenant_id,
            created_by_user_id,
            job_id,
        )


def load_documents(
    documents: list[Document],
    driver,
    update_mode: bool = False,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata, applied to every document
        in the batch (see ``load_file``).

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
    tenants = _batch_tenants(tenant_id, (doc.tenant_id for doc in documents))
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        shared_version = _resolve_batch_version(session, all_paths, update_mode, tenants)

    logger.info("Batch version resolved: %d for %d path(s)", shared_version, len(all_paths))

    errors: dict[str, Exception] = {}
    doc_names: list[str] = []
    for doc in documents:
        try:
            doc_name = _load_document_object(
                doc,
                driver,
                update_mode=update_mode,
                shared_version=shared_version,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
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
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
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
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata, applied to every document
        in the folder (see ``load_file``).

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
    tenants = _batch_tenants(tenant_id, (_read_tenant_id(f) for f in json_files))
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        shared_version = _resolve_batch_version(session, all_paths, update_mode, tenants)

    logger.info("Batch version resolved: %d for %d path(s)", shared_version, len(all_paths))

    errors: dict[Path, Exception] = {}
    doc_names: list[str] = []
    for json_file in json_files:
        try:
            doc_name = load_file(
                json_file,
                driver,
                update_mode=update_mode,
                shared_version=shared_version,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
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
