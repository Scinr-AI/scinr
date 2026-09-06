"""
stages/ingestion.py — Stage 2: Load Documents into Neo4j, plus replacement helpers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from scinr.newton.config import get_config
from scinr.newton.ingest.config import get_driver
from scinr.newton.ingest.loader import load_documents, load_files, load_folder
from scinr.newton.ingest.schema import setup_schema
from scinr.newton.results import DocumentResult, StageResult

logger = logging.getLogger(__name__)


async def run_ingestion(
    output_folder: str | None = None,
    files: list[Path] | None = None,
    documents: list | None = None,
    update_mode: bool = False,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> StageResult:
    """Load extracted documents into Neo4j.

    Accepts documents either from disk (via *output_folder* or *files*) or
    directly as in-memory :class:`~models.document_structure.Document` objects
    (via *documents*). Exactly one source must be provided.

    Parameters
    ----------
    output_folder:
        Path to the directory containing ``extract-*.json`` files. Used when
        neither *files* nor *documents* is provided.
    files:
        Explicit list of JSON extraction file paths to ingest. Takes priority
        over *output_folder* when both are given.
    documents:
        List of in-memory Document objects from run_extraction() (in-memory
        mode). When provided, *output_folder* and *files* are ignored.
    update_mode:
        If True, wipe existing structure of the latest version and re-insert
        without creating a new version.
    tenant_id, created_by_user_id, job_id:
        Optional caller-supplied provenance metadata written onto every
        :Document node created by this stage (see ``ingest.loader.load_file``).

    Returns
    -------
    StageResult
        Stage result with one DocumentResult per ingested document.

    Raises
    ------
    ValueError
        If no source is provided.
    """
    t0 = time.monotonic()

    if documents is None and files is None and output_folder is None:
        raise ValueError(
            "At least one of documents, files, or output_folder must be provided."
        )

    driver = get_driver()
    doc_results: list[DocumentResult] = []
    try:
        setup_schema(driver)

        if documents is not None:
            # In-memory mode
            doc_names = load_documents(
                documents,
                driver,
                update_mode=update_mode,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            for doc in documents:
                success = doc.document_name in doc_names
                doc_results.append(DocumentResult(
                    document_name=doc.document_name,
                    nodes_processed=1 if success else 0,
                    nodes_failed=0 if success else 1,
                    errors=[] if success else [f"Failed to ingest '{doc.document_name}'"],
                ))
        elif files is not None:
            doc_names = load_files(
                files,
                driver,
                update_mode=update_mode,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            for path in files:
                doc_name = path.stem.removeprefix("extract-")
                success = doc_name in doc_names
                doc_results.append(DocumentResult(
                    document_name=doc_name,
                    nodes_processed=1 if success else 0,
                    nodes_failed=0 if success else 1,
                    errors=[] if success else [f"Failed to ingest '{path.name}'"],
                ))
        else:
            output_path = Path(output_folder)
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Output folder not found: '{output_folder}'. "
                    f"Run run_extraction() first."
                )
            extract_files = list(output_path.rglob("extract-*.json"))
            if not extract_files:
                raise FileNotFoundError(
                    f"No 'extract-*.json' files found in '{output_folder}'. "
                    f"Run run_extraction() first."
                )
            doc_names = load_folder(
                output_path,
                driver,
                update_mode=update_mode,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            for path in extract_files:
                doc_name = path.stem.removeprefix("extract-")
                success = doc_name in doc_names
                doc_results.append(DocumentResult(
                    document_name=doc_name,
                    nodes_processed=1 if success else 0,
                    nodes_failed=0 if success else 1,
                    errors=[] if success else [f"Failed to ingest '{path.name}'"],
                ))
    finally:
        driver.close()

    total_failed = sum(1 for r in doc_results if r.nodes_failed > 0)
    duration = time.monotonic() - t0
    return StageResult(
        stage="ingestion",
        success=total_failed == 0,
        documents=doc_results,
        total_processed=sum(r.nodes_processed for r in doc_results),
        total_failed=total_failed,
        duration_seconds=duration,
    )


def preflight_check_replaces(driver, replaces_name: str) -> dict:
    """Verify that the document to be replaced exists in Neo4j.

    Queries for Document nodes with the given name and latest=True.
    Aborts the process if:

    - No document is found.
    - Multiple documents with the same name and latest=True are found
      (ambiguous — user should use a more specific path).

    Parameters
    ----------
    driver:
        An open, authenticated Neo4j driver.
    replaces_name:
        The ``name`` property of the document to replace.

    Returns
    -------
    dict
        A dict with ``path`` and ``version`` of the found document.

    Raises
    ------
    SystemExit
        If the document is not found or if the match is ambiguous.
    """
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        result = session.run(
            """
            MATCH (d:Document {name: $name, latest: true})
            RETURN d.path AS path, d.version AS version
            """,
            name=replaces_name,
        )
        rows = [dict(r) for r in result]

    if len(rows) == 0:
        raise SystemExit(
            f"error: --replaces '{replaces_name}' not found in Neo4j.\n"
            f"       No Document with name='{replaces_name}' and latest=true exists.\n"
            f"       Check the document name and try again."
        )
    if len(rows) > 1:
        paths_str = "\n".join(
            f"  - path={r['path']!r}, version={r['version']!r}" for r in rows
        )
        raise SystemExit(
            f"error: --replaces '{replaces_name}' is ambiguous.\n"
            f"       Multiple documents with name='{replaces_name}' and latest=true were found:\n"
            f"{paths_str}\n"
            f"       Rename your documents to disambiguate before using --replaces."
        )
    return rows[0]


def apply_replacement(driver, replaces_name: str, new_root_doc_names: list[str]) -> None:
    """After ingestion, link the old document to the new root document(s) via HAS_NEWER_VERSION.

    Sets old document latest=False and creates HAS_NEWER_VERSION relationships.

    Parameters
    ----------
    driver:
        An open, authenticated Neo4j driver.
    replaces_name:
        The ``name`` property of the old document being replaced.
    new_root_doc_names:
        Names of the newly ingested root documents (those without an IS_COMPOSED_OF parent).
    """
    if not new_root_doc_names:
        logger.warning("apply_replacement: no new root documents found; skipping.")
        return
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        # Find the new root documents (those just ingested with no IS_COMPOSED_OF parent)
        result = session.run(
            """
            MATCH (d:Document {latest: true})
            WHERE d.name IN $names AND NOT ()-[:IS_COMPOSED_OF]->(d)
            RETURN d.path AS path, d.version AS version, d.name AS name
            """,
            names=new_root_doc_names,
        )
        new_roots = [dict(r) for r in result]

    if not new_roots:
        logger.warning(
            "apply_replacement: could not find any root documents among %s; skipping.",
            new_root_doc_names,
        )
        return
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        for new_root in new_roots:
            session.run(
                """
                MATCH (old:Document {name: $old_name, latest: true})
                MATCH (new:Document {path: $new_path, version: $new_version})
                SET old.latest = false
                MERGE (old)-[:HAS_NEWER_VERSION]->(new)
                """,
                old_name=replaces_name,
                new_path=new_root["path"],
                new_version=new_root["version"],
            )
            logger.info(
                "apply_replacement: linked '%s' (latest=false) -[:HAS_NEWER_VERSION]-> '%s' (path=%s, v=%s)",
                replaces_name,
                new_root["name"],
                new_root["path"],
                new_root["version"],
            )
