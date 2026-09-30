"""
ingest/freeze.py — Document freezing (snapshot export + graph reduction).

``freeze_document()`` serialises the subtree of one or more :Document nodes
to a single JSON snapshot (streamed, never materialised in memory), stores it
through the configured freeze backend (``freeze/factory.py``) and reduces the
:Document node(s) to stubs marked ``frozen = true``, deleting the rest of the
subtree (minus what the ``keep_*`` flags keep). ``restore_document()``
(``ingest/restore.py``) rebuilds it from the snapshot.

``export_document_snapshot()`` is the serialisation primitive on its own:
the same snapshot, returned as a ``dict``, written to a file, or stored in
the freeze backend — without touching the graph.

The selector is the one of ``delete_document()`` (``ingest/deletion.py``):
``tenant_id`` mandatory and always applied, exactly one of ``path`` /
``job_id``, and the ``IS_COMPOSED_OF*`` cascade **downwards** from the
matched documents. The document set is resolved once and every later step
(export, mutation, stub marking) works on those explicit
``(tenant_id, path, version)`` keys, so what is archived is exactly what is
frozen.

Public API
----------
    result = await freeze_document("a.pdf", tenant_id="acme")
    result = await freeze_document(job_id="job-1", tenant_id="acme", keep_annotations=True)
    result = await freeze_document("a.pdf", tenant_id=None, delete_after_export=False)  # backup
    snapshot = await export_document_snapshot("a.pdf", tenant_id="acme")  # dict
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from scinr.newton.config import get_config
from scinr.newton.exceptions import FreezeError
from scinr.newton.ingest._cascade import (
    DELETE_BATCH_ROWS,
    DOCS_BY_KEY,
    SUBTREE_COUNTER_FIELDS,
    run_count_query,
    run_subtree_steps,
    subtree_steps,
)
from scinr.newton.ingest._gc import GC_REACHABILITY_MAX_HOPS, OrphanCollector, sweep_tenant
from scinr.newton.ingest._json_stream import (
    DictSnapshotWriter,
    JsonSnapshotWriter,
    SnapshotWriter,
)
from scinr.newton.ingest.config import get_driver
from scinr.newton.ingest.deletion import _as_list, _build_doc_match
from scinr.newton.results import FreezeResult
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1
"""Version of the snapshot format written by :func:`export_document_snapshot`."""

# ---------------------------------------------------------------------------
# Node identity (business keys) — shared with ingest/restore.py
# ---------------------------------------------------------------------------

_DOCUMENT_KEY_FIELDS = ("tenant_id", "path", "version")

# Global catalog nodes: never exported as nodes nor deleted, only referenced
# as relationship targets by their own key.
CATALOG_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "CatalogModel": ("name",),
    "ModelField": ("name", "model"),
    "EntityLabel": ("label",),
    "Theme": ("path",),
}

# Per-document and merged node labels keyed by ``uid``.
UID_LABELS = (
    "InfoUnit",
    "ModelDecision",
    "ProposedModel",
    "ProposedField",
    "ComplementaryMatch",
    "SupplementaryField",
    "ExtractionResult",
    "ModelInstance",
    "Entity",
    "LabeledEntity",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def node_identity(labels: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Return ``(primary label, business-key fields)`` for a node's labels.

    The primary label is the one whose constraint/index backs the key
    (``StructureNode`` for the multi-label structure nodes). Raises
    :class:`FreezeError` for a node type the snapshot does not know how to
    rebuild.
    """
    if "Document" in labels:
        return "Document", _DOCUMENT_KEY_FIELDS
    if "StructureNode" in labels:
        return "StructureNode", ("id",)
    for label, fields in CATALOG_KEY_FIELDS.items():
        if label in labels:
            return label, fields
    for label in UID_LABELS:
        if label in labels:
            return label, ("uid",)
    raise FreezeError(f"Node with labels {list(labels)!r} has no known business key.")


def is_catalog(primary_label: str) -> bool:
    return primary_label in CATALOG_KEY_FIELDS


def cypher_identifier(name: str) -> str:
    """Backquote a label / relationship type after validating it."""
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise FreezeError(f"Invalid label or relationship type in snapshot: {name!r}.")
    return f"`{name}`"


def _ref(primary_label: str, key: dict[str, Any]) -> str:
    """Hashable identity of a node inside one document entry."""
    return f"{primary_label}|{json.dumps(key, sort_keys=True, default=str)}"


def _ordered_labels(labels: Sequence[str], primary_label: str) -> list[str]:
    return [primary_label, *sorted(label for label in labels if label != primary_label)]


# ---------------------------------------------------------------------------
# Selector — the delete_document() selector plus the IS_COMPOSED_OF* cascade
# ---------------------------------------------------------------------------


def selector_filters(
    func_name: str,
    path: str | None,
    version: int | None,
    tenant_id: str | None,
    created_by_user_id: str | Sequence[str] | None,
    job_id: str | Sequence[str] | None,
) -> dict[str, Any]:
    """Validate the selector arguments and return the ``_build_doc_match`` filters.

    Same rules as ``delete_document()``: exactly one of *path* / *job_id*
    (``ValueError``), empty lists rejected, *tenant_id* converted to the
    stored key (``""`` → ``ValueError``).
    """
    if (path is None) == (job_id is None):
        raise ValueError(
            f"{func_name} requires exactly one of 'path' or 'job_id' "
            f"(got path={path!r}, job_id={job_id!r})."
        )
    return {
        "path": path,
        "version": version,
        "tenant_id": tenant_key(tenant_id),
        "created_by_user_id": _as_list("created_by_user_id", created_by_user_id),
        "job_id": _as_list("job_id", job_id),
    }


def selector_repr(filters: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in filters.items() if v is not None)


# Matched Documents (``is_seed``) plus their IS_COMPOSED_OF* descendants,
# each once. Appended to the _build_doc_match() prefix (tenant always set).
_DOCUMENT_SET_TAIL = """
OPTIONAL MATCH (d)-[:IS_COMPOSED_OF*]->(cd:Document)
WITH collect(DISTINCT d) AS seeds, collect(DISTINCT cd) AS descendants
UNWIND seeds + descendants AS doc
WITH DISTINCT doc, seeds
RETURN properties(doc) AS properties, doc IN seeds AS is_seed
ORDER BY doc.path, doc.version
"""


# (keep_structure_nodes, keep_annotations, keep_extraction_results)
KeepFlags = tuple[bool, bool, bool]


@dataclass(frozen=True)
class DocumentInfo:
    """One :Document of the resolved set (matched or descendant)."""

    path: str
    version: int
    tenant_id: str
    is_seed: bool
    properties: dict[str, Any]

    @property
    def frozen(self) -> bool:
        return bool(self.properties.get("frozen"))

    @property
    def frozen_blob_id(self) -> str | None:
        return self.properties.get("frozen_blob_id")

    @property
    def cleanup_pending(self) -> bool:
        """Marked frozen by a freeze that did not get to delete the whole subtree."""
        return self.frozen and bool(self.properties.get("frozen_cleanup_pending"))

    @property
    def keep_flags(self) -> KeepFlags:
        """The ``keep_*`` flags the stub was frozen with."""
        return (
            bool(self.properties.get("frozen_keep_structure_nodes")),
            bool(self.properties.get("frozen_keep_annotations")),
            bool(self.properties.get("frozen_keep_extraction_results")),
        )

    @property
    def key(self) -> dict[str, Any]:
        return {"path": self.path, "version": self.version}


def fetch_document_set(driver, database: str, filters: dict[str, Any]) -> list[DocumentInfo]:
    """Resolve the selector to the matched Documents and their descendants.

    A descendant of another tenant (a data relationship crossing tenants —
    impossible under the write invariant) raises :class:`FreezeError`.
    """
    match_clause, params = _build_doc_match(filters)
    tenant = filters["tenant_id"]

    def _do_query() -> list[DocumentInfo]:
        with driver.session(database=database) as session:
            result = session.run(match_clause + _DOCUMENT_SET_TAIL, **params)
            return [
                DocumentInfo(
                    path=record["properties"].get("path"),
                    version=record["properties"].get("version"),
                    tenant_id=record["properties"].get("tenant_id"),
                    is_seed=record["is_seed"],
                    properties=dict(record["properties"]),
                )
                for record in result
            ]

    docs = with_neo4j_retry_sync(_do_query)
    for doc in docs:
        if doc.tenant_id != tenant:
            raise FreezeError(
                f"Document path={doc.path!r} version={doc.version!r} reached via "
                f"IS_COMPOSED_OF belongs to tenant {doc.tenant_id!r}, not {tenant!r}."
            )
        if doc.path is None or doc.version is None:
            raise FreezeError(f"Document without path/version in the selection: {doc.properties!r}")
    return docs


# ---------------------------------------------------------------------------
# Snapshot export — Neo4j → writer, streaming
# ---------------------------------------------------------------------------

_CATALOG_PREDICATE = " OR ".join(f"n:{label}" for label in CATALOG_KEY_FIELDS)

# Every node of one document's subtree, once: the whole HAS_STRUCTURE /
# HAS_CHILD tree (unbounded — a strict hierarchy), what hangs from each
# StructureNode (InfoUnit, ModelDecision and its children, ExtractionResult),
# and everything reachable from each ExtractionResult through any outgoing
# relationship within GC_REACHABILITY_MAX_HOPS hops (ModelInstance / Entity /
# LabeledEntity may form cycles), catalog nodes excluded.
_SUBTREE_NODES_FRAGMENT = f"""
MATCH (d:Document {{tenant_id: $tenant_id, path: $path, version: $version}})
CALL (d) {{
  MATCH (d)-[:HAS_STRUCTURE]->(:StructureNode)-[:HAS_CHILD*0..]->(s:StructureNode)
  WITH DISTINCT s
  CALL (s) {{
    RETURN s AS n
    UNION
    MATCH (s)-[:HAS_INFO_UNIT]->(n:InfoUnit)
    RETURN n
    UNION
    MATCH (s)-[:HAS_MODEL_DECISION]->(n:ModelDecision)
    RETURN n
    UNION
    MATCH (s)-[:HAS_MODEL_DECISION]->(:ModelDecision)
          -[:HAS_PROPOSED_MODEL|HAS_COMPLEMENTARY_MATCH|HAS_SUPPLEMENTARY_FIELD]->(n)
    RETURN n
    UNION
    MATCH (s)-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(:ProposedModel)
          -[:HAS_PROPOSED_FIELD]->(n:ProposedField)
    RETURN n
    UNION
    MATCH (s)-[:HAS_EXTRACTION]->(n:ExtractionResult)
    RETURN n
    UNION
    MATCH (s)-[:HAS_EXTRACTION]->(:ExtractionResult)-[*1..{GC_REACHABILITY_MAX_HOPS}]->(n)
    WHERE NOT ({_CATALOG_PREDICATE})
    RETURN n
  }}
  RETURN n
}}
WITH DISTINCT n
"""

_SUBTREE_NODES_EXPORT_QUERY = (
    _SUBTREE_NODES_FRAGMENT + "RETURN labels(n) AS labels, properties(n) AS properties\n"
)

# Only the properties any business key is made of (never the whole node).
_KEY_PROJECTION = "{.id, .uid, .name, .model, .label, .path, .tenant_id, .version}"

# Outgoing relationships of every subtree node. Targets outside the entry's
# node set (beyond the hop bound, other documents) are dropped in Python;
# catalog targets are kept as references.
_SUBTREE_RELATIONSHIPS_EXPORT_QUERY = (
    _SUBTREE_NODES_FRAGMENT
    + f"""MATCH (n)-[r]->(m)
RETURN type(r) AS type,
       labels(n) AS from_labels, n {_KEY_PROJECTION} AS from_key,
       labels(m) AS to_labels, m {_KEY_PROJECTION} AS to_key,
       properties(r) AS properties
"""
)

# The Document's own relationships: HAS_STRUCTURE into its tree, and its
# links to other Documents of the tenant (folder IS_COMPOSED_OF, version chain
# HAS_NEWER_VERSION) in both directions — what restore_document() needs to
# re-link a Document it has to recreate (deleted, or restored from a backup).
_DOCUMENT_RELATIONSHIPS_EXPORT_QUERY = f"""
MATCH (d:Document {{tenant_id: $tenant_id, path: $path, version: $version}})
CALL (d) {{
  MATCH (d)-[r:HAS_STRUCTURE]->(m:StructureNode)
  RETURN d AS n, r, m
  UNION
  MATCH (d)-[r:IS_COMPOSED_OF|HAS_NEWER_VERSION]->(m:Document)
  WHERE m.tenant_id = $tenant_id
  RETURN d AS n, r, m
  UNION
  MATCH (n:Document)-[r:IS_COMPOSED_OF|HAS_NEWER_VERSION]->(d)
  WHERE n.tenant_id = $tenant_id
  RETURN n, r, d AS m
}}
RETURN type(r) AS type,
       labels(n) AS from_labels, n {_KEY_PROJECTION} AS from_key,
       labels(m) AS to_labels, m {_KEY_PROJECTION} AS to_key,
       properties(r) AS properties
"""


def _endpoint(labels: Sequence[str], key_props: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    primary, key_fields = node_identity(labels)
    key = {field: key_props.get(field) for field in key_fields}
    return primary, {"labels": _ordered_labels(labels, primary), "key": key}


def _export_document(session, writer: SnapshotWriter, doc: DocumentInfo, tenant: str) -> int:
    """Write one ``documents[]`` entry; return the number of nodes written."""
    params = {"tenant_id": tenant, "path": doc.path, "version": doc.version}
    writer.begin_document(doc.properties)

    emitted: set[str] = set()
    for record in session.run(_SUBTREE_NODES_EXPORT_QUERY, **params):
        labels, props = record["labels"], record["properties"]
        primary, key_fields = node_identity(labels)
        if props.get("tenant_id") != tenant:
            raise FreezeError(
                f"{primary} node reachable from Document path={doc.path!r} "
                f"version={doc.version!r} has tenant_id={props.get('tenant_id')!r}, "
                f"expected {tenant!r}: the tenant write invariant is violated, "
                "so the subtree cannot be archived."
            )
        key = {field: props.get(field) for field in key_fields}
        if any(value is None for value in key.values()):
            raise FreezeError(f"{primary} node without business key {key_fields!r}: {props!r}")
        ref = _ref(primary, key)
        if ref in emitted:
            continue
        emitted.add(ref)
        writer.write_node(
            {
                "labels": _ordered_labels(labels, primary),
                "key": key,
                "properties": {k: v for k, v in props.items() if k not in key_fields},
            }
        )

    for query in (_DOCUMENT_RELATIONSHIPS_EXPORT_QUERY, _SUBTREE_RELATIONSHIPS_EXPORT_QUERY):
        for record in session.run(query, **params):
            _, source = _endpoint(record["from_labels"], record["from_key"])
            to_primary, target = _endpoint(record["to_labels"], record["to_key"])
            # The Document's own relationships are all kept; subtree ones only
            # towards the entry's nodes or catalog nodes.
            if (
                query is _SUBTREE_RELATIONSHIPS_EXPORT_QUERY
                and not is_catalog(to_primary)
                and _ref(to_primary, target["key"]) not in emitted
            ):
                continue
            writer.write_relationship(
                {
                    "type": record["type"],
                    "from": source,
                    "to": target,
                    "properties": dict(record["properties"]),
                }
            )

    writer.end_document()
    return len(emitted)


def _stream_snapshot(
    driver,
    database: str,
    docs: Sequence[DocumentInfo],
    header: dict[str, Any],
    writer: SnapshotWriter,
) -> int:
    """Write the whole snapshot to *writer*; return the number of nodes written."""
    tenant = header["tenant_id"]
    nodes = 0
    writer.begin(header)
    with driver.session(database=database) as session:
        for doc in docs:
            nodes += _export_document(session, writer, doc, tenant)
    writer.end()
    return nodes


def _write_snapshot_file(
    driver, database: str, docs: Sequence[DocumentInfo], header: dict[str, Any], dest: Path
) -> None:
    """Stream the snapshot to *dest*. A retried attempt starts the file over;
    a failed export removes the partial file."""

    def _do_write() -> int:
        with dest.open("w", encoding="utf-8") as fh:
            return _stream_snapshot(driver, database, docs, header, JsonSnapshotWriter(fh))

    try:
        nodes = with_neo4j_retry_sync(_do_write)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    logger.debug("Snapshot written to %s (%d documents, %d nodes)", dest, len(docs), nodes)


def _build_snapshot_dict(
    driver, database: str, docs: Sequence[DocumentInfo], header: dict[str, Any]
) -> dict[str, Any]:
    def _do_build() -> dict[str, Any]:
        writer = DictSnapshotWriter()
        _stream_snapshot(driver, database, docs, header, writer)
        return writer.snapshot

    return with_neo4j_retry_sync(_do_build)


def _snapshot_header(
    tenant: str, timestamp: str, keep_flags: dict[str, bool] | None, mode: str
) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "tenant_id": tenant,
        "frozen_at": timestamp,
        "mode": mode,
        "keep_flags": keep_flags,
    }


def _document_metadata(doc: DocumentInfo) -> dict[str, Any]:
    """The freeze backend's per-document metadata (what find_snapshots() matches)."""
    return {
        **doc.key,
        "job_id": doc.properties.get("job_id"),
        "created_by_user_id": doc.properties.get("created_by_user_id"),
    }


async def _export_to_storage(
    driver,
    database: str,
    docs: Sequence[DocumentInfo],
    header: dict[str, Any],
    *,
    mode: str,
    path: str | None,
    version: int | None,
    tenant_id: str | None,
    created_by_user_id: str | Sequence[str] | None,
    job_id: str | Sequence[str] | None,
    timings: dict[str, float] | None = None,
) -> str:
    """Neo4j → temporary file → freeze backend; return the ``frozen_blob_id``.

    The backend is resolved first, so a ``ConfigurationError`` (backend
    ``"none"``) is raised before reading the graph. With *timings*, the
    seconds spent reading the graph (``export``) and storing the snapshot
    (``upload``) are written to it.
    """
    from scinr.newton.freeze.factory import get_freeze_storage

    repository = get_freeze_storage()
    fd, tmp_name = tempfile.mkstemp(prefix="scinr-snapshot-", suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        started = time.perf_counter()
        await asyncio.to_thread(_write_snapshot_file, driver, database, docs, header, tmp_path)
        uploading = time.perf_counter()
        blob_id = await repository.store_snapshot(
            tmp_path,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
            metadata={
                "schema_version": header["schema_version"],
                "frozen_at": header["frozen_at"],
                "keep_flags": header["keep_flags"],
                "mode": mode,
                "path": path,
                "version": version,
                "documents": [_document_metadata(doc) for doc in docs],
            },
        )
        if timings is not None:
            timings["export"] = uploading - started
            timings["upload"] = time.perf_counter() - uploading
        return blob_id
    finally:
        tmp_path.unlink(missing_ok=True)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _format_timings(timings: dict[str, float]) -> str:
    return ", ".join(f"{phase}={seconds:.2f}s" for phase, seconds in timings.items())


# ---------------------------------------------------------------------------
# Graph mutation — freeze (§7.3) and backup marking
# ---------------------------------------------------------------------------

# Every query below starts from explicit (tenant_id, path, version) keys:
# the document set resolved (and exported) before any mutation.
_DOCS_BY_KEY = DOCS_BY_KEY

# In-transaction guard: every document still there and none frozen yet
# (a concurrent freeze between the existence check and the mutation).
_FREEZE_GUARD_QUERY = (
    _DOCS_BY_KEY
    + "RETURN count(d) AS found, count(CASE WHEN d.frozen = true THEN 1 END) AS frozen\n"
)

# frozen_cleanup_pending stays until the subtree is gone (see _reduce_to_stubs).
_MARK_FROZEN_QUERY = (
    _DOCS_BY_KEY
    + """SET d.frozen = true,
    d.frozen_blob_id = $blob_id,
    d.frozen_at = $frozen_at,
    d.frozen_keep_structure_nodes = $keep_structure_nodes,
    d.frozen_keep_annotations = $keep_annotations,
    d.frozen_keep_extraction_results = $keep_extraction_results,
    d.frozen_cleanup_pending = true
RETURN count(d) AS n
"""
)

_CLEAR_PENDING_QUERY = (
    _DOCS_BY_KEY + "REMOVE d.frozen_cleanup_pending, d.deletion_pending\nRETURN count(d) AS n\n"
)

_MARK_BACKUP_QUERY = (
    _DOCS_BY_KEY
    + """SET d.last_backup_blob_id = $blob_id, d.last_backup_at = $backup_at
RETURN count(d) AS n
"""
)

_FREEZE_COUNTER_FIELDS = SUBTREE_COUNTER_FIELDS


def _mark_frozen(
    driver,
    database: str,
    tenant: str,
    docs: Sequence[DocumentInfo],
    *,
    blob_id: str,
    frozen_at: str,
    keep_flags: KeepFlags,
) -> None:
    """Mark the stubs — in one write transaction, **before** anything is deleted.

    The subtree is then deleted in batches (``_reduce_to_stubs``), which is
    not atomic: marking first means that whatever happens next, every
    document already points at the snapshot that holds its whole subtree.
    """
    keys = [doc.key for doc in docs]
    base = {"tenant_id": tenant, "keys": keys}
    keep_structure_nodes, keep_annotations, keep_extraction_results = keep_flags

    def _do_mark() -> None:
        with driver.session(database=database) as session:
            with session.begin_transaction() as tx:
                try:
                    guard = tx.run(_FREEZE_GUARD_QUERY, **base).single()
                    if guard["found"] != len(keys) or guard["frozen"]:
                        raise FreezeError(
                            "The documents to freeze changed during the export "
                            f"(expected {len(keys)} unfrozen, found {guard['found']} "
                            f"with {guard['frozen']} already frozen); nothing was mutated."
                        )
                    tx.run(
                        _MARK_FROZEN_QUERY,
                        **base,
                        blob_id=blob_id,
                        frozen_at=frozen_at,
                        keep_structure_nodes=keep_structure_nodes,
                        keep_annotations=keep_annotations,
                        keep_extraction_results=keep_extraction_results,
                    ).single()
                    tx.commit()
                except Exception:
                    tx.rollback()
                    logger.exception(
                        "freeze_document: stub marking rolled back (tenant=%r, keys=%r)",
                        tenant,
                        keys,
                    )
                    raise

    with_neo4j_retry_sync(_do_mark)


def _reduce_to_stubs(
    driver,
    database: str,
    tenant: str,
    groups: dict[KeepFlags, list[dict[str, Any]]],
    *,
    interrupted: bool,
    timings: dict[str, float],
) -> tuple[dict[str, int], dict[str, int]]:
    """Delete the subtree of the marked stubs, collect the orphans and clear
    ``frozen_cleanup_pending``.

    *groups* maps each ``keep_*`` combination to the keys frozen with it.
    Everything runs in bounded transactions (``ingest/_cascade.py``) and only
    touches what is still there, so after a failure the same call finishes
    the job. The pending mark goes last — after the GC — so a failed GC is
    retried too.

    The orphans are collected among what the deleted :ExtractionResult nodes
    pointed at (``OrphanCollector``). Those candidates only live in this
    process: with *interrupted* — some document was left half-way by an
    earlier freeze or delete, whose candidates are lost — the whole tenant is
    swept instead.

    The seconds of each phase are written to *timings* (``delete``, ``gc``,
    ``clear_pending``) as it ends, so a failure leaves the finished ones.
    """
    counters = dict.fromkeys(_FREEZE_COUNTER_FIELDS, 0)
    started = time.perf_counter()
    orphans = None
    if not interrupted:
        orphans = OrphanCollector(
            driver, tenant_id=tenant, database=database, caller="freeze_document"
        )
    for keep_flags, keys in groups.items():
        keep_extraction_results = keep_flags[2]
        counts = run_subtree_steps(
            driver,
            database,
            tenant,
            keys,
            subtree_steps(*keep_flags),
            caller="freeze_document",
            # Kept ExtractionResults orphan nothing.
            orphans=None if keep_extraction_results else orphans,
        )
        for name, value in counts.items():
            counters[name] += value
    timings["delete"] = time.perf_counter() - started
    started = time.perf_counter()
    if orphans is not None:
        orphans.collect()
        gc_counts = orphans.counts()
    else:
        gc_counts = sweep_tenant(
            driver, tenant_id=tenant, database=database, caller="freeze_document"
        )
    timings["gc"] = time.perf_counter() - started
    started = time.perf_counter()
    keys = [key for group in groups.values() for key in group]
    for start in range(0, len(keys), DELETE_BATCH_ROWS):
        run_count_query(
            driver,
            database,
            _CLEAR_PENDING_QUERY,
            tenant_id=tenant,
            keys=keys[start : start + DELETE_BATCH_ROWS],
        )
    timings["clear_pending"] = time.perf_counter() - started
    return counters, gc_counts


def _mark_backup(
    driver, database: str, tenant: str, docs: Sequence[DocumentInfo], blob_id: str, backup_at: str
) -> None:
    keys = [doc.key for doc in docs]

    def _do_mark() -> None:
        with driver.session(database=database) as session:
            session.execute_write(
                lambda tx: tx.run(
                    _MARK_BACKUP_QUERY,
                    tenant_id=tenant,
                    keys=keys,
                    blob_id=blob_id,
                    backup_at=backup_at,
                ).consume()
            )

    with_neo4j_retry_sync(_do_mark)


def _refuse_frozen(docs: Sequence[DocumentInfo], func_name: str) -> None:
    # frozen / frozen_blob_id are read from the raw node properties (the
    # Document node carries them since freeze_document() set them).
    frozen = [doc for doc in docs if doc.frozen]
    if frozen:
        listed = ", ".join(f"{doc.path!r} v{doc.version}" for doc in frozen)
        raise FreezeError(
            f"{func_name}: already frozen: {listed}. Restore it with restore_document() first."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def export_document_snapshot(
    path: str | None = None,
    version: int | None = None,
    *,
    tenant_id: str | None,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
    destination: Literal["dict", "file", "storage"] = "dict",
    file_path: Path | str | None = None,
) -> dict | Path | str:
    """Serialise the subtree of the selected document(s) to a snapshot.

    Read-only: the graph is never modified. The selector is the one of
    :func:`freeze_document` (and ``delete_document()``): *tenant_id*
    mandatory, exactly one of *path* / *job_id*, the ``IS_COMPOSED_OF*``
    descendants of the matched documents included.

    Parameters
    ----------
    destination:
        - ``"dict"`` (default): return the snapshot as a ``dict``, built in
          memory — for small documents and debugging.
        - ``"file"``: stream it to *file_path* and return that ``Path``.
        - ``"storage"``: stream it to a temporary file, upload it to the
          freeze backend (``get_freeze_storage()``) and return the
          ``frozen_blob_id``. Only this destination needs a freeze backend.
    file_path:
        Destination file; required with ``destination="file"`` only.

    Raises
    ------
    FreezeError
        If nothing matches, a selected document is frozen (its subtree is no
        longer in the graph — restore it first), or a node of the subtree
        belongs to another tenant.
    ConfigurationError
        With ``destination="storage"`` when the freeze backend resolves to
        ``"none"``.
    ValueError
        Invalid selector, unknown *destination*, or *file_path* missing /
        given without ``destination="file"``.
    """
    if destination not in ("dict", "file", "storage"):
        raise ValueError(f"Unknown destination {destination!r}: use 'dict', 'file' or 'storage'.")
    if (destination == "file") != (file_path is not None):
        raise ValueError("file_path is required with destination='file' and only valid with it.")
    filters = selector_filters(
        "export_document_snapshot", path, version, tenant_id, created_by_user_id, job_id
    )
    database = get_config().neo4j_database

    driver = get_driver()
    try:
        docs = await asyncio.to_thread(fetch_document_set, driver, database, filters)
        if not docs:
            raise FreezeError(f"export_document_snapshot: no Document found for {selector_repr(filters)}.")
        _refuse_frozen(docs, "export_document_snapshot")
        header = _snapshot_header(filters["tenant_id"], _now_iso(), None, "export")

        if destination == "dict":
            return await asyncio.to_thread(_build_snapshot_dict, driver, database, docs, header)
        if destination == "file":
            dest = Path(file_path)
            await asyncio.to_thread(_write_snapshot_file, driver, database, docs, header, dest)
            return dest
        return await _export_to_storage(
            driver,
            database,
            docs,
            header,
            mode="export",
            path=path,
            version=version,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )
    finally:
        driver.close()


async def freeze_document(
    path: str | None = None,
    version: int | None = None,
    *,
    tenant_id: str | None,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
    keep_structure_nodes: bool = False,
    keep_annotations: bool = False,
    keep_extraction_results: bool = False,
    delete_after_export: bool = True,
) -> FreezeResult:
    """Archive the subtree of the selected document(s) and reduce them to stubs.

    Target selection is identical to ``delete_document()``: *tenant_id*
    mandatory (keyword-only, no default; ``None`` / ``"__public__"`` = public
    documents), exactly one of *path* (all versions, or *version*) or
    *job_id* (one value or several), *created_by_user_id* as an extra AND
    filter, and the ``IS_COMPOSED_OF*`` cascade **downwards** from the
    matched documents (freezing a folder freezes its whole subtree; freezing
    a leaf by *path* leaves its folder and siblings alone).

    Steps:

    1. Resolve the document set; ``found=False`` (nothing touched) if empty,
       :class:`FreezeError` if any of them is already frozen (except the
       stubs of an interrupted freeze — see below).
    2. Export the complete snapshot (always the whole subtree, whatever the
       ``keep_*`` flags) to the freeze backend. Fail-fast: if it fails, the
       graph is not touched.
    3. ``delete_after_export=False`` (backup mode): only set
       ``last_backup_blob_id`` / ``last_backup_at`` on the documents and stop.
    4. Otherwise mark every Document ``frozen=true`` with ``frozen_blob_id``,
       ``frozen_at``, the ``frozen_keep_*`` flags and
       ``frozen_cleanup_pending=true`` (one transaction), and then, in
       bounded transactions (``ingest/_cascade.py``): re-link kept
       ModelDecision / ExtractionResult nodes to their Document when their
       StructureNode goes, and delete the subtree (InfoUnits always;
       StructureNodes, annotations and extraction results unless kept). The
       Document nodes themselves stay.
    5. Delete the :Entity / :ModelInstance / :LabeledEntity nodes of the
       tenant this freeze leaves orphaned (``ingest/_gc.py``), and remove
       ``frozen_cleanup_pending``. Only what hung from the deleted
       :ExtractionResult nodes is checked, not the whole tenant (that is
       ``collect_orphans()``); a call that finishes an interrupted freeze
       has lost those candidates and sweeps the whole tenant instead.

    The duration of each phase is logged (``export`` = graph → snapshot
    file, ``upload``, ``mark``, ``delete``, ``gc``, ``clear_pending``).

    The subtree is **not** deleted atomically (one transaction holding the
    subtree of hundreds of documents overruns the server's transaction
    memory). If step 4 or 5 fails, the documents are already frozen stubs
    pointing at the complete snapshot, with part of their subtree still in
    the graph and ``frozen_cleanup_pending=true``. Calling
    ``freeze_document()`` again on them finishes the deletion (no new
    export; the ``keep_*`` flags of the first call apply, the ones passed
    are ignored for those documents), and ``restore_document()`` brings them
    back whole.

    Documental storage (raw files, converted pages) is not touched: a frozen
    document keeps its ``raw_file_id``.

    Parameters
    ----------
    keep_structure_nodes:
        Keep the whole :StructureNode tree (only :InfoUnit nodes go).
    keep_annotations:
        Keep :ModelDecision nodes and their children.
    keep_extraction_results:
        Keep :ExtractionResult nodes (and so what hangs from them within
        the GC hop bound).
    delete_after_export:
        ``False`` = backup only (no ``keep_*`` flag may be set).

    Raises
    ------
    TypeError
        If *tenant_id* is not passed.
    ValueError
        Invalid selector, or ``delete_after_export=False`` with a ``keep_*`` flag.
    FreezeError
        A selected document is already frozen, or the subtree violates the
        tenant invariant.
    ConfigurationError
        If the freeze backend resolves to ``"none"``.
    """
    filters = selector_filters("freeze_document", path, version, tenant_id, created_by_user_id, job_id)
    if not delete_after_export and (
        keep_structure_nodes or keep_annotations or keep_extraction_results
    ):
        raise ValueError(
            "delete_after_export=False (backup mode) deletes nothing, so the keep_* "
            "flags do not apply; leave them False."
        )
    mode: Literal["freeze", "backup"] = "freeze" if delete_after_export else "backup"
    tenant = filters["tenant_id"]
    selector = selector_repr(filters)
    timings: dict[str, float] = {}
    result_scope = {
        "path": path,
        "version": version,
        "job_id": job_id,
        "tenant_id": tenant_id,
        "created_by_user_id": created_by_user_id,
    }
    database = get_config().neo4j_database

    driver = get_driver()
    try:
        docs = await asyncio.to_thread(fetch_document_set, driver, database, filters)
        if not docs:
            logger.warning("freeze_document: no Document found for %s; nothing to freeze.", selector)
            return FreezeResult(
                **result_scope,
                found=False,
                mode=mode,
                frozen_blob_id=None,
                versions_frozen=[],
                documents_frozen=0,
            )
        # A freeze that was interrupted while deleting left its stubs marked
        # frozen_cleanup_pending: this call finishes it instead of refusing.
        resumed = [doc for doc in docs if doc.cleanup_pending] if delete_after_export else []
        fresh = [doc for doc in docs if not (delete_after_export and doc.cleanup_pending)]
        _refuse_frozen(fresh, "freeze_document")
        versions = sorted({doc.version for doc in docs if doc.is_seed})

        timestamp = _now_iso()
        keep_flags: KeepFlags = (keep_structure_nodes, keep_annotations, keep_extraction_results)
        blob_id: str | None = None
        if fresh:
            header = _snapshot_header(
                tenant,
                timestamp,
                {
                    "structure_nodes": keep_structure_nodes,
                    "annotations": keep_annotations,
                    "extraction_results": keep_extraction_results,
                },
                mode,
            )
            blob_id = await _export_to_storage(
                driver,
                database,
                fresh,
                header,
                mode=mode,
                path=path,
                version=version,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
                timings=timings,
            )
            logger.info(
                "freeze_document: snapshot stored for %s (%d documents) → frozen_blob_id=%s",
                selector,
                len(fresh),
                blob_id,
            )

            started = time.perf_counter()
            if not delete_after_export:
                await asyncio.to_thread(
                    _mark_backup, driver, database, tenant, fresh, blob_id, timestamp
                )
                timings["mark"] = time.perf_counter() - started
                logger.info(
                    "freeze_document: backup complete for %s. documents_backed_up=%d timings: %s",
                    selector,
                    len(fresh),
                    _format_timings(timings),
                )
                return FreezeResult(
                    **result_scope,
                    found=True,
                    mode="backup",
                    frozen_blob_id=blob_id,
                    versions_frozen=versions,
                    documents_frozen=len(fresh),
                )

            await asyncio.to_thread(
                _mark_frozen,
                driver,
                database,
                tenant,
                fresh,
                blob_id=blob_id,
                frozen_at=timestamp,
                keep_flags=keep_flags,
            )
            timings["mark"] = time.perf_counter() - started

        if resumed:
            logger.info(
                "freeze_document: finishing the interrupted freeze of %d document(s) for %s "
                "(with the keep_* flags they were frozen with).",
                len(resumed),
                selector,
            )
            if blob_id is None:
                seeds = [doc for doc in resumed if doc.is_seed]
                blob_id = (seeds or resumed)[0].frozen_blob_id

        groups: dict[KeepFlags, list[dict[str, Any]]] = {}
        for doc in fresh:
            groups.setdefault(keep_flags, []).append(doc.key)
        for doc in resumed:
            groups.setdefault(doc.keep_flags, []).append(doc.key)
        try:
            counters, gc_counts = await asyncio.to_thread(
                _reduce_to_stubs,
                driver,
                database,
                tenant,
                groups,
                interrupted=bool(resumed)
                or any(doc.properties.get("deletion_pending") for doc in fresh),
                timings=timings,
            )
        except Exception:
            logger.error(
                "freeze_document: interrupted while deleting the subtree of %s. The documents "
                "are marked frozen and their snapshot is stored, but part of their subtree is "
                "still in the graph (frozen_cleanup_pending = true). Call freeze_document() "
                "again with the same selector to finish, or restore_document() to bring them "
                "back. timings of the finished phases: %s",
                selector,
                _format_timings(timings),
            )
            raise
        logger.info(
            "freeze_document: complete for %s. documents_frozen=%d structure_nodes_deleted=%d "
            "info_units_deleted=%d gc_entity_model_instance_deleted=%d "
            "gc_labeled_entity_deleted=%d timings: %s",
            selector,
            len(docs),
            counters["structure_nodes_deleted"],
            counters["info_units_deleted"],
            gc_counts["gc_entity_model_instance_deleted"],
            gc_counts["gc_labeled_entity_deleted"],
            _format_timings(timings),
        )
        return FreezeResult(
            **result_scope,
            found=True,
            mode="freeze",
            frozen_blob_id=blob_id,
            versions_frozen=versions,
            documents_frozen=len(docs),
            **counters,
            **gc_counts,
        )
    finally:
        driver.close()
