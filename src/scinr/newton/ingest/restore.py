"""
ingest/restore.py — Rebuild frozen documents from their snapshot.

The inverse of :func:`~scinr.newton.ingest.freeze.freeze_document`: download
the snapshot from the freeze backend, validate it (tenant, format) in a first
streaming pass, rebuild the subtree in batches in a second one, remove the
temporary ``(:Document)-[:HAS_MODEL_DECISION|HAS_EXTRACTION]->()`` links,
unmark the stubs and delete the snapshot once no Document references it.
Every write runs in a bounded transaction (row batches for the rebuild, 50
documents at a time for the finish), so the graph database's transaction memory depends on
``batch_size × concurrency``, not on how much is restored.

Rebuild semantics per node type (see ``docs/user-guides/document-freezing.md``):

- :StructureNode / :InfoUnit — ``MERGE`` by ``id`` / ``uid``: a kept
  StructureNode is reused as is.
- :ModelInstance / :Entity / :LabeledEntity — ``MERGE`` by ``uid``; a node
  still alive (shared with another document of the tenant) keeps its
  properties and only gains the snapshot's ``created_by_user_ids`` /
  ``job_ids`` (set union).
- :ModelDecision / :ExtractionResult and their children — when the
  StructureNode was kept and the family was not, whatever hangs from it now
  (a re-annotation / re-extraction while frozen) is deleted first with the
  pipeline's own idempotency helpers, so the snapshot wins; then ``MERGE`` by
  ``uid`` (a kept node is reused).
- Catalog nodes — ``MERGE`` by their key, as the pipeline does.
- Relationships — created only when the same relationship (type, endpoints,
  properties) does not exist yet.

Every write is idempotent: if a restore fails half-way, the stubs stay
frozen and the snapshot stays stored, so calling ``restore_document()``
again completes it.

Public API
----------
    result = await restore_document("a.pdf", tenant_id="acme")
    result = await restore_document(job_id="job-1", tenant_id="acme")
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import ijson

from scinr.newton.config import get_config
from scinr.newton.exceptions import FreezeError
from scinr.newton.graph import get_graph_store
from scinr.newton.graph.base import RestoreWriter
from scinr.newton.graph.types import DocumentInfo, RestoreCounters
from scinr.newton.ingest.freeze import (
    SNAPSHOT_SCHEMA_VERSION,
    fetch_document_set,
    is_catalog,
    node_identity,
    selector_filters,
    selector_repr,
    validate_identifier,
)
from scinr.newton.results import RestoreResult

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000
"""Rows per write transaction of the rebuild."""

DEFAULT_CONCURRENCY = 4
"""Write transactions of the rebuild in flight at once."""

# Indexes the rebuild's MERGE / MATCH by key rely on and that graphs ingested
# before document freezing may lack (setup_schema() only runs at ingestion).
# The others (Document, StructureNode, InfoUnit, ExtractionResult,
# ModelInstance, Entity, LabeledEntity, ModelField, EntityLabel) are backed by
# uniqueness constraints that predate it.
_RESTORE_INDEXES = (
    "idx_model_decision_uid",
    "idx_proposed_model_uid",
    "idx_proposed_field_uid",
    "idx_complementary_match_uid",
    "idx_supplementary_field_uid",
    "idx_catalog_model_name",
)

# ---------------------------------------------------------------------------
# Incremental snapshot reader
# ---------------------------------------------------------------------------

_ITEM_KINDS = {
    "documents.item.document": "document",
    "documents.item.nodes.item": "node",
    "documents.item.relationships.item": "relationship",
}


def _build_value(events: Iterator, event: str, value: Any) -> Any:
    """Assemble the JSON value that starts with (*event*, *value*)."""
    builder = ijson.ObjectBuilder()
    builder.event(event, value)
    depth = 1
    for _, ev, val in events:
        builder.event(ev, val)
        if ev in ("start_map", "start_array"):
            depth += 1
        elif ev in ("end_map", "end_array"):
            depth -= 1
            if depth == 0:
                break
    return builder.value


def iter_snapshot(fh: BinaryIO) -> Iterator[tuple[str, int, Any]]:
    """Yield ``(kind, entry_index, value)`` from a snapshot file, one element
    at a time: ``("header", -1, {...})`` once, then per ``documents[]`` entry
    its ``"document"``, every ``"node"`` and every ``"relationship"``.

    Only one element is ever held in memory (``ijson``); numbers are parsed
    as ``int`` / ``float`` (never ``Decimal``).
    """
    events = iter(ijson.parse(fh, use_float=True))
    header: dict[str, Any] = {}
    entry = -1
    for prefix, event, value in events:
        if prefix == "documents":
            if event == "start_array":
                yield "header", -1, header
        elif prefix == "documents.item":
            if event == "start_map":
                entry += 1
        elif prefix in _ITEM_KINDS:
            if event in ("start_map", "start_array"):
                yield _ITEM_KINDS[prefix], entry, _build_value(events, event, value)
        elif prefix and "." not in prefix:
            # Top-level header field.
            if event in ("start_map", "start_array"):
                header[prefix] = _build_value(events, event, value)
            elif event not in ("map_key", "end_map", "end_array"):
                header[prefix] = value


# ---------------------------------------------------------------------------
# Pass 1 — validation (nothing written)
# ---------------------------------------------------------------------------


@dataclass
class _EntryPlan:
    """What to do with one ``documents[]`` entry of a snapshot."""

    doc: DocumentInfo
    delete_stale_model_decisions: bool = False
    delete_stale_extraction_results: bool = False
    structure_node_ids: list[str] = field(default_factory=list)
    # True when the :Document itself is gone (deleted, or a backup being
    # restored) and must be recreated from the entry's properties.
    recreate: bool = False

    @property
    def needs_structure_ids(self) -> bool:
        return self.delete_stale_model_decisions or self.delete_stale_extraction_results


def _entry_plan(doc: DocumentInfo) -> _EntryPlan:
    # The stub's frozen_keep_* flags (raw node properties) tell what survived.
    props = doc.properties
    kept_structure = bool(props.get("frozen_keep_structure_nodes"))
    return _EntryPlan(
        doc=doc,
        delete_stale_model_decisions=kept_structure and not props.get("frozen_keep_annotations"),
        delete_stale_extraction_results=kept_structure
        and not props.get("frozen_keep_extraction_results"),
    )


DocKey = tuple[str, int]


@dataclass
class SnapshotScan:
    """What the validation pass learned about a snapshot (nothing written)."""

    header: dict[str, Any]
    # entry index → the :Document properties of the entry
    entries: dict[int, dict[str, Any]] = field(default_factory=dict)
    # (path, version) → child (path, version) keys, from the IS_COMPOSED_OF
    # relationships the snapshot holds (snapshots written since those are
    # exported; older ones have none)
    children: dict[DocKey, list[DocKey]] = field(default_factory=dict)
    # (path, version) → StructureNode ids, for the keys asked for
    structure_node_ids: dict[DocKey, list[str]] = field(default_factory=dict)

    def entry_index(self) -> dict[DocKey, int]:
        return {(props.get("path"), props.get("version")): i for i, props in self.entries.items()}


def _check_endpoint(endpoint: Any, tenant: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("key"), dict):
        raise FreezeError(f"Malformed relationship endpoint in snapshot: {endpoint!r}")
    labels = endpoint.get("labels") or []
    primary, key_fields = node_identity(labels)
    for label in labels:
        validate_identifier(label)
    key = endpoint["key"]
    if any(key.get(f) is None for f in key_fields):
        raise FreezeError(f"Relationship endpoint without its {key_fields!r} key: {endpoint!r}")
    if primary == "Document" and key["tenant_id"] != tenant:
        raise FreezeError(
            f"Snapshot relationship points at Document {key!r} of another tenant; "
            "nothing was restored."
        )
    return primary, key


def _check_relationship(
    value: dict[str, Any], tenant: str, doc_key: tuple | None
) -> list[tuple[str, dict[str, Any]]]:
    """A relationship may touch a :Document only if it is the entry's own one;
    between two Documents (folder / version links) one side must be it.
    Returns the (primary label, key) of both endpoints."""
    rel_type = value.get("type")
    validate_identifier(rel_type)
    endpoints = [_check_endpoint(value.get(side), tenant) for side in ("from", "to")]
    documents = [
        (key["tenant_id"], key["path"], key["version"])
        for primary, key in endpoints
        if primary == "Document"
    ]
    if documents and doc_key not in documents:
        outside = documents[0]
        raise FreezeError(
            f"Snapshot relationship points at Document {outside!r}, outside its entry {doc_key!r}."
        )
    return endpoints


def scan_snapshot(
    path: Path, tenant: str, *, structure_ids_for: set[DocKey] | frozenset = frozenset()
) -> SnapshotScan:
    """Validation pass over the snapshot at *path*: nothing is written.

    Raises :class:`FreezeError` when the header, a document entry, a node
    or a Document endpoint carries another tenant, an identifier is not a
    valid label / relationship type, or the file is malformed. Collects the
    StructureNode ids of the entries in *structure_ids_for*.
    """
    scan: SnapshotScan | None = None
    doc_key: tuple | None = None
    entry_key: DocKey | None = None
    with path.open("rb") as fh:
        for kind, entry, value in iter_snapshot(fh):
            if kind == "header":
                if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
                    raise FreezeError(
                        f"Unsupported snapshot schema_version {value.get('schema_version')!r}."
                    )
                if value.get("tenant_id") != tenant:
                    raise FreezeError(
                        f"Snapshot belongs to tenant {value.get('tenant_id')!r}, "
                        f"not to {tenant!r}; nothing was restored."
                    )
                scan = SnapshotScan(header=value)
            elif kind == "document":
                if value.get("tenant_id") != tenant:
                    raise FreezeError(
                        f"Snapshot document entry of tenant {value.get('tenant_id')!r} "
                        f"inside a {tenant!r} snapshot; nothing was restored."
                    )
                if value.get("path") is None or value.get("version") is None:
                    raise FreezeError(f"Snapshot document entry without path/version: {value!r}")
                doc_key = (tenant, value["path"], value["version"])
                entry_key = (value["path"], value["version"])
                scan.entries[entry] = value
            elif kind == "node":
                labels = value.get("labels") or []
                primary, key_fields = node_identity(labels)
                if is_catalog(primary) or primary == "Document":
                    raise FreezeError(f"Unexpected {primary} node in snapshot: {value!r}")
                for label in labels:
                    validate_identifier(label)
                props = value.get("properties") or {}
                if props.get("tenant_id") != tenant:
                    raise FreezeError(
                        f"Snapshot {primary} node {value.get('key')!r} has tenant_id="
                        f"{props.get('tenant_id')!r}, expected {tenant!r}; nothing was restored."
                    )
                key = value.get("key") or {}
                if any(key.get(f) is None for f in key_fields):
                    raise FreezeError(f"Snapshot {primary} node without its key: {value!r}")
                if primary == "StructureNode" and entry_key in structure_ids_for:
                    scan.structure_node_ids.setdefault(entry_key, []).append(key["id"])
            elif kind == "relationship":
                (from_label, parent), (to_label, child) = _check_relationship(value, tenant, doc_key)
                # IS_COMPOSED_OF also links StructureNodes inside the subtree;
                # only Document → Document is a folder link.
                if (
                    value.get("type") == "IS_COMPOSED_OF"
                    and from_label == "Document"
                    and to_label == "Document"
                ):
                    children = scan.children.setdefault((parent["path"], parent["version"]), [])
                    if (child["path"], child["version"]) not in children:
                        children.append((child["path"], child["version"]))

    if scan is None:
        raise FreezeError("Malformed snapshot: no 'documents' array.")
    return scan


def _selector_matches(props: dict[str, Any], filters: dict[str, Any]) -> bool:
    """The restore selector (``DocumentSelector`` filters) applied to one
    Document's properties, as stored in a snapshot."""
    if filters["path"] is not None and props.get("path") != filters["path"]:
        return False
    if filters["version"] is not None and props.get("version") != filters["version"]:
        return False
    for name in ("job_id", "created_by_user_id"):
        if filters[name] is not None and props.get(name) not in filters[name]:
            return False
    return True


def _with_descendants(
    seeds: Sequence[DocKey], children: dict[DocKey, list[DocKey]], exclude: set[DocKey]
) -> list[DocKey]:
    """*seeds* plus their IS_COMPOSED_OF* descendants, in order, each once."""
    ordered: list[DocKey] = []
    pending = list(seeds)
    while pending:
        key = pending.pop(0)
        if key in ordered or (key in exclude and key not in seeds):
            continue
        ordered.append(key)
        pending.extend(children.get(key, []))
    return ordered


# ---------------------------------------------------------------------------
# Pass 2 — rebuild (async writer, batches)
# ---------------------------------------------------------------------------


def _sort_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class _Rebuilder:
    """Groups snapshot elements into batches and hands them to a RestoreWriter.

    Node batches are grouped by label set (one lane each: a label set is
    written by one transaction at a time, so labels without a uniqueness
    constraint never see two concurrent MERGEs of the same key). Every node
    batch is finished before the first relationship batch of the entry is
    sent, since relationships MATCH their endpoints. Relationship batches are
    grouped by (type, source label, target label), one lane each, except that
    all groups pointing at catalog nodes share a single lane (every document
    points at the same few catalog nodes). Rows are sorted by key inside each
    batch so concurrent transactions take their locks in the same order.
    """

    def __init__(self, writer: RestoreWriter, batch_size: int) -> None:
        self.writer = writer
        self.batch_size = batch_size
        self.nodes: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        self.relationships: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self.nodes_created = 0
        self.nodes_reused = 0
        self.relationships_created = 0
        self._nodes_in_flight = False

    def _count_nodes(self, rows: int) -> Callable[[RestoreCounters], None]:
        def _done(counters: RestoreCounters) -> None:
            self.nodes_created += counters.nodes_created
            self.nodes_reused += rows - counters.nodes_created

        return _done

    def _count_relationships(self, counters: RestoreCounters) -> None:
        self.relationships_created += counters.relationships_created

    async def _flush_node_group(self, labels: tuple[str, ...]) -> None:
        rows = self.nodes.pop(labels, None)
        if rows:
            rows.sort(key=lambda row: _sort_key(row["key"]))
            self._nodes_in_flight = True
            await self.writer.merge_nodes(
                ("nodes", labels), labels, rows, self._count_nodes(len(rows))
            )

    async def _flush_relationship_group(self, group: tuple[str, str, str]) -> None:
        rows = self.relationships.pop(group, None)
        if rows:
            rows.sort(key=lambda row: (_sort_key(row["from"]), _sort_key(row["to"])))
            lane = ("catalog",) if is_catalog(group[2]) else ("relationships", group)
            await self.writer.create_relationships(
                lane, group, rows, self._count_relationships
            )

    async def _finish_nodes(self) -> None:
        for labels in list(self.nodes):
            await self._flush_node_group(labels)
        if self._nodes_in_flight:
            await self.writer.drain()
            self._nodes_in_flight = False

    async def flush(self) -> None:
        """Write everything pending and wait for it — nodes first."""
        await self._finish_nodes()
        for group in list(self.relationships):
            await self._flush_relationship_group(group)
        await self.writer.drain()

    async def add_node(self, node: dict[str, Any]) -> None:
        labels = tuple(node["labels"])
        rows = self.nodes.setdefault(labels, [])
        rows.append({"key": node["key"], "props": node.get("properties") or {}})
        if len(rows) >= self.batch_size:
            await self._flush_node_group(labels)

    async def add_relationship(self, relationship: dict[str, Any]) -> None:
        if self.nodes or self._nodes_in_flight:
            await self._finish_nodes()
        source, target = relationship["from"], relationship["to"]
        group = (
            relationship["type"],
            node_identity(source["labels"])[0],
            node_identity(target["labels"])[0],
        )
        rows = self.relationships.setdefault(group, [])
        rows.append(
            {
                "from": source["key"],
                "to": target["key"],
                "props": relationship.get("properties") or {},
            }
        )
        if len(rows) >= self.batch_size:
            await self._flush_relationship_group(group)


async def _delete_stale_subtrees(
    writer: RestoreWriter, plans: dict[int, _EntryPlan], batch_size: int
) -> None:
    """Stale ModelDecision / ExtractionResult subtrees on kept StructureNodes,
    removed before any snapshot node is merged (a merge could otherwise reuse
    a stale node that is about to be deleted). One lane: the deletes of two
    batches can meet on a shared (tabular) ModelDecision."""
    for plan in plans.values():
        ids = plan.structure_node_ids
        for start in range(0, len(ids), batch_size):
            await writer.delete_stale(
                ("stale",),
                ids[start : start + batch_size],
                model_decisions=plan.delete_stale_model_decisions,
                extraction_results=plan.delete_stale_extraction_results,
            )
    await writer.drain()


async def _rebuild_snapshot(
    path: Path,
    plans: dict[int, _EntryPlan],
    *,
    batch_size: int,
    concurrency: int,
) -> dict[str, int]:
    """Second pass: write the planned entries of the snapshot at *path*."""
    writer = get_graph_store().restore.batch_writer(concurrency)
    rebuilder = _Rebuilder(writer, batch_size)
    try:
        started = time.perf_counter()
        await _delete_stale_subtrees(writer, plans, batch_size)
        stale_seconds = time.perf_counter() - started

        started = time.perf_counter()
        current: _EntryPlan | None = None
        with path.open("rb") as fh:
            for kind, entry, value in iter_snapshot(fh):
                if kind == "document":
                    await rebuilder.flush()
                    current = plans.get(entry)
                    if current is not None and current.recreate:
                        await rebuilder.add_node(_document_node(value))
                elif current is None:
                    continue
                elif kind == "node":
                    await rebuilder.add_node(value)
                elif kind == "relationship":
                    await rebuilder.add_relationship(value)
        await rebuilder.flush()
        rebuild_seconds = time.perf_counter() - started
    except BaseException:
        await writer.abort()
        raise

    return {
        "nodes_created": rebuilder.nodes_created,
        "nodes_reused": rebuilder.nodes_reused,
        "relationships_created": rebuilder.relationships_created,
        "stale_seconds": stale_seconds,
        "rebuild_seconds": rebuild_seconds,
    }


_DOCUMENT_KEY_FIELDS = ("tenant_id", "path", "version")


def _document_node(props: dict[str, Any]) -> dict[str, Any]:
    """The :Document of a snapshot entry, as a node to MERGE (recreate)."""
    return {
        "labels": ["Document"],
        "key": {name: props[name] for name in _DOCUMENT_KEY_FIELDS},
        "properties": {
            name: value
            for name, value in props.items()
            if name not in _DOCUMENT_KEY_FIELDS
            and name != "frozen"
            and not name.startswith("frozen_")
        },
    }


# ---------------------------------------------------------------------------
# Finishing — temporary links, stubs, snapshot references
# ---------------------------------------------------------------------------

_NO_GC = {
    "gc_entity_model_instance_deleted": 0,
    "gc_entity_model_instance_passes": 0,
    "gc_labeled_entity_deleted": 0,
    "gc_labeled_entity_passes": 0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class _BlobWork:
    """One snapshot to restore from, and what to restore from it."""

    blob_id: str
    # True when the snapshot was reached through frozen stubs (the classic
    # restore); False when looked up / named explicitly (no stub needed).
    from_stubs: bool
    # from_stubs: the frozen stubs to restore from this snapshot
    stubs: list[DocumentInfo] = field(default_factory=list)
    # not from_stubs: the matched keys (lookup), or None = match the selector
    # against the snapshot's entries (explicit frozen_blob_id)
    seed_keys: list[DocKey] | None = None
    # keys another (newer) snapshot restores instead
    exclude: set[DocKey] = field(default_factory=set)
    file: Path | None = None
    scan: SnapshotScan | None = None
    plans: dict[int, _EntryPlan] = field(default_factory=dict)
    seeds: list[DocKey] = field(default_factory=list)
    # not from_stubs: entry index by key, and the keys to restore (seeds and
    # their descendants in the snapshot)
    index: dict[DocKey, int] = field(default_factory=dict)
    targets: list[DocKey] = field(default_factory=list)


def _plan_from_stubs(work: _BlobWork) -> None:
    index = work.scan.entry_index()
    missing = [
        (doc.path, doc.version) for doc in work.stubs if (doc.path, doc.version) not in index
    ]
    if missing:
        raise FreezeError(f"The snapshot has no entry for {missing!r}; nothing was restored.")
    for doc in work.stubs:
        key = (doc.path, doc.version)
        plan = _entry_plan(doc)
        plan.structure_node_ids = work.scan.structure_node_ids.get(key, [])
        work.plans[index[key]] = plan
        if doc.is_seed:
            work.seeds.append(key)


def _plan_from_snapshot(work: _BlobWork, existing: dict[DocKey, DocumentInfo]) -> None:
    """Classify the snapshot's selected entries against the graph: a missing
    Document is recreated, a stub frozen with this snapshot is restored as
    usual, anything else is refused before any write."""
    for key in work.targets:
        entry = work.index[key]
        graph_doc = existing.get(key)
        if graph_doc is None:
            props = work.scan.entries[entry]
            doc = DocumentInfo(
                path=key[0],
                version=key[1],
                tenant_id=props["tenant_id"],
                is_seed=key in work.seeds,
                properties=props,
            )
            work.plans[entry] = _EntryPlan(doc=doc, recreate=True)
        elif graph_doc.frozen and graph_doc.frozen_blob_id == work.blob_id:
            plan = _entry_plan(graph_doc)
            plan.structure_node_ids = work.scan.structure_node_ids.get(key, [])
            work.plans[entry] = plan
        elif graph_doc.frozen:
            raise FreezeError(
                f"restore_document: Document path={key[0]!r} version={key[1]!r} is frozen with "
                f"another snapshot ({graph_doc.frozen_blob_id!r}); restore it without "
                "frozen_blob_id. Nothing was restored."
            )
        else:
            raise FreezeError(
                f"restore_document: Document path={key[0]!r} version={key[1]!r} exists and is "
                "not frozen; delete it first to restore it from a snapshot. Nothing was restored."
            )


def _select_snapshot_targets(work: _BlobWork, filters: dict[str, Any], selector: str) -> None:
    index = work.scan.entry_index()
    if work.seed_keys is None:
        seeds = [
            key for key, i in index.items() if _selector_matches(work.scan.entries[i], filters)
        ]
        if not seeds:
            raise FreezeError(
                f"restore_document: snapshot {work.blob_id!r} has no document matching {selector}."
            )
    else:
        missing = [key for key in work.seed_keys if key not in index]
        if missing:
            raise FreezeError(
                f"Snapshot {work.blob_id!r} has no entry for {missing!r}, although its metadata "
                "lists them; nothing was restored."
            )
        seeds = list(work.seed_keys)
    work.seeds = seeds
    work.index = index
    work.targets = _with_descendants(seeds, work.scan.children, work.exclude)


async def _warn_missing_raw_files(docs: Sequence[DocumentInfo], tenant: str) -> None:
    """A recreated Document may reference a raw file deleted with it
    (``delete_document()`` removes it from storage): warn, never fail."""
    if getattr(get_config(), "storage_backend", "none") == "none":
        return
    try:
        from scinr.newton.storage.factory import get_storage

        raw_files, _ = get_storage()
        for doc in docs:
            raw_file_id = doc.properties.get("raw_file_id")
            if raw_file_id and await raw_files.get(raw_file_id, tenant_id=tenant) is None:
                logger.warning(
                    "restore_document: Document path=%r version=%r restored with raw_file_id=%r, "
                    "which no longer exists in storage (raw files are not part of snapshots).",
                    doc.path,
                    doc.version,
                    raw_file_id,
                )
    except Exception as exc:
        logger.warning(
            "restore_document: could not check the raw files of restored documents: %s", exc
        )


async def restore_document(
    path: str | None = None,
    version: int | None = None,
    *,
    tenant_id: str | None,
    created_by_user_id: str | Sequence[str] | None = None,
    job_id: str | Sequence[str] | None = None,
    frozen_blob_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> RestoreResult:
    """Rebuild document(s) from a snapshot: frozen stubs, deleted documents or backups.

    The selector is the one of ``freeze_document()`` / ``delete_document()``
    (*tenant_id* mandatory, exactly one of *path* / *job_id*,
    *created_by_user_id* as an AND filter). Where the snapshot comes from:

    - **Frozen stubs** (no *frozen_blob_id*, the selector matches Documents
      in the graph): the frozen ones are restored from their own snapshot,
      together with their ``IS_COMPOSED_OF*`` descendants frozen in the same
      snapshot (restoring a folder restores what freezing it froze). If
      none of them is frozen, :class:`FreezeError` — to restore a backup of
      a document that still exists, delete it first.
    - **No Document in the graph** (deleted, or never frozen and only
      backed up): the newest snapshot of the tenant holding each matched
      document is looked up in the freeze backend
      (:meth:`FreezeRepository.find_snapshots`), and the Documents are
      recreated from it — with their links to other Documents (folder,
      version chain) where those still exist, and ``latest`` recomputed.
    - **An explicit** *frozen_blob_id*: that snapshot, whatever the graph
      holds; its entries matching the selector (plus their descendants in
      the snapshot) are restored.

    Either way, a selected Document that exists but is not frozen (or is
    frozen with another snapshot) raises :class:`FreezeError` before any
    write. A snapshot of a freeze is deleted from the backend once no
    Document of the tenant references it; backups and exports are kept.

    Steps: ensure the lookup indexes the rebuild needs → download each
    snapshot (tenant-scoped) → validate it without writing (tenant of
    header, entries, nodes and Document links) → rebuild in batches → drop
    the temporary Document links, unmark the stubs → sweep the tenant's
    orphans if stale ExtractionResults had to be deleted (``graph/neo4j/gc.py``) →
    delete snapshots no longer referenced. The duration of each phase is
    logged. Raw files are not part of snapshots: a recreated Document whose
    raw file was deleted keeps its ``raw_file_id`` and a warning is logged.

    Parameters
    ----------
    frozen_blob_id:
        Restore from this snapshot instead of the stubs' / the newest one.
    batch_size:
        Rows per write transaction of the rebuild. Larger batches mean fewer
        round trips but longer transactions: more graph heap per transaction
        and locks held longer (other writers — e.g. a running ingestion —
        wait on the shared catalog / entity nodes meanwhile).
    concurrency:
        Rebuild transactions in flight at once (each on its own pooled
        connection). Lower it (e.g. ``1``) while an ingestion is writing to
        the same graph database; graph heap in use by the rebuild grows roughly with
        ``batch_size × concurrency``.

    Returns
    -------
    RestoreResult
        ``found=False`` (nothing written) when neither a Document nor a
        snapshot matches.

    Raises
    ------
    TypeError
        If *tenant_id* is not passed.
    ValueError
        Invalid selector, or *batch_size* / *concurrency* below 1.
    FreezeError
        Nothing restorable (matched Documents not frozen), a selected
        Document exists unfrozen, a snapshot is missing / belongs to another
        tenant / is malformed, or the backend cannot look snapshots up.
    ConfigurationError
        If the freeze backend resolves to ``"none"``.
    GraphError
        If the graph engine fails (after the retries).
    """
    filters = selector_filters(
        "restore_document", path, version, tenant_id, created_by_user_id, job_id
    )
    if batch_size < 1 or concurrency < 1:
        raise ValueError(
            f"batch_size and concurrency must be >= 1 (got {batch_size}, {concurrency})."
        )
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
    not_found = RestoreResult(
        **result_scope, found=False, versions_restored=[], documents_restored=0
    )
    from scinr.newton.freeze.factory import get_freeze_storage

    store = get_graph_store()
    with store.sync_scope():
        docs = await asyncio.to_thread(fetch_document_set, filters)
        seeds = [doc for doc in docs if doc.is_seed]
        repository = get_freeze_storage()
        works: list[_BlobWork] = []

        if frozen_blob_id is None and seeds:
            # Classic restore: through the frozen stubs.
            # frozen / frozen_blob_id / frozen_keep_* are the stub's raw properties.
            frozen_seeds = [doc for doc in seeds if doc.frozen]
            if not frozen_seeds:
                raise FreezeError(
                    f"restore_document: nothing to restore for {selector} — the matched "
                    "document(s) exist and are not frozen. To restore a backup of a document "
                    "that still exists, delete it first."
                )
            if any(not doc.frozen_blob_id for doc in frozen_seeds):
                raise FreezeError("restore_document: a frozen Document has no frozen_blob_id.")
            blob_ids = sorted({doc.frozen_blob_id for doc in frozen_seeds})
            targets = frozen_seeds + [
                doc
                for doc in docs
                if not doc.is_seed and doc.frozen and doc.frozen_blob_id in blob_ids
            ]
            for blob_id in blob_ids:
                works.append(
                    _BlobWork(
                        blob_id=blob_id,
                        from_stubs=True,
                        stubs=[doc for doc in targets if doc.frozen_blob_id == blob_id],
                    )
                )
        elif frozen_blob_id is not None:
            works.append(_BlobWork(blob_id=frozen_blob_id, from_stubs=False))
        else:
            # No Document in the graph: the newest snapshot holding each one.
            records = await repository.find_snapshots(
                tenant_id=tenant,
                path=filters["path"],
                version=filters["version"],
                job_id=filters["job_id"],
                created_by_user_id=filters["created_by_user_id"],
            )
            assigned: dict[DocKey, str] = {}
            for record in records:
                for meta in record.documents:
                    key = (meta.get("path"), meta.get("version"))
                    if _selector_matches(meta, filters):
                        assigned.setdefault(key, record.frozen_blob_id)
            if not assigned:
                logger.warning(
                    "restore_document: no Document nor snapshot found for %s; nothing to restore.",
                    selector,
                )
                return not_found
            for record in records:
                keys = [key for key, blob in assigned.items() if blob == record.frozen_blob_id]
                if keys:
                    works.append(
                        _BlobWork(
                            blob_id=record.frozen_blob_id,
                            from_stubs=False,
                            seed_keys=keys,
                            exclude={k for k, b in assigned.items() if b != record.frozen_blob_id},
                        )
                    )

        started = time.perf_counter()
        await asyncio.to_thread(store.schema.ensure_indexes, _RESTORE_INDEXES)
        timings["indexes"] = time.perf_counter() - started

        counters = {"nodes_created": 0, "nodes_reused": 0, "relationships_created": 0}
        with tempfile.TemporaryDirectory(prefix="scinr-restore-") as tmp_dir:
            # Download, validate and plan every snapshot before writing anything.
            timings["download"] = timings["validate"] = 0.0
            for index, work in enumerate(works):
                if not work.from_stubs:
                    work.stubs = await asyncio.to_thread(
                        store.restore.frozen_stubs_of_blob, tenant, work.blob_id
                    )
                structure_ids_for = {
                    (doc.path, doc.version)
                    for doc in work.stubs
                    if _entry_plan(doc).needs_structure_ids
                }

                work.file = Path(tmp_dir) / f"snapshot-{index}.json"
                started = time.perf_counter()
                found = await repository.read_snapshot_to_file(
                    work.blob_id, work.file, tenant_id=tenant
                )
                timings["download"] += time.perf_counter() - started
                if not found:
                    raise FreezeError(
                        f"restore_document: snapshot frozen_blob_id={work.blob_id!r} not found in "
                        f"the freeze backend for tenant {tenant!r}; nothing was restored."
                    )

                started = time.perf_counter()
                work.scan = await asyncio.to_thread(
                    scan_snapshot, work.file, tenant, structure_ids_for=structure_ids_for
                )
                if work.from_stubs:
                    _plan_from_stubs(work)
                else:
                    _select_snapshot_targets(work, filters, selector)
                    existing = await asyncio.to_thread(
                        store.restore.existing_documents,
                        tenant,
                        [{"path": p, "version": v} for p, v in work.targets],
                    )
                    _plan_from_snapshot(work, {(doc.path, doc.version): doc for doc in existing})
                timings["validate"] += time.perf_counter() - started

            # Read before the rebuild deletes them: see the GC below.
            needs_gc = await asyncio.to_thread(
                store.restore.has_stale_extraction_results,
                [
                    plan.structure_node_ids
                    for work in works
                    for plan in work.plans.values()
                    if plan.delete_stale_extraction_results
                ],
            )

            timings["stale_delete"] = timings["rebuild"] = 0.0
            for work in works:
                written = await _rebuild_snapshot(
                    work.file,
                    work.plans,
                    batch_size=batch_size,
                    concurrency=concurrency,
                )
                timings["stale_delete"] += written.pop("stale_seconds")
                timings["rebuild"] += written.pop("rebuild_seconds")
                for name, value in written.items():
                    counters[name] += value

        plans = [plan for work in works for plan in work.plans.values()]
        restored = [plan.doc for plan in plans]
        recreated = [plan.doc for plan in plans if plan.recreate]
        started = time.perf_counter()
        await asyncio.to_thread(
            store.restore.finish_restore,
            tenant,
            restored,
            sorted({doc.path for doc in recreated}),
        )
        timings["finish"] = time.perf_counter() - started
        # A restore only creates, except for the stale ExtractionResults it
        # removes from kept StructureNodes (something extracted onto a frozen
        # document): without those nothing can have been orphaned, and the
        # GC — a scan of every :Entity / :ModelInstance of the tenant — is
        # skipped.
        started = time.perf_counter()
        gc_counts = dict(_NO_GC)
        if needs_gc:
            gc_counts = await asyncio.to_thread(store.gc.sweep_tenant, tenant, "restore_document")
        timings["gc"] = time.perf_counter() - started
        await _warn_missing_raw_files(recreated, tenant)

        snapshots_deleted = 0
        for work in works:
            mode = work.scan.header.get("mode")
            if not work.from_stubs and mode != "freeze":
                logger.info(
                    "restore_document: snapshot %s kept (mode=%r: backups and exports are kept).",
                    work.blob_id,
                    mode,
                )
                continue
            remaining = await asyncio.to_thread(
                store.restore.count_blob_references, tenant, work.blob_id
            )
            if remaining == 0:
                await repository.delete_snapshot(work.blob_id, tenant_id=tenant)
                snapshots_deleted += 1
            else:
                logger.info(
                    "restore_document: snapshot %s kept — still referenced by %d frozen "
                    "Document(s) of tenant %r.",
                    work.blob_id,
                    remaining,
                    tenant,
                )

        logger.info(
            "restore_document: complete for %s. documents_restored=%d documents_recreated=%d "
            "nodes_created=%d nodes_reused=%d relationships_created=%d snapshots_deleted=%d "
            "(batch_size=%d, concurrency=%d) timings: %s",
            selector,
            len(restored),
            len(recreated),
            counters["nodes_created"],
            counters["nodes_reused"],
            counters["relationships_created"],
            snapshots_deleted,
            batch_size,
            concurrency,
            ", ".join(f"{phase}={seconds:.2f}s" for phase, seconds in timings.items()),
        )
        return RestoreResult(
            **result_scope,
            found=True,
            versions_restored=sorted({key[1] for work in works for key in work.seeds}),
            documents_restored=len(restored),
            documents_recreated=len(recreated),
            frozen_blob_ids=[work.blob_id for work in works],
            snapshots_deleted=snapshots_deleted,
            **counters,
            **gc_counts,
        )
