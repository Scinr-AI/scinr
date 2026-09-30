"""
ingest/restore.py — Rebuild frozen documents from their snapshot.

The inverse of :func:`~scinr.newton.ingest.freeze.freeze_document`: download
the snapshot from the freeze backend, validate it (tenant, format) in a first
streaming pass, rebuild the subtree in batches in a second one, remove the
temporary ``(:Document)-[:HAS_MODEL_DECISION|HAS_EXTRACTION]->()`` links,
unmark the stubs and delete the snapshot once no Document references it.
Every write runs in a bounded transaction (row batches for the rebuild, 50
documents at a time for the finish), so Neo4j's transaction memory depends on
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
from collections.abc import Awaitable, Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import ijson

from scinr.newton.annotation.neo4j_ops import delete_stale_model_decision
from scinr.newton.config import get_config
from scinr.newton.entity_extraction.graph_mapper import delete_stale_extraction_result
from scinr.newton.exceptions import FreezeError
from scinr.newton.ingest._cascade import DELETE_BATCH_ROWS, DOCUMENTS_PER_QUERY, run_count_query
from scinr.newton.ingest._gc import sweep_tenant
from scinr.newton.ingest.config import get_async_driver, get_driver
from scinr.newton.ingest.freeze import (
    _DOCS_BY_KEY,
    SNAPSHOT_SCHEMA_VERSION,
    DocumentInfo,
    cypher_identifier,
    fetch_document_set,
    is_catalog,
    node_identity,
    selector_filters,
    selector_repr,
)
from scinr.newton.ingest.schema import ensure_indexes
from scinr.newton.results import RestoreResult
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync

logger = logging.getLogger(__name__)

# Nodes merged across documents of a tenant (business-key uid): on match they
# only gain the snapshot's provenance.
_MERGED_LABELS = frozenset({"ModelInstance", "Entity", "LabeledEntity"})

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
        cypher_identifier(label)
    key = endpoint["key"]
    if any(key.get(f) is None for f in key_fields):
        raise FreezeError(f"Relationship endpoint without its {key_fields!r} key: {endpoint!r}")
    if primary == "Document" and key["tenant_id"] != tenant:
        raise FreezeError(
            f"Snapshot relationship points at Document {key!r} of another tenant; "
            "nothing was restored."
        )
    return primary, key


def _check_relationship(value: dict[str, Any], tenant: str, doc_key: tuple | None) -> None:
    """A relationship may touch a :Document only if it is the entry's own one;
    between two Documents (folder / version links) one side must be it."""
    rel_type = value.get("type")
    cypher_identifier(rel_type)
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
                    cypher_identifier(label)
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
                _check_relationship(value, tenant, doc_key)
                if value.get("type") == "IS_COMPOSED_OF":
                    parent, child = value["from"]["key"], value["to"]["key"]
                    children = scan.children.setdefault((parent["path"], parent["version"]), [])
                    if (child["path"], child["version"]) not in children:
                        children.append((child["path"], child["version"]))

    if scan is None:
        raise FreezeError("Malformed snapshot: no 'documents' array.")
    return scan


def _selector_matches(props: dict[str, Any], filters: dict[str, Any]) -> bool:
    """The restore selector (``_build_doc_match`` filters) applied to one
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
# Pass 2 — rebuild (async driver, batches)
# ---------------------------------------------------------------------------


def _key_map(source: str, key_fields: Sequence[str]) -> str:
    # key_fields come from node_identity() — fixed names, never snapshot data.
    return ", ".join(f"{name}: {source}.{name}" for name in key_fields)


def _union(field_name: str) -> str:
    return (
        f"n.{field_name} = reduce(acc = coalesce(n.{field_name}, []), "
        f"x IN coalesce(row.props.{field_name}, []) | "
        "CASE WHEN x IN acc THEN acc ELSE acc + x END)"
    )


@lru_cache(maxsize=256)
def _node_merge_query(labels: tuple[str, ...]) -> str:
    """``MERGE`` a batch of nodes sharing *labels*; props only on create."""
    primary, key_fields = node_identity(labels)
    extra_labels = "".join(
        f", n:{cypher_identifier(label)}" for label in labels if label != primary
    )
    query = (
        "UNWIND $rows AS row\n"
        f"MERGE (n:{cypher_identifier(primary)} {{{_key_map('row.key', key_fields)}}})\n"
        f"ON CREATE SET n += row.props{extra_labels}\n"
    )
    if primary in _MERGED_LABELS:
        query += f"ON MATCH SET {_union('created_by_user_ids')}, {_union('job_ids')}\n"
    return query


@lru_cache(maxsize=1024)
def _relationship_query(rel_type: str, from_label: str, to_label: str) -> str:
    """Create a batch of relationships unless an identical one exists.

    Catalog targets are ``MERGE``d by their key (like the pipeline does).
    """
    _, from_fields = node_identity([from_label])
    _, to_fields = node_identity([to_label])
    rel = cypher_identifier(rel_type)
    target = "MERGE" if is_catalog(to_label) else "MATCH"
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (a:{cypher_identifier(from_label)} {{{_key_map('row.from', from_fields)}}})\n"
        f"{target} (b:{cypher_identifier(to_label)} {{{_key_map('row.to', to_fields)}}})\n"
        "WITH a, b, row\n"
        f"WHERE NOT EXISTS {{ MATCH (a)-[x:{rel}]->(b) WHERE properties(x) = row.props }}\n"
        f"CREATE (a)-[r:{rel}]->(b)\n"
        "SET r = row.props\n"
    )


async def _run_batch(tx, query: str, rows: list[dict[str, Any]]):
    result = await tx.run(query, rows=rows)
    summary = await result.consume()
    return summary.counters


async def _delete_stale(tx, node_ids: list[str], model_decisions: bool, extraction_results: bool):
    if extraction_results:
        await delete_stale_extraction_result(tx, node_ids)
    if model_decisions:
        await delete_stale_model_decision(tx, node_ids)


def _sort_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class _BatchWriter:
    """Runs write transactions with at most *concurrency* in flight.

    Each batch runs in its own session (an async session is not safe for
    concurrent use; sessions borrow pooled connections). Batches of the same
    *lane* run one after the other, lanes in parallel. The producer waits for
    a free slot **before** handing over a batch, so a fast reader never piles
    up more than *concurrency* batches in memory.
    """

    def __init__(self, driver, database: str, concurrency: int) -> None:
        self._driver = driver
        self._database = database
        self._slots = asyncio.Semaphore(concurrency)
        self._lanes: dict[Hashable, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()
        self._error: BaseException | None = None

    async def _run(
        self,
        lane: Hashable,
        work: Callable[..., Awaitable[Any]],
        args: tuple,
        on_done: Callable[[Any], None],
    ) -> None:
        try:
            async with self._lanes.setdefault(lane, asyncio.Lock()):
                async with self._driver.session(database=self._database) as session:
                    on_done(await session.execute_write(work, *args))
        except Exception as exc:
            # Kept for drain() to raise: the task itself never fails, so a
            # finished task nobody awaits leaves no "exception never retrieved".
            if self._error is None:
                self._error = exc
        finally:
            self._slots.release()

    async def submit(
        self,
        lane: Hashable,
        work: Callable[..., Awaitable[Any]],
        *args,
        on_done: Callable[[Any], None] = lambda _: None,
    ) -> None:
        if self._error is not None:
            await self.drain()
        await self._slots.acquire()
        task = asyncio.create_task(self._run(lane, work, args, on_done))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for every batch in flight; re-raise the first failure."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._error is not None:
            raise self._error

    async def abort(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


class _Rebuilder:
    """Groups snapshot elements into batches and hands them to a _BatchWriter.

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

    def __init__(self, writer: _BatchWriter, batch_size: int) -> None:
        self.writer = writer
        self.batch_size = batch_size
        self.nodes: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        self.relationships: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self.nodes_created = 0
        self.nodes_reused = 0
        self.relationships_created = 0
        self._nodes_in_flight = False

    def _count_nodes(self, rows: int) -> Callable[[Any], None]:
        def _done(counters) -> None:
            self.nodes_created += counters.nodes_created
            self.nodes_reused += rows - counters.nodes_created

        return _done

    def _count_relationships(self, counters) -> None:
        self.relationships_created += counters.relationships_created

    async def _flush_node_group(self, labels: tuple[str, ...]) -> None:
        rows = self.nodes.pop(labels, None)
        if rows:
            rows.sort(key=lambda row: _sort_key(row["key"]))
            self._nodes_in_flight = True
            await self.writer.submit(
                ("nodes", labels),
                _run_batch,
                _node_merge_query(labels),
                rows,
                on_done=self._count_nodes(len(rows)),
            )

    async def _flush_relationship_group(self, group: tuple[str, str, str]) -> None:
        rows = self.relationships.pop(group, None)
        if rows:
            rows.sort(key=lambda row: (_sort_key(row["from"]), _sort_key(row["to"])))
            lane = ("catalog",) if is_catalog(group[2]) else ("relationships", group)
            await self.writer.submit(
                lane,
                _run_batch,
                _relationship_query(*group),
                rows,
                on_done=self._count_relationships,
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
    writer: _BatchWriter, plans: dict[int, _EntryPlan], batch_size: int
) -> None:
    """Stale ModelDecision / ExtractionResult subtrees on kept StructureNodes,
    removed before any snapshot node is merged (a merge could otherwise reuse
    a stale node that is about to be deleted). One lane: the deletes of two
    batches can meet on a shared (tabular) ModelDecision."""
    for plan in plans.values():
        ids = plan.structure_node_ids
        for start in range(0, len(ids), batch_size):
            await writer.submit(
                ("stale",),
                _delete_stale,
                ids[start : start + batch_size],
                plan.delete_stale_model_decisions,
                plan.delete_stale_extraction_results,
            )
    await writer.drain()


async def _rebuild_snapshot(
    path: Path,
    plans: dict[int, _EntryPlan],
    database: str,
    *,
    batch_size: int,
    concurrency: int,
) -> dict[str, int]:
    """Second pass: write the planned entries of the snapshot at *path*."""
    writer = _BatchWriter(get_async_driver(), database, concurrency)
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

# A Document never has outgoing HAS_MODEL_DECISION / HAS_EXTRACTION outside
# a freeze, so any such link on a restored stub is a temporary one. There is
# one per kept ModelDecision / ExtractionResult, so they go in batches (an
# auto-commit query, like the deletes of ingest/_cascade.py); grouping by the
# link makes the query read them all before the first batch commits.
_TEMPORARY_LINKS_DELETE_QUERY = (
    _DOCS_BY_KEY
    + f"""MATCH (d)-[link:HAS_MODEL_DECISION|HAS_EXTRACTION]->()
WITH link, count(*) AS hits
CALL (link) {{
  DELETE link
}} IN TRANSACTIONS OF {DELETE_BATCH_ROWS} ROWS
RETURN count(*) AS n
"""
)

_RESTORE_FINISH_QUERY = (
    _DOCS_BY_KEY
    + """SET d.frozen = false
REMOVE d.frozen_blob_id, d.frozen_at, d.frozen_keep_structure_nodes,
       d.frozen_keep_annotations, d.frozen_keep_extraction_results,
       d.frozen_cleanup_pending
RETURN count(d) AS n
"""
)

# ``latest`` is the newest version of a path: a recreated Document may be an
# older version than one ingested since, or the newest one again.
_RECOMPUTE_LATEST_QUERY = """
UNWIND $paths AS p
MATCH (d:Document {tenant_id: $tenant_id, path: p})
WITH p, collect(d) AS versions, max(d.version) AS newest
UNWIND versions AS d
SET d.latest = (d.version = newest)
"""

_EXISTING_DOCUMENTS_QUERY = _DOCS_BY_KEY + "RETURN properties(d) AS properties\n"

_STUBS_OF_BLOB_QUERY = """
MATCH (d:Document {tenant_id: $tenant_id})
WHERE d.frozen = true AND d.frozen_blob_id = $blob_id
RETURN properties(d) AS properties
"""

_BLOB_REFERENCES_QUERY = """
MATCH (d:Document {tenant_id: $tenant_id})
WHERE d.frozen_blob_id = $blob_id
RETURN count(d) AS n
"""


def _finish_restore(
    driver, database: str, tenant: str, docs: Sequence[DocumentInfo], recreated_paths: list[str]
) -> None:
    """Remove the temporary links and unmark the stubs, ``DOCUMENTS_PER_QUERY``
    documents at a time, then recompute ``latest`` for the recreated paths.

    Not one transaction: a freeze that kept annotations or extraction results
    leaves one link per kept node, and deleting those of hundreds of
    documents at once is what a transaction memory pool cannot hold. A
    document is unmarked only once its links are gone; if this stops
    half-way the remaining documents are still frozen stubs, so calling
    ``restore_document()`` again finishes them.
    """
    keys = [doc.key for doc in docs]
    for start in range(0, len(keys), DOCUMENTS_PER_QUERY):
        chunk = keys[start : start + DOCUMENTS_PER_QUERY]
        run_count_query(
            driver, database, _TEMPORARY_LINKS_DELETE_QUERY, tenant_id=tenant, keys=chunk
        )

        def _do_unmark(chunk: list[dict[str, Any]] = chunk) -> None:
            with driver.session(database=database) as session:
                session.execute_write(
                    lambda tx: tx.run(_RESTORE_FINISH_QUERY, tenant_id=tenant, keys=chunk).consume()
                )

        with_neo4j_retry_sync(_do_unmark)

    if recreated_paths:

        def _do_latest() -> None:
            with driver.session(database=database) as session:
                session.execute_write(
                    lambda tx: tx.run(
                        _RECOMPUTE_LATEST_QUERY, tenant_id=tenant, paths=recreated_paths
                    ).consume()
                )

        with_neo4j_retry_sync(_do_latest)


_NO_GC = {
    "gc_entity_model_instance_deleted": 0,
    "gc_entity_model_instance_passes": 0,
    "gc_labeled_entity_deleted": 0,
    "gc_labeled_entity_passes": 0,
}

_STALE_EXTRACTION_RESULTS_QUERY = """
UNWIND $node_ids AS nid
MATCH (:StructureNode {id: nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
RETURN count(er) AS n
"""


def _has_stale_extraction_results(driver, database: str, plans: Sequence[_EntryPlan]) -> bool:
    """Whether any kept StructureNode still carries an ExtractionResult the
    restore is about to delete — the only thing a restore deletes that can
    orphan an :Entity / :ModelInstance / :LabeledEntity."""
    for plan in plans:
        if not plan.delete_stale_extraction_results:
            continue
        ids = plan.structure_node_ids
        for start in range(0, len(ids), DELETE_BATCH_ROWS):
            batch = ids[start : start + DELETE_BATCH_ROWS]
            if run_count_query(driver, database, _STALE_EXTRACTION_RESULTS_QUERY, node_ids=batch):
                return True
    return False


def _read_documents(driver, database: str, query: str, **params) -> list[DocumentInfo]:
    def _do_read() -> list[DocumentInfo]:
        with driver.session(database=database) as session:
            return [
                DocumentInfo(
                    path=record["properties"].get("path"),
                    version=record["properties"].get("version"),
                    tenant_id=record["properties"].get("tenant_id"),
                    is_seed=False,
                    properties=dict(record["properties"]),
                )
                for record in session.run(query, **params)
            ]

    return with_neo4j_retry_sync(_do_read)


def _count_blob_references(driver, database: str, tenant: str, blob_id: str) -> int:
    def _do_count() -> int:
        with driver.session(database=database) as session:
            record = session.run(_BLOB_REFERENCES_QUERY, tenant_id=tenant, blob_id=blob_id).single()
            return record["n"] if record else 0

    return with_neo4j_retry_sync(_do_count)


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
    orphans if stale ExtractionResults had to be deleted (``ingest/_gc.py``) →
    delete snapshots no longer referenced. The duration of each phase is
    logged. Raw files are not part of snapshots: a recreated Document whose
    raw file was deleted keeps its ``raw_file_id`` and a warning is logged.

    Parameters
    ----------
    frozen_blob_id:
        Restore from this snapshot instead of the stubs' / the newest one.
    batch_size:
        Rows per write transaction of the rebuild. Larger batches mean fewer
        round trips but longer transactions: more Neo4j heap per transaction
        and locks held longer (other writers — e.g. a running ingestion —
        wait on the shared catalog / entity nodes meanwhile).
    concurrency:
        Rebuild transactions in flight at once (each on its own pooled
        connection). Lower it (e.g. ``1``) while an ingestion is writing to
        the same Neo4j; Neo4j heap in use by the rebuild grows roughly with
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
    database = get_config().neo4j_database

    from scinr.newton.freeze.factory import get_freeze_storage

    driver = get_driver()
    try:
        docs = await asyncio.to_thread(fetch_document_set, driver, database, filters)
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
        await asyncio.to_thread(ensure_indexes, driver, _RESTORE_INDEXES, database=database)
        timings["indexes"] = time.perf_counter() - started

        counters = {"nodes_created": 0, "nodes_reused": 0, "relationships_created": 0}
        with tempfile.TemporaryDirectory(prefix="scinr-restore-") as tmp_dir:
            # Download, validate and plan every snapshot before writing anything.
            timings["download"] = timings["validate"] = 0.0
            for index, work in enumerate(works):
                if not work.from_stubs:
                    work.stubs = await asyncio.to_thread(
                        _read_documents,
                        driver,
                        database,
                        _STUBS_OF_BLOB_QUERY,
                        tenant_id=tenant,
                        blob_id=work.blob_id,
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
                        _read_documents,
                        driver,
                        database,
                        _EXISTING_DOCUMENTS_QUERY,
                        tenant_id=tenant,
                        keys=[{"path": p, "version": v} for p, v in work.targets],
                    )
                    _plan_from_snapshot(work, {(doc.path, doc.version): doc for doc in existing})
                timings["validate"] += time.perf_counter() - started

            # Read before the rebuild deletes them: see the GC below.
            needs_gc = await asyncio.to_thread(
                _has_stale_extraction_results,
                driver,
                database,
                [plan for work in works for plan in work.plans.values()],
            )

            timings["stale_delete"] = timings["rebuild"] = 0.0
            for work in works:
                written = await _rebuild_snapshot(
                    work.file,
                    work.plans,
                    database,
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
            _finish_restore,
            driver,
            database,
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
            gc_counts = await asyncio.to_thread(
                sweep_tenant,
                driver,
                tenant_id=tenant,
                database=database,
                caller="restore_document",
            )
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
                _count_blob_references, driver, database, tenant, work.blob_id
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
    finally:
        driver.close()
