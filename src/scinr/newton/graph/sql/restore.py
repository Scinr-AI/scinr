"""
graph/sql/restore.py — ``store.restore`` over SQL: batch writer, finishing statements, reads.

Translation of ``graph/neo4j/restore.py``. What goes in each batch, and in which order, is
decided by the neutral ``_Rebuilder`` of ``ingest/restore.py``; this module only runs the
batches:

- :class:`SqlRestoreWriter` — runs write transactions on the asynchronous engine with at most
  ``dialect.max_concurrent_writes(concurrency)`` in flight (SQLite: one); batches of the same
  *lane* run one after the other. The producer waits for a free slot **before** handing over
  a batch, so memory stays ``O(batch_size × concurrency)``.
- the statements that finish a restore (temporary links, unmarking the stubs, ``latest``),
  the snapshot-reference count and the stale-extraction check.

``_document_key`` of the restored nodes: the rows of ``merge_nodes`` carry no document (the
interface, and the Neo4j queries, only know ``{"key", "props"}``), so a per-document node is
created with ``_document_key`` (and an InfoUnit with ``_structure_node_id``) unset, and
:func:`assign_document_keys` fills them in from the ownership relationships
(``HAS_STRUCTURE``, ``HAS_CHILD``, ``HAS_INFO_UNIT``…) when the writer is drained after
relationship batches. A restore drains before it finishes, so no row is left without them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Hashable, Iterable, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from scinr.newton.graph.base import RestoreOps, RestoreWriter
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._shared import (
    delete_stale_extraction_result,
    delete_stale_model_decision,
)
from scinr.newton.graph.sql.deletion import document_keys
from scinr.newton.graph.sql.freeze import document_info
from scinr.newton.graph.types import (
    DELETE_BATCH_ROWS,
    DOCUMENTS_PER_QUERY,
    DocumentInfo,
    RestoreCounters,
)
from scinr.newton.ingest.freeze import is_catalog, node_identity, validate_identifier
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import async_sql_errors, sql_errors
from scinr.newton.sql.keys import encode_key
from scinr.newton.sql.names import KEY_COLUMN, LABEL_TABLE, PER_DOCUMENT_TABLES
from scinr.newton.sql.retry import with_sql_retry
from scinr.newton.sql.rows import props_to_row
from scinr.newton.sql.tables import LIST_COLUMNS

__all__ = ["SqlRestoreOps", "SqlRestoreWriter", "assign_document_keys"]

logger = logging.getLogger(__name__)

D = t.documents
E = t.edges

#: Nodes merged across documents of a tenant (business-key uid): on match they only gain
#: the snapshot's provenance (``_MERGED_LABELS`` of the Neo4j store).
_MERGED_LABELS = frozenset({"ModelInstance", "Entity", "LabeledEntity"})
_UNION_FIELDS = ("created_by_user_ids", "job_ids")

#: Bound parameters per statement (SQLite allows 32766, Postgres 65535).
_MAX_BINDS = 30_000

#: Bound of the ``HAS_CHILD`` propagation of :func:`assign_document_keys` (tree depth).
_MAX_TREE_DEPTH = 1000

#: ``(child label, relationship type, owner label)``: the relationships a per-document node
#: hangs from, in the order :func:`assign_document_keys` follows them.
_OWNERSHIP: tuple[tuple[str, str, str], ...] = (
    ("StructureNode", "HAS_STRUCTURE", "Document"),
    ("StructureNode", "HAS_CHILD", "StructureNode"),
    ("InfoUnit", "HAS_INFO_UNIT", "StructureNode"),
    ("ModelDecision", "HAS_MODEL_DECISION", "StructureNode"),
    ("ModelDecision", "HAS_MODEL_DECISION", "Document"),
    ("ProposedModel", "HAS_PROPOSED_MODEL", "ModelDecision"),
    ("ComplementaryMatch", "HAS_COMPLEMENTARY_MATCH", "ModelDecision"),
    ("SupplementaryField", "HAS_SUPPLEMENTARY_FIELD", "ModelDecision"),
    ("ProposedField", "HAS_PROPOSED_FIELD", "ProposedModel"),
    ("ExtractionResult", "HAS_EXTRACTION", "StructureNode"),
    ("ExtractionResult", "HAS_EXTRACTION", "Document"),
)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _chunks(rows: Sequence[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    """Slices of *rows* small enough for the bound-parameter limit of one statement."""
    if not rows:
        return []
    per_statement = max(1, _MAX_BINDS // max(1, len(rows[0])))
    return c.batched(list(rows), per_statement)


def _uniform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows with the same columns each (``None`` where a row has no value), for a
    multi-row ``VALUES``."""
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    return [{name: row.get(name) for name in columns} for row in rows]


def _node_row(label: str, key: dict[str, Any], props: dict[str, Any]) -> dict[str, Any]:
    """The row of a node created by ``MERGE (n {key}) ON CREATE SET n += props``."""
    target = LABEL_TABLE[label]
    row = props_to_row(target, {**key, **props})
    for name in LIST_COLUMNS.get(target, frozenset()):
        if name in row and not isinstance(row[name], list):
            row[name] = list(row[name])
    if KEY_COLUMN[label] == "_key":
        row["_key"] = encode_key(label, key)
    return row


def _union(current: list[Any] | None, added: Any) -> list[Any]:
    """``reduce(acc = coalesce(n.x, []), x IN coalesce(row.props.x, []) | CASE WHEN x IN acc
    THEN acc ELSE acc + x END)``."""
    result = list(current or [])
    for item in added or []:
        if item not in result:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Batch work (each runs inside one transaction)
# ---------------------------------------------------------------------------


async def _delete_stale(
    connection: AsyncConnection,
    node_ids: list[str],
    model_decisions: bool,
    extraction_results: bool,
) -> None:
    """Cypher: ``delete_stale_extraction_result`` then ``delete_stale_model_decision``, in one
    transaction."""
    if extraction_results:
        await delete_stale_extraction_result(connection, node_ids)
    if model_decisions:
        await delete_stale_model_decision(connection, node_ids)


async def _existing_lists(
    connection: AsyncConnection, label: str, keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """``{key: {list field: value}}`` of the existing nodes among *keys* (locked for update
    where the engine has row locks)."""
    target, key_column = c.table(label), c.key_column(label)
    columns = [target.c[name] for name in _UNION_FIELDS] if label in _MERGED_LABELS else []
    found: dict[str, dict[str, Any]] = {}
    for batch in c.batched(list(keys), c.IN_BATCH):
        statement = sa.select(key_column, *columns).where(key_column.in_(batch)).with_for_update()
        for row in (await connection.execute(statement)).all():
            found[row[0]] = {name: row[i + 1] for i, name in enumerate(_UNION_FIELDS) if columns}
    return found


async def _merge_nodes(
    connection: AsyncConnection, labels: tuple[str, ...], rows: list[dict[str, Any]]
) -> RestoreCounters:
    """Cypher: ``UNWIND $rows MERGE (n:<primary> {key}) ON CREATE SET n += row.props`` and, for
    the merged labels, ``ON MATCH SET`` the union of ``created_by_user_ids`` / ``job_ids``.

    The rows are applied in order, as ``UNWIND`` does: the first row of a new key creates the
    node, every later row of the same key (and every row of an existing key) is a match.
    """
    primary, key_fields = node_identity(labels)
    merged = primary in _MERGED_LABELS
    # key -> (key properties, the props of every row of that key, in order)
    by_key: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for row in rows:
        key = {name: row["key"][name] for name in key_fields}
        props = dict(row.get("props") or {})
        by_key.setdefault(encode_key(primary, key), (key, []))[1].append(props)

    existing = await _existing_lists(connection, primary, list(by_key))
    new_keys = [key for key in by_key if key not in existing]

    # A new key: its first row creates the node; with the later ones as matches (lists).
    inserts: list[dict[str, Any]] = []
    for encoded in new_keys:
        key, all_props = by_key[encoded]
        props = dict(all_props[0])
        if merged and len(all_props) > 1:
            for name in _UNION_FIELDS:
                value = props.get(name)
                for later in all_props[1:]:
                    value = _union(value, later.get(name))
                props[name] = value
        row = _node_row(primary, key, props)
        if LABEL_TABLE[primary] in PER_DOCUMENT_TABLES and primary != "Document":
            row["_document_key"] = None  # filled in by assign_document_keys()
        inserts.append(row)

    target, key_column = c.table(primary), c.key_column(primary)
    created: set[str] = set()
    for batch in _chunks(_uniform(inserts)):
        statement = (
            c.dialect()
            .insert(target)
            .values(batch)
            .on_conflict_do_nothing(index_elements=[KEY_COLUMN[primary]])
            .returning(key_column)
        )
        created.update(c.column_values(await connection.execute(statement)))

    if merged:
        # A key created meanwhile by another writer: every row of it is a match.
        raced = [key for key in new_keys if key not in created]
        if raced:
            existing.update(await _existing_lists(connection, primary, raced))
        updates = []
        for encoded, current in existing.items():
            values = dict(current)
            for props in by_key[encoded][1]:
                for name in _UNION_FIELDS:
                    values[name] = _union(values.get(name), props.get(name))
            if values != current:
                updates.append({"_k": encoded, **{f"_{n}": values[n] for n in _UNION_FIELDS}})
        if updates:
            statement = (
                sa.update(target)
                .where(key_column == sa.bindparam("_k"))
                .values(
                    {
                        name: sa.bindparam(f"_{name}", type_=target.c[name].type)
                        for name in _UNION_FIELDS
                    }
                )
            )
            await connection.execute(statement, updates)
    return RestoreCounters(nodes_created=len(created))


async def _tenants(
    connection: AsyncConnection, label: str, keys: Iterable[str]
) -> dict[str, str | None]:
    """``{key: tenant_id}`` of the existing nodes of *label* among *keys* (``None`` for
    catalog nodes, which have no tenant)."""
    target, key_column = c.table(label), c.key_column(label)
    tenant = target.c.tenant_id if "tenant_id" in target.c else sa.null()
    found: dict[str, str | None] = {}
    for batch in c.batched(sorted(set(keys)), c.IN_BATCH):
        statement = sa.select(key_column, tenant).where(key_column.in_(batch))
        found.update({row[0]: row[1] for row in (await connection.execute(statement)).all()})
    return found


async def _create_relationships(
    connection: AsyncConnection, group: tuple[str, str, str], rows: list[dict[str, Any]]
) -> RestoreCounters:
    """Cypher: ``UNWIND $rows MATCH (a {from}) MERGE|MATCH (b {to}) WHERE NOT EXISTS
    { (a)-[x:R]->(b) WHERE properties(x) = row.props } CREATE (a)-[r:R]->(b) SET r = row.props``.

    The source must exist; a catalog target is created if missing (``MERGE``), any other
    must exist. A relationship already there with the same identity is left alone
    (``ON CONFLICT DO NOTHING``; difference D1, see :class:`SqlRestoreOps`).
    """
    rel_type, from_label, to_label = group
    _, from_fields = node_identity([from_label])
    _, to_fields = node_identity([to_label])
    wanted: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = []
    for row in rows:
        source = {name: (row.get("from") or {}).get(name) for name in from_fields}
        target = {name: (row.get("to") or {}).get(name) for name in to_fields}
        if any(value is None for value in (*source.values(), *target.values())):
            continue  # a null key never MATCHes
        wanted.append(
            (encode_key(from_label, source), target, encode_key(to_label, target), row.get("props"))
        )
    if not wanted:
        return RestoreCounters()

    sources = await _tenants(connection, from_label, (w[0] for w in wanted))
    wanted = [w for w in wanted if w[0] in sources]
    nodes_created = 0
    if is_catalog(to_label):
        missing: dict[str, dict[str, Any]] = {}
        for _, key, encoded, _ in wanted:
            missing.setdefault(encoded, key)
        known = await _tenants(connection, to_label, missing)
        inserts = [_node_row(to_label, key, {}) for k, key in missing.items() if k not in known]
        for batch in _chunks(_uniform(inserts)):
            statement = (
                c.dialect()
                .insert(c.table(to_label))
                .values(batch)
                .on_conflict_do_nothing(index_elements=[KEY_COLUMN[to_label]])
                .returning(c.key_column(to_label))
            )
            nodes_created += len((await connection.execute(statement)).all())
        targets: dict[str, str | None] = dict.fromkeys(missing)
    else:
        targets = await _tenants(connection, to_label, (w[2] for w in wanted))

    edge_rows: dict[Any, list[dict[str, Any]]] = {}
    for src_key, _, dst_key, props in wanted:
        if dst_key not in targets:
            continue
        tenant = sources[src_key] if sources[src_key] is not None else targets[dst_key]
        table, edge = c.edge_row(
            rel_type, from_label, src_key, to_label, dst_key, tenant=tenant, props=props
        )
        edge_rows.setdefault(table, []).append(edge)

    relationships_created = 0
    for table, table_rows in edge_rows.items():
        for batch in _chunks(_uniform(table_rows)):
            statement = (
                c.dialect()
                .insert(table)
                .values(batch)
                .on_conflict_do_nothing(index_elements=list(c.EDGE_IDENTITY_COLUMNS))
                .returning(table.c.id)
            )
            relationships_created += len((await connection.execute(statement)).all())
    return RestoreCounters(nodes_created=nodes_created, relationships_created=relationships_created)


def _owner_document_key(child_label: str, rel_type: str, owner_label: str) -> Any:
    """Scalar subquery: the ``_document_key`` of the owner a *child_label* row hangs from
    through *rel_type* (the owner's ``_key`` when the owner is the Document)."""
    link = E.alias("owner_link")
    condition = sa.and_(
        link.c.type == rel_type,
        link.c.src_label == owner_label,
        link.c.dst_label == child_label,
        link.c.dst_key == c.key_column(child_label),
    )
    if owner_label == "Document":
        value = link.c.src_key
        statement = sa.select(value).where(condition)
    else:
        owner = c.table(owner_label).alias("owner")
        value = owner.c._document_key
        statement = (
            sa.select(value)
            .select_from(link.join(owner, owner.c[KEY_COLUMN[owner_label]] == link.c.src_key))
            .where(condition, value.is_not(None))
        )
    return statement.order_by(value).limit(1).scalar_subquery()


async def assign_document_keys(connection: AsyncConnection) -> int:
    """Fill in the unset ``_document_key`` of per-document nodes (and ``_structure_node_id`` of
    InfoUnits) from the relationships they hang from (:data:`_OWNERSHIP`). Returns the rows
    updated. Nodes that hang from nothing keep it unset."""
    updated = 0
    for child_label, rel_type, owner_label in _OWNERSHIP:
        child = c.table(child_label)
        owner_key = _owner_document_key(child_label, rel_type, owner_label)
        statement = (
            sa.update(child)
            .where(child.c._document_key.is_(None), owner_key.is_not(None))
            .values(_document_key=owner_key)
        )
        rounds = _MAX_TREE_DEPTH if child_label == owner_label else 1
        for _ in range(rounds):
            count = (await connection.execute(statement)).rowcount
            updated += count
            if not count:
                break
    info_units = c.table("InfoUnit")
    link = E.alias("owner_link")
    structure_node = (
        sa.select(link.c.src_key)
        .where(
            link.c.type == "HAS_INFO_UNIT",
            link.c.src_label == "StructureNode",
            link.c.dst_label == "InfoUnit",
            link.c.dst_key == info_units.c.uid,
        )
        .order_by(link.c.src_key)
        .limit(1)
        .scalar_subquery()
    )
    statement = (
        sa.update(info_units)
        .where(info_units.c._structure_node_id.is_(None), structure_node.is_not(None))
        .values(_structure_node_id=structure_node)
    )
    updated += (await connection.execute(statement)).rowcount
    return updated


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


class SqlRestoreWriter(RestoreWriter):
    """Runs write transactions with at most *concurrency* in flight (as the Neo4j writer).

    Each batch runs in its own transaction on a pooled connection of the async engine,
    retried whole on transient errors. Batches of the same *lane* run one after the other,
    lanes in parallel. The producer waits for a free slot **before** handing over a batch.

    Every public method may raise a :class:`~scinr.newton.exceptions.GraphError` (the first
    error of a batch is kept and raised, translated, by the next ``delete_stale`` /
    ``merge_nodes`` / ``create_relationships`` that has to wait, or by ``drain()``).
    """

    def __init__(self, engine: AsyncEngine, concurrency: int) -> None:
        self._engine = engine
        self._slots = asyncio.Semaphore(concurrency)
        self._lanes: dict[Hashable, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()
        self._error: BaseException | None = None
        self._linked = False

    async def _transaction(self, work: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        async def attempt() -> Any:
            async with self._engine.begin() as connection:
                return await work(connection, *args)

        return await with_sql_retry(attempt)

    async def _run(
        self,
        lane: Hashable,
        work: Callable[..., Awaitable[Any]],
        args: tuple,
        on_done: Callable[[Any], None],
    ) -> None:
        try:
            async with self._lanes.setdefault(lane, asyncio.Lock()):
                on_done(await self._transaction(work, *args))
        except Exception as exc:
            # Kept for drain() to raise: the task itself never fails.
            if self._error is None:
                self._error = exc
        finally:
            self._slots.release()

    async def _submit(
        self,
        lane: Hashable,
        work: Callable[..., Awaitable[Any]],
        *args: Any,
        on_done: Callable[[Any], None] = lambda _: None,
    ) -> None:
        if self._error is not None:
            await self._drain()
        await self._slots.acquire()
        task = asyncio.create_task(self._run(lane, work, args, on_done))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._error is not None:
            raise self._error

    @async_sql_errors()
    async def delete_stale(
        self,
        lane: Hashable,
        node_ids: Sequence[str],
        *,
        model_decisions: bool,
        extraction_results: bool,
    ) -> None:
        """Delete the stale ModelDecision / ExtractionResult subtrees hanging from *node_ids*
        (one transaction per batch)."""
        await self._submit(lane, _delete_stale, list(node_ids), model_decisions, extraction_results)

    @async_sql_errors()
    async def merge_nodes(
        self,
        lane: Hashable,
        labels: tuple[str, ...],
        rows: Sequence[dict[str, Any]],
        on_done: Callable[[RestoreCounters], None],
    ) -> None:
        """``MERGE`` a batch of nodes sharing *labels* (props only on create).

        Raises:
            FreezeError: If a label is not a valid identifier or has no known business key.
        """
        for label in labels:
            validate_identifier(label)
        node_identity(labels)
        await self._submit(lane, _merge_nodes, tuple(labels), list(rows), on_done=on_done)

    @async_sql_errors()
    async def create_relationships(
        self,
        lane: Hashable,
        group: tuple[str, str, str],
        rows: Sequence[dict[str, Any]],
        on_done: Callable[[RestoreCounters], None],
    ) -> None:
        """Create a batch of relationships of *group* unless an identical one exists.

        Raises:
            FreezeError: If the type or a label is not a valid identifier.
        """
        rel_type, from_label, to_label = group
        validate_identifier(rel_type)
        node_identity([validate_identifier(from_label)])
        node_identity([validate_identifier(to_label)])
        self._linked = True
        await self._submit(lane, _create_relationships, tuple(group), list(rows), on_done=on_done)

    @async_sql_errors()
    async def drain(self) -> None:
        """Wait for every batch in flight; re-raise (translated) the first failure. After
        relationship batches, fill in the ``_document_key`` of the nodes they linked."""
        await self._drain()
        if self._linked:
            self._linked = False
            await self._transaction(assign_document_keys)

    @async_sql_errors()
    async def abort(self) -> None:
        """Cancel every batch in flight."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# The store group
# ---------------------------------------------------------------------------


def _read_documents(statement: sa.Select) -> list[DocumentInfo]:
    return c.sync_read(
        lambda connection: [
            document_info(row, is_seed=False) for row in connection.execute(statement).mappings()
        ]
    )


class SqlRestoreOps(RestoreOps):
    """``store.restore`` over SQL. Synchronous methods use the process-wide sync engine.

    **Known difference D1 (accepted).** The Neo4j restore creates a relationship unless one
    with **all** the same properties exists between the two nodes; SQL cannot hold two
    relationships with the same identity (type, ends and ``_ident``), so a snapshot
    relationship whose identity already exists with other properties is not created. It only
    diverges for a snapshot holding two relationships of equal identity and different
    properties, which the pipeline never produces.
    """

    def batch_writer(self, concurrency: int) -> RestoreWriter:
        """A :class:`SqlRestoreWriter` on the async engine, with
        ``dialect.max_concurrent_writes(concurrency)`` slots.

        Must be called from the event loop that will use it.

        Raises:
            GraphConfigurationError: If ``sql_url`` is not configured.
        """
        with sql_errors():
            engine = c.async_write_engine()
            return SqlRestoreWriter(engine, c.dialect().max_concurrent_writes(concurrency))

    @sql_errors()
    def existing_documents(self, tenant: str, keys: Sequence[dict[str, Any]]) -> list[DocumentInfo]:
        """The :Document nodes among *keys* that exist, in the order of *keys*
        (``DOCS_BY_KEY`` + ``RETURN properties(d)``).

        Raises:
            GraphError: If the engine fails.
        """
        doc_keys = document_keys(tenant, list(keys))

        def read(connection: Connection) -> dict[str, DocumentInfo]:
            found: dict[str, DocumentInfo] = {}
            for batch in c.batched(sorted(set(doc_keys)), c.IN_BATCH):
                statement = sa.select(D).where(D.c._key.in_(batch))
                for row in connection.execute(statement).mappings():
                    found[row["_key"]] = document_info(row, is_seed=False)
            return found

        found = c.sync_read(read)
        return [found[key] for key in doc_keys if key in found]

    @sql_errors()
    def frozen_stubs_of_blob(self, tenant: str, blob_id: str) -> list[DocumentInfo]:
        """The documents of *tenant* frozen with the snapshot *blob_id* (``_STUBS_OF_BLOB_QUERY``),
        ordered by path and version.

        Raises:
            GraphError: If the engine fails.
        """
        statement = (
            sa.select(D)
            .where(D.c.tenant_id == tenant, D.c.frozen == sa.true(), D.c.frozen_blob_id == blob_id)
            .order_by(c.dialect().order_text(D.c.path), D.c.version)
        )
        return _read_documents(statement)

    @sql_errors()
    def finish_restore(
        self, tenant: str, docs: Sequence[DocumentInfo], recreated_paths: Sequence[str]
    ) -> None:
        """Remove the temporary links and unmark the stubs, :data:`DOCUMENTS_PER_QUERY`
        documents at a time, then recompute ``latest`` for the recreated paths.

        Cypher: ``_TEMPORARY_LINKS_DELETE_QUERY`` (every ``HAS_MODEL_DECISION`` /
        ``HAS_EXTRACTION`` leaving the Documents, read first, deleted
        :data:`DELETE_BATCH_ROWS` per transaction), ``_RESTORE_FINISH_QUERY`` (one transaction
        per chunk) and ``_RECOMPUTE_LATEST_QUERY`` (one transaction).

        Raises:
            GraphError: If the engine fails.
        """
        doc_keys = document_keys(tenant, [doc.key for doc in docs])
        unmark = c.set_columns(
            D,
            {
                "frozen": False,
                "frozen_blob_id": None,
                "frozen_at": None,
                "frozen_keep_structure_nodes": None,
                "frozen_keep_annotations": None,
                "frozen_keep_extraction_results": None,
                "frozen_cleanup_pending": None,
            },
        )
        for chunk in c.batched(doc_keys, DOCUMENTS_PER_QUERY):
            links = sa.select(E.c.id).where(
                E.c.src_label == "Document",
                E.c.src_key.in_(chunk),
                E.c.type.in_(("HAS_MODEL_DECISION", "HAS_EXTRACTION")),
            )
            ids = c.sync_read(
                lambda connection, links=links: c.column_values(connection.execute(links))
            )
            for batch in c.batched(ids, DELETE_BATCH_ROWS):
                statement = sa.delete(E).where(E.c.id.in_(batch))
                c.sync_write(lambda connection, statement=statement: connection.execute(statement))
            statement = sa.update(D).where(D.c._key.in_(chunk)).values(**unmark)
            c.sync_write(lambda connection, statement=statement: connection.execute(statement))

        if recreated_paths:
            paths = sorted(set(recreated_paths))
            newer = D.alias("newer")
            newest = (
                sa.select(sa.func.max(newer.c.version))
                .where(newer.c.tenant_id == D.c.tenant_id, newer.c.path == D.c.path)
                .scalar_subquery()
            )

            def recompute(connection: Connection) -> None:
                for batch in c.batched(paths, c.IN_BATCH):
                    connection.execute(
                        sa.update(D)
                        .where(D.c.tenant_id == tenant, D.c.path.in_(batch))
                        .values(latest=D.c.version == newest)
                    )

            c.sync_write(recompute)

    @sql_errors()
    def count_blob_references(self, tenant: str, blob_id: str) -> int:
        """How many documents of *tenant* still reference the snapshot *blob_id*.

        Raises:
            GraphError: If the engine fails.
        """
        statement = (
            sa.select(sa.func.count())
            .select_from(D)
            .where(D.c.tenant_id == tenant, D.c.frozen_blob_id == blob_id)
        )
        return c.sync_read(lambda connection: connection.execute(statement).scalar_one())

    @sql_errors()
    def has_stale_extraction_results(self, node_id_groups: Iterable[Sequence[str]]) -> bool:
        """Whether any of the StructureNodes still carries an ExtractionResult
        (``_STALE_EXTRACTION_RESULTS_QUERY``, :data:`DELETE_BATCH_ROWS` ids at a time).

        Raises:
            GraphError: If the engine fails.
        """
        for ids in node_id_groups:
            for batch in c.batched(list(ids), DELETE_BATCH_ROWS):
                found = c.follow(
                    c.nodes_with_keys("StructureNode", batch),
                    [("HAS_EXTRACTION", "ExtractionResult")],
                ).subquery()
                statement = sa.select(sa.func.count()).select_from(found)
                if c.sync_read(lambda connection, s=statement: connection.execute(s).scalar_one()):
                    return True
        return False
