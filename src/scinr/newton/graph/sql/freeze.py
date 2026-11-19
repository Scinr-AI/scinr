"""
graph/sql/freeze.py — ``store.freeze`` over SQL: set resolution, export reads, marks, reduction.

Translation of ``graph/neo4j/freeze.py``. The snapshot format, the validation of what is read
and the orchestration stay in ``ingest/freeze.py``; this module only reads and marks.

One export is one connection: :meth:`SqlFreezeOps.export_snapshot` retries the *whole*
attempt (the caller's callback opens the output file inside it). The
:class:`~scinr.newton.graph.types.ExportReader` it hands out streams: each method is a
generator over a result read with ``stream_results`` / ``yield_per``, so nothing is
materialised beyond one batch of :data:`DELETE_BATCH_ROWS` rows.

The subtree of a Document is the one of the Cypher (``_SUBTREE_NODES_FRAGMENT``): its
StructureNodes (selected by ``_document_key``, the ``HAS_STRUCTURE`` / ``HAS_CHILD*`` tree of
the Cypher), what hangs from each one, and every non-catalog node reachable from its
ExtractionResults within :data:`GC_REACHABILITY_MAX_HOPS` hops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from scinr.newton.exceptions import FreezeError
from scinr.newton.graph.base import FreezeOps, OrphanCollector
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._traversal import reachable
from scinr.newton.graph.sql.deletion import (
    _with_descendants,
    doc_match,
    document_keys,
    run_subtree_steps,
    subtree_nodes,
    subtree_steps,
)
from scinr.newton.graph.types import (
    DELETE_BATCH_ROWS,
    GC_REACHABILITY_MAX_HOPS,
    DocumentInfo,
    DocumentSelector,
    ExportReader,
    KeepFlags,
    RawNode,
    RawRelationship,
)
from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.sql import tables as t
from scinr.newton.sql.dialect import SQLITE_DEFERRED_OPTION
from scinr.newton.sql.engine import get_sync_engine
from scinr.newton.sql.errors import sql_errors
from scinr.newton.sql.keys import decode_key
from scinr.newton.sql.names import CATALOG_LABELS, LABEL_TABLE
from scinr.newton.sql.retry import with_sql_retry_sync

__all__ = ["SqlFreezeOps", "document_info", "subtree_handles"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

D = t.documents
SN = t.structure_nodes
E = t.edges

_MD: c.Step = ("HAS_MODEL_DECISION", "ModelDecision")
_DECISION_CHILDREN = ("HAS_PROPOSED_MODEL", "HAS_COMPLEMENTARY_MATCH", "HAS_SUPPLEMENTARY_FIELD")
_DOCUMENT_LINKS = ("IS_COMPOSED_OF", "HAS_NEWER_VERSION")


def document_info(row: Any, *, is_seed: bool) -> DocumentInfo:
    """A :class:`DocumentInfo` of a ``documents`` row (``properties(d)`` of the Cypher)."""
    properties = c.node_from_row("Document", row)[1]
    return DocumentInfo(
        path=properties.get("path"),
        version=properties.get("version"),
        tenant_id=properties.get("tenant_id"),
        is_seed=is_seed,
        properties=properties,
    )


# ---------------------------------------------------------------------------
# Snapshot export — reads
# ---------------------------------------------------------------------------


def subtree_handles(doc_key: str) -> sa.CompoundSelect:
    """``(label, key)`` of every node of the subtree of one Document, each once.

    Cypher: ``_SUBTREE_NODES_FRAGMENT`` — the StructureNodes, then from each one its
    InfoUnits, its ModelDecision and the ProposedModel / ComplementaryMatch /
    SupplementaryField below it (any label), the ProposedFields of its ProposedModels, its
    ExtractionResults and every non-catalog node within ``GC_REACHABILITY_MAX_HOPS`` hops
    of them.
    """
    structure = subtree_nodes([doc_key])
    results = c.follow(structure, [("HAS_EXTRACTION", "ExtractionResult")])
    below = reachable(results, max_hops=GC_REACHABILITY_MAX_HOPS, name="export_reach")
    return sa.union(
        structure,
        c.follow(structure, [("HAS_INFO_UNIT", "InfoUnit")]),
        c.follow(structure, [_MD]),
        c.follow(structure, [_MD, (_DECISION_CHILDREN, None)]),
        c.follow(
            structure,
            [_MD, ("HAS_PROPOSED_MODEL", "ProposedModel"), ("HAS_PROPOSED_FIELD", "ProposedField")],
        ),
        results,
        sa.select(below.c.label, below.c.key).where(below.c.label.not_in(sorted(CATALOG_LABELS))),
    )


def _stream(connection: Connection, statement: Any) -> Any:
    """Execute *statement* streaming its rows, :data:`DELETE_BATCH_ROWS` at a time."""
    return connection.execute(
        statement.execution_options(stream_results=True, yield_per=DELETE_BATCH_ROWS)
    )


def _labels(label: str, role: Any) -> tuple[str, ...]:
    """``labels(n)`` of a node from its primary label (and the ``role`` of a StructureNode)."""
    role_label = ROLE_TO_LABEL.get(role or "") if label == "StructureNode" else None
    return (label, role_label) if role_label else (label,)


def _relationships_with_ends(condition: Any) -> sa.Select:
    """``edges`` matching *condition*, with the ``role`` of StructureNode ends (for labels)."""
    src_sn, dst_sn = SN.alias("src_sn"), SN.alias("dst_sn")
    return (
        sa.select(
            E.c.type,
            E.c.src_label,
            E.c.src_key,
            E.c.dst_label,
            E.c.dst_key,
            E.c.props,
            src_sn.c.role.label("src_role"),
            dst_sn.c.role.label("dst_role"),
        )
        .select_from(
            E.outerjoin(
                src_sn, sa.and_(E.c.src_label == "StructureNode", src_sn.c.id == E.c.src_key)
            ).outerjoin(
                dst_sn, sa.and_(E.c.dst_label == "StructureNode", dst_sn.c.id == E.c.dst_key)
            )
        )
        .where(condition)
        .order_by(E.c.id)
    )


def _to_relationship(row: Any) -> RawRelationship:
    return RawRelationship(
        type=row.type,
        from_labels=_labels(row.src_label, row.src_role),
        from_key=decode_key(row.src_label, row.src_key),
        to_labels=_labels(row.dst_label, row.dst_role),
        to_key=decode_key(row.dst_label, row.dst_key),
        properties=dict(row.props or {}),
    )


class _SqlExportReader:
    """:class:`ExportReader` over the open connection of one ``export_snapshot`` attempt.

    Each method is a generator over a streamed result. An engine error raised while
    iterating is **not** translated here: it must reach the retry of ``export_snapshot`` raw.
    """

    def __init__(self, connection: Connection, tenant: str) -> None:
        self._connection = connection
        self._tenant = tenant

    def _doc_key(self, doc: DocumentInfo) -> str:
        return c.document_key(self._tenant, doc.path, doc.version)

    def _document_exists(self, doc_key: str) -> bool:
        statement = sa.select(D.c._key).where(D.c._key == doc_key)
        return self._connection.execute(statement).first() is not None

    def subtree_nodes(self, doc: DocumentInfo) -> Iterator[RawNode]:
        """Cypher: ``_SUBTREE_NODES_EXPORT_QUERY`` (labels and properties of each node).

        The ``(label, key)`` handles are streamed ordered by label; each run of at most
        :data:`DELETE_BATCH_ROWS` keys of one label is then read from its table.
        """
        doc_key = self._doc_key(doc)
        if not self._document_exists(doc_key):
            return
        handles = subtree_handles(doc_key).subquery("handles")
        ordered = sa.select(handles.c.label, handles.c.key).order_by(handles.c.label, handles.c.key)
        pending: list[str] = []
        pending_label: str | None = None
        for label, key in _stream(self._connection, ordered):
            if pending and (label != pending_label or len(pending) >= DELETE_BATCH_ROWS):
                yield from self._nodes(pending_label, pending)
                pending = []
            pending_label = label
            pending.append(key)
        if pending:
            yield from self._nodes(pending_label, pending)

    def _nodes(self, label: str | None, keys: list[str]) -> Iterator[RawNode]:
        if label not in LABEL_TABLE:
            return
        statement = sa.select(c.table(label)).where(c.key_column(label).in_(keys))
        for row in self._connection.execute(statement).mappings():
            labels, properties = c.node_from_row(label, row)
            yield RawNode(labels=labels, properties=properties)

    def document_relationships(self, doc: DocumentInfo) -> Iterator[RawRelationship]:
        """Cypher: ``_DOCUMENT_RELATIONSHIPS_EXPORT_QUERY`` — ``HAS_STRUCTURE`` into its tree and
        ``IS_COMPOSED_OF`` / ``HAS_NEWER_VERSION`` with Documents of the tenant, both ways."""
        doc_key = self._doc_key(doc)
        if not self._document_exists(doc_key):
            return
        tenant_docs = sa.select(D.c._key).where(D.c.tenant_id == self._tenant)
        outgoing = sa.and_(E.c.src_label == "Document", E.c.src_key == doc_key)
        incoming = sa.and_(E.c.dst_label == "Document", E.c.dst_key == doc_key)
        condition = sa.or_(
            sa.and_(outgoing, E.c.type == "HAS_STRUCTURE", E.c.dst_label == "StructureNode"),
            sa.and_(
                outgoing,
                E.c.type.in_(_DOCUMENT_LINKS),
                E.c.dst_label == "Document",
                E.c.dst_key.in_(tenant_docs),
            ),
            sa.and_(
                incoming,
                E.c.type.in_(_DOCUMENT_LINKS),
                E.c.src_label == "Document",
                E.c.src_key.in_(tenant_docs),
            ),
        )
        for row in _stream(self._connection, _relationships_with_ends(condition)):
            yield _to_relationship(row)

    def subtree_relationships(self, doc: DocumentInfo) -> Iterator[RawRelationship]:
        """Cypher: ``_SUBTREE_RELATIONSHIPS_EXPORT_QUERY`` — every outgoing relationship of
        every subtree node (the caller drops the targets outside the entry)."""
        doc_key = self._doc_key(doc)
        if not self._document_exists(doc_key):
            return
        handles = subtree_handles(doc_key).cte("subtree_handles")
        condition = sa.exists().where(
            handles.c.label == E.c.src_label, handles.c.key == E.c.src_key
        )
        for row in _stream(self._connection, _relationships_with_ends(condition)):
            yield _to_relationship(row)


# ---------------------------------------------------------------------------
# The store group
# ---------------------------------------------------------------------------


def _keys(docs: Sequence[DocumentInfo]) -> list[dict[str, Any]]:
    return [doc.key for doc in docs]


class SqlFreezeOps(FreezeOps):
    """``store.freeze`` over SQL (synchronous, process-wide sync engine)."""

    @sql_errors()
    def fetch_document_set(self, selector: DocumentSelector) -> list[DocumentInfo]:
        """The matched Documents (``is_seed``) and their ``IS_COMPOSED_OF*`` descendants, each
        once, ordered by path and version.

        Cypher: ``doc_match`` + ``_DOCUMENT_SET_TAIL``.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        seeds, descendants = _with_descendants(doc_match(selector))
        seed_keys = sa.select(seeds.subquery().c.key)
        statement = (
            sa.select(D, D.c._key.in_(seed_keys).label("_is_seed"))
            .where(sa.or_(D.c._key.in_(seed_keys), D.c._key.in_(descendants)))
            .order_by(c.dialect().order_text(D.c.path), D.c.version)
        )

        def read(connection: Connection) -> list[DocumentInfo]:
            return [
                document_info(row, is_seed=bool(row["_is_seed"]))
                for row in connection.execute(statement).mappings()
            ]

        return c.sync_read(read)

    @sql_errors()
    def export_snapshot(
        self, docs: Sequence[DocumentInfo], tenant: str, attempt: Callable[[ExportReader], T]
    ) -> T:
        """Run *attempt* with an :class:`ExportReader` over one connection, retrying it whole.

        Cypher: one session per attempt, retried with ``with_neo4j_retry_sync``. An exception
        from *attempt* that is not an engine error passes through; an engine error raised
        while the reader is iterated is retried like any other and translated when it
        finally leaves this method.

        Raises:
            GraphError: If the engine fails after the retries.
        """

        def _do_export() -> T:
            engine = get_sync_engine().execution_options(**{SQLITE_DEFERRED_OPTION: True})
            with engine.connect() as connection:
                return attempt(_SqlExportReader(connection, tenant))

        return with_sql_retry_sync(_do_export)

    @sql_errors()
    def mark_frozen(
        self,
        tenant: str,
        docs: Sequence[DocumentInfo],
        *,
        blob_id: str,
        frozen_at: str,
        keep_flags: KeepFlags,
    ) -> None:
        """Mark the stubs in one transaction, **before** anything is deleted.

        Cypher: ``_FREEZE_GUARD_QUERY`` (every document still there, none frozen; counted per
        key of ``UNWIND $keys``) then ``_MARK_FROZEN_QUERY``, in one explicit transaction.

        Raises:
            FreezeError: If the documents changed since they were read (one is missing or
                already frozen); nothing is marked.
            GraphError: If the engine fails.
        """
        keys = _keys(docs)
        doc_keys = document_keys(tenant, keys)
        keep_structure_nodes, keep_annotations, keep_extraction_results = keep_flags
        values = c.set_columns(
            D,
            {
                "frozen": True,
                "frozen_blob_id": blob_id,
                "frozen_at": frozen_at,
                "frozen_keep_structure_nodes": keep_structure_nodes,
                "frozen_keep_annotations": keep_annotations,
                "frozen_keep_extraction_results": keep_extraction_results,
                "frozen_cleanup_pending": True,
            },
        )

        def mark(connection: Connection) -> None:
            try:
                state: dict[str, bool] = {}
                for batch in c.batched(sorted(set(doc_keys)), c.IN_BATCH):
                    statement = sa.select(D.c._key, D.c.frozen).where(D.c._key.in_(batch))
                    for key, frozen in connection.execute(statement):
                        state[key] = bool(frozen)
                found = sum(1 for key in doc_keys if key in state)
                frozen = sum(1 for key in doc_keys if state.get(key))
                if found != len(keys) or frozen:
                    raise FreezeError(
                        "The documents to freeze changed during the export "
                        f"(expected {len(keys)} unfrozen, found {found} "
                        f"with {frozen} already frozen); nothing was mutated."
                    )
                for batch in c.batched(sorted(set(doc_keys)), c.IN_BATCH):
                    connection.execute(sa.update(D).where(D.c._key.in_(batch)).values(**values))
            except Exception:
                logger.exception(
                    "freeze_document: stub marking rolled back (tenant=%r, keys=%r)",
                    tenant,
                    keys,
                )
                raise

        c.sync_write(mark)

    @sql_errors()
    def mark_backup(
        self, tenant: str, docs: Sequence[DocumentInfo], blob_id: str, backup_at: str
    ) -> None:
        """Record ``last_backup_blob_id`` / ``last_backup_at`` on the documents.

        Cypher: ``_MARK_BACKUP_QUERY`` over every key, in one write transaction (here: one
        transaction, one ``UPDATE`` per :data:`~scinr.newton.graph.sql._common.IN_BATCH` keys).

        Raises:
            GraphError: If the engine fails.
        """
        doc_keys = document_keys(tenant, _keys(docs))
        values = c.set_columns(D, {"last_backup_blob_id": blob_id, "last_backup_at": backup_at})

        def mark(connection: Connection) -> None:
            for batch in c.batched(doc_keys, c.IN_BATCH):
                connection.execute(sa.update(D).where(D.c._key.in_(batch)).values(**values))

        c.sync_write(mark)

    @sql_errors()
    def reduce_to_stubs(
        self,
        tenant: str,
        docs: Sequence[DocumentInfo],
        keep_flags: KeepFlags,
        collector: OrphanCollector | None,
        *,
        caller: str,
    ) -> dict[str, int]:
        """Delete the subtree of *docs* that *keep_flags* do not keep, in bounded transactions
        (the steps of ``subtree_steps(*keep_flags)``, as Neo4j). Resumable; *collector* is
        handed what the ExtractionResults about to be deleted point at.

        Raises:
            GraphError: If the engine fails.
        """
        return run_subtree_steps(
            tenant, _keys(docs), subtree_steps(*keep_flags), caller=caller, orphans=collector
        )

    @sql_errors()
    def clear_cleanup_pending(self, tenant: str, docs: Sequence[DocumentInfo]) -> None:
        """Remove ``frozen_cleanup_pending`` / ``deletion_pending`` from *docs*,
        :data:`DELETE_BATCH_ROWS` per transaction (``_CLEAR_PENDING_QUERY``, auto-commit).

        Raises:
            GraphError: If the engine fails.
        """
        values = c.set_columns(D, {"frozen_cleanup_pending": None, "deletion_pending": None})
        for batch in c.batched(document_keys(tenant, _keys(docs)), DELETE_BATCH_ROWS):
            statement = sa.update(D).where(D.c._key.in_(batch)).values(**values)
            c.sync_write(lambda connection, statement=statement: connection.execute(statement))
