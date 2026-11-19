"""
graph/sql/deletion.py — ``store.deletion`` over SQL: selection, batched cascade, marks.

Translation of ``graph/neo4j/deletion.py``. The cascade is shared by ``delete_document()``
and ``freeze_document()`` and keeps the bounds of the Cypher:

- the documents go :data:`DOCUMENTS_PER_QUERY` at a time;
- every delete step first reads every node it is about to delete (as the aggregation before
  ``CALL { … } IN TRANSACTIONS`` does) and then deletes them :data:`DELETE_BATCH_ROWS` per
  transaction, each batch retried whole.

Not atomic, like Neo4j: every step only deletes what is still there, so the callers make the
operation resumable. The subtree of a Document is found through ``_document_key`` (the
StructureNodes of the document), not by walking ``HAS_STRUCTURE`` / ``HAS_CHILD``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from scinr.newton.config import get_config
from scinr.newton.exceptions import TenantAccessError
from scinr.newton.graph.base import DeletionOps
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._traversal import reachable
from scinr.newton.graph.types import DELETE_BATCH_ROWS, DOCUMENTS_PER_QUERY, DocumentSelector
from scinr.newton.results import DeletedDocument
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import sql_errors
from scinr.newton.sql.schema import ensure_schema, rls_enabled

if TYPE_CHECKING:
    from scinr.newton.graph.base import OrphanCollector

__all__ = [
    "SUBTREE_COUNTER_FIELDS",
    "SqlDeletionOps",
    "delete_documents",
    "doc_match",
    "document_keys",
    "run_subtree_steps",
    "subtree_nodes",
    "subtree_steps",
]

logger = logging.getLogger(__name__)

D = t.documents
SN = t.structure_nodes
E = t.edges

#: Hop bound of the unbounded ``IS_COMPOSED_OF*`` of Cypher (``UNION`` stops on cycles).
COMPOSITION_MAX_DEPTH = 1000

SUBTREE_COUNTER_FIELDS = (
    "structure_nodes_deleted",
    "structure_nodes_kept",
    "info_units_deleted",
    "model_decisions_deleted",
    "model_decisions_kept",
    "proposed_models_deleted",
    "proposed_fields_deleted",
    "extraction_results_deleted",
    "extraction_results_kept",
)

_MD = ("HAS_MODEL_DECISION", "ModelDecision")
PROPOSED_FIELD_PATH: list[c.Step] = [
    _MD,
    ("HAS_PROPOSED_MODEL", "ProposedModel"),
    ("HAS_PROPOSED_FIELD", "ProposedField"),
]
PROPOSED_MODEL_PATH: list[c.Step] = [_MD, ("HAS_PROPOSED_MODEL", "ProposedModel")]
COMPLEMENTARY_MATCH_PATH: list[c.Step] = [_MD, ("HAS_COMPLEMENTARY_MATCH", "ComplementaryMatch")]
SUPPLEMENTARY_FIELD_PATH: list[c.Step] = [_MD, ("HAS_SUPPLEMENTARY_FIELD", "SupplementaryField")]
MODEL_DECISION_PATH: list[c.Step] = [_MD]
EXTRACTION_RESULT_PATH: list[c.Step] = [("HAS_EXTRACTION", "ExtractionResult")]
INFO_UNIT_PATH: list[c.Step] = [("HAS_INFO_UNIT", "InfoUnit")]


def document_keys(tenant: str, keys: Sequence[dict[str, Any]]) -> list[str]:
    """The ``_key`` of each ``{"path", "version"}`` of *tenant* (the ``DOCS_BY_KEY`` of Cypher)."""
    return [c.document_key(tenant, key["path"], key["version"]) for key in keys]


def subtree_nodes(doc_keys: Sequence[str]) -> sa.Select:
    """``(label, key)`` of the StructureNodes of the documents (``SUBTREE_STRUCTURE``)."""
    return sa.select(
        sa.literal("StructureNode", sa.Text).label("label"), SN.c.id.label("key")
    ).where(SN.c._document_key.in_(list(doc_keys)))


# ── Steps ────────────────────────────────────────────────────────────────────

#: One step of the cascade: ``(tenant, document keys) -> count``.
SubtreeStep = Callable[[str, list[str]], int]


def _delete_handles(handles: Sequence[tuple[str, str]]) -> int:
    """DETACH DELETE the ``(label, key)`` nodes, :data:`DELETE_BATCH_ROWS` per transaction."""
    deleted = 0
    for batch in c.batched(list(handles), DELETE_BATCH_ROWS):

        def attempt(connection: Connection, batch: list[tuple[str, str]] = batch) -> int:
            count = 0
            for label, keys in c.group_by_label(batch).items():
                statements = c.detach_delete(label, keys)
                for statement in statements[:-1]:
                    connection.execute(statement)
                count += connection.execute(statements[-1]).rowcount
            return count

        deleted += c.sync_write(attempt)
    return deleted


def _read_handles(found: sa.Select) -> list[tuple[str, str]]:
    return c.sync_read(lambda connection: [(row[0], row[1]) for row in connection.execute(found)])


def _path_step(path: list[c.Step], action: str) -> SubtreeStep:
    """The ``subtree_query(pattern, action)`` of Cypher: delete or count the nodes at the end
    of *path* from the StructureNodes of the documents."""

    def step(_tenant: str, doc_keys: list[str]) -> int:
        found = c.follow(subtree_nodes(doc_keys), path)
        if action == "count":
            sub = found.subquery()
            return c.sync_read(
                lambda connection: connection.execute(
                    sa.select(sa.func.count()).select_from(sub)
                ).scalar_one()
            )
        return _delete_handles(_read_handles(found))

    return step


def _structure_nodes_step(action: str) -> SubtreeStep:
    def step(_tenant: str, doc_keys: list[str]) -> int:
        found = subtree_nodes(doc_keys)
        if action == "count":
            sub = found.subquery()
            return c.sync_read(
                lambda connection: connection.execute(
                    sa.select(sa.func.count()).select_from(sub)
                ).scalar_one()
            )
        return _delete_handles(_read_handles(found))

    return step


def _relink_step(rel_type: str, label: str) -> SubtreeStep:
    """``MATCH (s)-[:rel]->(x) WITH DISTINCT d, x MERGE (d)-[:rel]->(x)`` (one transaction)."""

    def step(tenant: str, doc_keys: list[str]) -> int:
        link = E.alias("link")
        pairs = (
            sa.select(SN.c._document_key, link.c.dst_key)
            .select_from(
                SN.join(
                    link,
                    sa.and_(
                        link.c.src_label == "StructureNode",
                        link.c.src_key == SN.c.id,
                        link.c.type == rel_type,
                        link.c.dst_label == label,
                    ),
                )
            )
            .where(SN.c._document_key.in_(doc_keys))
            .distinct()
        )

        def attempt(connection: Connection) -> int:
            rows = [
                c.edge_row(rel_type, "Document", doc_key, label, key, tenant=tenant)[1]
                for doc_key, key in connection.execute(pairs).all()
            ]
            for batch in c.batched(rows, DELETE_BATCH_ROWS):
                connection.execute(c.merge_edges_many("Document", label), batch)
            return 0

        return c.sync_write(attempt)

    return step


def subtree_steps(
    keep_structure_nodes: bool = False,
    keep_annotations: bool = False,
    keep_extraction_results: bool = False,
) -> list[tuple[str | None, SubtreeStep]]:
    """Ordered ``(counter, step)`` removing the subtree of a Document (as Neo4j's)."""
    steps: list[tuple[str | None, SubtreeStep]] = []
    if not keep_structure_nodes:
        if keep_annotations:
            steps.append((None, _relink_step("HAS_MODEL_DECISION", "ModelDecision")))
        if keep_extraction_results:
            steps.append((None, _relink_step("HAS_EXTRACTION", "ExtractionResult")))
    if keep_annotations:
        steps.append(("model_decisions_kept", _path_step(MODEL_DECISION_PATH, "count")))
    else:
        steps += [
            ("proposed_fields_deleted", _path_step(PROPOSED_FIELD_PATH, "delete")),
            ("proposed_models_deleted", _path_step(PROPOSED_MODEL_PATH, "delete")),
            (None, _path_step(COMPLEMENTARY_MATCH_PATH, "delete")),
            (None, _path_step(SUPPLEMENTARY_FIELD_PATH, "delete")),
            ("model_decisions_deleted", _path_step(MODEL_DECISION_PATH, "delete")),
        ]
    if keep_extraction_results:
        steps.append(("extraction_results_kept", _path_step(EXTRACTION_RESULT_PATH, "count")))
    else:
        steps.append(("extraction_results_deleted", _path_step(EXTRACTION_RESULT_PATH, "delete")))
    steps.append(("info_units_deleted", _path_step(INFO_UNIT_PATH, "delete")))
    if keep_structure_nodes:
        steps.append(("structure_nodes_kept", _structure_nodes_step("count")))
    else:
        steps.append(("structure_nodes_deleted", _structure_nodes_step("delete")))
    return steps


def run_subtree_steps(
    tenant: str,
    keys: Sequence[dict[str, Any]],
    steps: Sequence[tuple[str | None, SubtreeStep]],
    *,
    caller: str,
    orphans: OrphanCollector | None = None,
) -> dict[str, int]:
    """Run *steps* over the documents at *keys*, :data:`DOCUMENTS_PER_QUERY` at a time, and
    return the summed counters (``gather`` before and ``chunk_done`` after each chunk)."""
    counters = dict.fromkeys(SUBTREE_COUNTER_FIELDS, 0)
    for start in range(0, len(keys), DOCUMENTS_PER_QUERY):
        chunk = list(keys[start : start + DOCUMENTS_PER_QUERY])
        doc_keys = document_keys(tenant, chunk)
        if orphans is not None:
            orphans.gather(chunk)
        for counter, step in steps:
            count = step(tenant, doc_keys)
            if counter is not None:
                counters[counter] += count
        if orphans is not None:
            orphans.chunk_done()
        logger.info(
            "%s: subtree of %d/%d document(s) processed.",
            caller,
            start + len(chunk),
            len(keys),
        )
    return counters


def delete_documents(tenant: str, keys: Sequence[dict[str, Any]]) -> int:
    """DETACH DELETE the Documents at *keys*, :data:`DELETE_BATCH_ROWS` per transaction, in order."""
    deleted = 0
    for batch in c.batched(document_keys(tenant, keys), DELETE_BATCH_ROWS):

        def attempt(connection: Connection, batch: list[str] = batch) -> int:
            statements = c.detach_delete("Document", batch)
            connection.execute(statements[0])
            return connection.execute(statements[1]).rowcount

        deleted += c.sync_write(attempt)
    return deleted


# ── Selection ────────────────────────────────────────────────────────────────

_IN_FIELDS = frozenset({"job_id", "created_by_user_id"})
_FILTERABLE_FIELDS = ("path", "job_id", "version", "created_by_user_id")


def doc_match(selector: DocumentSelector) -> Any:
    """``WHERE`` of the "filters" selection: the tenant always, each other field when set.

    Raises:
        ValueError: If the selector carries no tenant.
    """
    tenant = selector.tenant_id
    if tenant is None:
        raise ValueError("doc_match requires the stored tenant key in selector.tenant_id.")
    conditions = [D.c.tenant_id == tenant]
    for name in _FILTERABLE_FIELDS:
        value = getattr(selector, name)
        if value is None:
            continue
        column = D.c[name]
        if name in _IN_FIELDS:
            conditions.append(column.in_([value] if isinstance(value, str) else list(value)))
        else:
            conditions.append(column == value)
    return sa.and_(*conditions)


def _with_descendants(condition: Any) -> tuple[sa.Select, Any]:
    """The matched Documents and the ``_key`` of their ``IS_COMPOSED_OF*`` descendants."""
    seeds = sa.select(sa.literal("Document", sa.Text).label("label"), D.c._key.label("key")).where(
        condition
    )
    below = reachable(
        seeds, max_hops=COMPOSITION_MAX_DEPTH, rel_types={"IS_COMPOSED_OF"}, name="composed"
    )
    return seeds, sa.select(below.c.key).where(below.c.label == "Document")


def _interrupted(deletion_pending: Any, props: dict[str, Any]) -> bool:
    """``coalesce(doc.deletion_pending, doc.frozen_cleanup_pending, false)``."""
    if deletion_pending is not None:
        return bool(deletion_pending)
    pending = props.get("frozen_cleanup_pending")
    return bool(pending) if pending is not None else False


class SqlDeletionOps(DeletionOps):
    """``store.deletion`` over SQL (synchronous, process-wide sync engine)."""

    @sql_errors()
    def fetch_document_keys(
        self, selector: DocumentSelector
    ) -> tuple[list[dict], list[DeletedDocument], bool]:
        """The keys of the matched Documents and their ``IS_COMPOSED_OF*`` descendants —
        descendants first, matched last —, the same as :class:`DeletedDocument`, and whether
        an earlier delete or freeze of any of them was interrupted.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        tenant = selector.tenant_id
        condition = doc_match(selector)
        seeds, descendants = _with_descendants(condition)
        seed_keys = sa.select(seeds.subquery().c.key)
        d = c.dialect()
        ensure_schema()  # the RLS mode is read when the schema is verified
        columns = [
            D.c.path,
            D.c.version,
            D.c.tenant_id,
            D.c.is_folder,
            D.c.deletion_pending,
            D.c.props,
            D.c._key.in_(seed_keys).label("is_seed"),
        ]
        guard_writes = rls_enabled()
        if guard_writes:
            can_write = getattr(sa.func, get_config().sql_schema).newton_can_write(D.c.tenant_id)
            columns.append(can_write.label("can_write"))
        statement = (
            sa.select(*columns)
            .where(sa.or_(D.c._key.in_(seed_keys), D.c._key.in_(descendants)))
            .order_by(d.order_text(D.c.path), D.c.version)
        )

        def read(connection: Connection) -> list[dict]:
            rows = []
            for row in connection.execute(statement).mappings():
                props = c.node_from_row("Document", row)[1]
                rows.append(
                    {
                        "path": row["path"],
                        "version": row["version"],
                        "tenant_id": row["tenant_id"],
                        "is_folder": bool(row["is_folder"])
                        if row["is_folder"] is not None
                        else False,
                        "is_seed": bool(row["is_seed"]),
                        "interrupted": _interrupted(row["deletion_pending"], props),
                        "can_write": bool(row["can_write"]) if guard_writes else True,
                    }
                )
            return rows

        rows = c.sync_read(read)
        if not all(row["can_write"] for row in rows):
            # Under RLS a DELETE of rows the role may see but not write removes nothing and
            # does not fail: refuse up front so the cascade is never "successful" silently.
            raise TenantAccessError(
                f"The SQL user has read-only access to tenant {tenant!r}: "
                "it cannot delete or freeze its documents."
            )
        addressable = [
            row
            for row in rows
            if row["path"] is not None and row["version"] is not None and row["tenant_id"] == tenant
        ]
        if len(addressable) != len(rows):
            logger.warning(
                "delete_document: %d Document(s) reached by the selector have no path/version "
                "or belong to another tenant than %r; they are left untouched.",
                len(rows) - len(addressable),
                tenant,
            )
        addressable.sort(key=lambda row: row["is_seed"])
        keys = [{"path": row["path"], "version": row["version"]} for row in addressable]
        documents = [
            DeletedDocument(
                path=row["path"], version=row["version"], is_folder=bool(row["is_folder"])
            )
            for row in addressable
        ]
        return keys, documents, any(row["interrupted"] for row in addressable)

    @sql_errors()
    def fetch_existing_versions(self, selector: DocumentSelector) -> list[int]:
        """The sorted versions of the matched Documents (``None`` dropped).

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        statement = sa.select(D.c.version).where(doc_match(selector))
        versions = c.sync_read(lambda connection: c.column_values(connection.execute(statement)))
        return sorted(v for v in versions if v is not None)

    @sql_errors()
    def fetch_raw_file_ids(self, selector: DocumentSelector) -> list[tuple[str, str | None]]:
        """Distinct non-empty ``(raw_file_id, tenant_id)`` of the matched Documents and their
        ``IS_COMPOSED_OF*`` descendants.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """
        seeds, descendants = _with_descendants(doc_match(selector))
        seed_keys = sa.select(seeds.subquery().c.key)
        d = c.dialect()
        statement = (
            sa.select(D.c.raw_file_id, D.c.tenant_id)
            .where(
                sa.or_(D.c._key.in_(seed_keys), D.c._key.in_(descendants)),
                D.c.raw_file_id.is_not(None),
                D.c.raw_file_id != "",
            )
            .distinct()
        )
        sub = statement.subquery()
        ordered = sa.select(sub.c.raw_file_id, sub.c.tenant_id).order_by(
            d.order_text(sub.c.raw_file_id), d.order_text(sub.c.tenant_id)
        )
        return c.sync_read(
            lambda connection: [(row[0], row[1]) for row in connection.execute(ordered)]
        )

    @sql_errors()
    def mark_deletion_pending(self, tenant: str, keys: Sequence[dict[str, Any]]) -> None:
        """Mark the documents at *keys* ``deletion_pending``, :data:`DELETE_BATCH_ROWS` per
        transaction, before anything is deleted.

        Raises:
            GraphError: If the engine fails.
        """
        for batch in c.batched(document_keys(tenant, keys), DELETE_BATCH_ROWS):
            statement = sa.update(D).where(D.c._key.in_(batch)).values(deletion_pending=True)
            c.sync_write(lambda connection, statement=statement: connection.execute(statement))

    @sql_errors()
    def cascade_delete(
        self,
        tenant: str,
        keys: Sequence[dict[str, Any]],
        collector: OrphanCollector | None,
        *,
        caller: str,
    ) -> dict[str, int]:
        """Delete the whole subtree of the documents at *keys* (the Documents stay), in
        bounded transactions, and return the summed counters.

        Raises:
            GraphError: If the engine fails.
        """
        return run_subtree_steps(tenant, keys, subtree_steps(), caller=caller, orphans=collector)

    @sql_errors()
    def delete_documents(self, tenant: str, keys: Sequence[dict[str, Any]]) -> int:
        """DETACH DELETE the Documents at *keys*, :data:`DELETE_BATCH_ROWS` per transaction, in
        the order given. Returns how many were deleted.

        Raises:
            GraphError: If the engine fails.
        """
        return delete_documents(tenant, keys)
