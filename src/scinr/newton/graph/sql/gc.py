"""
graph/sql/gc.py — Garbage collection of orphaned extraction nodes, over SQL.

Translation of ``graph/neo4j/gc.py``; the same two rules and the same two scopes:

- an :Entity / :ModelInstance is an orphan when no :ExtractionResult reaches it within
  :data:`GC_REACHABILITY_MAX_HOPS` hops (a correlated recursive CTE walking ``edges``
  backwards; ``catalog_edges`` never leads back to data, invariant I1);
- a :LabeledEntity is an orphan when no relationship points at it.

The collector holds opaque ``(label, key)`` handles (the ``elementId`` of Neo4j). Every
delete goes :data:`DELETE_BATCH_ROWS` per transaction, retried whole. The tenant-wide pass
reads every orphan of the tenant first and then deletes them in batches (the eager
``CALL { … } IN TRANSACTIONS`` of Cypher).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from scinr.newton.graph.base import GcOps, GcPassKind
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._traversal import reaches_from
from scinr.newton.graph.sql.deletion import document_keys, subtree_nodes
from scinr.newton.graph.types import (  # noqa: F401 — the GC bounds live there
    DELETE_BATCH_ROWS,
    GC_FLUSH_CANDIDATES,
    GC_MAX_PASSES,
    GC_REACHABILITY_MAX_HOPS,
)
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import sql_errors
from scinr.newton.sql.retry import with_sql_retry_sync

__all__ = ["SqlGcOps", "SqlOrphanCollector", "run_gc_pass", "sweep_tenant"]

logger = logging.getLogger(__name__)

E = t.edges

Handle = tuple[str, str]

_EMI_LABELS = ("Entity", "ModelInstance")
_LE_LABEL = "LabeledEntity"
_CANDIDATE_LABELS = (*_EMI_LABELS, _LE_LABEL)


def _orphan_condition(label: str) -> Any:
    """The orphan rule of *label*, correlated with the rows of its table."""
    key = c.key_column(label)
    if label == _LE_LABEL:
        return ~sa.exists().where(E.c.dst_label == label, E.c.dst_key == key)
    return ~reaches_from(
        sa.literal(label, sa.Text),
        key,
        source_label="ExtractionResult",
        max_hops=GC_REACHABILITY_MAX_HOPS,
        name="gc_reach",
    )


def _orphans_of(label: str, tenant: str, keys: Sequence[str] | None) -> sa.Select:
    """Keys of the orphans of *label* in *tenant* (among *keys*, or every one)."""
    target = c.table(label)
    key = c.key_column(label)
    statement = sa.select(key).where(target.c.tenant_id == tenant, _orphan_condition(label))
    if keys is not None:
        statement = statement.where(key.in_(list(keys)))
    return statement.order_by(c.dialect().order_text(key))


def _candidates(dst: sa.Select, tenant: str) -> sa.CompoundSelect:
    """The ``(label, key)`` among the rows of *dst* that are candidates of the tenant
    (Entity, ModelInstance or LabeledEntity with ``tenant_id = $tenant_id``)."""
    sub = dst.subquery()
    parts = []
    for label in _CANDIDATE_LABELS:
        target = c.table(label)
        key = c.key_column(label)
        parts.append(
            sa.select(sa.literal(label, sa.Text).label("label"), key.label("key")).where(
                key.in_(sa.select(sub.c.key).where(sub.c.label == label)),
                target.c.tenant_id == tenant,
            )
        )
    return sa.union(*parts)


def _rows(connection: Connection, statement: Any) -> list[Handle]:
    return [(row[0], row[1]) for row in connection.execute(statement)]


class SqlOrphanCollector:
    """Collects the orphans one operation leaves, from its candidates (as Neo4j's)."""

    def __init__(self, *, tenant_id: str, caller: str) -> None:
        self._tenant = tenant_id
        self._caller = caller
        self._entity_model_instance: set[Handle] = set()
        self._labeled_entity: set[Handle] = set()
        self._counts = {
            "gc_entity_model_instance_deleted": 0,
            "gc_entity_model_instance_passes": 0,
            "gc_labeled_entity_deleted": 0,
            "gc_labeled_entity_passes": 0,
        }

    def _add(self, handle: Handle) -> None:
        (self._labeled_entity if handle[0] == _LE_LABEL else self._entity_model_instance).add(
            handle
        )

    @sql_errors()
    def gather(self, keys: Sequence[dict[str, Any]]) -> None:
        """Add what the ExtractionResults of the documents at *keys* point at (one hop).
        Call it before they are deleted.

        Raises:
            GraphError: If the engine fails.
        """
        results = c.follow(
            subtree_nodes(document_keys(self._tenant, keys)),
            [("HAS_EXTRACTION", "ExtractionResult")],
        )
        statement = _candidates(c.follow(results, [(None, None)]), self._tenant)
        for handle in c.sync_read(lambda connection: _rows(connection, statement)):
            self._add(handle)

    def chunk_done(self) -> None:
        """The ExtractionResults of every gathered document are gone: collect now if enough
        candidates have piled up."""
        if len(self._entity_model_instance) + len(self._labeled_entity) >= GC_FLUSH_CANDIDATES:
            self.collect()

    def _run_round(self, handles: set[Handle]) -> int:
        """Delete the orphans among *handles*; what they pointed at becomes a candidate."""
        ordered = sorted(handles)
        deleted = 0
        for batch in c.batched(ordered, DELETE_BATCH_ROWS):
            by_label = c.group_by_label(batch)

            def read(
                connection: Connection, by_label: dict[str, list[str]] = by_label
            ) -> tuple[list[Handle], list[Handle]]:
                orphans: list[Handle] = []
                for label, keys in by_label.items():
                    orphans += [
                        (label, key)
                        for key in c.column_values(
                            connection.execute(_orphans_of(label, self._tenant, keys))
                        )
                    ]
                if not orphans:
                    return [], []
                dependents: list[Handle] = []
                for label, keys in c.group_by_label(orphans).items():
                    dst = c.follow(c.nodes_with_keys(label, keys), [(None, None)])
                    dependents += _rows(connection, _candidates(dst, self._tenant))
                return orphans, dependents

            orphans, dependents = c.sync_read(read)
            if not orphans:
                continue
            for handle in dependents:
                self._add(handle)

            def delete(connection: Connection, orphans: list[Handle] = orphans) -> int:
                count = 0
                for label, keys in c.group_by_label(orphans).items():
                    # Checked again: a node linked in between is left alone.
                    still = c.column_values(
                        connection.execute(_orphans_of(label, self._tenant, keys))
                    )
                    if not still:
                        continue
                    statements = c.detach_delete(label, still)
                    connection.execute(statements[0])
                    count += connection.execute(statements[1]).rowcount
                return count

            deleted += c.sync_write(delete)
        return deleted

    @sql_errors()
    def collect(self) -> None:
        """Delete every candidate that is an orphan, round after round.

        Raises:
            GraphError: If the engine fails.
        """
        rounds = (
            ("_entity_model_instance", "gc_entity_model_instance", "Entity|ModelInstance"),
            ("_labeled_entity", "gc_labeled_entity", "LabeledEntity"),
        )
        while self._entity_model_instance or self._labeled_entity:
            for attribute, counter, label in rounds:
                handles = getattr(self, attribute)
                if not handles:
                    continue
                setattr(self, attribute, set())
                deleted = self._run_round(handles)
                self._counts[f"{counter}_deleted"] += deleted
                self._counts[f"{counter}_passes"] += 1
                logger.info(
                    "%s: GC (%s) deleted %d of %d candidate(s).",
                    self._caller,
                    label,
                    deleted,
                    len(handles),
                )

    def counts(self) -> dict[str, int]:
        """The ``gc_*`` counters shared by ``DeletionResult`` and ``FreezeResult``."""
        return dict(self._counts)


# ── Tenant-wide sweep ────────────────────────────────────────────────────────

_PASS_LABELS: dict[str, tuple[tuple[str, ...], str]] = {
    "entity_model_instance": (_EMI_LABELS, "Entity|ModelInstance"),
    "labeled_entity": ((_LE_LABEL,), "LabeledEntity"),
}


def _sweep_once(labels: Sequence[str], tenant_id: str) -> int:
    """Every orphan of *labels* in the tenant, read first, deleted in batches."""
    deleted = 0
    for label in labels:
        statement = _orphans_of(label, tenant_id, None)
        keys = c.sync_read(
            lambda connection, statement=statement: c.column_values(connection.execute(statement)),
            retry=False,
        )
        for batch in c.batched(keys, DELETE_BATCH_ROWS):

            def delete(connection: Connection, label: str = label, batch: list[str] = batch) -> int:
                statements = c.detach_delete(label, batch)
                connection.execute(statements[0])
                return connection.execute(statements[1]).rowcount

            deleted += c.sync_write(delete, retry=False)
    return deleted


def run_gc_pass(kind: str, *, tenant_id: str, caller: str) -> tuple[int, int]:
    """One tenant-wide pass of *kind* up to :data:`GC_MAX_PASSES` times, stopping as soon
    as an execution deletes nothing. Returns ``(deleted, iterations)``."""
    labels, label_text = _PASS_LABELS[kind]
    total_deleted = 0
    passes_run = 0

    def iteration() -> int:
        try:
            return _sweep_once(labels, tenant_id)
        except Exception:
            logger.exception(
                "%s: GC pass (%s) iteration %d raised an exception.",
                caller,
                label_text,
                passes_run + 1,
            )
            raise

    for _ in range(GC_MAX_PASSES):
        deleted = with_sql_retry_sync(iteration)
        passes_run += 1
        total_deleted += deleted
        logger.info(
            "%s: GC pass (%s) iteration %d deleted %d node(s).",
            caller,
            label_text,
            passes_run,
            deleted,
        )
        if deleted == 0:
            break
    return total_deleted, passes_run


def sweep_tenant(*, tenant_id: str, caller: str) -> dict[str, int]:
    """Both passes in order (:Entity/:ModelInstance, then :LabeledEntity)."""
    emi_deleted, emi_passes = run_gc_pass(
        "entity_model_instance", tenant_id=tenant_id, caller=caller
    )
    le_deleted, le_passes = run_gc_pass("labeled_entity", tenant_id=tenant_id, caller=caller)
    return {
        "gc_entity_model_instance_deleted": emi_deleted,
        "gc_entity_model_instance_passes": emi_passes,
        "gc_labeled_entity_deleted": le_deleted,
        "gc_labeled_entity_passes": le_passes,
    }


class SqlGcOps(GcOps):
    """``store.gc`` over SQL (synchronous, process-wide sync engine)."""

    def collector(self, tenant: str, caller: str) -> SqlOrphanCollector:
        """A new :class:`SqlOrphanCollector` for one operation."""
        return SqlOrphanCollector(tenant_id=tenant, caller=caller)

    @sql_errors()
    def sweep_tenant(self, tenant_id: str, caller: str) -> dict[str, int]:
        """Collect every orphan of the tenant (see :func:`sweep_tenant`).

        Raises:
            GraphError: If the engine fails.
        """
        return sweep_tenant(tenant_id=tenant_id, caller=caller)

    @sql_errors()
    def run_pass(self, kind: GcPassKind, tenant_id: str, caller: str) -> tuple[int, int]:
        """One tenant-wide pass of *kind* (see :func:`run_gc_pass`).

        Raises:
            GraphError: If the engine fails.
        """
        return run_gc_pass(kind, tenant_id=tenant_id, caller=caller)
