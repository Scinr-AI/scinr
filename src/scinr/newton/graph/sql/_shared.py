"""
graph/sql/_shared.py — Steps shared by several SQL operation groups (annotation, extraction,
tabular, restore).

Translation of ``graph/neo4j/_shared.py``. The steps receive the ``AsyncConnection`` of the
caller, already inside the caller's transaction: a caller that writes in autocommit (as
``write_annotation`` does) opens one transaction per step, like one Cypher statement each.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._traversal import reachable
from scinr.newton.graph.types import Provenance, RawProvenance

__all__ = [
    "STALE_EXTRACTION_MAX_HOPS",
    "STALE_MODEL_DECISION_STEPS",
    "delete_found",
    "delete_stale_extraction_result",
    "delete_stale_model_decision",
    "node_provenance",
    "prov_values",
    "provenance_on_create",
    "provenance_on_match",
]

#: Hop bound of the ModelInstance purge of ``delete_stale_extraction_result`` (Cypher ``*1..10``).
STALE_EXTRACTION_MAX_HOPS = 10

_MD = ("HAS_MODEL_DECISION", "ModelDecision")
_PM = ("HAS_PROPOSED_MODEL", "ProposedModel")

Step = Callable[[AsyncConnection, list[str]], Awaitable[None]]


def _ids(node_ids: str | Sequence[str]) -> list[str]:
    return [node_ids] if isinstance(node_ids, str) else list(node_ids)


async def delete_found(connection: AsyncConnection, found: sa.Select) -> int:
    """``DETACH DELETE`` every ``(label, key)`` row of *found*; returns the nodes deleted."""
    deleted = 0
    for label, keys in c.group_by_label(await connection.execute(found)).items():
        statements = c.detach_delete(label, keys)
        for statement in statements[:-1]:
            await connection.execute(statement)
        deleted += (await connection.execute(statements[-1])).rowcount
    return deleted


def _path_step(path: list[c.Step]) -> Step:
    async def step(connection: AsyncConnection, node_ids: list[str]) -> None:
        await delete_found(connection, c.follow(c.nodes_with_keys("StructureNode", node_ids), path))

    return step


#: The five statements of ``_STALE_MODEL_DECISION_DELETE_QUERIES``, leaf first:
#: ProposedField, ProposedModel, SupplementaryField, ComplementaryMatch, ModelDecision.
STALE_MODEL_DECISION_STEPS: tuple[Step, ...] = (
    _path_step([_MD, _PM, ("HAS_PROPOSED_FIELD", None)]),
    _path_step([_MD, _PM]),
    _path_step([_MD, ("HAS_SUPPLEMENTARY_FIELD", None)]),
    _path_step([_MD, ("HAS_COMPLEMENTARY_MATCH", "ComplementaryMatch")]),
    _path_step([_MD]),
)


async def delete_stale_model_decision(
    connection: AsyncConnection, node_ids: str | Sequence[str]
) -> None:
    """Delete the ModelDecision subtree(s) hanging from the StructureNode(s), on *connection*.

    Cypher: the five leaf-first DETACH DELETE statements of ``_STALE_MODEL_DECISION_DELETE_QUERIES``.
    :CatalogModel singletons are never deleted. A ModelDecision shared by several
    StructureNodes (the tabular one) goes as soon as one of them is passed.
    """
    ids = _ids(node_ids)
    for step in STALE_MODEL_DECISION_STEPS:
        await step(connection, ids)


async def delete_stale_extraction_result(
    connection: AsyncConnection, node_ids: str | Sequence[str]
) -> None:
    """Delete the ExtractionResult(s) of the StructureNode(s) and every :ModelInstance
    reachable from them within 10 hops (one statement in Cypher, one transaction here).

    Cypher: MATCH (n {id})-[:HAS_EXTRACTION]->(er) OPTIONAL MATCH (er)-[*1..10]->(child:
    ModelInstance) DETACH DELETE child, er. Like Neo4j, it does not check whether another
    ExtractionResult still reaches the instance (the GC cleans up what is left orphaned).
    """
    ids = _ids(node_ids)
    results = c.follow(
        c.nodes_with_keys("StructureNode", ids), [("HAS_EXTRACTION", "ExtractionResult")]
    )
    er_keys = c.column_values(await connection.execute(sa.select(results.subquery().c.key)))
    if not er_keys:
        return
    below = reachable(
        c.nodes_with_keys("ExtractionResult", er_keys),
        max_hops=STALE_EXTRACTION_MAX_HOPS,
        name="stale",
    )
    children = c.column_values(
        await connection.execute(
            sa.select(below.c.key).where(below.c.label == "ModelInstance").distinct()
        )
    )
    for batch in c.batched(children, c.IN_BATCH):
        for statement in c.detach_delete("ModelInstance", batch):
            await connection.execute(statement)
    for statement in c.detach_delete("ExtractionResult", er_keys):
        await connection.execute(statement)


# ── Provenance ───────────────────────────────────────────────────────────────


async def node_provenance(
    connection: AsyncConnection, node_id: str
) -> tuple[RawProvenance, str | None] | None:
    """``(provenance, _document_key)`` of a StructureNode as stored, or ``None`` if missing."""
    sn = c.table("StructureNode")
    row = (
        await connection.execute(
            sa.select(
                sn.c.tenant_id, sn.c.created_by_user_id, sn.c.job_id, sn.c._document_key
            ).where(sn.c.id == node_id)
        )
    ).first()
    if row is None:
        return None
    return RawProvenance(row.tenant_id, row.created_by_user_id, row.job_id), row._document_key


def prov_values(prov: RawProvenance | Provenance) -> dict[str, Any]:
    """The scalar provenance properties copied onto a per-document node (``null`` removes)."""
    return {
        "tenant_id": prov.tenant_id,
        "created_by_user_id": prov.created_by_user_id,
        "job_id": prov.job_id,
    }


def provenance_on_create(prov: Provenance) -> dict[str, Any]:
    """``provenance_set_clause`` of a node being created: tenant plus one-element lists."""
    return {
        "tenant_id": prov.tenant_id,
        "created_by_user_ids": [] if prov.created_by_user_id is None else [prov.created_by_user_id],
        "job_ids": [] if prov.job_id is None else [prov.job_id],
    }


def provenance_on_match(target: sa.Table, prov: Provenance) -> dict[str, Any]:
    """``provenance_set_clause`` of an existing node: tenant plus append-unique of each list."""
    d = c.dialect()
    return {
        "tenant_id": prov.tenant_id,
        "created_by_user_ids": d.append_unique(
            target.c.created_by_user_ids, prov.created_by_user_id
        ),
        "job_ids": d.append_unique(target.c.job_ids, prov.job_id),
    }
