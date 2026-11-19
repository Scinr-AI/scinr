"""navigation/sql/_info_units.py — Group C: InfoUnits (SQL)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import sqlalchemy as sa

from scinr.newton.navigation import _map
from scinr.newton.navigation.models import (
    DocumentRef,
    InfoUnitRef,
    ScoredInfoUnit,
    StructureNodeRef,
)
from scinr.newton.navigation.scope import make_scope, resolve_selector
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._walk import reached, walk
from scinr.newton.sql import tables as T

_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")


class _InfoUnitsMixin(_SqlRuntime):
    @staticmethod
    def _unit_refs(rows: list[dict[str, Any]]) -> list[InfoUnitRef]:
        return [_map.info_unit_ref(_SqlRuntime._node("info_units", r)) for r in rows]

    async def get_info_units(
        self,
        node_id: str,
        *,
        order_by: str = "order",
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[InfoUnitRef]:
        """The info units of a structure node, by ``order`` (or by title)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn, iu, e = T.structure_nodes, T.info_units, T.edges
        anchor = sa.select(sn.c.id).where(sn.c.id == node_id, *self._scope(sn, sc, "tenant_only"))
        linked = sa.select(e.c.dst_key).where(
            e.c.type == "HAS_INFO_UNIT",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(anchor),
            e.c.dst_label == "InfoUnit",
        )
        if order_by == "order":
            ordering = (self._dl.order_value(iu.c.order), self._dl.order_text(iu.c.uid))
        else:
            ordering = (self._dl.order_text(iu.c.title), self._dl.order_text(iu.c.uid))
        stmt = (
            sa.select(iu)
            .where(iu.c.uid.in_(linked), *self._scope(iu, sc, tenant=False))
            .order_by(*ordering)
        )
        return self._unit_refs(await self._read(stmt))

    async def count_info_units(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Number of info units of the structure nodes below a document."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        iu, e = T.info_units, T.edges
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_STRUCTURE_EDGES)
        below = reached(cte, "StructureNode")
        linked = sa.select(e.c.dst_key).where(
            e.c.type == "HAS_INFO_UNIT",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(below.c.key)),
            e.c.dst_label == "InfoUnit",
        )
        rec = await self._read_one(
            sa.select(sa.func.count().label("c")).where(
                iu.c.uid.in_(linked), *self._scope(iu, sc, tenant=False)
            )
        )
        return int(rec["c"]) if rec else 0

    async def search_info_units(
        self,
        text: str,
        *,
        field: Literal["title", "description", "both"] = "both",
        document: str | DocumentRef | None = None,
        limit: int = 25,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ScoredInfoUnit]:
        """Full-text search over info-unit titles and/or descriptions.

        The query syntax is the engine's, not Lucene's: Postgres reads *text* with
        ``websearch_to_tsquery`` (words are AND-ed, ``"phrases"``, ``or``, ``-negation``);
        SQLite translates it to FTS5 (words and ``"phrases"`` AND-ed, ``or``; negation is
        ignored). The ``score`` is the engine's own relevance and is only comparable within
        one engine; results with several fields keep the best score of each unit.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        fields: list[Literal["title", "description"]] = (
            ["title", "description"] if field == "both" else [field]
        )
        iu, sn, e = T.info_units, T.structure_nodes, T.edges
        best: dict[str, ScoredInfoUnit] = {}
        for index in fields:
            hits = self._dl.fulltext_units(index, text).subquery("hits")
            conditions: list[sa.ColumnElement[bool]] = []
            if document is not None:
                path, sc = resolve_selector(document, sc)
                conditions.append(self._in_document_path(path, sn.c.id))
            # Scope is applied to the hits *before* the LIMIT, so a scoped search still
            # returns up to `limit` rows of the scope.
            conditions.extend(self._scope(iu, sc))
            stmt = (
                sa.select(
                    iu,
                    hits.c.score.label("_score"),
                    sn.c.id.label("_node_id"),
                    sn.c.title.label("_node_title"),
                )
                .select_from(
                    hits.join(iu, iu.c.uid == hits.c.uid)
                    .join(
                        e,
                        sa.and_(
                            e.c.type == "HAS_INFO_UNIT",
                            e.c.src_label == "StructureNode",
                            e.c.dst_label == "InfoUnit",
                            e.c.dst_key == iu.c.uid,
                        ),
                    )
                    .join(sn, sn.c.id == e.c.src_key)
                )
                .where(*conditions)
                .order_by(hits.c.score.desc(), self._dl.order_text(iu.c.uid))
                .limit(int(limit))
            )
            for row in await self._read(stmt):
                hit = _map.scored_info_unit(
                    self._node("info_units", row),
                    score=row["_score"],
                    node_id=row["_node_id"],
                    node_title=row["_node_title"],
                )
                current = best.get(hit.uid)
                if current is None or hit.score > current.score:
                    best[hit.uid] = hit
        return sorted(best.values(), key=lambda x: x.score, reverse=True)[: int(limit)]

    async def get_info_unit(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> InfoUnitRef | None:
        """The info unit with this uid, if it is in scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        iu = T.info_units
        rec = await self._read_one(
            sa.select(iu).where(iu.c.uid == uid, *self._scope(iu, sc)).limit(1)
        )
        return self._unit_refs([rec])[0] if rec else None

    async def get_node_for_info_unit(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureNodeRef | None:
        """The structure node an info unit belongs to."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        iu, sn, e = T.info_units, T.structure_nodes, T.edges
        anchor = sa.select(iu.c.uid).where(iu.c.uid == uid, *self._scope(iu, sc, "scalar"))
        owners = sa.select(e.c.src_key).where(
            e.c.type == "HAS_INFO_UNIT",
            e.c.src_label == "StructureNode",
            e.c.dst_label == "InfoUnit",
            e.c.dst_key.in_(anchor),
        )
        rec = await self._read_one(
            sa.select(sn).where(sn.c.id.in_(owners)).order_by(self._dl.order_text(sn.c.id)).limit(1)
        )
        if rec is None:
            return None
        return _map.structure_node_ref(self._node("structure_nodes", rec))


__all__ = ["_InfoUnitsMixin"]
