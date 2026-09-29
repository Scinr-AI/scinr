"""navigation/neo4j/_info_units.py — Group C: InfoUnits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from scinr.newton.navigation.models import (
    DocumentRef,
    InfoUnitRef,
    ScoredInfoUnit,
    StructureNodeRef,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.scope import make_scope, resolve_selector

_FT_INDEX = {"description": "infoUnitDescription", "title": "infoUnitTitle"}


class _InfoUnitsMixin(_Neo4jRuntime):
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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        key = "u.order, u.uid" if order_by == "order" else "u.title, u.uid"
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("sn", params, sc)
        clauses: list[str] = []
        self._scope_where("u", clauses, params, sc, tenant=False)
        u_where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rows = await self._read(
            f"MATCH (sn:StructureNode {{id: $node_id}}) {anchor_where}"
            f"MATCH (sn)-[:HAS_INFO_UNIT]->(u:InfoUnit) {u_where}"
            f"RETURN u ORDER BY {key}",
            **params,
        )
        return [_map.info_unit_ref(r["u"]) for r in rows]

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        clauses: list[str] = []
        self._scope_where("u", clauses, params, sc, tenant=False)
        u_where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rec = await self._read_one(
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(:StructureNode)-[:HAS_INFO_UNIT]->(u:InfoUnit) "
            f"{u_where}RETURN count(u) AS c",
            **params,
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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        indexes = (
            [_FT_INDEX["title"], _FT_INDEX["description"]]
            if field == "both"
            else [_FT_INDEX[field]]
        )
        params: dict[str, Any] = {"q": text, "lim": int(limit)}
        clauses: list[str] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                "EXISTS { MATCH (sn)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) }"
            )
            params["doc_path"] = path
        # Scope is applied to the hits *before* the LIMIT, so a scoped search
        # still returns up to `limit` rows of the scope (not `limit` minus
        # the ones filtered away).
        self._scope_where("node", clauses, params, sc)
        doc_filter = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        best: dict[str, ScoredInfoUnit] = {}
        for index in indexes:
            rows = await self._read(
                "CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score "
                "MATCH (sn:StructureNode)-[:HAS_INFO_UNIT]->(node) "
                f"{doc_filter}"
                "RETURN node, score, sn.id AS node_id, sn.title AS node_title "
                "ORDER BY score DESC LIMIT $lim",
                index=index,
                **params,
            )
            for r in rows:
                iu = _map.scored_info_unit(
                    r["node"], score=r["score"], node_id=r["node_id"], node_title=r["node_title"]
                )
                cur = best.get(iu.uid)
                if cur is None or iu.score > cur.score:
                    best[iu.uid] = iu
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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses: list[str] = []
        params: dict[str, Any] = {"uid": uid}
        self._scope_where("u", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rec = await self._read_one(
            f"MATCH (u:InfoUnit {{uid: $uid}}) {where_sql}RETURN u", **params
        )
        return _map.info_unit_ref(rec["u"]) if rec else None

    async def get_node_for_info_unit(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureNodeRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid}
        u_where = self._doc_where("u", params, sc, "scalar")
        rec = await self._read_one(
            f"MATCH (u:InfoUnit {{uid: $uid}}) {u_where}"
            "MATCH (n:StructureNode)-[:HAS_INFO_UNIT]->(u) "
            "RETURN n { .*, _labels: labels(n) } AS n LIMIT 1",
            **params,
        )
        return _map.structure_node_ref(rec["n"]) if rec else None
