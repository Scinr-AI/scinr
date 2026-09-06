"""navigation/neo4j/_info_units.py — Group C: InfoUnits."""

from __future__ import annotations

from typing import Any, Literal

from scinr.newton.navigation.models import (
    DocumentRef,
    InfoUnitRef,
    ScoredInfoUnit,
    StructureNodeRef,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime, selector_path

_FT_INDEX = {"description": "infoUnitDescription", "title": "infoUnitTitle"}


class _InfoUnitsMixin(_Neo4jRuntime):
    async def get_info_units(
        self, node_id: str, *, order_by: str = "order"
    ) -> list[InfoUnitRef]:
        key = "u.order, u.uid" if order_by == "order" else "u.title, u.uid"
        rows = await self._read(
            "MATCH (:StructureNode {id: $node_id})-[:HAS_INFO_UNIT]->(u:InfoUnit) "
            f"RETURN u ORDER BY {key}",
            node_id=node_id,
        )
        return [_map.info_unit_ref(r["u"]) for r in rows]

    async def count_info_units(
        self, document: str | DocumentRef, *, version: int | None = None, depth: int | None = None
    ) -> int:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        rec = await self._read_one(
            f"MATCH {self._doc_match('doc', version=version)} "
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(:StructureNode)-[:HAS_INFO_UNIT]->(u:InfoUnit) "
            "RETURN count(u) AS c",
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
    ) -> list[ScoredInfoUnit]:
        indexes = (
            [_FT_INDEX["title"], _FT_INDEX["description"]]
            if field == "both"
            else [_FT_INDEX[field]]
        )
        doc_filter = ""
        params: dict[str, Any] = {"q": text, "lim": int(limit)}
        if document is not None:
            doc_filter = (
                "WHERE EXISTS { MATCH (sn)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) } "
            )
            params["doc_path"] = selector_path(document)
        best: dict[str, ScoredInfoUnit] = {}
        for index in indexes:
            rows = await self._read(
                f"CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score "
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

    async def get_info_unit(self, uid: str) -> InfoUnitRef | None:
        rec = await self._read_one(
            "MATCH (u:InfoUnit {uid: $uid}) RETURN u", uid=uid
        )
        return _map.info_unit_ref(rec["u"]) if rec else None

    async def get_node_for_info_unit(self, uid: str) -> StructureNodeRef | None:
        rec = await self._read_one(
            "MATCH (n:StructureNode)-[:HAS_INFO_UNIT]->(:InfoUnit {uid: $uid}) "
            "RETURN n { .*, _labels: labels(n) } AS n LIMIT 1",
            uid=uid,
        )
        return _map.structure_node_ref(rec["n"]) if rec else None
