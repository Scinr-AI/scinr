"""navigation/neo4j/_power.py — Group H: generic power tools + execute_raw."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import GraphNode, NodeSelector, PathResult, Subgraph
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._safe import assert_read_only, safe_ident


def _sel(selector: NodeSelector, alias: str) -> tuple[str, dict[str, Any]]:
    t = safe_ident(selector.type, kind="node type")
    k = safe_ident(selector.key, kind="selector key")
    pname = f"{alias}_val"
    return f"({alias}:{t} {{`{k}`: ${pname}}})", {pname: selector.value}


class _PowerMixin(_Neo4jRuntime):
    async def neighbors(
        self,
        selector: NodeSelector,
        *,
        edge_types: Sequence[str] | None = None,
        direction: Literal["out", "in", "both"] = "both",
        target_types: Sequence[str] | None = None,
        depth: int | None = 1,
        limit: int | None = None,
    ) -> list[GraphNode]:
        d = self._resolve_depth(depth)
        pat, params = _sel(selector, "s")
        if direction == "out":
            hop = f"(s)-[r*1..{d}]->(o)"
        elif direction == "in":
            hop = f"(o)-[r*1..{d}]->(s)"
        else:
            hop = f"(s)-[r*1..{d}]-(o)"
        params["edge_types"] = list(edge_types) if edge_types else None
        params["target_types"] = list(target_types) if target_types else None
        rows = await self._read(
            f"MATCH {pat} MATCH {hop} "
            "WHERE ($edge_types IS NULL OR all(x IN r WHERE type(x) IN $edge_types)) "
            "AND ($target_types IS NULL OR any(l IN labels(o) WHERE l IN $target_types)) "
            f"RETURN DISTINCT o {{ .*, _labels: labels(o) }} AS o{self._limit_clause(limit)}",
            **params,
        )
        return [_map.graph_node(r["o"]) for r in rows]

    async def shortest_path(
        self,
        from_selector: NodeSelector,
        to_selector: NodeSelector,
        *,
        max_hops: int = 6,
        edge_types: Sequence[str] | None = None,
    ) -> PathResult | None:
        h = int(max_hops)
        if h < 1:
            raise NavigationError("max_hops must be >= 1")
        pa, params_a = _sel(from_selector, "a")
        pb, params_b = _sel(to_selector, "b")
        params: dict[str, Any] = {**params_a, **params_b}
        params["edge_types"] = list(edge_types) if edge_types else None
        rec = await self._read_one(
            f"MATCH {pa}, {pb} "
            f"MATCH p = shortestPath( (a)-[*..{h}]-(b) ) "
            "WHERE $edge_types IS NULL OR all(x IN relationships(p) WHERE type(x) IN $edge_types) "
            "RETURN [n IN nodes(p) | n { .*, _labels: labels(n) }] AS nodes, "
            "[r IN relationships(p) | {type: type(r), props: properties(r)}] AS rels, "
            "length(p) AS length LIMIT 1",
            **params,
        )
        if not rec:
            return None
        return PathResult(
            raw=dict(rec),
            length=int(rec["length"]),
            nodes=[_map.graph_node(n) for n in rec["nodes"]],
            relationships=list(rec["rels"]),
        )

    async def subgraph(
        self,
        selector: NodeSelector,
        *,
        depth: int = 2,
        edge_types: Sequence[str] | None = None,
        max_nodes: int = 500,
    ) -> Subgraph:
        d = int(depth)
        if d < 1:
            raise NavigationError("depth must be >= 1")
        pat, params = _sel(selector, "s")
        params["max_nodes"] = int(max_nodes)
        rel_filter = "|".join(safe_ident(e, kind="edge type") for e in edge_types) if edge_types else None
        params["rel_filter"] = rel_filter
        try:
            rec = await self._read_one(
                f"MATCH {pat} "
                "CALL apoc.path.subgraphAll(s, {maxLevel: $lvl, relationshipFilter: $rel_filter, "
                "limit: $max_nodes}) YIELD nodes, relationships "
                "RETURN [n IN nodes | n { .*, _labels: labels(n) }] AS nodes, "
                "[r IN relationships | {type: type(r), props: properties(r), "
                "start: elementId(startNode(r)), end: elementId(endNode(r))}] AS rels",
                lvl=d,
                **params,
            )
        except Exception:  # noqa: BLE001 — APOC missing → pure-Cypher fallback
            rec = await self._read_one(
                f"MATCH {pat} "
                f"MATCH p = (s)-[*0..{d}]-(o) "
                "WITH collect(DISTINCT o)[0..$max_nodes] AS ns "
                "UNWIND ns AS n1 "
                "RETURN [n IN ns | n { .*, _labels: labels(n) }] AS nodes, "
                "[ (n1)-[e]->(n2) WHERE n2 IN ns | {type: type(e), props: properties(e)} ] AS rels",
                **params,
            )
        if not rec:
            return Subgraph(raw={})
        return Subgraph(
            raw={"n_nodes": len(rec["nodes"])},
            nodes=[_map.graph_node(n) for n in rec["nodes"]],
            edges=list(rec["rels"] or []),
        )

    # -- raw escape hatch --------------------------------------------------

    async def execute_raw(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        dialect: str | None = None,
    ) -> list[dict[str, Any]]:
        if dialect is not None and dialect != self.dialect:
            raise NavigationError(
                f"execute_raw called with dialect={dialect!r} on a {self.dialect!r} backend"
            )
        assert_read_only(query)
        return await self._read(query, **dict(params or {}))
