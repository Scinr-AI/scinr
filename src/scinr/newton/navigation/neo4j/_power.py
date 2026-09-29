"""navigation/neo4j/_power.py — Group H: generic power tools + execute_raw."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import GraphNode, NodeSelector, PathResult, Subgraph
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._safe import assert_read_only, safe_ident
from scinr.newton.navigation.scope import Scope, make_scope

#: Global catalogue labels: they carry no tenant, and are the bridge through
#: which a free graph walk could hop from one tenant's data to another's
#: (``ModelDecision(A) -[:MATCHED_MODEL]-> CatalogModel <-[:MATCHED_MODEL]- ModelDecision(B)``).
_CATALOG_LABELS = ["CatalogModel", "ModelField", "EntityLabel", "Theme"]

_IS_CATALOG = "any(l IN labels({a}) WHERE l IN $scope_catalog_labels)"


def _sel(selector: NodeSelector, alias: str) -> tuple[str, dict[str, Any]]:
    t = safe_ident(selector.type, kind="node type")
    k = safe_ident(selector.key, kind="selector key")
    pname = f"{alias}_val"
    return f"({alias}:{t} {{`{k}`: ${pname}}})", {pname: selector.value}


def _prov_any(alias: str, sc: Scope) -> list[str]:
    """User / job predicates that work on either provenance form (scalar or array)."""
    out: list[str] = []
    if sc.user_ids is not None:
        out.append(
            f"({alias}.created_by_user_id IN $scope_user_ids OR "
            f"any(x IN $scope_user_ids WHERE x IN coalesce({alias}.created_by_user_ids, [])))"
        )
    if sc.job_ids is not None:
        out.append(
            f"({alias}.job_id IN $scope_job_ids OR "
            f"any(x IN $scope_job_ids WHERE x IN coalesce({alias}.job_ids, [])))"
        )
    return out


def _allowed(alias: str, sc: Scope) -> str | None:
    """Predicate "*alias* may be seen": a catalogue node, or a node inside the scope."""
    parts = list(_prov_any(alias, sc))
    tenant = sc.tenant_clause(alias)
    if tenant:
        parts.insert(0, tenant)
    if not parts:
        return None
    return f"({_IS_CATALOG.format(a=alias)} OR ({' AND '.join(parts)}))"


def _scope_params(sc: Scope) -> dict[str, Any]:
    return {**sc.params(), "scope_catalog_labels": _CATALOG_LABELS}


class _PowerMixin(_Neo4jRuntime):
    # Scope notes (navigation/scope.py). These are free graph walks, so the
    # anchor is checked and — when a tenant is given — every intermediate node
    # of a path must belong to it (catalogue nodes are only allowed as the far
    # end, never as a bridge). With ``tenant_id=None`` ("all") nothing restricts
    # the walk. ``execute_raw`` cannot be scoped: it is an administrative tool
    # outside the isolation contract.

    async def neighbors(
        self,
        selector: NodeSelector,
        *,
        edge_types: Sequence[str] | None = None,
        direction: Literal["out", "in", "both"] = "both",
        target_types: Sequence[str] | None = None,
        depth: int | None = 1,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[GraphNode]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        pat, params = _sel(selector, "s")
        if direction == "out":
            hop = f"p = (s)-[r*1..{d}]->(o)"
        elif direction == "in":
            hop = f"p = (o)-[r*1..{d}]->(s)"
        else:
            hop = f"p = (s)-[r*1..{d}]-(o)"
        params["edge_types"] = list(edge_types) if edge_types else None
        params["target_types"] = list(target_types) if target_types else None
        params.update(_scope_params(sc))
        clauses = [
            "($edge_types IS NULL OR all(x IN r WHERE type(x) IN $edge_types))",
            "($target_types IS NULL OR any(l IN labels(o) WHERE l IN $target_types))",
        ]
        anchor = _allowed("s", sc)
        target = _allowed("o", sc)
        if anchor:
            clauses.append(anchor)
        if target:
            clauses.append(target)
        tenant_mid = sc.tenant_clause("m")
        if tenant_mid:
            # nothing but the far end may be a catalogue node / another tenant's
            clauses.append(f"all(m IN nodes(p)[1..-1] WHERE {tenant_mid})")
        rows = await self._read(
            f"MATCH {pat} MATCH {hop} "
            f"WHERE {' AND '.join(clauses)} "
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
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> PathResult | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        h = int(max_hops)
        if h < 1:
            raise NavigationError("max_hops must be >= 1")
        pa, params_a = _sel(from_selector, "a")
        pb, params_b = _sel(to_selector, "b")
        params: dict[str, Any] = {**params_a, **params_b}
        params["edge_types"] = list(edge_types) if edge_types else None
        params.update(_scope_params(sc))
        clauses = ["($edge_types IS NULL OR all(x IN relationships(p) WHERE type(x) IN $edge_types))"]
        for alias in ("a", "b"):
            allowed = _allowed(alias, sc)
            if allowed:
                clauses.append(allowed)
        tenant_mid = sc.tenant_clause("m")
        if tenant_mid:
            # a catalogue node may only be one of the two endpoints
            clauses.append(f"all(m IN nodes(p) WHERE m = a OR m = b OR {tenant_mid})")
        rec = await self._read_one(
            f"MATCH {pa}, {pb} "
            f"MATCH p = shortestPath( (a)-[*..{h}]-(b) ) "
            f"WHERE {' AND '.join(clauses)} "
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
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> Subgraph:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = int(depth)
        if d < 1:
            raise NavigationError("depth must be >= 1")
        pat, params = _sel(selector, "s")
        params["max_nodes"] = int(max_nodes)
        rel_filter = "|".join(safe_ident(e, kind="edge type") for e in edge_types) if edge_types else None
        params["rel_filter"] = rel_filter
        if not sc.is_unfiltered:
            return await self._scoped_subgraph(pat, params, sc, d, edge_types)
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

    async def _scoped_subgraph(
        self,
        pat: str,
        params: dict[str, Any],
        sc: Scope,
        depth: int,
        edge_types: Sequence[str] | None,
    ) -> Subgraph:
        """Scope-aware subgraph: plain Cypher, since ``apoc.path.subgraphAll``
        cannot be told to stay inside a tenant."""
        params = {**params, **_scope_params(sc)}
        params["edge_types"] = list(edge_types) if edge_types else None
        clauses = ["($edge_types IS NULL OR all(x IN relationships(p) WHERE type(x) IN $edge_types))"]
        for alias in ("s", "o"):
            allowed = _allowed(alias, sc)
            if allowed:
                clauses.append(allowed)
        tenant_mid = sc.tenant_clause("m")
        if tenant_mid:
            clauses.append(f"all(m IN nodes(p)[1..-1] WHERE {tenant_mid})")
        rec = await self._read_one(
            f"MATCH {pat} "
            f"MATCH p = (s)-[*0..{depth}]-(o) WHERE {' AND '.join(clauses)} "
            "WITH collect(DISTINCT o)[0..$max_nodes] AS ns "
            "RETURN [n IN ns | n { .*, _labels: labels(n) }] AS nodes, "
            "reduce(acc = [], n1 IN ns | acc + "
            "[(n1)-[e]->(n2) WHERE n2 IN ns AND ($edge_types IS NULL OR type(e) IN $edge_types) "
            "| {type: type(e), props: properties(e)}]) AS rels",
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
