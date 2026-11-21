"""navigation/neo4j/_power.py — Group H: generic power tools + execute_raw."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scinr.newton.exceptions import GraphConnectionError, NavigationError, QueryTimeoutError
from scinr.newton.graph.identity import PRIMARY_LABELS
from scinr.newton.navigation import _browse as browse
from scinr.newton.navigation._labels import resolve_label
from scinr.newton.navigation._search_fields import SEARCH_FIELDS, SPECIAL_SEARCH_LABELS
from scinr.newton.navigation.models import (
    GraphEdge,
    GraphNode,
    NodeExpansion,
    NodeMatch,
    NodeSelector,
    PathResult,
    RelationshipCount,
    Subgraph,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._safe import assert_read_only, check_raw_limits, safe_ident
from scinr.newton.navigation.scope import Scope, make_scope
from scinr.newton.utils.redaction import redact_secrets

#: Global catalogue labels: they carry no tenant, and are the bridge through
#: which a free graph walk could hop from one tenant's data to another's
#: (``ModelDecision(A) -[:MATCHED_MODEL]-> CatalogModel <-[:MATCHED_MODEL]- ModelDecision(B)``).
_CATALOG_LABELS = ["CatalogModel", "ModelField", "EntityLabel", "Theme"]

#: ``code`` prefix of a transaction aborted by its timeout.
_TIMEOUT_CODE_PREFIX = "Neo.ClientError.Transaction.TransactionTimedOut"

_IS_CATALOG = "any(l IN labels({a}) WHERE l IN $scope_catalog_labels)"

#: ``CASE`` giving the primary label of ``m`` (the order of ``graph.identity.node_identity``);
#: ``null`` for a node of an unknown kind.
_PRIMARY_OF_M = (
    "CASE "
    + " ".join(f"WHEN '{label}' IN labels(m) THEN '{label}'" for label in PRIMARY_LABELS)
    + " END"
)

#: Map projection of every property that is part of some label's key.
_IDENT_OF_M = "m { " + ", ".join(f".{f}" for f in browse.ident_fields()) + " }"

_DIRECTIONS = ("out", "in", "both")


def _where(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)} " if clauses else ""


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


def _lucene_phrase(text: str) -> str:
    """*text* as one quoted Lucene phrase: its words in that order, no query syntax inside."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


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

    # -- node browsing (portable identity: label + key) -----------------------

    def _anchor(
        self, ref: tuple[str, dict[str, Any]], sc: Scope
    ) -> tuple[str, dict[str, Any], list[str]]:
        """``(pattern, params, scope clauses)`` of the node ``n`` named by *ref*.

        The scope is the one of the node's own kind (a catalogue node has none).
        """
        label, fields = ref
        props = ", ".join(f"`{name}`: $anchor_{i}" for i, name in enumerate(fields))
        params: dict[str, Any] = {f"anchor_{i}": value for i, value in enumerate(fields.values())}
        clauses: list[str] = []
        kind = resolve_label(label).scope_kind
        if kind is not None:
            self._scope_where("n", clauses, params, sc, kind)
        return f"(n:`{label}` {{{props}}})", params, clauses

    async def get_node(
        self,
        label: str,
        key: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> GraphNode | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ref = browse.node_ref(label, key)
        if ref is None:
            return None
        pattern, params, clauses = self._anchor(ref, sc)
        rec = await self._read_one(
            f"MATCH {pattern} {_where(clauses)}RETURN n {{ .*, _labels: labels(n) }} AS n LIMIT 1",
            **params,
        )
        return _map.graph_node(rec["n"]) if rec else None

    async def count_node_relationships(
        self,
        label: str,
        key: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[RelationshipCount]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ref = browse.node_ref(label, key)
        if ref is None:
            return []
        pattern, params, clauses = self._anchor(ref, sc)
        params.update(_scope_params(sc))
        allowed = _allowed("m", sc)
        neighbour_clauses = [allowed] if allowed else []
        rows: list[tuple[str, str, str, int]] = []
        for direction, hop in (("out", "(n)-[r]->(m)"), ("in", "(n)<-[r]-(m)")):
            found = await self._read(
                f"MATCH {pattern} {_where(clauses)}"
                f"MATCH {hop} {_where(neighbour_clauses)}"
                "RETURN type(r) AS type, labels(m) AS labels, count(r) AS count",
                **params,
            )
            for rec in found:
                neighbour = browse.primary_label(rec["labels"])
                if neighbour is not None:
                    rows.append((rec["type"], direction, neighbour, int(rec["count"])))
        return browse.count_relationships(rows)

    async def expand_node(
        self,
        label: str,
        key: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        edge_types: Sequence[str] | None = None,
        neighbor_labels: Sequence[str] | None = None,
        skip: int = 0,
        limit: int | None = 100,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> NodeExpansion:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if direction not in _DIRECTIONS:
            raise NavigationError(f"direction must be one of {_DIRECTIONS}, got {direction!r}.")
        browse.check_page(skip, limit, allow_unlimited=True)
        types = browse.check_edge_types(edge_types)
        wanted = browse.check_labels(neighbor_labels)
        ref = browse.node_ref(label, key)
        if ref is None:
            return NodeExpansion()
        pattern, params, clauses = self._anchor(ref, sc)
        params.update(_scope_params(sc))
        params["edge_types"] = types
        params["neighbor_labels"] = wanted
        hop = {"out": "(n)-[r]->(m)", "in": "(n)<-[r]-(m)", "both": "(n)-[r]-(m)"}[direction]
        neighbour_clauses = [
            "m <> n",
            "($edge_types IS NULL OR type(r) IN $edge_types)",
            "($neighbor_labels IS NULL OR any(l IN labels(m) WHERE l IN $neighbor_labels))",
        ]
        allowed = _allowed("m", sc)
        if allowed:
            neighbour_clauses.append(allowed)
        # The neighbours are ordered in Python (by primary label and encoded key: a composite
        # key has no Cypher expression), so the first read brings only their identity.
        found = await self._read(
            f"MATCH {pattern} {_where(clauses)}"
            f"MATCH {hop} {_where(neighbour_clauses)}"
            "WITH DISTINCT n, m "
            f"WITH n, m, {_PRIMARY_OF_M} AS label WHERE label IS NOT NULL "
            f"RETURN elementId(n) AS anchor, elementId(m) AS eid, label, {_IDENT_OF_M} AS ident",
            **params,
        )
        neighbours: dict[tuple[str, str], str] = {}
        anchor_id = None
        for rec in found:
            anchor_id = rec["anchor"]
            neighbour_key = browse.ident_key(rec["label"], rec["ident"])
            if neighbour_key is not None:
                neighbours[(rec["label"], neighbour_key)] = rec["eid"]
        ordered = sorted(neighbours)
        stop = None if limit is None else skip + limit
        page = ordered[skip:stop]
        if not page:
            return NodeExpansion(total=len(ordered))
        by_eid = {neighbours[pair]: pair for pair in page}
        node_rows = await self._read(
            "MATCH (m) WHERE elementId(m) IN $eids "
            "RETURN elementId(m) AS eid, m { .*, _labels: labels(m) } AS m",
            eids=list(by_eid),
        )
        by_pair = {by_eid[r["eid"]]: _map.graph_node(r["m"]) for r in node_rows if r["eid"] in by_eid}
        edge_rows = await self._read(
            "MATCH (n) WHERE elementId(n) = $anchor "
            "MATCH (m) WHERE elementId(m) IN $eids "
            "MATCH (n)-[r]-(m) WHERE ($edge_types IS NULL OR type(r) IN $edge_types) "
            "RETURN elementId(m) AS eid, type(r) AS type, properties(r) AS props, "
            "startNode(r) = n AS outgoing",
            anchor=anchor_id,
            eids=list(by_eid),
            edge_types=types,
        )
        anchor_pair = (ref[0], browse.encode_ref(ref))
        edges = []
        for rec in edge_rows:
            if rec["eid"] not in by_eid:
                continue
            near = by_eid[rec["eid"]]
            source, target = (anchor_pair, near) if rec["outgoing"] else (near, anchor_pair)
            edges.append(
                GraphEdge(
                    type=rec["type"],
                    source_label=source[0],
                    source_key=source[1],
                    target_label=target[0],
                    target_key=target[1],
                    properties=dict(rec["props"] or {}),
                )
            )
        return NodeExpansion(
            nodes=[by_pair[pair] for pair in page if pair in by_pair],
            edges=browse.sort_edges(edges),
            total=len(ordered),
        )

    def _fulltext_text(self, text: str) -> str:
        """The text of a full-text index query made on behalf of ``search_nodes`` (a phrase)."""
        return _lucene_phrase(text)

    async def search_nodes(
        self,
        text: str,
        *,
        labels: Sequence[str] | None = None,
        limit: int = 20,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[NodeMatch]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        browse.check_page(skip, limit)
        wanted = browse.search_labels(labels)
        if not isinstance(text, str) or len(text) < browse.MIN_SEARCH_LENGTH:
            return []
        scope = {
            "tenant_id": tenant_id,
            "include_public": include_public,
            "created_by_user_id": created_by_user_id,
            "job_id": job_id,
        }
        hits: list[browse.Hit] = []
        element_ids: dict[tuple[str, str], str] = {}
        for label in wanted:
            if label in SPECIAL_SEARCH_LABELS:
                hits.extend(await browse.special_hits(self, label, text, skip + limit, scope))
            else:
                hits.extend(await self._search_label(label, text, sc, element_ids))
        hits.sort(key=lambda hit: hit.order)
        page = hits[skip : skip + limit]
        missing = [element_ids[(hit.label, hit.key)] for hit in page if hit.node is None]
        fetched: dict[tuple[str, str], GraphNode] = {}
        if missing:
            by_eid = {element_ids[(hit.label, hit.key)]: (hit.label, hit.key) for hit in page if hit.node is None}
            for rec in await self._read(
                "MATCH (m) WHERE elementId(m) IN $eids "
                "RETURN elementId(m) AS eid, m { .*, _labels: labels(m) } AS m",
                eids=missing,
            ):
                if rec["eid"] in by_eid:
                    fetched[by_eid[rec["eid"]]] = _map.graph_node(rec["m"])
        out = []
        for hit in page:
            node = hit.node or fetched.get((hit.label, hit.key))
            if node is not None:
                out.append(hit.match(node))
        return out

    async def _search_label(
        self,
        label: str,
        text: str,
        sc: Scope,
        element_ids: dict[tuple[str, str], str],
    ) -> list[browse.Hit]:
        """Candidates of one label (``SEARCH_FIELDS`` and the exact key); records their ids."""
        params: dict[str, Any] = {"needle": text.lower()}
        clauses: list[str] = []
        kind = resolve_label(label).scope_kind
        if kind is not None:
            self._scope_where("n", clauses, params, sc, kind)
        cases = []
        ref = browse.node_ref(label, text)
        if ref is not None:
            tests = []
            for i, (name, value) in enumerate(ref[1].items()):
                tests.append(f"n.`{name}` = $key_{i}")
                params[f"key_{i}"] = value
            cases.append(f"WHEN {' AND '.join(tests)} THEN 'key'")
        for name in SEARCH_FIELDS.get(label, ()):
            cases.append(f"WHEN toLower(n.`{name}`) CONTAINS $needle THEN '{name}'")
        if not cases:
            return []
        identity = "n { " + ", ".join(f".{f}" for f in browse.ident_fields()) + " }"
        rows = await self._read(
            f"MATCH (n:`{label}`) {_where(clauses)}"
            f"WITH n, CASE {' '.join(cases)} END AS field WHERE field IS NOT NULL "
            f"RETURN elementId(n) AS eid, field, {identity} AS ident",
            **params,
        )
        hits = []
        for rec in rows:
            node_key = browse.ident_key(label, rec["ident"])
            if node_key is not None:
                element_ids[(label, node_key)] = rec["eid"]
                hits.append(browse.Hit(label, node_key, rec["field"]))
        return hits

    # -- raw escape hatch --------------------------------------------------

    async def execute_raw(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        dialect: str | None = None,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        if dialect is not None and dialect != self.dialect:
            raise NavigationError(
                f"execute_raw called with dialect={dialect!r} on a {self.dialect!r} backend"
            )
        assert_read_only(query)
        check_raw_limits(timeout_seconds, max_rows)
        from neo4j import unit_of_work
        from neo4j.exceptions import Neo4jError, ServiceUnavailable

        from scinr.newton.graph.neo4j.retry import with_neo4j_retry

        driver = self._live_driver()
        values = dict(params or {})

        async def _tx(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(query, values)
            rows: list[dict[str, Any]] = []
            async for record in result:
                rows.append(record.data())
                if max_rows is not None and len(rows) > max_rows:
                    break
            await result.consume()
            return rows

        work = unit_of_work(timeout=timeout_seconds)(_tx) if timeout_seconds is not None else _tx

        async def _run() -> list[dict[str, Any]]:
            async with driver.session(database=self._database()) as session:
                return await session.execute_read(work)

        try:
            return await with_neo4j_retry(_run)
        except Neo4jError as exc:
            code = getattr(exc, "code", None)
            message = redact_secrets(str(getattr(exc, "message", None) or exc))
            if isinstance(code, str) and code.startswith(_TIMEOUT_CODE_PREFIX):
                raise QueryTimeoutError(message, code=code) from exc
            if isinstance(code, str) and code.startswith("Neo.ClientError."):
                raise NavigationError(message, code=code) from exc
            raise
        except ServiceUnavailable as exc:
            raise GraphConnectionError(redact_secrets(str(exc))) from exc
