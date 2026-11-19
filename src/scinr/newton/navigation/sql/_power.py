"""navigation/sql/_power.py — Group H: generic power tools + execute_raw (SQL)."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement

from scinr.newton.exceptions import NavigationError
from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.navigation import _map
from scinr.newton.navigation._safe import safe_ident
from scinr.newton.navigation.models import GraphNode, NodeSelector, PathResult, Subgraph
from scinr.newton.navigation.scope import Scope, make_scope
from scinr.newton.navigation.sql import _bfs as bfs
from scinr.newton.navigation.sql._bfs import Node
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import prop_equals
from scinr.newton.navigation.sql._walk import LABEL_SEP, PATH_SEP, walk
from scinr.newton.sql import tables as T
from scinr.newton.sql.dialect import SQLITE_DEFERRED_OPTION
from scinr.newton.sql.engine import get_async_engine
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.names import KEY_COLUMN, LABEL_TABLE

#: Global catalogue labels: they carry no tenant, and are the bridge through which a free
#: graph walk could hop from one tenant's data to another's
#: (``ModelDecision(A) -[:MATCHED_MODEL]-> CatalogModel <-[:MATCHED_MODEL]- ModelDecision(B)``).
_CATALOG_LABELS = ("CatalogModel", "ModelField", "EntityLabel", "Theme")

_READ_ONLY_MARKERS = ("read-only", "read only", "readonly")


def _by_label(nodes: Iterable[Node]) -> dict[str, list[str]]:
    """The keys of ``(label, key)`` nodes, by label."""
    out: dict[str, list[str]] = {}
    for label, key in nodes:
        out.setdefault(label, []).append(key)
    return out


def _roles_of(labels: Sequence[str]) -> list[str]:
    """The ``role`` values whose role label is in *labels*."""
    wanted = set(labels)
    return sorted(role for role, label in ROLE_TO_LABEL.items() if label in wanted)


class _PowerMixin(_SqlRuntime):
    # Scope notes (navigation/scope.py). These are free graph walks, so the anchor is checked
    # and — when a tenant is given — every intermediate node of a path must belong to it
    # (catalogue nodes are only allowed as the far end, never as a bridge). With
    # ``tenant_id=None`` ("all") nothing restricts the walk. ``execute_raw`` cannot be
    # scoped: it is an administrative tool outside the isolation contract.

    def _node_index(
        self, name: str = "node_index", labels: Collection[str] | None = None
    ) -> sa.Subquery:
        """``(label, key, role, tenant_id, user, job, users, jobs)`` of every node of the graph.

        The provenance columns that a table does not have are ``NULL``. Catalogue tables have
        no tenant either. *labels* restricts the index to the nodes with those primary labels
        (at least one of them must be a known label).
        """
        text_null = sa.cast(sa.null(), sa.Text)
        list_null = sa.cast(sa.null(), self._dl.list_type)
        parts = []
        for label, table_name in LABEL_TABLE.items():
            if labels is not None and label not in labels:
                continue
            table = T.metadata.tables[table_name]
            columns = table.c
            parts.append(
                sa.select(
                    sa.literal(label, sa.Text).label("label"),
                    columns[KEY_COLUMN[label]].label("key"),
                    (columns.role if "role" in columns else text_null).label("role"),
                    (columns.tenant_id if "tenant_id" in columns else text_null).label("tenant_id"),
                    (
                        columns.created_by_user_id if "created_by_user_id" in columns else text_null
                    ).label("user"),
                    (columns.job_id if "job_id" in columns else text_null).label("job"),
                    (
                        columns.created_by_user_ids
                        if "created_by_user_ids" in columns
                        else list_null
                    ).label("users"),
                    (columns.job_ids if "job_ids" in columns else list_null).label("jobs"),
                )
            )
        return sa.union_all(*parts).subquery(name)

    def _allowed(self, index: Any, sc: Scope) -> ColumnElement[bool] | None:
        """Predicate "the node may be seen": a catalogue node, or a node inside the scope."""
        parts: list[ColumnElement[bool]] = []
        if sc.tenants is not None:
            parts.append(index.c.tenant_id.in_(list(sc.tenants)))
        if sc.user_ids is not None:
            users = list(sc.user_ids)
            parts.append(
                sa.or_(index.c.user.in_(users), self._dl.list_overlaps(index.c.users, users))
            )
        if sc.job_ids is not None:
            jobs = list(sc.job_ids)
            parts.append(sa.or_(index.c.job.in_(jobs), self._dl.list_overlaps(index.c.jobs, jobs)))
        if not parts:
            return None
        return sa.or_(index.c.label.in_(_CATALOG_LABELS), sa.and_(*parts))

    def _selector_start(self, selector: NodeSelector) -> sa.Select:
        """``(label, key)`` of the nodes a selector names (nothing for an unknown type)."""
        label = safe_ident(selector.type, kind="node type")
        key = safe_ident(selector.key, kind="selector key")
        table_name = LABEL_TABLE.get(label)
        if table_name is None:
            return sa.select(
                sa.literal(label, sa.Text).label("label"), sa.literal("", sa.Text).label("key")
            ).where(sa.false())
        table = T.metadata.tables[table_name]
        return self._node_start(
            label,
            table.c[KEY_COLUMN[label]],
            [prop_equals(table, key, selector.value, self._dl)],
        )

    def _tenant_guard(self, sc: Scope, index: Any) -> Any:
        """Hop guard: only the start node and nodes inside the tenant may be walked through."""
        if sc.tenants is None:
            return None
        tenants = list(sc.tenants)

        def guard(cte: Any) -> ColumnElement[bool]:
            inside = sa.exists().where(
                index.c.label == cte.c.label,
                index.c.key == cte.c.key,
                index.c.tenant_id.in_(tenants),
            )
            return sa.or_(cte.c.depth == 0, inside)

        return guard

    async def _fetch_nodes(self, pairs: Sequence[tuple[str, str]]) -> dict[tuple[str, str], Any]:
        """The node dicts (with ``_labels``) of ``(label, key)`` pairs, by pair."""
        by_label: dict[str, list[str]] = {}
        for label, key in pairs:
            by_label.setdefault(label, []).append(key)
        found: dict[tuple[str, str], Any] = {}
        for label, keys in by_label.items():
            table_name = LABEL_TABLE.get(label)
            if table_name is None:
                continue
            table = T.metadata.tables[table_name]
            key_column = table.c[KEY_COLUMN[label]]
            for row in await self._read(sa.select(table).where(key_column.in_(keys))):
                found[(label, row[KEY_COLUMN[label]])] = self._node(table_name, row)
        return found

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
        """The distinct nodes within 1..depth hops of the selected node(s)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        start = self._selector_start(selector)
        index = self._node_index()
        links = T.all_edges
        cte = walk(
            start,
            max_hops=d,
            direction=direction,
            no_backtrack=True,
            types=list(edge_types) if edge_types else None,
            guard=self._tenant_guard(sc, index),
            edge_table=links,
            name="neighbors",
            dialect=self._dl,
        )
        conditions: list[ColumnElement[bool]] = [cte.c.depth >= 1]
        if target_types:
            wanted = list(target_types)
            roles = _roles_of(wanted)
            matches = [index.c.label.in_(wanted)]
            if roles:
                matches.append(sa.and_(index.c.label == "StructureNode", index.c.role.in_(roles)))
            conditions.append(sa.or_(*matches))
        target_scope = self._allowed(index, sc)
        if target_scope is not None:
            conditions.append(target_scope)
            conditions.append(self._anchor_allowed(start, sc))
        hits = (
            sa.select(cte.c.label.label("label"), cte.c.key.label("key"))
            .select_from(
                cte.join(index, sa.and_(index.c.label == cte.c.label, index.c.key == cte.c.key))
            )
            .where(*conditions)
            .distinct()
            .order_by("label", "key")
        )
        pairs = [(r["label"], r["key"]) for r in await self._read(self._limit(hits, limit))]
        nodes = await self._fetch_nodes(pairs)
        return [_map.graph_node(nodes[p]) for p in pairs if p in nodes]

    def _anchor_allowed(self, start: sa.Select, sc: Scope) -> ColumnElement[bool]:
        """The start node(s) of a walk must be inside the scope (``_allowed("s")``)."""
        index = self._node_index("anchor_index")
        anchors = start.subquery("anchor_pairs")
        allowed = self._allowed(index, sc)
        assert allowed is not None
        inside = (
            sa.select(sa.literal(1))
            .select_from(
                anchors.join(
                    index,
                    sa.and_(index.c.label == anchors.c.label, index.c.key == anchors.c.key),
                )
            )
            .where(allowed)
        )
        return sa.exists(inside)

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
        """The shortest undirected path between two selected nodes, up to *max_hops* hops.

        Cypher: ``MATCH p = shortestPath((a)-[*..h]-(b))``, every relationship of an allowed
        type; when scoped, both ends allowed and every inner node inside the tenant.

        The search is breadth-first from both ends, one query per level (``_bfs.py``), instead
        of a recursive CTE of every walk (exponential in *max_hops*). It returns what that CTE
        returned: the smallest ``(length, path string)`` over the start/goal pairs whose keys
        differ, no inner node outside the tenant (the start is exempt), the goal allowed by
        the scope and, when scoped, at least one start allowed. Ties between shortest paths
        go to the smallest path string by code point (``Dialect.order_text``) on both engines.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        h = int(max_hops)
        if h < 1:
            raise NavigationError("max_hops must be >= 1")
        start = self._selector_start(from_selector)
        goal_select = self._selector_start(to_selector)
        starts = {(r["label"], r["key"]) for r in await self._read(start)}
        goals = {(r["label"], r["key"]) for r in await self._read(goal_select)}
        if len(starts) == 1 and starts == goals:
            # Neo4j refuses a shortest path between a node and itself.
            raise NavigationError("shortest_path: from_selector and to_selector name the same node")
        if not starts or not goals:
            return None
        if not sc.is_unfiltered:  # ``_allowed`` has a predicate
            if not await self._any_allowed(start, {label for label, _ in starts}, sc):
                return None
            goals = await self._allowed_pairs(goal_select, {label for label, _ in goals}, sc)
        found = await self._search_shortest(starts, goals, h, edge_types, sc)
        if found is None:
            return None
        length, path = found
        tokens = [tuple(t.split(LABEL_SEP, 1)) for t in path.split(PATH_SEP)]
        pairs = [(label, key) for label, key in tokens]
        nodes = await self._fetch_nodes(pairs)
        relationships = await self._path_relationships(pairs, edge_types)
        return PathResult(
            raw={"length": length},
            length=length,
            nodes=[_map.graph_node(nodes[p]) for p in pairs if p in nodes],
            relationships=relationships,
        )

    async def _any_allowed(self, start: sa.Select, labels: Collection[str], sc: Scope) -> bool:
        """``_anchor_allowed``, evaluated: is any start node inside the scope?"""
        index = self._node_index("anchor_index", labels)
        anchors = start.subquery("anchor_pairs")
        allowed = self._allowed(index, sc)
        assert allowed is not None
        statement = (
            sa.select(sa.literal(1).label("one"))
            .select_from(
                anchors.join(
                    index, sa.and_(index.c.label == anchors.c.label, index.c.key == anchors.c.key)
                )
            )
            .where(allowed)
            .limit(1)
        )
        return await self._read_one(statement) is not None

    async def _allowed_pairs(
        self, select: sa.Select, labels: Collection[str], sc: Scope
    ) -> set[Node]:
        """The ``(label, key)`` of *select* that the scope allows (``_allowed``)."""
        index = self._node_index("end_index", labels)
        ends = select.subquery("end_pairs")
        allowed = self._allowed(index, sc)
        assert allowed is not None
        statement = (
            sa.select(ends.c.label, ends.c.key)
            .select_from(
                ends.join(index, sa.and_(index.c.label == ends.c.label, index.c.key == ends.c.key))
            )
            .where(allowed)
        )
        return {(r["label"], r["key"]) for r in await self._read(statement)}

    async def _search_shortest(
        self,
        starts: set[Node],
        goals: set[Node],
        max_hops: int,
        edge_types: Sequence[str] | None,
        sc: Scope,
    ) -> tuple[int, str] | None:
        """The smallest ``(length, path string)`` over the start/goal pairs whose keys differ.

        A start may not end at a goal with its own key (the CTE's ``key != origin``). The
        pairs are split into searches whose starts and goals never share a key: the starts
        whose key is no goal's, against every goal; then each shared key's starts against the
        goals with another key.
        """
        shared = sorted({key for _, key in starts} & {key for _, key in goals})
        searches: list[tuple[set[Node], set[Node]]] = []
        plain = {n for n in starts if n[1] not in shared}
        if plain:
            searches.append((plain, goals))
        for key in shared:
            searches.append(({n for n in starts if n[1] == key}, {n for n in goals if n[1] != key}))
        types = sorted(edge_types) if edge_types else None
        neighbours = self._neighbours_query(types)
        passable = self._passable_query(sorted(sc.tenants)) if sc.tenants is not None else None
        best: tuple[int, str] | None = None
        for sources, targets in searches:
            found = await bfs.shortest_path(
                sources,
                targets,
                max_hops=max_hops if best is None else best[0],
                neighbours=neighbours,
                passable=passable,
            )
            if found is not None and (best is None or found < best):
                best = found
        return best

    def _neighbours_query(self, types: list[str] | None) -> bfs.Neighbours:
        """One level of the search: the relationships (of *types*) with an end in a batch."""
        links = T.all_edges

        def hop(near: tuple[Any, Any], far: tuple[Any, Any], batch: Sequence[Node]) -> sa.Select:
            match = sa.or_(
                *[
                    sa.and_(near[0] == label, near[1].in_(keys))
                    for label, keys in _by_label(batch).items()
                ]
            )
            conditions: list[ColumnElement[bool]] = [match]
            if types is not None:
                conditions.append(links.c.type.in_(types))
            return sa.select(
                near[0].label("near_label"),
                near[1].label("near_key"),
                far[0].label("far_label"),
                far[1].label("far_key"),
            ).where(*conditions)

        source = (links.c.src_label, links.c.src_key)
        destination = (links.c.dst_label, links.c.dst_key)

        async def neighbours(batch: Sequence[Node]) -> list[tuple[Node, Node]]:
            statement = sa.union(hop(source, destination, batch), hop(destination, source, batch))
            return [
                ((r["near_label"], r["near_key"]), (r["far_label"], r["far_key"]))
                for r in await self._read(statement)
            ]

        return neighbours

    def _passable_query(self, tenants: list[str]) -> bfs.Passable:
        """The tenant guard of ``_tenant_guard``: the nodes of a batch inside *tenants*."""

        async def passable(batch: Sequence[Node]) -> set[Node]:
            parts = []
            for label, keys in _by_label(batch).items():
                table_name = LABEL_TABLE.get(label)
                if table_name is None:
                    continue
                table = T.metadata.tables[table_name]
                if "tenant_id" not in table.c:
                    continue  # catalogue nodes have no tenant
                key = table.c[KEY_COLUMN[label]]
                parts.append(
                    sa.select(sa.literal(label, sa.Text).label("label"), key.label("key")).where(
                        key.in_(keys), table.c.tenant_id.in_(tenants)
                    )
                )
            if not parts:
                return set()
            statement = parts[0] if len(parts) == 1 else sa.union_all(*parts)
            return {(r["label"], r["key"]) for r in await self._read(statement)}

        return passable

    async def _path_relationships(
        self, pairs: list[tuple[str, str]], edge_types: Sequence[str] | None
    ) -> list[dict[str, Any]]:
        """One relationship (``{type, props}``) between each pair of consecutive path nodes."""
        links = T.all_edges
        out: list[dict[str, Any]] = []
        for (a_label, a_key), (b_label, b_key) in zip(pairs, pairs[1:], strict=False):
            forward = sa.and_(
                links.c.src_label == a_label,
                links.c.src_key == a_key,
                links.c.dst_label == b_label,
                links.c.dst_key == b_key,
            )
            backward = sa.and_(
                links.c.src_label == b_label,
                links.c.src_key == b_key,
                links.c.dst_label == a_label,
                links.c.dst_key == a_key,
            )
            conditions = [sa.or_(forward, backward)]
            if edge_types:
                conditions.append(links.c.type.in_(list(edge_types)))
            row = await self._read_one(
                sa.select(links.c.type, links.c.props)
                .where(*conditions)
                .order_by(self._dl.order_text(links.c.type))
                .limit(1)
            )
            if row is not None:
                out.append({"type": row["type"], "props": row["props"] or {}})
        return out

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
        """The nodes within *depth* hops of the selected node and the edges among them."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = int(depth)
        if d < 1:
            raise NavigationError("depth must be >= 1")
        for edge in edge_types or ():
            safe_ident(edge, kind="edge type")
        start = self._selector_start(selector)
        index = self._node_index()
        links = T.all_edges
        cte = walk(
            start,
            max_hops=d,
            direction="both",
            no_backtrack=True,
            types=list(edge_types) if edge_types else None,
            guard=self._tenant_guard(sc, index),
            edge_table=links,
            name="subgraph",
            dialect=self._dl,
        )
        conditions: list[ColumnElement[bool]] = []
        end_allowed = self._allowed(index, sc)
        if end_allowed is not None:
            conditions.extend([end_allowed, self._anchor_allowed(start, sc)])
        hits = (
            sa.select(cte.c.label.label("label"), cte.c.key.label("key"))
            .select_from(
                cte.join(index, sa.and_(index.c.label == cte.c.label, index.c.key == cte.c.key))
            )
            .where(*conditions)
            .group_by(cte.c.label, cte.c.key)
            .order_by(sa.func.min(cte.c.depth), cte.c.label, cte.c.key)
            .limit(int(max_nodes))
        )
        pairs = [(r["label"], r["key"]) for r in await self._read(hits)]
        if not pairs and sc.is_unfiltered:
            if not await self._read(start.limit(1)):
                return Subgraph(raw={})
        nodes = await self._fetch_nodes(pairs)
        graph_nodes = [nodes[p] for p in pairs if p in nodes]
        return Subgraph(
            raw={"n_nodes": len(graph_nodes)},
            nodes=[_map.graph_node(n) for n in graph_nodes],
            edges=await self._edges_among(pairs, edge_types),
        )

    async def _edges_among(
        self, pairs: list[tuple[str, str]], edge_types: Sequence[str] | None
    ) -> list[dict[str, Any]]:
        """``{type, props}`` of the relationships whose two ends are in *pairs*."""
        if not pairs:
            return []
        links = T.all_edges
        wanted = set(pairs)
        out: list[dict[str, Any]] = []
        by_label: dict[str, list[str]] = {}
        for label, key in pairs:
            by_label.setdefault(label, []).append(key)
        source_match = sa.or_(
            *[
                sa.and_(links.c.src_label == label, links.c.src_key.in_(keys))
                for label, keys in by_label.items()
            ]
        )
        conditions: list[ColumnElement[bool]] = [source_match]
        if edge_types:
            conditions.append(links.c.type.in_(list(edge_types)))
        rows = await self._read(
            sa.select(links)
            .where(*conditions)
            .order_by(self._dl.order_text(links.c.type), links.c.src_key, links.c.dst_key)
        )
        for row in rows:
            if (row["dst_label"], row["dst_key"]) in wanted:
                out.append({"type": row["type"], "props": row["props"] or {}})
        return out

    # -- raw escape hatch --------------------------------------------------

    async def execute_raw(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        dialect: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read-only SQL query (``:name`` parameters); the engine enforces read-only.

        Postgres runs it in a ``READ ONLY`` transaction with the configured schema first in
        the ``search_path`` (so table names need no schema); SQLite with ``PRAGMA
        query_only``. A statement that writes fails and is reported as a
        :class:`NavigationError`.
        """
        if dialect is not None and dialect != self.dialect:
            raise NavigationError(
                f"execute_raw called with dialect={dialect!r} on a {self.dialect!r} backend"
            )
        if not isinstance(query, str) or not query.strip():
            raise NavigationError("execute_raw: query must be a non-empty string.")
        from scinr.newton.config import get_config

        schema = get_config().sql_schema
        dl = self._dl
        statement = sa.text(query)
        values = dict(params or {})

        def run(connection: Any) -> list[dict[str, Any]]:
            with dl.read_only(connection):
                dl.use_schema(connection, schema)
                result = connection.execute(statement, values)
                if not result.returns_rows:
                    return []
                return [dict(row._mapping) for row in result]

        engine = get_async_engine().execution_options(**{SQLITE_DEFERRED_OPTION: True})
        try:
            async with async_sql_errors(), engine.connect() as connection:
                return await connection.run_sync(run)
        except Exception as exc:
            if any(marker in str(exc).lower() for marker in _READ_ONLY_MARKERS):
                raise NavigationError(
                    "execute_raw is read-only; the statement was refused."
                ) from exc
            raise


__all__ = ["_PowerMixin"]
