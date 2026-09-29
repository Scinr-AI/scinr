"""navigation/neo4j/_structure.py — Group B: structure nodes & the document tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scinr.newton.navigation.models import (
    DocumentRef,
    NodeDescription,
    NodePath,
    StructureNodeRef,
    StructureTree,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._translate import translate_where
from scinr.newton.navigation.scope import make_scope, resolve_selector

_NODE_PROJ = "n { .*, _labels: labels(n) }"

# Scope notes (navigation/scope.py). A :StructureNode carries scalar
# tenant_id / created_by_user_id / job_id copied from its document, and its
# id embeds the tenant. Document-anchored methods pin the *document* to the
# tenant and filter the returned nodes by user/job; node-anchored methods
# guard the anchor with the full scope (a foreign id does not resolve) and
# filter returned nodes by user/job. Edges never cross tenants, so the anchor
# is enough — no traversal back up to :Document.


class _StructureMixin(_Neo4jRuntime):
    async def get_structure_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        roles: Sequence[str] | None = None,
        title_contains: str | None = None,
        theme: str | None = None,
        where: Mapping[str, Any] | None = None,
        depth: int | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        clauses: list[str] = []
        if roles:
            clauses.append("n.role IN $roles")
            params["roles"] = list(roles)
        if title_contains is not None:
            clauses.append("toLower(n.title) CONTAINS toLower($title_contains)")
            params["title_contains"] = title_contains
        if theme is not None:
            clauses.append("n.theme = $theme")
            params["theme"] = theme
        wfrag, wparams = translate_where(where, alias="n")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        self._scope_where("n", clauses, params, sc, tenant=False)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode) {where_sql}"
            f"RETURN DISTINCT {_NODE_PROJ} AS n, doc.path AS dp, doc.version AS dv "
            f"ORDER BY n.appearance_order, n.id{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy, **params)
        return [
            _map.structure_node_ref(r["n"], document_path=r["dp"], document_version=r["dv"])
            for r in rows
        ]

    async def count_structure_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        roles: Sequence[str] | None = None,
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
        if roles:
            clauses.append("n.role IN $roles")
            params["roles"] = list(roles)
        self._scope_where("n", clauses, params, sc, tenant=False)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rec = await self._read_one(
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode) {where_sql}"
            "RETURN count(DISTINCT n) AS c",
            **params,
        )
        return int(rec["c"]) if rec else 0

    async def get_root_structure_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        clauses: list[str] = []
        self._scope_where("n", clauses, params, sc, tenant=False)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            f"MATCH (doc)-[:HAS_STRUCTURE]->(n:StructureNode) {where_sql}"
            f"RETURN {_NODE_PROJ} AS n, doc.path AS dp, doc.version AS dv "
            "ORDER BY n.appearance_order, n.id"
        )
        rows = await self._read(cy, **params)
        return [
            _map.structure_node_ref(r["n"], document_path=r["dp"], document_version=r["dv"])
            for r in rows
        ]

    async def get_structure_node(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureNodeRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses: list[str] = []
        params: dict[str, Any] = {"node_id": node_id}
        self._scope_where("n", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rec = await self._read_one(
            f"MATCH (n:StructureNode {{id: $node_id}}) {where_sql}RETURN {_NODE_PROJ} AS n",
            **params,
        )
        return _map.structure_node_ref(rec["n"]) if rec else None

    async def get_structure_nodes_by_ids(
        self,
        node_ids: Sequence[str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return []
        clauses: list[str] = ["n.id IN $ids"]
        params: dict[str, Any] = {"ids": ids}
        self._scope_where("n", clauses, params, sc)
        rows = await self._read(
            f"MATCH (n:StructureNode) WHERE {' AND '.join(clauses)} RETURN {_NODE_PROJ} AS n",
            **params,
        )
        by_id = {r["n"]["id"]: _map.structure_node_ref(r["n"]) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    async def get_child_nodes(
        self,
        node_id: str,
        *,
        depth: int | None = 1,
        roles: Sequence[str] | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("a", params, sc)
        clauses: list[str] = []
        if roles:
            clauses.append("c.role IN $roles")
            params["roles"] = list(roles)
        self._scope_where("c", clauses, params, sc, tenant=False)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH (a:StructureNode {{id: $node_id}}) {anchor_where}"
            f"MATCH (a)-[:HAS_CHILD*1..{d}]->(c:StructureNode) {where_sql}"
            f"RETURN DISTINCT c {{ .*, _labels: labels(c) }} AS n "
            f"ORDER BY n.appearance_order, n.id{self._limit_clause(limit)}"
        )
        rows = await self._read(cy, **params)
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_structure_subtree(
        self,
        node_id: str,
        *,
        depth: int | None = None,
        include_info_units: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureTree | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"node_id": node_id}
        # user/job filters apply to the root only: pruning inner nodes would
        # leave their children orphaned.
        root_where = self._doc_where("root", params, sc, "scalar")
        iu = (
            "OPTIONAL MATCH (c)-[:HAS_INFO_UNIT]->(u:InfoUnit) "
            if include_info_units
            else ""
        )
        ret_units = "collect(DISTINCT u) AS units" if include_info_units else "[] AS units"
        cy = (
            f"MATCH (root:StructureNode {{id: $node_id}}) {root_where}"
            f"OPTIONAL MATCH p = (root)-[:HAS_CHILD*1..{d}]->(c:StructureNode) "
            f"{iu}"
            f"RETURN root {{ .*, _labels: labels(root) }} AS root, "
            f"c {{ .*, _labels: labels(c) }} AS c, [x IN nodes(p) | x.id] AS lineage, {ret_units}"
        )
        rows = await self._read(cy, **params)
        if not rows:
            return None
        root = StructureTree(
            **_map.structure_node_ref(rows[0]["root"]).model_dump(), depth=0
        )
        by_id: dict[str, StructureTree] = {root.id: root}
        for r in sorted((r for r in rows if r.get("c")), key=lambda r: len(r["lineage"])):
            lineage = r["lineage"]
            node = StructureTree(
                **_map.structure_node_ref(r["c"]).model_dump(),
                depth=len(lineage) - 1,
                info_units=[_map.info_unit_ref(u) for u in (r["units"] or [])] or None,
            )
            by_id[node.id] = node
            parent = by_id.get(lineage[-2]) if len(lineage) >= 2 else root
            if parent is not None:
                parent.children.append(node)
        return root

    async def get_parent_node(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureNodeRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("a", params, sc)
        clauses: list[str] = []
        self._scope_where("p", clauses, params, sc, tenant=False)
        p_where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rec = await self._read_one(
            f"MATCH (a:StructureNode {{id: $node_id}}) {anchor_where}"
            f"MATCH (p:StructureNode)-[:HAS_CHILD]->(a) {p_where}"
            "RETURN p { .*, _labels: labels(p) } AS n LIMIT 1",
            **params,
        )
        return _map.structure_node_ref(rec["n"]) if rec else None

    async def get_node_ancestors(
        self,
        node_id: str,
        *,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"node_id": node_id}
        # the anchor carries the whole scope; the spine above it is not pruned
        t_where = self._doc_where("t", params, sc, "scalar")
        rec = await self._read_one(
            f"MATCH (t:StructureNode {{id: $node_id}}) {t_where}"
            "MATCH p = (doc:Document)-[:HAS_STRUCTURE]->(a:StructureNode)"
            f"-[:HAS_CHILD*0..{d}]->(t) "
            "RETURN [x IN nodes(p) WHERE x:StructureNode][0..-1] AS anc, "
            "doc.path AS dp, doc.version AS dv "
            "ORDER BY length(p) DESC LIMIT 1",
            **params,
        )
        if not rec or not rec.get("anc"):
            return []
        return [
            _map.structure_node_ref(
                {**n, "_labels": n.get("_labels", [])}, document_path=rec["dp"], document_version=rec["dv"]
            )
            for n in rec["anc"]
        ]

    async def get_node_path(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> NodePath | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"node_id": node_id}
        t_where = self._doc_where("t", params, sc, "scalar")
        rec = await self._read_one(
            f"MATCH (t:StructureNode {{id: $node_id}}) {t_where}"
            "MATCH p = (doc:Document)-[:HAS_STRUCTURE]->(a:StructureNode)"
            "-[:HAS_CHILD*0..]->(t) "
            "RETURN doc AS doc, [x IN nodes(p) WHERE x:StructureNode] AS nodes "
            "ORDER BY length(p) DESC LIMIT 1",
            **params,
        )
        if not rec:
            return None
        return NodePath(
            raw=dict(rec),
            document=_map.document_ref(rec["doc"]) if rec.get("doc") else None,
            nodes=[
                _map.structure_node_ref({**n, "_labels": n.get("_labels", [])})
                for n in (rec["nodes"] or [])
            ],
        )

    async def get_document_of_node(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"node_id": node_id}
        n_where = self._doc_where("n", params, sc, "scalar")
        rec = await self._read_one(
            f"MATCH (n:StructureNode {{id: $node_id}}) {n_where}"
            "MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(d:Document) "
            "RETURN d LIMIT 1",
            **params,
        )
        return _map.document_ref(rec["d"]) if rec else None

    async def get_sibling_nodes(
        self,
        node_id: str,
        *,
        include_self: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("a", params, sc)
        clauses: list[str] = []
        self._scope_where("s", clauses, params, sc, tenant=False)
        s_where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        # Try HAS_CHILD parent first; fall back to HAS_STRUCTURE (root-level node).
        rows = await self._read(
            f"MATCH (a:StructureNode {{id: $node_id}}) {anchor_where}"
            "MATCH (p:StructureNode)-[:HAS_CHILD]->(a) "
            f"MATCH (p)-[:HAS_CHILD]->(s:StructureNode) {s_where}"
            "RETURN s { .*, _labels: labels(s) } AS n ORDER BY n.appearance_order, n.id",
            **params,
        )
        if not rows:
            rows = await self._read(
                f"MATCH (a:StructureNode {{id: $node_id}}) {anchor_where}"
                "MATCH (d:Document)-[:HAS_STRUCTURE]->(a) "
                f"MATCH (d)-[:HAS_STRUCTURE]->(s:StructureNode) {s_where}"
                "RETURN s { .*, _labels: labels(s) } AS n ORDER BY n.appearance_order, n.id",
                **params,
            )
        out = [_map.structure_node_ref(r["n"]) for r in rows]
        if not include_self:
            out = [s for s in out if s.id != node_id]
        return out

    async def find_structure_nodes(
        self,
        *,
        title_contains: str | None = None,
        node_id: str | None = None,
        role: str | None = None,
        theme: str | None = None,
        document: str | DocumentRef | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if title_contains is not None:
            clauses.append("toLower(n.title) CONTAINS toLower($title_contains)")
            params["title_contains"] = title_contains
        if node_id is not None:
            clauses.append("n.node_id = $node_id")
            params["node_id"] = node_id
        if role is not None:
            clauses.append("n.role = $role")
            params["role"] = role
        if theme is not None:
            clauses.append("n.theme = $theme")
            params["theme"] = theme
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                "EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) }"
            )
            params["doc_path"] = path
        wfrag, wparams = translate_where(where, alias="n")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        self._scope_where("n", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH (n:StructureNode) {where_sql}"
            f"RETURN {_NODE_PROJ} AS n ORDER BY n.id{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy, **params)
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_nodes_by_theme(
        self,
        theme: str,
        *,
        document: str | DocumentRef | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"theme": theme}
        clauses: list[str] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                "EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) }"
            )
            params["doc_path"] = path
        self._scope_where("n", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH (n:StructureNode {{theme: $theme}}) {where_sql}"
            f"RETURN {_NODE_PROJ} AS n ORDER BY n.id{self._limit_clause(limit)}"
        )
        rows = await self._read(cy, **params)
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def describe_node(
        self,
        node_id: str,
        *,
        include_source_text: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> NodeDescription | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        scope_kw: dict[str, Any] = {
            "tenant_id": tenant_id,
            "include_public": include_public,
            "created_by_user_id": created_by_user_id,
            "job_id": job_id,
        }
        node = await self.get_structure_node(node_id, **scope_kw)  # type: ignore[attr-defined]
        if node is None:
            return None
        # The anchor passed the full scope; what hangs off it lives in the
        # same tenant, so only the tenant is pinned for the sub-queries.
        sub_kw: dict[str, Any] = {"tenant_id": node.raw.get("tenant_id") or tenant_id, "include_public": False}
        ancestors = await self.get_node_ancestors(node_id, **sub_kw)  # type: ignore[attr-defined]
        info_units = await self.get_info_units(node_id, **sub_kw)  # type: ignore[attr-defined]
        decision = await self.get_model_decision(node_id, **sub_kw)  # type: ignore[attr-defined]
        extraction = await self.get_extraction_result(node_id, **sub_kw)  # type: ignore[attr-defined]
        params: dict[str, Any] = {"node_id": node_id}
        n_where = self._doc_where("n", params, sc, "scalar")
        counts = await self._read_one(
            f"""MATCH (n:StructureNode {{id: $node_id}}) {n_where}
            CALL {{ WITH n MATCH (n)-[:HAS_CHILD]->(c) RETURN count(c) AS child_count }}
            CALL {{ WITH n OPTIONAL MATCH (n)-[:HAS_EXTRACTION]->(er:ExtractionResult)-[hr*1..{self._containment_depth(None)}]->(mi:ModelInstance)
                    WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
                    RETURN count(DISTINCT mi) AS mi_count }}
            RETURN child_count, mi_count""",
            **params,
        )
        source_text = None
        if include_source_text:
            try:
                from scinr.newton.navigation import pages

                # Reuses the already-resolved node: only its pages are read.
                source_text = await pages._node_source_text(node)
            except Exception:  # noqa: BLE001 — source text is best-effort
                source_text = None
        return NodeDescription(
            raw=dict(counts or {}),
            node=node,
            ancestors=ancestors,
            info_units=info_units,
            model_decision=decision,
            extraction=extraction,
            model_instance_count=int((counts or {}).get("mi_count", 0) or 0),
            child_count=int((counts or {}).get("child_count", 0) or 0),
            source_page_ids=list(node.source_page_ids),
            source_text=source_text,
        )
