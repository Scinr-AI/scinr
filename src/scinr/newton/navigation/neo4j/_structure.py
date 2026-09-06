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
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime, selector_path
from scinr.newton.navigation.neo4j._translate import translate_where

_NODE_PROJ = "n { .*, _labels: labels(n) }"


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
    ) -> list[StructureNodeRef]:
        path = selector_path(document)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
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
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH {self._doc_match('doc', version=version)} "
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
    ) -> int:
        path = selector_path(document)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        where_sql = ""
        if roles:
            where_sql = "WHERE n.role IN $roles "
            params["roles"] = list(roles)
        rec = await self._read_one(
            f"MATCH {self._doc_match('doc', version=version)} "
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode) {where_sql}"
            "RETURN count(DISTINCT n) AS c",
            **params,
        )
        return int(rec["c"]) if rec else 0

    async def get_root_structure_nodes(
        self, document: str | DocumentRef, *, version: int | None = None
    ) -> list[StructureNodeRef]:
        path = selector_path(document)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        cy = (
            f"MATCH {self._doc_match('doc', version=version)}-[:HAS_STRUCTURE]->(n:StructureNode) "
            f"RETURN {_NODE_PROJ} AS n, doc.path AS dp, doc.version AS dv "
            "ORDER BY n.appearance_order, n.id"
        )
        rows = await self._read(cy, **params)
        return [
            _map.structure_node_ref(r["n"], document_path=r["dp"], document_version=r["dv"])
            for r in rows
        ]

    async def get_structure_node(self, node_id: str) -> StructureNodeRef | None:
        rec = await self._read_one(
            f"MATCH (n:StructureNode {{id: $node_id}}) RETURN {_NODE_PROJ} AS n", node_id=node_id
        )
        return _map.structure_node_ref(rec["n"]) if rec else None

    async def get_child_nodes(
        self,
        node_id: str,
        *,
        depth: int | None = 1,
        roles: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[StructureNodeRef]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"node_id": node_id}
        where_sql = ""
        if roles:
            where_sql = "WHERE c.role IN $roles "
            params["roles"] = list(roles)
        cy = (
            f"MATCH (:StructureNode {{id: $node_id}})-[:HAS_CHILD*1..{d}]->(c:StructureNode) {where_sql}"
            f"RETURN DISTINCT c {{ .*, _labels: labels(c) }} AS n "
            f"ORDER BY n.appearance_order, n.id{self._limit_clause(limit)}"
        )
        rows = await self._read(cy, **params)
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_structure_subtree(
        self, node_id: str, *, depth: int | None = None, include_info_units: bool = False
    ) -> StructureTree | None:
        d = self._resolve_depth(depth)
        iu = (
            "OPTIONAL MATCH (c)-[:HAS_INFO_UNIT]->(u:InfoUnit) "
            if include_info_units
            else ""
        )
        ret_units = "collect(DISTINCT u) AS units" if include_info_units else "[] AS units"
        cy = (
            "MATCH (root:StructureNode {id: $node_id}) "
            f"OPTIONAL MATCH p = (root)-[:HAS_CHILD*1..{d}]->(c:StructureNode) "
            f"{iu}"
            f"RETURN root {{ .*, _labels: labels(root) }} AS root, "
            f"c {{ .*, _labels: labels(c) }} AS c, [x IN nodes(p) | x.id] AS lineage, {ret_units}"
        )
        rows = await self._read(cy, node_id=node_id)
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

    async def get_parent_node(self, node_id: str) -> StructureNodeRef | None:
        rec = await self._read_one(
            "MATCH (p:StructureNode)-[:HAS_CHILD]->(:StructureNode {id: $node_id}) "
            "RETURN p { .*, _labels: labels(p) } AS n LIMIT 1",
            node_id=node_id,
        )
        return _map.structure_node_ref(rec["n"]) if rec else None

    async def get_node_ancestors(
        self, node_id: str, *, depth: int | None = None
    ) -> list[StructureNodeRef]:
        d = self._resolve_depth(depth)
        rec = await self._read_one(
            "MATCH p = (doc:Document)-[:HAS_STRUCTURE]->(a:StructureNode)"
            f"-[:HAS_CHILD*0..{d}]->(t:StructureNode {{id: $node_id}}) "
            "RETURN [x IN nodes(p) WHERE x:StructureNode][0..-1] AS anc, "
            "doc.path AS dp, doc.version AS dv "
            "ORDER BY length(p) DESC LIMIT 1",
            node_id=node_id,
        )
        if not rec or not rec.get("anc"):
            return []
        return [
            _map.structure_node_ref(
                {**n, "_labels": n.get("_labels", [])}, document_path=rec["dp"], document_version=rec["dv"]
            )
            for n in rec["anc"]
        ]

    async def get_node_path(self, node_id: str) -> NodePath | None:
        rec = await self._read_one(
            "MATCH p = (doc:Document)-[:HAS_STRUCTURE]->(a:StructureNode)"
            "-[:HAS_CHILD*0..]->(t:StructureNode {id: $node_id}) "
            "RETURN doc AS doc, [x IN nodes(p) WHERE x:StructureNode] AS nodes "
            "ORDER BY length(p) DESC LIMIT 1",
            node_id=node_id,
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

    async def get_document_of_node(self, node_id: str) -> DocumentRef | None:
        rec = await self._read_one(
            "MATCH (n:StructureNode {id: $node_id})<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(d:Document) "
            "RETURN d LIMIT 1",
            node_id=node_id,
        )
        return _map.document_ref(rec["d"]) if rec else None

    async def get_sibling_nodes(
        self, node_id: str, *, include_self: bool = False
    ) -> list[StructureNodeRef]:
        # Try HAS_CHILD parent first; fall back to HAS_STRUCTURE (root-level node).
        rows = await self._read(
            "MATCH (p:StructureNode)-[:HAS_CHILD]->(:StructureNode {id: $node_id}) "
            "MATCH (p)-[:HAS_CHILD]->(s:StructureNode) "
            "RETURN s { .*, _labels: labels(s) } AS n ORDER BY n.appearance_order, n.id",
            node_id=node_id,
        )
        if not rows:
            rows = await self._read(
                "MATCH (d:Document)-[:HAS_STRUCTURE]->(:StructureNode {id: $node_id}) "
                "MATCH (d)-[:HAS_STRUCTURE]->(s:StructureNode) "
                "RETURN s { .*, _labels: labels(s) } AS n ORDER BY n.appearance_order, n.id",
                node_id=node_id,
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
    ) -> list[StructureNodeRef]:
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
            clauses.append(
                "EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) }"
            )
            params["doc_path"] = selector_path(document)
        wfrag, wparams = translate_where(where, alias="n")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH (n:StructureNode) {where_sql}"
            f"RETURN {_NODE_PROJ} AS n ORDER BY n.id{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy, **params)
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_nodes_by_theme(
        self, theme: str, *, document: str | DocumentRef | None = None, limit: int | None = None
    ) -> list[StructureNodeRef]:
        params: dict[str, Any] = {"theme": theme}
        where_sql = ""
        if document is not None:
            where_sql = (
                "WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) } "
            )
            params["doc_path"] = selector_path(document)
        cy = (
            f"MATCH (n:StructureNode {{theme: $theme}}) {where_sql}"
            f"RETURN {_NODE_PROJ} AS n ORDER BY n.id{self._limit_clause(limit)}"
        )
        rows = await self._read(cy, **params)
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def describe_node(
        self, node_id: str, *, include_source_text: bool = False
    ) -> NodeDescription | None:
        node = await self.get_structure_node(node_id)  # type: ignore[attr-defined]
        if node is None:
            return None
        ancestors = await self.get_node_ancestors(node_id)  # type: ignore[attr-defined]
        info_units = await self.get_info_units(node_id)  # type: ignore[attr-defined]
        decision = await self.get_model_decision(node_id)  # type: ignore[attr-defined]
        extraction = await self.get_extraction_result(node_id)  # type: ignore[attr-defined]
        counts = await self._read_one(
            f"""MATCH (n:StructureNode {{id: $node_id}})
            CALL {{ WITH n MATCH (n)-[:HAS_CHILD]->(c) RETURN count(c) AS child_count }}
            CALL {{ WITH n OPTIONAL MATCH (n)-[:HAS_EXTRACTION]->(er:ExtractionResult)-[hr*1..{self._containment_depth(None)}]->(mi:ModelInstance)
                    WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
                    RETURN count(DISTINCT mi) AS mi_count }}
            RETURN child_count, mi_count""",
            node_id=node_id,
        )
        source_text = None
        if include_source_text:
            try:
                from scinr.newton.navigation.pages import get_node_source_text

                source_text = await get_node_source_text(self, node_id)  # type: ignore[arg-type]
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
