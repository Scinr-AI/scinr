"""navigation/neo4j/_documents.py — Group A: documents & folder hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import DocumentRef, DocumentStats, DocumentTree
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._translate import translate_where


class _DocumentsMixin(_Neo4jRuntime):
    async def list_root_documents(
        self,
        *,
        latest_only: bool = True,
        only_folders: bool = False,
        only_leaves: bool = False,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[DocumentRef]:
        if only_folders and only_leaves:
            raise NavigationError("only_folders and only_leaves are mutually exclusive")
        where = ["NOT ( ()-[:IS_COMPOSED_OF]->(d) )"]
        if latest_only:
            where.append("d.latest = true")
        if only_folders:
            where.append("d.is_folder = true")
        if only_leaves:
            where.append("d.is_folder = false")
        cy = (
            f"MATCH (d:Document) WHERE {' AND '.join(where)} "
            f"RETURN d ORDER BY d.path, d.version{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy)
        return [_map.document_ref(r["d"]) for r in rows]

    async def count_root_documents(
        self, *, latest_only: bool = True, only_folders: bool = False, only_leaves: bool = False
    ) -> int:
        if only_folders and only_leaves:
            raise NavigationError("only_folders and only_leaves are mutually exclusive")
        where = ["NOT ( ()-[:IS_COMPOSED_OF]->(d) )"]
        if latest_only:
            where.append("d.latest = true")
        if only_folders:
            where.append("d.is_folder = true")
        if only_leaves:
            where.append("d.is_folder = false")
        rec = await self._read_one(
            f"MATCH (d:Document) WHERE {' AND '.join(where)} RETURN count(d) AS c"
        )
        return int(rec["c"]) if rec else 0

    async def get_one_document(self, path: str, version: int) -> DocumentRef | None:
        rec = await self._read_one(
            "MATCH (d:Document {path: $path, version: $version}) RETURN d",
            path=path,
            version=int(version),
        )
        return _map.document_ref(rec["d"]) if rec else None

    async def get_documents(
        self,
        *,
        path: str | None = None,
        name_contains: str | None = None,
        version: int | None = None,
        latest_only: bool = True,
        is_folder: bool | None = None,
        path_prefix: str | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[DocumentRef]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if path is not None:
            clauses.append("d.path = $path")
            params["path"] = path
        if name_contains is not None:
            clauses.append("toLower(d.name) CONTAINS toLower($name_contains)")
            params["name_contains"] = name_contains
        if version is not None:
            clauses.append("d.version = $version")
            params["version"] = int(version)
        if latest_only and version is None:
            clauses.append("d.latest = true")
        if is_folder is not None:
            clauses.append("d.is_folder = $is_folder")
            params["is_folder"] = bool(is_folder)
        if path_prefix is not None:
            clauses.append("d.path STARTS WITH $path_prefix")
            params["path_prefix"] = path_prefix
        wfrag, wparams = translate_where(where, alias="d")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH (d:Document) {where_sql}"
            f"RETURN d ORDER BY d.path, d.version{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["d"]) for r in rows]

    async def document_exists(self, path: str, *, version: int | None = None) -> bool:
        if version is None:
            rec = await self._read_one(
                "RETURN EXISTS { MATCH (:Document {path: $path}) } AS found", path=path
            )
        else:
            rec = await self._read_one(
                "RETURN EXISTS { MATCH (:Document {path: $path, version: $version}) } AS found",
                path=path,
                version=int(version),
            )
        return bool(rec and rec["found"])

    async def get_child_documents(
        self,
        path: str,
        *,
        depth: int | None = 1,
        version: int | None = None,
        is_folder: bool | None = None,
        limit: int | None = None,
    ) -> list[DocumentRef]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        extra = ""
        if is_folder is not None:
            extra = " WHERE c.is_folder = $is_folder"
            params["is_folder"] = bool(is_folder)
        cy = (
            f"MATCH {self._doc_match('root', version=version)} "
            f"MATCH (root)-[:IS_COMPOSED_OF*1..{d}]->(c:Document){extra} "
            f"RETURN DISTINCT c ORDER BY c.path{self._limit_clause(limit)}"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["c"]) for r in rows]

    async def get_document_tree(
        self, path: str, *, depth: int | None = None, version: int | None = None
    ) -> DocumentTree | None:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        cy = (
            f"MATCH {self._doc_match('root', version=version)} "
            f"OPTIONAL MATCH p = (root)-[:IS_COMPOSED_OF*1..{d}]->(c:Document) "
            "RETURN root, c, [x IN nodes(p) | x.path] AS lineage"
        )
        rows = await self._read(cy, **params)
        if not rows:
            return None
        root_props = rows[0]["root"]
        root = DocumentTree(**_map.document_ref(root_props).model_dump(), depth=0)
        by_path: dict[str, DocumentTree] = {root.path: root}
        edges = sorted(
            (r for r in rows if r.get("c")),
            key=lambda r: len(r["lineage"]),
        )
        for r in edges:
            lineage = r["lineage"]
            child = _map.document_ref(r["c"])
            parent_path = lineage[-2] if len(lineage) >= 2 else root.path
            node = DocumentTree(**child.model_dump(), depth=len(lineage) - 1)
            by_path[node.path] = node
            parent = by_path.get(parent_path)
            if parent is not None:
                parent.children.append(node)
        return root

    async def get_document_parent(
        self, path: str, *, version: int | None = None
    ) -> DocumentRef | None:
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        cy = (
            f"MATCH (p:Document)-[:IS_COMPOSED_OF]->{self._doc_match('d', version=version)} "
            "RETURN p ORDER BY p.version DESC LIMIT 1"
        )
        rec = await self._read_one(cy, **params)
        return _map.document_ref(rec["p"]) if rec else None

    async def get_document_ancestors(
        self, path: str, *, version: int | None = None, depth: int | None = None
    ) -> DocumentTree | None:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        cy = (
            f"MATCH {self._doc_match('d', version=version)} "
            f"MATCH p = (root:Document)-[:IS_COMPOSED_OF*1..{d}]->(d) "
            "WHERE NOT ( ()-[:IS_COMPOSED_OF]->(root) ) "
            "WITH p ORDER BY length(p) DESC LIMIT 1 "
            "RETURN [x IN nodes(p)[0..-1] | x] AS spine"
        )
        rec = await self._read_one(cy, **params)
        if not rec or not rec.get("spine"):
            return None
        spine = rec["spine"]
        head = DocumentTree(**_map.document_ref(spine[0]).model_dump(), depth=0)
        cur = head
        for i, props in enumerate(spine[1:], start=1):
            child = DocumentTree(**_map.document_ref(props).model_dump(), depth=i)
            cur.children.append(child)
            cur = child
        return head

    async def get_document_leaves(
        self, path: str, *, version: int | None = None, depth: int | None = None
    ) -> list[DocumentRef]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        cy = (
            f"MATCH {self._doc_match('root', version=version)} "
            f"MATCH (root)-[:IS_COMPOSED_OF*1..{d}]->(leaf:Document) "
            "WHERE NOT (leaf)-[:IS_COMPOSED_OF]->(:Document) "
            "RETURN DISTINCT leaf ORDER BY leaf.path"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["leaf"]) for r in rows]

    async def list_document_versions(self, path: str) -> list[DocumentRef]:
        rows = await self._read(
            "MATCH (d:Document {path: $path}) RETURN d ORDER BY d.version", path=path
        )
        return [_map.document_ref(r["d"]) for r in rows]

    async def get_latest_version(self, path: str) -> DocumentRef | None:
        rec = await self._read_one(
            "MATCH (d:Document {path: $path, latest: true}) RETURN d LIMIT 1", path=path
        )
        return _map.document_ref(rec["d"]) if rec else None

    async def get_version_chain(self, path: str) -> list[DocumentRef]:
        # HAS_NEWER_VERSION is a clean chain; ordering by version is sufficient.
        rows = await self._read(
            "MATCH (d:Document {path: $path}) RETURN d ORDER BY d.version", path=path
        )
        return [_map.document_ref(r["d"]) for r in rows]

    async def get_document_stats(
        self, path: str, *, version: int | None = None
    ) -> DocumentStats | None:
        cdepth = self._containment_depth(None)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        base = self._doc_match("d", version=version)
        core = await self._read_one(
            f"""
            MATCH {base}
            CALL {{ WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
                    RETURN count(n) AS n_nodes }}
            CALL {{ WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_INFO_UNIT]->(u:InfoUnit)
                    RETURN count(u) AS n_units }}
            CALL {{ WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
                    RETURN count(md) AS n_dec,
                           sum(CASE WHEN md.matched_model_class IS NOT NULL THEN 1 ELSE 0 END) AS n_matched,
                           sum(CASE WHEN md.propose_new_model THEN 1 ELSE 0 END) AS n_proposed }}
            CALL {{ WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)
                    OPTIONAL MATCH (er)-[hr*1..{cdepth}]->(mi:ModelInstance)
                    WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
                    RETURN count(DISTINCT er) AS n_er, count(DISTINCT mi) AS n_mi }}
            RETURN d.version AS version, n_nodes, n_units, n_dec, n_matched, n_proposed, n_er, n_mi
            """,
            **params,
        )
        if core is None:
            return None
        roles = await self._read(
            f"MATCH {base}-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode) "
            "RETURN n.role AS role, count(*) AS c",
            **params,
        )
        classes = await self._read(
            f"""MATCH {base}-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)
            MATCH (er)-[hr*1..{cdepth}]->(mi:ModelInstance)
            WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
            RETURN mi.model_class AS mc, count(DISTINCT mi) AS c""",
            **params,
        )
        labels = await self._read(
            f"""MATCH {base}-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)
            -[hr*1..{cdepth}]->(mi:ModelInstance)-[:REFERENCES]->(le:LabeledEntity)
            WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
            RETURN le.label AS lbl, count(DISTINCT le) AS c""",
            **params,
        )
        triples = await self._read_one(
            f"""MATCH {base}-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)
            -[:HAS_EXTRACTION]->(er:ExtractionResult {{model_class:'Triple'}})
            MATCH (er)-[:HAS_ENTITY {{role:'subject'}}]->(s:Entity)
            MATCH (s)-[p]->(o:Entity)<-[:HAS_ENTITY {{role:'object'}}]-(er)
            RETURN count(p) AS c""",
            **params,
        )
        return DocumentStats(
            raw=dict(core),
            path=path,
            version=int(core["version"]),
            structure_nodes=int(core["n_nodes"] or 0),
            structure_nodes_by_role={r["role"]: int(r["c"]) for r in roles if r["role"]},
            info_units=int(core["n_units"] or 0),
            model_decisions=int(core["n_dec"] or 0),
            model_decisions_matched=int(core["n_matched"] or 0),
            model_decisions_proposed=int(core["n_proposed"] or 0),
            extraction_results=int(core["n_er"] or 0),
            model_instances=int(core["n_mi"] or 0),
            model_instances_by_class={r["mc"]: int(r["c"]) for r in classes if r["mc"]},
            labeled_entities=sum(int(r["c"]) for r in labels),
            labeled_entities_by_label={r["lbl"]: int(r["c"]) for r in labels if r["lbl"]},
            triples=int(triples["c"]) if triples else 0,
        )
