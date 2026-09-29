"""navigation/neo4j/_documents.py — Group A: documents & folder hierarchy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import DocumentRef, DocumentStats, DocumentTree
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._translate import translate_where
from scinr.newton.navigation.scope import make_scope

# Scope notes (see navigation/scope.py). A :Document is keyed by
# (tenant_id, path, version): the same path can exist once per tenant.
#  * list methods return every match in scope (each DocumentRef carries its tenant);
#  * single-document methods resolve the tenant first (``_single_document_scope``)
#    and raise NavigationError when the path is ambiguous;
#  * folder documents are re-MERGEd on every ingestion, so their job_id /
#    created_by_user_id are those of the *last* run that touched them.


def _where(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)} " if clauses else ""


class _DocumentsMixin(_Neo4jRuntime):
    async def list_root_documents(
        self,
        *,
        latest_only: bool = True,
        only_folders: bool = False,
        only_leaves: bool = False,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        if only_folders and only_leaves:
            raise NavigationError("only_folders and only_leaves are mutually exclusive")
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        where = ["NOT ( ()-[:IS_COMPOSED_OF]->(d) )"]
        params: dict[str, Any] = {}
        if latest_only:
            where.append("d.latest = true")
        if only_folders:
            where.append("d.is_folder = true")
        if only_leaves:
            where.append("d.is_folder = false")
        self._scope_where("d", where, params, sc)
        cy = (
            f"MATCH (d:Document) WHERE {' AND '.join(where)} "
            f"RETURN d ORDER BY d.path, d.version{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["d"]) for r in rows]

    async def count_root_documents(
        self,
        *,
        latest_only: bool = True,
        only_folders: bool = False,
        only_leaves: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        if only_folders and only_leaves:
            raise NavigationError("only_folders and only_leaves are mutually exclusive")
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        where = ["NOT ( ()-[:IS_COMPOSED_OF]->(d) )"]
        params: dict[str, Any] = {}
        if latest_only:
            where.append("d.latest = true")
        if only_folders:
            where.append("d.is_folder = true")
        if only_leaves:
            where.append("d.is_folder = false")
        self._scope_where("d", where, params, sc)
        rec = await self._read_one(
            f"MATCH (d:Document) WHERE {' AND '.join(where)} RETURN count(d) AS c", **params
        )
        return int(rec["c"]) if rec else 0

    async def get_one_document(
        self,
        path: str,
        version: int,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        clauses: list[str] = []
        params: dict[str, Any] = {"path": path, "version": int(version)}
        self._scope_where("d", clauses, params, sc)
        rec = await self._read_one(
            f"MATCH (d:Document {{path: $path, version: $version}}) {_where(clauses)}RETURN d",
            **params,
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
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
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
        self._scope_where("d", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH (d:Document) {where_sql}"
            f"RETURN d ORDER BY d.path, d.version{self._limit_clause(limit, skip)}"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["d"]) for r in rows]

    async def document_exists(
        self,
        path: str,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> bool:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses = ["d.path = $path"]
        params: dict[str, Any] = {"path": path}
        if version is not None:
            clauses.append("d.version = $version")
            params["version"] = int(version)
        self._scope_where("d", clauses, params, sc)
        rec = await self._read_one(
            f"RETURN EXISTS {{ MATCH (d:Document) WHERE {' AND '.join(clauses)} }} AS found",
            **params,
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
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        root_where = self._doc_where("root", params, sc)
        clauses: list[str] = []
        if is_folder is not None:
            clauses.append("c.is_folder = $is_folder")
            params["is_folder"] = bool(is_folder)
        self._scope_where("c", clauses, params, sc, tenant=False)
        extra = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cy = (
            f"MATCH {self._doc_match('root', version=version)} {root_where}"
            f"MATCH (root)-[:IS_COMPOSED_OF*1..{d}]->(c:Document){extra} "
            f"RETURN DISTINCT c ORDER BY c.path{self._limit_clause(limit)}"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["c"]) for r in rows]

    async def get_document_tree(
        self,
        path: str,
        *,
        depth: int | None = None,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentTree | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        # user/job filters apply to the anchor only: pruning inner folders
        # would orphan their children.
        root_where = self._doc_where("root", params, sc, "scalar")
        cy = (
            f"MATCH {self._doc_match('root', version=version)} {root_where}"
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
        self,
        path: str,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        d_where = self._doc_where("d", params, sc)
        clauses: list[str] = []
        self._scope_where("p", clauses, params, sc, tenant=False)
        p_where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        cy = (
            f"MATCH {self._doc_match('d', version=version)} {d_where}"
            f"MATCH (p:Document)-[:IS_COMPOSED_OF]->(d) {p_where}"
            "RETURN p ORDER BY p.version DESC LIMIT 1"
        )
        rec = await self._read_one(cy, **params)
        return _map.document_ref(rec["p"]) if rec else None

    async def get_document_ancestors(
        self,
        path: str,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentTree | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        # user/job filters apply to the anchor only (the spine must stay whole).
        d_where = self._doc_where("d", params, sc, "scalar")
        cy = (
            f"MATCH {self._doc_match('d', version=version)} {d_where}"
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
        self,
        path: str,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        root_where = self._doc_where("root", params, sc)
        clauses = ["NOT (leaf)-[:IS_COMPOSED_OF]->(:Document)"]
        self._scope_where("leaf", clauses, params, sc, tenant=False)
        cy = (
            f"MATCH {self._doc_match('root', version=version)} {root_where}"
            f"MATCH (root)-[:IS_COMPOSED_OF*1..{d}]->(leaf:Document) "
            f"WHERE {' AND '.join(clauses)} "
            "RETURN DISTINCT leaf ORDER BY leaf.path"
        )
        rows = await self._read(cy, **params)
        return [_map.document_ref(r["leaf"]) for r in rows]

    async def list_document_versions(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses = ["d.path = $path"]
        params: dict[str, Any] = {"path": path}
        self._scope_where("d", clauses, params, sc)
        rows = await self._read(
            f"MATCH (d:Document) WHERE {' AND '.join(clauses)} "
            "RETURN d ORDER BY d.tenant_id, d.version",
            **params,
        )
        return [_map.document_ref(r["d"]) for r in rows]

    async def get_version_chain(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses = ["d.path = $path"]
        params: dict[str, Any] = {"path": path}
        self._scope_where("d", clauses, params, sc)
        rows = await self._read(
            f"MATCH (d:Document) WHERE {' AND '.join(clauses)} "
            "RETURN d ORDER BY d.tenant_id, d.version",
            **params,
        )
        return [_map.document_ref(r["d"]) for r in rows]

    async def get_latest_version(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, None, sc)
        clauses = ["d.path = $path", "d.latest = true"]
        params: dict[str, Any] = {"path": path}
        self._scope_where("d", clauses, params, sc)
        rec = await self._read_one(
            f"MATCH (d:Document) WHERE {' AND '.join(clauses)} RETURN d LIMIT 1", **params
        )
        return _map.document_ref(rec["d"]) if rec else None

    async def get_document_stats(
        self,
        path: str,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentStats | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        cdepth = self._containment_depth(None)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        # The anchor carries the whole scope; everything below hangs off it.
        base = f"{self._doc_match('d', version=version)} {self._doc_where('d', params, sc, 'scalar')}"
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
            f"MATCH {base} MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode) "
            "RETURN n.role AS role, count(*) AS c",
            **params,
        )
        classes = await self._read(
            f"""MATCH {base}
            MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)
            MATCH (er)-[hr*1..{cdepth}]->(mi:ModelInstance)
            WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
            RETURN mi.model_class AS mc, count(DISTINCT mi) AS c""",
            **params,
        )
        labels = await self._read(
            f"""MATCH {base}
            MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)
            -[hr*1..{cdepth}]->(mi:ModelInstance)-[:REFERENCES]->(le:LabeledEntity)
            WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
            RETURN le.label AS lbl, count(DISTINCT le) AS c""",
            **params,
        )
        triples = await self._read_one(
            f"""MATCH {base}
            MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)
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
