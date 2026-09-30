"""navigation/neo4j/_instances.py — Group E: extraction & model instances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import (
    DocumentRef,
    ExtractionResultRef,
    ExtractionResultWithNode,
    ModelInstanceRef,
    ModelInstanceRelation,
    ModelInstanceTree,
    RelTypeStat,
    StructureNodeRef,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._safe import safe_ident
from scinr.newton.navigation.neo4j._translate import translate_where
from scinr.newton.navigation.scope import make_scope, resolve_selector

# containment = only HAS_* edges between ExtractionResult/ModelInstance nodes
_HAS_ONLY = "all(r IN rels WHERE type(r) STARTS WITH 'HAS_')"

# Scope notes (navigation/scope.py). ExtractionResult carries scalar
# provenance; ModelInstance / LabeledEntity are merge-deduplicated, so their
# tenant is folded into the uid (scalar tenant_id) and job / user accumulate in
# ``job_ids`` / ``created_by_user_ids`` (kind="array": any-of match). Anchors
# guard the tenant; the returned nodes are filtered by user/job. Only the
# "primary" getters (get_model_instance, ...) and the tree root apply the whole
# scope to the anchor. Not re-checking the tenant past the anchor is safe ONLY
# because no data relationship crosses tenants (public included) — see the
# note in _entities.py and docs/user-guides/graph-navigation.md.


def _and(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)} " if clauses else ""


class _ModelInstancesMixin(_Neo4jRuntime):
    async def _mi_refs(
        self, rows: list[dict[str, Any]], *, key: str = "mi"
    ) -> list[ModelInstanceRef]:
        out: list[ModelInstanceRef] = []
        for r in rows:
            node = r[key]
            out.append(
                _map.model_instance_ref(
                    node,
                    via_rel=r.get("via_rel"),
                    direction=r.get("direction"),
                    index=r.get("index"),
                    is_shell=await self._is_shell(node),
                )
            )
        return out

    # -- extraction results -------------------------------------------------

    async def get_extraction_result(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> ExtractionResultRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("sn", params, sc)
        clauses: list[str] = []
        self._scope_where("er", clauses, params, sc, tenant=False)
        rec = await self._read_one(
            f"MATCH (sn:StructureNode {{id: $node_id}}) {anchor_where}"
            f"MATCH (sn)-[:HAS_EXTRACTION]->(er:ExtractionResult) {_and(clauses)}"
            "RETURN er, [(er)-[:USES_PRIMARY_MODEL]->(cm) | cm.name][0] AS primary_model, "
            "[(er)-[:USES_COMPLEMENTARY_MODEL]->(cm) | cm.name] AS complementary_models",
            **params,
        )
        if not rec:
            return None
        return _map.extraction_result_ref(
            rec["er"],
            primary_model=rec.get("primary_model"),
            complementary_models=rec.get("complementary_models"),
        )

    async def get_document_extraction_results(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        model_class: str | None = None,
        depth: int | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ExtractionResultWithNode]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        clauses: list[str] = []
        if model_class is not None:
            clauses.append("er.model_class = $model_class")
            params["model_class"] = model_class
        self._scope_where("er", clauses, params, sc, tenant=False)
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"{_and(clauses)}"
            "RETURN er, n.id AS node_id, n.title AS node_title, "
            "[(er)-[:USES_PRIMARY_MODEL]->(cm) | cm.name][0] AS primary_model, "
            "[(er)-[:USES_COMPLEMENTARY_MODEL]->(cm) | cm.name] AS complementary_models "
            f"ORDER BY n.appearance_order{self._limit_clause(limit)}",
            **params,
        )
        out: list[ExtractionResultWithNode] = []
        for r in rows:
            base = _map.extraction_result_ref(
                r["er"],
                primary_model=r.get("primary_model"),
                complementary_models=r.get("complementary_models"),
            )
            out.append(
                ExtractionResultWithNode(
                    **base.model_dump(), node_id=r["node_id"], node_title=r.get("node_title")
                )
            )
        return out

    # -- model instances of a node / document -----------------------------

    async def get_node_model_instances(
        self,
        node_id: str,
        *,
        model_class: str | None = None,
        where: Mapping[str, Any] | None = None,
        depth: int | None = None,
        direct_only: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = 1 if direct_only else self._containment_depth(depth)
        clauses: list[str] = [_HAS_ONLY]
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("sn", params, sc)
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        self._scope_where("mi", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH (sn:StructureNode {{id: $node_id}}) {anchor_where}"
            "MATCH (sn)-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[rels*1..{d}]->(mi:ModelInstance) "
            f"WHERE {' AND '.join(clauses)} "
            "RETURN DISTINCT mi ORDER BY mi.model_class, mi.uid",
            **params,
        )
        return await self._mi_refs(rows)

    async def get_document_model_instances(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        model_class: str | None = None,
        where: Mapping[str, Any] | None = None,
        depth: int | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        cd = self._containment_depth(depth)
        clauses: list[str] = [_HAS_ONLY]
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        self._scope_where("mi", clauses, params, sc, "array", tenant=False)
        head = (
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            "MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[rels*1..{cd}]->(mi:ModelInstance) "
            f"WHERE {' AND '.join(clauses)} "
        )
        rows = await self._read(
            f"{head}RETURN DISTINCT mi ORDER BY mi.uid{self._limit_clause(limit, skip)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def count_document_model_instances(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        model_class: str | None = None,
        where: Mapping[str, Any] | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        cd = self._containment_depth(depth)
        clauses: list[str] = [_HAS_ONLY]
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        self._scope_where("mi", clauses, params, sc, "array", tenant=False)
        head = (
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            "MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[rels*1..{cd}]->(mi:ModelInstance) "
            f"WHERE {' AND '.join(clauses)} "
        )
        rec = await self._read_one(f"{head}RETURN count(DISTINCT mi) AS c", **params)
        return int(rec["c"]) if rec else 0

    async def get_model_instances_by_class(
        self,
        model_class: str,
        *,
        where: Mapping[str, Any] | None = None,
        document: str | DocumentRef | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        clauses: list[str] = []
        params: dict[str, Any] = {"model_class": model_class}
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                f"EXISTS {{ MATCH (mi)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }"
            )
            params["doc_path"] = path
        self._scope_where("mi", clauses, params, sc, "array")
        order_col = f"mi.`{safe_ident(order_by, kind='order_by field')}`" if order_by else "mi.uid"
        rows = await self._read(
            "MATCH (mi:ModelInstance {model_class: $model_class}) "
            f"{_and(clauses)}RETURN mi ORDER BY {order_col}{self._limit_clause(limit, skip)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def count_model_instances_by_class(
        self,
        model_class: str,
        *,
        where: Mapping[str, Any] | None = None,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        clauses: list[str] = []
        params: dict[str, Any] = {"model_class": model_class}
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                f"EXISTS {{ MATCH (mi)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }"
            )
            params["doc_path"] = path
        self._scope_where("mi", clauses, params, sc, "array")
        rec = await self._read_one(
            f"MATCH (mi:ModelInstance {{model_class: $model_class}}) {_and(clauses)}RETURN count(mi) AS c",
            **params,
        )
        return int(rec["c"]) if rec else 0

    async def get_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> ModelInstanceRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses: list[str] = []
        params: dict[str, Any] = {"uid": uid}
        self._scope_where("mi", clauses, params, sc, "array")
        rec = await self._read_one(
            f"MATCH (mi:ModelInstance {{uid: $uid}}) {_and(clauses)}RETURN mi", **params
        )
        if not rec:
            return None
        return _map.model_instance_ref(rec["mi"], is_shell=await self._is_shell(rec["mi"]))

    async def get_model_instance_by_key(
        self,
        model_class: str,
        key_fields: Mapping[str, str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> ModelInstanceRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        from scinr.newton.utils.uid import make_instance_uid, normalize_key

        # The uid embeds the tenant, so it cannot be "all tenants": the lookup
        # needs a concrete tenant (or "__public__"), and include_public tries
        # the tenant first, then the public one.
        if sc.tenants is None:
            raise NavigationError(
                "get_model_instance_by_key needs a concrete tenant_id (or '__public__'): "
                "the instance uid embeds the tenant"
            )
        norm = {k: normalize_key(str(v)) for k, v in key_fields.items()}
        for tenant in sc.tenants:
            ref = await self.get_model_instance(
                make_instance_uid(model_class, norm, tenant),
                tenant_id=tenant,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            if ref is not None:
                return ref
        return None

    # -- provenance: which node / document / ER owns an instance ----------

    async def get_structure_nodes_for_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("mi", params, sc)
        clauses = [_HAS_ONLY]
        self._scope_where("sn", clauses, params, sc, tenant=False)
        rows = await self._read(
            f"MATCH (mi:ModelInstance {{uid: $uid}}) {anchor_where}"
            "MATCH p = (sn:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)"
            f"-[rels*1..{cd}]->(mi) "
            f"WHERE {' AND '.join(clauses)} "
            "RETURN DISTINCT sn { .*, _labels: labels(sn) } AS n, min(length(p)) AS hops "
            "ORDER BY hops",
            **params,
        )
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_documents_for_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("mi", params, sc)
        clauses: list[str] = []
        self._scope_where("d", clauses, params, sc, tenant=False)
        rows = await self._read(
            f"MATCH (mi:ModelInstance {{uid: $uid}}) {anchor_where}"
            f"MATCH (er:ExtractionResult)-[rels*1..{cd}]->(mi) WHERE {_HAS_ONLY} "
            "MATCH (er)<-[:HAS_EXTRACTION]-(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(d:Document) "
            f"{_and(clauses)}"
            "RETURN DISTINCT d ORDER BY d.path, d.version",
            **params,
        )
        return [_map.document_ref(r["d"]) for r in rows]

    async def get_extraction_results_for_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ExtractionResultRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("mi", params, sc)
        clauses = [_HAS_ONLY]
        self._scope_where("er", clauses, params, sc, tenant=False)
        rows = await self._read(
            f"MATCH (mi:ModelInstance {{uid: $uid}}) {anchor_where}"
            f"MATCH (er:ExtractionResult)-[rels*1..{cd}]->(mi) WHERE {' AND '.join(clauses)} "
            "RETURN DISTINCT er",
            **params,
        )
        return [_map.extraction_result_ref(r["er"]) for r in rows]

    # -- outgoing / incoming (any rel type) ------------------------------

    async def get_incoming_model_instances(
        self,
        uid: str,
        *,
        rel_type: str | None = None,
        depth: int | None = 1,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        return await self._directional(
            uid, "in", rel_type=rel_type, depth=depth, limit=limit,
            tenant_id=tenant_id, include_public=include_public,
            created_by_user_id=created_by_user_id, job_id=job_id,
        )

    async def get_outgoing_model_instances(
        self,
        uid: str,
        *,
        rel_type: str | None = None,
        depth: int | None = 1,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        return await self._directional(
            uid, "out", rel_type=rel_type, depth=depth, limit=limit,
            tenant_id=tenant_id, include_public=include_public,
            created_by_user_id=created_by_user_id, job_id=job_id,
        )

    async def _directional(
        self,
        uid: str,
        direction: Literal["in", "out"],
        *,
        rel_type: str | None,
        depth: int | None,
        limit: int | None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("a", params, sc)
        clauses: list[str] = []
        if rel_type is not None:
            clauses.append("all(x IN r WHERE type(x) = $rel_type)")
            params["rel_type"] = rel_type
        self._scope_where("o", clauses, params, sc, "array", tenant=False)
        if direction == "out":
            pattern = f"(a)-[r*1..{d}]->(o:ModelInstance)"
        else:
            pattern = f"(o:ModelInstance)-[r*1..{d}]->(a)"
        rows = await self._read(
            f"MATCH (a:ModelInstance {{uid: $uid}}) {anchor_where}"
            f"MATCH {pattern} {_and(clauses)}"
            f"RETURN DISTINCT o AS mi, type(r[{0 if direction == 'out' else -1}]) AS via_rel, "
            f"'{direction}' AS direction "
            f"ORDER BY via_rel, mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def get_model_instance_subtree(
        self,
        uid: str,
        *,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> ModelInstanceTree | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"uid": uid}
        # user/job filters apply to the root only (inner nodes are not pruned).
        root_where = self._doc_where("root", params, sc, "array")
        rows = await self._read(
            f"MATCH (root:ModelInstance {{uid: $uid}}) {root_where}"
            f"OPTIONAL MATCH p = (root)-[rels*1..{d}]->(c:ModelInstance) "
            "WHERE all(n IN nodes(p) WHERE n:ModelInstance) "
            "RETURN root, c, [x IN nodes(p) | x.uid] AS lineage, "
            "[x IN relationships(p) | type(x)] AS rtypes",
            **params,
        )
        if not rows:
            return None
        root = ModelInstanceTree(
            **_map.model_instance_ref(rows[0]["root"], is_shell=await self._is_shell(rows[0]["root"])).model_dump(),
            depth=0,
        )
        by_uid: dict[str, ModelInstanceTree] = {root.uid: root}
        for r in sorted((r for r in rows if r.get("c")), key=lambda r: len(r["lineage"])):
            lineage = r["lineage"]
            rtypes = r.get("rtypes") or []
            node = ModelInstanceTree(
                **_map.model_instance_ref(
                    r["c"],
                    via_rel=rtypes[-1] if rtypes else None,
                    is_shell=await self._is_shell(r["c"]),
                ).model_dump(),
                depth=len(lineage) - 1,
            )
            by_uid[node.uid] = node
            parent = by_uid.get(lineage[-2]) if len(lineage) >= 2 else root
            if parent is not None:
                parent.children.append(node)
        return root

    async def get_model_instance_relationships(
        self,
        uid: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        rel_type: str | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRelation]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("mi", params, sc)
        clauses: list[str] = []
        if rel_type is not None:
            clauses.append("type(r) = $rel_type")
            params["rel_type"] = rel_type
        self._scope_where("o", clauses, params, sc, "array", tenant=False)
        rt = _and(clauses)
        parts: list[str] = []
        if direction in ("out", "both"):
            parts.append(
                f"MATCH (mi:ModelInstance {{uid: $uid}}) {anchor_where}"
                f"MATCH (mi)-[r]->(o:ModelInstance) {rt}"
                "RETURN type(r) AS rel_type, 'out' AS direction, o"
            )
        if direction in ("in", "both"):
            parts.append(
                f"MATCH (mi:ModelInstance {{uid: $uid}}) {anchor_where}"
                f"MATCH (mi)<-[r]-(o:ModelInstance) {rt}"
                "RETURN type(r) AS rel_type, 'in' AS direction, o"
            )
        rows = await self._read(" UNION ".join(parts), **params)
        out: list[ModelInstanceRelation] = []
        for r in rows:
            out.append(
                ModelInstanceRelation(
                    raw=dict(r),
                    rel_type=r["rel_type"],
                    direction=r["direction"],
                    other=_map.model_instance_ref(r["o"], is_shell=await self._is_shell(r["o"])),
                )
            )
        return out

    async def get_related_model_instances(
        self,
        uid: str,
        rel_type: str,
        *,
        direction: Literal["out", "in"] = "out",
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid, "rel_type": rel_type}
        anchor_where = self._doc_where("a", params, sc)
        clauses = ["type(r) = $rel_type"]
        self._scope_where("o", clauses, params, sc, "array", tenant=False)
        if direction == "out":
            pattern = "(a)-[r]->(o:ModelInstance)"
        else:
            pattern = "(o:ModelInstance)-[r]->(a)"
        rows = await self._read(
            f"MATCH (a:ModelInstance {{uid: $uid}}) {anchor_where}"
            f"MATCH {pattern} WHERE {' AND '.join(clauses)} "
            f"RETURN DISTINCT o AS mi, type(r) AS via_rel, '{direction}' AS direction "
            "ORDER BY mi.uid",
            **params,
        )
        return await self._mi_refs(rows)

    async def find_shell_model_instances(
        self,
        *,
        model_class: str | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {}
        # The population the "shell" average is computed over is the scope
        # itself, so one tenant's data never skews another's.
        # Both MATCHes filter model class and tenant in their own WHERE (not
        # after a WITH), so each can seek on (tenant_id, model_class).
        pop: list[str] = []
        if model_class is not None:
            pop.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        self._scope_where("mi", pop, params, sc, "array")
        sel: list[str] = []
        self._scope_where("mi2", sel, params, sc, "array")
        sel.append("(size(keys(mi2)) < avgk * 0.5 OR size(keys(mi2)) <= 3)")
        rows = await self._read(
            f"MATCH (mi:ModelInstance) {_and(pop)}"
            "WITH mi.model_class AS m, avg(size(keys(mi))) AS avgk "
            f"MATCH (mi2:ModelInstance {{model_class: m}}) WHERE {' AND '.join(sel)} "
            f"RETURN mi2 AS mi ORDER BY mi.model_class, mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def list_model_instance_relationship_types(
        self,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[RelTypeStat]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        params: dict[str, Any] = {}
        clauses = ["NOT type(r) STARTS WITH 'HAS_'"]
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                f"EXISTS {{ MATCH (a)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_') }"
            )
            params["doc_path"] = path
        self._scope_where("a", clauses, params, sc, "array")
        rows = await self._read(
            "MATCH (a:ModelInstance)-[r]->(b:ModelInstance) "
            f"{_and(clauses)}"
            "RETURN a.model_class AS source_model, type(r) AS rel_type, "
            "b.model_class AS target_model, count(*) AS count ORDER BY count DESC",
            **params,
        )
        return [_map.rel_type_stat(r) for r in rows]
