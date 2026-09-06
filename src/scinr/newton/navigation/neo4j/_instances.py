"""navigation/neo4j/_instances.py — Group E: extraction & model instances."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

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
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime, selector_path
from scinr.newton.navigation.neo4j._safe import safe_ident
from scinr.newton.navigation.neo4j._translate import translate_where

# containment = only HAS_* edges between ExtractionResult/ModelInstance nodes
_HAS_ONLY = "all(r IN rels WHERE type(r) STARTS WITH 'HAS_')"


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

    async def get_extraction_result(self, node_id: str) -> ExtractionResultRef | None:
        rec = await self._read_one(
            "MATCH (:StructureNode {id: $node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            "RETURN er, [(er)-[:USES_PRIMARY_MODEL]->(cm) | cm.name][0] AS primary_model, "
            "[(er)-[:USES_COMPLEMENTARY_MODEL]->(cm) | cm.name] AS complementary_models",
            node_id=node_id,
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
    ) -> list[ExtractionResultWithNode]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        clause = ""
        if model_class is not None:
            clause = "WHERE er.model_class = $model_class "
            params["model_class"] = model_class
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} "
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"{clause}"
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
    ) -> list[ModelInstanceRef]:
        d = 1 if direct_only else self._containment_depth(depth)
        clauses: list[str] = [_HAS_ONLY]
        params: dict[str, Any] = {"node_id": node_id}
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        rows = await self._read(
            "MATCH (:StructureNode {id: $node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult) "
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
    ) -> list[ModelInstanceRef]:
        cd = self._containment_depth(depth)
        clauses: list[str] = [_HAS_ONLY]
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} "
            "MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[rels*1..{cd}]->(mi:ModelInstance) "
            f"WHERE {' AND '.join(clauses)} "
            f"RETURN DISTINCT mi ORDER BY mi.uid{self._limit_clause(limit, skip)}",
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
    ) -> int:
        cd = self._containment_depth(depth)
        clauses: list[str] = [_HAS_ONLY]
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        rec = await self._read_one(
            f"MATCH {self._doc_match('doc', version=version)} "
            "MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[rels*1..{cd}]->(mi:ModelInstance) "
            f"WHERE {' AND '.join(clauses)} "
            "RETURN count(DISTINCT mi) AS c",
            **params,
        )
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
    ) -> list[ModelInstanceRef]:
        cd = self._containment_depth(None)
        clauses: list[str] = []
        params: dict[str, Any] = {"model_class": model_class}
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        if document is not None:
            clauses.append(
                f"EXISTS {{ MATCH (mi)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }"
            )
            params["doc_path"] = selector_path(document)
        order_col = f"mi.`{safe_ident(order_by, kind='order_by field')}`" if order_by else "mi.uid"
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rows = await self._read(
            "MATCH (mi:ModelInstance {model_class: $model_class}) "
            f"{where_sql}RETURN mi ORDER BY {order_col}{self._limit_clause(limit, skip)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def count_model_instances_by_class(
        self,
        model_class: str,
        *,
        where: Mapping[str, Any] | None = None,
        document: str | DocumentRef | None = None,
    ) -> int:
        cd = self._containment_depth(None)
        clauses: list[str] = []
        params: dict[str, Any] = {"model_class": model_class}
        wfrag, wparams = translate_where(where, alias="mi")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        if document is not None:
            clauses.append(
                f"EXISTS {{ MATCH (mi)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }"
            )
            params["doc_path"] = selector_path(document)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rec = await self._read_one(
            f"MATCH (mi:ModelInstance {{model_class: $model_class}}) {where_sql}RETURN count(mi) AS c",
            **params,
        )
        return int(rec["c"]) if rec else 0

    async def get_model_instance(self, uid: str) -> ModelInstanceRef | None:
        rec = await self._read_one("MATCH (mi:ModelInstance {uid: $uid}) RETURN mi", uid=uid)
        if not rec:
            return None
        return _map.model_instance_ref(rec["mi"], is_shell=await self._is_shell(rec["mi"]))

    async def get_model_instance_by_key(
        self, model_class: str, key_fields: Mapping[str, str]
    ) -> ModelInstanceRef | None:
        from scinr.newton.utils.uid import make_instance_uid, normalize_key

        norm = {k: normalize_key(str(v)) for k, v in key_fields.items()}
        uid = make_instance_uid(model_class, norm)
        return await self.get_model_instance(uid)

    # -- provenance: which node / document / ER owns an instance ----------

    async def get_structure_nodes_for_model_instance(self, uid: str) -> list[StructureNodeRef]:
        cd = self._containment_depth(None)
        rows = await self._read(
            "MATCH (mi:ModelInstance {uid: $uid}) "
            "MATCH p = (sn:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)"
            f"-[rels*1..{cd}]->(mi) "
            f"WHERE {_HAS_ONLY} "
            "RETURN DISTINCT sn { .*, _labels: labels(sn) } AS n, min(length(p)) AS hops "
            "ORDER BY hops",
            uid=uid,
        )
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_documents_for_model_instance(self, uid: str) -> list[DocumentRef]:
        cd = self._containment_depth(None)
        rows = await self._read(
            "MATCH (mi:ModelInstance {uid: $uid}) "
            f"MATCH (er:ExtractionResult)-[rels*1..{cd}]->(mi) WHERE {_HAS_ONLY} "
            "MATCH (er)<-[:HAS_EXTRACTION]-(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(d:Document) "
            "RETURN DISTINCT d ORDER BY d.path, d.version",
            uid=uid,
        )
        return [_map.document_ref(r["d"]) for r in rows]

    async def get_extraction_results_for_model_instance(
        self, uid: str
    ) -> list[ExtractionResultRef]:
        cd = self._containment_depth(None)
        rows = await self._read(
            "MATCH (mi:ModelInstance {uid: $uid}) "
            f"MATCH (er:ExtractionResult)-[rels*1..{cd}]->(mi) WHERE {_HAS_ONLY} "
            "RETURN DISTINCT er",
            uid=uid,
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
    ) -> list[ModelInstanceRef]:
        return await self._directional(uid, "in", rel_type=rel_type, depth=depth, limit=limit)

    async def get_outgoing_model_instances(
        self,
        uid: str,
        *,
        rel_type: str | None = None,
        depth: int | None = 1,
        limit: int | None = None,
    ) -> list[ModelInstanceRef]:
        return await self._directional(uid, "out", rel_type=rel_type, depth=depth, limit=limit)

    async def _directional(
        self,
        uid: str,
        direction: Literal["in", "out"],
        *,
        rel_type: str | None,
        depth: int | None,
        limit: int | None,
    ) -> list[ModelInstanceRef]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"uid": uid}
        rt = ""
        if rel_type is not None:
            rt = "WHERE all(x IN r WHERE type(x) = $rel_type) "
            params["rel_type"] = rel_type
        if direction == "out":
            pattern = f"(:ModelInstance {{uid: $uid}})-[r*1..{d}]->(o:ModelInstance)"
        else:
            pattern = f"(o:ModelInstance)-[r*1..{d}]->(:ModelInstance {{uid: $uid}})"
        rows = await self._read(
            f"MATCH {pattern} {rt}"
            f"RETURN DISTINCT o AS mi, type(r[{0 if direction == 'out' else -1}]) AS via_rel, "
            f"'{direction}' AS direction "
            f"ORDER BY via_rel, mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def get_model_instance_subtree(
        self, uid: str, *, depth: int | None = None
    ) -> ModelInstanceTree | None:
        d = self._resolve_depth(depth)
        rows = await self._read(
            "MATCH (root:ModelInstance {uid: $uid}) "
            f"OPTIONAL MATCH p = (root)-[rels*1..{d}]->(c:ModelInstance) "
            "WHERE all(n IN nodes(p) WHERE n:ModelInstance) "
            "RETURN root, c, [x IN nodes(p) | x.uid] AS lineage, "
            "[x IN relationships(p) | type(x)] AS rtypes",
            uid=uid,
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
    ) -> list[ModelInstanceRelation]:
        params: dict[str, Any] = {"uid": uid}
        rt = ""
        if rel_type is not None:
            rt = "WHERE type(r) = $rel_type "
            params["rel_type"] = rel_type
        parts: list[str] = []
        if direction in ("out", "both"):
            parts.append(
                "MATCH (mi:ModelInstance {uid: $uid})-[r]->(o:ModelInstance) "
                f"{rt}RETURN type(r) AS rel_type, 'out' AS direction, o"
            )
        if direction in ("in", "both"):
            parts.append(
                "MATCH (mi:ModelInstance {uid: $uid})<-[r]-(o:ModelInstance) "
                f"{rt}RETURN type(r) AS rel_type, 'in' AS direction, o"
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
        self, uid: str, rel_type: str, *, direction: Literal["out", "in"] = "out"
    ) -> list[ModelInstanceRef]:
        params = {"uid": uid, "rel_type": rel_type}
        if direction == "out":
            pattern = "(:ModelInstance {uid: $uid})-[r]->(o:ModelInstance)"
        else:
            pattern = "(o:ModelInstance)-[r]->(:ModelInstance {uid: $uid})"
        rows = await self._read(
            f"MATCH {pattern} WHERE type(r) = $rel_type "
            f"RETURN DISTINCT o AS mi, type(r) AS via_rel, '{direction}' AS direction "
            "ORDER BY mi.uid",
            **params,
        )
        return await self._mi_refs(rows)

    async def find_shell_model_instances(
        self, *, model_class: str | None = None, limit: int | None = None
    ) -> list[ModelInstanceRef]:
        params: dict[str, Any] = {}
        clause = ""
        if model_class is not None:
            clause = "AND mi2.model_class = $model_class "
            params["model_class"] = model_class
        rows = await self._read(
            "MATCH (mi:ModelInstance) WITH mi.model_class AS m, avg(size(keys(mi))) AS avgk "
            "MATCH (mi2:ModelInstance {model_class: m}) "
            "WITH mi2, avgk WHERE (size(keys(mi2)) < avgk * 0.5 OR size(keys(mi2)) <= 3) "
            f"{clause}"
            f"RETURN mi2 AS mi ORDER BY mi.model_class, mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return await self._mi_refs(rows)

    async def list_model_instance_relationship_types(
        self, *, document: str | DocumentRef | None = None
    ) -> list[RelTypeStat]:
        cd = self._containment_depth(None)
        params: dict[str, Any] = {}
        clause = "WHERE NOT type(r) STARTS WITH 'HAS_' "
        if document is not None:
            clause += (
                f"AND EXISTS {{ MATCH (a)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_') } "
            )
            params["doc_path"] = selector_path(document)
        rows = await self._read(
            "MATCH (a:ModelInstance)-[r]->(b:ModelInstance) "
            f"{clause}"
            "RETURN a.model_class AS source_model, type(r) AS rel_type, "
            "b.model_class AS target_model, count(*) AS count ORDER BY count DESC",
            **params,
        )
        return [_map.rel_type_stat(r) for r in rows]
