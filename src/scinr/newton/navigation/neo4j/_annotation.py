"""navigation/neo4j/_annotation.py — Group D: annotation (ModelDecision)."""

from __future__ import annotations

from typing import Any

from scinr.newton.navigation.models import (
    AnnotationCoverage,
    DocumentModelProfile,
    DocumentRef,
    ModelDecisionRef,
    ModelDecisionWithNode,
    ProposedModelRef,
    StructureNodeRef,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime, selector_path

_DECISION_EXTRAS = (
    "[(md)-[:MATCHED_MODEL]->(cm) | cm.name][0] AS matched_model, "
    "[(md)-[:HAS_COMPLEMENTARY_MATCH]->()-[:REFERS_TO_MODEL]->(cm) | cm.name] AS complementary_models, "
    "[(md)-[:HAS_SUPPLEMENTARY_FIELD]->(sf) | sf.field_name] AS supplementary_fields"
)


class _AnnotationMixin(_Neo4jRuntime):
    async def get_model_decision(self, node_id: str) -> ModelDecisionRef | None:
        rec = await self._read_one(
            "MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(md:ModelDecision) "
            f"RETURN md, {_DECISION_EXTRAS}",
            node_id=node_id,
        )
        if not rec:
            return None
        return _map.model_decision_ref(
            rec["md"],
            matched_model=rec.get("matched_model"),
            complementary_models=rec.get("complementary_models"),
            supplementary_fields=rec.get("supplementary_fields"),
        )

    async def get_document_model_decisions(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        matched_only: bool | None = None,
        depth: int | None = None,
    ) -> list[ModelDecisionWithNode]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        clause = ""
        if matched_only is True:
            clause = "WHERE md.matched_model_class IS NOT NULL "
        elif matched_only is False:
            clause = "WHERE md.matched_model_class IS NULL "
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} "
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode)"
            "-[:HAS_MODEL_DECISION]->(md:ModelDecision) "
            f"{clause}"
            f"RETURN md, {_DECISION_EXTRAS}, n.id AS node_id, n.title AS node_title "
            "ORDER BY n.appearance_order",
            **params,
        )
        out: list[ModelDecisionWithNode] = []
        for r in rows:
            base = _map.model_decision_ref(
                r["md"],
                matched_model=r.get("matched_model"),
                complementary_models=r.get("complementary_models"),
                supplementary_fields=r.get("supplementary_fields"),
            )
            out.append(
                ModelDecisionWithNode(
                    **base.model_dump(), node_id=r["node_id"], node_title=r.get("node_title")
                )
            )
        return out

    async def get_document_model_profile(
        self, document: str | DocumentRef, *, version: int | None = None, depth: int | None = None
    ) -> DocumentModelProfile | None:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        base = self._doc_match("doc", version=version)
        core = await self._read_one(
            f"""MATCH {base}
            MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode)
            OPTIONAL MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
            OPTIONAL MATCH (md)-[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)
            RETURN doc.version AS version, count(n) AS total,
                   sum(CASE WHEN md IS NULL THEN 1 ELSE 0 END) AS unannotated,
                   collect(DISTINCT pm.schema_name) AS proposed""",
            **params,
        )
        if core is None:
            return None
        matched = await self._read(
            f"""MATCH {base}
            MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(:StructureNode)
            -[:HAS_MODEL_DECISION]->(:ModelDecision)-[:MATCHED_MODEL]->(cm:CatalogModel)
            RETURN cm.name AS model_class, count(*) AS count ORDER BY count DESC""",
            **params,
        )
        complementary = await self._read(
            f"""MATCH {base}
            MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(:StructureNode)
            -[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_COMPLEMENTARY_MATCH]->()
            -[:REFERS_TO_MODEL]->(cm:CatalogModel)
            RETURN cm.name AS model_class, count(*) AS count ORDER BY count DESC""",
            **params,
        )
        return DocumentModelProfile(
            raw=dict(core),
            path=selector_path(document),
            version=int(core["version"]),
            matched=[_map.model_class_stat(r, kind="matched") for r in matched if r["model_class"]],
            complementary=[
                _map.model_class_stat(r, kind="complementary")
                for r in complementary
                if r["model_class"]
            ],
            proposed=[p for p in (core.get("proposed") or []) if p],
            unannotated_nodes=int(core.get("unannotated", 0) or 0),
        )

    async def get_nodes_by_annotated_model(
        self, model_class: str, *, document: str | DocumentRef | None = None
    ) -> list[StructureNodeRef]:
        params: dict[str, Any] = {"mc": model_class}
        clause = ""
        if document is not None:
            clause = (
                "WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) } "
            )
            params["doc_path"] = selector_path(document)
        rows = await self._read(
            "MATCH (n:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)"
            "-[:MATCHED_MODEL]->(:CatalogModel {name: $mc}) "
            f"{clause}"
            "RETURN DISTINCT n { .*, _labels: labels(n) } AS n ORDER BY n.id",
            **params,
        )
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_unannotated_nodes(
        self, document: str | DocumentRef, *, version: int | None = None, depth: int | None = None
    ) -> list[StructureNodeRef]:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} "
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode) "
            "WHERE NOT (n)-[:HAS_MODEL_DECISION]->() "
            "RETURN n { .*, _labels: labels(n) } AS n ORDER BY n.appearance_order",
            **params,
        )
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_proposed_models(
        self, *, document: str | DocumentRef | None = None
    ) -> list[ProposedModelRef]:
        params: dict[str, Any] = {}
        clause = ""
        if document is not None:
            clause = (
                "WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) } "
            )
            params["doc_path"] = selector_path(document)
        rows = await self._read(
            "MATCH (n:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)"
            "-[:HAS_PROPOSED_MODEL]->(pm:ProposedModel) "
            f"{clause}"
            "RETURN pm, [(pm)-[:HAS_PROPOSED_FIELD]->(f) | f { .* }] AS fields, n.id AS node_id "
            "ORDER BY pm.uid",
            **params,
        )
        return [
            _map.proposed_model_ref(r["pm"], fields=r.get("fields"), node_id=r.get("node_id"))
            for r in rows
        ]

    async def get_annotation_coverage(
        self, document: str | DocumentRef, *, version: int | None = None, depth: int | None = None
    ) -> AnnotationCoverage | None:
        d = self._resolve_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        rec = await self._read_one(
            f"MATCH {self._doc_match('doc', version=version)} "
            f"MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..{d}]->(n:StructureNode) "
            "OPTIONAL MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision) "
            "RETURN doc.version AS version, count(n) AS total, count(md) AS annotated, "
            "sum(CASE WHEN md IS NULL THEN 1 ELSE 0 END) AS unannotated, "
            "sum(CASE WHEN md.matched_model_class IS NOT NULL THEN 1 ELSE 0 END) AS matched, "
            "sum(CASE WHEN md.propose_new_model THEN 1 ELSE 0 END) AS proposed",
            **params,
        )
        if rec is None:
            return None
        total = int(rec["total"] or 0)
        annotated = int(rec["annotated"] or 0)
        return AnnotationCoverage(
            raw=dict(rec),
            path=selector_path(document),
            version=int(rec["version"]),
            total_nodes=total,
            annotated=annotated,
            unannotated=int(rec["unannotated"] or 0),
            matched=int(rec["matched"] or 0),
            proposed=int(rec["proposed"] or 0),
            ratio=(annotated / total) if total else 0.0,
        )
