"""navigation/neo4j/_entities.py — Group F: LabeledEntity, Entity, triples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from scinr.newton.navigation.models import (
    DocumentRef,
    EntityLabelStat,
    EntityRelation,
    LabeledEntityRef,
    ModelInstanceRef,
    StructureNodeRef,
    Triple,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime, selector_path
from scinr.newton.navigation.neo4j._translate import translate_where


class _EntitiesMixin(_Neo4jRuntime):
    async def get_model_instance_entities(
        self, uid: str, *, label: str | None = None
    ) -> list[LabeledEntityRef]:
        params: dict[str, Any] = {"uid": uid}
        clause = ""
        if label is not None:
            clause = "WHERE le.label = $label "
            params["label"] = label
        rows = await self._read(
            "MATCH (:ModelInstance {uid: $uid})-[r:REFERENCES]->(le:LabeledEntity) "
            f"{clause}"
            "RETURN le, r.field_name AS field_name, r.list_index AS list_index "
            "ORDER BY le.label, le.value",
            **params,
        )
        return [
            _map.labeled_entity_ref(r["le"], field_name=r.get("field_name"), list_index=r.get("list_index"))
            for r in rows
        ]

    async def get_node_entities(
        self, node_id: str, *, label: str | None = None, depth: int | None = None
    ) -> list[LabeledEntityRef]:
        cd = self._containment_depth(depth)
        params: dict[str, Any] = {"node_id": node_id}
        clause = ""
        if label is not None:
            clause = "AND le.label = $label "
            params["label"] = label
        rows = await self._read(
            "MATCH (:StructureNode {id: $node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[hr*1..{cd}]->(mi:ModelInstance) "
            "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') "
            "MATCH (mi)-[:REFERENCES]->(le:LabeledEntity) "
            f"WHERE true {clause}"
            "RETURN DISTINCT le ORDER BY le.label, le.value",
            **params,
        )
        return [_map.labeled_entity_ref(r["le"]) for r in rows]

    async def get_document_entities(
        self,
        document: str | DocumentRef,
        *,
        label: str | None = None,
        version: int | None = None,
        depth: int | None = None,
        limit: int | None = None,
    ) -> list[LabeledEntityRef]:
        cd = self._containment_depth(depth)
        params: dict[str, Any] = {"path": selector_path(document)}
        if version is not None:
            params["version"] = int(version)
        clause = ""
        if label is not None:
            clause = "AND le.label = $label "
            params["label"] = label
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} "
            "MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[hr*1..{cd}]->(mi:ModelInstance) "
            "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') "
            "MATCH (mi)-[:REFERENCES]->(le:LabeledEntity) "
            f"WHERE true {clause}"
            f"RETURN DISTINCT le ORDER BY le.label, le.value{self._limit_clause(limit)}",
            **params,
        )
        return [_map.labeled_entity_ref(r["le"]) for r in rows]

    async def list_entity_labels(self) -> list[EntityLabelStat]:
        rows = await self._read(
            "MATCH (le:LabeledEntity) RETURN le.label AS label, count(*) AS count ORDER BY count DESC"
        )
        return [_map.entity_label_stat(r) for r in rows]

    async def get_labeled_entities(
        self,
        *,
        label: str | None = None,
        value: str | None = None,
        normalized_value: str | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[LabeledEntityRef]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if label is not None:
            clauses.append("le.label = $label")
            params["label"] = label
        if value is not None:
            clauses.append("le.value = $value")
            params["value"] = value
        if normalized_value is not None:
            clauses.append("le.normalized_value = $normalized_value")
            params["normalized_value"] = normalized_value
        wfrag, wparams = translate_where(where, alias="le")
        if wfrag:
            clauses.append(wfrag)
            params.update(wparams)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rows = await self._read(
            f"MATCH (le:LabeledEntity) {where_sql}"
            f"RETURN le ORDER BY le.label, le.value{self._limit_clause(limit, skip)}",
            **params,
        )
        return [_map.labeled_entity_ref(r["le"]) for r in rows]

    async def get_labeled_entity(self, uid: str) -> LabeledEntityRef | None:
        rec = await self._read_one("MATCH (le:LabeledEntity {uid: $uid}) RETURN le", uid=uid)
        return _map.labeled_entity_ref(rec["le"]) if rec else None

    async def get_model_instances_referencing_entity(
        self, uid: str, *, model_class: str | None = None, limit: int | None = None
    ) -> list[ModelInstanceRef]:
        params: dict[str, Any] = {"uid": uid}
        clause = ""
        if model_class is not None:
            clause = "WHERE mi.model_class = $model_class "
            params["model_class"] = model_class
        rows = await self._read(
            "MATCH (mi:ModelInstance)-[:REFERENCES]->(:LabeledEntity {uid: $uid}) "
            f"{clause}"
            f"RETURN DISTINCT mi ORDER BY mi.model_class, mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return [
            _map.model_instance_ref(r["mi"], is_shell=await self._is_shell(r["mi"])) for r in rows
        ]

    async def get_nodes_referencing_entity(
        self, uid: str, *, depth: int | None = None, limit: int | None = None
    ) -> list[StructureNodeRef]:
        cd = self._containment_depth(depth)
        rows = await self._read(
            "MATCH (:LabeledEntity {uid: $uid})<-[:REFERENCES]-(mi:ModelInstance) "
            f"MATCH (mi)<-[hr*1..{cd}]-(er:ExtractionResult) "
            "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') "
            "MATCH (sn:StructureNode)-[:HAS_EXTRACTION]->(er) "
            f"RETURN DISTINCT sn {{ .*, _labels: labels(sn) }} AS n ORDER BY n.id{self._limit_clause(limit)}",
            uid=uid,
        )
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_entity_relationships(
        self,
        uid: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        rel_type: str | None = None,
    ) -> list[EntityRelation]:
        params: dict[str, Any] = {"uid": uid}
        rt = ""
        if rel_type is not None:
            rt = "WHERE type(r) = $rel_type "
            params["rel_type"] = rel_type
        parts: list[str] = []
        if direction in ("out", "both"):
            parts.append(
                "MATCH (le:LabeledEntity {uid: $uid})-[r]->(o:LabeledEntity) "
                f"{rt}RETURN type(r) AS rel_type, 'out' AS direction, o"
            )
        if direction in ("in", "both"):
            parts.append(
                "MATCH (le:LabeledEntity {uid: $uid})<-[r]-(o:LabeledEntity) "
                f"{rt}RETURN type(r) AS rel_type, 'in' AS direction, o"
            )
        rows = await self._read(" UNION ".join(parts), **params)
        return [
            EntityRelation(
                raw=dict(r),
                rel_type=r["rel_type"],
                direction=r["direction"],
                other=_map.labeled_entity_ref(r["o"]),
            )
            for r in rows
        ]

    async def get_related_entities(
        self, uid: str, rel_type: str, *, direction: Literal["out", "in"] = "out"
    ) -> list[LabeledEntityRef]:
        params = {"uid": uid, "rel_type": rel_type}
        if direction == "out":
            pattern = "(:LabeledEntity {uid: $uid})-[r]->(o:LabeledEntity)"
        else:
            pattern = "(o:LabeledEntity)-[r]->(:LabeledEntity {uid: $uid})"
        rows = await self._read(
            f"MATCH {pattern} WHERE type(r) = $rel_type RETURN DISTINCT o ORDER BY o.label, o.value",
            **params,
        )
        return [_map.labeled_entity_ref(r["o"]) for r in rows]

    async def get_triples(self, node_id: str) -> list[Triple]:
        rows = await self._read(
            "MATCH (:StructureNode {id: $node_id})-[:HAS_EXTRACTION]->"
            "(er:ExtractionResult {model_class: 'Triple'}) "
            "MATCH (er)-[:HAS_ENTITY {role: 'subject'}]->(s:Entity) "
            "OPTIONAL MATCH (s)-[p]->(o:Entity)<-[:HAS_ENTITY {role: 'object'}]-(er) "
            "RETURN s.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, "
            "o.value AS object",
            node_id=node_id,
        )
        return [_map.triple(r, node_id=node_id) for r in rows]

    async def get_entity_triples(
        self, value_or_uid: str, *, direction: Literal["out", "in", "both"] = "both"
    ) -> list[Triple]:
        params = {"k": value_or_uid, "kl": value_or_uid.lower()}
        parts: list[str] = []
        if direction in ("out", "both"):
            parts.append(
                "MATCH (e:Entity)-[p]->(o:Entity) "
                "WHERE e.uid = $k OR e.value = $k OR e.normalized_value = $kl "
                "RETURN e.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, o.value AS object"
            )
        if direction in ("in", "both"):
            parts.append(
                "MATCH (s:Entity)-[p]->(e:Entity) "
                "WHERE e.uid = $k OR e.value = $k OR e.normalized_value = $kl "
                "RETURN s.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, e.value AS object"
            )
        rows = await self._read(" UNION ".join(parts), **params)
        return [_map.triple(r) for r in rows]
