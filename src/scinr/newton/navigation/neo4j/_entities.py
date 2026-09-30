"""navigation/neo4j/_entities.py — Group F: LabeledEntity, Entity, triples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._translate import translate_where
from scinr.newton.navigation.scope import make_scope, resolve_selector

# Scope notes (navigation/scope.py): LabeledEntity / Entity / ModelInstance are
# merge-deduplicated — scalar tenant_id (folded into the uid) plus accumulated
# ``job_ids`` / ``created_by_user_ids`` (kind="array"). Anchors guard the
# tenant; returned nodes are filtered by user/job only (``tenant=False``).
# That is safe ONLY because no data relationship crosses tenants (public
# included): every link targets a uid hashed with the writer's own tenant.
# If tenant data is ever linked to public data, every reached node must be
# tenant-checked too — see "Why a traversal cannot leave the tenant" in
# docs/user-guides/graph-navigation.md.
#
# ``get_entity_triples`` matches by *value*: with tenant_id=None the same value
# extracted by several tenants comes back once per tenant (they are distinct
# :Entity nodes). Pass a tenant to avoid mixing them.


def _and(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)} " if clauses else ""


class _EntitiesMixin(_Neo4jRuntime):
    async def get_model_instance_entities(
        self,
        uid: str,
        *,
        label: str | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[LabeledEntityRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("mi", params, sc)
        clauses: list[str] = []
        if label is not None:
            clauses.append("le.label = $label")
            params["label"] = label
        self._scope_where("le", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH (mi:ModelInstance {{uid: $uid}}) {anchor_where}"
            f"MATCH (mi)-[r:REFERENCES]->(le:LabeledEntity) {_and(clauses)}"
            "RETURN le, r.field_name AS field_name, r.list_index AS list_index "
            "ORDER BY le.label, le.value",
            **params,
        )
        return [
            _map.labeled_entity_ref(r["le"], field_name=r.get("field_name"), list_index=r.get("list_index"))
            for r in rows
        ]

    async def get_node_entities(
        self,
        node_id: str,
        *,
        label: str | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[LabeledEntityRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(depth)
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("sn", params, sc)
        clauses: list[str] = []
        if label is not None:
            clauses.append("le.label = $label")
            params["label"] = label
        self._scope_where("le", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH (sn:StructureNode {{id: $node_id}}) {anchor_where}"
            "MATCH (sn)-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[hr*1..{cd}]->(mi:ModelInstance) "
            "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') "
            "MATCH (mi)-[:REFERENCES]->(le:LabeledEntity) "
            f"{_and(clauses)}"
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
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[LabeledEntityRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        cd = self._containment_depth(depth)
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        doc_where = self._doc_where("doc", params, sc)
        clauses: list[str] = []
        if label is not None:
            clauses.append("le.label = $label")
            params["label"] = label
        self._scope_where("le", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH {self._doc_match('doc', version=version)} {doc_where}"
            "MATCH (doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult) "
            f"MATCH (er)-[hr*1..{cd}]->(mi:ModelInstance) "
            "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') "
            "MATCH (mi)-[:REFERENCES]->(le:LabeledEntity) "
            f"{_and(clauses)}"
            f"RETURN DISTINCT le ORDER BY le.label, le.value{self._limit_clause(limit)}",
            **params,
        )
        return [_map.labeled_entity_ref(r["le"]) for r in rows]

    async def list_entity_labels(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[EntityLabelStat]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses: list[str] = []
        params: dict[str, Any] = {}
        self._scope_where("le", clauses, params, sc, "array")
        rows = await self._read(
            f"MATCH (le:LabeledEntity) {_and(clauses)}"
            "RETURN le.label AS label, count(*) AS count ORDER BY count DESC",
            **params,
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
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[LabeledEntityRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
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
        self._scope_where("le", clauses, params, sc, "array")
        rows = await self._read(
            f"MATCH (le:LabeledEntity) {_and(clauses)}"
            f"RETURN le ORDER BY le.label, le.value{self._limit_clause(limit, skip)}",
            **params,
        )
        return [_map.labeled_entity_ref(r["le"]) for r in rows]

    async def get_labeled_entity(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> LabeledEntityRef | None:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        clauses: list[str] = []
        params: dict[str, Any] = {"uid": uid}
        self._scope_where("le", clauses, params, sc, "array")
        rec = await self._read_one(
            f"MATCH (le:LabeledEntity {{uid: $uid}}) {_and(clauses)}RETURN le", **params
        )
        return _map.labeled_entity_ref(rec["le"]) if rec else None

    async def get_model_instances_referencing_entity(
        self,
        uid: str,
        *,
        model_class: str | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("le", params, sc)
        clauses: list[str] = []
        if model_class is not None:
            clauses.append("mi.model_class = $model_class")
            params["model_class"] = model_class
        self._scope_where("mi", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH (le:LabeledEntity {{uid: $uid}}) {anchor_where}"
            f"MATCH (mi:ModelInstance)-[:REFERENCES]->(le) {_and(clauses)}"
            f"RETURN DISTINCT mi ORDER BY mi.model_class, mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return [
            _map.model_instance_ref(r["mi"], is_shell=await self._is_shell(r["mi"])) for r in rows
        ]

    async def get_nodes_referencing_entity(
        self,
        uid: str,
        *,
        depth: int | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(depth)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("le", params, sc)
        clauses: list[str] = []
        self._scope_where("sn", clauses, params, sc, tenant=False)
        rows = await self._read(
            f"MATCH (le:LabeledEntity {{uid: $uid}}) {anchor_where}"
            "MATCH (le)<-[:REFERENCES]-(mi:ModelInstance) "
            f"MATCH (mi)<-[hr*1..{cd}]-(er:ExtractionResult) "
            "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') "
            f"MATCH (sn:StructureNode)-[:HAS_EXTRACTION]->(er) {_and(clauses)}"
            f"RETURN DISTINCT sn {{ .*, _labels: labels(sn) }} AS n ORDER BY n.id{self._limit_clause(limit)}",
            **params,
        )
        return [_map.structure_node_ref(r["n"]) for r in rows]

    async def get_entity_relationships(
        self,
        uid: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        rel_type: str | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[EntityRelation]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid}
        anchor_where = self._doc_where("le", params, sc)
        clauses: list[str] = []
        if rel_type is not None:
            clauses.append("type(r) = $rel_type")
            params["rel_type"] = rel_type
        self._scope_where("o", clauses, params, sc, "array", tenant=False)
        rt = _and(clauses)
        parts: list[str] = []
        if direction in ("out", "both"):
            parts.append(
                f"MATCH (le:LabeledEntity {{uid: $uid}}) {anchor_where}"
                f"MATCH (le)-[r]->(o:LabeledEntity) {rt}"
                "RETURN type(r) AS rel_type, 'out' AS direction, o"
            )
        if direction in ("in", "both"):
            parts.append(
                f"MATCH (le:LabeledEntity {{uid: $uid}}) {anchor_where}"
                f"MATCH (le)<-[r]-(o:LabeledEntity) {rt}"
                "RETURN type(r) AS rel_type, 'in' AS direction, o"
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
        self,
        uid: str,
        rel_type: str,
        *,
        direction: Literal["out", "in"] = "out",
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[LabeledEntityRef]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"uid": uid, "rel_type": rel_type}
        anchor_where = self._doc_where("a", params, sc)
        clauses = ["type(r) = $rel_type"]
        self._scope_where("o", clauses, params, sc, "array", tenant=False)
        if direction == "out":
            pattern = "(a)-[r]->(o:LabeledEntity)"
        else:
            pattern = "(o:LabeledEntity)-[r]->(a)"
        rows = await self._read(
            f"MATCH (a:LabeledEntity {{uid: $uid}}) {anchor_where}"
            f"MATCH {pattern} WHERE {' AND '.join(clauses)} "
            "RETURN DISTINCT o ORDER BY o.label, o.value",
            **params,
        )
        return [_map.labeled_entity_ref(r["o"]) for r in rows]

    async def get_triples(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[Triple]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"node_id": node_id}
        anchor_where = self._doc_where("sn", params, sc)
        clauses: list[str] = []
        self._scope_where("er", clauses, params, sc, tenant=False)
        rows = await self._read(
            f"MATCH (sn:StructureNode {{id: $node_id}}) {anchor_where}"
            "MATCH (sn)-[:HAS_EXTRACTION]->(er:ExtractionResult {model_class: 'Triple'}) "
            f"{_and(clauses)}"
            "MATCH (er)-[:HAS_ENTITY {role: 'subject'}]->(s:Entity) "
            "OPTIONAL MATCH (s)-[p]->(o:Entity)<-[:HAS_ENTITY {role: 'object'}]-(er) "
            "RETURN s.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, "
            "o.value AS object",
            **params,
        )
        return [_map.triple(r, node_id=node_id) for r in rows]

    async def get_entity_triples(
        self,
        value_or_uid: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[Triple]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {"k": value_or_uid, "kl": value_or_uid.lower()}
        e_clauses = ["(e.uid = $k OR e.value = $k OR e.normalized_value = $kl)"]
        self._scope_where("e", e_clauses, params, sc, "array")
        e_where = f"WHERE {' AND '.join(e_clauses)} "
        parts: list[str] = []
        if direction in ("out", "both"):
            parts.append(
                f"MATCH (e:Entity)-[p]->(o:Entity) {e_where}"
                "RETURN e.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, o.value AS object"
            )
        if direction in ("in", "both"):
            parts.append(
                f"MATCH (s:Entity)-[p]->(e:Entity) {e_where}"
                "RETURN s.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, e.value AS object"
            )
        rows = await self._read(" UNION ".join(parts), **params)
        return [_map.triple(r) for r in rows]
