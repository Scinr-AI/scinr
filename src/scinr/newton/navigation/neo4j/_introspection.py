"""navigation/neo4j/_introspection.py — Group G: catalogue & schema introspection."""

from __future__ import annotations

from typing import Any

from scinr.newton.navigation.models import (
    CatalogGraph,
    CatalogModelRef,
    DocumentRef,
    GraphSummary,
    ModelClassStat,
    RoleStat,
    ThemeRef,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime, selector_path


class _IntrospectionMixin(_Neo4jRuntime):
    async def list_catalog_models(
        self, *, include_fields: bool = False
    ) -> list[CatalogModelRef]:
        if include_fields:
            rows = await self._read(
                "MATCH (cm:CatalogModel) "
                "OPTIONAL MATCH (cm)-[hf:HAS_FIELD]->(f:ModelField) "
                "OPTIONAL MATCH (cm)-[:BELONGS_TO_THEME]->(t:Theme) "
                "RETURN cm, "
                "collect(DISTINCT {name: f.name, type: f.type, entity_label: f.entity_label, "
                "is_instance_key: f.is_instance_key, required: hf.required, description: hf.description}) AS fields, "
                "collect(DISTINCT t.name) AS themes ORDER BY cm.name"
            )
            return [
                _map.catalog_model_ref(r["cm"], fields=r["fields"], themes=r["themes"])
                for r in rows
            ]
        rows = await self._read("MATCH (cm:CatalogModel) RETURN cm ORDER BY cm.name")
        return [_map.catalog_model_ref(r["cm"]) for r in rows]

    async def get_catalog_graph(
        self, *, include_fields: bool = True, include_relationships: bool = True
    ) -> CatalogGraph:
        models = await self.list_catalog_models(include_fields=include_fields)
        labels_rows = await self._read("MATCH (el:EntityLabel) RETURN el.label AS label ORDER BY label")
        entity_labels = [r["label"] for r in labels_rows if r["label"]]
        relationships = []
        if include_relationships:
            rows = await self._read(
                "MATCH (a)-[r]->(b) "
                "WHERE (a:CatalogModel OR a:EntityLabel) AND (b:CatalogModel OR b:EntityLabel) "
                "AND NOT type(r) IN ['HAS_FIELD', 'BELONGS_TO_THEME'] "
                "RETURN labels(a)[0] AS source_kind, coalesce(a.name, a.label) AS source, "
                "type(r) AS rel_type, properties(r) AS props, "
                "labels(b)[0] AS target_kind, coalesce(b.name, b.label) AS target"
            )
            relationships = [_map.catalog_relation(r) for r in rows]
        return CatalogGraph(
            raw={"models": len(models), "relationships": len(relationships)},
            models=models,
            entity_labels=entity_labels,
            relationships=relationships,
        )

    async def list_model_classes_in_use(
        self, *, document: str | DocumentRef | None = None
    ) -> list[ModelClassStat]:
        cd = self._containment_depth(None)
        params: dict[str, Any] = {}
        clause = ""
        if document is not None:
            clause = (
                f"WHERE EXISTS {{ MATCH (mi)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') } "
            )
            params["doc_path"] = selector_path(document)
        rows = await self._read(
            f"MATCH (mi:ModelInstance) {clause}"
            "RETURN mi.model_class AS model_class, count(*) AS count ORDER BY count DESC",
            **params,
        )
        return [_map.model_class_stat(r, kind="in_use") for r in rows if r["model_class"]]

    async def get_model_properties(
        self, model_class: str, *, document: str | DocumentRef | None = None
    ) -> dict[str, list[str]]:
        declared_rec = await self._read_one(
            "MATCH (:CatalogModel {name: $mc})-[:HAS_FIELD]->(f:ModelField) "
            "RETURN collect(DISTINCT f.name) AS declared",
            mc=model_class,
        )
        observed_rec = await self._read_one(
            "MATCH (mi:ModelInstance {model_class: $mc}) WITH mi LIMIT 500 "
            "UNWIND keys(mi) AS k WITH DISTINCT k WHERE NOT k IN ['uid', 'model_class'] "
            "RETURN collect(k) AS observed",
            mc=model_class,
        )
        return {
            "declared": sorted((declared_rec or {}).get("declared", []) or []),
            "observed": sorted((observed_rec or {}).get("observed", []) or []),
        }

    async def list_node_roles(
        self, *, document: str | DocumentRef | None = None
    ) -> list[RoleStat]:
        params: dict[str, Any] = {}
        clause = ""
        if document is not None:
            clause = (
                "WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) } "
            )
            params["doc_path"] = selector_path(document)
        rows = await self._read(
            f"MATCH (n:StructureNode) {clause}"
            "RETURN n.role AS role, count(*) AS count ORDER BY count DESC",
            **params,
        )
        return [_map.role_stat(r) for r in rows if r["role"]]

    async def list_themes(self) -> list[ThemeRef]:
        rows = await self._read("MATCH (t:Theme) RETURN t ORDER BY t.path")
        if rows:
            return [_map.theme_ref(r["t"]) for r in rows]
        # Fallback: distinct StructureNode.theme values.
        rows = await self._read(
            "MATCH (n:StructureNode) WHERE n.theme IS NOT NULL "
            "RETURN DISTINCT n.theme AS name ORDER BY name"
        )
        return [ThemeRef(raw=r, name=r["name"]) for r in rows if r["name"]]

    async def _structural_rel_types(self) -> list[str]:
        rows = await self._read(
            "MATCH (a)-[r]->(b) "
            "WITH type(r) AS t, collect(DISTINCT (labels(a)[0] + '->' + labels(b)[0])) AS pairs "
            "WHERE NOT all(p IN pairs WHERE p = 'Entity->Entity') "
            "RETURN collect(DISTINCT t) AS types"
        )
        return sorted(rows[0]["types"]) if rows else []

    async def list_relationship_types(self, *, structural_only: bool = True) -> list[str]:
        if structural_only:
            return await self._structural_rel_types()
        rows = await self._read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t ORDER BY t"
        )
        return [r["t"] for r in rows]

    async def list_node_labels(self) -> list[str]:
        rows = await self._read("CALL db.labels() YIELD label RETURN label ORDER BY label")
        return [r["label"] for r in rows]

    async def get_graph_summary(self) -> GraphSummary:
        try:
            scalars = await self._read_one(
                "CALL apoc.meta.stats() YIELD nodeCount, relCount, labelCount, relTypeCount "
                "RETURN nodeCount, relCount, labelCount, relTypeCount"
            )
        except Exception:  # noqa: BLE001 — APOC not available
            scalars = None
        if scalars is None:
            n = await self._read_one("MATCH (x) RETURN count(x) AS c")
            r = await self._read_one("MATCH ()-[e]->() RETURN count(e) AS c")
            scalars = {
                "nodeCount": (n or {}).get("c", 0),
                "relCount": (r or {}).get("c", 0),
                "labelCount": 0,
                "relTypeCount": 0,
            }
        label_rows = await self._read(
            "CALL db.labels() YIELD label "
            "CALL { WITH label MATCH (x) WHERE label IN labels(x) RETURN count(x) AS c } "
            "RETURN label, c ORDER BY c DESC"
        )
        structural = await self._structural_rel_types()
        rel_rows = await self._read(
            "MATCH ()-[r]->() WITH type(r) AS t WHERE t IN $types "
            "RETURN t, count(*) AS c ORDER BY c DESC",
            types=structural,
        )
        docs = await self._read_one(
            "MATCH (d:Document) RETURN count(d) AS total, "
            "sum(CASE WHEN d.latest THEN 1 ELSE 0 END) AS latest"
        )
        return GraphSummary(
            raw=dict(scalars),
            node_counts={r["label"]: int(r["c"]) for r in label_rows},
            relationship_counts={r["t"]: int(r["c"]) for r in rel_rows},
            total_nodes=int(scalars.get("nodeCount", 0) or 0),
            total_relationships=int(scalars.get("relCount", 0) or 0),
            total_relationship_types=int(scalars.get("relTypeCount", 0) or 0),
            documents=int((docs or {}).get("total", 0) or 0),
            latest_documents=int((docs or {}).get("latest", 0) or 0),
        )
