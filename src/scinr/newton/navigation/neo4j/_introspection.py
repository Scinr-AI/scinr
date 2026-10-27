"""navigation/neo4j/_introspection.py — Group G: catalogue & schema introspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.models import (
    CatalogGraph,
    CatalogModelRef,
    CatalogPresence,
    CatalogRelation,
    DocumentRef,
    GraphSummary,
    ModelClassStat,
    RoleStat,
    ThemeRef,
)
from scinr.newton.navigation.neo4j import _map
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.scope import Scope, ScopeKind, make_scope, resolve_selector

#: Labels that carry ``tenant_id`` (and how they store their provenance) — the
#: ones a scoped :meth:`get_graph_summary` counts.
_SCOPED_LABELS: tuple[tuple[str, str], ...] = (
    ("Document", "scalar"),
    ("StructureNode", "scalar"),
    ("InfoUnit", "scalar"),
    ("ModelDecision", "scalar"),
    ("ProposedModel", "scalar"),
    ("ProposedField", "scalar"),
    ("ExtractionResult", "scalar"),
    ("ModelInstance", "array"),
    ("LabeledEntity", "array"),
    ("Entity", "array"),
)
#: Global catalogue labels (no tenant): reported unfiltered.
_CATALOG_LABELS: tuple[str, ...] = ("CatalogModel", "ModelField", "EntityLabel", "Theme")


class _IntrospectionMixin(_Neo4jRuntime):
    async def _catalog_model_refs(
        self, include_fields: bool, where: str = "", params: dict[str, Any] | None = None
    ) -> list[CatalogModelRef]:
        """``:CatalogModel`` nodes (optionally filtered by *where* on ``cm``)."""
        match = "MATCH (cm:CatalogModel) " + (f"WHERE {where} " if where else "")
        if include_fields:
            rows = await self._read(
                match
                + "OPTIONAL MATCH (cm)-[hf:HAS_FIELD]->(f:ModelField) "
                "OPTIONAL MATCH (cm)-[:BELONGS_TO_THEME]->(t:Theme) "
                "RETURN cm, "
                "collect(DISTINCT {name: f.name, type: f.type, entity_label: f.entity_label, "
                "is_instance_key: f.is_instance_key, required: hf.required, description: hf.description}) AS fields, "
                "collect(DISTINCT t.name) AS themes ORDER BY cm.name",
                **(params or {}),
            )
            return [
                _map.catalog_model_ref(r["cm"], fields=r["fields"], themes=r["themes"])
                for r in rows
            ]
        rows = await self._read(match + "RETURN cm ORDER BY cm.name", **(params or {}))
        return [_map.catalog_model_ref(r["cm"]) for r in rows]

    async def _catalog_relations(self) -> list[CatalogRelation]:
        """The declared relationships between ``:CatalogModel`` / ``:EntityLabel`` nodes."""
        rows = await self._read(
            "MATCH (a)-[r]->(b) "
            "WHERE (a:CatalogModel OR a:EntityLabel) AND (b:CatalogModel OR b:EntityLabel) "
            "AND NOT type(r) IN ['HAS_FIELD', 'BELONGS_TO_THEME'] "
            "RETURN labels(a)[0] AS source_kind, coalesce(a.name, a.label) AS source, "
            "type(r) AS rel_type, properties(r) AS props, "
            "labels(b)[0] AS target_kind, coalesce(b.name, b.label) AS target"
        )
        return [_map.catalog_relation(r) for r in rows]

    def _present_in_scope(
        self,
        outer: str,
        label: str,
        key: str,
        inner: str,
        kind: ScopeKind,
        sc: Scope,
        params: dict[str, Any],
    ) -> str:
        """``EXISTS { MATCH (inner:label {key: outer}) WHERE <scope> }``.

        *label* and *key* are code constants, never user input.
        """
        clauses: list[str] = []
        self._scope_where(inner, clauses, params, sc, kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return f"EXISTS {{ MATCH ({inner}:{label} {{{key}: {outer}}}){where} }}"

    def _model_extracted_in_scope(self, sc: Scope, params: dict[str, Any]) -> str:
        """Predicate on ``cm``: a ``:ModelInstance`` or an ``:ExtractionResult`` in the scope.

        ``:ModelInstance`` stores its provenance as arrays, ``:ExtractionResult``
        as scalars (see :data:`_SCOPED_LABELS`); both share the ``$scope_*`` params.
        """
        return (
            "("
            + self._present_in_scope("cm.name", "ModelInstance", "model_class", "mi", "array", sc, params)
            + " OR "
            + self._present_in_scope("cm.name", "ExtractionResult", "model_class", "er", "scalar", sc, params)
            + ")"
        )

    def _model_annotated_in_scope(self, sc: Scope, params: dict[str, Any]) -> str:
        """Predicate on ``cm``: a ``:ModelDecision`` in the scope matched it, as primary or complementary."""
        clauses: list[str] = []
        self._scope_where("md", clauses, params, sc, "scalar")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return (
            f"(EXISTS {{ MATCH (md:ModelDecision)-[:MATCHED_MODEL]->(cm){where} }}"
            f" OR EXISTS {{ MATCH (md:ModelDecision)-[:HAS_COMPLEMENTARY_MATCH]->(:ComplementaryMatch)"
            f"-[:REFERS_TO_MODEL]->(cm){where} }})"
        )

    def _model_present_in_scope(
        self, sc: Scope, params: dict[str, Any], presence: CatalogPresence
    ) -> str:
        """Predicate on ``cm`` for *presence* (see :meth:`list_model_classes_present`)."""
        if presence == "model_instances":
            return self._model_extracted_in_scope(sc, params)
        if presence == "annotated":
            return self._model_annotated_in_scope(sc, params)
        if presence == "both":
            return (
                "("
                + self._model_extracted_in_scope(sc, params)
                + " OR "
                + self._model_annotated_in_scope(sc, params)
                + ")"
            )
        raise NavigationError(f"unknown presence: {presence!r}")

    async def list_catalog_models(
        self, *, include_fields: bool = False
    ) -> list[CatalogModelRef]:
        return await self._catalog_model_refs(include_fields)

    async def get_catalog_graph(
        self, *, include_fields: bool = True, include_relationships: bool = True
    ) -> CatalogGraph:
        models = await self.list_catalog_models(include_fields=include_fields)
        labels_rows = await self._read("MATCH (el:EntityLabel) RETURN el.label AS label ORDER BY label")
        entity_labels = [r["label"] for r in labels_rows if r["label"]]
        relationships = await self._catalog_relations() if include_relationships else []
        return CatalogGraph(
            raw={"models": len(models), "relationships": len(relationships)},
            models=models,
            entity_labels=entity_labels,
            relationships=relationships,
        )

    async def list_model_classes_present(
        self,
        *,
        model_classes: Sequence[str] | None = None,
        presence: CatalogPresence = "both",
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[str]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {}
        # Built before the empty-list shortcut so an unknown *presence* always raises.
        present = self._model_present_in_scope(sc, params, presence)
        if model_classes is not None and not model_classes:
            return []
        clauses: list[str] = []
        if model_classes is not None:
            clauses.append("cm.name IN $model_classes")
            params["model_classes"] = list(model_classes)
        clauses.append(present)
        rows = await self._read(
            f"MATCH (cm:CatalogModel) WHERE {' AND '.join(clauses)} "
            "RETURN cm.name AS name ORDER BY name",
            **params,
        )
        return [r["name"] for r in rows]

    async def get_catalog_graph_in_use(
        self,
        *,
        include_fields: bool = True,
        include_relationships: bool = True,
        presence: CatalogPresence = "both",
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> CatalogGraph:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        model_params: dict[str, Any] = {}
        models = await self._catalog_model_refs(
            include_fields,
            where=self._model_present_in_scope(sc, model_params, presence),
            params=model_params,
        )
        label_params: dict[str, Any] = {}
        label_exists = self._present_in_scope(
            "el.label", "LabeledEntity", "label", "le", "array", sc, label_params
        )
        labels_rows = await self._read(
            f"MATCH (el:EntityLabel) WHERE {label_exists} RETURN el.label AS label ORDER BY label",
            **label_params,
        )
        entity_labels = [r["label"] for r in labels_rows if r["label"]]
        relationships: list[CatalogRelation] = []
        if include_relationships:
            names = {m.name for m in models}
            labels = set(entity_labels)

            def kept(name: str, kind: str) -> bool:
                return name in names if kind == "CatalogModel" else name in labels

            relationships = [
                r
                for r in await self._catalog_relations()
                if kept(r.source, r.source_kind) and kept(r.target, r.target_kind)
            ]
        return CatalogGraph(
            raw={"models": len(models), "relationships": len(relationships), "presence": presence},
            models=models,
            entity_labels=entity_labels,
            relationships=relationships,
        )

    async def list_model_classes_in_use(
        self,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelClassStat]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        params: dict[str, Any] = {}
        mi_clauses: list[str] = []
        er_clauses: list[str] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            mi_clauses.append(
                f"EXISTS {{ MATCH (mi)<-[hr*1..{cd}]-(:ExtractionResult)<-[:HAS_EXTRACTION]-"
                "(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) "
                "WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }"
            )
            er_clauses.append(
                "EXISTS { MATCH (er)<-[:HAS_EXTRACTION]-(:StructureNode)"
                "<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) }"
            )
            params["doc_path"] = path
        self._scope_where("mi", mi_clauses, params, sc, "array")
        self._scope_where("er", er_clauses, params, sc, "scalar")
        mi_where = f"WHERE {' AND '.join(mi_clauses)} " if mi_clauses else ""
        er_where = f"WHERE {' AND '.join(er_clauses)} " if er_clauses else ""
        rows = await self._read(
            "CALL () { "
            f"MATCH (mi:ModelInstance) {mi_where}"
            "RETURN mi.model_class AS model_class, count(*) AS model_instances, 0 AS extraction_results "
            "UNION ALL "
            f"MATCH (er:ExtractionResult) {er_where}"
            "RETURN er.model_class AS model_class, 0 AS model_instances, count(*) AS extraction_results "
            "} "
            "WITH model_class, sum(model_instances) AS model_instances, "
            "sum(extraction_results) AS extraction_results "
            "WHERE model_class IS NOT NULL "
            "RETURN model_class, model_instances, extraction_results, "
            "model_instances + extraction_results AS count "
            "ORDER BY count DESC, model_class",
            **params,
        )
        return [_map.model_class_stat(r, kind="in_use") for r in rows]

    async def get_model_properties(
        self,
        model_class: str,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> dict[str, list[str]]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        # ``declared`` comes from the (global) catalogue; ``observed`` is
        # sampled only from the instances inside the scope.
        declared_rec = await self._read_one(
            "MATCH (:CatalogModel {name: $mc})-[:HAS_FIELD]->(f:ModelField) "
            "RETURN collect(DISTINCT f.name) AS declared",
            mc=model_class,
        )
        clauses: list[str] = []
        params: dict[str, Any] = {"mc": model_class}
        self._scope_where("mi", clauses, params, sc, "array")
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        observed_rec = await self._read_one(
            f"MATCH (mi:ModelInstance {{model_class: $mc}}) {where_sql}WITH mi LIMIT 500 "
            "UNWIND keys(mi) AS k WITH DISTINCT k WHERE NOT k IN ['uid', 'model_class'] "
            "RETURN collect(k) AS observed",
            **params,
        )
        return {
            "declared": sorted((declared_rec or {}).get("declared", []) or []),
            "observed": sorted((observed_rec or {}).get("observed", []) or []),
        }

    async def list_node_roles(
        self,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[RoleStat]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {}
        clauses: list[str] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            clauses.append(
                "EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $doc_path}) }"
            )
            params["doc_path"] = path
        self._scope_where("n", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rows = await self._read(
            f"MATCH (n:StructureNode) {where_sql}"
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

    async def get_graph_summary(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> GraphSummary:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if not sc.is_unfiltered:
            return await self._scoped_graph_summary(sc)
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

    async def _scoped_graph_summary(self, sc: Scope) -> GraphSummary:
        """Summary restricted to the scope.

        Per-label counts of the tenant-carrying labels are taken label by label
        (each is an index seek on ``tenant_id``). Catalogue nodes have no
        tenant: they are global and reported unfiltered under their own labels.
        """
        node_counts: dict[str, int] = {}
        rel_counts: dict[str, int] = {}
        for label, kind in _SCOPED_LABELS:
            clauses: list[str] = []
            params: dict[str, Any] = {}
            self._scope_where("x", clauses, params, sc, kind)
            where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
            rec = await self._read_one(
                f"MATCH (x:{label}) {where_sql}RETURN count(x) AS c", **params
            )
            count = int((rec or {}).get("c", 0) or 0)
            if not count:
                continue
            node_counts[label] = count
            for r in await self._read(
                f"MATCH (x:{label})-[r]->() {where_sql}"
                "RETURN type(r) AS t, count(*) AS c",
                **params,
            ):
                rel_counts[r["t"]] = rel_counts.get(r["t"], 0) + int(r["c"])
        for label in _CATALOG_LABELS:
            rec = await self._read_one(f"MATCH (x:{label}) RETURN count(x) AS c")
            count = int((rec or {}).get("c", 0) or 0)
            if count:
                node_counts[label] = count
        clauses = []
        params = {}
        self._scope_where("d", clauses, params, sc)
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        docs = await self._read_one(
            f"MATCH (d:Document) {where_sql}RETURN count(d) AS total, "
            "sum(CASE WHEN d.latest THEN 1 ELSE 0 END) AS latest",
            **params,
        )
        return GraphSummary(
            raw={"scope": {"tenants": sc.tenants, "user_ids": sc.user_ids, "job_ids": sc.job_ids}},
            node_counts=dict(sorted(node_counts.items(), key=lambda kv: -kv[1])),
            relationship_counts=dict(sorted(rel_counts.items(), key=lambda kv: -kv[1])),
            total_nodes=sum(node_counts.values()),
            total_relationships=sum(rel_counts.values()),
            total_relationship_types=len(rel_counts),
            documents=int((docs or {}).get("total", 0) or 0),
            latest_documents=int((docs or {}).get("latest", 0) or 0),
        )
