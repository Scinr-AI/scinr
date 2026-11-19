"""navigation/sql/_introspection.py — Group G: catalogue & schema introspection (SQL)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa

from scinr.newton.exceptions import NavigationError
from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.navigation import _map
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
from scinr.newton.navigation.scope import Scope, make_scope, resolve_selector
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.sql import tables as T
from scinr.newton.sql.names import LABEL_TABLE

#: Labels that carry ``tenant_id`` (and how they store their provenance) — the ones a scoped
#: :meth:`get_graph_summary` counts.
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

#: Relationship types of the catalogue that ``_catalog_relations`` does not report.
_CATALOG_STRUCTURE_TYPES = ("HAS_FIELD", "BELONGS_TO_THEME")
#: The only property keys of the observed-properties sample that are not reported.
_SAMPLE_SIZE = 500

_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")


class _IntrospectionMixin(_SqlRuntime):
    async def _catalog_model_refs(
        self, include_fields: bool, conditions: Sequence[Any] = ()
    ) -> list[CatalogModelRef]:
        """``catalog_models`` rows (optionally filtered by *conditions* on that table)."""
        cm = T.catalog_models
        rows = await self._read(
            sa.select(cm).where(*conditions).order_by(self._dl.order_text(cm.c.name))
        )
        if not include_fields:
            return [_map.catalog_model_ref(self._node("catalog_models", r)) for r in rows]
        names = [r["name"] for r in rows]
        fields = await self._fields_by_model(names)
        themes = await self._themes_by_model(names)
        return [
            _map.catalog_model_ref(
                self._node("catalog_models", r),
                fields=fields.get(r["name"], []),
                themes=themes.get(r["name"], []),
            )
            for r in rows
        ]

    async def _fields_by_model(self, names: list[str]) -> dict[str, list[dict[str, Any]]]:
        """The ``HAS_FIELD`` fields of each model: the field's properties plus the ``required``
        and ``description`` of the relationship, in creation order."""
        if not names:
            return {}
        link, mf = T.catalog_edges, T.model_fields
        stmt = (
            sa.select(mf, link.c.src_key.label("_model"), link.c.props.label("_edge"))
            .select_from(link.join(mf, mf.c._key == link.c.dst_key))
            .where(
                link.c.type == "HAS_FIELD",
                link.c.src_label == "CatalogModel",
                link.c.src_key.in_(names),
                link.c.dst_label == "ModelField",
            )
            .order_by(link.c.id)
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for row in await self._read(stmt):
            field = self._node("model_fields", row)
            edge = row["_edge"] or {}
            out.setdefault(row["_model"], []).append(
                {
                    "name": field.get("name"),
                    "type": field.get("type"),
                    "entity_label": field.get("entity_label"),
                    "is_instance_key": field.get("is_instance_key"),
                    "required": edge.get("required"),
                    "description": edge.get("description"),
                }
            )
        return out

    async def _themes_by_model(self, names: list[str]) -> dict[str, list[str]]:
        if not names:
            return {}
        link, th = T.catalog_edges, T.themes
        name = self._dl.json_text(th.c.props, "name")
        stmt = (
            sa.select(link.c.src_key.label("_model"), name.label("name"))
            .select_from(link.join(th, th.c.path == link.c.dst_key))
            .where(
                link.c.type == "BELONGS_TO_THEME",
                link.c.src_label == "CatalogModel",
                link.c.src_key.in_(names),
                link.c.dst_label == "Theme",
            )
            .order_by(link.c.id)
        )
        out: dict[str, list[str]] = {}
        for row in await self._read(stmt):
            themes = out.setdefault(row["_model"], [])
            if row["name"] is not None and row["name"] not in themes:
                themes.append(row["name"])
        return out

    async def _catalog_relations(self) -> list[CatalogRelation]:
        """The declared relationships between ``CatalogModel`` / ``EntityLabel`` nodes."""
        link = T.catalog_edges
        kinds = ("CatalogModel", "EntityLabel")
        rows = await self._read(
            sa.select(link)
            .where(
                link.c.src_label.in_(kinds),
                link.c.dst_label.in_(kinds),
                link.c.type.not_in(_CATALOG_STRUCTURE_TYPES),
            )
            .order_by(link.c.id)
        )
        return [
            _map.catalog_relation(
                {
                    "source_kind": r["src_label"],
                    "source": r["src_key"],
                    "rel_type": r["type"],
                    "props": r["props"] or {},
                    "target_kind": r["dst_label"],
                    "target": r["dst_key"],
                }
            )
            for r in rows
        ]

    def _present_in_scope(
        self, table: Any, column: Any, outer: Any, kind: Any, sc: Scope
    ) -> sa.ColumnElement[bool]:
        """``EXISTS { MATCH (inner {<column>: outer}) WHERE <scope> }`` (correlated)."""
        return sa.exists().where(column == outer, *self._scope(table, sc, kind))

    def _model_extracted_in_scope(self, sc: Scope) -> sa.ColumnElement[bool]:
        """Predicate on ``catalog_models``: a model instance or an extraction result of the
        scope has the model class (instances keep their provenance as arrays, extraction
        results as scalars)."""
        cm = T.catalog_models
        mi, er = T.model_instances, T.extraction_results
        return sa.or_(
            self._present_in_scope(mi, mi.c.model_class, cm.c.name, "array", sc),
            self._present_in_scope(er, er.c.model_class, cm.c.name, "scalar", sc),
        )

    def _model_annotated_in_scope(self, sc: Scope) -> sa.ColumnElement[bool]:
        """Predicate on ``catalog_models``: a decision of the scope matched it, as primary or
        complementary model."""
        cm = T.catalog_models
        md = T.model_decisions
        matched, has_match, refers = (T.edges.alias(n) for n in ("mm", "hc", "rt"))
        primary = sa.exists().where(
            matched.c.type == "MATCHED_MODEL",
            matched.c.src_label == "ModelDecision",
            matched.c.src_key == md.c.uid,
            matched.c.dst_label == "CatalogModel",
            matched.c.dst_key == cm.c.name,
            *self._scope(md, sc, "scalar"),
        )
        complementary = sa.exists().where(
            has_match.c.type == "HAS_COMPLEMENTARY_MATCH",
            has_match.c.src_label == "ModelDecision",
            has_match.c.src_key == md.c.uid,
            has_match.c.dst_label == "ComplementaryMatch",
            refers.c.type == "REFERS_TO_MODEL",
            refers.c.src_label == "ComplementaryMatch",
            refers.c.src_key == has_match.c.dst_key,
            refers.c.dst_label == "CatalogModel",
            refers.c.dst_key == cm.c.name,
            *self._scope(md, sc, "scalar"),
        )
        return sa.or_(primary, complementary)

    def _model_present_in_scope(
        self, sc: Scope, presence: CatalogPresence
    ) -> sa.ColumnElement[bool]:
        """Predicate on ``catalog_models`` for *presence* (see ``list_model_classes_present``)."""
        if presence == "model_instances":
            return self._model_extracted_in_scope(sc)
        if presence == "annotated":
            return self._model_annotated_in_scope(sc)
        if presence == "both":
            return sa.or_(self._model_extracted_in_scope(sc), self._model_annotated_in_scope(sc))
        raise NavigationError(f"unknown presence: {presence!r}")

    async def _entity_labels(self, conditions: Sequence[Any] = ()) -> list[str]:
        el = T.entity_labels
        rows = await self._read(
            sa.select(el.c.label).where(*conditions).order_by(self._dl.order_text(el.c.label))
        )
        return [r["label"] for r in rows if r["label"]]

    async def list_catalog_models(self, *, include_fields: bool = False) -> list[CatalogModelRef]:
        """The catalogue models, by name."""
        return await self._catalog_model_refs(include_fields)

    async def get_catalog_graph(
        self, *, include_fields: bool = True, include_relationships: bool = True
    ) -> CatalogGraph:
        """The whole catalogue: models, entity labels and declared relationships."""
        models = await self.list_catalog_models(include_fields=include_fields)
        entity_labels = await self._entity_labels()
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
        """The catalogue models that have instances and/or decisions in the scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        # Built before the empty-list shortcut so an unknown *presence* always raises.
        present = self._model_present_in_scope(sc, presence)
        if model_classes is not None and not model_classes:
            return []
        cm = T.catalog_models
        conditions: list[sa.ColumnElement[bool]] = []
        if model_classes is not None:
            conditions.append(cm.c.name.in_(list(model_classes)))
        conditions.append(present)
        rows = await self._read(
            sa.select(cm.c.name).where(*conditions).order_by(self._dl.order_text(cm.c.name))
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
        """The part of the catalogue that is present in the scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        models = await self._catalog_model_refs(
            include_fields, [self._model_present_in_scope(sc, presence)]
        )
        el, le = T.entity_labels, T.labeled_entities
        entity_labels = await self._entity_labels(
            [self._present_in_scope(le, le.c.label, el.c.label, "array", sc)]
        )
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
        """How many instances and extraction results each model class has in the scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(None)
        mi, er, e = T.model_instances, T.extraction_results, T.edges
        mi_conditions: list[sa.ColumnElement[bool]] = []
        er_conditions: list[sa.ColumnElement[bool]] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            mi_conditions.append(mi.c.uid.in_(self._instances_of_path(path, cd)))
            er_conditions.append(
                er.c.uid.in_(
                    sa.select(e.c.dst_key).where(
                        e.c.type == "HAS_EXTRACTION",
                        e.c.src_label == "StructureNode",
                        e.c.dst_label == "ExtractionResult",
                        self._in_document_path(path, e.c.src_key),
                    )
                )
            )
        mi_conditions.extend(self._scope(mi, sc, "array"))
        er_conditions.extend(self._scope(er, sc, "scalar"))
        counts: dict[str, list[int]] = {}
        for table, conditions, slot in ((mi, mi_conditions, 0), (er, er_conditions, 1)):
            rows = await self._read(
                sa.select(table.c.model_class.label("model_class"), sa.func.count().label("c"))
                .where(*conditions)
                .group_by(table.c.model_class)
            )
            for row in rows:
                if row["model_class"] is not None:
                    counts.setdefault(row["model_class"], [0, 0])[slot] += int(row["c"])
        stats = [
            {
                "model_class": name,
                "model_instances": n[0],
                "extraction_results": n[1],
                "count": n[0] + n[1],
            }
            for name, n in counts.items()
        ]
        stats.sort(key=lambda s: (-s["count"], s["model_class"]))
        return [_map.model_class_stat(s, kind="in_use") for s in stats]

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
        """Declared fields of a model and the properties observed on its instances.

        ``declared`` comes from the (global) catalogue; ``observed`` is sampled only from
        (up to 500 of) the instances inside the scope.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        link, mf, mi = T.catalog_edges, T.model_fields, T.model_instances
        declared_rows = await self._read(
            sa.select(mf.c.name)
            .select_from(link.join(mf, mf.c._key == link.c.dst_key))
            .where(
                link.c.type == "HAS_FIELD",
                link.c.src_label == "CatalogModel",
                link.c.src_key == model_class,
                link.c.dst_label == "ModelField",
            )
        )
        sample = await self._read(
            sa.select(mi)
            .where(mi.c.model_class == model_class, *self._scope(mi, sc, "array"))
            .order_by(self._dl.order_text(mi.c.uid))
            .limit(_SAMPLE_SIZE)
        )
        observed: set[str] = set()
        for row in sample:
            observed.update(k for k in self._node("model_instances", row) if k != "_labels")
        observed -= {"uid", "model_class"}
        return {
            "declared": sorted({r["name"] for r in declared_rows if r["name"] is not None}),
            "observed": sorted(observed),
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
        """How many structure nodes each role has, most frequent first."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn = T.structure_nodes
        conditions: list[sa.ColumnElement[bool]] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(self._in_document_path(path, sn.c.id))
        conditions.extend(self._scope(sn, sc))
        count = sa.func.count().label("count")
        rows = await self._read(
            sa.select(sn.c.role.label("role"), count)
            .where(*conditions)
            .group_by(sn.c.role)
            .order_by(count.desc(), self._dl.order_text(sn.c.role))
        )
        return [_map.role_stat(r) for r in rows if r["role"]]

    async def list_themes(self) -> list[ThemeRef]:
        """The catalogue themes (or, without any, the distinct themes of the nodes)."""
        th, sn = T.themes, T.structure_nodes
        rows = await self._read(sa.select(th).order_by(self._dl.order_text(th.c.path)))
        if rows:
            return [_map.theme_ref(self._node("themes", r)) for r in rows]
        # Fallback: distinct StructureNode.theme values.
        rows = await self._read(
            sa.select(sn.c.theme.label("name"))
            .where(sn.c.theme.is_not(None))
            .group_by(sn.c.theme)
            .order_by(self._dl.order_text(sn.c.theme))
        )
        return [ThemeRef(raw=r, name=r["name"]) for r in rows if r["name"]]

    async def _structural_rel_types(self) -> list[str]:
        """Relationship types that are not only ``Entity -> Entity`` (triple predicates)."""
        links = T.all_edges
        other_pair = sa.case(
            (sa.and_(links.c.src_label == "Entity", links.c.dst_label == "Entity"), 0), else_=1
        )
        rows = await self._read(
            sa.select(links.c.type).group_by(links.c.type).having(sa.func.sum(other_pair) > 0)
        )
        return sorted(r["type"] for r in rows)

    async def list_relationship_types(self, *, structural_only: bool = True) -> list[str]:
        """The relationship types of the graph (by default without triple predicates)."""
        if structural_only:
            return await self._structural_rel_types()
        links = T.all_edges
        rows = await self._read(
            sa.select(links.c.type)
            .group_by(links.c.type)
            .order_by(self._dl.order_text(links.c.type))
        )
        return [r["type"] for r in rows]

    async def _table_counts(self) -> dict[str, int]:
        """Row count of the node table of each primary label."""
        parts = [
            sa.select(
                sa.literal(label, sa.Text).label("label"), sa.func.count().label("c")
            ).select_from(T.metadata.tables[table])
            for label, table in LABEL_TABLE.items()
        ]
        rows = await self._read(sa.union_all(*parts))
        return {r["label"]: int(r["c"]) for r in rows}

    async def _role_label_counts(self) -> dict[str, int]:
        """Node count of each role label (``Section``, ``Table``…) of the structure nodes."""
        sn = T.structure_nodes
        rows = await self._read(
            sa.select(sn.c.role, sa.func.count().label("c")).group_by(sn.c.role)
        )
        counts: dict[str, int] = {}
        for row in rows:
            label = ROLE_TO_LABEL.get(row["role"] or "")
            if label:
                counts[label] = counts.get(label, 0) + int(row["c"])
        return counts

    async def list_node_labels(self) -> list[str]:
        """The labels in use: the primary label of each non-empty table and the role labels."""
        labels = {k for k, v in (await self._table_counts()).items() if v}
        labels.update(await self._role_label_counts())
        return sorted(labels)

    async def get_graph_summary(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> GraphSummary:
        """Node and relationship counts of the graph (or of the scope)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if not sc.is_unfiltered:
            return await self._scoped_graph_summary(sc)
        links, doc = T.all_edges, T.documents
        node_counts = {k: v for k, v in (await self._table_counts()).items() if v}
        node_counts.update(await self._role_label_counts())
        total_nodes = sum(v for k, v in (await self._table_counts()).items())
        edge_rows = await self._read(
            sa.select(links.c.type, sa.func.count().label("c")).group_by(links.c.type)
        )
        structural = set(await self._structural_rel_types())
        rel_counts = {r["type"]: int(r["c"]) for r in edge_rows if r["type"] in structural}
        documents = await self._read_one(
            sa.select(
                sa.func.count().label("total"),
                sa.func.sum(sa.case((doc.c.latest == sa.true(), 1), else_=0)).label("latest"),
            )
        )
        scalars = {
            "nodeCount": total_nodes,
            "relCount": sum(int(r["c"]) for r in edge_rows),
            "labelCount": len(node_counts),
            "relTypeCount": len(edge_rows),
        }
        return GraphSummary(
            raw=dict(scalars),
            node_counts=dict(sorted(node_counts.items(), key=lambda kv: -kv[1])),
            relationship_counts=dict(sorted(rel_counts.items(), key=lambda kv: -kv[1])),
            total_nodes=scalars["nodeCount"],
            total_relationships=scalars["relCount"],
            total_relationship_types=scalars["relTypeCount"],
            documents=int((documents or {}).get("total", 0) or 0),
            latest_documents=int((documents or {}).get("latest", 0) or 0),
        )

    async def _scoped_graph_summary(self, sc: Scope) -> GraphSummary:
        """Summary restricted to the scope.

        Per-label counts of the tenant-carrying labels are taken label by label. Catalogue
        nodes have no tenant: they are global and reported unfiltered under their own labels.
        """
        node_counts: dict[str, int] = {}
        rel_counts: dict[str, int] = {}
        e = T.edges
        for label, kind in _SCOPED_LABELS:
            table = T.metadata.tables[LABEL_TABLE[label]]
            from scinr.newton.sql.names import KEY_COLUMN

            key = table.c[KEY_COLUMN[label]]
            scoped = self._scope(table, sc, kind)  # type: ignore[arg-type]
            rec = await self._read_one(sa.select(sa.func.count().label("c")).where(*scoped))
            count = int((rec or {}).get("c", 0) or 0)
            if not count:
                continue
            node_counts[label] = count
            rows = await self._read(
                sa.select(e.c.type, sa.func.count().label("c"))
                .where(e.c.src_label == label, e.c.src_key.in_(sa.select(key).where(*scoped)))
                .group_by(e.c.type)
            )
            for r in rows:
                rel_counts[r["type"]] = rel_counts.get(r["type"], 0) + int(r["c"])
        for label in _CATALOG_LABELS:
            rec = await self._read_one(
                sa.select(sa.func.count().label("c")).select_from(
                    T.metadata.tables[LABEL_TABLE[label]]
                )
            )
            count = int((rec or {}).get("c", 0) or 0)
            if count:
                node_counts[label] = count
        doc = T.documents
        docs = await self._read_one(
            sa.select(
                sa.func.count().label("total"),
                sa.func.sum(sa.case((doc.c.latest == sa.true(), 1), else_=0)).label("latest"),
            ).where(*self._scope(doc, sc))
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


__all__ = ["_IntrospectionMixin"]
