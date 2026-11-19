"""navigation/sql/_annotation.py — Group D: annotation (ModelDecision) (SQL)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa

from scinr.newton.navigation import _map
from scinr.newton.navigation.models import (
    AnnotationCoverage,
    DocumentModelProfile,
    DocumentRef,
    ModelDecisionRef,
    ModelDecisionWithNode,
    ProposedModelRef,
    StructureNodeRef,
)
from scinr.newton.navigation.scope import make_scope, resolve_selector
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import prop_equals
from scinr.newton.navigation.sql._walk import reached, walk
from scinr.newton.sql import tables as T

_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")


class _AnnotationMixin(_SqlRuntime):
    async def _decision_extras(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        """``matched_model`` / ``complementary_models`` / ``supplementary_fields`` per decision.

        The pattern comprehensions of the Cypher, in relationship creation order.
        """
        extras: dict[str, dict[str, Any]] = {
            u: {"matched_model": None, "complementary_models": [], "supplementary_fields": []}
            for u in uids
        }
        if not uids:
            return extras
        e1, e2, sf = T.edges.alias("e1"), T.edges.alias("e2"), T.supplementary_fields
        matched = await self._read(
            sa.select(e1.c.src_key, e1.c.dst_key)
            .where(
                e1.c.type == "MATCHED_MODEL",
                e1.c.src_label == "ModelDecision",
                e1.c.src_key.in_(uids),
                e1.c.dst_label == "CatalogModel",
            )
            .order_by(e1.c.id)
        )
        for row in matched:
            if extras[row["src_key"]]["matched_model"] is None:
                extras[row["src_key"]]["matched_model"] = row["dst_key"]
        complementary = await self._read(
            sa.select(e1.c.src_key, e2.c.dst_key)
            .select_from(
                e1.join(
                    e2,
                    sa.and_(
                        e2.c.type == "REFERS_TO_MODEL",
                        e2.c.src_label == "ComplementaryMatch",
                        e2.c.src_key == e1.c.dst_key,
                        e2.c.dst_label == "CatalogModel",
                    ),
                )
            )
            .where(
                e1.c.type == "HAS_COMPLEMENTARY_MATCH",
                e1.c.src_label == "ModelDecision",
                e1.c.src_key.in_(uids),
                e1.c.dst_label == "ComplementaryMatch",
            )
            .order_by(e1.c.id, e2.c.id)
        )
        for row in complementary:
            extras[row["src_key"]]["complementary_models"].append(row["dst_key"])
        field_name = self._dl.json_text(sf.c.props, "field_name")
        supplementary = await self._read(
            sa.select(e1.c.src_key, field_name.label("field_name"))
            .select_from(
                e1.join(
                    sf,
                    sa.and_(e1.c.dst_label == "SupplementaryField", sf.c.uid == e1.c.dst_key),
                )
            )
            .where(
                e1.c.type == "HAS_SUPPLEMENTARY_FIELD",
                e1.c.src_label == "ModelDecision",
                e1.c.src_key.in_(uids),
            )
            .order_by(e1.c.id)
        )
        for row in supplementary:
            if row["field_name"] is not None:
                extras[row["src_key"]]["supplementary_fields"].append(row["field_name"])
        return extras

    async def get_model_decision(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> ModelDecisionRef | None:
        """The model decision of a structure node."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        md, e = T.model_decisions, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")
        linked = sa.select(e.c.dst_key).where(
            e.c.type == "HAS_MODEL_DECISION",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(anchor.c.key)),
            e.c.dst_label == "ModelDecision",
        )
        rec = await self._read_one(
            sa.select(md)
            .where(md.c.uid.in_(linked), *self._scope(md, sc, tenant=False))
            .order_by(self._dl.order_text(md.c.uid))
            .limit(1)
        )
        if not rec:
            return None
        extras = (await self._decision_extras([rec["uid"]]))[rec["uid"]]
        return _map.model_decision_ref(self._node("model_decisions", rec), **extras)

    async def get_document_model_decisions(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        matched_only: bool | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelDecisionWithNode]:
        """The model decisions of the structure nodes below a document, in document order."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        md, sn, e = T.model_decisions, T.structure_nodes, T.edges
        conditions: list[sa.ColumnElement[bool]] = []
        if matched_only is True:
            conditions.append(md.c.matched_model_class.is_not(None))
        elif matched_only is False:
            conditions.append(md.c.matched_model_class.is_(None))
        conditions.extend(self._scope(md, sc, tenant=False))
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_STRUCTURE_EDGES)
        below = reached(cte, "StructureNode")
        stmt = (
            sa.select(md, sn.c.id.label("_node_id"), sn.c.title.label("_node_title"))
            .select_from(
                sn.join(below, below.c.key == sn.c.id)
                .join(
                    e,
                    sa.and_(
                        e.c.type == "HAS_MODEL_DECISION",
                        e.c.src_label == "StructureNode",
                        e.c.src_key == sn.c.id,
                        e.c.dst_label == "ModelDecision",
                    ),
                )
                .join(md, md.c.uid == e.c.dst_key)
            )
            .where(*conditions)
            .order_by(
                self._dl.order_value(sn.c.appearance_order),
                self._dl.order_text(sn.c.id),
                self._dl.order_text(md.c.uid),
            )
        )
        rows = await self._read(stmt)
        extras = await self._decision_extras([r["uid"] for r in rows])
        out: list[ModelDecisionWithNode] = []
        for r in rows:
            base = _map.model_decision_ref(self._node("model_decisions", r), **extras[r["uid"]])
            out.append(
                ModelDecisionWithNode(
                    **base.model_dump(), node_id=r["_node_id"], node_title=r["_node_title"]
                )
            )
        return out

    def _annotated_nodes(self, path: str, version: int | None, scope: Any, depth: int) -> Any:
        """``(doc)-[:HAS_STRUCTURE|HAS_CHILD*1..d]->(n)`` with its decision and proposal, as
        left joins: one row per (node, decision, proposed model) like the Cypher's
        ``OPTIONAL MATCH`` chain. Returns ``(select_from, md, pm, doc)``."""
        sn, doc, e1, e2 = T.structure_nodes, T.documents, T.edges.alias("hd"), T.edges.alias("hp")
        md, pm = T.model_decisions.alias("md"), T.proposed_models.alias("pm")
        cte = walk(
            self._doc_start(path, version, scope, "scalar"), max_hops=depth, types=_STRUCTURE_EDGES
        )
        below = reached(cte, "StructureNode")
        source = (
            sn.join(below, below.c.key == sn.c.id)
            .join(doc, doc.c._key == below.c.origin)
            .outerjoin(
                e1,
                sa.and_(
                    e1.c.type == "HAS_MODEL_DECISION",
                    e1.c.src_label == "StructureNode",
                    e1.c.src_key == sn.c.id,
                    e1.c.dst_label == "ModelDecision",
                ),
            )
            .outerjoin(md, md.c.uid == e1.c.dst_key)
            .outerjoin(
                e2,
                sa.and_(
                    e2.c.type == "HAS_PROPOSED_MODEL",
                    e2.c.src_label == "ModelDecision",
                    e2.c.src_key == md.c.uid,
                    e2.c.dst_label == "ProposedModel",
                ),
            )
            .outerjoin(pm, pm.c.uid == e2.c.dst_key)
        )
        return source, md, pm, doc

    async def get_document_model_profile(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentModelProfile | None:
        """How the structure nodes of a document were annotated with catalogue models."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        sc = await self._single_document_scope(path, version, sc)
        d = self._resolve_depth(depth)
        # Single document: the anchor carries the whole scope, everything below it belongs
        # to it.
        source, md, pm, doc = self._annotated_nodes(path, version, sc, d)
        sn = T.structure_nodes
        core = await self._read_one(
            sa.select(
                doc.c.version.label("version"),
                sa.func.count(sn.c.id).label("total"),
                sa.func.sum(sa.case((md.c.uid.is_(None), 1), else_=0)).label("unannotated"),
            )
            .select_from(source)
            .group_by(doc.c.version)
        )
        if core is None:
            return None
        proposed_names = await self._read(
            sa.select(self._dl.json_text(pm.c.props, "schema_name").label("name"))
            .select_from(source)
            .where(pm.c.uid.is_not(None))
            .distinct()
        )
        # Matched and complementary model counts, walking from the same structure nodes.
        cte = walk(
            self._doc_start(path, version, sc, "scalar"),
            max_hops=d,
            types=_STRUCTURE_EDGES,
            name="profile",
        )
        below = reached(cte, "StructureNode", name="profile_r")
        e_dec, e_mm = T.edges.alias("pd"), T.edges.alias("pm_")
        count = sa.func.count().label("count")
        matched = await self._read(
            sa.select(e_mm.c.dst_key.label("model_class"), count)
            .select_from(
                e_dec.join(
                    e_mm,
                    sa.and_(
                        e_mm.c.type == "MATCHED_MODEL",
                        e_mm.c.src_label == "ModelDecision",
                        e_mm.c.src_key == e_dec.c.dst_key,
                        e_mm.c.dst_label == "CatalogModel",
                    ),
                )
            )
            .where(
                e_dec.c.type == "HAS_MODEL_DECISION",
                e_dec.c.src_label == "StructureNode",
                e_dec.c.src_key.in_(sa.select(below.c.key)),
                e_dec.c.dst_label == "ModelDecision",
            )
            .group_by(e_mm.c.dst_key)
            .order_by(count.desc(), self._dl.order_text(e_mm.c.dst_key))
        )
        e_cm, e_rf = T.edges.alias("cm_"), T.edges.alias("rf_")
        complementary = await self._read(
            sa.select(e_rf.c.dst_key.label("model_class"), count)
            .select_from(
                e_dec.join(
                    e_cm,
                    sa.and_(
                        e_cm.c.type == "HAS_COMPLEMENTARY_MATCH",
                        e_cm.c.src_label == "ModelDecision",
                        e_cm.c.src_key == e_dec.c.dst_key,
                    ),
                ).join(
                    e_rf,
                    sa.and_(
                        e_rf.c.type == "REFERS_TO_MODEL",
                        e_rf.c.src_key == e_cm.c.dst_key,
                        e_rf.c.dst_label == "CatalogModel",
                    ),
                )
            )
            .where(
                e_dec.c.type == "HAS_MODEL_DECISION",
                e_dec.c.src_label == "StructureNode",
                e_dec.c.src_key.in_(sa.select(below.c.key)),
                e_dec.c.dst_label == "ModelDecision",
            )
            .group_by(e_rf.c.dst_key)
            .order_by(count.desc(), self._dl.order_text(e_rf.c.dst_key))
        )
        raw = {
            "version": core["version"],
            "total": core["total"],
            "unannotated": core["unannotated"],
            "proposed": sorted(r["name"] for r in proposed_names if r["name"]),
        }
        return DocumentModelProfile(
            raw=raw,
            path=path,
            version=int(core["version"]),
            matched=[_map.model_class_stat(r, kind="matched") for r in matched if r["model_class"]],
            complementary=[
                _map.model_class_stat(r, kind="complementary")
                for r in complementary
                if r["model_class"]
            ],
            proposed=list(raw["proposed"]),
            unannotated_nodes=int(core.get("unannotated", 0) or 0),
        )

    async def get_nodes_by_annotated_model(
        self,
        model_class: str,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The structure nodes whose decision matched *model_class* as primary model."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn, e1, e2 = T.structure_nodes, T.edges.alias("hd"), T.edges.alias("mm")
        conditions: list[sa.ColumnElement[bool]] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(self._in_document_path(path, sn.c.id))
        conditions.extend(self._scope(sn, sc))
        annotated = (
            sa.select(e1.c.src_key)
            .select_from(
                e1.join(
                    e2,
                    sa.and_(
                        e2.c.type == "MATCHED_MODEL",
                        e2.c.src_label == "ModelDecision",
                        e2.c.src_key == e1.c.dst_key,
                        e2.c.dst_label == "CatalogModel",
                        e2.c.dst_key == model_class,
                    ),
                )
            )
            .where(
                e1.c.type == "HAS_MODEL_DECISION",
                e1.c.src_label == "StructureNode",
                e1.c.dst_label == "ModelDecision",
            )
        )
        stmt = (
            sa.select(sn)
            .where(sn.c.id.in_(annotated), *conditions)
            .order_by(self._dl.order_text(sn.c.id))
        )
        return [
            _map.structure_node_ref(self._node("structure_nodes", r))
            for r in await self._read(stmt)
        ]

    async def get_unannotated_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The structure nodes of a document that have no model decision."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        sn, e = T.structure_nodes, T.edges
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_STRUCTURE_EDGES)
        below = reached(cte, "StructureNode")
        has_decision = sa.exists().where(
            e.c.type == "HAS_MODEL_DECISION",
            e.c.src_label == "StructureNode",
            e.c.src_key == sn.c.id,
        )
        stmt = (
            sa.select(sn)
            .where(
                sn.c.id.in_(sa.select(below.c.key)),
                ~has_decision,
                *self._scope(sn, sc, tenant=False),
            )
            .order_by(self._dl.order_value(sn.c.appearance_order), self._dl.order_text(sn.c.id))
        )
        return [
            _map.structure_node_ref(self._node("structure_nodes", r))
            for r in await self._read(stmt)
        ]

    async def get_proposed_models(
        self,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ProposedModelRef]:
        """The models proposed by decisions (one row per node that proposed them)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn, pm = T.structure_nodes, T.proposed_models
        e1, e2 = T.edges.alias("hd"), T.edges.alias("hp")
        conditions: list[sa.ColumnElement[bool]] = []
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(self._in_document_path(path, sn.c.id))
        conditions.extend(self._scope(pm, sc))
        stmt = (
            sa.select(pm, sn.c.id.label("_node_id"))
            .select_from(
                sn.join(
                    e1,
                    sa.and_(
                        e1.c.type == "HAS_MODEL_DECISION",
                        e1.c.src_label == "StructureNode",
                        e1.c.src_key == sn.c.id,
                        e1.c.dst_label == "ModelDecision",
                    ),
                )
                .join(
                    e2,
                    sa.and_(
                        e2.c.type == "HAS_PROPOSED_MODEL",
                        e2.c.src_label == "ModelDecision",
                        e2.c.src_key == e1.c.dst_key,
                        e2.c.dst_label == "ProposedModel",
                    ),
                )
                .join(pm, pm.c.uid == e2.c.dst_key)
            )
            .where(*conditions)
            .order_by(self._dl.order_text(pm.c.uid), self._dl.order_text(sn.c.id))
        )
        rows = await self._read(stmt)
        fields = await self._proposed_fields([r["uid"] for r in rows])
        return [
            _map.proposed_model_ref(
                self._node("proposed_models", r),
                fields=fields.get(r["uid"], []),
                node_id=r["_node_id"],
            )
            for r in rows
        ]

    async def _proposed_fields(self, uids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """The fields (``f { .* }``) of each proposed model, in creation order."""
        if not uids:
            return {}
        pf, e = T.proposed_fields, T.edges
        stmt = (
            sa.select(pf, e.c.src_key.label("_model"))
            .select_from(
                pf.join(e, sa.and_(e.c.dst_label == "ProposedField", e.c.dst_key == pf.c.uid))
            )
            .where(
                e.c.type == "HAS_PROPOSED_FIELD",
                e.c.src_label == "ProposedModel",
                e.c.src_key.in_(sorted(set(uids))),
            )
            .order_by(e.c.id)
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for row in await self._read(stmt):
            node = self._node("proposed_fields", row)
            node.pop("_labels", None)
            out.setdefault(row["_model"], []).append(node)
        return out

    async def get_annotation_coverage(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> AnnotationCoverage | None:
        """How many structure nodes of a document have a model decision."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        sc = await self._single_document_scope(path, version, sc)
        d = self._resolve_depth(depth)
        # Single document: the anchor carries the whole scope, so both sides of the ratio
        # (annotated / total) are computed over the same nodes.
        source, md, _pm, doc = self._annotated_nodes(path, version, sc, d)
        sn = T.structure_nodes
        rec = await self._read_one(
            sa.select(
                doc.c.version.label("version"),
                sa.func.count(sn.c.id).label("total"),
                sa.func.count(md.c.uid).label("annotated"),
                sa.func.sum(sa.case((md.c.uid.is_(None), 1), else_=0)).label("unannotated"),
                sa.func.sum(sa.case((md.c.matched_model_class.is_not(None), 1), else_=0)).label(
                    "matched"
                ),
                sa.func.sum(
                    sa.case((prop_equals(md, "propose_new_model", True, self._dl), 1), else_=0)
                ).label("proposed"),
            )
            .select_from(source)
            .group_by(doc.c.version)
        )
        if rec is None:
            return None
        total = int(rec["total"] or 0)
        annotated = int(rec["annotated"] or 0)
        return AnnotationCoverage(
            raw=dict(rec),
            path=path,
            version=int(rec["version"]),
            total_nodes=total,
            annotated=annotated,
            unannotated=int(rec["unannotated"] or 0),
            matched=int(rec["matched"] or 0),
            proposed=int(rec["proposed"] or 0),
            ratio=(annotated / total) if total else 0.0,
        )


__all__ = ["_AnnotationMixin"]
