"""navigation/sql/_instances.py — Group E: extraction & model instances (SQL)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import sqlalchemy as sa

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation import _map
from scinr.newton.navigation._safe import safe_ident
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
from scinr.newton.navigation.scope import make_scope, resolve_selector
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import translate_where
from scinr.newton.navigation.sql._walk import UNBOUNDED_HOPS, reached, split_path, walk
from scinr.newton.sql import tables as T

# Scope notes (navigation/scope.py). ExtractionResult carries scalar provenance;
# ModelInstance / LabeledEntity are merge-deduplicated, so their tenant is folded into the uid
# (scalar tenant_id) and job / user accumulate in ``job_ids`` / ``created_by_user_ids``
# (kind="array": any-of match). Anchors guard the tenant; the returned nodes are filtered by
# user/job. Only the "primary" getters (get_model_instance, ...) and the tree root apply the
# whole scope to the anchor. Not re-checking the tenant past the anchor is safe ONLY because
# no data relationship crosses tenants (public included).

_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")


class _ModelInstancesMixin(_SqlRuntime):
    async def _mi_refs(
        self, rows: list[dict[str, Any]], *, via: Mapping[str, Any] | None = None
    ) -> list[ModelInstanceRef]:
        """Refs of model-instance rows; *via* adds ``via_rel`` / ``direction`` to every row."""
        out: list[ModelInstanceRef] = []
        for r in rows:
            node = self._node("model_instances", r)
            out.append(
                _map.model_instance_ref(
                    node,
                    via_rel=r.get("_via_rel", (via or {}).get("via_rel")),
                    direction=(via or {}).get("direction"),
                    is_shell=await self._is_shell(node),
                )
            )
        return out

    def _mi_anchor(self, uid: str, scope: Any, kind: Any = "tenant_only") -> sa.Select:
        """``(label, key)`` of the anchor model instance."""
        mi = T.model_instances
        return self._node_start(
            "ModelInstance", mi.c.uid, [mi.c.uid == uid, *self._scope(mi, scope, kind)]
        )

    async def _er_extras(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        """``primary_model`` / ``complementary_models`` of each extraction result."""
        extras: dict[str, dict[str, Any]] = {
            u: {"primary_model": None, "complementary_models": []} for u in uids
        }
        if not uids:
            return extras
        e = T.edges
        rows = await self._read(
            sa.select(e.c.src_key, e.c.type, e.c.dst_key)
            .where(
                e.c.type.in_(("USES_PRIMARY_MODEL", "USES_COMPLEMENTARY_MODEL")),
                e.c.src_label == "ExtractionResult",
                e.c.src_key.in_(uids),
                e.c.dst_label == "CatalogModel",
            )
            .order_by(e.c.id)
        )
        for row in rows:
            extra = extras[row["src_key"]]
            if row["type"] == "USES_PRIMARY_MODEL":
                if extra["primary_model"] is None:
                    extra["primary_model"] = row["dst_key"]
            else:
                extra["complementary_models"].append(row["dst_key"])
        return extras

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
        """The extraction result of a structure node."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        er, e = T.extraction_results, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")
        linked = sa.select(e.c.dst_key).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(anchor.c.key)),
            e.c.dst_label == "ExtractionResult",
        )
        rec = await self._read_one(
            sa.select(er)
            .where(er.c.uid.in_(linked), *self._scope(er, sc, tenant=False))
            .order_by(self._dl.order_text(er.c.uid))
            .limit(1)
        )
        if not rec:
            return None
        extras = (await self._er_extras([rec["uid"]]))[rec["uid"]]
        return _map.extraction_result_ref(self._node("extraction_results", rec), **extras)

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
        """The extraction results of the structure nodes below a document."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        er, sn, e = T.extraction_results, T.structure_nodes, T.edges
        conditions: list[sa.ColumnElement[bool]] = []
        if model_class is not None:
            conditions.append(er.c.model_class == model_class)
        conditions.extend(self._scope(er, sc, tenant=False))
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_STRUCTURE_EDGES)
        below = reached(cte, "StructureNode")
        stmt = (
            sa.select(er, sn.c.id.label("_node_id"), sn.c.title.label("_node_title"))
            .select_from(
                sn.join(below, below.c.key == sn.c.id)
                .join(
                    e,
                    sa.and_(
                        e.c.type == "HAS_EXTRACTION",
                        e.c.src_label == "StructureNode",
                        e.c.src_key == sn.c.id,
                        e.c.dst_label == "ExtractionResult",
                    ),
                )
                .join(er, er.c.uid == e.c.dst_key)
            )
            .where(*conditions)
            .order_by(
                self._dl.order_value(sn.c.appearance_order),
                self._dl.order_text(sn.c.id),
                self._dl.order_text(er.c.uid),
            )
        )
        rows = await self._read(self._limit(stmt, limit))
        extras = await self._er_extras([r["uid"] for r in rows])
        out: list[ExtractionResultWithNode] = []
        for r in rows:
            base = _map.extraction_result_ref(
                self._node("extraction_results", r), **extras[r["uid"]]
            )
            out.append(
                ExtractionResultWithNode(
                    **base.model_dump(), node_id=r["_node_id"], node_title=r["_node_title"]
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
        """The model instances contained in a node's extraction results."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = 1 if direct_only else self._containment_depth(depth)
        mi, e = T.model_instances, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")
        results = sa.select(
            sa.literal("ExtractionResult", sa.Text).label("label"), e.c.dst_key.label("key")
        ).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(anchor.c.key)),
            e.c.dst_label == "ExtractionResult",
        )
        contained = reached(walk(results, max_hops=d, prefix="HAS_"), "ModelInstance")
        conditions: list[sa.ColumnElement[bool]] = [mi.c.uid.in_(sa.select(contained.c.key))]
        if model_class is not None:
            conditions.append(mi.c.model_class == model_class)
        conditions.extend(translate_where(where, mi, self._dl))
        conditions.extend(self._scope(mi, sc, "array", tenant=False))
        stmt = (
            sa.select(mi)
            .where(*conditions)
            .order_by(self._dl.order_text(mi.c.model_class), self._dl.order_text(mi.c.uid))
        )
        return await self._mi_refs(await self._read(stmt))

    def _document_instances_statement(
        self,
        path: str,
        version: int | None,
        sc: Any,
        model_class: str | None,
        where: Mapping[str, Any] | None,
        depth: int | None,
    ) -> tuple[Any, list[sa.ColumnElement[bool]]]:
        mi = T.model_instances
        cd = self._containment_depth(depth)
        keys = self._instances_from(self._doc_start(path, version, sc), cd, "doc_mi")
        conditions: list[sa.ColumnElement[bool]] = [mi.c.uid.in_(keys)]
        if model_class is not None:
            conditions.append(mi.c.model_class == model_class)
        conditions.extend(translate_where(where, mi, self._dl))
        conditions.extend(self._scope(mi, sc, "array", tenant=False))
        return mi, conditions

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
        """The model instances extracted from the structure nodes of a document."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        mi, conditions = self._document_instances_statement(
            path, version, sc, model_class, where, depth
        )
        stmt = sa.select(mi).where(*conditions).order_by(self._dl.order_text(mi.c.uid))
        return await self._mi_refs(await self._read(self._limit(stmt, limit, skip)))

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
        """Number of model instances ``get_document_model_instances`` would return."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        mi, conditions = self._document_instances_statement(
            path, version, sc, model_class, where, depth
        )
        rec = await self._read_one(sa.select(sa.func.count().label("c")).where(*conditions))
        return int(rec["c"]) if rec else 0

    def _by_class_conditions(
        self,
        model_class: str,
        where: Mapping[str, Any] | None,
        document: str | DocumentRef | None,
        sc: Any,
    ) -> tuple[list[sa.ColumnElement[bool]], Any]:
        mi = T.model_instances
        conditions: list[sa.ColumnElement[bool]] = [mi.c.model_class == model_class]
        conditions.extend(translate_where(where, mi, self._dl))
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(
                mi.c.uid.in_(self._instances_of_path(path, self._containment_depth(None)))
            )
        conditions.extend(self._scope(mi, sc, "array"))
        return conditions, sc

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
        """The instances of one model class, optionally of one document and ordered."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        mi = T.model_instances
        conditions, sc = self._by_class_conditions(model_class, where, document, sc)
        if order_by:
            safe_ident(order_by, kind="order_by field")
            ordering = self._order_by_property(mi, order_by)
        else:
            ordering = [self._dl.order_text(mi.c.uid)]
        stmt = sa.select(mi).where(*conditions).order_by(*ordering, self._dl.order_text(mi.c.uid))
        return await self._mi_refs(await self._read(self._limit(stmt, limit, skip)))

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
        """Number of instances ``get_model_instances_by_class`` would return."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        conditions, sc = self._by_class_conditions(model_class, where, document, sc)
        rec = await self._read_one(sa.select(sa.func.count().label("c")).where(*conditions))
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
        """The model instance with this uid, if it is in scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        mi = T.model_instances
        rec = await self._read_one(
            sa.select(mi).where(mi.c.uid == uid, *self._scope(mi, sc, "array")).limit(1)
        )
        if not rec:
            return None
        node = self._node("model_instances", rec)
        return _map.model_instance_ref(node, is_shell=await self._is_shell(node))

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
        """The instance whose key fields are *key_fields* (the uid is derived from them)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        from scinr.newton.utils.uid import make_instance_uid, normalize_key

        # The uid embeds the tenant, so it cannot be "all tenants": the lookup needs a
        # concrete tenant (or "__public__"), and include_public tries the tenant first, then
        # the public one.
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

    def _owning_results(self, uid: str, scope: Any, name: str) -> Any:
        """``(er_key, hops)`` walking back from the instance through ``HAS_*`` edges."""
        cte = walk(
            self._mi_anchor(uid, scope),
            max_hops=self._containment_depth(None),
            direction="in",
            prefix="HAS_",
            name=name,
        )
        return cte

    async def get_structure_nodes_for_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The structure nodes whose extraction results contain the instance."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn, e = T.structure_nodes, T.edges
        cte = self._owning_results(uid, sc, "owners")
        owners = (
            sa.select(cte.c.key.label("er"), sa.func.min(cte.c.depth).label("depth"))
            .where(cte.c.label == "ExtractionResult", cte.c.depth >= 1)
            .group_by(cte.c.key)
            .subquery("owners_r")
        )
        hops = (
            sa.select(e.c.src_key.label("node"), (sa.func.min(owners.c.depth) + 1).label("hops"))
            .select_from(
                e.join(
                    owners,
                    sa.and_(e.c.dst_label == "ExtractionResult", e.c.dst_key == owners.c.er),
                )
            )
            .where(e.c.type == "HAS_EXTRACTION", e.c.src_label == "StructureNode")
            .group_by(e.c.src_key)
            .subquery("hops")
        )
        stmt = (
            sa.select(sn)
            .select_from(sn.join(hops, hops.c.node == sn.c.id))
            .where(*self._scope(sn, sc, tenant=False))
            .order_by(hops.c.hops, self._dl.order_text(sn.c.id))
        )
        return [
            _map.structure_node_ref(self._node("structure_nodes", r))
            for r in await self._read(stmt)
        ]

    async def get_documents_for_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """The documents that contain the instance (through their extraction results)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        doc, e = T.documents, T.edges
        cte = self._owning_results(uid, sc, "owners")
        owners = reached(cte, "ExtractionResult", name="owners_r")
        nodes = sa.select(
            sa.literal("StructureNode", sa.Text).label("label"), e.c.src_key.label("key")
        ).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.dst_label == "ExtractionResult",
            e.c.dst_key.in_(sa.select(owners.c.key)),
        )
        above = reached(
            walk(
                nodes,
                max_hops=UNBOUNDED_HOPS,
                direction="in",
                types=_STRUCTURE_EDGES,
                name="doc_up",
            ),
            "Document",
            name="doc_up_r",
        )
        stmt = (
            sa.select(doc)
            .where(doc.c._key.in_(sa.select(above.c.key)), *self._scope(doc, sc, tenant=False))
            .order_by(self._dl.order_text(doc.c.path), self._dl.order_value(doc.c.version))
        )
        return [_map.document_ref(self._node("documents", r)) for r in await self._read(stmt)]

    async def get_extraction_results_for_model_instance(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ExtractionResultRef]:
        """The extraction results that contain the instance."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        er = T.extraction_results
        owners = reached(
            self._owning_results(uid, sc, "owners"), "ExtractionResult", name="owners_r"
        )
        stmt = (
            sa.select(er)
            .where(er.c.uid.in_(sa.select(owners.c.key)), *self._scope(er, sc, tenant=False))
            .order_by(self._dl.order_text(er.c.uid))
        )
        return [
            _map.extraction_result_ref(self._node("extraction_results", r))
            for r in await self._read(stmt)
        ]

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
        """Instances that reach this one in 1..depth hops."""
        return await self._directional(
            uid,
            "in",
            rel_type=rel_type,
            depth=depth,
            limit=limit,
            tenant_id=tenant_id,
            include_public=include_public,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
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
        """Instances this one reaches in 1..depth hops."""
        return await self._directional(
            uid,
            "out",
            rel_type=rel_type,
            depth=depth,
            limit=limit,
            tenant_id=tenant_id,
            include_public=include_public,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
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
        mi = T.model_instances
        cte = walk(
            self._mi_anchor(uid, sc),
            max_hops=d,
            direction=direction,
            types=None if rel_type is None else (rel_type,),
            track_types=True,
        )
        hits = (
            sa.select(cte.c.key.label("key"), cte.c.first_type.label("via"))
            .where(cte.c.label == "ModelInstance", cte.c.depth >= 1)
            .distinct()
            .subquery("hits")
        )
        stmt = (
            sa.select(mi, hits.c.via.label("_via_rel"))
            .select_from(mi.join(hits, hits.c.key == mi.c.uid))
            .where(*self._scope(mi, sc, "array", tenant=False))
            .order_by(self._dl.order_text(hits.c.via), self._dl.order_text(mi.c.uid))
        )
        rows = await self._read(self._limit(stmt, limit))
        return await self._mi_refs(rows, via={"direction": direction})

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
        """An instance and the instances below it as a tree (user/job apply to the root)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        mi = T.model_instances
        # user/job filters apply to the root only (inner nodes are not pruned).
        root_row = await self._read_one(
            sa.select(mi).where(mi.c.uid == uid, *self._scope(mi, sc, "array")).limit(1)
        )
        if root_row is None:
            return None
        root_node = self._node("model_instances", root_row)
        root = ModelInstanceTree(
            **_map.model_instance_ref(
                root_node, is_shell=await self._is_shell(root_node)
            ).model_dump(),
            depth=0,
        )
        cte = walk(
            self._mi_anchor(uid, sc, "array"),
            max_hops=d,
            track_path=True,
            track_types=True,
            only_labels=("ModelInstance",),
        )
        stmt = (
            sa.select(mi, cte.c.path.label("_lineage"), cte.c.last_type.label("_rtype"))
            .select_from(
                mi.join(cte, sa.and_(cte.c.label == "ModelInstance", cte.c.key == mi.c.uid))
            )
            .where(cte.c.depth >= 1)
            .order_by(cte.c.depth, self._dl.order_text(mi.c.uid))
        )
        by_uid: dict[str, ModelInstanceTree] = {root.uid: root}
        for row in await self._read(stmt):
            lineage = split_path(row["_lineage"])
            node_props = self._node("model_instances", row)
            node = ModelInstanceTree(
                **_map.model_instance_ref(
                    node_props, via_rel=row["_rtype"], is_shell=await self._is_shell(node_props)
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
        """The direct relationships between this instance and other instances."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        mi, e = T.model_instances, T.edges
        anchor = self._mi_anchor(uid, sc).subquery("anchor")
        parts: list[tuple[str, str, str]] = []
        if direction in ("out", "both"):
            parts.append(("out", "src_key", "dst_key"))
        if direction in ("in", "both"):
            parts.append(("in", "dst_key", "src_key"))
        out: list[ModelInstanceRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for way, near, far in parts:
            conditions: list[sa.ColumnElement[bool]] = [
                e.c.src_label == "ModelInstance",
                e.c.dst_label == "ModelInstance",
                e.c[near].in_(sa.select(anchor.c.key)),
                *self._scope(mi, sc, "array", tenant=False),
            ]
            if rel_type is not None:
                conditions.append(e.c.type == rel_type)
            stmt = (
                sa.select(mi, e.c.type.label("_rel_type"))
                .select_from(e.join(mi, mi.c.uid == e.c[far]))
                .where(*conditions)
                .order_by(self._dl.order_text(e.c.type), self._dl.order_text(mi.c.uid))
            )
            for row in await self._read(stmt):
                marker = (way, row["_rel_type"], row["uid"])
                if marker in seen:
                    continue  # UNION removes duplicate rows
                seen.add(marker)
                other = self._node("model_instances", row)
                out.append(
                    ModelInstanceRelation(
                        raw={"rel_type": row["_rel_type"], "direction": way, "o": other},
                        rel_type=row["_rel_type"],
                        direction=way,  # type: ignore[arg-type]
                        other=_map.model_instance_ref(other, is_shell=await self._is_shell(other)),
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
        """The instances linked to this one by a relationship of *rel_type*."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        mi, e = T.model_instances, T.edges
        anchor = self._mi_anchor(uid, sc).subquery("anchor")
        near, far = ("src_key", "dst_key") if direction == "out" else ("dst_key", "src_key")
        linked = sa.select(e.c[far]).where(
            e.c.type == rel_type,
            e.c.src_label == "ModelInstance",
            e.c.dst_label == "ModelInstance",
            e.c[near].in_(sa.select(anchor.c.key)),
        )
        stmt = (
            sa.select(mi)
            .where(mi.c.uid.in_(linked), *self._scope(mi, sc, "array", tenant=False))
            .order_by(self._dl.order_text(mi.c.uid))
        )
        rows = await self._read(stmt)
        return await self._mi_refs(
            [{**r, "_via_rel": rel_type} for r in rows], via={"direction": direction}
        )

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
        """Instances with far fewer properties than the average of their class.

        The population the average is computed over is the scope itself (and the class
        filter, if any), so one tenant's data never skews another's.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        mi = T.model_instances
        conditions: list[sa.ColumnElement[bool]] = [mi.c.model_class.is_not(None)]
        if model_class is not None:
            conditions.append(mi.c.model_class == model_class)
        conditions.extend(self._scope(mi, sc, "array"))
        rows = await self._read(sa.select(mi).where(*conditions))
        sizes = {r["uid"]: len(self._node("model_instances", r)) - 1 for r in rows}
        totals: dict[str, list[int]] = {}
        for r in rows:
            totals.setdefault(r["model_class"], []).append(sizes[r["uid"]])
        average = {m: sum(v) / len(v) for m, v in totals.items()}
        shells = [
            r
            for r in rows
            if sizes[r["uid"]] < average[r["model_class"]] * 0.5 or sizes[r["uid"]] <= 3
        ]
        shells.sort(key=lambda r: (r["model_class"], r["uid"]))
        if limit is not None and limit >= 0:
            shells = shells[: int(limit)]
        return await self._mi_refs(shells)

    async def list_model_instance_relationship_types(
        self,
        *,
        document: str | DocumentRef | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[RelTypeStat]:
        """Relationship types between model instances (containment ``HAS_*`` excluded)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        a, b, e = T.model_instances.alias("a"), T.model_instances.alias("b"), T.edges
        conditions: list[sa.ColumnElement[bool]] = [
            e.c.src_label == "ModelInstance",
            e.c.dst_label == "ModelInstance",
            sa.func.substr(e.c.type, 1, 4) != "HAS_",
        ]
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(
                a.c.uid.in_(self._instances_of_path(path, self._containment_depth(None)))
            )
        conditions.extend(self._scope(a, sc, "array"))
        count = sa.func.count().label("count")
        stmt = (
            sa.select(
                a.c.model_class.label("source_model"),
                e.c.type.label("rel_type"),
                b.c.model_class.label("target_model"),
                count,
            )
            .select_from(e.join(a, a.c.uid == e.c.src_key).join(b, b.c.uid == e.c.dst_key))
            .where(*conditions)
            .group_by(a.c.model_class, e.c.type, b.c.model_class)
            .order_by(
                count.desc(),
                self._dl.order_text(a.c.model_class),
                self._dl.order_text(e.c.type),
                self._dl.order_text(b.c.model_class),
            )
        )
        return [_map.rel_type_stat(r) for r in await self._read(stmt)]


__all__ = ["_ModelInstancesMixin"]
