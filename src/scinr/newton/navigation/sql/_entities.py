"""navigation/sql/_entities.py — Group F: LabeledEntity, Entity, triples (SQL)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import sqlalchemy as sa

from scinr.newton.navigation import _map
from scinr.newton.navigation.models import (
    DocumentRef,
    EntityLabelStat,
    EntityRelation,
    LabeledEntityRef,
    ModelInstanceRef,
    StructureNodeRef,
    Triple,
)
from scinr.newton.navigation.scope import make_scope, resolve_selector
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import translate_where
from scinr.newton.navigation.sql._walk import reached, walk
from scinr.newton.sql import tables as T

# Scope notes (navigation/scope.py): LabeledEntity / Entity / ModelInstance are
# merge-deduplicated — scalar tenant_id (folded into the uid) plus accumulated ``job_ids`` /
# ``created_by_user_ids`` (kind="array"). Anchors guard the tenant; returned nodes are
# filtered by user/job only (``tenant=False``). That is safe ONLY because no data
# relationship crosses tenants (public included): every link targets a uid hashed with the
# writer's own tenant.
#
# ``get_entity_triples`` matches by *value*: with tenant_id=None the same value extracted by
# several tenants comes back once per tenant (they are distinct Entity nodes). Pass a tenant
# to avoid mixing them.

_PRED = "predicate_raw"


def _labeled(rows: list[dict[str, Any]]) -> list[LabeledEntityRef]:
    return [_map.labeled_entity_ref(_SqlRuntime._node("labeled_entities", r)) for r in rows]


class _EntitiesMixin(_SqlRuntime):
    def _le_anchor(self, uid: str, scope: Any) -> sa.Select:
        """Uid of the anchor labeled entity, guarded by the tenant."""
        le = T.labeled_entities
        return sa.select(le.c.uid).where(le.c.uid == uid, *self._scope(le, scope, "tenant_only"))

    def _mi_uid_anchor(self, uid: str, scope: Any) -> sa.Select:
        mi = T.model_instances
        return sa.select(mi.c.uid).where(mi.c.uid == uid, *self._scope(mi, scope, "tenant_only"))

    @staticmethod
    def _entities_of(instances: Any, label: str | None, conditions_extra: list[Any]) -> sa.Select:
        """The labeled entities referenced by the given model-instance keys."""
        e, le = T.edges, T.labeled_entities
        linked = sa.select(e.c.dst_key).where(
            e.c.type == "REFERENCES",
            e.c.src_label == "ModelInstance",
            e.c.src_key.in_(instances),
            e.c.dst_label == "LabeledEntity",
        )
        conditions = [le.c.uid.in_(linked), *conditions_extra]
        if label is not None:
            conditions.append(le.c.label == label)
        return sa.select(le).where(*conditions)

    def _order_entities(self, stmt: sa.Select) -> sa.Select:
        le = T.labeled_entities
        return stmt.order_by(
            self._dl.order_text(le.c.label),
            self._dl.order_text(le.c.value),
            self._dl.order_text(le.c.uid),
        )

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
        """The labeled entities a model instance references, with the referencing field."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        le, e = T.labeled_entities, T.edges
        conditions: list[sa.ColumnElement[bool]] = [
            e.c.type == "REFERENCES",
            e.c.src_label == "ModelInstance",
            e.c.src_key.in_(self._mi_uid_anchor(uid, sc)),
            *self._scope(le, sc, "array", tenant=False),
        ]
        if label is not None:
            conditions.append(le.c.label == label)
        stmt = self._order_entities(
            sa.select(
                le,
                self._dl.json_text(e.c.props, "field_name").label("_field_name"),
                self._dl.json_number(e.c.props, "list_index").label("_list_index"),
            )
            .select_from(
                e.join(le, sa.and_(e.c.dst_label == "LabeledEntity", le.c.uid == e.c.dst_key))
            )
            .where(*conditions)
        )
        return [
            _map.labeled_entity_ref(
                self._node("labeled_entities", r),
                field_name=r["_field_name"],
                list_index=None if r["_list_index"] is None else int(r["_list_index"]),
            )
            for r in await self._read(stmt)
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
        """The labeled entities referenced by the instances extracted from a node."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(depth)
        le, e = T.labeled_entities, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")
        results = sa.select(
            sa.literal("ExtractionResult", sa.Text).label("label"), e.c.dst_key.label("key")
        ).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(anchor.c.key)),
            e.c.dst_label == "ExtractionResult",
        )
        contained = reached(walk(results, max_hops=cd, prefix="HAS_"), "ModelInstance")
        stmt = self._order_entities(
            self._entities_of(
                sa.select(contained.c.key), label, self._scope(le, sc, "array", tenant=False)
            )
        )
        return _labeled(await self._read(stmt))

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
        """The labeled entities referenced by the instances extracted from a document."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        cd = self._containment_depth(depth)
        le = T.labeled_entities
        instances = self._instances_from(self._doc_start(path, version, sc), cd, "doc_ent")
        stmt = self._order_entities(
            self._entities_of(instances, label, self._scope(le, sc, "array", tenant=False))
        )
        return _labeled(await self._read(self._limit(stmt, limit)))

    async def list_entity_labels(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[EntityLabelStat]:
        """How many labeled entities each label has, most frequent first."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        le = T.labeled_entities
        count = sa.func.count().label("count")
        stmt = (
            sa.select(le.c.label.label("label"), count)
            .where(*self._scope(le, sc, "array"))
            .group_by(le.c.label)
            .order_by(count.desc(), self._dl.order_text(le.c.label))
        )
        return [_map.entity_label_stat(r) for r in await self._read(stmt)]

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
        """Labeled entities filtered by label, value and ``where``."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        le = T.labeled_entities
        conditions: list[sa.ColumnElement[bool]] = []
        if label is not None:
            conditions.append(le.c.label == label)
        if value is not None:
            conditions.append(le.c.value == value)
        if normalized_value is not None:
            conditions.append(le.c.normalized_value == normalized_value)
        conditions.extend(translate_where(where, le, self._dl))
        conditions.extend(self._scope(le, sc, "array"))
        stmt = self._order_entities(sa.select(le).where(*conditions))
        return _labeled(await self._read(self._limit(stmt, limit, skip)))

    async def get_labeled_entity(
        self,
        uid: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> LabeledEntityRef | None:
        """The labeled entity with this uid, if it is in scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        le = T.labeled_entities
        rec = await self._read_one(
            sa.select(le).where(le.c.uid == uid, *self._scope(le, sc, "array")).limit(1)
        )
        return _labeled([rec])[0] if rec else None

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
        """The model instances that reference a labeled entity."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        mi, e = T.model_instances, T.edges
        referencing = sa.select(e.c.src_key).where(
            e.c.type == "REFERENCES",
            e.c.src_label == "ModelInstance",
            e.c.dst_label == "LabeledEntity",
            e.c.dst_key.in_(self._le_anchor(uid, sc)),
        )
        conditions: list[sa.ColumnElement[bool]] = [mi.c.uid.in_(referencing)]
        if model_class is not None:
            conditions.append(mi.c.model_class == model_class)
        conditions.extend(self._scope(mi, sc, "array", tenant=False))
        stmt = (
            sa.select(mi)
            .where(*conditions)
            .order_by(self._dl.order_text(mi.c.model_class), self._dl.order_text(mi.c.uid))
        )
        out: list[ModelInstanceRef] = []
        for row in await self._read(self._limit(stmt, limit)):
            node = self._node("model_instances", row)
            out.append(_map.model_instance_ref(node, is_shell=await self._is_shell(node)))
        return out

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
        """The structure nodes whose extractions reference a labeled entity."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        cd = self._containment_depth(depth)
        sn, e = T.structure_nodes, T.edges
        mi_start = sa.select(
            sa.literal("ModelInstance", sa.Text).label("label"), e.c.src_key.label("key")
        ).where(
            e.c.type == "REFERENCES",
            e.c.src_label == "ModelInstance",
            e.c.dst_label == "LabeledEntity",
            e.c.dst_key.in_(self._le_anchor(uid, sc)),
        )
        owners = reached(
            walk(mi_start, max_hops=cd, direction="in", prefix="HAS_"), "ExtractionResult"
        )
        extracted_from = sa.select(e.c.src_key).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.dst_label == "ExtractionResult",
            e.c.dst_key.in_(sa.select(owners.c.key)),
        )
        stmt = (
            sa.select(sn)
            .where(sn.c.id.in_(extracted_from), *self._scope(sn, sc, tenant=False))
            .order_by(self._dl.order_text(sn.c.id))
        )
        rows = await self._read(self._limit(stmt, limit))
        return [_map.structure_node_ref(self._node("structure_nodes", r)) for r in rows]

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
        """The direct relationships between this labeled entity and other ones."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        le, e = T.labeled_entities, T.edges
        ways: list[tuple[str, str, str]] = []
        if direction in ("out", "both"):
            ways.append(("out", "src_key", "dst_key"))
        if direction in ("in", "both"):
            ways.append(("in", "dst_key", "src_key"))
        out: list[EntityRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for way, near, far in ways:
            conditions: list[sa.ColumnElement[bool]] = [
                e.c.src_label == "LabeledEntity",
                e.c.dst_label == "LabeledEntity",
                e.c[near].in_(self._le_anchor(uid, sc)),
                *self._scope(le, sc, "array", tenant=False),
            ]
            if rel_type is not None:
                conditions.append(e.c.type == rel_type)
            stmt = (
                sa.select(le, e.c.type.label("_rel_type"))
                .select_from(e.join(le, le.c.uid == e.c[far]))
                .where(*conditions)
                .order_by(self._dl.order_text(e.c.type), self._dl.order_text(le.c.uid))
            )
            for row in await self._read(stmt):
                marker = (way, row["_rel_type"], row["uid"])
                if marker in seen:
                    continue  # UNION removes duplicate rows
                seen.add(marker)
                other = self._node("labeled_entities", row)
                out.append(
                    EntityRelation(
                        raw={"rel_type": row["_rel_type"], "direction": way, "o": other},
                        rel_type=row["_rel_type"],
                        direction=way,  # type: ignore[arg-type]
                        other=_map.labeled_entity_ref(other),
                    )
                )
        return out

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
        """The labeled entities linked to this one by a relationship of *rel_type*."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        le, e = T.labeled_entities, T.edges
        near, far = ("src_key", "dst_key") if direction == "out" else ("dst_key", "src_key")
        linked = sa.select(e.c[far]).where(
            e.c.type == rel_type,
            e.c.src_label == "LabeledEntity",
            e.c.dst_label == "LabeledEntity",
            e.c[near].in_(self._le_anchor(uid, sc)),
        )
        stmt = self._order_entities(
            sa.select(le).where(le.c.uid.in_(linked), *self._scope(le, sc, "array", tenant=False))
        )
        return _labeled(await self._read(stmt))

    async def get_triples(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[Triple]:
        """The (subject, predicate, object) triples extracted from a node.

        A subject without a matching object comes back once with ``predicate`` and
        ``object`` unset (the Cypher's ``OPTIONAL MATCH``).
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        er, ent, e = T.extraction_results, T.entities, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")
        subject_edge, link, obj_edge = e.alias("subj"), e.alias("link"), e.alias("obj")
        subject, obj = ent.alias("s"), ent.alias("o")
        role = self._dl.json_text
        extracted = sa.select(e.c.dst_key).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(anchor.c.key)),
            e.c.dst_label == "ExtractionResult",
        )
        matched = (
            sa.select(
                link.c.src_key.label("s_key"),
                obj_edge.c.src_key.label("er_key"),
                link.c.type.label("predicate"),
                self._dl.json_text(link.c.props, _PRED).label("predicate_raw"),
                obj.c.value.label("object"),
            )
            .select_from(
                link.join(
                    obj, sa.and_(link.c.dst_label == "Entity", obj.c.uid == link.c.dst_key)
                ).join(
                    obj_edge,
                    sa.and_(
                        obj_edge.c.type == "HAS_ENTITY",
                        obj_edge.c.src_label == "ExtractionResult",
                        obj_edge.c.dst_label == "Entity",
                        obj_edge.c.dst_key == obj.c.uid,
                        role(obj_edge.c.props, "role") == "object",
                    ),
                )
            )
            .where(link.c.src_label == "Entity")
            .subquery("matched")
        )
        stmt = (
            sa.select(
                subject.c.value.label("subject"),
                matched.c.predicate,
                matched.c.predicate_raw,
                matched.c.object,
            )
            .select_from(
                er.join(
                    subject_edge,
                    sa.and_(
                        subject_edge.c.type == "HAS_ENTITY",
                        subject_edge.c.src_label == "ExtractionResult",
                        subject_edge.c.src_key == er.c.uid,
                        subject_edge.c.dst_label == "Entity",
                        role(subject_edge.c.props, "role") == "subject",
                    ),
                )
                .join(subject, subject.c.uid == subject_edge.c.dst_key)
                .outerjoin(
                    matched,
                    sa.and_(matched.c.s_key == subject.c.uid, matched.c.er_key == er.c.uid),
                )
            )
            .where(
                er.c.uid.in_(extracted),
                er.c.model_class == "Triple",
                *self._scope(er, sc, tenant=False),
            )
            .order_by(
                self._dl.order_text(subject.c.value),
                self._dl.order_text(matched.c.predicate),
                self._dl.order_text(matched.c.object),
            )
        )
        return [_map.triple(r, node_id=node_id) for r in await self._read(stmt)]

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
        """The triples an entity takes part in, matched by uid, value or normalized value."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ent, other, link = T.entities, T.entities.alias("x"), T.edges
        anchor = [
            sa.or_(
                ent.c.uid == value_or_uid,
                ent.c.value == value_or_uid,
                ent.c.normalized_value == value_or_uid.lower(),
            ),
            *self._scope(ent, sc, "array"),
        ]
        rows: list[dict[str, Any]] = []
        ways: list[tuple[str, str, bool]] = []
        if direction in ("out", "both"):
            ways.append(("src_key", "dst_key", True))
        if direction in ("in", "both"):
            ways.append(("dst_key", "src_key", False))
        for near, far, is_out in ways:
            stmt = (
                sa.select(
                    ent.c.value.label("anchor_value"),
                    other.c.value.label("other_value"),
                    link.c.type.label("predicate"),
                    self._dl.json_text(link.c.props, _PRED).label("predicate_raw"),
                )
                .select_from(
                    link.join(ent, ent.c.uid == link.c[near]).join(
                        other, other.c.uid == link.c[far]
                    )
                )
                .where(link.c.src_label == "Entity", link.c.dst_label == "Entity", *anchor)
                .order_by(
                    self._dl.order_text(link.c.type),
                    self._dl.order_text(ent.c.value),
                    self._dl.order_text(other.c.value),
                )
            )
            for r in await self._read(stmt):
                subject, obj = (
                    (r["anchor_value"], r["other_value"])
                    if is_out
                    else (r["other_value"], r["anchor_value"])
                )
                rows.append(
                    {
                        "subject": subject,
                        "predicate": r["predicate"],
                        "predicate_raw": r["predicate_raw"],
                        "object": obj,
                    }
                )
        seen: set[tuple[Any, ...]] = set()
        out: list[Triple] = []
        for r in rows:  # UNION removes duplicate rows
            marker = tuple(r.values())
            if marker not in seen:
                seen.add(marker)
                out.append(_map.triple(r))
        return out


__all__ = ["_EntitiesMixin"]
