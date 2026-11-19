"""navigation/sql/_structure.py — Group B: structure nodes & the document tree (SQL)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa

from scinr.newton.navigation import _map
from scinr.newton.navigation.models import (
    DocumentRef,
    NodeDescription,
    NodePath,
    StructureNodeRef,
    StructureTree,
)
from scinr.newton.navigation.scope import make_scope, resolve_selector
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import translate_where
from scinr.newton.navigation.sql._walk import UNBOUNDED_HOPS, reached, split_path, walk
from scinr.newton.sql import tables as T

# Scope notes (navigation/scope.py). A structure node carries scalar tenant_id /
# created_by_user_id / job_id copied from its document, and its id embeds the tenant.
# Document-anchored methods pin the *document* to the tenant and filter the returned nodes by
# user/job; node-anchored methods guard the anchor with the full scope (a foreign id does not
# resolve) and filter returned nodes by user/job. Edges never cross tenants, so the anchor is
# enough — no traversal back up to the document.

_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")
_CHILD = ("HAS_CHILD",)


class _StructureMixin(_SqlRuntime):
    def _structure_refs(
        self, rows: list[dict[str, Any]], *, with_document: bool = False
    ) -> list[StructureNodeRef]:
        return [
            _map.structure_node_ref(
                self._node("structure_nodes", r),
                document_path=r.get("_dp") if with_document else None,
                document_version=r.get("_dv") if with_document else None,
            )
            for r in rows
        ]

    def _order_nodes(self, stmt: sa.Select) -> sa.Select:
        sn = T.structure_nodes
        return stmt.order_by(
            self._dl.order_value(sn.c.appearance_order), self._dl.order_text(sn.c.id)
        )

    async def get_structure_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        roles: Sequence[str] | None = None,
        title_contains: str | None = None,
        theme: str | None = None,
        where: Mapping[str, Any] | None = None,
        depth: int | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The structure nodes below a document (any depth up to *depth*), in document order."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        sn, doc = T.structure_nodes, T.documents
        conditions: list[sa.ColumnElement[bool]] = []
        if roles:
            conditions.append(sn.c.role.in_(list(roles)))
        if title_contains is not None:
            conditions.append(self._dl.contains_ci(sn.c.title, title_contains))
        if theme is not None:
            conditions.append(sn.c.theme == theme)
        conditions.extend(translate_where(where, sn, self._dl))
        conditions.extend(self._scope(sn, sc, tenant=False))
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_STRUCTURE_EDGES)
        below = reached(cte, "StructureNode")
        stmt = (
            sa.select(sn, doc.c.path.label("_dp"), doc.c.version.label("_dv"))
            .select_from(
                sn.join(below, below.c.key == sn.c.id).join(doc, doc.c._key == below.c.origin)
            )
            .where(*conditions)
        )
        rows = await self._read(self._limit(self._order_nodes(stmt), limit, skip))
        return self._structure_refs(rows, with_document=True)

    async def count_structure_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        roles: Sequence[str] | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Number of structure nodes ``get_structure_nodes`` would return."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        d = self._resolve_depth(depth)
        sn = T.structure_nodes
        conditions: list[sa.ColumnElement[bool]] = []
        if roles:
            conditions.append(sn.c.role.in_(list(roles)))
        conditions.extend(self._scope(sn, sc, tenant=False))
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_STRUCTURE_EDGES)
        below = reached(cte, "StructureNode")
        rec = await self._read_one(
            sa.select(sa.func.count(sa.distinct(sn.c.id)).label("c"))
            .select_from(sn.join(below, below.c.key == sn.c.id))
            .where(*conditions)
        )
        return int(rec["c"]) if rec else 0

    async def get_root_structure_nodes(
        self,
        document: str | DocumentRef,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The top-level structure nodes of a document (``HAS_STRUCTURE``)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        path, sc = resolve_selector(document, sc)
        sn, doc = T.structure_nodes, T.documents
        cte = walk(self._doc_start(path, version, sc), max_hops=1, types=("HAS_STRUCTURE",))
        below = reached(cte, "StructureNode")
        stmt = (
            sa.select(sn, doc.c.path.label("_dp"), doc.c.version.label("_dv"))
            .select_from(
                sn.join(below, below.c.key == sn.c.id).join(doc, doc.c._key == below.c.origin)
            )
            .where(*self._scope(sn, sc, tenant=False))
        )
        return self._structure_refs(await self._read(self._order_nodes(stmt)), with_document=True)

    async def get_structure_node(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureNodeRef | None:
        """The structure node with this id, if it is in scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn = T.structure_nodes
        rec = await self._read_one(
            sa.select(sn).where(sn.c.id == node_id, *self._scope(sn, sc)).limit(1)
        )
        return self._structure_refs([rec])[0] if rec else None

    async def get_structure_nodes_by_ids(
        self,
        node_ids: Sequence[str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The structure nodes with these ids, in the order given (missing ones skipped)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return []
        sn = T.structure_nodes
        rows = await self._read(sa.select(sn).where(sn.c.id.in_(ids), *self._scope(sn, sc)))
        by_id = {r["id"]: r for r in rows}
        return self._structure_refs([by_id[i] for i in ids if i in by_id])

    async def get_child_nodes(
        self,
        node_id: str,
        *,
        depth: int | None = 1,
        roles: Sequence[str] | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The descendants of a node through ``HAS_CHILD``, up to *depth* levels."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        sn = T.structure_nodes
        cte = walk(self._node_anchor(node_id, sc), max_hops=d, types=_CHILD)
        below = reached(cte, "StructureNode")
        conditions: list[sa.ColumnElement[bool]] = [sn.c.id.in_(sa.select(below.c.key))]
        if roles:
            conditions.append(sn.c.role.in_(list(roles)))
        conditions.extend(self._scope(sn, sc, tenant=False))
        stmt = self._order_nodes(sa.select(sn).where(*conditions))
        return self._structure_refs(await self._read(self._limit(stmt, limit)))

    async def get_structure_subtree(
        self,
        node_id: str,
        *,
        depth: int | None = None,
        include_info_units: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureTree | None:
        """A node and its descendants as a tree (user/job filters apply to the root)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        sn = T.structure_nodes
        # user/job filters apply to the root only: pruning inner nodes would leave their
        # children orphaned.
        root_row = await self._read_one(
            sa.select(sn).where(sn.c.id == node_id, *self._scope(sn, sc, "scalar")).limit(1)
        )
        if root_row is None:
            return None
        root = StructureTree(**self._structure_refs([root_row])[0].model_dump(), depth=0)
        cte = walk(
            self._node_anchor(node_id, sc, "scalar"),
            max_hops=d,
            types=_CHILD,
            track_path=True,
        )
        stmt = (
            sa.select(sn, cte.c.path.label("_lineage"))
            .select_from(
                sn.join(cte, sa.and_(cte.c.label == "StructureNode", cte.c.key == sn.c.id))
            )
            .where(cte.c.depth >= 1)
            .order_by(
                cte.c.depth,
                self._dl.order_value(sn.c.appearance_order),
                self._dl.order_text(sn.c.id),
            )
        )
        rows = await self._read(stmt)
        units: dict[str, list[Any]] = {}
        if include_info_units and rows:
            units = await self._units_by_node([r["id"] for r in rows])
        by_id: dict[str, StructureTree] = {root.id: root}
        for row in rows:
            lineage = split_path(row["_lineage"])
            ref = self._structure_refs([row])[0]
            node = StructureTree(
                **ref.model_dump(),
                depth=len(lineage) - 1,
                info_units=[_map.info_unit_ref(u) for u in units.get(row["id"], [])] or None,
            )
            by_id[node.id] = node
            parent = by_id.get(lineage[-2]) if len(lineage) >= 2 else root
            if parent is not None:
                parent.children.append(node)
        return root

    async def _units_by_node(self, node_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """The info units (as node dicts) of each structure node, by ``HAS_INFO_UNIT``."""
        iu, e = T.info_units, T.edges
        stmt = (
            sa.select(iu, e.c.src_key.label("_node"))
            .select_from(iu.join(e, sa.and_(e.c.dst_label == "InfoUnit", e.c.dst_key == iu.c.uid)))
            .where(
                e.c.type == "HAS_INFO_UNIT",
                e.c.src_label == "StructureNode",
                e.c.src_key.in_(node_ids),
            )
            .order_by(self._dl.order_value(iu.c.order), self._dl.order_text(iu.c.uid))
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for row in await self._read(stmt):
            out.setdefault(row["_node"], []).append(self._node("info_units", row))
        return out

    async def get_parent_node(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> StructureNodeRef | None:
        """The ``HAS_CHILD`` parent of a node."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn, e = T.structure_nodes, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")
        parents = sa.select(e.c.src_key).where(
            e.c.type == "HAS_CHILD",
            e.c.src_label == "StructureNode",
            e.c.dst_label == "StructureNode",
            e.c.dst_key.in_(sa.select(anchor.c.key)),
        )
        rec = await self._read_one(
            sa.select(sn)
            .where(sn.c.id.in_(parents), *self._scope(sn, sc, tenant=False))
            .order_by(self._dl.order_text(sn.c.id))
            .limit(1)
        )
        return self._structure_refs([rec])[0] if rec else None

    async def _spine(self, node_id: str, sc: Any, max_hops: int) -> tuple[list[str], str] | None:
        """``(structure keys top→…→node, document key)`` of the longest path up to *node_id*.

        The Cypher ``(doc)-[:HAS_STRUCTURE]->(a)-[:HAS_CHILD*0..N]->(t)`` with the longest
        path kept: walk backwards from *t* (``N`` ``HAS_CHILD`` hops plus the ``HAS_STRUCTURE``
        one) and keep the deepest document reached.
        """
        cte = walk(
            self._node_anchor(node_id, sc, "scalar"),
            max_hops=max_hops + 1,
            direction="in",
            types=_STRUCTURE_EDGES,
            track_path=True,
        )
        rec = await self._read_one(
            sa.select(cte.c.path)
            .where(cte.c.label == "Document", cte.c.depth >= 1)
            .order_by(cte.c.depth.desc(), cte.c.path)
            .limit(1)
        )
        if rec is None:
            return None
        keys = split_path(rec["path"])  # node → … → a → document
        return list(reversed(keys[:-1])), keys[-1]

    async def _structure_rows(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        sn = T.structure_nodes
        return {r["id"]: r for r in await self._read(sa.select(sn).where(sn.c.id.in_(keys)))}

    async def get_node_ancestors(
        self,
        node_id: str,
        *,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The chain of ancestors from the top-level node down to the node's parent.

        Like the Cypher (which returns bare node maps, without labels), the refs have empty
        ``types``.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        spine = await self._spine(node_id, sc, d)
        if spine is None:
            return []
        keys, doc_key = spine
        ancestors = keys[:-1]
        if not ancestors:
            return []
        rows = await self._structure_rows(ancestors)
        doc = T.documents
        doc_row = await self._read_one(sa.select(doc).where(doc.c._key == doc_key))
        doc_props = self._node("documents", doc_row) if doc_row else {}
        return [
            _map.structure_node_ref(
                {**self._node("structure_nodes", rows[k]), "_labels": []},
                document_path=doc_props.get("path"),
                document_version=doc_props.get("version"),
            )
            for k in ancestors
            if k in rows
        ]

    async def get_node_path(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> NodePath | None:
        """The document and the structure nodes from the top level down to this node.

        Like the Cypher (which returns bare node maps, without labels), the refs have empty
        ``types``.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        spine = await self._spine(node_id, sc, UNBOUNDED_HOPS)
        if spine is None:
            return None
        keys, doc_key = spine
        rows = await self._structure_rows(keys)
        doc = T.documents
        doc_row = await self._read_one(sa.select(doc).where(doc.c._key == doc_key))
        if doc_row is None:
            return None
        doc_props = {k: v for k, v in self._node("documents", doc_row).items() if k != "_labels"}
        node_props = [
            {k: v for k, v in self._node("structure_nodes", rows[key]).items() if k != "_labels"}
            for key in keys
            if key in rows
        ]
        return NodePath(
            raw={"doc": doc_props, "nodes": node_props},
            document=_map.document_ref(doc_props),
            nodes=[_map.structure_node_ref({**n, "_labels": []}) for n in node_props],
        )

    async def get_document_of_node(
        self,
        node_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        """A document that contains the node (through ``HAS_STRUCTURE`` / ``HAS_CHILD``)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        doc = T.documents
        cte = walk(
            self._node_anchor(node_id, sc, "scalar"),
            max_hops=UNBOUNDED_HOPS,
            direction="in",
            types=_STRUCTURE_EDGES,
        )
        above = reached(cte, "Document")
        rec = await self._read_one(
            sa.select(doc)
            .where(doc.c._key.in_(sa.select(above.c.key)))
            .order_by(self._dl.order_text(doc.c.path), self._dl.order_value(doc.c.version))
            .limit(1)
        )
        return _map.document_ref(self._node("documents", rec)) if rec else None

    async def get_sibling_nodes(
        self,
        node_id: str,
        *,
        include_self: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """The nodes sharing the node's parent (or the document, for a top-level node)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn, e = T.structure_nodes, T.edges
        anchor = self._node_anchor(node_id, sc).subquery("anchor")

        def siblings(via: str, parent_label: str) -> sa.Select:
            parents = sa.select(e.c.src_key).where(
                e.c.type == via,
                e.c.src_label == parent_label,
                e.c.dst_label == "StructureNode",
                e.c.dst_key.in_(sa.select(anchor.c.key)),
            )
            children = sa.select(e.c.dst_key).where(
                e.c.type == via,
                e.c.src_label == parent_label,
                e.c.src_key.in_(parents),
                e.c.dst_label == "StructureNode",
            )
            return self._order_nodes(
                sa.select(sn).where(sn.c.id.in_(children), *self._scope(sn, sc, tenant=False))
            )

        # Try the HAS_CHILD parent first; fall back to HAS_STRUCTURE (a top-level node).
        rows = await self._read(siblings("HAS_CHILD", "StructureNode"))
        if not rows:
            rows = await self._read(siblings("HAS_STRUCTURE", "Document"))
        out = self._structure_refs(rows)
        if not include_self:
            out = [s for s in out if s.id != node_id]
        return out

    async def find_structure_nodes(
        self,
        *,
        title_contains: str | None = None,
        node_id: str | None = None,
        role: str | None = None,
        theme: str | None = None,
        document: str | DocumentRef | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """Structure nodes matching the given fields, ordered by id."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn = T.structure_nodes
        conditions: list[sa.ColumnElement[bool]] = []
        if title_contains is not None:
            conditions.append(self._dl.contains_ci(sn.c.title, title_contains))
        if node_id is not None:
            conditions.append(sn.c.node_id == node_id)
        if role is not None:
            conditions.append(sn.c.role == role)
        if theme is not None:
            conditions.append(sn.c.theme == theme)
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(self._in_document_path(path, sn.c.id))
        conditions.extend(translate_where(where, sn, self._dl))
        conditions.extend(self._scope(sn, sc))
        stmt = sa.select(sn).where(*conditions).order_by(self._dl.order_text(sn.c.id))
        return self._structure_refs(await self._read(self._limit(stmt, limit, skip)))

    async def get_nodes_by_theme(
        self,
        theme: str,
        *,
        document: str | DocumentRef | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[StructureNodeRef]:
        """Structure nodes tagged with *theme*, ordered by id."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sn = T.structure_nodes
        conditions: list[sa.ColumnElement[bool]] = [sn.c.theme == theme]
        if document is not None:
            path, sc = resolve_selector(document, sc)
            conditions.append(self._in_document_path(path, sn.c.id))
        conditions.extend(self._scope(sn, sc))
        stmt = sa.select(sn).where(*conditions).order_by(self._dl.order_text(sn.c.id))
        return self._structure_refs(await self._read(self._limit(stmt, limit)))

    async def describe_node(
        self,
        node_id: str,
        *,
        include_source_text: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> NodeDescription | None:
        """A node with its ancestors, info units, decision, extraction and counts."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        scope_kw: dict[str, Any] = {
            "tenant_id": tenant_id,
            "include_public": include_public,
            "created_by_user_id": created_by_user_id,
            "job_id": job_id,
        }
        node = await self.get_structure_node(node_id, **scope_kw)  # type: ignore[attr-defined]
        if node is None:
            return None
        # The anchor passed the full scope; what hangs off it lives in the same tenant, so
        # only the tenant is pinned for the sub-queries.
        sub_kw: dict[str, Any] = {
            "tenant_id": node.raw.get("tenant_id") or tenant_id,
            "include_public": False,
        }
        ancestors = await self.get_node_ancestors(node_id, **sub_kw)  # type: ignore[attr-defined]
        info_units = await self.get_info_units(node_id, **sub_kw)  # type: ignore[attr-defined]
        decision = await self.get_model_decision(node_id, **sub_kw)  # type: ignore[attr-defined]
        extraction = await self.get_extraction_result(node_id, **sub_kw)  # type: ignore[attr-defined]
        e, sn = T.edges, T.structure_nodes
        anchor = await self._read_one(
            sa.select(sn.c.id).where(sn.c.id == node_id, *self._scope(sn, sc, "scalar"))
        )
        child_count = mi_count = 0
        if anchor is not None:
            children = await self._read_one(
                sa.select(sa.func.count().label("c")).where(
                    e.c.type == "HAS_CHILD",
                    e.c.src_label == "StructureNode",
                    e.c.src_key == node_id,
                )
            )
            child_count = int((children or {}).get("c", 0) or 0)
            results = sa.select(
                sa.literal("ExtractionResult", sa.Text).label("label"), e.c.dst_key.label("key")
            ).where(
                e.c.type == "HAS_EXTRACTION",
                e.c.src_label == "StructureNode",
                e.c.src_key == node_id,
                e.c.dst_label == "ExtractionResult",
            )
            cte = walk(results, max_hops=self._containment_depth(None), prefix="HAS_")
            contained = reached(cte, "ModelInstance")
            counted = await self._read_one(
                sa.select(sa.func.count(sa.distinct(contained.c.key)).label("c"))
            )
            mi_count = int((counted or {}).get("c", 0) or 0)
        source_text = None
        if include_source_text:
            try:
                from scinr.newton.navigation import pages

                # Reuses the already-resolved node: only its pages are read.
                source_text = await pages._node_source_text(node)
            except Exception:  # noqa: BLE001 — source text is best-effort
                source_text = None
        return NodeDescription(
            raw={"child_count": child_count, "mi_count": mi_count},
            node=node,
            ancestors=ancestors,
            info_units=info_units,
            model_decision=decision,
            extraction=extraction,
            model_instance_count=mi_count,
            child_count=child_count,
            source_page_ids=list(node.source_page_ids),
            source_text=source_text,
        )


__all__ = ["_StructureMixin"]
