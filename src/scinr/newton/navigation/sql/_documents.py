"""navigation/sql/_documents.py — Group A: documents & folder hierarchy (SQL)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation import _map
from scinr.newton.navigation.models import DocumentRef, DocumentStats, DocumentTree
from scinr.newton.navigation.scope import make_scope
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import prop_equals, starts_with, translate_where
from scinr.newton.navigation.sql._walk import reached, split_path, walk
from scinr.newton.sql import tables as T
from scinr.newton.sql.keys import decode_key

# Scope notes (see navigation/scope.py). A document is keyed by (tenant_id, path, version):
# the same path can exist once per tenant.
#  * list methods return every match in scope (each DocumentRef carries its tenant);
#  * single-document methods resolve the tenant first (``_single_document_scope``) and raise
#    NavigationError when the path is ambiguous;
#  * folder documents are re-merged on every ingestion, so their job_id / created_by_user_id
#    are those of the *last* run that touched them.

_COMPOSED = ("IS_COMPOSED_OF",)


def _doc_path(key: str) -> str:
    return decode_key("Document", key)["path"]


class _DocumentsMixin(_SqlRuntime):
    def _document_refs(self, rows: list[dict[str, Any]]) -> list[DocumentRef]:
        return [_map.document_ref(self._node("documents", r)) for r in rows]

    @staticmethod
    def _has_no_parent() -> sa.ColumnElement[bool]:
        """``NOT ( ()-[:IS_COMPOSED_OF]->(d) )`` for the documents table."""
        return ~sa.exists().where(
            T.edges.c.type == "IS_COMPOSED_OF",
            T.edges.c.dst_label == "Document",
            T.edges.c.dst_key == T.documents.c._key,
        )

    def _root_conditions(
        self,
        latest_only: bool,
        only_folders: bool,
        only_leaves: bool,
        scope: Any,
    ) -> list[sa.ColumnElement[bool]]:
        if only_folders and only_leaves:
            raise NavigationError("only_folders and only_leaves are mutually exclusive")
        doc = T.documents
        conditions = [self._has_no_parent()]
        if latest_only:
            conditions.append(doc.c.latest == sa.true())
        if only_folders:
            conditions.append(doc.c.is_folder == sa.true())
        if only_leaves:
            conditions.append(doc.c.is_folder == sa.false())
        conditions.extend(self._scope(doc, scope))
        return conditions

    def _order_documents(self, stmt: sa.Select) -> sa.Select:
        doc = T.documents
        return stmt.order_by(
            self._dl.order_text(doc.c.path),
            self._dl.order_value(doc.c.version),
            self._dl.order_text(doc.c.tenant_id),
        )

    async def list_root_documents(
        self,
        *,
        latest_only: bool = True,
        only_folders: bool = False,
        only_leaves: bool = False,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Documents no ``IS_COMPOSED_OF`` points at, by path and version."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        conditions = self._root_conditions(latest_only, only_folders, only_leaves, sc)
        stmt = self._order_documents(sa.select(T.documents).where(*conditions))
        return self._document_refs(await self._read(self._limit(stmt, limit, skip)))

    async def count_root_documents(
        self,
        *,
        latest_only: bool = True,
        only_folders: bool = False,
        only_leaves: bool = False,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Number of documents ``list_root_documents`` would return."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        conditions = self._root_conditions(latest_only, only_folders, only_leaves, sc)
        rec = await self._read_one(sa.select(sa.func.count().label("c")).where(*conditions))
        return int(rec["c"]) if rec else 0

    async def get_one_document(
        self,
        path: str,
        version: int,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        """The document at *path* and *version* (tenant resolved first)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        doc = T.documents
        rec = await self._read_one(
            sa.select(doc).where(
                doc.c.path == path, doc.c.version == int(version), *self._scope(doc, sc)
            )
        )
        return self._document_refs([rec])[0] if rec else None

    async def get_documents(
        self,
        *,
        path: str | None = None,
        name_contains: str | None = None,
        version: int | None = None,
        latest_only: bool = True,
        is_folder: bool | None = None,
        path_prefix: str | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        skip: int = 0,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Documents filtered by path, name, version, folder flag, prefix and ``where``."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        doc = T.documents
        conditions: list[sa.ColumnElement[bool]] = []
        if path is not None:
            conditions.append(doc.c.path == path)
        if name_contains is not None:
            conditions.append(self._dl.contains_ci(doc.c.name, name_contains))
        if version is not None:
            conditions.append(doc.c.version == int(version))
        if latest_only and version is None:
            conditions.append(doc.c.latest == sa.true())
        if is_folder is not None:
            conditions.append(doc.c.is_folder == bool(is_folder))
        if path_prefix is not None:
            conditions.append(starts_with(doc.c.path, path_prefix))
        conditions.extend(translate_where(where, doc, self._dl))
        conditions.extend(self._scope(doc, sc))
        stmt = self._order_documents(sa.select(doc).where(*conditions))
        return self._document_refs(await self._read(self._limit(stmt, limit, skip)))

    async def document_exists(
        self,
        path: str,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> bool:
        """Whether a document at *path* (and *version*) is in scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        doc = T.documents
        conditions = [doc.c.path == path]
        if version is not None:
            conditions.append(doc.c.version == int(version))
        conditions.extend(self._scope(doc, sc))
        rec = await self._read_one(sa.select(sa.exists().where(*conditions).label("found")))
        return bool(rec and rec["found"])

    async def get_child_documents(
        self,
        path: str,
        *,
        depth: int | None = 1,
        version: int | None = None,
        is_folder: bool | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Documents below *path* through ``IS_COMPOSED_OF``, up to *depth* levels."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        doc = T.documents
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_COMPOSED)
        below = reached(cte, "Document")
        conditions: list[sa.ColumnElement[bool]] = [doc.c._key.in_(sa.select(below.c.key))]
        if is_folder is not None:
            conditions.append(doc.c.is_folder == bool(is_folder))
        conditions.extend(self._scope(doc, sc, tenant=False))
        stmt = sa.select(doc).where(*conditions).order_by(self._dl.order_text(doc.c.path))
        return self._document_refs(await self._read(self._limit(stmt, limit)))

    async def get_document_tree(
        self,
        path: str,
        *,
        depth: int | None = None,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentTree | None:
        """The document and its descendants as a tree (user/job filters apply to the root)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        d = self._resolve_depth(depth)
        doc = T.documents
        # user/job filters apply to the anchor only: pruning inner folders would orphan
        # their children.
        conditions = self._doc_conditions(path, version, sc, "scalar")
        root_row = await self._read_one(
            self._order_documents(sa.select(doc).where(*conditions)).limit(1)
        )
        if root_row is None:
            return None
        root = DocumentTree(**self._document_refs([root_row])[0].model_dump(), depth=0)
        cte = walk(
            self._doc_start(path, version, sc, "scalar"),
            max_hops=d,
            types=_COMPOSED,
            track_path=True,
        )
        stmt = (
            sa.select(doc, cte.c.path.label("_lineage"))
            .select_from(doc.join(cte, sa.and_(cte.c.label == "Document", cte.c.key == doc.c._key)))
            .where(cte.c.depth >= 1)
            .order_by(cte.c.depth, self._dl.order_text(doc.c.path))
        )
        by_path: dict[str, DocumentTree] = {root.path: root}
        for row in await self._read(stmt):
            lineage = [_doc_path(k) for k in split_path(row["_lineage"])]
            child = self._document_refs([row])[0]
            node = DocumentTree(**child.model_dump(), depth=len(lineage) - 1)
            by_path[node.path] = node
            parent = by_path.get(lineage[-2] if len(lineage) >= 2 else root.path)
            if parent is not None:
                parent.children.append(node)
        return root

    async def get_document_parent(
        self,
        path: str,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        """The document that ``IS_COMPOSED_OF``-contains *path* (highest version first)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        doc = T.documents
        e = T.edges
        anchor = self._doc_start(path, version, sc).subquery("anchor")
        parents = sa.select(e.c.src_key).where(
            e.c.type == "IS_COMPOSED_OF",
            e.c.src_label == "Document",
            e.c.dst_label == "Document",
            e.c.dst_key.in_(sa.select(anchor.c.key)),
        )
        stmt = (
            sa.select(doc)
            .where(doc.c._key.in_(parents), *self._scope(doc, sc, tenant=False))
            .order_by(self._dl.order_value(doc.c.version, desc=True))
            .limit(1)
        )
        rec = await self._read_one(stmt)
        return self._document_refs([rec])[0] if rec else None

    async def get_document_ancestors(
        self,
        path: str,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentTree | None:
        """The chain from the topmost root down to *path*'s parent, as a one-branch tree."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        d = self._resolve_depth(depth)
        # user/job filters apply to the anchor only (the spine must stay whole).
        cte = walk(
            self._doc_start(path, version, sc, "scalar"),
            max_hops=d,
            direction="in",
            types=_COMPOSED,
            track_path=True,
        )
        e = T.edges
        is_root = ~sa.exists().where(
            e.c.type == "IS_COMPOSED_OF",
            e.c.dst_label == "Document",
            e.c.dst_key == cte.c.key,
        )
        rec = await self._read_one(
            sa.select(cte.c.path)
            .where(cte.c.depth >= 1, cte.c.label == "Document", is_root)
            .order_by(cte.c.depth.desc(), cte.c.path)
            .limit(1)
        )
        if not rec:
            return None
        # The walk goes d → … → root; the spine is root → … → parent of d.
        spine_keys = list(reversed(split_path(rec["path"])))[:-1]
        doc = T.documents
        rows = {
            r["_key"]: r for r in await self._read(sa.select(doc).where(doc.c._key.in_(spine_keys)))
        }
        spine = [rows[k] for k in spine_keys if k in rows]
        if not spine:
            return None
        head = DocumentTree(**self._document_refs([spine[0]])[0].model_dump(), depth=0)
        cur = head
        for i, row in enumerate(spine[1:], start=1):
            child = DocumentTree(**self._document_refs([row])[0].model_dump(), depth=i)
            cur.children.append(child)
            cur = child
        return head

    async def get_document_leaves(
        self,
        path: str,
        *,
        version: int | None = None,
        depth: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Descendants of *path* that contain no further document."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        d = self._resolve_depth(depth)
        doc = T.documents
        e = T.edges
        cte = walk(self._doc_start(path, version, sc), max_hops=d, types=_COMPOSED)
        below = reached(cte, "Document")
        has_child = sa.exists().where(
            e.c.type == "IS_COMPOSED_OF",
            e.c.src_label == "Document",
            e.c.src_key == doc.c._key,
            e.c.dst_label == "Document",
        )
        stmt = (
            sa.select(doc)
            .where(
                doc.c._key.in_(sa.select(below.c.key)),
                ~has_child,
                *self._scope(doc, sc, tenant=False),
            )
            .order_by(self._dl.order_text(doc.c.path))
        )
        return self._document_refs(await self._read(stmt))

    async def _versions_of(self, path: str, scope: Any) -> list[DocumentRef]:
        doc = T.documents
        stmt = (
            sa.select(doc)
            .where(doc.c.path == path, *self._scope(doc, scope))
            .order_by(self._dl.order_text(doc.c.tenant_id), self._dl.order_value(doc.c.version))
        )
        return self._document_refs(await self._read(stmt))

    async def list_document_versions(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Every version of *path* in scope, by tenant and version."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return await self._versions_of(path, sc)

    async def get_version_chain(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[DocumentRef]:
        """Every version of *path* in scope, by tenant and version (as in the Cypher)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        return await self._versions_of(path, sc)

    async def get_latest_version(
        self,
        path: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentRef | None:
        """The ``latest`` version of *path* (tenant resolved first)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, None, sc)
        doc = T.documents
        rec = await self._read_one(
            sa.select(doc)
            .where(doc.c.path == path, doc.c.latest == sa.true(), *self._scope(doc, sc))
            .order_by(
                self._dl.order_text(doc.c.tenant_id), self._dl.order_value(doc.c.version, desc=True)
            )
            .limit(1)
        )
        return self._document_refs([rec])[0] if rec else None

    async def get_document_stats(
        self,
        path: str,
        *,
        version: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> DocumentStats | None:
        """Counts of what hangs from the document.

        Everything below a document carries its ``_document_key``, so the per-table counts
        need no traversal; only the model instances (shared between documents) are reached
        by walking the ``HAS_*`` containment edges from the extraction results.
        """
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        sc = await self._single_document_scope(path, version, sc)
        cdepth = self._containment_depth(None)
        doc = T.documents
        # The anchor carries the whole scope; everything below hangs off it.
        anchor = await self._read_one(
            sa.select(doc.c._key, doc.c.version).where(
                *self._doc_conditions(path, version, sc, "scalar")
            )
        )
        if anchor is None:
            return None
        key = anchor["_key"]
        sn, iu, md, er = T.structure_nodes, T.info_units, T.model_decisions, T.extraction_results

        async def _count(table: sa.Table) -> int:
            rec = await self._read_one(
                sa.select(sa.func.count().label("c")).where(table.c._document_key == key)
            )
            return int((rec or {}).get("c", 0) or 0)

        roles = await self._read(
            sa.select(sn.c.role, sa.func.count().label("c"))
            .where(sn.c._document_key == key)
            .group_by(sn.c.role)
        )
        matched = await self._read_one(
            sa.select(sa.func.count().label("c")).where(
                md.c._document_key == key, md.c.matched_model_class.is_not(None)
            )
        )
        proposed = await self._read_one(
            sa.select(sa.func.count().label("c")).where(
                md.c._document_key == key, prop_equals(md, "propose_new_model", True, self._dl)
            )
        )
        er_keys = sa.select(
            sa.literal("ExtractionResult", sa.Text).label("label"), er.c.uid.label("key")
        ).where(er.c._document_key == key)
        mi = T.model_instances
        cte = walk(er_keys, max_hops=cdepth, prefix="HAS_", name="contained")
        instances = sa.select(cte.c.key).where(cte.c.depth >= 1, cte.c.label == "ModelInstance")
        classes = await self._read(
            sa.select(mi.c.model_class.label("mc"), sa.func.count().label("c"))
            .where(mi.c.uid.in_(instances))
            .group_by(mi.c.model_class)
        )
        n_mi = await self._read_one(
            sa.select(sa.func.count().label("c")).where(mi.c.uid.in_(instances))
        )
        e = T.edges
        le = T.labeled_entities
        referenced = (
            sa.select(e.c.dst_key)
            .where(
                e.c.type == "REFERENCES",
                e.c.src_label == "ModelInstance",
                e.c.dst_label == "LabeledEntity",
            )
            .where(e.c.src_key.in_(instances))
        )
        labels = await self._read(
            sa.select(le.c.label.label("lbl"), sa.func.count().label("c"))
            .where(le.c.uid.in_(referenced))
            .group_by(le.c.label)
        )
        triples = await self._read_one(self._triple_count_statement(key))
        core = {
            "version": anchor["version"],
            "n_nodes": await _count(sn),
            "n_units": await _count(iu),
            "n_dec": await _count(md),
            "n_matched": int((matched or {}).get("c", 0) or 0),
            "n_proposed": int((proposed or {}).get("c", 0) or 0),
            "n_er": await _count(er),
            "n_mi": int((n_mi or {}).get("c", 0) or 0),
        }
        return DocumentStats(
            raw=dict(core),
            path=path,
            version=int(core["version"]),
            structure_nodes=int(core["n_nodes"] or 0),
            structure_nodes_by_role={r["role"]: int(r["c"]) for r in roles if r["role"]},
            info_units=int(core["n_units"] or 0),
            model_decisions=int(core["n_dec"] or 0),
            model_decisions_matched=int(core["n_matched"] or 0),
            model_decisions_proposed=int(core["n_proposed"] or 0),
            extraction_results=int(core["n_er"] or 0),
            model_instances=int(core["n_mi"] or 0),
            model_instances_by_class={r["mc"]: int(r["c"]) for r in classes if r["mc"]},
            labeled_entities=sum(int(r["c"]) for r in labels),
            labeled_entities_by_label={r["lbl"]: int(r["c"]) for r in labels if r["lbl"]},
            triples=int(triples["c"]) if triples else 0,
        )

    def _triple_count_statement(self, document_key: str) -> sa.Select:
        """Count of ``(er)-[HAS_ENTITY subject]->(s)-[p]->(o)<-[HAS_ENTITY object]-(er)``."""
        er = T.extraction_results
        subject = T.edges.alias("subject")
        link = T.edges.alias("link")
        obj = T.edges.alias("object")
        role = self._dl.json_text
        return (
            sa.select(sa.func.count().label("c"))
            .select_from(
                er.join(
                    subject,
                    sa.and_(
                        subject.c.src_label == "ExtractionResult",
                        subject.c.src_key == er.c.uid,
                        subject.c.type == "HAS_ENTITY",
                        subject.c.dst_label == "Entity",
                        role(subject.c.props, "role") == "subject",
                    ),
                )
                .join(
                    link,
                    sa.and_(
                        link.c.src_label == "Entity",
                        link.c.src_key == subject.c.dst_key,
                        link.c.dst_label == "Entity",
                    ),
                )
                .join(
                    obj,
                    sa.and_(
                        obj.c.src_label == "ExtractionResult",
                        obj.c.src_key == er.c.uid,
                        obj.c.type == "HAS_ENTITY",
                        obj.c.dst_label == "Entity",
                        obj.c.dst_key == link.c.dst_key,
                        role(obj.c.props, "role") == "object",
                    ),
                )
            )
            .where(er.c._document_key == document_key, er.c.model_class == "Triple")
        )


__all__ = ["_DocumentsMixin"]
