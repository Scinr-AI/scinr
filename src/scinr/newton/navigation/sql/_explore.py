"""navigation/sql/_explore.py — Group J: memory summaries, explore & label keys (SQL).

The summaries and the record search aggregate in Python over the node rows (a record's
fields live in typed columns *and* in the ``props`` JSON, which has no portable "count the
keys" in SQL); the work that narrows the rows (scope, documents, relationships, ordering and
limits) is SQL. See ``navigation/_records.py`` for the definitions both backends share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa

from scinr.newton.exceptions import NavigationError
from scinr.newton.ingest.nodes import ROLE_TO_LABEL
from scinr.newton.navigation import _map
from scinr.newton.navigation._labels import resolve_label
from scinr.newton.navigation._records import (
    build_sources,
    class_summaries,
    field_keys,
    link_summaries,
    search_hits,
)
from scinr.newton.navigation._safe import safe_ident, safe_keys
from scinr.newton.navigation.filters import Contains
from scinr.newton.navigation.models import (
    ExtractionLink,
    ModelClassSummary,
    ModelInstanceLink,
    ModelInstanceMatch,
    ModelInstanceRef,
    ModelLinkSummary,
    SourceSummary,
)
from scinr.newton.navigation.scope import Scope, make_scope
from scinr.newton.navigation.sql._common import _SqlRuntime
from scinr.newton.navigation.sql._translate import translate_where
from scinr.newton.navigation.sql._walk import UNBOUNDED_HOPS, reached, walk
from scinr.newton.sql import tables as T
from scinr.newton.sql.names import LABEL_TABLE

_STRUCTURE_EDGES = ("HAS_STRUCTURE", "HAS_CHILD")
_ROLE_OF_LABEL = {label: role for role, label in ROLE_TO_LABEL.items()}


def _prefixed(table: Any, prefix: str) -> list[Any]:
    """Every column of *table* labelled ``<prefix><name>`` (two nodes in one row)."""
    return [column.label(f"{prefix}{column.name}") for column in table.c]


def _unprefixed(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix) :]: v for k, v in row.items() if k.startswith(prefix)}


def _paths(document_paths: Sequence[str] | None) -> list[str] | None:
    """Validated ``document_paths``: ``None`` (every document) or a non-empty list."""
    if document_paths is None:
        return None
    if isinstance(document_paths, (str, bytes)) or not all(
        isinstance(p, str) for p in document_paths
    ):
        raise NavigationError("document_paths must be a list of path strings or None.")
    if not document_paths:
        raise NavigationError("document_paths must not be an empty list (use None for all).")
    return list(document_paths)


class _ExploreMixin(_SqlRuntime):
    # -- shared pieces -------------------------------------------------------

    @staticmethod
    def _extracted(mi: Any) -> sa.ColumnElement[bool]:
        """``EXISTS { (:ExtractionResult)-->(mi) }``."""
        e = T.edges.alias("extracted_by")
        return sa.exists(
            sa.select(sa.literal(1)).where(
                e.c.src_label == "ExtractionResult",
                e.c.dst_label == "ModelInstance",
                e.c.dst_key == mi.c.uid,
            )
        )

    def _records_statement(
        self, sc: Scope, extra: Sequence[sa.ColumnElement[bool]] = ()
    ) -> sa.Select:
        """The records with a ``model_class`` in the scope, with their ``_extracted`` flag."""
        mi = T.model_instances
        return sa.select(mi, self._extracted(mi).label("_extracted")).where(
            mi.c.model_class.is_not(None), *self._scope(mi, sc, "array"), *extra
        )

    def _documents_start(self, paths: list[str] | None, sc: Scope) -> sa.Select:
        """``(label, key)`` of the ``latest`` documents (of *paths*, or all) in the scope."""
        doc = T.documents
        conditions = [doc.c.latest == sa.true(), *self._scope(doc, sc, "tenant_only")]
        if paths is not None:
            conditions.append(doc.c.path.in_(paths))
        return self._node_start("Document", doc.c._key, conditions)

    def _extraction_result_keys(self, paths: list[str] | None, sc: Scope, name: str) -> sa.Select:
        """Keys of the ``ExtractionResult`` nodes below the documents (any depth)."""
        e = T.edges
        structure = reached(
            walk(
                self._documents_start(paths, sc),
                max_hops=UNBOUNDED_HOPS,
                types=_STRUCTURE_EDGES,
                name=f"{name}_sn",
            ),
            "StructureNode",
            name=f"{name}_snr",
        )
        return sa.select(e.c.dst_key.label("key")).where(
            e.c.type == "HAS_EXTRACTION",
            e.c.src_label == "StructureNode",
            e.c.src_key.in_(sa.select(structure.c.key)),
            e.c.dst_label == "ExtractionResult",
        )

    def _extraction_triples(
        self,
        paths: list[str] | None,
        sc: Scope,
        field_contains: Sequence[tuple[str, str]],
        limit: int | None,
    ) -> sa.Subquery:
        """Distinct ``(s, t, d)`` = (extraction result uid, rel type, record uid), ordered."""
        er, mi, link = T.extraction_results, T.model_instances, T.edges.alias("link")
        conditions: list[sa.ColumnElement[bool]] = [
            link.c.src_label == "ExtractionResult",
            link.c.dst_label == "ModelInstance",
            link.c.src_key.in_(self._extraction_result_keys(paths, sc, "xl")),
            *self._scope(er, sc, "scalar", tenant=False),
            *self._scope(mi, sc, "array", tenant=False),
        ]
        for key, value in field_contains:
            conditions.append(
                sa.or_(
                    translate_where({key: Contains(value=value)}, er, self._dl)[0],
                    translate_where({key: Contains(value=value)}, mi, self._dl)[0],
                )
            )
        dl = self._dl
        stmt = (
            sa.select(
                link.c.src_key.label("s"), link.c.type.label("t"), link.c.dst_key.label("d")
            )
            .select_from(
                link.join(er, er.c.uid == link.c.src_key).join(mi, mi.c.uid == link.c.dst_key)
            )
            .where(*conditions)
            .group_by(link.c.src_key, link.c.type, link.c.dst_key)
            .order_by(
                dl.order_text(link.c.src_key), dl.order_text(link.c.type), dl.order_text(link.c.dst_key)
            )
        )
        return self._limit(stmt, limit).subquery("triples")

    @staticmethod
    def _check_field_contains(field_contains: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for item in field_contains:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise NavigationError("field_contains must be a list of (key, value) pairs.")
            key, value = item
            safe_ident(key, kind="field_contains key")
            if not isinstance(value, str):
                raise NavigationError(f"field_contains value for {key!r} must be a string.")
            pairs.append((key, value))
        return pairs

    # -- 2.2 summaries -----------------------------------------------------------

    async def summarize_model_classes(
        self,
        *,
        ignore_keys: Sequence[str] = (),
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelClassSummary]:
        """Records, empty records and placeholders of each ``model_class``."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        records = [
            (self._node("model_instances", row), bool(row["_extracted"]))
            for row in await self._read(self._records_statement(sc))
        ]
        selectable: dict[str, bool | None] = {}
        classes = sorted({node["model_class"] for node, _ in records})
        if classes:
            cm = T.catalog_models
            for row in await self._read(sa.select(cm).where(cm.c.name.in_(classes))):
                value = self._node("catalog_models", row).get("selectable")
                selectable[row["name"]] = None if value is None else bool(value)
        return class_summaries(records, selectable, ignore, limit)

    async def summarize_model_instance_links(
        self,
        *,
        ignore_keys: Sequence[str] = (),
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelLinkSummary]:
        """Relationships between records, aggregated by (from class, type, to class)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        a, b, e = T.model_instances.alias("a"), T.model_instances.alias("b"), T.edges
        stmt = (
            sa.select(
                e.c.type.label("_rel_type"),
                a.c.uid.label("_a_uid"),
                a.c.model_class.label("_a_class"),
                b,
                self._extracted(b).label("_extracted"),
            )
            .select_from(e.join(a, a.c.uid == e.c.src_key).join(b, b.c.uid == e.c.dst_key))
            .where(
                e.c.src_label == "ModelInstance",
                e.c.dst_label == "ModelInstance",
                a.c.model_class.is_not(None),
                b.c.model_class.is_not(None),
                *self._scope(a, sc, "array"),
                *self._scope(b, sc, "array"),
            )
        )
        links = [
            (
                row["_a_uid"],
                row["_a_class"],
                row["_rel_type"],
                self._node("model_instances", row),
                bool(row["_extracted"]),
            )
            for row in await self._read(stmt)
        ]
        return link_summaries(links, ignore, limit)

    async def summarize_sources(
        self,
        *,
        ignore_keys: Sequence[str] = (),
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[SourceSummary]:
        """The latest version of each source document and the records it brought."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        doc, latest = T.documents, T.documents.alias("latest")
        latest_rows = await self._read(
            sa.select(latest).where(
                latest.c.latest == sa.true(),
                sa.or_(latest.c.is_folder.is_(None), latest.c.is_folder == sa.false()),
                *self._scope(latest, sc, "scalar"),
            )
        )
        if not latest_rows:
            return []
        paths = sorted({r["path"] for r in latest_rows})
        versions: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in await self._read(sa.select(doc).where(doc.c.path.in_(paths))):
            versions.setdefault((row["tenant_id"], row["path"]), []).append(
                self._node("documents", row)
            )

        records = [
            (self._node("model_instances", row), bool(row["_extracted"]))
            for row in await self._read(self._records_statement(sc))
        ]
        latest = [self._node("documents", row) for row in latest_rows]
        return build_sources(latest, versions, records, ignore, limit)

    # -- 2.3 explore -------------------------------------------------------------

    async def get_model_instance_links(
        self,
        uids: Sequence[str],
        *,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceLink]:
        """The relationships between records that touch any of *uids*."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if isinstance(uids, (str, bytes)) or not all(isinstance(u, str) for u in uids):
            raise NavigationError("uids must be a list of uid strings.")
        wanted = sorted(set(uids))
        if not wanted:
            return []
        mi, e, dl = T.model_instances, T.edges, self._dl
        a, b = mi.alias("a"), mi.alias("b")
        triples = self._limit(
            sa.select(e.c.src_key.label("s"), e.c.type.label("t"), e.c.dst_key.label("d"))
            .select_from(e.join(a, a.c.uid == e.c.src_key).join(b, b.c.uid == e.c.dst_key))
            .where(
                e.c.src_label == "ModelInstance",
                e.c.dst_label == "ModelInstance",
                sa.or_(e.c.src_key.in_(wanted), e.c.dst_key.in_(wanted)),
                *self._scope(a, sc, "array"),
                *self._scope(b, sc, "array"),
            )
            .group_by(e.c.src_key, e.c.type, e.c.dst_key)
            .order_by(
                dl.order_text(e.c.src_key), dl.order_text(e.c.type), dl.order_text(e.c.dst_key)
            ),
            limit,
        ).subquery("triples")
        src, dst = mi.alias("src"), mi.alias("dst")
        stmt = (
            sa.select(triples.c.t.label("_rel_type"), *_prefixed(src, "s_"), *_prefixed(dst, "d_"))
            .select_from(
                triples.join(src, src.c.uid == triples.c.s).join(dst, dst.c.uid == triples.c.d)
            )
            .order_by(
                dl.order_text(triples.c.s), dl.order_text(triples.c.t), dl.order_text(triples.c.d)
            )
        )
        out: list[ModelInstanceLink] = []
        for row in await self._read(stmt):
            source = self._node("model_instances", _unprefixed(row, "s_"))
            target = self._node("model_instances", _unprefixed(row, "d_"))
            out.append(
                ModelInstanceLink(
                    source=_map.model_instance_ref(source, is_shell=await self._is_shell(source)),
                    rel_type=row["_rel_type"],
                    target=_map.model_instance_ref(target, is_shell=await self._is_shell(target)),
                )
            )
        return out

    async def get_extraction_links(
        self,
        *,
        document_paths: Sequence[str] | None = None,
        field_contains: Sequence[tuple[str, str]] = (),
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ExtractionLink]:
        """Extraction result → record relationships below some documents."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        paths = _paths(document_paths)
        pairs = self._check_field_contains(field_contains)
        mi, er, dl = T.model_instances, T.extraction_results, self._dl
        triples = self._extraction_triples(paths, sc, pairs, limit)
        src, dst = er.alias("src"), mi.alias("dst")
        stmt = (
            sa.select(triples.c.t.label("_rel_type"), *_prefixed(src, "s_"), *_prefixed(dst, "d_"))
            .select_from(
                triples.join(src, src.c.uid == triples.c.s).join(dst, dst.c.uid == triples.c.d)
            )
            .order_by(
                dl.order_text(triples.c.s), dl.order_text(triples.c.t), dl.order_text(triples.c.d)
            )
        )
        out: list[ExtractionLink] = []
        for row in await self._read(stmt):
            source = self._node("extraction_results", _unprefixed(row, "s_"))
            source.pop("embedding", None)
            target = self._node("model_instances", _unprefixed(row, "d_"))
            out.append(
                ExtractionLink(
                    source=_map.graph_node({**source, "_labels": ["ExtractionResult"]}),
                    rel_type=row["_rel_type"],
                    target=_map.model_instance_ref(target, is_shell=await self._is_shell(target)),
                )
            )
        return out

    async def list_extraction_field_keys(
        self,
        *,
        document_paths: Sequence[str] | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[str]:
        """Property keys of the extraction results and records below some documents."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        paths = _paths(document_paths)
        er, mi = T.extraction_results, T.model_instances
        triples = self._extraction_triples(paths, sc, [], None)
        keys: set[str] = set()
        for table, name, column in ((er, "extraction_results", "s"), (mi, "model_instances", "d")):
            rows = await self._read(
                sa.select(table).where(table.c.uid.in_(sa.select(triples.c[column])))
            )
            for row in rows:
                keys.update(k for k in self._node(name, row) if k != "_labels")
        return sorted(keys)

    async def get_model_instances_in_documents(
        self,
        model_class: str,
        *,
        document_paths: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceRef]:
        """Records of a class extracted from some documents, and their neighbours."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        paths = _paths(document_paths)
        mi, e = T.model_instances, T.edges
        link = e.alias("extracted")
        extracted = sa.select(link.c.dst_key.label("key")).where(
            link.c.src_label == "ExtractionResult",
            link.c.dst_label == "ModelInstance",
            link.c.src_key.in_(self._extraction_result_keys(paths, sc, "xi")),
        )
        out_edge, in_edge = e.alias("out_edge"), e.alias("in_edge")
        candidates = sa.union(
            extracted,
            sa.select(out_edge.c.dst_key.label("key")).where(
                out_edge.c.src_label == "ModelInstance",
                out_edge.c.dst_label == "ModelInstance",
                out_edge.c.src_key.in_(extracted),
            ),
            sa.select(in_edge.c.src_key.label("key")).where(
                in_edge.c.src_label == "ModelInstance",
                in_edge.c.dst_label == "ModelInstance",
                in_edge.c.dst_key.in_(extracted),
            ),
        ).subquery("candidates")
        stmt = (
            sa.select(mi)
            .where(
                mi.c.uid.in_(sa.select(candidates.c.key)),
                mi.c.model_class == model_class,
                *translate_where(where, mi, self._dl),
                *self._scope(mi, sc, "array"),
            )
            .order_by(self._dl.order_text(mi.c.uid))
        )
        return await self._mi_refs(await self._read(self._limit(stmt, limit)))

    async def search_model_instances(
        self,
        text: str,
        *,
        ignore_keys: Sequence[str] = (),
        limit: int | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ModelInstanceMatch]:
        """Records whose uid is *text* or with a field containing it (case insensitive)."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if not isinstance(text, str) or not text:
            raise NavigationError("text must be a non-empty string.")
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        mi = T.model_instances
        rows = await self._read(
            sa.select(mi).where(mi.c.model_class.is_not(None), *self._scope(mi, sc, "array"))
        )
        by_uid = {row["uid"]: row for row in rows}
        hits = search_hits((self._node("model_instances", r) for r in rows), text, ignore, limit)
        refs = await self._mi_refs([by_uid[node["uid"]] for node, _ in hits])
        return [ModelInstanceMatch(instance=ref, field=field) for ref, (_, field) in zip(refs, hits, strict=True)]

    async def list_model_instance_field_keys(
        self,
        model_class: str,
        *,
        ignore_keys: Sequence[str] = (),
        sample_limit: int = 200,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[str]:
        """Field keys with a value in the first *sample_limit* records of a class."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit < 0:
            raise NavigationError(f"sample_limit must be an integer >= 0, got {sample_limit!r}.")
        mi = T.model_instances
        rows = await self._read(
            sa.select(mi)
            .where(mi.c.model_class == model_class, *self._scope(mi, sc, "array"))
            .order_by(self._dl.order_text(mi.c.uid))
            .limit(sample_limit)
        )
        return field_keys((self._node("model_instances", row) for row in rows), ignore)

    # -- 2.4 label keys ------------------------------------------------------------

    async def get_node_property_keys(
        self,
        label: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[str]:
        """Property keys of the first node of a label inside the scope."""
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        info = resolve_label(label)
        name = LABEL_TABLE[info.primary]
        table = T.metadata.tables[name]
        conditions: list[sa.ColumnElement[bool]] = []
        if info.label != info.primary:
            conditions.append(table.c.role == _ROLE_OF_LABEL[info.label])
        if info.scope_kind is not None:
            conditions.extend(self._scope(table, sc, info.scope_kind))
        from scinr.newton.sql.names import KEY_COLUMN

        row = await self._read_one(
            sa.select(table)
            .where(*conditions)
            .order_by(
                self._dl.order_text(table.c[info.key]),
                self._dl.order_text(table.c[KEY_COLUMN[info.primary]]),
            )
            .limit(1)
        )
        if row is None:
            return []
        return sorted(k for k in self._node(name, row) if k != "_labels")


__all__ = ["_ExploreMixin"]
