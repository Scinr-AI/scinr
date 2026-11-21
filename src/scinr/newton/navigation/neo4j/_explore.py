"""navigation/neo4j/_explore.py — Group J: memory summaries, explore & label keys (Cypher).

Cypher narrows the rows (scope, documents, relationships, ordering and limits); the definitions
of *field*, *empty* and *placeholder* and the aggregation are the engine-neutral ones of
``navigation/_records.py``, shared with the SQL backend so both count alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation import _map
from scinr.newton.navigation._labels import resolve_label
from scinr.newton.navigation._records import (
    build_sources,
    class_summaries,
    field_keys,
    link_summaries,
    search_hits,
)
from scinr.newton.navigation.models import (
    ExtractionLink,
    ModelClassSummary,
    ModelInstanceLink,
    ModelInstanceMatch,
    ModelInstanceRef,
    ModelLinkSummary,
    SourceSummary,
)
from scinr.newton.navigation.neo4j._common import _Neo4jRuntime
from scinr.newton.navigation.neo4j._safe import safe_ident, safe_keys
from scinr.newton.navigation.neo4j._translate import translate_where
from scinr.newton.navigation.scope import make_scope

_EXTRACTED = "EXISTS { (:ExtractionResult)-->(%s) }"


def _where(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)} " if clauses else ""


def _paths(document_paths: Sequence[str] | None) -> list[str] | None:
    if document_paths is None:
        return None
    if isinstance(document_paths, (str, bytes)) or not all(
        isinstance(p, str) for p in document_paths
    ):
        raise NavigationError("document_paths must be a list of path strings or None.")
    if not document_paths:
        raise NavigationError("document_paths must not be an empty list (use None for all).")
    return list(document_paths)


def _pairs(field_contains: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
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


class _ExploreMixin(_Neo4jRuntime):
    async def _records(self, sc: Any) -> list[tuple[dict[str, Any], bool]]:
        params: dict[str, Any] = {}
        clauses = ["mi.model_class IS NOT NULL"]
        self._scope_where("mi", clauses, params, sc, "array")
        rows = await self._read(
            f"MATCH (mi:ModelInstance) {_where(clauses)}"
            f"RETURN mi, {_EXTRACTED % 'mi'} AS extracted",
            **params,
        )
        return [(r["mi"], bool(r["extracted"])) for r in rows]

    def _extraction_match(
        self, sc: Any, paths: list[str] | None, pairs: list[tuple[str, str]], params: dict[str, Any]
    ) -> str:
        """``MATCH`` of document -> structure -> extraction result -> record, with its filters."""
        doc_clauses = ["d.latest = true"]
        self._scope_where("d", doc_clauses, params, sc, "tenant_only")
        if paths is not None:
            doc_clauses.append("d.path IN $document_paths")
            params["document_paths"] = paths
        clauses: list[str] = []
        self._scope_where("er", clauses, params, sc, "scalar", tenant=False)
        self._scope_where("mi", clauses, params, sc, "array", tenant=False)
        for i, (key, value) in enumerate(pairs):
            params[f"fk{i}"] = key
            params[f"fv{i}"] = value
            clauses.append(f"(er[$fk{i}] CONTAINS $fv{i} OR mi[$fk{i}] CONTAINS $fv{i})")
        return (
            f"MATCH (d:Document) {_where(doc_clauses)}"
            "MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(er:ExtractionResult)-[r]->(mi:ModelInstance) "
            f"{_where(clauses)}"
        )

    # -- summaries --------------------------------------------------------------

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        records = await self._records(sc)
        selectable: dict[str, bool | None] = {}
        classes = sorted({node["model_class"] for node, _ in records})
        if classes:
            for r in await self._read(
                "MATCH (cm:CatalogModel) WHERE cm.name IN $classes "
                "RETURN cm.name AS name, cm.selectable AS selectable",
                classes=classes,
            ):
                selectable[r["name"]] = None if r["selectable"] is None else bool(r["selectable"])
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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        params: dict[str, Any] = {}
        clauses = ["a.model_class IS NOT NULL", "b.model_class IS NOT NULL"]
        self._scope_where("a", clauses, params, sc, "array")
        self._scope_where("b", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH (a:ModelInstance)-[r]->(b:ModelInstance) {_where(clauses)}"
            "RETURN a.uid AS from_uid, a.model_class AS from_class, type(r) AS rel_type, b, "
            f"{_EXTRACTED % 'b'} AS extracted",
            **params,
        )
        links = [
            (r["from_uid"], r["from_class"], r["rel_type"], r["b"], bool(r["extracted"]))
            for r in rows
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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        params: dict[str, Any] = {}
        clauses = ["d.latest = true", "coalesce(d.is_folder, false) = false"]
        self._scope_where("d", clauses, params, sc, "scalar")
        latest_rows = await self._read(f"MATCH (d:Document) {_where(clauses)}RETURN d", **params)
        if not latest_rows:
            return []
        latest = [r["d"] for r in latest_rows]
        versions: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in await self._read(
            "MATCH (d:Document) WHERE d.path IN $paths RETURN d",
            paths=sorted({n["path"] for n in latest}),
        ):
            versions.setdefault((r["d"].get("tenant_id"), r["d"]["path"]), []).append(r["d"])
        return build_sources(latest, versions, await self._records(sc), ignore, limit)

    # -- explore ------------------------------------------------------------------

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if isinstance(uids, (str, bytes)) or not all(isinstance(u, str) for u in uids):
            raise NavigationError("uids must be a list of uid strings.")
        if not uids:
            return []
        params: dict[str, Any] = {"uids": sorted(set(uids))}
        clauses = ["(a.uid IN $uids OR b.uid IN $uids)"]
        self._scope_where("a", clauses, params, sc, "array")
        self._scope_where("b", clauses, params, sc, "array", tenant=False)
        rows = await self._read(
            f"MATCH (a:ModelInstance)-[r]->(b:ModelInstance) {_where(clauses)}"
            "RETURN DISTINCT a, type(r) AS rel_type, b "
            f"ORDER BY a.uid, type(r), b.uid{self._limit_clause(limit)}",
            **params,
        )
        return [
            ModelInstanceLink(
                source=_map.model_instance_ref(r["a"], is_shell=await self._is_shell(r["a"])),
                rel_type=r["rel_type"],
                target=_map.model_instance_ref(r["b"], is_shell=await self._is_shell(r["b"])),
            )
            for r in rows
        ]

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {}
        match = self._extraction_match(sc, _paths(document_paths), _pairs(field_contains), params)
        rows = await self._read(
            f"{match}RETURN DISTINCT er {{.*, embedding: null}} AS er, type(r) AS rel_type, mi "
            f"ORDER BY er.uid, type(r), mi.uid{self._limit_clause(limit)}",
            **params,
        )
        out: list[ExtractionLink] = []
        for r in rows:
            er = {k: v for k, v in r["er"].items() if k != "embedding"}
            out.append(
                ExtractionLink(
                    source=_map.graph_node({**er, "_labels": ["ExtractionResult"]}),
                    rel_type=r["rel_type"],
                    target=_map.model_instance_ref(r["mi"], is_shell=await self._is_shell(r["mi"])),
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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        params: dict[str, Any] = {}
        match = self._extraction_match(sc, _paths(document_paths), [], params)
        rows = await self._read(
            f"{match}WITH collect(DISTINCT er) AS ers, collect(DISTINCT mi) AS mis "
            "UNWIND ers + mis AS n UNWIND keys(n) AS k RETURN DISTINCT k ORDER BY k",
            **params,
        )
        return [r["k"] for r in rows]

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        paths = _paths(document_paths)
        params: dict[str, Any] = {"model_class": model_class}
        doc_clauses = ["d.latest = true"]
        self._scope_where("d", doc_clauses, params, sc, "tenant_only")
        if paths is not None:
            doc_clauses.append("d.path IN $document_paths")
            params["document_paths"] = paths
        clauses = ["mi.model_class = $model_class"]
        self._scope_where("mi", clauses, params, sc, "array")
        if where:
            frag, wp = translate_where(where, alias="mi")
            clauses.append(frag)
            params.update(wp)
        rows = await self._read(
            f"MATCH (d:Document) {_where(doc_clauses)}"
            "MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)"
            "-[:HAS_EXTRACTION]->(:ExtractionResult)-->(x:ModelInstance) "
            "WITH DISTINCT x OPTIONAL MATCH (x)--(n:ModelInstance) "
            "WITH collect(DISTINCT x) + collect(DISTINCT n) AS candidates "
            "UNWIND candidates AS mi WITH DISTINCT mi "
            f"{_where(clauses)}"
            f"RETURN mi ORDER BY mi.uid{self._limit_clause(limit)}",
            **params,
        )
        return await self._mi_refs(rows)

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        if not isinstance(text, str) or not text:
            raise NavigationError("text must be a non-empty string.")
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        records = await self._records(sc)
        hits = search_hits((node for node, _ in records), text, ignore, limit)
        refs = await self._mi_refs([{"mi": node} for node, _ in hits])
        return [
            ModelInstanceMatch(instance=ref, field=field)
            for ref, (_, field) in zip(refs, hits, strict=True)
        ]

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
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        ignore = safe_keys(ignore_keys, kind="ignore_keys key")
        if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit < 0:
            raise NavigationError(f"sample_limit must be an integer >= 0, got {sample_limit!r}.")
        params: dict[str, Any] = {"model_class": model_class}
        clauses = ["mi.model_class = $model_class"]
        self._scope_where("mi", clauses, params, sc, "array")
        rows = await self._read(
            f"MATCH (mi:ModelInstance) {_where(clauses)}"
            f"RETURN mi ORDER BY mi.uid LIMIT {int(sample_limit)}",
            **params,
        )
        return field_keys((r["mi"] for r in rows), ignore)

    async def get_node_property_keys(
        self,
        label: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[str]:
        sc = make_scope(tenant_id, include_public, created_by_user_id, job_id)
        info = resolve_label(label)
        params: dict[str, Any] = {}
        clauses: list[str] = []
        if info.scope_kind is not None:
            self._scope_where("n", clauses, params, sc, info.scope_kind)
        rows = await self._read(
            f"MATCH (n:`{info.label}`) {_where(clauses)}"
            f"WITH n ORDER BY n.`{info.key}` LIMIT 1 RETURN keys(n) AS ks",
            **params,
        )
        return sorted(rows[0]["ks"]) if rows else []


__all__ = ["_ExploreMixin"]
