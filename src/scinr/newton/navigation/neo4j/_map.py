"""
navigation/neo4j/_map.py — Neo4j record dict → engine-neutral model mapping.

Queries return nodes as map projections (``n { .* }``, plus
``_labels: labels(n)`` where the label set matters). These helpers turn those
plain dicts into ``navigation.models`` instances. Unknown / extra keys are kept
on ``raw``.
"""

from __future__ import annotations

from typing import Any

from scinr.newton.navigation.models import (
    CatalogFieldRef,
    CatalogModelRef,
    CatalogRelation,
    DocumentRef,
    EntityLabelStat,
    ExtractionResultRef,
    GraphNode,
    InfoUnitRef,
    LabeledEntityRef,
    ModelClassStat,
    ModelDecisionRef,
    ModelInstanceRef,
    ProposedFieldRef,
    ProposedModelRef,
    RelTypeStat,
    RoleStat,
    ScoredInfoUnit,
    StructureNodeRef,
    ThemeRef,
    Triple,
)

_INTERNAL = ("_labels", "_elementId", "_id")


def _clean(props: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if k not in _INTERNAL}


def document_ref(n: dict[str, Any]) -> DocumentRef:
    return DocumentRef(
        raw=_clean(n),
        path=n["path"],
        name=n.get("name", n["path"]),
        version=int(n["version"]),
        latest=bool(n.get("latest", False)),
        is_folder=bool(n.get("is_folder", False)),
        raw_file_id=n.get("raw_file_id"),
        load_date=n.get("load_date"),
        tenant_id=n.get("tenant_id"),
        created_by_user_id=n.get("created_by_user_id"),
        job_id=n.get("job_id"),
        context_instructions=n.get("context_instructions"),
    )


def structure_node_ref(
    n: dict[str, Any],
    *,
    document_path: str | None = None,
    document_version: int | None = None,
) -> StructureNodeRef:
    labels = [str(x) for x in (n.get("_labels") or [])]
    return StructureNodeRef(
        raw=_clean(n),
        id=n["id"],
        node_id=n.get("node_id", ""),
        title=n.get("title"),
        role=n.get("role", ""),
        types=labels,
        appearance_order=int(n.get("appearance_order") or 0),
        theme=n.get("theme"),
        source_page_ids=list(n.get("source_page_ids") or []),
        row_index=n.get("row_index"),
        document_path=document_path,
        document_version=document_version,
    )


def info_unit_ref(n: dict[str, Any]) -> InfoUnitRef:
    return InfoUnitRef(
        raw=_clean(n),
        uid=n["uid"],
        info_unit_id=n.get("info_unit_id"),
        title=n.get("title", ""),
        description=n.get("description", ""),
        order=int(n.get("order") or 0),
    )


def scored_info_unit(
    n: dict[str, Any], *, score: float, node_id: str, node_title: str | None
) -> ScoredInfoUnit:
    return ScoredInfoUnit(
        raw=_clean(n),
        uid=n["uid"],
        info_unit_id=n.get("info_unit_id"),
        title=n.get("title", ""),
        description=n.get("description", ""),
        order=int(n.get("order") or 0),
        node_id=node_id,
        node_title=node_title,
        score=float(score),
    )


def model_decision_ref(
    n: dict[str, Any],
    *,
    matched_model: str | None = None,
    complementary_models: list[str] | None = None,
    supplementary_fields: list[str] | None = None,
) -> ModelDecisionRef:
    gaps = n.get("coverage_gaps")
    if isinstance(gaps, str):
        gaps = [gaps]
    return ModelDecisionRef(
        raw=_clean(n),
        uid=n["uid"],
        matched_model_class=n.get("matched_model_class"),
        confidence=None if n.get("confidence") is None else str(n.get("confidence")),
        rationale=n.get("rationale"),
        coverage_gaps=list(gaps or []),
        propose_new_model=n.get("propose_new_model"),
        proposed_model_description=n.get("proposed_model_description"),
        document_name=n.get("document_name"),
        timestamp=n.get("timestamp"),
        source=n.get("source"),
        matched_model=matched_model,
        complementary_models=list(complementary_models or []),
        supplementary_fields=list(supplementary_fields or []),
    )


def proposed_model_ref(
    n: dict[str, Any], *, fields: list[dict[str, Any]] | None = None, node_id: str | None = None
) -> ProposedModelRef:
    return ProposedModelRef(
        raw=_clean(n),
        uid=n["uid"],
        schema_name=n.get("schema_name") or n.get("name"),
        description=n.get("description"),
        fields=[proposed_field_ref(f) for f in (fields or []) if f],
        node_id=node_id,
    )


def proposed_field_ref(f: dict[str, Any]) -> ProposedFieldRef:
    return ProposedFieldRef(
        raw=_clean(f),
        field_name=f.get("field_name") or f.get("name") or "",
        field_type=f.get("field_type") or f.get("type"),
        description=f.get("description"),
        required=f.get("required"),
    )


def extraction_result_ref(
    n: dict[str, Any],
    *,
    primary_model: str | None = None,
    complementary_models: list[str] | None = None,
) -> ExtractionResultRef:
    mc = n.get("model_class", "")
    return ExtractionResultRef(
        raw=_clean(n),
        uid=n["uid"],
        node_full_id=n.get("node_full_id"),
        model_class=mc,
        document_name=n.get("document_name"),
        timestamp=n.get("timestamp"),
        is_triple=(mc == "Triple"),
        primary_model=primary_model,
        complementary_models=list(complementary_models or []),
    )


def model_instance_ref(
    n: dict[str, Any],
    *,
    via_rel: str | None = None,
    direction: str | None = None,
    index: int | None = None,
    is_shell: bool | None = None,
) -> ModelInstanceRef:
    props = {k: v for k, v in n.items() if k not in ("uid", "model_class", *_INTERNAL)}
    return ModelInstanceRef(
        raw=_clean(n),
        uid=n["uid"],
        model_class=n.get("model_class", ""),
        properties=props,
        is_shell=is_shell,
        via_rel=via_rel,
        direction=direction,  # type: ignore[arg-type]
        index=index,
    )


def labeled_entity_ref(
    n: dict[str, Any], *, field_name: str | None = None, list_index: int | None = None
) -> LabeledEntityRef:
    return LabeledEntityRef(
        raw=_clean(n),
        uid=n["uid"],
        label=n.get("label", ""),
        value=n.get("value", ""),
        normalized_value=n.get("normalized_value", ""),
        field_name=field_name,
        list_index=list_index,
    )


def catalog_model_ref(
    n: dict[str, Any],
    *,
    fields: list[dict[str, Any]] | None = None,
    themes: list[str] | None = None,
) -> CatalogModelRef:
    return CatalogModelRef(
        raw=_clean(n),
        name=n.get("name", ""),
        description=n.get("description"),
        selectable=n.get("selectable"),
        fields=[
            CatalogFieldRef(
                raw=_clean(f),
                name=f.get("name", ""),
                type=f.get("type"),
                entity_label=f.get("entity_label"),
                is_instance_key=f.get("is_instance_key"),
                required=f.get("required"),
                description=f.get("description"),
            )
            for f in (fields or [])
            if f and f.get("name")
        ],
        themes=[t for t in (themes or []) if t],
    )


def catalog_relation(r: dict[str, Any]) -> CatalogRelation:
    return CatalogRelation(
        raw=_clean(r),
        source=r["source"],
        target=r["target"],
        rel_type=r["rel_type"],
        source_kind=r.get("source_kind", "CatalogModel"),
        target_kind=r.get("target_kind", "CatalogModel"),
        properties=dict(r.get("props") or {}),
    )


def theme_ref(n: dict[str, Any]) -> ThemeRef:
    return ThemeRef(raw=_clean(n), name=n.get("name", ""), path=n.get("path"))


def triple(row: dict[str, Any], *, node_id: str | None = None) -> Triple:
    return Triple(
        raw=dict(row),
        subject=row.get("subject", ""),
        predicate=row.get("predicate"),
        object=row.get("object"),
        predicate_raw=row.get("predicate_raw"),
        node_id=node_id,
    )


def graph_node(n: dict[str, Any]) -> GraphNode:
    labels = [str(x) for x in (n.get("_labels") or [])]
    props = {k: v for k, v in n.items() if k not in _INTERNAL}
    return GraphNode(raw=_clean(n), types=labels, properties=props)


def model_class_stat(row: dict[str, Any], *, kind: str | None = None) -> ModelClassStat:
    return ModelClassStat(
        raw=dict(row),
        model_class=row.get("model_class") or row.get("name") or "",
        count=int(row.get("count", 0) or 0),
        kind=kind,  # type: ignore[arg-type]
    )


def role_stat(row: dict[str, Any]) -> RoleStat:
    return RoleStat(raw=dict(row), role=row.get("role", ""), count=int(row.get("count", 0) or 0))


def entity_label_stat(row: dict[str, Any]) -> EntityLabelStat:
    return EntityLabelStat(
        raw=dict(row), label=row.get("label", ""), count=int(row.get("count", 0) or 0)
    )


def rel_type_stat(row: dict[str, Any]) -> RelTypeStat:
    return RelTypeStat(
        raw=dict(row),
        source_model=row.get("source_model"),
        rel_type=row.get("rel_type", ""),
        target_model=row.get("target_model"),
        count=int(row.get("count", 0) or 0),
    )
