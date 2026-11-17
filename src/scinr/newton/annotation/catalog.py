"""
annotation/catalog.py — Domain side of the model catalog: no Cypher, no engine.

Introspection of the Pydantic models (which models a registry reaches, how a
field type reads, which fields reference other models, which relationship
types are safe), the traversal that produces the catalog and the theme
structure (:func:`ensure_catalog_models`, :func:`ensure_theme_structure`) and
the process-level memoization of both. The traversal calls a
:class:`~scinr.newton.graph.sinks.CatalogSink` once per statement; the sink is
what ``store.annotation.catalog_sink()`` opens (see :mod:`scinr.newton.graph.base`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import typing
from typing import TYPE_CHECKING

from scinr.newton.config import get_config
from scinr.newton.graph import get_graph_store

if TYPE_CHECKING:
    from scinr.newton.graph.sinks import CatalogSink
    from scinr.newton.utils.theme_registry import ThemeRegistry

log = logging.getLogger(__name__)


def _field_type_str(annotation: object) -> str:
    """Convert a type annotation to a readable string for storage."""
    if annotation is None:
        return "Any"
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        args = typing.get_args(annotation)
        return _field_type_str(args[0]) if args else "Any"
    if origin is not None:
        return str(annotation).replace("typing.", "")
    return getattr(annotation, "__name__", str(annotation))


def _extract_model_refs(annotation: object) -> list[tuple[type, bool]]:
    """
    Extract all Pydantic BaseModel subclass references from a type annotation.

    Returns a list of (model_class, is_list) tuples.
    Handles: Model | None, list[Model], Annotated[Union[A, B], ...], direct references.
    Does NOT recurse into list[list[X]].
    """
    import types as _builtin_types

    from pydantic import BaseModel

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Annotated[X, ...] — unwrap to first arg
    if origin is typing.Annotated:
        return _extract_model_refs(args[0]) if args else []

    # list[X] — mark results as is_list=True
    if origin is list:
        if not args:
            return []
        return [(cls, True) for cls, _ in _extract_model_refs(args[0])]

    # typing.Union[X, Y] (covers Optional[X]) OR Python 3.10+ X | Y syntax
    is_union = origin is typing.Union
    if not is_union and hasattr(_builtin_types, "UnionType"):
        is_union = isinstance(annotation, _builtin_types.UnionType)
    if is_union:
        results: list[tuple[type, bool]] = []
        for arg in args:
            if arg is type(None):
                continue
            results.extend(_extract_model_refs(arg))
        return results

    # Direct Pydantic model class reference
    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [(annotation, False)]
    except TypeError:
        pass

    return []


def _collect_all_model_classes(seed_classes: dict[str, type]) -> dict[str, type]:
    """
    BFS discovery of all Pydantic BaseModel subclasses reachable from seed_classes
    via their model fields. Returns the complete set including seeds.
    """
    all_models: dict[str, type] = dict(seed_classes)
    queue: list[type] = list(seed_classes.values())

    while queue:
        cls = queue.pop()
        if not hasattr(cls, "model_fields"):
            continue
        for field_info in cls.model_fields.values():
            for ref_cls, _ in _extract_model_refs(field_info.annotation):
                name = ref_cls.__name__
                if name not in all_models:
                    all_models[name] = ref_cls
                    queue.append(ref_cls)

    return all_models


_VALID_REL_TYPE_RE = _re.compile(r'^[A-Z][A-Z0-9_]*$')


def _validate_rel_type(rel_type: str, context: str) -> bool:
    """Return True if rel_type is a valid Neo4j relationship type identifier.

    Valid: SCREAMING_SNAKE_CASE starting with a letter (e.g. HAS_PROCEDURE_TYPE).
    Logs a warning and returns False if invalid, preventing unsafe Cypher interpolation.
    """
    if _VALID_REL_TYPE_RE.match(rel_type):
        return True
    log.warning(
        "ensure_catalog_models: invalid rel_type %r in %s — "
        "must match [A-Z][A-Z0-9_]*, skipping",
        rel_type, context,
    )
    return False


async def ensure_catalog_models(registry: ThemeRegistry, sink: CatalogSink) -> None:
    """
    Write a (:CatalogModel) singleton for every Pydantic model reachable from the
    registered themes and the pharmaceutical model hierarchy.

    Writes, through *sink*:
      (:CatalogModel {name, description, selectable})  — one per unique model class
      (:CatalogModel)-[:HAS_FIELD]->(:ModelField)       — one per Pydantic field
      (:CatalogModel)-[:AGGREGATES {field_name, is_list, required}]->(:CatalogModel)
                                                        — when a field references another model

    selectable=True  : model appears in a theme's SELECTABLE_MODELS list (valid annotation target)
    selectable=False : sub-model or aggregator, not directly selectable for annotation

    The instance-key indexes are requested in the same pass (best effort: the
    sink logs a failed index and does not raise).

    Idempotent. Call once at agent startup (see :func:`ensure_catalog_models_once`).

    Parameters
    ----------
    registry:
        The theme registry whose models seed the catalog.
    sink:
        Open catalog sink; every statement is emitted as the traversal reaches it.
    """
    from scinr.newton.utils.theme_registry import ThemeRegistry

    # ── 1. Collect selectable model names and seed set from all themes ────────
    selectable_names: set[str] = set()
    seed_classes: dict[str, type] = {}
    for theme_node in registry._themes.values():
        for cls in theme_node.models:
            selectable_names.add(cls.__name__)
            seed_classes[cls.__name__] = cls

    # ── 2. Recursively discover every referenced Pydantic model via BFS ───────
    all_models = _collect_all_model_classes(seed_classes)

    field_count = 0
    agg_count = 0

    # ── 4. Create / update CatalogModel nodes ─────────────────────────────────
    for model_name, cls in all_models.items():
        description = ThemeRegistry._get_docstring_description(cls, full_docstring=get_config().full_docstring)
        selectable = model_name in selectable_names
        await sink.merge_catalog_model(model_name, description, selectable)

    # ── 5. Create HAS_FIELD relationships ─────────────────────────────────────
    for model_name, cls in all_models.items():
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            field_type = _field_type_str(field_info.annotation)
            field_description = field_info.description or ""
            field_required = field_info.is_required()
            await sink.merge_model_field(model_name, field_name, field_type, field_description, field_required)
            field_count += 1

    # ── 6. Create AGGREGATES relationships between CatalogModel nodes ──────────
    for model_name, cls in all_models.items():
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            refs = _extract_model_refs(field_info.annotation)
            if not refs:
                continue
            required = field_info.is_required()
            for ref_cls, is_list in refs:
                ref_name = ref_cls.__name__
                if ref_name not in all_models:
                    continue  # Target not in the known model graph
                await sink.merge_aggregates(model_name, ref_name, field_name, is_list, required)
                agg_count += 1

    # Pass A: Enrich ModelField nodes with json_schema_extra metadata
    entity_label_count = 0
    instance_key_count = 0
    instance_key_props: set[str] = set()
    for model_name, cls in all_models.items():
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if not isinstance(extra, dict):
                continue
            is_instance_key = bool(extra.get("instance_key", False))
            entity_label = extra.get("entity_label")  # str or None
            if not is_instance_key and entity_label is None:
                continue  # Nothing to enrich
            await sink.set_field_metadata(model_name, field_name, is_instance_key, entity_label)
            if is_instance_key:
                instance_key_count += 1
                instance_key_props.add(field_name)
            if entity_label is not None:
                entity_label_count += 1

    # Pass A': one (tenant_id, <key>) index per distinct instance_key property
    await sink.ensure_instance_key_indexes(instance_key_props)

    # Pass B: Create EntityLabel schema nodes and PRODUCES_ENTITY relationships
    produces_entity_count = 0
    for model_name, cls in all_models.items():
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if not isinstance(extra, dict):
                continue
            entity_label = extra.get("entity_label")
            if entity_label is None:
                continue
            await sink.link_produces_entity(model_name, field_name, entity_label)
            produces_entity_count += 1

    # Pass C: Create schema-level EntityLabel→EntityLabel edges from field_relationships
    field_rel_count = 0
    for model_name, cls in all_models.items():
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if not isinstance(extra, dict):
                continue
            field_rels = extra.get("field_relationships")
            if not field_rels:
                continue
            source_entity_label = extra.get("entity_label")
            if source_entity_label is None:
                continue  # source field must have entity_label
            for rel in field_rels:
                to_field_name = rel.get("to_field")
                rel_type = rel.get("rel_type")
                if not to_field_name or not rel_type:
                    continue
                # Resolve entity_label of the to_field on the same model
                to_field_info = cls.model_fields.get(to_field_name)
                if to_field_info is None:
                    continue
                to_extra = to_field_info.json_schema_extra or {}
                if not isinstance(to_extra, dict):
                    continue
                target_entity_label = to_extra.get("entity_label")
                if target_entity_label is None:
                    continue  # to_field must also have entity_label
                if not _validate_rel_type(rel_type, f"{model_name}.{field_name}.field_relationships"):
                    continue
                await sink.merge_field_relationship(
                    rel_type, source_entity_label, target_entity_label, model_name, field_name, to_field_name
                )
                field_rel_count += 1

    # Pass D: Create schema-level CatalogModel→CatalogModel edges from instance_relationships
    instance_rel_count = 0
    for model_name, cls in all_models.items():
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if not isinstance(extra, dict):
                continue
            instance_rels = extra.get("instance_relationships")
            if not instance_rels:
                continue
            for rel in instance_rels:
                target_model = rel.get("target_model")
                join_via = rel.get("join_via", {})
                rel_type = rel.get("rel_type")
                if not target_model or not rel_type:
                    continue
                if target_model not in all_models:
                    log.warning(
                        "ensure_catalog_models: instance_relationship on "
                        "%s.%s references unknown target_model=%r — skipping",
                        model_name, field_name, target_model,
                    )
                    continue
                is_composite = len(join_via) > 1
                join_via_json = json.dumps(join_via, sort_keys=True)
                if not _validate_rel_type(rel_type, f"{model_name}.{field_name}.instance_relationships"):
                    continue
                await sink.merge_instance_relationship(
                    rel_type, model_name, target_model, field_name, join_via_json, is_composite
                )
                instance_rel_count += 1

    log.info(
        "ensure_catalog_models: %d CatalogModel nodes (%d selectable), "
        "%d HAS_FIELD, %d AGGREGATES, %d PRODUCES_ENTITY relationships, "
        "%d ModelField nodes with entity_label, %d instance_key fields, "
        "%d field_relationship edges, %d instance_relationship edges",
        len(all_models),
        len(selectable_names),
        field_count,
        agg_count,
        produces_entity_count,
        entity_label_count,
        instance_key_count,
        field_rel_count,
        instance_rel_count,
    )


async def ensure_theme_structure(registry: ThemeRegistry, sink: CatalogSink) -> None:
    """
    Write the (:Theme) nodes mirroring the models/ folder hierarchy, through *sink*.

    Writes:
      - (:Theme {path, name}) for each discovered theme folder
      - (:Theme)-[:HAS_SUBTOPIC]->(:Theme) for nested theme relationships
      - (:CatalogModel)-[:BELONGS_TO_THEME]->(:Theme) for each model-to-theme mapping

    Idempotent. Call once at agent startup alongside :func:`ensure_catalog_models`
    (see :func:`ensure_theme_structure_once`).
    """
    structure = registry.get_neo4j_theme_structure()

    # Create all Theme nodes
    for item in structure:
        await sink.merge_theme(item["path"], item["name"])

    # Create HAS_SUBTOPIC relationships between nested themes
    for item in structure:
        if item["parent_path"]:
            await sink.link_subtopic(item["parent_path"], item["path"])

    # Create BELONGS_TO_THEME relationships from CatalogModel to Theme
    for item in structure:
        for model_name in item["model_names"]:
            await sink.link_model_to_theme(model_name, item["path"])

    log.info(
        "ensure_theme_structure: %d Theme nodes and their relationships merged",
        len(structure),
    )


# ---------------------------------------------------------------------------
# Process-level memoization guards
#
# ensure_catalog_models()/ensure_theme_structure() are idempotent in the graph (they
# use MERGE), but not idempotent at the Python-process level: if called N
# times concurrently (e.g. once per document/task), they perform N redundant
# round-trips to Neo4j. These "_once" wrappers ensure the underlying function
# runs exactly once per process, even under concurrent asyncio callers, using
# a check-lock-check pattern (a plain bool is not safe here: two tasks could
# both read False before the first finishes writing).
# ---------------------------------------------------------------------------

_catalog_models_ensured: bool = False
_catalog_models_lock = None

_theme_structure_ensured: bool = False
_theme_structure_lock = None


def _get_catalog_models_lock():
    """Return (creating if needed) the lock guarding ensure_catalog_models_once."""
    global _catalog_models_lock
    if _catalog_models_lock is None:
        _catalog_models_lock = asyncio.Lock()
    return _catalog_models_lock


def _get_theme_structure_lock():
    """Return (creating if needed) the lock guarding ensure_theme_structure_once."""
    global _theme_structure_lock
    if _theme_structure_lock is None:
        _theme_structure_lock = asyncio.Lock()
    return _theme_structure_lock


async def ensure_catalog_models_once(registry: ThemeRegistry) -> None:
    """
    Process-memoized wrapper around :func:`ensure_catalog_models`.

    Runs the real (Neo4j-idempotent) setup exactly once per process, even if
    called concurrently from multiple tasks. Subsequent calls are a no-op.
    Use reset_catalog_memoization() to force a re-run (e.g. in tests or after
    configure() changes the underlying registry/driver).
    """
    global _catalog_models_ensured
    if _catalog_models_ensured:
        return
    async with _get_catalog_models_lock():
        if _catalog_models_ensured:
            return
        async with get_graph_store().annotation.catalog_sink() as sink:
            await ensure_catalog_models(registry, sink)
        _catalog_models_ensured = True


async def ensure_theme_structure_once(registry: ThemeRegistry) -> None:
    """
    Process-memoized wrapper around :func:`ensure_theme_structure`.

    Runs the real (Neo4j-idempotent) setup exactly once per process, even if
    called concurrently from multiple tasks. Subsequent calls are a no-op.
    Use reset_catalog_memoization() to force a re-run (e.g. in tests or after
    configure() changes the underlying registry/driver).
    """
    global _theme_structure_ensured
    if _theme_structure_ensured:
        return
    async with _get_theme_structure_lock():
        if _theme_structure_ensured:
            return
        async with get_graph_store().annotation.catalog_sink() as sink:
            await ensure_theme_structure(registry, sink)
        _theme_structure_ensured = True


def reset_catalog_memoization() -> None:
    """
    Reset the process-level memoization guards for ensure_catalog_models_once
    and ensure_theme_structure_once.

    Call this after configure() so the next pipeline run re-executes the
    Neo4j schema setup (e.g. after the theme registry or driver changed).
    Also drops the cached locks, since an asyncio.Lock created under a
    now-closed event loop should not be reused by a later run.
    """
    global _catalog_models_ensured, _catalog_models_lock
    global _theme_structure_ensured, _theme_structure_lock
    _catalog_models_ensured = False
    _catalog_models_lock = None
    _theme_structure_ensured = False
    _theme_structure_lock = None
