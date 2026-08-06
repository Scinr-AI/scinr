from __future__ import annotations

import hashlib
import json
import logging
import re as _re
import typing
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from neo4j import AsyncDriver

from scinr.newton.annotation.models import AnnotationDecision
from scinr.newton.utils.neo4j_retry import with_neo4j_retry

if TYPE_CHECKING:
    from scinr.newton.utils.theme_registry import ThemeRegistry

log = logging.getLogger(__name__)


def _make_uid(*parts: str) -> str:
    """16-char SHA-256 hex digest — deterministic UID from one or more string parts."""
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Node fetching
# ---------------------------------------------------------------------------


async def fetch_nodes_to_annotate(
    driver: AsyncDriver,
    document_name: str,
    only_unannotated: bool = False,
) -> list[dict]:
    """
    Fetch all StructureNodes that have at least one InfoUnit for this document.

    Traverses both direct HAS_STRUCTURE children and all HAS_CHILD descendants.
    Returns nodes ordered by appearance_order.

    When *only_unannotated* is True, nodes that already have a
    :HAS_MODEL_DECISION relationship are excluded.

    Returns
    -------
    list[dict]
        Each dict has keys: full_id, node_id, title, role, appearance_order, theme
    """
    extra_filter = ""
    if only_unannotated:
        extra_filter = "AND NOT (n)-[:HAS_MODEL_DECISION]->()"

    query = f"""
    MATCH (d:Document {{name: $doc_name, latest: true}})-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
    WHERE (n)-[:HAS_INFO_UNIT]->()
    {extra_filter}
    RETURN DISTINCT
           n.id               AS full_id,
           n.node_id          AS node_id,
           n.title            AS title,
           n.role             AS role,
           n.appearance_order AS appearance_order,
           coalesce(n.theme, 'default') AS theme
    """
    async with driver.session() as session:
        result = await session.run(query, doc_name=document_name)
        return await result.data()


async def _fetch_info_units_for_node(session, full_node_id: str) -> list[dict]:
    """
    Fetch all InfoUnits directly attached to a StructureNode.

    Parameters
    ----------
    session:
        An already-open Neo4j async session shared for the whole subtree traversal.
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.

    Returns
    -------
    list[dict]
        Each dict has: info_unit_id, title, description, order.
    """
    query = """
    MATCH (n:StructureNode {id: $node_id})-[:HAS_INFO_UNIT]->(iu:InfoUnit)
    RETURN iu.uid         AS info_unit_id,
           iu.title       AS title,
           iu.description AS description,
           iu.order       AS order
    ORDER BY iu.uid
    """
    result = await session.run(query, node_id=full_node_id)
    return await result.data()


async def _fetch_qualifying_children(session, full_node_id: str) -> list[dict]:
    """
    Fetch direct StructureNode children whose role is freeform_block, table, or field_group.

    Parameters
    ----------
    session:
        An already-open Neo4j async session shared for the whole subtree traversal.
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.

    Returns
    -------
    list[dict]
        Each dict has: full_id, node_id, title, role, appearance_order.
    """
    query = """
    MATCH (n:StructureNode {id: $node_id})-[:HAS_CHILD]->(c:StructureNode)
    WHERE c.role IN ['freeform_block', 'table', 'field_group']
    RETURN c.id               AS full_id,
           c.node_id          AS node_id,
           c.title            AS title,
           c.role             AS role,
           c.appearance_order AS appearance_order
    ORDER BY c.appearance_order
    """
    result = await session.run(query, node_id=full_node_id)
    return await result.data()


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


async def _fetch_node_context_with_session(
    session,
    full_node_id: str,
    node_id: str,
    title: str | None = None,
    role: str | None = None,
    depth: int = 0,
    max_depth: int = 3,
    visited: set[str] | None = None,
) -> dict:
    """
    Recursively build a context dict for a StructureNode using a shared session.

    This private function contains the full recursive logic and is called by
    the public ``fetch_node_context`` wrapper which owns the session lifecycle.
    Reusing a single session across the entire subtree avoids opening O(b^depth)
    concurrent connections and eliminates any risk of semaphore deadlocks.

    Parameters
    ----------
    session:
        An already-open Neo4j async session. Shared across all recursive calls
        for a single subtree traversal — never opened or closed here.
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.
    node_id:
        Short node_id (already known from parent query or node_data).
        Avoids a redundant round-trip to Neo4j.
    title:
        Node title, if already known from the parent query.
    role:
        Node role, if already known from the parent query.
    depth:
        Current recursion depth (0 = root node).
    max_depth:
        Maximum recursion depth (default 3).
    visited:
        Set of full_node_ids already visited in this traversal (cycle guard).

    Returns
    -------
    dict
        NodeContext-shaped dict with keys:
        node_id, full_id, title, role, info_units, children,
        depth, depth_limit_reached.

    .. warning::
        The ``session`` parameter is shared across all recursive calls.
        Do **not** call this function with ``asyncio.gather`` over multiple
        children — that would use the session concurrently across coroutines,
        which is not safe. The children loop must remain sequential (``for``).
    """
    if visited is None:
        visited = set()

    if depth >= max_depth or full_node_id in visited:
        return {
            "full_id": full_node_id,
            "node_id": node_id,
            "title": title,
            "role": role,
            "depth": depth,
            "depth_limit_reached": True,
            "info_units": [],
            "children": [],
        }

    visited = visited | {full_node_id}

    info_units = await _fetch_info_units_for_node(session, full_node_id)
    children_meta = await _fetch_qualifying_children(session, full_node_id)

    children = []
    # NOTE: intentionally sequential — do NOT convert to asyncio.gather.
    # The session is shared across all recursive calls; concurrent use would
    # violate neo4j AsyncSession's single-transaction-at-a-time contract.
    for child in children_meta:
        child_ctx = await _fetch_node_context_with_session(
            session=session,
            full_node_id=child["full_id"],
            node_id=child["node_id"],
            title=child.get("title"),
            role=child.get("role"),
            depth=depth + 1,
            max_depth=max_depth,
            visited=visited,
        )
        children.append(child_ctx)

    return {
        "full_id": full_node_id,
        "node_id": node_id,
        "title": title,
        "role": role,
        "depth": depth,
        "depth_limit_reached": False,
        "info_units": info_units,
        "children": children,
    }


async def fetch_node_context(
    driver: AsyncDriver,
    full_node_id: str,
    node_id: str,
    title: str | None = None,
    role: str | None = None,
    depth: int = 0,
    max_depth: int = 3,
    visited: set[str] | None = None,
) -> dict:
    """
    Recursively build a context dict for a StructureNode.

    Opens exactly **one** Neo4j session for the entire subtree traversal and
    delegates all recursive work to ``_fetch_node_context_with_session``.
    This ensures that the number of concurrent connections is bounded by the
    number of root-level calls to this function, regardless of tree depth or
    branching factor.

    For each node:
    - Fetch its InfoUnits
    - Fetch direct qualifying children (freeform_block / table / field_group)
    - Recursively apply the same logic to qualifying children up to max_depth

    Guards:
    - max_depth=3 prevents excessive query fan-out
    - visited set prevents infinite loops on graph cycles

    Parameters
    ----------
    driver:
        Async Neo4j driver instance (singleton from get_async_driver()).
    full_node_id:
        Composite StructureNode.id as stored in Neo4j.
    node_id:
        Short node_id already known from the caller (from fetch_nodes_to_annotate
        or from the parent _fetch_qualifying_children result). Passed through to
        avoid a redundant Neo4j round-trip.
    title:
        Node title, if already known.
    role:
        Node role, if already known.
    depth:
        Starting recursion depth (always 0 for external callers).
    max_depth:
        Maximum recursion depth (default 3).
    visited:
        Cycle-guard set; should be None for external callers.

    Returns
    -------
    dict
        NodeContext-shaped dict with keys:
        node_id, full_id, title, role, info_units, children,
        depth, depth_limit_reached.
    """
    async with driver.session() as session:
        return await _fetch_node_context_with_session(
            session=session,
            full_node_id=full_node_id,
            node_id=node_id,
            title=title,
            role=role,
            depth=depth,
            max_depth=max_depth,
            visited=visited,
        )


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------


async def ensure_catalog_models(driver: AsyncDriver) -> None:
    """
    Create (:CatalogModel) singleton nodes in Neo4j for every Pydantic model reachable
    from the registered themes and the pharmaceutical model hierarchy.

    Creates:
      (:CatalogModel {name, description, selectable})  — one per unique model class
      (:CatalogModel)-[:HAS_FIELD]->(:ModelField)       — one per Pydantic field
      (:CatalogModel)-[:AGGREGATES {field_name, is_list, required}]->(:CatalogModel)
                                                        — when a field references another model

    selectable=True  : model appears in a theme's SELECTABLE_MODELS list (valid annotation target)
    selectable=False : sub-model or aggregator, not directly selectable for annotation

    Idempotent (uses MERGE). Call once at agent startup.
    """
    from scinr.newton.utils.theme_registry import ThemeRegistry, get_theme_registry
    theme_registry = get_theme_registry()

    # ── 1. Collect selectable model names and seed set from all themes ────────
    selectable_names: set[str] = set()
    seed_classes: dict[str, type] = {}
    for theme_node in theme_registry._themes.values():
        for cls in theme_node.models:
            selectable_names.add(cls.__name__)
            seed_classes[cls.__name__] = cls

    # ── 2. Recursively discover every referenced Pydantic model via BFS ───────
    all_models = _collect_all_model_classes(seed_classes)

    field_count = 0
    agg_count = 0

    async with driver.session() as session:
        # ── 4. Create / update CatalogModel nodes ─────────────────────────────
        for model_name, cls in all_models.items():
            description = ThemeRegistry._get_docstring_summary(cls)
            selectable = model_name in selectable_names
            await session.run(
                """
                MERGE (cm:CatalogModel {name: $name})
                SET cm.description = $description,
                    cm.selectable  = $selectable
                """,
                name=model_name,
                description=description,
                selectable=selectable,
            )

        # ── 5. Create HAS_FIELD relationships ─────────────────────────────────
        for model_name, cls in all_models.items():
            if not hasattr(cls, "model_fields"):
                continue
            for field_name, field_info in cls.model_fields.items():
                field_type = _field_type_str(field_info.annotation)
                field_description = field_info.description or ""
                field_required = field_info.is_required()
                await session.run(
                    """
                    MERGE (mf:ModelField {name: $field_name, model: $model_name})
                    ON CREATE SET mf.type = $field_type
                    WITH mf
                    MATCH (cm:CatalogModel {name: $model_name})
                    MERGE (cm)-[r:HAS_FIELD]->(mf)
                    SET r.description = $description,
                        r.required    = $required
                    """,
                    field_name=field_name,
                    field_type=field_type,
                    model_name=model_name,
                    description=field_description,
                    required=field_required,
                )
                field_count += 1

        # ── 6. Create AGGREGATES relationships between CatalogModel nodes ──────
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
                    await session.run(
                        """
                        MATCH (src:CatalogModel {name: $src_name})
                        MATCH (tgt:CatalogModel {name: $tgt_name})
                        MERGE (src)-[r:AGGREGATES {field_name: $field_name}]->(tgt)
                        SET r.is_list  = $is_list,
                            r.required = $required
                        """,
                        src_name=model_name,
                        tgt_name=ref_name,
                        field_name=field_name,
                        is_list=is_list,
                        required=required,
                    )
                    agg_count += 1

        # Pass A: Enrich ModelField nodes with json_schema_extra metadata
        entity_label_count = 0
        instance_key_count = 0
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
                await session.run(
                    """
                    MERGE (mf:ModelField {name: $field_name, model: $model_name})
                    SET mf.is_instance_key = $is_instance_key,
                        mf.entity_label    = $entity_label
                    """,
                    field_name=field_name,
                    model_name=model_name,
                    is_instance_key=is_instance_key,
                    entity_label=entity_label,
                )
                if is_instance_key:
                    instance_key_count += 1
                if entity_label is not None:
                    entity_label_count += 1

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
                await session.run(
                    """
                    MERGE (el:EntityLabel {label: $entity_label})
                    WITH el
                    MATCH (cm:CatalogModel {name: $model_name})
                    MERGE (cm)-[:PRODUCES_ENTITY {field_name: $field_name}]->(el)
                    """,
                    entity_label=entity_label,
                    model_name=model_name,
                    field_name=field_name,
                )
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
                    # Use a parameterised rel_type via apoc or build query dynamically
                    # Since Neo4j does not support parameterised relationship types, build the query string
                    if not _validate_rel_type(rel_type, f"{model_name}.{field_name}.field_relationships"):
                        continue
                    cypher = f"""
                    MERGE (el_src:EntityLabel {{label: $source_label}})
                    MERGE (el_tgt:EntityLabel {{label: $target_label}})
                    MERGE (el_src)-[r:`{rel_type}` {{via_model: $via_model, from_field: $from_field, to_field: $to_field}}]->(el_tgt)
                    """
                    await session.run(
                        cypher,
                        source_label=source_entity_label,
                        target_label=target_entity_label,
                        via_model=model_name,
                        from_field=field_name,
                        to_field=to_field_name,
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
                    # Use dynamic rel_type in query string (not parameterisable in Neo4j)
                    if not _validate_rel_type(rel_type, f"{model_name}.{field_name}.instance_relationships"):
                        continue
                    cypher = f"""
                    MERGE (src:CatalogModel {{name: $src_name}})
                    MERGE (tgt:CatalogModel {{name: $tgt_name}})
                    MERGE (src)-[r:`{rel_type}` {{via_field: $via_field, join_via: $join_via_json}}]->(tgt)
                    SET r.composite = $is_composite
                    """
                    await session.run(
                        cypher,
                        src_name=model_name,
                        tgt_name=target_model,
                        via_field=field_name,
                        join_via_json=join_via_json,
                        is_composite=is_composite,
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


async def ensure_theme_structure(driver: AsyncDriver, registry: ThemeRegistry) -> None:
    """
    Create (:Theme) nodes in Neo4j mirroring the models/ folder hierarchy.

    Creates:
      - (:Theme {path, name}) for each discovered theme folder
      - (:Theme)-[:HAS_SUBTOPIC]->(:Theme) for nested theme relationships
      - (:CatalogModel)-[:BELONGS_TO_THEME]->(:Theme) for each model-to-theme mapping

    Idempotent (uses MERGE). Call once at agent startup alongside ensure_catalog_models.
    """
    structure = registry.get_neo4j_theme_structure()

    async with driver.session() as session:
        # Create all Theme nodes
        for item in structure:
            await session.run(
                "MERGE (t:Theme {path: $path}) SET t.name = $name",
                path=item["path"],
                name=item["name"],
            )

        # Create HAS_SUBTOPIC relationships between nested themes
        for item in structure:
            if item["parent_path"]:
                await session.run(
                    """
                    MATCH (parent:Theme {path: $parent_path})
                    MATCH (child:Theme {path: $child_path})
                    MERGE (parent)-[:HAS_SUBTOPIC]->(child)
                    """,
                    parent_path=item["parent_path"],
                    child_path=item["path"],
                )

        # Create BELONGS_TO_THEME relationships from CatalogModel to Theme
        for item in structure:
            for model_name in item["model_names"]:
                await session.run(
                    """
                    MERGE (cm:CatalogModel {name: $model_name})
                    MERGE (t:Theme {path: $path})
                    MERGE (cm)-[:BELONGS_TO_THEME]->(t)
                    """,
                    model_name=model_name,
                    path=item["path"],
                )

    log.info(
        "ensure_theme_structure: %d Theme nodes and their relationships merged",
        len(structure),
    )


# ---------------------------------------------------------------------------
# Process-level memoization guards
#
# ensure_catalog_models/ensure_theme_structure are idempotent in Neo4j (they
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
    import asyncio

    global _catalog_models_lock
    if _catalog_models_lock is None:
        _catalog_models_lock = asyncio.Lock()
    return _catalog_models_lock


def _get_theme_structure_lock():
    """Return (creating if needed) the lock guarding ensure_theme_structure_once."""
    import asyncio

    global _theme_structure_lock
    if _theme_structure_lock is None:
        _theme_structure_lock = asyncio.Lock()
    return _theme_structure_lock


async def ensure_catalog_models_once(driver: AsyncDriver) -> None:
    """
    Process-memoized wrapper around ensure_catalog_models.

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
        await ensure_catalog_models(driver)
        _catalog_models_ensured = True


async def ensure_theme_structure_once(driver: AsyncDriver, registry: ThemeRegistry) -> None:
    """
    Process-memoized wrapper around ensure_theme_structure.

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
        await ensure_theme_structure(driver, registry)
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


# ---------------------------------------------------------------------------
# Writing decisions
# ---------------------------------------------------------------------------


async def write_annotation(
    driver: AsyncDriver,
    full_node_id: str,
    decision: AnnotationDecision,
    document_name: str,
) -> None:
    """
    Write an AnnotationDecision subgraph to Neo4j for a StructureNode.

    Creates or replaces:
      (:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)
      (:ModelDecision)-[:MATCHED_MODEL]->(:CatalogModel)                    [when matched_model_class is set]
      (:ModelDecision)-[:HAS_COMPLEMENTARY_MATCH]->(:ComplementaryMatch)
      (:ComplementaryMatch)-[:REFERS_TO_MODEL]->(:CatalogModel)
      (:ModelDecision)-[:HAS_SUPPLEMENTARY_FIELD]->(:SupplementaryField)    [when matched and gaps exist]
      (:ModelDecision)-[:HAS_PROPOSED_MODEL]->(:ProposedModel)              [when no match]
      (:ProposedModel)-[:HAS_PROPOSED_FIELD]->(:ProposedField)

    Idempotency: existing :ProposedField, :ProposedModel, :SupplementaryField,
    :ComplementaryMatch, and :ModelDecision nodes for this StructureNode are
    deleted before recreation. :CatalogModel singletons are never deleted.

    Raises
    ------
    RuntimeError
        If the StructureNode does not exist in the database.
    """
    decision_uid = _make_uid(
        "model_decision",
        full_node_id,
        decision.matched_model_class
        if decision.matched_model_class is not None
        else "null",
    )
    timestamp = datetime.now(UTC).isoformat()

    async with driver.session() as session:
        # ── Guard: verify StructureNode exists ────────────────────────────
        result = await session.run(
            "MATCH (n:StructureNode {id: $node_id}) RETURN count(n) AS cnt",
            node_id=full_node_id,
        )
        record = await result.single()
        if record["cnt"] == 0:
            raise RuntimeError(
                f"write_annotation: StructureNode not found for full_node_id={full_node_id!r}"
            )

        # ── Idempotency: delete stale ProposedField nodes ─────────────────
        await session.run(
            """
            MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                  -[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)-[:HAS_PROPOSED_FIELD]->(pf)
            DETACH DELETE pf
            """,
            node_id=full_node_id,
        )

        # ── Idempotency: delete stale ProposedModel nodes ─────────────────
        await session.run(
            """
            MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                  -[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)
            DETACH DELETE pm
            """,
            node_id=full_node_id,
        )

        # ── Idempotency: delete stale SupplementaryField nodes ────────────
        await session.run(
            """
            MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                  -[:HAS_SUPPLEMENTARY_FIELD]->(sf)
            DETACH DELETE sf
            """,
            node_id=full_node_id,
        )

        # ── Idempotency: delete stale ComplementaryMatch nodes first ──────
        await session.run(
            """
            MATCH (:StructureNode {id: $node_id})
                  -[:HAS_MODEL_DECISION]->(:ModelDecision)
                  -[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
            DETACH DELETE cm
            """,
            node_id=full_node_id,
        )

        # ── Idempotency: delete stale ModelDecision node ──────────────────
        await session.run(
            """
            MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(md:ModelDecision)
            DETACH DELETE md
            """,
            node_id=full_node_id,
        )

        # ── Create ModelDecision node + HAS_MODEL_DECISION relationship ───
        await session.run(
            """
            MATCH (n:StructureNode {id: $node_id})
            CREATE (md:ModelDecision {
                uid:                       $uid,
                matched_model_class:       $matched_model_class,
                confidence:                $confidence,
                rationale:                 $rationale,
                coverage_gaps:             $coverage_gaps,
                propose_new_model:         $propose_new_model,
                proposed_model_description: $proposed_model_description,
                document_name:             $document_name,
                timestamp:                 $timestamp
            })
            CREATE (n)-[:HAS_MODEL_DECISION]->(md)
            """,
            node_id=full_node_id,
            uid=decision_uid,
            matched_model_class=decision.matched_model_class,
            confidence=decision.confidence,
            rationale=decision.rationale,
            coverage_gaps=decision.coverage_gaps,
            propose_new_model=decision.propose_new_model,
            proposed_model_description=decision.proposed_model_description,
            document_name=document_name,
            timestamp=timestamp,
        )

        # ── MATCHED_MODEL → CatalogModel (only when a model was matched) ──
        if decision.matched_model_class is not None:
            _params = dict(
                decision_uid=decision_uid,
                model_name=decision.matched_model_class,
            )
            await with_neo4j_retry(lambda: session.run(
                """
                MATCH (md:ModelDecision {uid: $decision_uid})
                MERGE (cat:CatalogModel {name: $model_name})
                MERGE (md)-[:MATCHED_MODEL]->(cat)
                """,
                **_params,
            ))

        # ── ComplementaryMatch nodes ───────────────────────────────────────
        for cm in decision.complementary_models:
            cm_uid = _make_uid(full_node_id, cm.model_class)
            _cm_params = dict(
                decision_uid=decision_uid,
                cm_uid=cm_uid,
                model_class=cm.model_class,
                coverage_note=cm.coverage_note,
            )
            await with_neo4j_retry(lambda: session.run(
                """
                MATCH (md:ModelDecision {uid: $decision_uid})
                CREATE (comp:ComplementaryMatch {
                    uid:           $cm_uid,
                    model_class:   $model_class,
                    coverage_note: $coverage_note
                })
                CREATE (md)-[:HAS_COMPLEMENTARY_MATCH]->(comp)
                WITH comp
                MERGE (cat:CatalogModel {name: $model_class})
                MERGE (comp)-[:REFERS_TO_MODEL]->(cat)
                """,
                **_cm_params,
            ))

        # ── ProposedModel + ProposedField nodes (when no model matched) ───
        if (
            decision.matched_model_class is None
            and decision.proposed_schema_name is not None
        ):
            pm_uid = _make_uid(full_node_id, "proposed_model")
            await session.run(
                """
                MATCH (md:ModelDecision {uid: $decision_uid})
                CREATE (pm:ProposedModel {
                    uid:         $pm_uid,
                    schema_name: $schema_name,
                    description: $description
                })
                CREATE (md)-[:HAS_PROPOSED_MODEL]->(pm)
                """,
                decision_uid=decision_uid,
                pm_uid=pm_uid,
                schema_name=decision.proposed_schema_name,
                description=decision.proposed_model_description or "",
            )
            for pf in decision.proposed_schema_fields:
                f_uid = _make_uid(full_node_id, "proposed_field", pf.field_name)
                await session.run(
                    """
                    MATCH (pm:ProposedModel {uid: $pm_uid})
                    CREATE (f:ProposedField {
                        uid:         $f_uid,
                        field_name:  $field_name,
                        field_type:  $field_type,
                        description: $description,
                        required:    $required
                    })
                    CREATE (pm)-[:HAS_PROPOSED_FIELD]->(f)
                    """,
                    pm_uid=pm_uid,
                    f_uid=f_uid,
                    field_name=pf.field_name,
                    field_type=pf.field_type,
                    description=pf.description,
                    required=pf.required,
                )

        # ── SupplementaryField nodes (when a model matched and gaps exist) ─
        if (
            decision.matched_model_class is not None
            and decision.supplementary_fields
        ):
            for sf in decision.supplementary_fields:
                s_uid = _make_uid(full_node_id, "supplementary_field", sf.field_name)
                await session.run(
                    """
                    MATCH (md:ModelDecision {uid: $decision_uid})
                    CREATE (s:SupplementaryField {
                        uid:         $s_uid,
                        field_name:  $field_name,
                        field_type:  $field_type,
                        description: $description,
                        required:    $required
                    })
                    CREATE (md)-[:HAS_SUPPLEMENTARY_FIELD]->(s)
                    """,
                    decision_uid=decision_uid,
                    s_uid=s_uid,
                    field_name=sf.field_name,
                    field_type=sf.field_type,
                    description=sf.description,
                    required=sf.required,
                )

    log.info(
        "write_annotation: subgraph written for full_node_id=%r "
        "(class=%s, confidence=%s, %d complementary, %d supplementary_fields, "
        "proposed_model=%s, %d proposed_fields)",
        full_node_id,
        decision.matched_model_class,
        decision.confidence,
        len(decision.complementary_models),
        len(decision.supplementary_fields),
        decision.proposed_schema_name,
        len(decision.proposed_schema_fields),
    )


async def write_manual_annotation(
    driver: AsyncDriver,
    document_name: str,
    matched_model_class: str,
) -> int:
    """
    Assign a manual ModelDecision with the given model class to all StructureNodes
    with InfoUnits for the specified document.

    Bypasses the LLM annotation agent and writes a simplified ModelDecision with
    source='manual'. Any existing ModelDecision and ExtractionResult subgraphs
    for each qualifying node are replaced before writing the new decision.

    The subgraph written per node is minimal:
      (:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision {source: 'manual', ...})
      (:ModelDecision)-[:MATCHED_MODEL]->(:CatalogModel)

    Parameters
    ----------
    driver:
        Open Neo4j driver.
    document_name:
        Exact Document.name as stored in Neo4j (must have latest=True).
    matched_model_class:
        CamelCase Pydantic model class name to assign to every qualifying node.

    Returns
    -------
    int
        Number of StructureNodes updated. Returns 0 if no qualifying nodes are found.
    """
    timestamp = datetime.now(UTC).isoformat()

    # ── 1. Fetch all StructureNodes with at least one InfoUnit ────────────────
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {name: $doc_name, latest: true})
                  -[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
            WHERE (n)-[:HAS_INFO_UNIT]->()
            RETURN DISTINCT n.id AS full_node_id
            """,
            doc_name=document_name,
        )
        node_ids = [r["full_node_id"] for r in await result.data()]

    if not node_ids:
        log.warning(
            "write_manual_annotation: no StructureNodes with InfoUnits found "
            "for document %r",
            document_name,
        )
        return 0

    # ── 2. For each node: clear stale subgraphs, write new ModelDecision ─────
    async with driver.session() as session:
        for full_node_id in node_ids:
            decision_uid = _make_uid(
                "manual_model_decision",
                full_node_id,
                matched_model_class,
            )

            # Delete stale ExtractionResult and its exclusive ModelInstance children
            await session.run(
                """
                MATCH (n:StructureNode {id: $node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult)
                OPTIONAL MATCH (er)-[*1..10]->(child:ModelInstance)
                DETACH DELETE child
                WITH er
                DETACH DELETE er
                """,
                node_id=full_node_id,
            )

            # Delete stale ModelDecision subgraph — leaf nodes first to avoid
            # dangling relationships (mirrors the idempotency pattern in write_annotation)
            await session.run(
                """
                MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)-[:HAS_PROPOSED_FIELD]->(pf)
                DETACH DELETE pf
                """,
                node_id=full_node_id,
            )
            await session.run(
                """
                MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)
                DETACH DELETE pm
                """,
                node_id=full_node_id,
            )
            await session.run(
                """
                MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_SUPPLEMENTARY_FIELD]->(sf)
                DETACH DELETE sf
                """,
                node_id=full_node_id,
            )
            await session.run(
                """
                MATCH (:StructureNode {id: $node_id})
                      -[:HAS_MODEL_DECISION]->(:ModelDecision)
                      -[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
                DETACH DELETE cm
                """,
                node_id=full_node_id,
            )
            await session.run(
                """
                MATCH (:StructureNode {id: $node_id})-[:HAS_MODEL_DECISION]->(md:ModelDecision)
                DETACH DELETE md
                """,
                node_id=full_node_id,
            )

            # Create new minimal ModelDecision with source='manual'
            await session.run(
                """
                MATCH (n:StructureNode {id: $node_id})
                CREATE (md:ModelDecision {
                    uid:                        $uid,
                    matched_model_class:        $matched_model_class,
                    confidence:                 'high',
                    rationale:                  'Manually assigned override',
                    coverage_gaps:              [],
                    propose_new_model:          false,
                    proposed_model_description: null,
                    document_name:              $document_name,
                    timestamp:                  $timestamp,
                    source:                     'manual'
                })
                CREATE (n)-[:HAS_MODEL_DECISION]->(md)
                """,
                node_id=full_node_id,
                uid=decision_uid,
                matched_model_class=matched_model_class,
                document_name=document_name,
                timestamp=timestamp,
            )

            # Link ModelDecision to CatalogModel singleton
            await session.run(
                """
                MATCH (md:ModelDecision {uid: $decision_uid})
                MERGE (cm:CatalogModel {name: $model_name})
                MERGE (md)-[:MATCHED_MODEL]->(cm)
                """,
                decision_uid=decision_uid,
                model_name=matched_model_class,
            )

    log.info(
        "write_manual_annotation: assigned '%s' to %d nodes in document '%s'",
        matched_model_class,
        len(node_ids),
        document_name,
    )
    return len(node_ids)


async def fetch_document_context_instructions(
    driver,
    document_name: str,
) -> str | None:
    """Fetches the context_instructions property from the latest (:Document) node."""
    query = """
        MATCH (d:Document {name: $document_name, latest: true})
        RETURN d.context_instructions AS context_instructions
    """
    async with driver.session() as session:
        result = await session.run(query, document_name=document_name)
        record = await result.single()
        if record is None:
            return None
        return record["context_instructions"]
