"""
entity_extraction/graph_mapper.py

Converts a populated Pydantic instance (ExtractionModel or CompositeSchema) into
a Neo4j subgraph using three relationship mechanisms:

  Level 1 — Entity labeling:
    Fields with json_schema_extra={"entity_label": "X"} become MERGE'd
    (:X {label, value, normalized_value}) nodes. Same tenant + same label +
    same normalized_value always resolves to the same node across all
    extractions of that tenant (tenant_id is folded into the uid, so entities
    are never shared between tenants).

  Level 2 — Field relationships:
    Fields with json_schema_extra={"field_relationships": [{"to_field": "...", "rel_type": "..."}]}
    trigger MERGE relationships between the source entity node and the
    target entity node (target must also have entity_label).

  Level 3 — Instance Key Relationships:
    Fields with json_schema_extra={"instance_key": True} define a composite
    key that makes ModelInstance nodes deduplicatable within a tenant (UID =
    make_instance_uid(model_class, {sorted key_fields}, tenant_id)). Analogous
    to LabeledEntity deduplication by (tenant_id, label, normalized_value).

  Provenance: every node written here carries a scalar ``tenant_id`` read
  back from the owning StructureNode — the *stored* tenant key (the tenant, or
  ``"__public__"`` for a public document; see ``utils/tenancy.py``), which is
  also the value hashed into every tenant-scoped uid. Merge-deduplicated nodes (ModelInstance
  with instance_key, LabeledEntity, Entity) also accumulate
  ``created_by_user_ids`` / ``job_ids`` arrays. Key-less ModelInstance nodes
  (never merged) get the same array properties with a single element, so every
  ModelInstance is filtered uniformly; ExtractionResult keeps them as scalars.

    Fields with json_schema_extra={"instance_relationships": [...]} trigger,
    for each item in a list[str] field, MERGE of a target ModelInstance shell
    (identified by its composite key) and MERGE of the typed relationship
    (src_mi)-[:REL_TYPE]->(tgt_mi). Enables forward references across
    StructureNode boundaries.

Graph produced per ExtractionResult:
  (:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)
  (:ExtractionResult)-[:USES_PRIMARY_MODEL]->(:CatalogModel)
  (:ExtractionResult)-[:USES_COMPLEMENTARY_MODEL]->(:CatalogModel)  [0..*]
  (:ExtractionResult)-[:HAS_<FIELDNAME> {index}]->(:ModelInstance)  [for nested models]
  (:ModelInstance | :ExtractionResult)-[:REFERENCES {field_name}]->(:LabeledEntity)
  (:LabeledEntity)-[:REL_TYPE]->(:LabeledEntity)                    [field_relationships]
  (:ModelInstance)-[:REL_TYPE]->(:ModelInstance)                    [instance_relationships, Level 3]

For Triple (fallback) extractions:
  (:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult {model_class: "Triple"})
  (:ExtractionResult)-[:HAS_ENTITY {role}]->(:Entity)
  (:Entity)-[:NORMALIZED_PREDICATE {predicate_raw}]->(:Entity)
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from collections.abc import Sequence
from datetime import UTC
from typing import Any

from neo4j import AsyncDriver
from pydantic import BaseModel

from scinr.newton.config import get_config
from scinr.newton.utils.neo4j_retry import with_neo4j_retry
from scinr.newton.utils.tenancy import PUBLIC_TENANT
from scinr.newton.utils.uid import make_instance_uid as _make_instance_uid
from scinr.newton.utils.uid import make_uid as _make_uid
from scinr.newton.utils.uid import normalize_key as _normalize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ``_normalize`` (lower-case, strip accents, collapse whitespace) is the shared
# ``scinr.newton.utils.uid.normalize_key`` — the navigation layer reuses the
# exact same transform to rebuild ModelInstance UIDs from raw key values.


def _to_rel_name(field_name: str) -> str:
    """
    Convert a Python field name to a safe Neo4j relationship type.
    Replaces any character that is not alphanumeric or underscore with underscore.
    """
    safe = re.sub(r"[^A-Z0-9_]", "_", field_name.upper())
    if not safe or safe[0].isdigit():
        safe = f"HAS_{safe}"
    return safe or "HAS_VALUE"


def _get_entity_label(field_info) -> str | None:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict):
        return extra.get("entity_label")
    return None


def _get_field_relationships(field_info) -> list[dict]:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict):
        return extra.get("field_relationships", [])
    return []


def _get_instance_key(field_info) -> bool:
    """Return True if the field is marked as an instance_key component."""
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict):
        return bool(extra.get("instance_key", False))
    return False


def _get_instance_relationships(field_info) -> list[dict]:
    """Return the instance_relationships list from json_schema_extra, or []."""
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict):
        return extra.get("instance_relationships", [])
    return []


def _stringify_if_dict(value: Any) -> Any:
    """
    Last-resort defensive coercion for values headed for a Neo4j scalar
    property. Neo4j only supports primitive types or arrays of primitives —
    a raw ``dict`` can never be written as a property value.

    This is an independent third layer of defense (in addition to the
    field-type sanitization and the ``mode="before"`` validator in
    schema_composer.py): it protects against ANY future dict-typed value
    that slips through by another path (a different theme, a future
    annotation mechanism, a misdeclared custom model) so that a single bad
    property never aborts the write of the entire extraction subgraph.

    Parameters
    ----------
    value:
        Candidate value for a Neo4j scalar property.

    Returns
    -------
    Any
        *value* unchanged if it is not a dict; otherwise a human-readable
        "key: value; key2: value2" string representation of it.
    """
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return value


def _get_instance_key_fields(instance: BaseModel) -> dict[str, str] | None:
    """
    If the instance has ≥1 field marked with instance_key=True, return a dict
    {field_name: normalized_value} for those fields. Otherwise return None.

    The normalized values are produced by _normalize() so they are ready to
    pass directly to make_instance_uid() without further processing.
    """
    key_fields: dict[str, str] = {}
    for field_name, field_info in instance.model_fields.items():
        if _get_instance_key(field_info):
            value = getattr(instance, field_name, None)
            if value is not None:
                key_fields[field_name] = _normalize(str(value))
    return key_fields if key_fields else None


def _require_stored_tenant(stored: str | None, node_full_id: str) -> str:
    """Return the stored tenant key read off a StructureNode, refusing a
    missing one: hashing a null tenant into a uid would silently create a
    bucket shared by every such node, outside any tenant."""
    if stored is None:
        raise RuntimeError(
            f"StructureNode {node_full_id!r} has no tenant_id; re-ingest its document."
        )
    return stored


def _provenance_set_clause(alias: str) -> str:
    """SET-clause fragment for a MERGE'd node: scalar ``tenant_id`` plus
    accumulate ``created_by_user_id`` / ``job_id`` into deduplicated arrays.

    Used identically in ``ON CREATE`` and ``ON MATCH`` branches — re-affirming
    ``tenant_id`` on every MATCH is harmless since it is already folded into the
    node's ``uid`` (a MATCH only ever happens within the same tenant). A ``None``
    provenance value is skipped rather than pushed into the array.
    """
    return (
        f"{alias}.tenant_id = $tenant_id, "
        f"{alias}.created_by_user_ids = CASE "
        f"WHEN $created_by_user_id IS NULL THEN coalesce({alias}.created_by_user_ids, []) "
        f"WHEN $created_by_user_id IN coalesce({alias}.created_by_user_ids, []) THEN {alias}.created_by_user_ids "
        f"ELSE coalesce({alias}.created_by_user_ids, []) + $created_by_user_id END, "
        f"{alias}.job_ids = CASE "
        f"WHEN $job_id IS NULL THEN coalesce({alias}.job_ids, []) "
        f"WHEN $job_id IN coalesce({alias}.job_ids, []) THEN {alias}.job_ids "
        f"ELSE coalesce({alias}.job_ids, []) + $job_id END"
    )


_STALE_EXTRACTION_RESULT_DELETE_QUERY = """
UNWIND $node_ids AS nid
MATCH (n:StructureNode {id: nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
OPTIONAL MATCH (er)-[*1..10]->(child:ModelInstance)
DETACH DELETE child
WITH DISTINCT er
DETACH DELETE er
"""


async def delete_stale_extraction_result(runner, node_ids: str | Sequence[str]) -> None:
    """Delete the ExtractionResult(s) hanging from the given StructureNode(s),
    together with every :ModelInstance reachable from them within 10 hops.

    The idempotency step of write_extraction_subgraph(),
    write_manual_annotation() and restore_document(). Note that the
    ModelInstance purge does not check whether another live ExtractionResult
    still reaches the instance (long-standing pipeline behaviour); the GC
    passes of ``ingest/_gc.py`` clean up whatever else is left orphaned.

    Parameters
    ----------
    runner:
        An ``AsyncSession`` or ``AsyncTransaction`` (anything with an async ``run``).
    node_ids:
        One ``StructureNode.id`` (tenant-prefixed) or several.
    """
    ids = [node_ids] if isinstance(node_ids, str) else list(node_ids)
    await runner.run(_STALE_EXTRACTION_RESULT_DELETE_QUERY, node_ids=ids)


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------


async def write_extraction_subgraph(
    driver: AsyncDriver,
    node_full_id: str,
    composite_instance: BaseModel,
    primary_model_class: str,
    complementary_model_classes: list[str],
    document_name: str,
    extraction_uid: str,
) -> None:
    """
    Write the complete extraction subgraph for one StructureNode to Neo4j.

    Creates:
      (:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)
      + all entity nodes and relationships derived from the composite instance.

    Idempotent: uses MERGE for all entity nodes. ExtractionResult is re-created
    fresh (existing one for this node is deleted first).

    Parameters
    ----------
    driver:
        Open Neo4j driver.
    node_full_id:
        The StructureNode.id composite key.
    composite_instance:
        Populated instance of the composite Pydantic schema.
    primary_model_class:
        Name of the primary CatalogModel.
    complementary_model_classes:
        Names of complementary CatalogModel classes.
    document_name:
        Document.name for provenance.
    extraction_uid:
        Deterministic UID for the ExtractionResult node.
    """
    from datetime import datetime

    timestamp = datetime.now(UTC).isoformat()
    cfg = get_config()
    async with driver.session(database=cfg.neo4j_database) as session:
        # ── Guard: verify StructureNode exists, read its provenance ───────
        # tenant_id / created_by_user_id / job_id are stamped on the
        # StructureNode at Stage-2 ingestion time (see ingest/nodes.py) — read
        # them back here rather than threading them through the whole
        # extraction call stack. This is the single source of truth for "which
        # tenant does this extraction belong to".
        result = await session.run(
            "MATCH (n:StructureNode {id: $nid}) "
            "RETURN count(n) AS cnt, n.tenant_id AS tenant_id, "
            "n.created_by_user_id AS created_by_user_id, n.job_id AS job_id",
            nid=node_full_id,
        )
        rec = await result.single()
        cnt = rec["cnt"] if rec else 0
        if cnt == 0:
            raise RuntimeError(
                f"write_extraction_subgraph: StructureNode not found: {node_full_id!r}"
            )
        tenant_id = _require_stored_tenant(rec["tenant_id"], node_full_id)
        created_by_user_id = rec["created_by_user_id"]
        job_id = rec["job_id"]

        # ── Idempotency: delete stale ExtractionResult subgraph ───────────
        await delete_stale_extraction_result(session, node_full_id)

        # ── Create ExtractionResult node ──────────────────────────────────
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})
            CREATE (er:ExtractionResult {
                uid:           $uid,
                node_full_id:  $nid,
                document_name: $doc_name,
                model_class:   $model_class,
                timestamp:     $timestamp,
                tenant_id:            $tenant_id,
                created_by_user_id:   $created_by_user_id,
                job_id:               $job_id
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=node_full_id,
            uid=extraction_uid,
            doc_name=document_name,
            model_class=primary_model_class,
            timestamp=timestamp,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )

        # ── Link primary CatalogModel ─────────────────────────────────────
        _primary_params = dict(uid=extraction_uid, name=primary_model_class)
        await with_neo4j_retry(
            lambda: session.run(
                """
            MATCH (er:ExtractionResult {uid: $uid})
            MERGE (cm:CatalogModel {name: $name})
            MERGE (er)-[:USES_PRIMARY_MODEL]->(cm)
            """,
                **_primary_params,
            )
        )

        # ── Link complementary CatalogModels that were actually extracted ─
        # Only write USES_COMPLEMENTARY_MODEL for Optional[BaseModel] fields
        # where the LLM actually returned a non-None value. Required (primary)
        # fields are skipped because type(None) is not among their Union args.
        import typing as _typing

        for field_name, field_info in composite_instance.model_fields.items():
            ann = field_info.annotation
            args = _typing.get_args(ann)
            if type(None) not in args:
                continue  # required field — not a complementary model slot
            model_arg = next(
                (
                    a
                    for a in args
                    if a is not type(None) and isinstance(a, type) and hasattr(a, "model_fields")
                ),
                None,
            )
            if model_arg is None:
                continue  # not a BaseModel field
            value = getattr(composite_instance, field_name, None)
            if value is None:
                continue  # LLM returned None — do not write the relationship
            # Actually extracted — write the relationship
            _comp_params = dict(uid=extraction_uid, name=model_arg.__name__)
            await with_neo4j_retry(
                lambda: session.run(
                    """
                MATCH (er:ExtractionResult {uid: $uid})
                MERGE (cm:CatalogModel {name: $name})
                MERGE (er)-[:USES_COMPLEMENTARY_MODEL]->(cm)
                """,
                    **_comp_params,
                )
            )

        # ── Collect all entity nodes for Level 2 relationship resolution ──
        entity_nodes: dict[str, str] = {}  # field_path → labeled_entity_uid

        # ── Write fields from the composite instance ──────────────────────
        await _write_model_fields(
            session=session,
            instance=composite_instance,
            parent_uid=extraction_uid,
            parent_label="ExtractionResult",
            field_path_prefix="",
            entity_nodes=entity_nodes,
            depth=0,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )

        # ── Level 2: resolve field_relationships ──────────────────────────
        await _apply_field_relationships(
            session=session,
            instance=composite_instance,
            entity_nodes=entity_nodes,
            field_path_prefix="",
        )

    log.info(
        "write_extraction_subgraph: wrote ExtractionResult %s for node %r (%d entity nodes)",
        extraction_uid,
        node_full_id,
        len(entity_nodes),
    )


_PARENT_LABELS = frozenset({"ExtractionResult", "ModelInstance"})


def _check_parent_label(label: str) -> None:
    """Reject any label other than the two a model subgraph can hang from."""
    if label not in _PARENT_LABELS:
        raise ValueError(
            f"parent label must be one of {sorted(_PARENT_LABELS)}, got {label!r}"
        )


async def _write_model_fields(
    session,
    instance: BaseModel,
    parent_uid: str,
    parent_label: str,
    field_path_prefix: str,
    entity_nodes: dict[str, str],
    depth: int,
    tenant_id: str = PUBLIC_TENANT,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
    list_index: int | None = None,
) -> None:
    """
    Recursively write all fields of *instance* under the parent node identified by *parent_uid*.

    - Scalar fields with entity_label → MERGE LabeledEntity + REFERENCES relationship
    - Scalar fields without entity_label → stored as properties on the parent ModelInstance
    - Nested ExtractionModel fields → create ModelInstance child node, recurse
    - list[ExtractionModel] fields → create multiple ModelInstance child nodes
    - list[scalar] with entity_label → create multiple LabeledEntity nodes

    Parameters
    ----------
    session:
        Open Neo4j session.
    instance:
        Pydantic model instance to process.
    parent_uid:
        Neo4j uid of the parent node.
    parent_label:
        Neo4j label of the parent node: ``"ExtractionResult"`` at the root,
        ``"ModelInstance"`` below. Every MATCH on the parent uses it, so the
        lookup is a seek on the label's uid constraint instead of a scan of
        every node in the graph.
    field_path_prefix:
        Dot-separated prefix for entity_nodes registry keys.
    entity_nodes:
        Accumulator: maps "field_path" → labeled_entity_uid for Level 2 resolution.
    depth:
        Current recursion depth (safety cap at 10).
    list_index:
        If this instance is an element of a list, its 0-based index.
    """
    # Interpolated into the Cypher: a closed set, never caller-controlled text.
    _check_parent_label(parent_label)

    if depth > 10:
        log.warning("_write_model_fields: max depth reached, stopping recursion")
        return

    if not hasattr(instance, "model_fields"):
        return

    # Collect scalar properties to batch-SET on parent node
    scalar_props: dict[str, Any] = {}

    for field_name, field_info in instance.model_fields.items():
        value = getattr(instance, field_name, None)
        field_path = f"{field_path_prefix}.{field_name}" if field_path_prefix else field_name

        if value is None:
            continue

        entity_label = _get_entity_label(field_info)

        # ── Case: nested BaseModel ────────────────────────────────────────
        if isinstance(value, BaseModel):
            key_fields = _get_instance_key_fields(value)
            rel_name = f"HAS_{_to_rel_name(field_name)}"
            if key_fields:
                # ModelInstance con clave compuesta: UID determinístico (incluye
                # tenant_id), MERGE — nunca se comparte entre tenants distintos.
                child_uid = _make_instance_uid(type(value).__name__, key_fields, tenant_id)
                key_props_set = ", ".join(f"child.`{k}` = ${k}" for k in key_fields)
                prov_set = _provenance_set_clause("child")
                await session.run(
                    f"""
                    MATCH (parent:{parent_label} {{uid: $parent_uid}})
                    MERGE (child:ModelInstance {{uid: $child_uid}})
                    ON CREATE SET child.model_class = $model_class, {key_props_set}, {prov_set}
                    ON MATCH  SET {key_props_set}, {prov_set}
                    MERGE (parent)-[:`{rel_name}`]->(child)
                    """,
                    parent_uid=parent_uid,
                    child_uid=child_uid,
                    model_class=type(value).__name__,
                    tenant_id=tenant_id,
                    created_by_user_id=created_by_user_id,
                    job_id=job_id,
                    **key_fields,
                )
            else:
                # ModelInstance sin clave: UID posicional, CREATE (comportamiento original)
                child_uid = uuid.uuid4().hex[:16]
                await session.run(
                    f"""
                    MATCH (parent:{parent_label} {{uid: $parent_uid}})
                    CREATE (child:ModelInstance {{
                        uid:         $child_uid,
                        model_class: $model_class,
                        tenant_id:   $tenant_id,
                        created_by_user_ids: CASE WHEN $created_by_user_id IS NULL THEN [] ELSE [$created_by_user_id] END,
                        job_ids:             CASE WHEN $job_id IS NULL THEN [] ELSE [$job_id] END
                    }})
                    CREATE (parent)-[:`{rel_name}`]->(child)
                    """,
                    parent_uid=parent_uid,
                    child_uid=child_uid,
                    model_class=type(value).__name__,
                    tenant_id=tenant_id,
                    created_by_user_id=created_by_user_id,
                    job_id=job_id,
                )
            await _write_model_fields(
                session=session,
                instance=value,
                parent_uid=child_uid,
                parent_label="ModelInstance",
                field_path_prefix=field_path,
                entity_nodes=entity_nodes,
                depth=depth + 1,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            continue

        # ── Case: list ────────────────────────────────────────────────────
        if isinstance(value, list):
            scalarValues = []
            for i, item in enumerate(value):
                if item is None:
                    continue
                item_path = f"{field_path}[{i}]"

                if isinstance(item, BaseModel):
                    key_fields = _get_instance_key_fields(item)
                    rel_name = f"HAS_{_to_rel_name(field_name)}"
                    if key_fields:
                        child_uid = _make_instance_uid(type(item).__name__, key_fields, tenant_id)
                        key_props_set = ", ".join(f"child.`{k}` = ${k}" for k in key_fields)
                        prov_set = _provenance_set_clause("child")
                        await session.run(
                            f"""
                            MATCH (parent:{parent_label} {{uid: $parent_uid}})
                            MERGE (child:ModelInstance {{uid: $child_uid}})
                            ON CREATE SET child.model_class = $model_class, {key_props_set}, {prov_set}
                            ON MATCH  SET {key_props_set}, {prov_set}
                            MERGE (parent)-[:`{rel_name}` {{index: $idx}}]->(child)
                            """,
                            parent_uid=parent_uid,
                            child_uid=child_uid,
                            model_class=type(item).__name__,
                            idx=i,
                            tenant_id=tenant_id,
                            created_by_user_id=created_by_user_id,
                            job_id=job_id,
                            **key_fields,
                        )
                    else:
                        child_uid = uuid.uuid4().hex[:16]
                        await session.run(
                            f"""
                            MATCH (parent:{parent_label} {{uid: $parent_uid}})
                            CREATE (child:ModelInstance {{
                                uid:         $child_uid,
                                model_class: $model_class,
                                tenant_id:   $tenant_id,
                                created_by_user_ids: CASE WHEN $created_by_user_id IS NULL THEN [] ELSE [$created_by_user_id] END,
                                job_ids:             CASE WHEN $job_id IS NULL THEN [] ELSE [$job_id] END
                            }})
                            CREATE (parent)-[:`{rel_name}` {{index: $idx}}]->(child)
                            """,
                            parent_uid=parent_uid,
                            child_uid=child_uid,
                            model_class=type(item).__name__,
                            idx=i,
                            tenant_id=tenant_id,
                            created_by_user_id=created_by_user_id,
                            job_id=job_id,
                        )
                    await _write_model_fields(
                        session=session,
                        instance=item,
                        parent_uid=child_uid,
                        parent_label="ModelInstance",
                        field_path_prefix=item_path,
                        entity_nodes=entity_nodes,
                        depth=depth + 1,
                        tenant_id=tenant_id,
                        created_by_user_id=created_by_user_id,
                        job_id=job_id,
                        list_index=i,
                    )
                else:
                    if isinstance(item, str) and entity_label:
                        # list[str] with entity_label
                        le_uid = await _merge_labeled_entity(
                            session,
                            entity_label,
                            str(item),
                            tenant_id=tenant_id,
                            created_by_user_id=created_by_user_id,
                            job_id=job_id,
                        )
                        await session.run(
                            f"""
                            MATCH (parent:{parent_label} {{uid: $parent_uid}})
                            MATCH (le:LabeledEntity {{uid: $le_uid}})
                            MERGE (parent)-[:REFERENCES {{field_name: $field_name, list_index: $idx}}]->(le)
                            """,
                            parent_uid=parent_uid,
                            le_uid=le_uid,
                            field_name=field_name,
                            idx=i,
                        )
                        entity_nodes[item_path] = le_uid
                    if isinstance(item, dict):
                        log.warning(
                            "_write_model_fields: flattening unexpected dict item at "
                            "field_path=%r to a string (Neo4j cannot store nested Maps)",
                            item_path,
                        )
                    scalarValues.append(
                        _stringify_if_dict(item)
                    )  # Siempre se insertan lo propiedades en las instancias, aunque se haga referencia a ellas en la labels (casos del if)
            if scalarValues:
                scalar_props[field_name] = scalarValues
            continue

        # ── Case: scalar with entity_label ────────────────────────────────
        # Se guarda tanto como referencia y como propiedad si tiene label.
        if entity_label and isinstance(value, (str, int, float, bool)):
            le_uid = await _merge_labeled_entity(
                session,
                entity_label,
                str(value),
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            await session.run(
                f"""
                MATCH (parent:{parent_label} {{uid: $parent_uid}})
                MATCH (le:LabeledEntity {{uid: $le_uid}})
                MERGE (parent)-[:REFERENCES {{field_name: $field_name}}]->(le)
                """,
                parent_uid=parent_uid,
                le_uid=le_uid,
                field_name=field_name,
            )
            entity_nodes[field_path] = le_uid

        # ── Case: scalar -> accumulate as property ────
        if isinstance(value, (str, int, float, bool)):
            scalar_props[field_name] = value
        elif isinstance(value, dict):
            log.warning(
                "_write_model_fields: flattening unexpected dict value at "
                "field_path=%r to a string (Neo4j cannot store nested Maps)",
                field_path,
            )
            scalar_props[field_name] = _stringify_if_dict(value)

    # ── Level 3: instance_relationships ──────────────────────────────────
    await _apply_instance_relationships(
        session=session,
        instance=instance,
        src_mi_uid=parent_uid,
        src_label=parent_label,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
    )

    # Batch-SET all scalar properties on the parent node
    if scalar_props:
        set_clause = ", ".join(f"parent.`{k}` = ${k}" for k in scalar_props)
        await session.run(
            f"MATCH (parent:{parent_label} {{uid: $parent_uid}}) SET {set_clause}",
            parent_uid=parent_uid,
            **scalar_props,
        )


async def _merge_labeled_entity(
    session,
    label: str,
    value: str,
    *,
    tenant_id: str = PUBLIC_TENANT,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> str:
    """
    MERGE a :LabeledEntity node with the given label and value.

    ``tenant_id`` is folded into the ``uid`` (see :func:`make_instance_uid` for
    the rationale) so entities never merge across tenants; the node's
    ``(label, normalized_value)`` uniqueness is therefore only enforced
    *within* a tenant. Returns the uid of the node.
    """
    normalized = _normalize(value)
    uid = _make_uid("le", tenant_id, label, normalized)
    prov_set = _provenance_set_clause("le")
    _le_params = dict(
        label=label,
        normalized_value=normalized,
        uid=uid,
        value=value,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
    )
    await with_neo4j_retry(
        lambda: session.run(
            f"""
        MERGE (le:LabeledEntity {{uid: $uid}})
        ON CREATE SET le.label = $label, le.normalized_value = $normalized_value,
                      le.value = $value, {prov_set}
        ON MATCH  SET {prov_set}
        """,
            **_le_params,
        )
    )
    return uid


async def _apply_field_relationships(
    session,
    instance: BaseModel,
    entity_nodes: dict[str, str],
    field_path_prefix: str,
) -> None:
    """
    Apply Level 2 field_relationships: for each field that declares field_relationships,
    MERGE the specified relationships between the source LabeledEntity and the target
    LabeledEntity (if both are present in entity_nodes).

    Recurses into nested models.
    """
    if not hasattr(instance, "model_fields"):
        return

    for field_name, field_info in instance.model_fields.items():
        value = getattr(instance, field_name, None)
        if value is None:
            continue

        field_path = f"{field_path_prefix}.{field_name}" if field_path_prefix else field_name
        relationships = _get_field_relationships(field_info)

        if relationships:
            src_uid = entity_nodes.get(field_path)
            if src_uid:
                for rel_def in relationships:
                    to_field = rel_def.get("to_field", "")
                    rel_type = rel_def.get("rel_type", "RELATED_TO")
                    # Build target field path (sibling field, same prefix)
                    if field_path_prefix:
                        target_path = f"{field_path_prefix}.{to_field}"
                    else:
                        target_path = to_field
                    tgt_uid = entity_nodes.get(target_path)
                    if tgt_uid:
                        await session.run(
                            f"""
                            MATCH (src:LabeledEntity {{uid: $src_uid}})
                            MATCH (tgt:LabeledEntity {{uid: $tgt_uid}})
                            MERGE (src)-[:`{rel_type}`]->(tgt)
                            """,
                            src_uid=src_uid,
                            tgt_uid=tgt_uid,
                        )
                        log.debug(
                            "_apply_field_relationships: %s -[%s]-> %s",
                            field_path,
                            rel_type,
                            target_path,
                        )
                    else:
                        log.debug(
                            "_apply_field_relationships: target field %r not in entity_nodes, "
                            "skipping relationship %s",
                            target_path,
                            rel_type,
                        )

        # Recurse into nested models
        if isinstance(value, BaseModel):
            await _apply_field_relationships(session, value, entity_nodes, field_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, BaseModel):
                    await _apply_field_relationships(
                        session, item, entity_nodes, f"{field_path}[{i}]"
                    )


async def _apply_instance_relationships(
    session,
    instance: BaseModel,
    src_mi_uid: str,
    *,
    src_label: str = "ModelInstance",
    tenant_id: str = PUBLIC_TENANT,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """
    Level 3 — Instance Key Relationships.

    For each field in *instance* that declares ``instance_relationships``,
    MERGE the target ModelInstance node (identified by its composite key)
    and MERGE the typed relationship (src_mi)-[:REL_TYPE]->(tgt_mi).

    This enables cross-StructureNode references: when VariationCodeModel lists
    condition_ids=['1','2'], two ConditionModel shell nodes are created/merged
    here with deterministic UIDs.  When ConditionModel is later extracted in a
    child section, its _write_model_fields call will MERGE the same node and
    populate the remaining fields (description, etc.).

    Parameters
    ----------
    session:
        Open Neo4j session.
    instance:
        Pydantic model instance whose fields are inspected for instance_relationships.
    src_mi_uid:
        UID of the ModelInstance (or ExtractionResult) node that *owns* this
        instance — it becomes the source of the typed relationship.
    src_label:
        Label of that node (``"ModelInstance"`` or ``"ExtractionResult"``).
    """
    _check_parent_label(src_label)
    if not hasattr(instance, "model_fields"):
        return

    for field_name, field_info in instance.model_fields.items():
        rel_defs = _get_instance_relationships(field_info)
        if not rel_defs:
            continue

        value = getattr(instance, field_name, None)
        if value is None:
            continue

        for rel_def in rel_defs:
            target_model: str = rel_def.get("target_model", "")
            join_via: dict[str, str] = rel_def.get("join_via", {})
            rel_type: str = rel_def.get("rel_type", "RELATED_TO")

            if not target_model or not join_via:
                log.warning(
                    "_apply_instance_relationships: campo %r tiene rel_def incompleto: %r",
                    field_name,
                    rel_def,
                )
                continue

            # ── Separar join_via en campo fan-out (la lista anotada) y campos fijos ──
            fixed_key_fields: dict[str, str] = {}
            fanout_remote_field: str | None = None

            empty_join_key = False
            for local_field, remote_field in join_via.items():
                if local_field == field_name:
                    # El campo anotado es la lista → fan-out
                    fanout_remote_field = remote_field
                else:
                    # Campo escalar fijo de la misma instancia
                    fixed_val = getattr(instance, local_field, None)
                    if (
                        fixed_val is None or str(fixed_val).strip() == ""
                    ):  # Si es empty string o none
                        log.warning(
                            "_apply_instance_relationships: campo fijo %r es None, "
                            "no se crearán relaciones para %r → %r",
                            local_field,
                            field_name,
                            rel_def,
                        )
                        empty_join_key = True
                        break
                    else:
                        fixed_key_fields[remote_field] = _normalize(str(fixed_val))
            if empty_join_key:
                continue
            if fanout_remote_field is None:
                log.warning(
                    "_apply_instance_relationships: join_via de campo %r no incluye "
                    "el propio campo como clave fan-out; skipping rel_def %r",
                    field_name,
                    rel_def,
                )
                continue

            # ── Fan-out: un ModelInstance target por cada item de la lista ──
            items = value if isinstance(value, list) else [value]
            for item in items:
                if item is None:
                    continue

                tgt_key_fields: dict[str, str] = {
                    **fixed_key_fields,
                    fanout_remote_field: _normalize(str(item)),
                }
                tgt_uid = _make_instance_uid(target_model, tgt_key_fields, tenant_id)

                # MERGE del nodo target (shell con solo las keys si es nuevo)
                key_set = ", ".join(f"tgt.`{k}` = ${k}" for k in tgt_key_fields)
                prov_set = _provenance_set_clause("tgt")
                merge_params: dict = {
                    "tgt_uid": tgt_uid,
                    "model_class": target_model,
                    "tenant_id": tenant_id,
                    "created_by_user_id": created_by_user_id,
                    "job_id": job_id,
                    **tgt_key_fields,
                }
                await with_neo4j_retry(
                    lambda p=merge_params, ks=key_set, ps=prov_set: session.run(
                        f"""
                    MERGE (tgt:ModelInstance {{uid: $tgt_uid}})
                    ON CREATE SET tgt.model_class = $model_class, {ks}, {ps}
                    ON MATCH  SET {ks}, {ps}
                    """,
                        **p,
                    )
                )

                # MERGE de la relación (src)-[:REL_TYPE]->(tgt)
                rel_params = {"src_uid": src_mi_uid, "tgt_uid": tgt_uid}
                await with_neo4j_retry(
                    lambda p=rel_params: session.run(
                        f"""
                    MATCH (src:{src_label} {{uid: $src_uid}})
                    MATCH (tgt:ModelInstance {{uid: $tgt_uid}})
                    MERGE (src)-[:`{rel_type}`]->(tgt)
                    """,
                        **p,
                    )
                )
                log.debug(
                    "_apply_instance_relationships: %s -[%s]-> %s(%r)",
                    src_mi_uid[:8],
                    rel_type,
                    target_model,
                    tgt_key_fields,
                )


# ---------------------------------------------------------------------------
# Triple (fallback) writer
# ---------------------------------------------------------------------------


def _normalize_rel_type(predicate: str) -> str:
    """
    Normalize a free-text predicate to a valid Neo4j relationship type in UPPER_SNAKE_CASE.

    Steps:
    1. Lowercase and strip leading/trailing whitespace.
    2. Strip unicode accents (NFKD normalization).
    3. Replace any character that is not alphanumeric or whitespace with a single underscore.
    4. Replace all whitespace sequences with a single underscore.
    5. Collapse consecutive underscores into one.
    6. Strip leading and trailing underscores.
    7. Uppercase the result.
    8. If the result is empty after normalization, return "RELATED_TO" as a safe default.

    Examples:
        "is manufactured by"      → "IS_MANUFACTURED_BY"
        "contains active ingredient" → "CONTAINS_ACTIVE_INGREDIENT"
        "has (primary) use"       → "HAS_PRIMARY_USE"
        "α-helix forms"           → "HELIX_FORMS"   (accent stripped, leading _ removed)
    """
    # Step 1: lowercase + strip
    text = predicate.strip().lower()
    # Step 2: strip accents
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Step 3: non-alphanumeric, non-whitespace → underscore
    text = re.sub(r"[^a-z0-9\s]", "_", text)
    # Step 4: whitespace → underscore
    text = re.sub(r"\s+", "_", text)
    # Step 5: collapse consecutive underscores
    text = re.sub(r"_+", "_", text)
    # Step 6: strip edge underscores
    text = text.strip("_")
    # Step 7: uppercase
    text = text.upper()
    # Step 8: safe default
    return text if text else "RELATED_TO"


async def write_triple_subgraph(
    driver: AsyncDriver,
    node_full_id: str,
    triple_instance: BaseModel,
    document_name: str,
    extraction_uid: str,
) -> None:
    """
    Write the Triple extraction subgraph for one StructureNode to Neo4j.

    Used for nodes where ModelDecision.matched_model_class IS NULL (no specific
    domain model matched). Extracts all subject-predicate-object statements and
    represents them as a graph of :Entity nodes.

    Graph produced:
      (:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult {model_class: "Triple"})
      (:ExtractionResult)-[:HAS_ENTITY]->(:Entity {value, normalized_value, uid})
      (:Entity {subject})-[:NORMALIZED_PRED {predicate_raw}]->(:Entity {object})

    Entity nodes are per-tenant singletons (MERGEd by uid = hash of tenant_id +
    normalized_value) — they are reused across extractions of the same tenant
    for the same canonical value, never across tenants. The ExtractionResult
    and its HAS_ENTITY relationships are re-created fresh on each run (idempotent
    via DELETE of the old ExtractionResult).

    Parameters
    ----------
    driver:
        Open Neo4j driver.
    node_full_id:
        The StructureNode.id composite key.
    triple_instance:
        Populated instance of the Triple Pydantic model (has a `triples` field
        which is a list of TripleItem objects, each with subject/predicate/object).
    document_name:
        Document.name for provenance.
    extraction_uid:
        Deterministic UID for the ExtractionResult node.
    """
    from datetime import datetime

    timestamp = datetime.now(UTC).isoformat()

    # Collect all TripleItem instances from the `triples` field
    triple_items = getattr(triple_instance, "triples", None) or []
    cfg = get_config()
    async with driver.session(database=cfg.neo4j_database) as session:
        # ── Guard: verify StructureNode exists, read its provenance ───────
        # (see write_extraction_subgraph for why provenance is read back here)
        result = await session.run(
            "MATCH (n:StructureNode {id: $nid}) "
            "RETURN count(n) AS cnt, n.tenant_id AS tenant_id, "
            "n.created_by_user_id AS created_by_user_id, n.job_id AS job_id",
            nid=node_full_id,
        )
        rec = await result.single()
        cnt = rec["cnt"] if rec else 0
        if cnt == 0:
            raise RuntimeError(f"write_triple_subgraph: StructureNode not found: {node_full_id!r}")
        tenant_id = _require_stored_tenant(rec["tenant_id"], node_full_id)
        created_by_user_id = rec["created_by_user_id"]
        job_id = rec["job_id"]

        # ── Idempotency: delete stale ExtractionResult ────────────────────
        # DETACH DELETE removes HAS_EXTRACTION and HAS_ENTITY relationships
        # but leaves :Entity nodes intact (they are deduplicated within a tenant).
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
            DETACH DELETE er
            """,
            nid=node_full_id,
        )

        # ── Create new ExtractionResult node ──────────────────────────────
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})
            CREATE (er:ExtractionResult {
                uid:           $uid,
                node_full_id:  $nid,
                document_name: $doc_name,
                model_class:   'Triple',
                timestamp:     $timestamp,
                tenant_id:            $tenant_id,
                created_by_user_id:   $created_by_user_id,
                job_id:               $job_id
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=node_full_id,
            uid=extraction_uid,
            doc_name=document_name,
            timestamp=timestamp,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )

        # ── Process each TripleItem ────────────────────────────────────────
        for item in triple_items:
            subject_val = getattr(item, "subject", None)
            predicate_val = getattr(item, "predicate", None)
            object_val = getattr(item, "object", None)

            if not subject_val or not predicate_val or not object_val:
                log.warning(
                    "write_triple_subgraph: skipping incomplete triple "
                    "(subject=%r, predicate=%r, object=%r) for node %r",
                    subject_val,
                    predicate_val,
                    object_val,
                    node_full_id,
                )
                continue

            # Normalize and build UIDs (tenant folded in — see make_instance_uid)
            subj_norm = _normalize(subject_val)
            obj_norm = _normalize(object_val)
            subj_uid = _make_uid("entity", tenant_id, subj_norm)
            obj_uid = _make_uid("entity", tenant_id, obj_norm)
            rel_type = _normalize_rel_type(predicate_val)
            entity_prov_set = _provenance_set_clause("e")

            # MERGE subject :Entity node
            _subj_params = dict(
                normalized_value=subj_norm,
                uid=subj_uid,
                value=subject_val,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            await with_neo4j_retry(
                lambda: session.run(
                    f"""
                MERGE (e:Entity {{uid: $uid}})
                ON CREATE SET e.normalized_value = $normalized_value,
                              e.value = $value, {entity_prov_set}
                ON MATCH  SET {entity_prov_set}
                """,
                    **_subj_params,
                )
            )

            # MERGE object :Entity node
            _obj_params = dict(
                normalized_value=obj_norm,
                uid=obj_uid,
                value=object_val,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            await with_neo4j_retry(
                lambda: session.run(
                    f"""
                MERGE (e:Entity {{uid: $uid}})
                ON CREATE SET e.normalized_value = $normalized_value,
                              e.value = $value, {entity_prov_set}
                ON MATCH  SET {entity_prov_set}
                """,
                    **_obj_params,
                )
            )

            # Link subject and object to ExtractionResult via HAS_ENTITY
            await session.run(
                """
                MATCH (er:ExtractionResult {uid: $er_uid})
                MATCH (subj:Entity {uid: $subj_uid})
                MERGE (er)-[:HAS_ENTITY {role: 'subject'}]->(subj)
                """,
                er_uid=extraction_uid,
                subj_uid=subj_uid,
            )
            await session.run(
                """
                MATCH (er:ExtractionResult {uid: $er_uid})
                MATCH (obj:Entity {uid: $obj_uid})
                MERGE (er)-[:HAS_ENTITY {role: 'object'}]->(obj)
                """,
                er_uid=extraction_uid,
                obj_uid=obj_uid,
            )

            # MERGE the predicate relationship between subject and object entities
            await session.run(
                f"""
                MATCH (subj:Entity {{uid: $subj_uid}})
                MATCH (obj:Entity {{uid: $obj_uid}})
                MERGE (subj)-[r:`{rel_type}`]->(obj)
                ON CREATE SET r.predicate_raw = $predicate_raw
                """,
                subj_uid=subj_uid,
                obj_uid=obj_uid,
                predicate_raw=predicate_val,
            )

    log.info(
        "write_triple_subgraph: wrote ExtractionResult %s for node %r (%d triples)",
        extraction_uid,
        node_full_id,
        len(triple_items),
    )
