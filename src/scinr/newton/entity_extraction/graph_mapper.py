"""
entity_extraction/graph_mapper.py

Converts a populated Pydantic instance (ExtractionModel or CompositeSchema) into
a Neo4j subgraph using three relationship mechanisms:

  Level 1 — Entity labeling:
    Fields with json_schema_extra={"entity_label": "X"} become MERGE'd
    (:X {label, value, normalized_value}) nodes. Same label + same
    normalized_value always resolves to the same node across all extractions.

  Level 2 — Field relationships:
    Fields with json_schema_extra={"field_relationships": [{"to_field": "...", "rel_type": "..."}]}
    trigger MERGE relationships between the source entity node and the
    target entity node (target must also have entity_label).

  Level 3 — Instance Key Relationships:
    Fields with json_schema_extra={"instance_key": True} define a composite
    key that makes ModelInstance nodes globally deduplicatable (UID =
    make_instance_uid(model_class, {sorted key_fields})). Analogous to
    LabeledEntity deduplication by (label, normalized_value).

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
from datetime import UTC
from typing import Any

from neo4j import AsyncDriver
from pydantic import BaseModel

from scinr.newton.utils.neo4j_retry import with_neo4j_retry
from scinr.newton.utils.uid import make_instance_uid as _make_instance_uid
from scinr.newton.utils.uid import make_uid as _make_uid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(value: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().lower()


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

    async with driver.session() as session:
        # ── Guard: verify StructureNode exists ────────────────────────────
        result = await session.run(
            "MATCH (n:StructureNode {id: $nid}) RETURN count(n) AS cnt",
            nid=node_full_id,
        )
        cnt = (await result.single())["cnt"]
        if cnt == 0:
            raise RuntimeError(
                f"write_extraction_subgraph: StructureNode not found: {node_full_id!r}"
            )

        # ── Idempotency: delete stale ExtractionResult subgraph ───────────
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
            OPTIONAL MATCH (er)-[*1..10]->(child:ModelInstance)
            DETACH DELETE child
            WITH er
            DETACH DELETE er
            """,
            nid=node_full_id,
        )

        # ── Create ExtractionResult node ──────────────────────────────────
        await session.run(
            """
            MATCH (n:StructureNode {id: $nid})
            CREATE (er:ExtractionResult {
                uid:           $uid,
                node_full_id:  $nid,
                document_name: $doc_name,
                model_class:   $model_class,
                timestamp:     $timestamp
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=node_full_id,
            uid=extraction_uid,
            doc_name=document_name,
            model_class=primary_model_class,
            timestamp=timestamp,
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


async def _write_model_fields(
    session,
    instance: BaseModel,
    parent_uid: str,
    parent_label: str,
    field_path_prefix: str,
    entity_nodes: dict[str, str],
    depth: int,
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
        Neo4j label of the parent node (used for SET properties).
    field_path_prefix:
        Dot-separated prefix for entity_nodes registry keys.
    entity_nodes:
        Accumulator: maps "field_path" → labeled_entity_uid for Level 2 resolution.
    depth:
        Current recursion depth (safety cap at 10).
    list_index:
        If this instance is an element of a list, its 0-based index.
    """
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
                # ModelInstance con clave compuesta: UID determinístico, MERGE
                child_uid = _make_instance_uid(type(value).__name__, key_fields)
                key_props_set = ", ".join(f"child.`{k}` = ${k}" for k in key_fields)
                await session.run(
                    f"""
                    MATCH (parent {{uid: $parent_uid}})
                    MERGE (child:ModelInstance {{uid: $child_uid}})
                    ON CREATE SET child.model_class = $model_class, {key_props_set}
                    ON MATCH  SET {key_props_set}
                    MERGE (parent)-[:`{rel_name}`]->(child)
                    """,
                    parent_uid=parent_uid,
                    child_uid=child_uid,
                    model_class=type(value).__name__,
                    **key_fields,
                )
            else:
                # ModelInstance sin clave: UID posicional, CREATE (comportamiento original)
                child_uid = uuid.uuid4().hex[:16]
                await session.run(
                    f"""
                    MATCH (parent {{uid: $parent_uid}})
                    CREATE (child:ModelInstance {{
                        uid:         $child_uid,
                        model_class: $model_class
                    }})
                    CREATE (parent)-[:`{rel_name}`]->(child)
                    """,
                    parent_uid=parent_uid,
                    child_uid=child_uid,
                    model_class=type(value).__name__,
                )
            await _write_model_fields(
                session=session,
                instance=value,
                parent_uid=child_uid,
                parent_label="ModelInstance",
                field_path_prefix=field_path,
                entity_nodes=entity_nodes,
                depth=depth + 1,
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
                        child_uid = _make_instance_uid(type(item).__name__, key_fields)
                        key_props_set = ", ".join(f"child.`{k}` = ${k}" for k in key_fields)
                        await session.run(
                            f"""
                            MATCH (parent {{uid: $parent_uid}})
                            MERGE (child:ModelInstance {{uid: $child_uid}})
                            ON CREATE SET child.model_class = $model_class, {key_props_set}
                            ON MATCH  SET {key_props_set}
                            MERGE (parent)-[:`{rel_name}` {{index: $idx}}]->(child)
                            """,
                            parent_uid=parent_uid,
                            child_uid=child_uid,
                            model_class=type(item).__name__,
                            idx=i,
                            **key_fields,
                        )
                    else:
                        child_uid = uuid.uuid4().hex[:16]
                        await session.run(
                            f"""
                            MATCH (parent {{uid: $parent_uid}})
                            CREATE (child:ModelInstance {{
                                uid:         $child_uid,
                                model_class: $model_class
                            }})
                            CREATE (parent)-[:`{rel_name}` {{index: $idx}}]->(child)
                            """,
                            parent_uid=parent_uid,
                            child_uid=child_uid,
                            model_class=type(item).__name__,
                            idx=i,
                        )
                    await _write_model_fields(
                        session=session,
                        instance=item,
                        parent_uid=child_uid,
                        parent_label="ModelInstance",
                        field_path_prefix=item_path,
                        entity_nodes=entity_nodes,
                        depth=depth + 1,
                        list_index=i,
                    )
                else:
                    if isinstance(item, str) and entity_label:
                        # list[str] with entity_label
                        le_uid = await _merge_labeled_entity(session, entity_label, str(item))
                        await session.run(
                            """
                            MATCH (parent {uid: $parent_uid})
                            MATCH (le:LabeledEntity {uid: $le_uid})
                            MERGE (parent)-[:REFERENCES {field_name: $field_name, list_index: $idx}]->(le)
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
            le_uid = await _merge_labeled_entity(session, entity_label, str(value))
            await session.run(
                """
                MATCH (parent {uid: $parent_uid})
                MATCH (le:LabeledEntity {uid: $le_uid})
                MERGE (parent)-[:REFERENCES {field_name: $field_name}]->(le)
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
    )

    # Batch-SET all scalar properties on the parent node
    if scalar_props:
        set_clause = ", ".join(f"parent.`{k}` = ${k}" for k in scalar_props)
        await session.run(
            f"MATCH (parent {{uid: $parent_uid}}) SET {set_clause}",
            parent_uid=parent_uid,
            **scalar_props,
        )


async def _merge_labeled_entity(session, label: str, value: str) -> str:
    """
    MERGE a :LabeledEntity node with the given label and value.
    Returns the uid of the node.
    """
    normalized = _normalize(value)
    uid = _make_uid("le", label, normalized)
    _le_params = dict(label=label, normalized_value=normalized, uid=uid, value=value)
    await with_neo4j_retry(
        lambda: session.run(
            """
        MERGE (le:LabeledEntity {label: $label, normalized_value: $normalized_value})
        ON CREATE SET le.uid   = $uid,
                      le.value = $value
        ON MATCH  SET le.uid   = $uid
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
    """
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
                tgt_uid = _make_instance_uid(target_model, tgt_key_fields)

                # MERGE del nodo target (shell con solo las keys si es nuevo)
                key_set = ", ".join(f"tgt.`{k}` = ${k}" for k in tgt_key_fields)
                merge_params: dict = {
                    "tgt_uid": tgt_uid,
                    "model_class": target_model,
                    **tgt_key_fields,
                }
                await with_neo4j_retry(
                    lambda p=merge_params, ks=key_set: session.run(
                        f"""
                    MERGE (tgt:ModelInstance {{uid: $tgt_uid}})
                    ON CREATE SET tgt.model_class = $model_class, {ks}
                    ON MATCH  SET {ks}
                    """,
                        **p,
                    )
                )

                # MERGE de la relación (src)-[:REL_TYPE]->(tgt)
                rel_params = {"src_uid": src_mi_uid, "tgt_uid": tgt_uid}
                await with_neo4j_retry(
                    lambda p=rel_params: session.run(
                        f"""
                    MATCH (src {{uid: $src_uid}})
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

    Entity nodes are global singletons (MERGEd by normalized_value) — they are
    reused across extractions for the same canonical value. The ExtractionResult
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

    async with driver.session() as session:
        # ── Guard: verify StructureNode exists ────────────────────────────
        result = await session.run(
            "MATCH (n:StructureNode {id: $nid}) RETURN count(n) AS cnt",
            nid=node_full_id,
        )
        cnt = (await result.single())["cnt"]
        if cnt == 0:
            raise RuntimeError(f"write_triple_subgraph: StructureNode not found: {node_full_id!r}")

        # ── Idempotency: delete stale ExtractionResult ────────────────────
        # DETACH DELETE removes HAS_EXTRACTION and HAS_ENTITY relationships
        # but leaves :Entity nodes intact (they are shared global singletons).
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
                timestamp:     $timestamp
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=node_full_id,
            uid=extraction_uid,
            doc_name=document_name,
            timestamp=timestamp,
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

            # Normalize and build UIDs
            subj_norm = _normalize(subject_val)
            obj_norm = _normalize(object_val)
            subj_uid = _make_uid("entity", subj_norm)
            obj_uid = _make_uid("entity", obj_norm)
            rel_type = _normalize_rel_type(predicate_val)

            # MERGE subject :Entity node
            _subj_params = dict(normalized_value=subj_norm, uid=subj_uid, value=subject_val)
            await with_neo4j_retry(
                lambda: session.run(
                    """
                MERGE (e:Entity {uid: $uid})
                ON CREATE SET e.normalized_value = $normalized_value,
                              e.value = $value
                """,
                    **_subj_params,
                )
            )

            # MERGE object :Entity node
            _obj_params = dict(normalized_value=obj_norm, uid=obj_uid, value=object_val)
            await with_neo4j_retry(
                lambda: session.run(
                    """
                MERGE (e:Entity {uid: $uid})
                ON CREATE SET e.normalized_value = $normalized_value,
                              e.value = $value
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
