"""
ingest/schema.py — Neo4j schema setup (constraints and indexes).

Call ``setup_schema(driver)`` once before any data is loaded to ensure all
unique constraints, regular indexes, and fulltext indexes are in place.

Schema overview
---------------
Unique constraints (MERGE targets):
    :Document(path, version)  — composite key
    :StructureNode(id)
    :InfoUnit(uid)
    :ExtractionResult(uid)
    :ModelField(name, model)  — composite key; prevents cross-model field node sharing
    :LabeledEntity(label, normalized_value)  — NODE KEY
    :EntityLabel(label)       — schema-level singleton per entity label string

Node existence constraints:
    :Document(name)
    :StructureNode(id)
    :InfoUnit(uid)

Regular indexes (query performance):
    :Document(name)
    :Document(latest)
    :Document(path)
    :StructureNode(role)
    :StructureNode(source_page_ids)
    :LabeledEntity(label)
    :ExtractionResult(node_full_id)
    :ModelInstance(model_class)

Fulltext indexes (semantic search):
    infoUnitDescription  → :InfoUnit(description)
    infoUnitTitle        → :InfoUnit(title)
"""

import logging
import re

from neo4j import Driver

from scinr.newton.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL statements — all use IF NOT EXISTS so they are idempotent
# ---------------------------------------------------------------------------

_UNIQUE_CONSTRAINTS: list[tuple[str, str]] = [
    (
        "constraint_document_path_version",
        "CREATE CONSTRAINT constraint_document_path_version IF NOT EXISTS "
        "FOR (d:Document) REQUIRE (d.path, d.version) IS UNIQUE",
    ),
    (
        "constraint_structure_node_id",
        "CREATE CONSTRAINT constraint_structure_node_id IF NOT EXISTS "
        "FOR (n:StructureNode) REQUIRE n.id IS UNIQUE",
    ),
    (
        "constraint_info_unit_uid",
        "CREATE CONSTRAINT constraint_info_unit_uid IF NOT EXISTS "
        "FOR (u:InfoUnit) REQUIRE u.uid IS UNIQUE",
    ),
    (
        "constraint_extraction_result_uid",
        "CREATE CONSTRAINT constraint_extraction_result_uid IF NOT EXISTS "
        "FOR (e:ExtractionResult) REQUIRE e.uid IS UNIQUE",
    ),
    (
        "constraint_model_instance_uid",
        "CREATE CONSTRAINT constraint_model_instance_uid IF NOT EXISTS "
        "FOR (mi:ModelInstance) REQUIRE mi.uid IS UNIQUE",
    ),
    # Composite key for ModelField: (name, model) ensures that two different
    # Pydantic models with a field of the same name get distinct ModelField nodes.
    (
        "constraint_model_field_name_model",
        "CREATE CONSTRAINT constraint_model_field_name_model IF NOT EXISTS "
        "FOR (mf:ModelField) REQUIRE (mf.name, mf.model) IS UNIQUE",
    ),
    (
        "constraint_labeled_entity_key",
        "CREATE CONSTRAINT constraint_labeled_entity_key IF NOT EXISTS "
        "FOR (le:LabeledEntity) REQUIRE le.uid IS UNIQUE",
    ),
    (
        "constraint_entity_key",
        "CREATE CONSTRAINT constraint_entity_key IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
    ),
    (
        "constraint_entity_label_label",
        "CREATE CONSTRAINT constraint_entity_label_label IF NOT EXISTS "
        "FOR (el:EntityLabel) REQUIRE el.label IS UNIQUE",
    ),
]

_EXISTENCE_CONSTRAINTS: list[tuple[str, str]] = [
    (
        "constraint_document_name_exists",
        "CREATE CONSTRAINT constraint_document_name_exists IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.name IS NOT NULL",
    ),
    (
        "constraint_structure_node_id_exists",
        "CREATE CONSTRAINT constraint_structure_node_id_exists IF NOT EXISTS "
        "FOR (n:StructureNode) REQUIRE n.id IS NOT NULL",
    ),
    (
        "constraint_info_unit_uid_exists",
        "CREATE CONSTRAINT constraint_info_unit_uid_exists IF NOT EXISTS "
        "FOR (u:InfoUnit) REQUIRE u.uid IS NOT NULL",
    ),
]

_REGULAR_INDEXES: list[tuple[str, str]] = [
    (
        "idx_document_name",
        "CREATE INDEX idx_document_name IF NOT EXISTS "
        "FOR (d:Document) ON (d.name)",
    ),
    (
        "idx_document_latest",
        "CREATE INDEX idx_document_latest IF NOT EXISTS "
        "FOR (d:Document) ON (d.latest)",
    ),
    (
        "idx_document_path",
        "CREATE INDEX idx_document_path IF NOT EXISTS "
        "FOR (d:Document) ON (d.path)",
    ),
    (
        "idx_structure_node_role",
        "CREATE INDEX idx_structure_node_role IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.role)",
    ),
    (
        "idx_structure_node_row_index",
        "CREATE INDEX idx_structure_node_row_index IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.row_index)",
    ),
    (
        "idx_structure_node_source_page_ids",
        "CREATE INDEX structurenode_source_page_ids IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.source_page_ids)",
    ),
    (
        "idx_labeled_entity_label",
        "CREATE INDEX idx_labeled_entity_label IF NOT EXISTS "
        "FOR (le:LabeledEntity) ON (le.label)",
    ),
    (
        "idx_extraction_result_node",
        "CREATE INDEX idx_extraction_result_node_full_id IF NOT EXISTS "
        "FOR (e:ExtractionResult) ON (e.node_full_id)",
    ),
    (
        "idx_model_instance_model_class",
        "CREATE INDEX idx_model_instance_model_class IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.model_class)",
    ),
]

_FULLTEXT_INDEXES: list[tuple[str, str]] = [
    (
        "infoUnitDescription",
        "CREATE FULLTEXT INDEX infoUnitDescription IF NOT EXISTS "
        "FOR (u:InfoUnit) ON EACH [u.description]",
    ),
    (
        "infoUnitTitle",
        "CREATE FULLTEXT INDEX infoUnitTitle IF NOT EXISTS "
        "FOR (u:InfoUnit) ON EACH [u.title]",
    ),
]


def setup_schema(driver: Driver) -> None:
    """Create all constraints and indexes in Neo4j (idempotent).

    Uses ``IF NOT EXISTS`` clauses so this function can be safely called
    multiple times without raising errors on an already-configured database.

    Also drops the legacy ``constraint_document_name`` unique constraint if it
    exists (replaced by the ``(path, version)`` composite constraint).

    Args:
        driver: An open, authenticated :class:`neo4j.Driver` instance.
    """
    # ------------------------------------------------------------------
    # Best-effort Neo4j minimum version check (>= 4.4 required)
    # ------------------------------------------------------------------
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        try:
            result = session.run(
                "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
            )
            record = result.single()
            if record:
                version_str = record["version"]
                parts = version_str.split(".")
                major, minor = int(parts[0]), int(parts[1])
                if (major, minor) < (4, 4):
                    from scinr.newton.exceptions import ConfigurationError
                    raise ConfigurationError(
                        f"Neo4j {version_str} is not supported. "
                        "scinr-ingest requires Neo4j >= 4.4. "
                        "Please upgrade your Neo4j instance."
                    )
            else:
                # Formato desconocido (por ejemplo, Aura: 27-aura).
                # No bloqueamos la conexión.
                logger.warning(
                    "Could not parse Neo4j version %r; skipping compatibility check.",
                    version_str,
                )

    # Drop legacy name-only unique constraint if it exists (replaced by path+version composite)
    with driver.session(database=cfg.neo4j_database) as session:
        try:
            session.execute_write(lambda tx: tx.run(
                "DROP CONSTRAINT constraint_document_name IF EXISTS"
            ))
            logger.info("Dropped legacy constraint_document_name if it existed.")
        except Exception as exc:
            logger.warning("Could not drop constraint_document_name: %s", exc)

    with driver.session(database=cfg.neo4j_database) as session:
        for name, cypher in _UNIQUE_CONSTRAINTS:
            logger.info("Ensuring unique constraint: %s", name)
            session.execute_write(lambda tx, q=cypher: tx.run(q))

        # for name, cypher in _EXISTENCE_CONSTRAINTS:
        #     logger.info("Ensuring existence constraint: %s", name)
        #     session.execute_write(lambda tx, q=cypher: tx.run(q))

        for name, cypher in _REGULAR_INDEXES:
            logger.info("Ensuring index: %s", name)
            session.execute_write(lambda tx, q=cypher: tx.run(q))

        for name, cypher in _FULLTEXT_INDEXES:
            logger.info("Ensuring fulltext index: %s", name)
            session.execute_write(lambda tx, q=cypher: tx.run(q))

    logger.info(
        "Schema setup complete: %d unique constraints, %d indexes, %d fulltext indexes.",
        len(_UNIQUE_CONSTRAINTS),
        len(_REGULAR_INDEXES),
        len(_FULLTEXT_INDEXES),
    )
