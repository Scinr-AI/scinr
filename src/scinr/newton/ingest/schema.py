"""
ingest/schema.py — Neo4j schema setup (constraints and indexes).

Call ``setup_schema(driver)`` once before any data is loaded to ensure all
unique constraints, regular indexes, and fulltext indexes are in place.

Schema overview
---------------
Unique constraints (MERGE targets):
    :Document(tenant_id, path, version)  — composite key; the tenant is part of
                              the document identity (``"__public__"`` for public
                              documents, see ``utils/tenancy.py``)
    :StructureNode(id)        — id is tenant-prefixed (see ``ingest/nodes.py``)
    :InfoUnit(uid)
    :ExtractionResult(uid)
    :ModelInstance(uid)       — uid hashes the tenant (``utils/uid.py``)
    :ModelField(name, model)  — composite key; prevents cross-model field node sharing
    :LabeledEntity(uid)       — uid hashes (tenant, label, normalized_value)
    :Entity(uid)              — uid hashes (tenant, normalized_value)
    :EntityLabel(label)       — schema-level singleton per entity label string

Node existence constraints:
    :Document(name)
    :StructureNode(id)
    :InfoUnit(uid)

Regular indexes (query performance):
    :Document(name)
    :Document(latest)
    :Document(path)
    :Document(tenant_id)
    :Document(created_by_user_id)
    :Document(job_id)
    :StructureNode(role)
    :StructureNode(source_page_ids)
    :StructureNode(tenant_id)
    :StructureNode(created_by_user_id)
    :StructureNode(job_id)
    :LabeledEntity(label)
    :LabeledEntity(tenant_id)
    :Entity(tenant_id)
    :ExtractionResult(node_full_id)
    :ExtractionResult(tenant_id)
    :ExtractionResult(created_by_user_id)
    :ExtractionResult(job_id)
    :ModelInstance(model_class)
    :ModelInstance(tenant_id)
    :InfoUnit(tenant_id)
    :ModelDecision(tenant_id)
    :ModelDecision(uid), :ProposedModel(uid), :ProposedField(uid),
    :ComplementaryMatch(uid), :SupplementaryField(uid)
        — lookups by uid (restore_document(), the tabular ModelDecision);
        plain indexes, not constraints: legacy graphs may hold duplicates
    :CatalogModel(name)
        — the MERGE of the catalog model by name (annotation, extraction,
        restore_document())

Composite tenant indexes (tenant-scoped filters on a second property):
    :ModelInstance(tenant_id, model_class)
    :Document(tenant_id, latest)
    :Document(tenant_id, name)
    :StructureNode(tenant_id, role)
    :LabeledEntity(tenant_id, label)

    The single-property indexes above are kept: a query over all tenants
    (``tenant_id=None``) has no tenant predicate and cannot use a composite
    index, which Neo4j only uses when every indexed property is constrained.

Fulltext indexes (semantic search):
    infoUnitDescription  → :InfoUnit(description)
    infoUnitTitle        → :InfoUnit(title)
"""

import logging
import re

from neo4j import Driver

from scinr.newton.config import get_config
from scinr.newton.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

DOCUMENT_KEY_CONSTRAINT = "constraint_document_tenant_path_version"
"""Name of the ``(tenant_id, path, version)`` uniqueness constraint on :Document
(lower-case, so it can be searched for in a lower-cased ``ConstraintError``)."""

_LEGACY_CONSTRAINTS = (
    # Name-only unique constraint, replaced by (path, version).
    "constraint_document_name",
    # (path, version), replaced by DOCUMENT_KEY_CONSTRAINT when the tenant
    # became part of the document identity.
    "constraint_document_path_version",
)

# ---------------------------------------------------------------------------
# DDL statements — all use IF NOT EXISTS so they are idempotent
# ---------------------------------------------------------------------------

_UNIQUE_CONSTRAINTS: list[tuple[str, str]] = [
    (
        DOCUMENT_KEY_CONSTRAINT,
        f"CREATE CONSTRAINT {DOCUMENT_KEY_CONSTRAINT} IF NOT EXISTS "
        "FOR (d:Document) REQUIRE (d.tenant_id, d.path, d.version) IS UNIQUE",
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
        "idx_document_tenant_id",
        "CREATE INDEX idx_document_tenant_id IF NOT EXISTS "
        "FOR (d:Document) ON (d.tenant_id)",
    ),
    (
        "idx_document_created_by_user_id",
        "CREATE INDEX idx_document_created_by_user_id IF NOT EXISTS "
        "FOR (d:Document) ON (d.created_by_user_id)",
    ),
    (
        "idx_document_job_id",
        "CREATE INDEX idx_document_job_id IF NOT EXISTS "
        "FOR (d:Document) ON (d.job_id)",
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
        "idx_structure_node_tenant_id",
        "CREATE INDEX idx_structure_node_tenant_id IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.tenant_id)",
    ),
    (
        "idx_structure_node_created_by_user_id",
        "CREATE INDEX idx_structure_node_created_by_user_id IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.created_by_user_id)",
    ),
    (
        "idx_structure_node_job_id",
        "CREATE INDEX idx_structure_node_job_id IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.job_id)",
    ),
    (
        "idx_labeled_entity_label",
        "CREATE INDEX idx_labeled_entity_label IF NOT EXISTS "
        "FOR (le:LabeledEntity) ON (le.label)",
    ),
    (
        "idx_labeled_entity_tenant_id",
        "CREATE INDEX idx_labeled_entity_tenant_id IF NOT EXISTS "
        "FOR (le:LabeledEntity) ON (le.tenant_id)",
    ),
    (
        "idx_entity_tenant_id",
        "CREATE INDEX idx_entity_tenant_id IF NOT EXISTS "
        "FOR (e:Entity) ON (e.tenant_id)",
    ),
    (
        "idx_extraction_result_node",
        "CREATE INDEX idx_extraction_result_node_full_id IF NOT EXISTS "
        "FOR (e:ExtractionResult) ON (e.node_full_id)",
    ),
    (
        "idx_extraction_result_tenant_id",
        "CREATE INDEX idx_extraction_result_tenant_id IF NOT EXISTS "
        "FOR (e:ExtractionResult) ON (e.tenant_id)",
    ),
    (
        "idx_extraction_result_created_by_user_id",
        "CREATE INDEX idx_extraction_result_created_by_user_id IF NOT EXISTS "
        "FOR (e:ExtractionResult) ON (e.created_by_user_id)",
    ),
    (
        "idx_extraction_result_job_id",
        "CREATE INDEX idx_extraction_result_job_id IF NOT EXISTS "
        "FOR (e:ExtractionResult) ON (e.job_id)",
    ),
    (
        "idx_model_instance_model_class",
        "CREATE INDEX idx_model_instance_model_class IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.model_class)",
    ),
    (
        "idx_model_instance_tenant_id",
        "CREATE INDEX idx_model_instance_tenant_id IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.tenant_id)",
    ),
    (
        "idx_info_unit_tenant_id",
        "CREATE INDEX idx_info_unit_tenant_id IF NOT EXISTS "
        "FOR (u:InfoUnit) ON (u.tenant_id)",
    ),
    (
        "idx_model_decision_tenant_id",
        "CREATE INDEX idx_model_decision_tenant_id IF NOT EXISTS "
        "FOR (md:ModelDecision) ON (md.tenant_id)",
    ),
    (
        "idx_model_decision_uid",
        "CREATE INDEX idx_model_decision_uid IF NOT EXISTS "
        "FOR (n:ModelDecision) ON (n.uid)",
    ),
    (
        "idx_proposed_model_uid",
        "CREATE INDEX idx_proposed_model_uid IF NOT EXISTS "
        "FOR (n:ProposedModel) ON (n.uid)",
    ),
    (
        "idx_proposed_field_uid",
        "CREATE INDEX idx_proposed_field_uid IF NOT EXISTS "
        "FOR (n:ProposedField) ON (n.uid)",
    ),
    (
        "idx_complementary_match_uid",
        "CREATE INDEX idx_complementary_match_uid IF NOT EXISTS "
        "FOR (n:ComplementaryMatch) ON (n.uid)",
    ),
    (
        "idx_supplementary_field_uid",
        "CREATE INDEX idx_supplementary_field_uid IF NOT EXISTS "
        "FOR (n:SupplementaryField) ON (n.uid)",
    ),
    (
        "idx_catalog_model_name",
        "CREATE INDEX idx_catalog_model_name IF NOT EXISTS "
        "FOR (n:CatalogModel) ON (n.name)",
    ),
    # -- Composite tenant indexes ------------------------------------------
    # Neo4j uses one index per node, so "tenant T and property P" with two
    # single indexes seeks on one and filters every node for the other.
    (
        "idx_model_instance_tenant_model_class",
        "CREATE INDEX idx_model_instance_tenant_model_class IF NOT EXISTS "
        "FOR (mi:ModelInstance) ON (mi.tenant_id, mi.model_class)",
    ),
    (
        "idx_document_tenant_latest",
        "CREATE INDEX idx_document_tenant_latest IF NOT EXISTS "
        "FOR (d:Document) ON (d.tenant_id, d.latest)",
    ),
    (
        "idx_document_tenant_name",
        "CREATE INDEX idx_document_tenant_name IF NOT EXISTS "
        "FOR (d:Document) ON (d.tenant_id, d.name)",
    ),
    (
        "idx_structure_node_tenant_role",
        "CREATE INDEX idx_structure_node_tenant_role IF NOT EXISTS "
        "FOR (n:StructureNode) ON (n.tenant_id, n.role)",
    ),
    (
        "idx_labeled_entity_tenant_label",
        "CREATE INDEX idx_labeled_entity_tenant_label IF NOT EXISTS "
        "FOR (le:LabeledEntity) ON (le.tenant_id, le.label)",
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

    Also drops the legacy :Document constraints if they exist
    (``constraint_document_name`` and ``constraint_document_path_version``,
    both superseded by the ``(tenant_id, path, version)`` composite
    constraint). The old ``(path, version)`` one must go: it would reject two
    tenants ingesting the same path.

    Args:
        driver: An open, authenticated :class:`neo4j.Driver` instance.
    """
    # ------------------------------------------------------------------
    # Best-effort Neo4j minimum version check (>= 4.4 required)
    # ------------------------------------------------------------------
    cfg = get_config()
    with driver.session(database=cfg.neo4j_database) as session:
        try:
            record = session.run(
                # Since 2025.x the procedure also lists the "Cypher" component
                # (versions ["5", "25"]): keep only the kernel row.
                "CALL dbms.components() YIELD name, versions "
                "WHERE name = 'Neo4j Kernel' RETURN versions[0] AS version"
            ).single()
            if record:
                version_str = str(record["version"])
                parts = version_str.split(".")
                major, minor = int(parts[0]), int(parts[1])
                if (major, minor) < (4, 4):
                    raise ConfigurationError(
                        f"Neo4j {version_str} is not supported. "
                        "scinr-ingest requires Neo4j >= 4.4. "
                        "Please upgrade your Neo4j instance."
                    )
            else:
                # Empty result (por ejemplo, Aura / formato no estándar).
                # Best-effort: no bloqueamos la conexión.
                logger.warning(
                    "Could not determine Neo4j version (empty result); skipping compatibility check."
                )
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.warning("Could not verify Neo4j version: %s", exc)

    # Drop legacy :Document constraints if they exist (see _LEGACY_CONSTRAINTS)
    with driver.session(database=cfg.neo4j_database) as session:
        for legacy in _LEGACY_CONSTRAINTS:
            try:
                session.execute_write(
                    lambda tx, q=f"DROP CONSTRAINT {legacy} IF EXISTS": tx.run(q)
                )
                logger.info("Dropped legacy %s if it existed.", legacy)
            except Exception as exc:
                logger.warning("Could not drop %s: %s", legacy, exc)

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


def ensure_indexes(
    driver: Driver, names: tuple[str, ...], *, database: str, await_timeout: int = 300
) -> int:
    """Create the named indexes of :data:`_REGULAR_INDEXES` if missing (idempotent).

    For operations that depend on indexes added after a graph was ingested
    (``restore_document()`` looks nodes up by ``uid``): ``setup_schema()``
    only runs at ingestion time. When at least one index is created, waits up
    to *await_timeout* seconds for it to come ONLINE, so the caller's
    queries can use it.

    Best-effort: a failure (e.g. a user without schema privileges) is logged
    and the caller carries on — only slower.

    Returns:
        The number of indexes created.
    """
    statements = dict(_REGULAR_INDEXES)
    unknown = [name for name in names if name not in statements]
    if unknown:
        raise ValueError(f"Unknown index name(s): {unknown!r}")
    created = 0
    try:
        with driver.session(database=database) as session:
            for name in names:
                summary = session.execute_write(
                    lambda tx, q=statements[name]: tx.run(q).consume()
                )
                if summary.counters.indexes_added:
                    logger.info("Created missing index %s.", name)
                    created += summary.counters.indexes_added
            if created:
                session.run("CALL db.awaitIndexes($timeout)", timeout=await_timeout).consume()
    except Exception as exc:
        logger.warning("Could not ensure indexes %s: %s", names, exc)
    return created
