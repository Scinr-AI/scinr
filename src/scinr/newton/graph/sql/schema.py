"""
graph/sql/schema.py — ``store.schema`` over SQL: the schema comes from the migrations.

The tables, keys and indexes are created (or verified) by
:func:`scinr.newton.sql.schema.ensure_schema`, from the packaged ``0001_init.sql`` of each
engine: the indexes that Neo4j creates on demand already exist. :meth:`ensure_indexes`
therefore only validates the Neo4j index names it is given.
"""

from __future__ import annotations

from collections.abc import Iterable

from scinr.newton.graph.base import SchemaOps
from scinr.newton.sql.engine import get_dialect_for_config
from scinr.newton.sql.errors import sql_errors

__all__ = ["KNOWN_INDEXES", "SqlSchemaOps"]

#: The index names of ``graph/neo4j/schema.py`` (``_REGULAR_INDEXES``); a unit test keeps
#: both lists equal. The SQL schema has an equivalent for each of them.
KNOWN_INDEXES: frozenset[str] = frozenset(
    {
        "idx_appendix_tenant_id",
        "idx_catalog_model_name",
        "idx_complementary_match_tenant_id",
        "idx_complementary_match_uid",
        "idx_document_created_by_user_id",
        "idx_document_job_id",
        "idx_document_latest",
        "idx_document_name",
        "idx_document_path",
        "idx_document_tenant_id",
        "idx_document_tenant_latest",
        "idx_document_tenant_name",
        "idx_entity_tenant_id",
        "idx_extraction_result_created_by_user_id",
        "idx_extraction_result_job_id",
        "idx_extraction_result_node",
        "idx_extraction_result_tenant_id",
        "idx_extraction_result_tenant_model_class",
        "idx_field_group_tenant_id",
        "idx_freeform_block_tenant_id",
        "idx_info_unit_tenant_id",
        "idx_labeled_entity_label",
        "idx_labeled_entity_tenant_id",
        "idx_labeled_entity_tenant_label",
        "idx_model_decision_tenant_id",
        "idx_model_decision_uid",
        "idx_model_instance_model_class",
        "idx_model_instance_tenant_id",
        "idx_model_instance_tenant_model_class",
        "idx_proposed_field_tenant_id",
        "idx_proposed_field_uid",
        "idx_proposed_model_tenant_id",
        "idx_proposed_model_uid",
        "idx_row_tenant_id",
        "idx_section_tenant_id",
        "idx_structure_node_created_by_user_id",
        "idx_structure_node_job_id",
        "idx_structure_node_role",
        "idx_structure_node_row_index",
        "idx_structure_node_source_page_ids",
        "idx_structure_node_tenant_id",
        "idx_structure_node_tenant_role",
        "idx_subsection_tenant_id",
        "idx_supplementary_field_tenant_id",
        "idx_supplementary_field_uid",
        "idx_table_tenant_id",
        "idx_theme_path",
    }
)


class SqlSchemaOps(SchemaOps):
    """``store.schema`` over SQL (both operations are synchronous)."""

    def setup(self) -> None:
        """Create, migrate or verify the schema (``ensure_schema()``); nothing else.

        Cypher: creates every constraint and index and checks the server version.

        Raises:
            ConfigurationError: If the schema cannot be created or is not at the
                version this newton needs.
            GraphConfigurationError: If ``sql_url`` is not configured.
            GraphError: If the engine fails.
        """
        from scinr.newton.sql.schema import ensure_schema

        with sql_errors():
            ensure_schema()

    def ensure_indexes(self, names: Iterable[str]) -> int:
        """Validate the index names and return ``0``: every index exists after ``setup()``.

        Cypher: creates the named indexes if they are missing and waits for them.

        Raises:
            ValueError: If a name is not a known index (same message as Neo4j).
            GraphConfigurationError: If ``sql_url`` is not configured.
        """
        get_dialect_for_config()
        names = tuple(names)
        unknown = [name for name in names if name not in KNOWN_INDEXES]
        if unknown:
            raise ValueError(f"Unknown index name(s): {unknown!r}")
        return 0
