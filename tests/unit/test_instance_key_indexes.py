"""
tests/unit/test_instance_key_indexes.py — The composite tenant indexes declared by the schema.

The ``(tenant_id, <key>)`` indexes on :ModelInstance for instance_key fields
are tested in tests/unit/graph_neo4j/test_annotation_ops.py.
"""

from __future__ import annotations

from scinr.newton.graph.neo4j.schema import _REGULAR_INDEXES
from scinr.newton.ingest.nodes import ROLE_TO_LABEL


def test_composite_tenant_indexes_are_declared():
    names = {name for name, _ in _REGULAR_INDEXES}
    for expected in (
        "idx_model_instance_tenant_model_class",
        "idx_extraction_result_tenant_model_class",
        "idx_document_tenant_latest",
        "idx_document_tenant_name",
        "idx_structure_node_tenant_role",
        "idx_labeled_entity_tenant_label",
    ):
        assert expected in names
    # The single tenant indexes stay: tenant_id=None queries cannot use a composite.
    assert "idx_model_instance_tenant_id" in names


def test_every_tenant_label_has_a_tenant_index():
    """graphApi's tenant guard filters each node pattern by tenant_id."""
    names = {name for name, _ in _REGULAR_INDEXES}
    for label in ("proposed_model", "proposed_field", "complementary_match", "supplementary_field"):
        assert f"idx_{label}_tenant_id" in names
    assert "idx_theme_path" in names


def test_every_role_label_has_a_tenant_index():
    """MATCH (n:Section) cannot use the :StructureNode index: each role label needs its own."""
    statements = dict(_REGULAR_INDEXES)
    for role, label in ROLE_TO_LABEL.items():
        assert statements[f"idx_{role}_tenant_id"] == (
            f"CREATE INDEX idx_{role}_tenant_id IF NOT EXISTS FOR (n:{label}) ON (n.tenant_id)"
        )
    assert len(statements) == len(_REGULAR_INDEXES)  # no duplicated index name
