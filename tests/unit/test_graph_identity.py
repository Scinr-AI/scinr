"""
tests/unit/test_graph_identity.py -- ``graph.identity`` and the ``label`` / ``key`` of ``GraphNode``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from scinr.newton.graph import identity
from scinr.newton.ingest import freeze
from scinr.newton.navigation import _map
from scinr.newton.sql import keys
from scinr.newton.sql.names import LABEL_TABLE

#: One node (labels and key properties) per primary label of the graph.
NODES: dict[str, tuple[list[str], dict[str, object], str]] = {
    "Document": (
        ["Document"],
        {"tenant_id": "acme", "path": "dir/a.pdf", "version": 2, "name": "a.pdf"},
        '["acme","dir/a.pdf",2]',
    ),
    "StructureNode": (
        ["StructureNode", "Section"],
        {"id": "acme::a.pdf::1::5_3", "role": "section"},
        "acme::a.pdf::1::5_3",
    ),
    "CatalogModel": (["CatalogModel"], {"name": "Drug"}, "Drug"),
    "ModelField": (["ModelField"], {"name": "dose", "model": "Drug"}, '["dose","Drug"]'),
    "EntityLabel": (["EntityLabel"], {"label": "ActiveSubstance"}, "ActiveSubstance"),
    "Theme": (["Theme"], {"path": "pharma/default"}, "pharma/default"),
    **{
        label: ([label], {"uid": f"{label}-1", "tenant_id": "acme"}, f"{label}-1")
        for label in identity.UID_LABELS
    },
}


def test_there_is_a_node_for_every_label_of_the_sql_tables() -> None:
    assert set(NODES) == set(LABEL_TABLE) == set(identity.PRIMARY_LABELS)


@pytest.mark.parametrize("label", sorted(NODES))
def test_graph_node_carries_label_and_key(label: str) -> None:
    labels, props, key = NODES[label]
    node = _map.graph_node({**props, "_labels": labels})
    assert (node.label, node.key) == (label, key)
    assert node.types == labels
    assert identity.decode_key(label, key) == {
        f: props[f] for f in identity.node_identity([label])[1]
    }


def test_a_node_without_known_labels_has_no_identity() -> None:
    node = _map.graph_node({"uid": "x", "_labels": ["Mystery"]})
    assert (node.label, node.key) == (None, None)
    assert _map.graph_node({"uid": "x"}).key is None  # no labels at all


def test_a_node_missing_a_key_property_has_no_identity() -> None:
    node = _map.graph_node({"tenant_id": "acme", "path": "a.pdf", "_labels": ["Document"]})
    assert (node.label, node.key) == (None, None)
    assert node.properties["path"] == "a.pdf"  # the rest of the node still maps


def test_the_old_import_paths_still_work() -> None:
    assert freeze.node_identity is identity.node_identity
    assert freeze.CATALOG_KEY_FIELDS is identity.CATALOG_KEY_FIELDS
    assert freeze.UID_LABELS is identity.UID_LABELS
    assert keys.encode_key is identity.encode_key
    assert keys.decode_key is identity.decode_key


def test_mapping_a_node_does_not_need_sqlalchemy() -> None:
    code = (
        "import sys; import scinr.newton.navigation._map; "
        "sys.exit(1 if 'sqlalchemy' in sys.modules else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
