"""Unit tests for scinr.newton.navigation.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scinr.newton.navigation.models import (
    DocumentRef,
    DocumentTree,
    ModelDecisionRef,
    ModelInstanceRef,
    ModelInstanceTree,
    ProposedModelRef,
    StructureNodeRef,
)


def _doc(**kw):
    base = dict(path="a/b", name="b", version=1, latest=True, is_folder=False)
    base.update(kw)
    return DocumentRef(**base)


def test_document_ref_frozen() -> None:
    d = _doc()
    with pytest.raises(ValidationError):
        d.path = "x"  # type: ignore[misc]


def test_raw_default_and_repr_hidden() -> None:
    d = _doc()
    assert d.raw == {}
    assert "raw=" not in repr(d)


def test_document_tree_recursion() -> None:
    child = DocumentTree(**_doc(path="a/b/c", name="c").model_dump(), depth=1)
    root = DocumentTree(**_doc().model_dump(), depth=0, children=[child])
    assert root.children[0].path == "a/b/c"


def test_model_decision_confidence_is_str_and_gaps_list() -> None:
    md = ModelDecisionRef(uid="x", confidence="high", coverage_gaps=["g1", "g2"])
    assert md.confidence == "high"
    assert md.coverage_gaps == ["g1", "g2"]


def test_proposed_model_name_alias() -> None:
    pm = ProposedModelRef(uid="x", schema_name="FooModel")
    assert pm.name == "FooModel"


def test_structure_node_types_and_optional_doc_ctx() -> None:
    n = StructureNodeRef(id="i", node_id="n", role="table", types=["StructureNode", "Table"])
    assert n.document_path is None
    assert "Table" in n.types


def test_model_instance_ref_properties_exclude_identity() -> None:
    mi = ModelInstanceRef(uid="u", model_class="M", properties={"a": 1})
    assert mi.properties == {"a": 1}
    assert mi.is_shell is None


def test_model_instance_tree_children() -> None:
    leaf = ModelInstanceTree(uid="c", model_class="M", depth=1)
    root = ModelInstanceTree(uid="r", model_class="M", depth=0, children=[leaf])
    assert root.children[0].uid == "c"


def test_model_dump_round_trip() -> None:
    d = _doc(raw_file_id="rf")
    assert DocumentRef(**d.model_dump()) == d
