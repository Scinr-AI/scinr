import pytest

from scinr.newton.extraction.compact_extraction import _dedup_siblings
from scinr.newton.models.document_structure import NodeRole, ParagraphRange, StructureNode


def _node(node_id, ranges, children=None):
    return StructureNode(
        node_id=node_id,
        role=NodeRole.SECTION,
        title=node_id,
        appearance_order=1,
        children=children or [],
        paragraph_ranges=[ParagraphRange(start=a, end=b) for a, b in ranges],
    )


def _pairs(node):
    return [(r.start, r.end) for r in node.paragraph_ranges]


def test_dedup_siblings_unites_overlapping_ranges():
    nodes = [_node("a", [(1, 3), (10, 10)]), _node("a", [(2, 5)])]

    _dedup_siblings(nodes)

    assert len(nodes) == 1
    assert _pairs(nodes[0]) == [(1, 5), (10, 10)]


def test_dedup_siblings_unites_contiguous_ranges():
    nodes = [_node("a", [(5, 7)]), _node("a", [(8, 9)])]

    _dedup_siblings(nodes)

    assert _pairs(nodes[0]) == [(5, 9)]


def test_dedup_siblings_merges_nested_duplicates():
    nodes = [
        _node("p", [(1, 1)], [_node("c", [(2, 2)])]),
        _node("p", [(3, 3)], [_node("c", [(3, 4)])]),
    ]

    _dedup_siblings(nodes)

    assert _pairs(nodes[0]) == [(1, 1), (3, 3)]
    assert len(nodes[0].children) == 1
    assert _pairs(nodes[0].children[0]) == [(2, 4)]


def test_dedup_siblings_without_ranges_stays_empty():
    nodes = [_node("a", []), _node("a", [])]

    _dedup_siblings(nodes)

    assert nodes[0].paragraph_ranges == []


@pytest.mark.parametrize("order", [("verbatim", "summary"), ("summary", "verbatim")])
def test_dedup_siblings_differing_auto_decisions_give_summary(order):
    from scinr.newton.extraction.compact_extraction import _dedup_siblings
    from scinr.newton.models.document_structure import StructureNode

    nodes = [
        StructureNode(node_id="1", title="T", role="section", appearance_order=1, info_units_mode=m)
        for m in order
    ]
    _dedup_siblings(nodes)
    assert len(nodes) == 1
    assert nodes[0].info_units_mode == "summary"


def test_dedup_siblings_both_verbatim_stays_verbatim():
    from scinr.newton.extraction.compact_extraction import _dedup_siblings
    from scinr.newton.models.document_structure import StructureNode

    nodes = [
        StructureNode(
            node_id="1", title="T", role="section", appearance_order=1, info_units_mode="verbatim"
        )
        for _ in range(2)
    ]
    _dedup_siblings(nodes)
    assert nodes[0].info_units_mode == "verbatim"
