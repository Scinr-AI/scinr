from scinr.newton.models import (
    DocumentStructure,
    DocumentStructureSummaryOutput,
    DocumentStructureVerbatimOutput,
    StructureNode,
    StructureNodeSummaryOutput,
    StructureNodeVerbatimOutput,
)


def _schema_text(model) -> str:
    return str(model.model_json_schema())


def test_legacy_node_without_new_fields_reads_with_defaults():
    node = StructureNode.model_validate(
        {
            "node_id": "1",
            "title": "T",
            "role": "section",
            "appearance_order": 1,
            "info_units": [{"title": "a", "order": 0, "description": "d"}],
            "source_page_ids": ["p1"],
        }
    )
    assert node.paragraph_ranges == []
    assert node.source_paragraphs == []
    assert node.info_units_mode == "summary"
    assert node.paragraph_segmenter_version is None


def test_verbatim_output_schema_has_no_info_units_or_page_ids():
    text = _schema_text(StructureNodeVerbatimOutput)
    assert "info_units" not in text
    assert "source_page_ids" not in text
    assert "paragraph_ranges" in text


def test_summary_output_schema_has_info_units_and_ranges_not_page_ids():
    text = _schema_text(StructureNodeSummaryOutput)
    assert "info_units" in text
    assert "paragraph_ranges" in text
    assert "source_page_ids" not in text


def test_verbatim_output_dump_validates_as_structure_node_with_children():
    out = StructureNodeVerbatimOutput(
        node_id="1",
        title="Parent",
        role="section",
        appearance_order=1,
        paragraph_ranges=[{"start": 1, "end": 2}],
        children=[
            StructureNodeVerbatimOutput(
                node_id="1_1",
                title="Child",
                role="subsection",
                appearance_order=1,
                paragraph_ranges=[{"start": 3, "end": 3}],
            )
        ],
    )
    node = StructureNode.model_validate(out.model_dump())
    assert node.paragraph_ranges[0].start == 1
    assert node.children[0].paragraph_ranges[0].end == 3
    assert node.info_units == []


def test_document_outputs_coerce_nodes_from_json_string():
    payload = '[{"node_id": "1", "title": "T", "role": "section", "appearance_order": 1}]'
    for model in (
        DocumentStructure,
        DocumentStructureSummaryOutput,
        DocumentStructureVerbatimOutput,
    ):
        assert model.model_validate({"nodes": payload}).nodes[0].node_id == "1"
