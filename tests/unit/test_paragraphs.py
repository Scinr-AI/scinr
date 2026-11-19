import logging

import pytest

from scinr.newton.extraction.paragraphs import (
    PARAGRAPH_SEGMENTER_VERSION,
    ParagraphIndex,
    ParagraphStats,
    build_verbatim_info_units,
    finalize_paragraph_references,
    merge_ranges,
    segment_page,
    segment_page_spans,
    split_paragraph,
    strip_markers,
    verbatim_title,
)
from scinr.newton.models.document_structure import (
    Document,
    InfoUnit,
    NodeRole,
    ParagraphRange,
    StructureNode,
)


def R(start, end=None):
    return ParagraphRange(start=start, end=start if end is None else end)


def pairs(ranges):
    return [(r.start, r.end) for r in ranges]


# ---------------- segmentation ----------------


def test_version_constant():
    assert PARAGRAPH_SEGMENTER_VERSION == 1


def test_multiline_text_block_is_one_paragraph():
    assert segment_page("line one\nline two\nline three") == ["line one\nline two\nline three"]


def test_multiple_blank_lines():
    assert segment_page("a\n\n\n\nb") == ["a", "b"]


def test_heading_glued_to_text():
    assert segment_page("# Title\nbody text\nmore") == ["# Title", "body text\nmore"]


def test_table_is_one_paragraph():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    assert segment_page(table) == [table]


def test_text_right_below_table_is_separate():
    assert segment_page("| a | b |\n| 1 | 2 |\nafter") == ["| a | b |\n| 1 | 2 |", "after"]


def test_code_block_with_blank_lines_and_hash():
    code = "```python\n# comment\n\nx = 1\n```"
    assert segment_page(f"before\n\n{code}\n\nafter") == ["before", code, "after"]


def test_unclosed_code_runs_to_end():
    text = "```\nline\n\n# not heading"
    assert segment_page(text) == [text]


def test_tilde_fence():
    code = "~~~\na\n\nb\n~~~"
    assert segment_page(code) == [code]


def test_fence_closing_needs_same_char():
    text = "```\na\n~~~\n\nb\n```"
    assert segment_page(text) == [text]


def test_list_items_are_paragraphs():
    assert segment_page("- a\n- b\n- c") == ["- a", "- b", "- c"]


def test_list_item_continuation():
    assert segment_page("- a\n  more of a\n- b") == ["- a\n  more of a", "- b"]


def test_non_indented_line_after_list_item_is_new_paragraph():
    assert segment_page("- a\nplain") == ["- a", "plain"]


def test_sub_item_with_two_spaces_is_new_paragraph():
    assert segment_page("- a\n  - sub") == ["- a", "  - sub"]


def test_numbered_list_markers():
    assert segment_page("1. one\n2) two") == ["1. one", "2) two"]


def test_image_only_line():
    assert segment_page("text\n![alt](img.png)\nmore") == ["text", "![alt](img.png)", "more"]


def test_empty_and_whitespace_pages():
    assert segment_page("") == []
    assert segment_page("   \n\n  \t\n") == []


def test_crlf():
    assert segment_page("a\r\nb\r\n\r\nc") == ["a\r\nb", "c"]


def test_text_is_exact_slice_of_page():
    page = "# H\nbody\n\n| a |\n| b |\n\n- x\n"
    for (s, e), text in zip(segment_page_spans(page), segment_page(page), strict=True):
        assert page[s:e] == text


def test_deterministic():
    page = "# H\nbody\n\n- a\n- b\n\n```\nx\n```"
    assert segment_page_spans(page) == segment_page_spans(page)


# ---------------- index ----------------


def make_index(pages=("A1\n\nA2", "", "B1\n\nB2\n\nB3"), indices=(0, 1, 2), ids=None):
    ids = {0: "p0", 1: "p1", 2: "p2"} if ids is None else ids
    return ParagraphIndex(list(pages), list(indices), ids)


def test_continuous_numbering_with_empty_page():
    idx = make_index()
    assert idx.total == 5
    assert [idx.text(n) for n in range(1, 6)] == ["A1", "A2", "B1", "B2", "B3"]
    assert idx.locate(3).page_index == 2
    assert idx.locate(3).local_number == 1
    assert idx.locate(3).page_position == 2


def test_page_indices_with_gaps():
    idx = ParagraphIndex(["a\n\nb", "c", "d", "e\n\nf"], [0, 1, 4, 5], {4: "x", 5: "y"})
    assert idx.locate(4).page_index == 4
    assert idx.locate(4).page_id == "x"
    assert idx.locate(6).page_index == 5
    assert idx.locate(6).local_number == 2
    assert idx.locate(1).page_id is None


def test_locate_bounds():
    idx = make_index()
    assert idx.locate(1).text == "A1"
    assert idx.locate(5).text == "B3"
    for bad in (0, 6, -1):
        with pytest.raises(ValueError):
            idx.locate(bad)


def test_number_range():
    idx = make_index()
    assert idx.number_range([0, 1, 2]) == (1, 5)
    assert idx.number_range([2]) == (3, 5)
    assert idx.number_range([1]) is None
    assert idx.number_range([0, 1]) == (1, 2)


def test_marked_page_exact_format():
    idx = make_index()
    assert idx.marked_page(0) == "[[1]]\nA1\n\n[[2]]\nA2"
    assert idx.marked_page(1) == ""
    assert idx.marked_page(2) == "[[3]]\nB1\n\n[[4]]\nB2\n\n[[5]]\nB3"


def test_page_ids_for_ranges():
    idx = make_index()
    assert idx.page_ids_for_ranges([R(2, 3)]) == ["p0", "p2"]
    assert idx.page_ids_for_ranges([R(3), R(4), R(1), R(99)]) == ["p0", "p2"]
    assert idx.page_ids_for_ranges([]) == []


def test_page_ids_without_ids():
    idx = make_index(ids={})
    assert idx.page_ids_for_ranges([R(1, 5)]) == []


def test_format_source_paragraphs():
    idx = ParagraphIndex(["a\n\nb\n\nc", "d\n\ne"], [4, 5], {})
    assert idx.format_source_paragraphs([R(2, 4)]) == ["5:2-3", "6:1-1"]
    assert idx.format_source_paragraphs([R(3)]) == ["5:3-3"]
    assert idx.format_source_paragraphs([R(1), R(5), R(40)]) == ["5:1-1", "6:2-2"]


# ---------------- ranges / markers ----------------


def test_merge_ranges():
    assert pairs(merge_ranges([R(5, 7), R(8, 9)])) == [(5, 9)]  # contiguous
    assert pairs(merge_ranges([R(1, 4), R(3, 6)])) == [(1, 6)]  # overlapping
    assert pairs(merge_ranges([R(2), R(2)])) == [(2, 2)]  # repeated
    assert pairs(merge_ranges([R(10), R(1, 2), R(4)])) == [(1, 2), (4, 4), (10, 10)]
    assert merge_ranges([]) == []


def test_strip_markers():
    assert strip_markers("[[3]]\nHello [[4]] world\n") == "Hello  world"
    assert strip_markers("no markers") == "no markers"


# ---------------- verbatim_title ----------------


def test_title_truncates_with_ellipsis():
    assert verbatim_title("one two three four five six seven eight nine", "fb") == (
        "one two three four five six seven eight…"
    )


def test_title_eight_words_no_ellipsis():
    assert verbatim_title("one two three four five six seven eight", "fb") == (
        "one two three four five six seven eight"
    )


def test_title_heading():
    assert verbatim_title("## 4.2 Plazos", "fb") == "4.2 Plazos"


def test_title_bold():
    assert verbatim_title("**Bold** and __more__", "fb") == "Bold and more"


def test_title_table_row():
    assert verbatim_title("| a | b |", "fb") == "a b"


def test_title_only_separators_uses_fallback():
    assert verbatim_title("|---|---|\n- ***", "fb") == "fb"


# ---------------- split_paragraph ----------------


def test_split_below_limit():
    assert split_paragraph("short", 100) == ["short"]


def test_split_on_newlines():
    text = "\n".join(["aaaa bbbb"] * 6)
    parts = split_paragraph(text, 25)
    assert len(parts) > 1
    assert all(p and len(p) <= 25 for p in parts)
    assert "\n".join(parts) == text


def test_split_single_long_line_at_spaces():
    text = " ".join(["word"] * 30)
    parts = split_paragraph(text, 20)
    assert all(p and len(p) <= 20 for p in parts)
    assert " ".join(parts) == text


def test_split_long_line_without_spaces():
    parts = split_paragraph("x" * 50, 20)
    assert parts == ["x" * 20, "x" * 20, "x" * 10]


def test_split_table_repeats_header():
    header = "| a | b |\n|---|---|"
    rows = [f"| {i} | v{i} |" for i in range(30)]
    text = "\n".join([header, *rows])
    parts = split_paragraph(text, 120)
    assert len(parts) > 1
    assert all(p.startswith(header + "\n") and len(p) <= 120 for p in parts)
    body = [ln for p in parts for ln in p.split("\n")[2:]]
    assert body == rows


def test_split_table_with_long_header_is_plain_text():
    header = "| " + "h" * 40 + " |\n|" + "-" * 40 + "|"
    text = header + "\n" + "\n".join(f"| row {i} |" for i in range(20))
    parts = split_paragraph(text, 60)
    assert all(p and len(p) <= 60 for p in parts)
    assert "\n".join(parts) == text


# ---------------- build_verbatim_info_units ----------------


def test_build_units_order_and_content():
    idx = ParagraphIndex(["# Head\n\nFirst body\n\nSecond body"], [0], {})
    units = build_verbatim_info_units([R(1, 3)], idx, node_title="N", max_chars=1000)
    assert [u.order for u in units] == [0, 1, 2]
    assert [u.description for u in units] == ["# Head", "First body", "Second body"]
    assert units[0].title == "Head"


def test_build_units_split_paragraph_gives_consecutive_units():
    long_text = " ".join(["word"] * 40)
    idx = ParagraphIndex(["short\n\n" + long_text], [3], {})
    units = build_verbatim_info_units([R(1, 2)], idx, node_title="N", max_chars=50)
    assert [u.order for u in units] == list(range(len(units)))
    assert len(units) > 2
    assert units[0].description == "short"
    assert " ".join(u.description for u in units[1:]) == long_text


def test_build_units_fallback_title_uses_page_number():
    idx = ParagraphIndex(["---"], [4], {})
    # '---' is a paragraph made only of separator characters
    units = build_verbatim_info_units([R(1)], idx, node_title="Node", max_chars=100)
    assert units[0].title == "Node · p. 5"


def test_build_units_no_ranges_or_out_of_range():
    idx = ParagraphIndex(["a\n\nb"], [0], {})
    assert build_verbatim_info_units([], idx, node_title="N", max_chars=100) == []
    assert build_verbatim_info_units([R(9)], idx, node_title="N", max_chars=100) == []


def test_build_units_dedups_overlapping_ranges():
    idx = ParagraphIndex(["a\n\nb\n\nc"], [0], {})
    units = build_verbatim_info_units([R(1, 2), R(2, 3)], idx, node_title="N", max_chars=100)
    assert [u.description for u in units] == ["a", "b", "c"]


# ---------------- finalize_paragraph_references ----------------


def fin_node(node_id, ranges, title=None, children=None, page_ids=None, units=None):
    return StructureNode(
        node_id=node_id,
        role=NodeRole.SECTION,
        title=title,
        appearance_order=1,
        children=children or [],
        paragraph_ranges=[R(a, b) for a, b in ranges],
        source_page_ids=page_ids or [],
        info_units=units or [],
    )


def fin_doc(*nodes):
    return Document(raw_file_id="raw", document_name="Doc",document_structure=list(nodes))


def run_finalize(doc, mode="summary", index=None, max_chars=4000, stats=None):
    finalize_paragraph_references(
        doc,
        index or make_index(),
        info_units_mode=mode,
        max_chars=max_chars,
        stats=stats or ParagraphStats(),
    )


def test_finalize_summary_keeps_llm_units_and_sets_references():
    unit = InfoUnit(title="LLM", order=0, description="summary text")
    node = fin_node("a", [(2, 4)], units=[unit], page_ids=["old"])
    doc = fin_doc(node)

    run_finalize(doc, "summary")

    assert node.info_units == [unit]
    assert node.source_page_ids == ["p0", "p2"]
    assert node.source_paragraphs == ["1:2-2", "3:1-2"]
    assert node.info_units_mode == "summary"


def test_finalize_verbatim_builds_units_in_order_with_original_text():
    node = fin_node("a", [(3, 4)], title="Node")
    run_finalize(fin_doc(node), "verbatim")

    assert [u.description for u in node.info_units] == ["B1", "B2"]
    assert [u.order for u in node.info_units] == [0, 1]
    assert node.info_units[0].title == "B1"
    assert node.info_units_mode == "verbatim"


def test_finalize_verbatim_splits_large_paragraph():
    index = make_index(pages=("word " * 40,), indices=(0,), ids={0: "p0"})
    node = fin_node("a", [(1, 1)], title="Node")

    run_finalize(fin_doc(node), "verbatim", index=index, max_chars=100)

    assert len(node.info_units) > 1
    assert all(len(u.description) <= 100 for u in node.info_units)


def test_finalize_verbatim_identical_paragraphs_give_identical_units():
    index = make_index(pages=("same\n\nsame",), indices=(0,), ids={0: "p0"})
    node = fin_node("a", [(1, 2)], title="Node")

    run_finalize(fin_doc(node), "verbatim", index=index)

    assert len(node.info_units) == 2
    assert node.info_units[0].title == node.info_units[1].title
    assert node.info_units[0].description == node.info_units[1].description


def test_finalize_node_without_ranges(caplog):
    node = fin_node("a", [], title="Node", page_ids=["keep"])
    node.source_paragraphs = ["stale"]
    with caplog.at_level(logging.INFO, logger="scinr.newton.extraction.paragraphs"):
        run_finalize(fin_doc(node), "verbatim")

    assert node.source_page_ids == ["keep"]
    assert node.source_paragraphs == []
    assert node.info_units == []
    assert "1 nodes without valid ranges" in caplog.text


def test_finalize_log_figures(caplog):
    child = fin_node("b", [(2, 3)])
    top = fin_node("a", [(1, 2)], children=[child])
    stats = ParagraphStats(discarded_ranges=2)
    with caplog.at_level(logging.INFO, logger="scinr.newton.extraction.paragraphs"):
        run_finalize(fin_doc(top), "summary", stats=stats)  # index.total == 5

    assert (
        "Paragraph anchors for 'Doc': 5 paragraphs, 3 cited (60.0%), 2 ranges discarded, "
        "0 nodes without valid ranges, info_units_mode=summary"
    ) in caplog.text


def test_finalize_log_empty_index_pct_zero(caplog):
    index = make_index(pages=(), indices=(), ids={})
    with caplog.at_level(logging.INFO, logger="scinr.newton.extraction.paragraphs"):
        run_finalize(fin_doc(fin_node("a", [])), index=index)

    assert "0 paragraphs, 0 cited (0.0%)" in caplog.text


def test_finalize_marks_every_node_recursively():
    grandchild = fin_node("c", [(1, 1)])
    child = fin_node("b", [], children=[grandchild])
    top = fin_node("a", [(2, 2)], children=[child])

    run_finalize(fin_doc(top), "verbatim")

    for n in (top, child, grandchild):
        assert n.paragraph_segmenter_version == 1
        assert n.info_units_mode == "verbatim"
