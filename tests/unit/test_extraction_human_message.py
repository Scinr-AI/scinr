"""
tests/unit/test_extraction_human_message.py — Regression tests for the
``<extraction_mode>`` tag injected by ``_build_human_message()`` when
``defer_hierarchy=True`` (scinr.newton.extraction.extraction).

Context
-------
Under ``fast_extraction=True`` (``defer_hierarchy=True``), each chunk is
extracted in isolation with no real cross-chunk hierarchy visibility. The
extraction prompt's own ambiguous-heading-level fallback defaults to the
higher-level role (SECTION) when ambiguity cannot be resolved from
``<previous_page>`` alone — a default that is safe in the legacy sequential
path (later chunks with full context can still correct it) but permanent and
uncorrectable in the fast/isolated path, since ``structure_consolidation.py``
never re-examines nodes already typed SECTION/APPENDIX.

To counteract this, ``_build_human_message()`` injects an ``<extraction_mode>``
block immediately before ``<active_hierarchy>`` whenever ``defer_hierarchy=True``,
telling the model to invert that fallback. This block must be entirely absent
when ``defer_hierarchy=False`` — the legacy human message must be unchanged.
"""

from __future__ import annotations

from scinr.newton.extraction.extraction import (
    _DEFERRED_HIERARCHY_NOTE,
    _FAST_EXTRACTION_MODE_NOTE,
    _build_human_message,
)


class TestExtractionModeTag:
    def test_extraction_mode_tag_present_when_defer_hierarchy_true(self):
        """The <extraction_mode> block (with its "fast" content) must appear
        in the built message when defer_hierarchy=True."""
        message = _build_human_message(
            prev_page="prev",
            curr_pages=["page one"],
            active_hierarchy="1_intro",
            defer_hierarchy=True,
        )

        assert "<extraction_mode>" in message
        assert "</extraction_mode>" in message
        assert "fast" in message
        assert _FAST_EXTRACTION_MODE_NOTE in message
        # active_hierarchy is ignored in this mode; the "(none)" sentinel is used.
        assert f"<active_hierarchy>\n{_DEFERRED_HIERARCHY_NOTE}\n</active_hierarchy>" in message

    def test_extraction_mode_tag_precedes_active_hierarchy(self):
        """<extraction_mode> must appear BEFORE <active_hierarchy> in the
        final message string — order matters for the model to read the mode
        note as framing context ahead of the (empty) hierarchy block."""
        message = _build_human_message(
            prev_page="prev",
            curr_pages=["page one"],
            active_hierarchy="1_intro",
            defer_hierarchy=True,
        )

        extraction_mode_idx = message.index("<extraction_mode>")
        active_hierarchy_idx = message.index("<active_hierarchy>")
        assert extraction_mode_idx < active_hierarchy_idx

    def test_extraction_mode_tag_absent_when_defer_hierarchy_false(self):
        """Legacy/sequential path (defer_hierarchy=False) must never contain
        the <extraction_mode> tag — zero footprint on that path."""
        message = _build_human_message(
            prev_page="prev",
            curr_pages=["page one"],
            active_hierarchy="1_intro",
            defer_hierarchy=False,
        )

        assert "<extraction_mode>" not in message
        assert "</extraction_mode>" not in message
        # The real active_hierarchy value is used untouched in this mode.
        assert "<active_hierarchy>\n1_intro\n</active_hierarchy>" in message

    def test_extraction_mode_tag_absent_by_default(self):
        """defer_hierarchy defaults to False, so calling without it must also
        omit the tag."""
        message = _build_human_message(
            prev_page="prev",
            curr_pages=["page one"],
            active_hierarchy="1_intro",
        )

        assert "<extraction_mode>" not in message
