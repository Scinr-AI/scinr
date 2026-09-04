"""
tests/unit/test_consolidation_prompt.py — Unit tests for
scinr.newton.prompts.consolidation_prompt.build_consolidation_prompt().
"""

from __future__ import annotations

from scinr.newton.prompts.consolidation_prompt import (
    _CONSOLIDATION_SYSTEM_PROMPT,
    build_consolidation_prompt,
)


class TestBuildConsolidationPrompt:
    def test_default_call_is_byte_for_byte_identical_to_pre_change_prompt(self):
        """No argument passed at all — must match the pre-existing prompt
        constant exactly (no partial-visibility notice), preserving the
        original single-call behaviour for callers that never opted in.
        """
        assert build_consolidation_prompt() == _CONSOLIDATION_SYSTEM_PROMPT

    def test_partial_visibility_false_is_identical_to_default(self):
        assert build_consolidation_prompt(partial_visibility=False) == _CONSOLIDATION_SYSTEM_PROMPT

    def test_partial_visibility_false_has_no_notice(self):
        prompt = build_consolidation_prompt(partial_visibility=False)
        assert "partial" not in prompt.lower()
        assert "reconsidered in a later call" not in prompt

    def test_partial_visibility_true_includes_the_notice(self):
        prompt = build_consolidation_prompt(partial_visibility=True)
        assert prompt.startswith(_CONSOLIDATION_SYSTEM_PROMPT)
        assert "partial" in prompt.lower()
        assert "decided_parent_id: null" in prompt
        assert "reconsidered in a later call" in prompt

    def test_partial_visibility_true_is_longer_than_false(self):
        short_prompt = build_consolidation_prompt(partial_visibility=False)
        long_prompt = build_consolidation_prompt(partial_visibility=True)
        assert len(long_prompt) > len(short_prompt)
