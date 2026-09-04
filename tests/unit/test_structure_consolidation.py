"""
tests/unit/test_structure_consolidation.py — Unit tests for
scinr.newton.extraction.structure_consolidation (the Stage 1
``fast_extraction=True`` Map-phase namespacing + consolidation + tree
assembly + checkpoint I/O module).

Mocking strategy
-----------------
``consolidate_structure()`` takes an explicit ``llm`` argument — no need to
patch ``scinr.newton.config.get_llm``. A minimal fake LLM is built here that
mimics the ``llm.with_structured_output(schema, include_raw=True).ainvoke(messages)``
shape consumed by ``with_llm_retry()`` (a dict with ``"raw"``, ``"parsed"``,
``"parsing_error"`` keys), following the same duck-typed contract exercised
elsewhere in this codebase (see ``tests/unit/test_normalization_engine_retry.py``
for a comparable pattern applied to a different structured-output call site).

``get_llm_semaphore()`` requires ``configure()`` to have run first (mirrors
``tests/unit/test_extraction_page_id_gaps.py``'s ``_configure_min`` fixture).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scinr.newton.config import configure
from scinr.newton.extraction.structure_consolidation import (
    assemble_tree,
    consolidate_structure,
    delete_map_checkpoint,
    namespace_node_ids,
    write_map_checkpoint,
)
from scinr.newton.models.consolidation import ParentDecision
from scinr.newton.models.document_structure import NodeRole, StructureNode


@pytest.fixture(autouse=True)
def _configure_min(mock_llm):
    """Minimal configure() so get_config()/get_llm_semaphore() work.

    Mirrors tests/unit/test_extraction_page_id_gaps.py's fixture of the same
    name/purpose.
    """
    configure(
        llm=mock_llm,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
    )


# ---------------------------------------------------------------------------
# Node-building helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    role: NodeRole,
    title: str | None = None,
    children: list[StructureNode] | None = None,
    appearance_order: int = 1,
    parent_id: str | None = None,
) -> StructureNode:
    return StructureNode(
        node_id=node_id,
        role=role,
        title=title,
        appearance_order=appearance_order,
        parent_id=parent_id,
        children=children or [],
    )


def _ids(nodes: list[StructureNode]) -> list[str]:
    """Flatten node_id of every node at every depth, depth-first."""
    out: list[str] = []
    for n in nodes:
        out.append(n.node_id)
        out.extend(_ids(n.children))
    return out


# ---------------------------------------------------------------------------
# 1. namespace_node_ids()
# ---------------------------------------------------------------------------


class TestNamespaceNodeIds:
    def test_single_page_chunk_namespaces_top_level_and_descendants(self):
        leaf = _node("2_1_1", NodeRole.TABLE, title="Leaf")
        mid = _node("2_1", NodeRole.SUBSECTION, title="Mid", children=[leaf])
        top = _node("2", NodeRole.SECTION, title="Top", children=[mid])

        namespace_node_ids([top], page_number=7)

        assert top.node_id == "page-7::2"
        assert top.children[0].node_id == "page-7::2_1"
        assert top.children[0].children[0].node_id == "page-7::2_1_1"

    def test_multi_page_chunk_uses_uniform_first_page_number(self):
        """A multi-page chunk (extraction_batch_size > 1) must namespace
        every node it returned — regardless of depth — with the SAME page
        number: the caller-supplied first absolute page index of the batch.
        """
        leaf = _node("a_1", NodeRole.FIELD_GROUP, title="Leaf")
        mid = _node("a", NodeRole.SUBSECTION, title="Mid", children=[leaf])
        other_top = _node("b", NodeRole.APPENDIX, title="Other top")

        nodes = [mid, other_top]
        namespace_node_ids(nodes, page_number=12)

        assert mid.node_id == "page-12::a"
        assert mid.children[0].node_id == "page-12::a_1"
        assert other_top.node_id == "page-12::b"

    def test_parent_id_left_untouched(self):
        """namespace_node_ids() must never rewrite parent_id — only node_id."""
        child = _node("x_1", NodeRole.TABLE, title="Child", parent_id="x")
        top = _node("x", NodeRole.SECTION, title="Top", children=[child])

        namespace_node_ids([top], page_number=3)

        assert top.node_id == "page-3::x"
        assert child.node_id == "page-3::x_1"
        # parent_id is untouched — still the pre-namespacing raw value.
        assert child.parent_id == "x"


# ---------------------------------------------------------------------------
# Fake LLM plumbing shared by consolidate_structure() tests
# ---------------------------------------------------------------------------


class _FakeAinvoke:
    """Callable mimicking ``structured_llm.ainvoke(messages)``.

    ``responder(messages) -> ConsolidationOutput | None`` decides what to
    return for each call. Returning ``None`` simulates a hard parse failure
    (parsed=None, no repair available) — not exercised by these tests since
    ``run_repair_loop`` is not itself under test here.
    """

    def __init__(self, responder, sleep_seconds: float = 0.0, tracker=None):
        self.responder = responder
        self.calls: list[str] = []
        self._sleep_seconds = sleep_seconds
        self._tracker = tracker

    async def __call__(self, messages):
        # messages[1] is the HumanMessage carrying the rendered pool + must_decide.
        human_text = messages[1].content
        self.calls.append(human_text)
        if self._tracker is not None:
            await self._tracker.track()
        elif self._sleep_seconds:
            await asyncio.sleep(self._sleep_seconds)
        parsed = self.responder(human_text)
        return {"raw": MagicMock(), "parsed": parsed, "parsing_error": None}


class _FakeStructuredLLM:
    def __init__(self, ainvoke_impl: _FakeAinvoke):
        self.ainvoke = ainvoke_impl


class _FakeLLM:
    def __init__(self, ainvoke_impl: _FakeAinvoke):
        self._structured = _FakeStructuredLLM(ainvoke_impl)

    def with_structured_output(self, schema, include_raw=True):
        return self._structured


class _ConcurrencyTracker:
    """Mirrors tests/unit/test_annotation_agent.py's ``_ConcurrencyTracker``."""

    def __init__(self, sleep_seconds: float = 0.03) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._sleep_seconds = sleep_seconds
        self._lock = asyncio.Lock()

    async def track(self) -> None:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self._sleep_seconds)
        async with self._lock:
            self.in_flight -= 1


def _consolidation_output(decisions: list[ParentDecision]):
    from scinr.newton.models.consolidation import ConsolidationOutput

    return ConsolidationOutput(decisions=decisions)


# ---------------------------------------------------------------------------
# 2. consolidate_structure()
# ---------------------------------------------------------------------------


class TestConsolidateStructureHappyPath:
    async def test_orphans_get_one_decision_each_including_deep_nested_target(self):
        """Pool: chunk0 has a SECTION top-level and an orphan SUBSECTION
        top-level; chunk1 has an APPENDIX top-level with a nested child
        several levels deep, and an orphan TABLE top-level. The mock LLM's
        decision for chunk0's orphan points at chunk1's deeply-nested node —
        proving cross-chunk, cross-depth targeting works.
        """
        chunk0_orphan = _node("orphan_a", NodeRole.SUBSECTION, title="Orphan A")
        chunk0_section = _node("sec1", NodeRole.SECTION, title="Section 1")

        deep_leaf = _node("deep_leaf", NodeRole.FIELD_GROUP, title="Deep leaf")
        deep_mid = _node("deep_mid", NodeRole.SUBSECTION, title="Deep mid", children=[deep_leaf])
        chunk1_appendix = _node(
            "appendixA", NodeRole.APPENDIX, title="Appendix A", children=[deep_mid]
        )
        chunk1_orphan = _node("orphan_b", NodeRole.TABLE, title="Orphan B")

        all_chunks = [
            [chunk0_section, chunk0_orphan],
            [chunk1_appendix, chunk1_orphan],
        ]

        def responder(human_text: str):
            assert "orphan_a" in human_text
            assert "orphan_b" in human_text
            # Both orphans must be listed in <must_decide>, sections/appendices
            # must not be.
            must_decide_block = human_text.split("<must_decide>")[1].split("</must_decide>")[0]
            assert "orphan_a" in must_decide_block
            assert "orphan_b" in must_decide_block
            assert "sec1" not in must_decide_block
            assert "appendixA" not in must_decide_block
            # Reference targets (including deeply-nested nodes) must still
            # be present in the pool itself, outside <must_decide>.
            assert "deep_leaf" in human_text
            return _consolidation_output(
                [
                    ParentDecision(node_id="orphan_a", decided_parent_id="deep_leaf"),
                    ParentDecision(node_id="orphan_b", decided_parent_id=None),
                ]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(decisions) == 2
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan_a": "deep_leaf", "orphan_b": None}
        assert len(fake_ainvoke.calls) == 1  # single partition, no ceiling exceeded.

    async def test_null_decided_parent_id_flows_through_as_none(self):
        orphan = _node("orphan_root", NodeRole.FREEFORM_BLOCK, title="Root-bound orphan")
        section = _node("sec1", NodeRole.SECTION, title="Section 1")
        all_chunks = [[section, orphan]]

        def responder(_human_text: str):
            return _consolidation_output(
                [ParentDecision(node_id="orphan_root", decided_parent_id=None)]
            )

        llm = _FakeLLM(_FakeAinvoke(responder))
        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(decisions) == 1
        assert decisions[0].node_id == "orphan_root"
        assert decisions[0].decided_parent_id is None

    async def test_no_orphans_skips_llm_call_entirely(self):
        section = _node("sec1", NodeRole.SECTION, title="Section 1")
        all_chunks = [[section]]

        fake_ainvoke = _FakeAinvoke(lambda _t: _consolidation_output([]))
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert decisions == []
        assert fake_ainvoke.calls == []  # never invoked — no orphans to resolve.


class TestConsolidateStructureMissingAndDuplicateDecisions:
    async def test_missing_decision_is_logged_and_falls_back_to_root(self, caplog):
        """The mock LLM omits a decision for one of two orphans — the
        omission must be logged, not raised, and the returned decisions
        list simply lacks an entry for it (assemble_tree() then treats a
        missing entry as decided_parent_id=None / root fallback).
        """
        orphan1 = _node("orphan1", NodeRole.SUBSECTION, title="Orphan 1")
        orphan2 = _node("orphan2", NodeRole.TABLE, title="Orphan 2")
        section = _node("sec1", NodeRole.SECTION, title="Section 1")
        all_chunks = [[section, orphan1, orphan2]]

        def responder(_human_text: str):
            # Only returns a decision for orphan1 — orphan2 is silently omitted.
            return _consolidation_output(
                [ParentDecision(node_id="orphan1", decided_parent_id=None)]
            )

        llm = _FakeLLM(_FakeAinvoke(responder))

        with caplog.at_level(logging.WARNING, logger="scinr.newton.extraction.structure_consolidation"):
            decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(decisions) == 1
        assert decisions[0].node_id == "orphan1"
        assert any(
            "no decision returned for orphan" in rec.message and "orphan2" in rec.message
            for rec in caplog.records
        )

    async def test_duplicate_decision_first_wins_and_logs_warning(self, caplog):
        """The mock LLM returns two decisions for the same orphan node_id
        with different targets — the first must win, and a warning must be
        logged for the discarded duplicate.
        """
        orphan = _node("orphan1", NodeRole.SUBSECTION, title="Orphan 1")
        target_a = _node("target_a", NodeRole.SECTION, title="Target A")
        target_b = _node("target_b", NodeRole.APPENDIX, title="Target B")
        all_chunks = [[target_a, target_b, orphan]]

        def responder(_human_text: str):
            return _consolidation_output(
                [
                    ParentDecision(node_id="orphan1", decided_parent_id="target_a"),
                    ParentDecision(node_id="orphan1", decided_parent_id="target_b"),
                ]
            )

        llm = _FakeLLM(_FakeAinvoke(responder))

        with caplog.at_level(logging.WARNING, logger="scinr.newton.extraction.structure_consolidation"):
            decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(decisions) == 1
        assert decisions[0].decided_parent_id == "target_a"  # first wins.
        assert any(
            "duplicate decision for node_id" in rec.message and "orphan1" in rec.message
            for rec in caplog.records
        )


class TestConsolidateStructurePartitioning:
    async def test_output_ceiling_forces_partition_split_and_merges_correctly(self, mock_llm):
        """Force consolidation_max_output_tokens low enough that even two
        orphans exceed the per-call output ceiling, verifying:
          - every orphan still gets exactly one decision after merging,
          - a decision from one partition's call correctly targets a node
            that only appears in a DIFFERENT partition's chunk data (the
            "full pool for reference" design),
          - partitions run concurrently (not sequentially).
        """
        # A tiny explicit ceiling — sized well below the output-token
        # estimate for even a single orphan (see
        # _estimate_decision_output_tokens: per-entry overhead alone is 40
        # chars -> 10 tokens after //4, plus each orphan's own + the
        # longest node_id's length), so max_orphans_per_call collapses to 1
        # and 2 orphans is guaranteed to split into 2 partitions.
        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
            consolidation_max_output_tokens=5,
        )

        orphan1 = _node("orphan1", NodeRole.SUBSECTION, title="Orphan 1")
        chunk0_section = _node("sec1", NodeRole.SECTION, title="Section 1")

        target_deep = _node("target_deep", NodeRole.FIELD_GROUP, title="Target deep")
        chunk1_appendix = _node(
            "appendixA", NodeRole.APPENDIX, title="Appendix A", children=[target_deep]
        )
        orphan2 = _node("orphan2", NodeRole.TABLE, title="Orphan 2")

        all_chunks = [
            [chunk0_section, orphan1],
            [chunk1_appendix, orphan2],
        ]

        tracker = _ConcurrencyTracker(sleep_seconds=0.03)

        def responder(human_text: str):
            must_decide_block = human_text.split("<must_decide>")[1].split("</must_decide>")[0]
            if "orphan1" in must_decide_block:
                assert "orphan2" not in must_decide_block  # each partition covers ONE orphan.
                # orphan1's partition targets a node (target_deep) that only
                # exists in chunk1's data — proving the full pool (not just
                # this partition's own orphans/chunk) was rendered as context.
                assert "target_deep" in human_text
                return _consolidation_output(
                    [ParentDecision(node_id="orphan1", decided_parent_id="target_deep")]
                )
            assert "orphan1" not in must_decide_block
            return _consolidation_output(
                [ParentDecision(node_id="orphan2", decided_parent_id=None)]
            )

        fake_ainvoke = _FakeAinvoke(responder, tracker=tracker)
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(fake_ainvoke.calls) == 2  # confirms the split actually happened.
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan1": "target_deep", "orphan2": None}
        # The two partition calls actually overlapped in time (ran
        # concurrently via asyncio.gather, not sequentially).
        assert tracker.max_in_flight == 2

        # Correctness of the merged result when fed into assemble_tree():
        # orphan1 (from chunk0) ends up nested under target_deep (only
        # present in chunk1's local subtree).
        tree = assemble_tree(all_chunks, decisions)
        target_deep_node = tree[1].children[0]  # appendixA -> target_deep
        assert target_deep_node.node_id == "target_deep"
        assert orphan1 in target_deep_node.children


# ---------------------------------------------------------------------------
# 2b. Token estimation — tiktoken (real counts, not the char/4 proxy)
# ---------------------------------------------------------------------------


class TestEstimateTokensUsesRealTiktoken:
    def test_matches_o200k_base_encoding_not_the_old_char_over_4_proxy(self):
        """`_estimate_tokens()` must return the real tiktoken (o200k_base)
        token count for its input — not `len(text) // 4` (the proxy this
        redesign replaced).
        """
        import tiktoken

        from scinr.newton.extraction.structure_consolidation import _estimate_tokens

        text = (
            "Regulatory Affairs — Módulo 3.2.P.5 Control of Drug Product "
            "(multibyte check: 你好, café, naïve)."
        )
        encoding = tiktoken.get_encoding("o200k_base")
        expected = len(encoding.encode(text))

        actual = _estimate_tokens(text)

        assert actual == expected
        # Guard against a false-positive pass: for this string the old
        # char//4 proxy must NOT coincidentally match the real count, or
        # this test would not actually be pinning the new mechanism.
        assert actual != len(text) // 4

    def test_encoding_singleton_is_reused_across_calls(self):
        """The lazy tiktoken encoding singleton is loaded once, not per call."""
        import scinr.newton.extraction.structure_consolidation as sc

        sc._TIKTOKEN_ENCODING = None  # force a fresh singleton for this test
        first = sc._get_encoding()
        second = sc._get_encoding()
        assert first is second


# ---------------------------------------------------------------------------
# 2c. consolidate_structure() — sliding-window batching redesign
# ---------------------------------------------------------------------------


def _long_filler(label: str, repeats: int = 40) -> str:
    """A long, deterministic filler string so a single node's rendered pool
    entry costs a realistic number of tokens (tens to hundreds) — needed to
    make ``consolidation_max_input_tokens`` ceilings meaningfully small
    relative to the module's fixed per-call overhead (system prompt +
    margin) without having to fabricate an enormous test document.
    """
    return f"{label} " + ("Detailed Regulatory Description Section Content Filler Words Here " * repeats)


class TestConsolidateStructureSingleBatchEquivalence:
    async def test_small_document_uses_exactly_one_call_like_the_pre_redesign_behavior(self):
        """Property required by the redesign: for a document whose full pool
        comfortably fits under the default `consolidation_max_input_tokens`
        ceiling, the sliding-window algorithm must collapse to exactly one
        LLM call — the same single-call behavior `consolidate_structure()`
        had before this change (see `fits_single_batch` in the
        implementation) — for the same final decisions.
        """
        chunk0_orphan = _node("orphan_a", NodeRole.SUBSECTION, title="Orphan A")
        chunk0_section = _node("sec1", NodeRole.SECTION, title="Section 1")
        chunk1_target = _node("target_x", NodeRole.FIELD_GROUP, title="Target X")
        chunk1_appendix = _node(
            "appendixA", NodeRole.APPENDIX, title="Appendix A", children=[chunk1_target]
        )
        chunk1_orphan = _node("orphan_b", NodeRole.TABLE, title="Orphan B")

        all_chunks = [
            [chunk0_section, chunk0_orphan],
            [chunk1_appendix, chunk1_orphan],
        ]

        def responder(_human_text: str):
            return _consolidation_output(
                [
                    ParentDecision(node_id="orphan_a", decided_parent_id="target_x"),
                    ParentDecision(node_id="orphan_b", decided_parent_id=None),
                ]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(fake_ainvoke.calls) == 1  # collapsed to a single batch/call.
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan_a": "target_x", "orphan_b": None}


class TestConsolidateStructureDefaultCeilingRegressesToSlidingWindow:
    """Regression test for the `consolidation_max_input_tokens` default
    lowered from `131072` to `65536` (see config.py).

    Motivation: the token estimate used by `fits_single_batch` relies on
    tiktoken's `o200k_base` encoding (an approximation of GPT-4o/GPT-5 — no
    public Claude tokenizer exists), which can undercount real Claude/Bedrock
    tokens for the kind of repetitive, quote/dash/colon-heavy text rendered
    into the node pool. With the OLD 131072 default, an honestly ~90k-token
    pool (comfortably below 131072, but well above the new 65536) would have
    wrongly collapsed `fits_single_batch` to True, silently skipping the
    sliding-window algorithm entirely. This test pins that with the new
    65536 default, the very same pool correctly activates the sliding-window
    batching path instead.
    """

    async def test_pool_between_new_and_old_default_activates_sliding_window(self):
        """No explicit `consolidation_max_input_tokens` override anywhere —
        relies entirely on the module-level `_configure_min` autouse
        fixture's plain `configure()` call, which resolves the ceiling to
        the library default (65536, not 131072).
        """
        n_chunks = 118
        all_chunks: list[list[StructureNode]] = []
        for i in range(n_chunks):
            section = _node(f"sec{i}", NodeRole.SECTION, title=_long_filler(f"Section {i}"))
            orphan = _node(f"orphan{i}", NodeRole.SUBSECTION, title=_long_filler(f"Orphan {i}"))
            all_chunks.append([section, orphan])

        # Calibration sanity check: the full pool's estimated size (system
        # prompt + rendered pool — the exact formula consolidate_structure()
        # itself uses to decide `fits_single_batch`) must land strictly
        # between the new default (65536) and the old one (131072), or this
        # test would not actually be exercising the reported scenario.
        from scinr.newton.extraction.structure_consolidation import (
            _estimate_tokens,
            _render_pool,
            build_consolidation_prompt,
        )

        orphan_ids = {
            node.node_id
            for chunk in all_chunks
            for node in chunk
            if node.role == NodeRole.SUBSECTION
        }
        full_pool_text = _render_pool(all_chunks, orphan_ids)
        system_prompt = build_consolidation_prompt(partial_visibility=False)
        full_input_est = _estimate_tokens(system_prompt) + _estimate_tokens(full_pool_text)
        assert 65536 < full_input_est < 131072, (
            f"Calibration drifted: full_input_est={full_input_est} must lie strictly "
            "between the new default (65536) and the old default (131072) for this "
            "test to actually pin the reported scenario."
        )

        def responder(human_text: str):
            # Resolve every orphan named in this call's <must_decide> block
            # immediately (never null) — the content of the decision is not
            # the point of this test, only the batching behavior is, and an
            # immediate resolution keeps `pending_orphans` from growing
            # unboundedly across batches.
            must_decide_block = human_text.split("<must_decide>")[1].split("</must_decide>")[0]
            ids_in_block = [
                line.split('node_id="')[1].split('"')[0]
                for line in must_decide_block.splitlines()
                if 'node_id="' in line
            ]
            return _consolidation_output(
                [
                    ParentDecision(node_id=nid, decided_parent_id="root_ref")
                    for nid in ids_in_block
                ]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        # The core assertion: with the OLD 131072 default this pool would
        # have collapsed to exactly one call (fits_single_batch=True,
        # wrongly). With the new 65536 default it must NOT collapse.
        assert len(fake_ainvoke.calls) > 1
        assert len(decisions) == n_chunks  # one decision per orphan, all resolved.


class TestConsolidateStructureSlidingWindowBatching:
    async def test_multiple_batches_and_backward_buffer_limits_visibility(self, mock_llm):
        """A document forced into multiple small batches: each LLM call must
        only ever see its own batch plus the immediately preceding batch —
        never anything from two or more batches back.
        """
        sec0 = _node("sec0", NodeRole.SECTION, title=_long_filler("Section Zero"))
        orphan1 = _node("orphan1", NodeRole.SUBSECTION, title=_long_filler("Orphan One"))
        sec2 = _node("sec2", NodeRole.SECTION, title=_long_filler("Section Two"))
        orphan3 = _node("orphan3", NodeRole.SUBSECTION, title=_long_filler("Orphan Three"))
        all_chunks = [[sec0], [orphan1], [sec2], [orphan3]]

        # A ceiling sized (via the module's own token estimator) so the
        # document does not fit in a single batch, and each per-chunk render
        # is sized so batches naturally land one chunk at a time. See the
        # Coder's design-note comment in structure_consolidation.py for why
        # a chunk's fitting alone right after being excluded from the prior
        # batch commonly re-triggers the atomic single-chunk safeguard —
        # that is expected here too, not a defect.
        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
            consolidation_max_input_tokens=1691,
        )

        def responder(human_text: str):
            must_decide_block = human_text.split("<must_decide>")[1].split("</must_decide>")[0]
            if "orphan1" in must_decide_block:
                assert "orphan3" not in must_decide_block
                assert "sec0" in human_text  # backward buffer from the immediately preceding batch.
                assert "sec2" not in human_text  # not visible yet — two batches ahead.
                return _consolidation_output(
                    [ParentDecision(node_id="orphan1", decided_parent_id="sec0")]
                )
            assert "orphan3" in must_decide_block
            assert "sec2" in human_text  # backward buffer from the immediately preceding batch.
            # sec0/orphan1 are more than one batch back by now — must not leak in.
            assert "sec0" not in human_text
            assert "orphan1" not in human_text
            return _consolidation_output(
                [ParentDecision(node_id="orphan3", decided_parent_id="sec2")]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(fake_ainvoke.calls) == 2  # only the two batches with orphans call the LLM.
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan1": "sec0", "orphan3": "sec2"}


class TestConsolidateStructureCarryForward:
    async def test_orphan_resolved_none_in_origin_batch_is_retried_and_resolved_next_batch(
        self, mock_llm
    ):
        """An orphan that gets `decided_parent_id=None` in its batch of
        origin must NOT be finalized as root immediately — it must be
        carried forward and retried in the next batch, and can still
        resolve to a concrete parent there.
        """
        ref_a = _node("ref_a", NodeRole.SECTION, title=_long_filler("Reference A"))
        orphan_x = _node("orphan_x", NodeRole.SUBSECTION, title=_long_filler("Orphan X"))
        ref_b = _node("ref_b", NodeRole.SECTION, title=_long_filler("Reference B"))
        all_chunks = [[ref_a], [orphan_x], [ref_b]]

        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
            consolidation_max_input_tokens=1691,
        )

        call_count_for_orphan_x = {"n": 0}

        def responder(human_text: str):
            must_decide_block = human_text.split("<must_decide>")[1].split("</must_decide>")[0]
            assert "orphan_x" in must_decide_block
            call_count_for_orphan_x["n"] += 1
            if call_count_for_orphan_x["n"] == 1:
                # First attempt: no confident match yet — must return null,
                # not force a placement.
                return _consolidation_output(
                    [ParentDecision(node_id="orphan_x", decided_parent_id=None)]
                )
            # Second attempt (next batch): ref_b is now visible — resolve it.
            assert "ref_b" in human_text
            return _consolidation_output(
                [ParentDecision(node_id="orphan_x", decided_parent_id="ref_b")]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        decisions = await consolidate_structure(all_chunks, llm=llm)

        assert call_count_for_orphan_x["n"] == 2  # retried exactly once more, not given up early.
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan_x": "ref_b"}  # NOT prematurely finalized as root.


class TestConsolidateStructureFinalFallback:
    async def test_orphan_never_resolved_falls_back_to_root_after_exhausting_all_batches(
        self, mock_llm, caplog
    ):
        """An orphan that never receives a concrete `decided_parent_id` in
        any batch — including retries — must be finalized as root only once
        every batch has been exhausted, with a clear warning log.
        """
        ref_a = _node("ref_a", NodeRole.SECTION, title=_long_filler("Reference A"))
        orphan_z = _node("orphan_z", NodeRole.SUBSECTION, title=_long_filler("Orphan Z"))
        ref_b = _node("ref_b", NodeRole.SECTION, title=_long_filler("Reference B"))
        all_chunks = [[ref_a], [orphan_z], [ref_b]]

        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
            consolidation_max_input_tokens=1691,
        )

        def responder(_human_text: str):
            return _consolidation_output(
                [ParentDecision(node_id="orphan_z", decided_parent_id=None)]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        with caplog.at_level(logging.WARNING, logger="scinr.newton.extraction.structure_consolidation"):
            decisions = await consolidate_structure(all_chunks, llm=llm)

        assert len(fake_ainvoke.calls) == 2  # retried once (origin batch + one more), then exhausted.
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan_z": None}  # finalized as root, present (not missing).
        assert any(
            "exhausted all batches" in rec.message and "orphan_z" in rec.message
            for rec in caplog.records
        )


class TestConsolidateStructureAtomicBatchSafeguard:
    async def test_single_chunk_exceeding_available_budget_becomes_atomic_batch_without_hanging(
        self, mock_llm, caplog
    ):
        """When even a single chunk's own rendered pool exceeds the
        available per-batch budget, it must still be processed — as an
        atomic batch of size 1, with a warning — rather than hanging,
        crashing, or making zero progress.
        """
        ref_a = _node("ref_a", NodeRole.SECTION, title=_long_filler("Reference A"))
        orphan_w = _node("orphan_w", NodeRole.SUBSECTION, title=_long_filler("Orphan W"))
        all_chunks = [[ref_a], [orphan_w]]

        # available_for_pool = consolidation_max_input_tokens
        #     - tokens(partial_visibility=True system prompt) - margin(500)
        # Sized deliberately tiny (a handful of tokens) so every chunk
        # (hundreds of tokens each, via _long_filler) exceeds it alone.
        configure(
            llm=mock_llm,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
            consolidation_max_input_tokens=1296,
        )

        def responder(_human_text: str):
            return _consolidation_output(
                [ParentDecision(node_id="orphan_w", decided_parent_id="ref_a")]
            )

        fake_ainvoke = _FakeAinvoke(responder)
        llm = _FakeLLM(fake_ainvoke)

        with caplog.at_level(logging.WARNING, logger="scinr.newton.extraction.structure_consolidation"):
            decisions = await consolidate_structure(all_chunks, llm=llm)

        # Completed without hanging/raising, and made progress on every chunk.
        by_id = {d.node_id: d.decided_parent_id for d in decisions}
        assert by_id == {"orphan_w": "ref_a"}
        assert any(
            "exceeds the remaining batch budget" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# 3. assemble_tree()
# ---------------------------------------------------------------------------


class TestAssembleTreeNoOp:
    def test_fully_self_contained_chunks_produce_untouched_root(self):
        """No orphans anywhere — every chunk's own SECTION/APPENDIX
        top-level nodes (with their local children fully intact) simply
        become the root list, in chunk order.
        """
        leaf = _node("leaf", NodeRole.TABLE, title="Leaf")
        chunk0_top = _node("sec1", NodeRole.SECTION, title="Section 1", children=[leaf])
        chunk1_top = _node("app1", NodeRole.APPENDIX, title="Appendix 1")

        all_chunks = [[chunk0_top], [chunk1_top]]

        tree = assemble_tree(all_chunks, decisions=[])

        assert [n.node_id for n in tree] == ["sec1", "app1"]
        assert tree[0].children == [leaf]
        assert tree[0].children[0].node_id == "leaf"
        assert tree[1].children == []


class TestAssembleTreeDeepAttachment:
    def test_orphan_attaches_to_nested_node_in_different_chunk(self):
        """The core new capability: an orphan top-level node from one chunk
        is placed as a child of a node several levels deep inside a
        DIFFERENT chunk's local subtree.
        """
        deep_leaf = _node("deep_leaf", NodeRole.FIELD_GROUP, title="Deep leaf")
        deep_mid = _node("deep_mid", NodeRole.SUBSECTION, title="Deep mid", children=[deep_leaf])
        chunk0_top = _node("sec1", NodeRole.SECTION, title="Section 1", children=[deep_mid])

        orphan = _node("orphan1", NodeRole.SUBSECTION, title="Orphan")
        all_chunks = [[chunk0_top], [orphan]]

        decisions = [ParentDecision(node_id="orphan1", decided_parent_id="deep_leaf")]
        tree = assemble_tree(all_chunks, decisions)

        assert [n.node_id for n in tree] == ["sec1"]  # orphan is NOT at root.
        placed_deep_leaf = tree[0].children[0].children[0]
        assert placed_deep_leaf.node_id == "deep_leaf"
        assert [c.node_id for c in placed_deep_leaf.children] == ["orphan1"]


class TestAssembleTreeCycles:
    def test_direct_self_reference_cycle_falls_back_to_root(self):
        """An orphan whose decided_parent_id is its own node_id must not
        hang or crash — it falls back to root."""
        orphan = _node("orphan1", NodeRole.SUBSECTION, title="Self-referencing orphan")
        section = _node("sec1", NodeRole.SECTION, title="Section 1")
        all_chunks = [[section, orphan]]

        decisions = [ParentDecision(node_id="orphan1", decided_parent_id="orphan1")]
        tree = assemble_tree(all_chunks, decisions)

        assert {n.node_id for n in tree} == {"sec1", "orphan1"}
        # orphan1 kept no children from the bogus self-attachment attempt.
        orphan_in_tree = next(n for n in tree if n.node_id == "orphan1")
        assert orphan_in_tree.children == []

    def test_compound_cycle_between_two_orphans_falls_back_both_to_root(self):
        """Orphan A's decision points to a node nested inside orphan B's
        local subtree, and B's decision points back into A's local subtree.
        Per the actual implementation (_has_cycle() is evaluated
        independently, starting from EACH orphan's own node_id), both A and
        B are detected as participating in a cycle and BOTH fall back to
        root — this is deterministic, not an arbitrary "sever one side"
        choice, because assemble_tree() runs the same cycle check
        independently once per orphan in the outer loop.
        """
        a_child = _node("a_child", NodeRole.TABLE, title="A's child")
        orphan_a = _node("orphan_a", NodeRole.SUBSECTION, title="Orphan A", children=[a_child])

        b_child = _node("b_child", NodeRole.TABLE, title="B's child")
        orphan_b = _node("orphan_b", NodeRole.SUBSECTION, title="Orphan B", children=[b_child])

        all_chunks = [[orphan_a], [orphan_b]]
        decisions = [
            ParentDecision(node_id="orphan_a", decided_parent_id="b_child"),
            ParentDecision(node_id="orphan_b", decided_parent_id="a_child"),
        ]

        tree = assemble_tree(all_chunks, decisions)

        assert {n.node_id for n in tree} == {"orphan_a", "orphan_b"}
        # Local children untouched — neither orphan actually got attached
        # anywhere, and their own local subtrees are unaffected by the
        # cycle detection short-circuit.
        a_in_tree = next(n for n in tree if n.node_id == "orphan_a")
        b_in_tree = next(n for n in tree if n.node_id == "orphan_b")
        assert [c.node_id for c in a_in_tree.children] == ["a_child"]
        assert [c.node_id for c in b_in_tree.children] == ["b_child"]


class TestAssembleTreeAdversarialInput:
    def test_bogus_decision_for_a_section_node_id_is_ignored(self):
        """Defense-in-depth: if consolidation somehow returns a decision
        entry keyed by a SECTION/APPENDIX node_id (which should never be in
        the "must decide" set), assemble_tree() must ignore it — that node
        still lands at root via its unconditional top-level-role placement,
        never consulting decision_by_id for it at all.
        """
        section = _node("sec1", NodeRole.SECTION, title="Section 1")
        appendix = _node("app1", NodeRole.APPENDIX, title="Appendix 1")
        all_chunks = [[section, appendix]]

        # Bogus: a decision entry for a SECTION's own node_id, pointing
        # nonsensically at the APPENDIX.
        decisions = [ParentDecision(node_id="sec1", decided_parent_id="app1")]

        tree = assemble_tree(all_chunks, decisions)

        assert [n.node_id for n in tree] == ["sec1", "app1"]
        assert tree[0].children == []
        assert tree[1].children == []


class TestAssembleTreeDuplicateMergePrePass:
    """See module docstring in structure_consolidation.py: the original
    design discussion called for a duplicate-node merge pre-pass (merging
    chunk-top-level duplicates by (source_page_ids, title, role)) before
    assembling the tree. Reading the actual implementation of
    assemble_tree() above (no such pre-pass exists — it only deduplicates
    parent DECISIONS via decision_by_id, never merges duplicate NODEs by
    content), this capability is absent from the shipped code.

    No test is written to pin behavior that does not exist; this class
    documents the absence per the test-authorship brief instead of
    asserting something that would only pass by accident.
    """

    def test_documented_absence_no_assertion_needed(self):
        pytest.skip(
            "Duplicate-node merge pre-pass (by source_page_ids/title/role) is not "
            "implemented in structure_consolidation.assemble_tree() — nothing to test. "
            "See class docstring."
        )


# ---------------------------------------------------------------------------
# 4. write_map_checkpoint() / delete_map_checkpoint()
# ---------------------------------------------------------------------------


class TestCheckpointIO:
    def test_write_map_checkpoint_writes_namespaced_content(self, tmp_path: Path):
        leaf = _node("page-1::leaf", NodeRole.TABLE, title="Leaf")
        top = _node("page-1::sec1", NodeRole.SECTION, title="Section 1", children=[leaf])
        all_chunks = [[top], []]

        checkpoint_path = tmp_path / "doc.map-checkpoint.json"
        write_map_checkpoint(checkpoint_path, all_chunks)

        assert checkpoint_path.exists()
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert len(payload) == 2  # one entry per chunk, including the empty one.
        assert payload[0][0]["node_id"] == "page-1::sec1"
        assert payload[0][0]["children"][0]["node_id"] == "page-1::leaf"
        assert payload[1] == []

    def test_delete_map_checkpoint_removes_existing_file(self, tmp_path: Path):
        checkpoint_path = tmp_path / "doc.map-checkpoint.json"
        checkpoint_path.write_text("[]", encoding="utf-8")
        assert checkpoint_path.exists()

        delete_map_checkpoint(checkpoint_path)

        assert not checkpoint_path.exists()

    def test_delete_map_checkpoint_on_missing_path_is_a_noop(self, tmp_path: Path):
        checkpoint_path = tmp_path / "does-not-exist.map-checkpoint.json"
        assert not checkpoint_path.exists()

        delete_map_checkpoint(checkpoint_path)  # must not raise.

        assert not checkpoint_path.exists()
