"""
tests/unit/test_tabular_route_row_values.py — Unit tests for the
containment-based value merging added to the tabular ingestion pipeline in
scinr.newton.tabular.neo4j_ops:

1. _merge_values() — pure merge/dedup-by-containment helper.
2. _classify_merge_type() — Pydantic annotation classifier (str / list_str /
   other), with Optional/Union unwrapping and fail-safe defaults.
3. _route_row_values() — full routing of a row dict into primary/
   supplementary/complementary kwargs, now type-aware for the primary target.
4. _instantiate_composite_from_row() — light end-to-end check that merged
   values actually land correctly on a constructed composite instance.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Union

import pytest
from pydantic import BaseModel

from scinr.newton.tabular.models import ColumnFieldMapping, ColumnMapping
from scinr.newton.tabular.neo4j_ops import (
    _classify_merge_type,
    _instantiate_composite_from_row,
    _merge_values,
    _route_row_values,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mapping(*entries: tuple[str, str, str, str]) -> ColumnMapping:
    """Build a ColumnMapping from (column_name, model_field_name, target_model,
    confidence) tuples, in the given order (order matters for determinism)."""
    return ColumnMapping(
        mappings=[
            ColumnFieldMapping(
                column_name=col,
                model_field_name=field,
                target_model=target,
                confidence=confidence,
            )
            for col, field, target, confidence in entries
        ]
    )


# ---------------------------------------------------------------------------
# 1. _merge_values() — direct unit tests
# ---------------------------------------------------------------------------


class TestMergeValues:
    def test_containment_by_substring_and_subset_forward_order(self):
        """Bigger value first, smaller/contained values later: only the
        biggest survives."""
        assert _merge_values(["Amox 500 mg", "500 mg", "mg"]) == ["Amox 500 mg"]

    def test_containment_reverse_order_same_result(self):
        """Same set of values in reverse order (smallest first) must yield
        the same final result — containment merging is order-independent in
        outcome, even though processing is sequential."""
        assert _merge_values(["mg", "500 mg", "Amox 500 mg"]) == ["Amox 500 mg"]

    def test_case_insensitive_dedup_keeps_first_trimmed_original(self):
        """Comparison is case-insensitive (via normalization), but the
        survivor keeps the ORIGINAL casing of the first-seen value, only
        trimmed — no lowercasing of the retained string."""
        result = _merge_values(["Madrid", "MADRID "])
        assert result == ["Madrid"]

    def test_no_containment_relation_keeps_both_in_order(self):
        """Unrelated values (no substring / token-subset relation) are both
        preserved, in order of first appearance."""
        assert _merge_values(["Amoxicillin", "Paracetamol"]) == [
            "Amoxicillin",
            "Paracetamol",
        ]

    def test_empty_and_whitespace_only_entries_are_filtered(self):
        """Empty strings and whitespace-only strings are dropped; only the
        real value survives."""
        assert _merge_values(["", "  ", "X"]) == ["X"]

    def test_single_whitespace_only_value_collapses_to_empty_list(self):
        """EDGE CASE (intentional behavior change vs. legacy code): when the
        ONLY value contributed to a field is whitespace-only (e.g. "   "),
        _merge_values() still applies its trim+empty-filter even though there
        is nothing to deduplicate against. The result is an empty list, NOT
        a list containing the literal whitespace string.

        This differs from the old "last column wins" behavior, which would
        have preserved "   " verbatim. This is a deliberate, accepted
        consequence of running everything (single or multiple contributing
        columns) through the same _merge_values() pipeline — documented here
        so it isn't mistaken for a regression later.
        """
        assert _merge_values(["   "]) == []

    def test_single_normal_value_passthrough(self):
        """A single non-empty value with no peers is trimmed but otherwise
        passed through unchanged."""
        assert _merge_values(["  Madrid  "]) == ["Madrid"]

    def test_empty_input_list(self):
        assert _merge_values([]) == []

    def test_bigger_value_arrives_after_smaller_survivor_replaces_it(self):
        """If a smaller/subset value is accepted first and a bigger/superset
        value arrives later, the smaller survivor is replaced."""
        assert _merge_values(["500 mg", "Amox 500 mg"]) == ["Amox 500 mg"]

    def test_word_token_subset_without_substring_relation(self):
        """Containment can also be a token-subset relation, not just a
        literal substring (e.g. word order differs)."""
        # "mg 500" tokens {"mg", "500"} are a subset of "Amox 500 mg" tokens
        # {"amox", "500", "mg"} even though "mg 500" is not a literal
        # substring of "Amox 500 mg".
        assert _merge_values(["Amox 500 mg", "mg 500"]) == ["Amox 500 mg"]

    def test_internal_whitespace_collapsed_for_comparison_only(self):
        """Multiple internal spaces are collapsed for comparison purposes,
        but the survivor's original spacing (post-trim) is preserved."""
        result = _merge_values(["Amox   500   mg", "500 mg"])
        assert result == ["Amox   500   mg"]


# ---------------------------------------------------------------------------
# 2. _classify_merge_type() — annotation classification tests
# ---------------------------------------------------------------------------


class TestClassifyMergeType:
    def test_plain_str(self):
        assert _classify_merge_type(str) == "str"

    def test_str_or_none_pep604(self):
        assert _classify_merge_type(str | None) == "str"

    def test_optional_str(self):
        assert _classify_merge_type(Optional[str]) == "str"

    def test_list_str(self):
        assert _classify_merge_type(list[str]) == "list_str"

    def test_list_str_or_none(self):
        assert _classify_merge_type(list[str] | None) == "list_str"

    def test_optional_list_str(self):
        assert _classify_merge_type(Optional[list[str]]) == "list_str"

    def test_bare_list_generic_unparametrized(self):
        """Bare `list` with no type args at all is fail-safe classified as
        list_str."""
        assert _classify_merge_type(list) == "list_str"

    def test_int_is_other(self):
        assert _classify_merge_type(int) == "other"

    def test_int_or_none_is_other(self):
        """Optional wrapping is unwrapped down to the real inner type (int),
        which is still "other" — Optional itself must not cause a fail-safe
        str/list_str misclassification."""
        assert _classify_merge_type(int | None) == "other"

    def test_optional_int_typing_style(self):
        assert _classify_merge_type(Optional[int]) == "other"

    def test_any_is_str_failsafe(self):
        assert _classify_merge_type(Any) == "str"

    def test_none_annotation_is_str_failsafe(self):
        """No annotation at all (None) — fail-safe default is str."""
        assert _classify_merge_type(None) == "str"

    def test_list_of_int_is_other(self):
        """list[int] (or any list of a non-str type) is not eligible for
        containment merging."""
        assert _classify_merge_type(list[int]) == "other"

    def test_complex_multi_member_union_is_str_failsafe(self):
        """A Union with more than one non-None member can't be confidently
        classified — fail-safe to str."""
        assert _classify_merge_type(Union[str, int]) == "str"

    def test_nested_basemodel_is_other(self):
        class _Nested(BaseModel):
            x: int = 0

        assert _classify_merge_type(_Nested) == "other"

    def test_nested_basemodel_optional_is_other(self):
        class _Nested(BaseModel):
            x: int = 0

        assert _classify_merge_type(_Nested | None) == "other"


# ---------------------------------------------------------------------------
# 3. _route_row_values() — integration tests
# ---------------------------------------------------------------------------


class _PrimaryStr(BaseModel):
    """Primary model with a plain str field (merge + join semantics)."""

    name: str | None = None


class _PrimaryList(BaseModel):
    """Primary model with a list[str] field (merge, real list, no join)."""

    tags: list[str] | None = None


class _PrimaryOther(BaseModel):
    """Primary model with a non-str/list field (legacy last-wins semantics)."""

    count: int | None = None


class _PrimaryMixed(BaseModel):
    """Primary model exercising all three categories at once."""

    name: str | None = None
    tags: list[str] | None = None
    count: int | None = None


class TestRouteRowValuesNoCollision:
    def test_single_column_single_field_unchanged(self):
        """No collision (1 column -> 1 field): behaves exactly as before —
        the raw value passes through unchanged (only merge-trimmed)."""
        mapping = _mapping(("Name", "name", "primary", "high"))
        row_dict = {"Name": "Amoxicillin"}

        primary_kwargs, supp_kwargs, comp_kwargs = _route_row_values(
            row_dict, mapping, _PrimaryStr
        )

        assert primary_kwargs == {"name": "Amoxicillin"}
        assert supp_kwargs == {}
        assert comp_kwargs == {}

    def test_single_column_list_str_field_yields_single_element_list(self):
        """No collision (1 column -> 1 list[str] field): the majority case
        for list-typed fields. The result must be a list containing exactly
        the one value, NOT a bare string — this end-to-end path was
        previously only covered for the 2+ column collision case."""
        mapping = _mapping(("ColA", "tags", "primary", "high"))
        row_dict = {"ColA": "antibiotic"}

        primary_kwargs, supp_kwargs, comp_kwargs = _route_row_values(
            row_dict, mapping, _PrimaryList
        )

        assert primary_kwargs == {"tags": ["antibiotic"]}
        assert isinstance(primary_kwargs["tags"], list)
        assert supp_kwargs == {}
        assert comp_kwargs == {}


class TestRouteRowValuesPrimaryStrCollision:
    def test_two_columns_same_str_field_merged_and_joined(self):
        mapping = _mapping(
            ("ColA", "name", "primary", "high"),
            ("ColB", "name", "primary", "high"),
        )
        row_dict = {"ColA": "Amox 500 mg", "ColB": "500 mg"}

        primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryStr)

        assert primary_kwargs == {"name": "Amox 500 mg"}

    def test_three_columns_disjoint_values_joined_with_semicolon(self):
        mapping = _mapping(
            ("ColA", "name", "primary", "high"),
            ("ColB", "name", "primary", "high"),
        )
        row_dict = {"ColA": "Amoxicillin", "ColB": "Paracetamol"}

        primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryStr)

        assert primary_kwargs == {"name": "Amoxicillin; Paracetamol"}


class TestRouteRowValuesPrimaryListCollision:
    def test_two_columns_same_list_field_merged_as_real_list(self):
        mapping = _mapping(
            ("ColA", "tags", "primary", "high"),
            ("ColB", "tags", "primary", "high"),
        )
        row_dict = {"ColA": "antibiotic", "ColB": "penicillin-class"}

        primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryList)

        assert primary_kwargs == {"tags": ["antibiotic", "penicillin-class"]}
        assert isinstance(primary_kwargs["tags"], list)

    def test_list_field_containment_dedup(self):
        mapping = _mapping(
            ("ColA", "tags", "primary", "high"),
            ("ColB", "tags", "primary", "high"),
            ("ColC", "tags", "primary", "high"),
        )
        row_dict = {"ColA": "Amox 500 mg", "ColB": "500 mg", "ColC": "mg"}

        primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryList)

        assert primary_kwargs == {"tags": ["Amox 500 mg"]}


class TestRouteRowValuesPrimaryOtherCollision:
    def test_other_type_last_column_wins_no_merge_and_warns(self, caplog):
        mapping = _mapping(
            ("ColA", "count", "primary", "high"),
            ("ColB", "count", "primary", "high"),
        )
        row_dict = {"ColA": "10", "ColB": "20"}

        with caplog.at_level(logging.WARNING):
            primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryOther)

        # Last column ("ColB" -> "20") wins, unmerged (str, no combining).
        assert primary_kwargs == {"count": "20"}
        assert any(
            "unsupported" in record.message and "count" in record.message
            for record in caplog.records
        )

    def test_other_type_single_column_no_warning(self, caplog):
        """A single contributing column for an OTHER-typed field must not
        trigger the multi-value warning (only fires when there is an actual
        conflict, i.e. >= 2 columns)."""
        mapping = _mapping(("ColA", "count", "primary", "high"),)
        row_dict = {"ColA": "10"}

        with caplog.at_level(logging.WARNING):
            primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryOther)

        assert primary_kwargs == {"count": "10"}
        assert not any("unsupported" in record.message for record in caplog.records)


class TestRouteRowValuesSupplementaryCollision:
    def test_supplementary_collision_merged_and_joined_regardless_of_type(self):
        """supplementary targets have no accessible Pydantic class, so they
        are always treated as str: merge + join, no type introspection."""
        mapping = _mapping(
            ("ColA", "notes", "supplementary", "high"),
            ("ColB", "notes", "supplementary", "high"),
        )
        row_dict = {"ColA": "Stored at 25C", "ColB": "25C"}

        _, supp_kwargs, _ = _route_row_values(row_dict, mapping, _PrimaryStr)

        assert supp_kwargs == {"notes": "Stored at 25C"}


class TestRouteRowValuesComplementaryCollision:
    def test_complementary_model_collision_merged_and_joined(self):
        """target = a complementary CamelCase class name (not 'primary' or
        'supplementary') gets the same str merge+join treatment as
        supplementary."""
        mapping = _mapping(
            ("ColA", "batch_id", "BatchAnalysis", "high"),
            ("ColB", "batch_id", "BatchAnalysis", "high"),
        )
        row_dict = {"ColA": "B-100", "ColB": "100"}

        _, _, comp_kwargs = _route_row_values(row_dict, mapping, _PrimaryStr)

        assert comp_kwargs == {"BatchAnalysis": {"batch_id": "B-100"}}


class TestRouteRowValuesLowConfidenceAndExtra:
    def test_low_confidence_entries_are_discarded_before_grouping(self):
        mapping = _mapping(
            ("ColA", "name", "primary", "high"),
            ("ColB", "name", "primary", "low"),
        )
        row_dict = {"ColA": "Amoxicillin", "ColB": "SomethingElse"}

        primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryStr)

        # The "low" confidence column must not participate at all.
        assert primary_kwargs == {"name": "Amoxicillin"}

    def test_extra_field_name_is_ignored(self):
        mapping = _mapping(
            ("ColA", "name", "primary", "high"),
            ("ColB", "__extra__", "primary", "high"),
        )
        row_dict = {"ColA": "Amoxicillin", "ColB": "Junk column value"}

        primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryStr)

        assert primary_kwargs == {"name": "Amoxicillin"}
        assert "__extra__" not in primary_kwargs


class TestRouteRowValuesDeterminism:
    def test_same_input_always_yields_same_output(self):
        mapping = _mapping(
            ("ColA", "name", "primary", "high"),
            ("ColB", "name", "primary", "high"),
            ("ColC", "tags", "primary", "high"),
            ("ColD", "tags", "primary", "high"),
        )
        row_dict = {
            "ColA": "Amox 500 mg",
            "ColB": "500 mg",
            "ColC": "antibiotic",
            "ColD": "penicillin",
        }

        result_1 = _route_row_values(row_dict, mapping, _PrimaryMixed)
        result_2 = _route_row_values(row_dict, mapping, _PrimaryMixed)

        assert result_1 == result_2


class TestRouteRowValuesUnknownField:
    def test_field_not_in_primary_model_warns_and_is_skipped(self, caplog):
        """A field_name that doesn't exist on primary_cls.model_fields must
        emit the existing warning and be silently skipped — must not raise."""
        mapping = _mapping(("ColA", "nonexistent_field", "primary", "high"),)
        row_dict = {"ColA": "some value"}

        with caplog.at_level(logging.WARNING):
            primary_kwargs, _, _ = _route_row_values(row_dict, mapping, _PrimaryStr)

        assert primary_kwargs == {}
        assert any(
            "not found in primary model" in record.message for record in caplog.records
        )


# ---------------------------------------------------------------------------
# 4. _instantiate_composite_from_row() — light end-to-end check
# ---------------------------------------------------------------------------


class _CompositeModel(BaseModel):
    """Minimal composite: primary field + one supplementary-style extra
    field, mirroring the shape compose_extraction_schema() would produce."""

    primary_mixed: _PrimaryMixed | None = None
    site_notes: str | None = None


class TestInstantiateCompositeFromRow:
    def test_merged_str_and_list_fields_land_correctly_on_instance(self):
        mapping = _mapping(
            ("ColA", "name", "primary", "high"),
            ("ColB", "name", "primary", "high"),
            ("ColC", "tags", "primary", "high"),
            ("ColD", "tags", "primary", "high"),
            ("ColE", "site_notes", "supplementary", "high"),
            ("ColF", "site_notes", "supplementary", "high"),
        )
        row_dict = {
            "ColA": "Amox 500 mg",
            "ColB": "500 mg",
            "ColC": "antibiotic",
            "ColD": "penicillin",
            "ColE": "Stored at 25C",
            "ColF": "25C",
        }

        instance = _instantiate_composite_from_row(
            primary_cls=_PrimaryMixed,
            composite_cls=_CompositeModel,
            primary_field_name="primary_mixed",
            mapping=mapping,
            row_dict=row_dict,
        )

        assert instance is not None
        assert instance.primary_mixed.name == "Amox 500 mg"
        assert instance.primary_mixed.tags == ["antibiotic", "penicillin"]
        assert isinstance(instance.primary_mixed.tags, list)
        assert instance.site_notes == "Stored at 25C"
