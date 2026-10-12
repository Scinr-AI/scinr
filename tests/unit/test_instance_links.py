"""
tests/unit/test_instance_links.py — Pure rules of same-extraction sibling
resolution and of the instance_relationships declaration checks
(scinr.newton.entity_extraction.instance_links). No Neo4j.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from pydantic import BaseModel, Field

from scinr.newton.entity_extraction.instance_links import (
    SiblingCandidate,
    check_instance_relationships,
    compare_join_value,
    join_key_fields,
    match_sibling,
    multi_column_join_fields,
    warn_instance_relationships,
)


def _cand(uid: str, model: str = "Target", keyed: bool = False, **values) -> SiblingCandidate:
    return SiblingCandidate(uid=uid, model_class=model, values=values, keyed=keyed)


# ---------------------------------------------------------------------------
# compare_join_value
# ---------------------------------------------------------------------------


def test_compare_equal_after_normalisation() -> None:
    assert compare_join_value("abc-1", "  ABC-1 ") == "equal"
    assert compare_join_value("cafe", "Café") == "equal"


def test_compare_contains_when_candidate_packs_several_values() -> None:
    # The production shape: "312947; CANTABNEW-0081" built from two columns.
    assert compare_join_value("312947", "312947; CANTABNEW-0081") == "contains"
    assert compare_join_value("cantabnew-0081", "312947; CANTABNEW-0081") == "contains"


def test_compare_contains_when_reference_packs_several_values() -> None:
    assert compare_join_value("312947; cantabnew-0081", "312947") == "contains"


def test_compare_list_candidate() -> None:
    assert compare_join_value("b", ["a", "b"]) == "contains"
    assert compare_join_value("a; b", ["b", "a"]) == "equal"


def test_compare_missing_and_different() -> None:
    assert compare_join_value("x", None) == "missing"
    assert compare_join_value("x", "  ") == "missing"
    assert compare_join_value("x", []) == "missing"
    assert compare_join_value("x", "y") == "different"
    # Partial overlap is not containment.
    assert compare_join_value("a; b", "b; c") == "different"


# ---------------------------------------------------------------------------
# match_sibling — the four behaviours
# ---------------------------------------------------------------------------


def test_no_sibling_of_target_class_means_none() -> None:
    m = match_sibling("Target", {"k": "1"}, [_cand("a", model="Other", k="1")])
    assert m.outcome == "none" and m.uid is None and not m.linked


def test_source_is_never_its_own_target() -> None:
    m = match_sibling("Target", {"k": "1"}, [_cand("self", k="1")], exclude_uid="self")
    assert m.outcome == "none"


def test_exact_match_among_several_candidates() -> None:
    cands = [_cand("a", k="1"), _cand("b", k="2"), _cand("c", k="3")]
    m = match_sibling("Target", {"k": "2"}, cands)
    assert (m.outcome, m.uid, m.candidates) == ("exact", "b", 3)


def test_exact_beats_compatible() -> None:
    cands = [_cand("a", k="1; 2"), _cand("b", k="2")]
    assert match_sibling("Target", {"k": "2"}, cands).uid == "b"


def test_compatible_match_by_containment() -> None:
    cands = [_cand("a", k="9"), _cand("b", k="312947; CANTABNEW-0081")]
    m = match_sibling("Target", {"k": "312947"}, cands)
    assert (m.outcome, m.uid) == ("compatible", "b")


def test_composite_key_every_field_must_agree() -> None:
    cands = [_cand("a", k="1", v="x"), _cand("b", k="1", v="y")]
    assert match_sibling("Target", {"k": "1", "v": "y"}, cands).uid == "b"


def test_unknown_tier_links_a_single_candidate_without_join_value() -> None:
    m = match_sibling("Target", {"k": "1"}, [_cand("a", keyed=True), _cand("b", k="2", keyed=True)])
    assert (m.outcome, m.uid) == ("unknown", "a")


def test_ambiguous_when_two_candidates_match_equally() -> None:
    m = match_sibling("Target", {"k": "1"}, [_cand("a", k="1"), _cand("b", k="1")])
    assert m.outcome == "ambiguous" and m.uid is None and not m.linked


def test_a_keyed_instance_repeated_in_the_extraction_counts_once() -> None:
    same = _cand("k1", keyed=True, k="1")
    m = match_sibling("Target", {"k": "1"}, [same, _cand("k2", keyed=True, k="2"), same])
    assert (m.outcome, m.uid, m.candidates) == ("exact", "k1", 2)


def test_sole_keyless_sibling_is_linked_even_when_values_differ() -> None:
    m = match_sibling("Target", {"k": "abc"}, [_cand("a", k="something else")])
    assert (m.outcome, m.uid) == ("sole", "a")


def test_sole_rule_does_not_apply_to_a_contradicting_keyed_sibling() -> None:
    # A keyed sibling with another key is a different record: the reference is
    # a cross-document one, resolved by key (shell).
    m = match_sibling("Target", {"k": "fr"}, [_cand("a", keyed=True, k="es")])
    assert m.outcome == "none"


def test_sole_rule_does_not_apply_to_a_fan_out() -> None:
    m = match_sibling("Target", {"k": "x"}, [_cand("a", k="y")], single_target=False)
    assert m.outcome == "none"


def test_several_contradicting_candidates_link_none() -> None:
    m = match_sibling("Target", {"k": "z"}, [_cand("a", k="x"), _cand("b", k="y")])
    assert m.outcome == "none" and m.candidates == 2


# ---------------------------------------------------------------------------
# Declaration analysis
# ---------------------------------------------------------------------------


class Record(BaseModel):
    record_id: str | None = Field(default=None, json_schema_extra={"instance_key": True})
    name: str | None = None


class LooseRecord(BaseModel):
    record_id: str | None = None
    name: str | None = None


class Spec(BaseModel):
    record_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "Record", "join_via": {"record_id": "record_id"}, "rel_type": "SPEC_FOR"}
            ]
        },
    )
    region: str | None = Field(default=None, json_schema_extra={"instance_key": True})
    note: str | None = None


class LooseSpec(BaseModel):
    record_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "LooseRecord", "join_via": {"record_id": "record_id"}, "rel_type": "SPEC_FOR"},
                {"target_model": "Ghost", "join_via": {"record_id": "id"}, "rel_type": "X"},
                {"target_model": "Record", "join_via": {"record_id": "nope"}, "rel_type": "Y"},
            ]
        },
    )


def test_join_key_fields_of_source_and_target() -> None:
    assert join_key_fields(Spec, [Record]) == {"record_id", "region"}
    # Record.record_id is an instance_key and also the remote side of Spec's join.
    assert join_key_fields(Record, [Spec]) == {"record_id"}
    # LooseRecord has no instance_key, but Spec joins on its record_id.
    assert join_key_fields(LooseRecord, [LooseSpec]) == {"record_id"}
    assert join_key_fields(LooseRecord) == set()


def test_check_instance_relationships_clean() -> None:
    assert check_instance_relationships({"Record": Record, "Spec": Spec}) == []


def test_check_instance_relationships_reports_every_problem() -> None:
    problems = check_instance_relationships(
        {"LooseRecord": LooseRecord, "LooseSpec": LooseSpec, "Record": Record}
    )
    joined = "\n".join(problems)
    assert len(problems) == 3, joined
    assert "LooseRecord.record_id is not instance_key" in joined
    assert "'Ghost' is not a known model" in joined
    assert "remote field 'nope' not on Record" in joined


def test_warn_instance_relationships_logs_warnings(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        warn_instance_relationships({"LooseRecord": LooseRecord, "LooseSpec": LooseSpec})
    assert any("not instance_key" in r.message for r in caplog.records)


def _m(column: str, field: str, target: str = "primary", confidence: str = "high"):
    return SimpleNamespace(
        column_name=column, model_field_name=field, target_model=target, confidence=confidence
    )


def test_multi_column_join_fields_flags_only_join_keys() -> None:
    mappings = [
        _m("ID Code", "record_id"),
        _m("ID", "record_id"),
        _m("Name A", "name"),
        _m("Name B", "name"),  # not a key: merging is fine
        _m("Spec ID", "record_id", target="Spec"),
        _m("Other", "record_id", target="Spec", confidence="low"),  # skipped by the router
    ]
    result = multi_column_join_fields(mappings, {"primary": Record, "Spec": Spec}, [Spec])
    assert result == {("primary", "record_id"): ["ID Code", "ID"]}
