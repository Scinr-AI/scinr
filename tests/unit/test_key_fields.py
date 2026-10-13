"""
tests/unit/test_key_fields.py — Key fields hold one value, and
instance_relationships declarations point at instance_key fields.

- entity_extraction.key_fields: which fields are keys; startup checks.
- tabular row routing: several columns mapped to a key field keep the first
  value (no "; " join), once-per-sheet warning.
- end to end on the production shape (a regulatory portfolio record and its
  specification, one row each): with the target keyed, the specification's
  reference computes the record's uid whichever is written first.

No Neo4j: writes go to an AsyncMock session.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

from pydantic import BaseModel, Field

from scinr.newton.entity_extraction.graph_mapper import _write_model_fields
from scinr.newton.entity_extraction.key_fields import (
    check_instance_relationships,
    key_fields,
    warn_instance_relationships,
)
from scinr.newton.tabular import neo4j_ops
from scinr.newton.tabular.models import ColumnFieldMapping, ColumnMapping
from scinr.newton.utils.uid import make_instance_uid

TENANT = "tenant_t"

_SPEC_REL = {
    "instance_relationships": [
        {
            "target_model": "PortfolioRecord",
            "join_via": {"portfolio_record_id": "portfolio_record_id"},
            "rel_type": "SPECIFICATION_FOR",
        }
    ]
}


class PortfolioRecord(BaseModel):
    """Fixed model: one identifier per field, the referenced one is the key."""

    portfolio_record_id: str | None = Field(default=None, json_schema_extra={"instance_key": True})
    authorisation_code: str | None = None
    product_name: str | None = None


class UnkeyedPortfolioRecord(BaseModel):
    """The production bug: the referenced field is not instance_key."""

    portfolio_record_id: str | None = None


class ProductSpec(BaseModel):
    portfolio_record_id: str | None = Field(default=None, json_schema_extra=_SPEC_REL)
    strength: str | None = None


class BrokenSpec(BaseModel):
    portfolio_record_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "UnkeyedPortfolioRecord",
                    "join_via": {"portfolio_record_id": "portfolio_record_id"},
                    "rel_type": "SPECIFICATION_FOR",
                },
                {"target_model": "Ghost", "join_via": {"portfolio_record_id": "x"}, "rel_type": "X"},
                {"target_model": "PortfolioRecord", "join_via": {"missing": "nope"}, "rel_type": "Y"},
            ]
        },
    )


class Condition(BaseModel):
    condition_ids: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "PortfolioRecord", "join_via": {"condition_ids": "portfolio_record_id"}, "rel_type": "R"}
            ]
        },
    )


# ---------------------------------------------------------------------------
# key_fields / startup checks
# ---------------------------------------------------------------------------


def test_key_fields() -> None:
    assert key_fields(PortfolioRecord) == {"portfolio_record_id"}
    assert key_fields(ProductSpec) == {"portfolio_record_id"}
    assert key_fields(UnkeyedPortfolioRecord) == set()


def test_check_passes_for_a_keyed_target() -> None:
    models = {"PortfolioRecord": PortfolioRecord, "ProductSpec": ProductSpec}
    assert check_instance_relationships(models) == []


def test_check_reports_the_production_bug_and_other_mistakes() -> None:
    problems = check_instance_relationships(
        {
            "UnkeyedPortfolioRecord": UnkeyedPortfolioRecord,
            "PortfolioRecord": PortfolioRecord,
            "BrokenSpec": BrokenSpec,
        }
    )
    text = "\n".join(problems)
    assert len(problems) == 4, text
    assert "UnkeyedPortfolioRecord.portfolio_record_id is not instance_key" in text
    assert "'Ghost' is not a known model" in text
    assert "join_via field 'missing' not on BrokenSpec" in text
    assert "join_via target field 'nope' not on PortfolioRecord" in text


def test_startup_check_logs_warnings(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        warn_instance_relationships(
            {"UnkeyedPortfolioRecord": UnkeyedPortfolioRecord, "BrokenSpec": BrokenSpec}
        )
    assert any('"instance_key": True' in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Tabular routing: no "; " join into key fields
# ---------------------------------------------------------------------------


def _mapping(*entries: tuple[str, str, str]) -> ColumnMapping:
    return ColumnMapping(
        mappings=[
            ColumnFieldMapping(column_name=c, model_field_name=f, target_model=t)
            for c, f, t in entries
        ]
    )


ROW = {
    "ID": "CANTABNEW-0081",
    "ID Code": "312947",
    "Name": "Alpha",
    "Name 2": "Alpha 10 mg",
    "Spec ID": "CANTABNEW-0081",
    "Spec ID 2": "X-1",
}


def test_primary_key_field_keeps_the_first_column_only() -> None:
    mapping = _mapping(
        ("ID", "portfolio_record_id", "primary"),
        ("ID Code", "portfolio_record_id", "primary"),
        ("Name", "product_name", "primary"),
        ("Name 2", "product_name", "primary"),
    )
    primary, _, _ = neo4j_ops._route_row_values(ROW, mapping, PortfolioRecord)
    assert primary["portfolio_record_id"] == "CANTABNEW-0081"
    # Non-key fields keep the usual containment merge ("Alpha" ⊂ "Alpha 10 mg").
    assert primary["product_name"] == "Alpha 10 mg"


def test_complementary_key_field_keeps_the_first_column_only() -> None:
    mapping = _mapping(
        ("Spec ID", "portfolio_record_id", "ProductSpec"),
        ("Spec ID 2", "portfolio_record_id", "ProductSpec"),
    )
    _, _, comp = neo4j_ops._route_row_values(
        ROW, mapping, PortfolioRecord, {"ProductSpec": ProductSpec}
    )
    assert comp["ProductSpec"]["portfolio_record_id"] == "CANTABNEW-0081"
    # Without the class map (older callers) the field is merged as before.
    _, _, comp = neo4j_ops._route_row_values(ROW, mapping, PortfolioRecord)
    assert comp["ProductSpec"]["portfolio_record_id"] == "CANTABNEW-0081; X-1"


def test_list_key_field_is_a_fan_out_and_keeps_every_value() -> None:
    mapping = _mapping(("ID", "condition_ids", "primary"), ("ID Code", "condition_ids", "primary"))
    primary, _, _ = neo4j_ops._route_row_values(ROW, mapping, Condition)
    assert primary["condition_ids"] == ["CANTABNEW-0081", "312947"]


def test_multi_column_key_field_is_warned_once_per_sheet(caplog) -> None:
    mapping = _mapping(
        ("ID", "portfolio_record_id", "primary"),
        ("ID Code", "portfolio_record_id", "primary"),
        ("Name", "product_name", "primary"),
        ("Name 2", "product_name", "primary"),  # not a key: no warning
    )
    with caplog.at_level(logging.WARNING, logger=neo4j_ops.logger.name):
        offending = neo4j_ops.warn_multi_column_key_fields(
            mapping, PortfolioRecord, {}, sheet_name="Portfolio"
        )
    assert offending == {("primary", "portfolio_record_id"): ["ID", "ID Code"]}
    (record,) = [r for r in caplog.records if "key field" in r.getMessage()]
    assert "PortfolioRecord.portfolio_record_id" in record.getMessage()
    assert "Only 'ID' is used" in record.getMessage()


# ---------------------------------------------------------------------------
# End to end: the reference lands on the record, whatever the order
# ---------------------------------------------------------------------------


async def _write(instance: BaseModel) -> list:
    session = AsyncMock()
    await _write_model_fields(
        session=session,
        instance=instance,
        parent_uid="er",
        parent_label="ExtractionResult",
        field_path_prefix="",
        entity_nodes={},
        depth=0,
        tenant_id=TENANT,
    )
    return session.run.call_args_list


class _RecordRow(BaseModel):
    portfolio_record: PortfolioRecord


class _SpecRow(BaseModel):
    product_spec: ProductSpec


class _UnkeyedRow(BaseModel):
    portfolio_record: UnkeyedPortfolioRecord


def _record_uid(calls) -> str:
    (uid,) = [c.kwargs["child_uid"] for c in calls if "(child:ModelInstance" in c.args[0]]
    return uid


def _shell_uid(calls) -> str:
    (uid,) = [c.kwargs["tgt_uid"] for c in calls if "MERGE (tgt:ModelInstance" in c.args[0]]
    return uid


async def test_keyed_record_and_reference_compute_the_same_uid_in_any_order() -> None:
    mapping = _mapping(
        ("ID", "portfolio_record_id", "primary"),
        ("ID Code", "portfolio_record_id", "primary"),  # packed before the fix
    )
    primary, _, _ = neo4j_ops._route_row_values(ROW, mapping, PortfolioRecord)
    record = _RecordRow(portfolio_record=PortfolioRecord(**primary))
    spec = _SpecRow(product_spec=ProductSpec(portfolio_record_id=" cantabnew-0081 "))

    spec_first = _shell_uid(await _write(spec))
    record_second = _record_uid(await _write(record))
    record_first = _record_uid(await _write(record))
    spec_second = _shell_uid(await _write(spec))

    expected = make_instance_uid("PortfolioRecord", {"portfolio_record_id": "cantabnew-0081"}, TENANT)
    assert spec_first == record_second == record_first == spec_second == expected


async def test_unkeyed_record_gets_a_random_uid_the_reference_never_reaches() -> None:
    """The production bug, for the record: without instance_key the record is
    CREATEd with a random uid, so a reference's shell is a different node."""
    calls = await _write(_UnkeyedRow(portfolio_record=UnkeyedPortfolioRecord(portfolio_record_id="A")))
    (create,) = [c for c in calls if "CREATE (child:ModelInstance" in c.args[0]]
    assert create.kwargs["child_uid"] != make_instance_uid(
        "UnkeyedPortfolioRecord", {"portfolio_record_id": "a"}, TENANT
    )


async def test_missing_identifier_means_no_key_and_no_link() -> None:
    """A dataset without the identifier: the record is key-less (random uid)
    and the specification writes no relationship — nothing to join on."""
    record_calls = await _write(_RecordRow(portfolio_record=PortfolioRecord(product_name="Alpha")))
    assert any("CREATE (child:ModelInstance" in c.args[0] for c in record_calls)
    spec_calls = await _write(_SpecRow(product_spec=ProductSpec(strength="10 mg")))
    assert not any("MERGE (tgt:ModelInstance" in c.args[0] for c in spec_calls)
