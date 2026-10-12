"""
tests/unit/test_same_row_linking.py — Same-row (same ExtractionResult)
instance_relationships in the tabular pipeline, end to end on small synthetic
tables: column mapping → row routing (tabular.neo4j_ops) → composite instance
→ write_extraction_subgraph (graph_mapper) on a recording fake session.

Covers the four behaviours of the resolution:
- sibling found (2 and 3 models per row, source written before or after its target);
- several candidates (exact pick, ambiguous → shell);
- no sibling → key-addressed shell, as before;
- later fill of a shell by the real instance of another row (cross-row reference);
plus the multi-column join-key warning and supplementary fields on the primary.
No Neo4j, no LLM.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from scinr.newton.entity_extraction import graph_mapper
from scinr.newton.entity_extraction.schema_composer import _to_snake_case, compose_extraction_schema
from scinr.newton.tabular import neo4j_ops
from scinr.newton.tabular.models import ColumnFieldMapping, ColumnMapping
from scinr.newton.utils.uid import make_instance_uid

TENANT = "tenant_t"

# ---------------------------------------------------------------------------
# Synthetic domain models (no catalog dependency)
# ---------------------------------------------------------------------------


class MarketingRecord(BaseModel):
    """Key-less primary: every row gets a random uid."""

    record_id: str | None = None
    product_name: str | None = None
    countries: list[CountryEntry] | None = None


class CountryEntry(BaseModel):
    code: str | None = None


class ProductSpec(BaseModel):
    record_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": "MarketingRecord",
                    "join_via": {"record_id": "record_id"},
                    "rel_type": "SPECIFICATION_FOR",
                }
            ]
        },
    )
    strength: str | None = None


class Site(BaseModel):
    site_id: str | None = Field(default=None, json_schema_extra={"instance_key": True})
    city: str | None = None


class Product(BaseModel):
    product_code: str | None = None
    name: str | None = None


class Batch(BaseModel):
    """Primary whose relationships target complementary models written after it."""

    batch_no: str | None = None
    product_code: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "Product", "join_via": {"product_code": "product_code"}, "rel_type": "OF_PRODUCT"}
            ]
        },
    )
    site_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "Site", "join_via": {"site_id": "site_id"}, "rel_type": "MADE_AT"}
            ]
        },
    )


class Component(BaseModel):
    code: str | None = None


class Formula(BaseModel):
    name: str | None = None
    components: list[Component] | None = None


class ComponentNote(BaseModel):
    component_code: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "Component", "join_via": {"component_code": "code"}, "rel_type": "NOTE_ON"}
            ]
        },
    )


class Part(BaseModel):
    """Keyed, self-referencing: the parent is another row of the same table."""

    part_id: str | None = Field(default=None, json_schema_extra={"instance_key": True})
    parent_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {"target_model": "Part", "join_via": {"parent_id": "part_id"}, "rel_type": "CHILD_OF"}
            ]
        },
    )


MarketingRecord.model_rebuild()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, record: dict | None) -> None:
        self._record = record

    async def single(self) -> dict | None:
        return self._record


class FakeSession:
    """Records every query; answers the StructureNode guard of write_extraction_subgraph."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, query: str, **params: Any) -> _Result:
        self.calls.append((query, params))
        if "RETURN count(n) AS cnt" in query:
            return _Result({"cnt": 1, "tenant_id": TENANT, "created_by_user_id": "u", "job_id": "j"})
        return _Result(None)

    # ── graph views over the recorded calls ──
    def instances(self) -> dict[str, str]:
        """uid → model_class of every ModelInstance written as a real instance."""
        out = {}
        for q, p in self.calls:
            if "(child:ModelInstance" in q and "child_uid" in p:
                out[p["child_uid"]] = p["model_class"]
        return out

    def uid_of(self, model_class: str) -> list[str]:
        return [uid for uid, cls in self.instances().items() if cls == model_class]

    def shells(self) -> list[dict[str, Any]]:
        return [p for q, p in self.calls if "MERGE (tgt:ModelInstance" in q]

    def links(self, rel_type: str | None = None) -> list[tuple[str, str, str]]:
        """(src_uid, tgt_uid, resolution) of every instance relationship MERGE."""
        out = []
        for q, p in self.calls:
            if "ON CREATE SET r.resolution" in q and (rel_type is None or f"`{rel_type}`" in q):
                out.append((p["src_uid"], p["tgt_uid"], p["resolution"]))
        return out


class FakeDriver:
    def __init__(self) -> None:
        self.session_obj = FakeSession()

    @asynccontextmanager
    async def _session(self):
        yield self.session_obj

    def session(self, database: str | None = None):
        return self._session()


@pytest.fixture(autouse=True)
def _fake_config(monkeypatch):
    cfg = SimpleNamespace(neo4j_database="neo4j", tabular_supplementary_on_primary=False)
    monkeypatch.setattr(graph_mapper, "get_config", lambda: cfg)
    monkeypatch.setattr(neo4j_ops, "get_config", lambda: cfg)
    return cfg


# ---------------------------------------------------------------------------
# Table → composite → graph
# ---------------------------------------------------------------------------


def _mapping(*entries: tuple[str, str, str]) -> ColumnMapping:
    return ColumnMapping(
        mappings=[
            ColumnFieldMapping(column_name=c, model_field_name=f, target_model=t) for c, f, t in entries
        ]
    )


async def _write_table(
    primary: type,
    complementary: list[type],
    mapping: ColumnMapping,
    headers: list[str],
    rows: list[list[str]],
    supplementary: list[dict] | None = None,
    supplementary_on_primary: bool = False,
) -> list[FakeSession]:
    """Write every row as its own ExtractionResult; return one session per row."""
    composite_cls = compose_extraction_schema(primary, complementary, supplementary or [])
    primary_field = _to_snake_case(primary.__name__)
    comp_map = {c.__name__: c for c in complementary}
    sessions = []
    for i, values in enumerate(rows):
        composite = neo4j_ops._instantiate_composite_from_row(
            primary_cls=primary,
            composite_cls=composite_cls,
            primary_field_name=primary_field,
            mapping=mapping,
            row_dict=neo4j_ops._build_row_dict(headers, values),
            comp_cls_map=comp_map,
        )
        driver = FakeDriver()
        await graph_mapper.write_extraction_subgraph(
            driver=driver,
            node_full_id=f"{TENANT}::doc/v1/table_1/row_{i + 1}",
            composite_instance=composite,
            primary_model_class=primary.__name__,
            complementary_model_classes=list(comp_map),
            document_name="doc",
            extraction_uid=f"er{i}",
            supplementary_fields=[s["field_name"] for s in supplementary or []]
            if supplementary_on_primary
            else (),
            primary_field_name=primary_field,
        )
        sessions.append(driver.session_obj)
    return sessions


# ── 1. Two models per row: the production shape ────────────────────────────


PORTFOLIO_HEADERS = ["ID Code", "ID", "Product", "Strength", "Spec ID"]
PORTFOLIO_ROWS = [
    ["312947", "CANTABNEW-0081", "Alpha", "10 mg", "312947"],
    ["400001", "CANTABNEW-0099", "Beta", "5 mg", "CANTABNEW-0099"],
    ["500002", "CANTABNEW-0100", "Gamma", "1 mg", "unrelated-value"],
]
PORTFOLIO_MAPPING = _mapping(
    ("ID Code", "record_id", "primary"),
    ("ID", "record_id", "primary"),  # two columns into one join field
    ("Product", "product_name", "primary"),
    ("Strength", "strength", "ProductSpec"),
    ("Spec ID", "record_id", "ProductSpec"),
)


async def test_two_models_per_row_link_to_the_same_row_sibling() -> None:
    sessions = await _write_table(
        MarketingRecord, [ProductSpec], PORTFOLIO_MAPPING, PORTFOLIO_HEADERS, PORTFOLIO_ROWS
    )
    for session in sessions:
        (record_uid,) = session.uid_of("MarketingRecord")
        (spec_uid,) = session.uid_of("ProductSpec")
        assert session.links("SPECIFICATION_FOR") == [(spec_uid, record_uid, "sibling")]
        assert session.shells() == [], "no key-addressed shell when the sibling is in the row"


async def test_sibling_links_are_independent_of_how_the_join_value_was_built(caplog) -> None:
    """Row 1 references one of the two packed values, row 2 the other one
    (compatible: containment), row 3 an unrelated value against the row's only,
    key-less record (sole sibling). All three link to the row's record."""
    with caplog.at_level(logging.INFO, logger=graph_mapper.log.name):
        await _write_table(
            MarketingRecord, [ProductSpec], PORTFOLIO_MAPPING, PORTFOLIO_HEADERS, PORTFOLIO_ROWS
        )
    outcomes = [
        r.args[-1] for r in caplog.records if r.msg.startswith("write_extraction_subgraph: wrote")
    ]
    assert outcomes == [{"compatible": 1}, {"compatible": 1}, {"sole": 1}]


async def test_multi_column_join_key_is_warned_once_per_sheet(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=neo4j_ops.logger.name):
        offending = neo4j_ops.warn_multi_column_join_fields(
            PORTFOLIO_MAPPING, MarketingRecord, {"ProductSpec": ProductSpec}, sheet_name="Portfolio"
        )
    assert offending == {("primary", "record_id"): ["ID Code", "ID"]}
    warnings = [r for r in caplog.records if "join-key field" in r.message]
    assert len(warnings) == 1
    assert "MarketingRecord.record_id" in warnings[0].message


# ── 2. Three models per row, source written before its targets ─────────────


async def test_three_models_per_row_resolve_after_all_siblings_are_written() -> None:
    headers = ["Batch", "Product", "Product Name", "Site", "City"]
    rows = [["B1", "P-1", "Alpha", "S-9", "Madrid"]]
    mapping = _mapping(
        ("Batch", "batch_no", "primary"),
        ("Product", "product_code", "primary"),
        ("Product", "product_code", "Product"),
        ("Product Name", "name", "Product"),
        ("Site", "site_id", "primary"),
        ("Site", "site_id", "Site"),
        ("City", "city", "Site"),
    )
    (session,) = await _write_table(Batch, [Product, Site], mapping, headers, rows)
    (batch_uid,) = session.uid_of("Batch")
    (product_uid,) = session.uid_of("Product")
    (site_uid,) = session.uid_of("Site")
    assert site_uid == make_instance_uid("Site", {"site_id": "s-9"}, TENANT)
    assert session.links("OF_PRODUCT") == [(batch_uid, product_uid, "sibling")]
    assert session.links("MADE_AT") == [(batch_uid, site_uid, "sibling")]
    assert session.shells() == []
    # Relationships are written after every instance (deferred resolution).
    first_link = next(i for i, (q, _) in enumerate(session.calls) if "r.resolution" in q)
    last_instance = max(i for i, (q, _) in enumerate(session.calls) if "(child:ModelInstance" in q)
    assert first_link > last_instance


# ── 3. Several candidates of the target class ──────────────────────────────


def _formula(*codes: str) -> BaseModel:
    composite_cls = compose_extraction_schema(Formula, [ComponentNote])
    return composite_cls.model_construct(
        formula=Formula.model_construct(
            name="F", components=[Component.model_construct(code=c) for c in codes]
        ),
        component_note=ComponentNote.model_construct(component_code="C2"),
    )


async def _write_composite(composite: BaseModel) -> FakeSession:
    driver = FakeDriver()
    await graph_mapper.write_extraction_subgraph(
        driver=driver,
        node_full_id=f"{TENANT}::doc/v1/n1",
        composite_instance=composite,
        primary_model_class="Formula",
        complementary_model_classes=["ComponentNote"],
        document_name="doc",
        extraction_uid="er",
    )
    return driver.session_obj


async def test_several_candidates_pick_the_one_whose_join_value_matches() -> None:
    session = await _write_composite(_formula("C1", "C2", "C3"))
    (note_uid,) = session.uid_of("ComponentNote")
    components = session.uid_of("Component")
    # Components are written in list order; C2 is the second.
    assert session.links("NOTE_ON") == [(note_uid, components[1], "sibling")]
    assert session.shells() == []


async def test_several_equally_matching_candidates_fall_back_to_the_shell(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=graph_mapper.log.name):
        session = await _write_composite(_formula("C2", "C2"))
    shell_uid = make_instance_uid("Component", {"code": "c2"}, TENANT)
    (note_uid,) = session.uid_of("ComponentNote")
    assert session.links("NOTE_ON") == [(note_uid, shell_uid, "key")]
    assert [s["tgt_uid"] for s in session.shells()] == [shell_uid]
    assert any("match" in r.message and "equally" in r.message for r in caplog.records)


async def test_several_non_matching_candidates_fall_back_to_the_shell() -> None:
    session = await _write_composite(_formula("C1", "C3"))
    shell_uid = make_instance_uid("Component", {"code": "c2"}, TENANT)
    assert [s["tgt_uid"] for s in session.shells()] == [shell_uid]


# ── 4. No sibling / cross-row reference: shell, later filled ───────────────


async def test_cross_row_reference_creates_a_shell_the_target_row_fills() -> None:
    headers = ["Part", "Parent"]
    rows = [["P-2", "P-1"], ["P-1", ""]]  # row 1 references row 2's part
    mapping = _mapping(("Part", "part_id", "primary"), ("Parent", "parent_id", "primary"))
    child_row, parent_row = await _write_table(Part, [], mapping, headers, rows)

    # Row 1: its only Part sibling is itself (excluded) → key-addressed shell.
    (shell,) = child_row.shells()
    expected = make_instance_uid("Part", {"part_id": "p-1"}, TENANT)
    assert shell["tgt_uid"] == expected
    (child_uid,) = child_row.uid_of("Part")
    assert child_row.links("CHILD_OF") == [(child_uid, expected, "key")]

    # Row 2 writes the real P-1 with MERGE on the very same uid: the shell fills.
    (parent_uid,) = parent_row.uid_of("Part")
    assert parent_uid == expected
    merge = next(q for q, p in parent_row.calls if p.get("child_uid") == parent_uid)
    assert "MERGE (child:ModelInstance {uid: $child_uid})" in merge
    assert parent_row.links() == []  # no parent_id → no relationship


async def test_contradicting_keyed_sibling_keeps_the_cross_document_reference() -> None:
    headers = ["Batch", "Product", "Site", "City"]
    rows = [["B1", "P-1", "S-9", "Madrid"]]
    mapping = _mapping(
        ("Batch", "batch_no", "primary"),
        ("Site", "site_id", "primary"),
        ("City", "city", "Site"),
    )
    # The row's Site is a keyed record "P-1"; the Batch references site "S-9":
    # a different record, so the reference stays key-addressed.
    mapping.mappings.append(
        ColumnFieldMapping(column_name="Product", model_field_name="site_id", target_model="Site")
    )
    (session,) = await _write_table(Batch, [Site], mapping, headers, rows)
    shell_uid = make_instance_uid("Site", {"site_id": "s-9"}, TENANT)
    (batch_uid,) = session.uid_of("Batch")
    assert session.links("MADE_AT") == [(batch_uid, shell_uid, "key")]


# ── Supplementary fields on the primary instance (opt-in) ──────────────────


SUPP = [
    {"field_name": "marketed", "field_type": "str", "description": "", "required": False},
    {"field_name": "region", "field_type": "str", "description": "", "required": False},
]


async def _supp_session(on_primary: bool) -> FakeSession:
    headers = ["ID", "Product", "Marketed", "Region"]
    mapping = _mapping(
        ("ID", "record_id", "primary"),
        ("Product", "product_name", "primary"),
        ("Marketed", "marketed", "supplementary"),
        ("Region", "region", "supplementary"),
    )
    (session,) = await _write_table(
        MarketingRecord,
        [],
        mapping,
        headers,
        [["1", "Alpha", "Yes", "EU"]],
        supplementary=SUPP,
        supplementary_on_primary=on_primary,
    )
    return session


async def test_supplementary_fields_stay_on_the_extraction_result_by_default() -> None:
    session = await _supp_session(on_primary=False)
    er_set = next(p for q, p in session.calls if "MATCH (parent:ExtractionResult" in q and "SET" in q)
    assert er_set["marketed"] == "Yes" and er_set["region"] == "EU"
    assert not any("SET mi += $props" in q for q, _ in session.calls)


async def test_supplementary_fields_are_copied_to_the_primary_instance_when_enabled() -> None:
    session = await _supp_session(on_primary=True)
    (record_uid,) = session.uid_of("MarketingRecord")
    (copy,) = [p for q, p in session.calls if "SET mi += $props" in q]
    assert copy == {"uid": record_uid, "props": {"marketed": "Yes", "region": "EU"}}
    # Still on the ExtractionResult too.
    er_set = next(p for q, p in session.calls if "MATCH (parent:ExtractionResult" in q and "SET" in q)
    assert er_set["marketed"] == "Yes"


async def test_write_row_batch_passes_supplementary_fields_only_when_configured(
    monkeypatch, _fake_config
) -> None:
    captured: list[dict] = []

    async def fake_write(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(graph_mapper, "write_extraction_subgraph", fake_write)

    class _Tx:
        async def run(self, *a, **k):
            return None

        async def commit(self):
            return None

        async def rollback(self):
            return None

    class _S:
        async def begin_transaction(self):
            return _Tx()

    class _D:
        @asynccontextmanager
        async def _s(self):
            yield _S()

        def session(self, database=None):
            return self._s()

    decision = SimpleNamespace(
        matched_model_class="MarketingRecord",
        supplementary_fields=[SimpleNamespace(field_name="marketed")],
    )
    composite = compose_extraction_schema(MarketingRecord, [], SUPP).model_construct(
        marketing_record=MarketingRecord.model_construct(record_id="1")
    )
    for enabled, expected in ((False, []), (True, ["marketed"])):
        _fake_config.tabular_supplementary_on_primary = enabled
        captured.clear()
        await neo4j_ops._write_row_batch(
            driver=_D(),
            table_composite_id="t",
            headers=["ID"],
            rows_batch=[["1"]],
            batch_start_index=0,
            decision_uid="md",
            decision=decision,
            mapping=_mapping(("ID", "record_id", "primary")),
            primary_cls=MarketingRecord,
            composite_cls=None,
            primary_field_name="marketing_record",
            comp_class_names=[],
            comp_cls_map={},
            document_name="doc",
            composite_results=[(0, ["1"], composite, "er")],
            row_indices=[0],
        )
        assert list(captured[0]["supplementary_fields"]) == expected
        assert captured[0]["primary_field_name"] == "marketing_record"
