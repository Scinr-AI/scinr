"""
tests/unit/test_relink.py — Repair of instance_relationships that point at
dead shells although a same-row sibling matches
(scinr.newton.ingest.relink). The planning is pure; the execution runs
against a small in-memory fake of the sync driver that applies the moves, so
idempotency is observable. No Neo4j.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest

from scinr.newton.ingest import relink
from scinr.newton.ingest.relink import Declaration, parse_declarations, plan_moves, run_relink

TENANT = "a705513c"

DECL = Declaration(
    src_class="RegulatoryProductSpecification",
    rel_type="SPECIFICATION_FOR",
    tgt_class="MarketingAuthorisationRecord",
    via_field="portfolio_record_id",
    join_via={"portfolio_record_id": "portfolio_record_id"},
)


def _row(i: int, shell_value: str, siblings: list[dict], via_value: Any = None, er: str | None = None):
    return {
        "rel_id": f"r{i}",
        "src_uid": f"spec{i}",
        "via_value": via_value if via_value is not None else shell_value,
        "shell_id": f"shell{i}",
        "shell_degree": 1,
        "shell_values": [shell_value],
        "er_uid": er or f"er{i}",
        "siblings": siblings,
    }


def _ma(i: int, value: str | None, keyed: bool = False) -> dict:
    return {"uid": f"ma{i}", "values": [value], "keyed": keyed}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_parse_declarations_reads_catalog_json() -> None:
    (decl,) = parse_declarations(
        [
            {
                "src_class": "S",
                "rel_type": "R",
                "tgt_class": "T",
                "via_field": "f",
                "join_via": json.dumps({"f": "g"}),
            }
        ]
    )
    assert decl.join_via == {"f": "g"} and decl.remote_fields == ["g"]


def test_plan_relinks_shells_to_the_row_sibling() -> None:
    rows = [
        _row(1, "312947", [_ma(1, "312947; CANTABNEW-0081")]),  # compatible
        _row(2, "zzz", [_ma(2, "400001; CANTABNEW-0099")]),  # sole key-less sibling
        _row(3, "500002; cantabnew-0100", [_ma(3, "500002; CANTABNEW-0100")]),  # exact
    ]
    moves, counts = plan_moves(DECL, rows)
    assert counts == {"relinked": 3}
    assert [(m.rel_id, m.src_uid, m.sib_uid) for m in moves] == [
        ("r1", "spec1", "ma1"),
        ("r2", "spec2", "ma2"),
        ("r3", "spec3", "ma3"),
    ]
    assert [m.outcome for m in moves] == ["compatible", "sole", "exact"]


def test_plan_keeps_ambiguous_and_unmatched() -> None:
    rows = [
        _row(1, "x", [_ma(1, "x"), {"uid": "ma1b", "values": ["x"], "keyed": False}]),
        _row(2, "x", [_ma(2, "y", keyed=True)]),  # keyed, contradicting: cross-document
        _row(3, "x", []),  # nothing of the target class in the row
    ]
    moves, counts = plan_moves(DECL, rows)
    assert moves == []
    assert counts == {"ambiguous": 1, "unmatched": 2}


def test_plan_source_shared_by_several_extractions_must_agree() -> None:
    agree = [_row(1, "x", [_ma(1, "x")], er="e1"), _row(1, "x", [_ma(1, "x")], er="e2")]
    disagree = [_row(2, "x", [_ma(2, "x")], er="e1"), _row(2, "x", [_ma(9, "x")], er="e2")]
    moves, counts = plan_moves(DECL, agree + disagree)
    assert [m.rel_id for m in moves] == ["r1"]
    assert counts == {"relinked": 1, "ambiguous": 1}


def test_plan_fan_out_does_not_use_the_sole_sibling_rule() -> None:
    rows = [_row(1, "a", [_ma(1, "zzz")], via_value=["a", "b"])]
    _, counts = plan_moves(DECL, rows)
    assert counts == {"unmatched": 1}


# ---------------------------------------------------------------------------
# Execution against a fake graph
# ---------------------------------------------------------------------------


class _Record(dict):
    def data(self) -> dict:
        return dict(self)


class _Result(list):
    def single(self):
        return self[0] if self else None


class FakeGraph:
    """Just enough state for run_relink: shell relationships (removed when
    moved), shells (deleted when isolated) and the queries issued."""

    def __init__(self, rows: list[dict], declarations: list[Declaration] = (DECL,)) -> None:
        self.rows = rows
        self.declarations = list(declarations)
        self.moved: list[dict] = []
        self.deleted_shells: set[str] = set()
        self.queries: list[tuple[str, dict]] = []

    def _live_rows(self) -> list[dict]:
        gone = {m["rel_id"] for m in self.moved}
        return [r for r in self.rows if r["rel_id"] not in gone]

    def _isolated_shells(self) -> set[str]:
        live = {r["shell_id"] for r in self._live_rows()}
        all_shells = {r["shell_id"] for r in self.rows}
        return all_shells - live - self.deleted_shells

    def run(self, query: str, **params) -> _Result:
        self.queries.append((query, params))
        if query == relink.DECLARATIONS_QUERY:
            return _Result(
                _Record(
                    src_class=d.src_class,
                    rel_type=d.rel_type,
                    tgt_class=d.tgt_class,
                    via_field=d.via_field,
                    join_via=json.dumps(dict(d.join_via)),
                )
                for d in self.declarations
            )
        if query == relink.INSTANCE_KEYS_QUERY:
            return _Result([])
        if "AS shell_degree" in query:
            assert params["tenant_id"] == TENANT
            return _Result(_Record(r) for r in self._live_rows())
        if "DELETE r_old" in query:
            assert params["tenant_id"] == TENANT
            self.moved.extend(params["rows"])
            return _Result([_Record(moved=len(params["rows"]))])
        if query == relink.COUNT_ISOLATED_SHELLS_QUERY:
            return _Result([_Record(n=0)])
        if query == relink.DELETE_ISOLATED_SHELLS_QUERY:
            assert params["tenant_id"] == TENANT
            isolated = self._isolated_shells()
            self.deleted_shells |= isolated
            return _Result([_Record(deleted=len(isolated))])
        raise AssertionError(f"unexpected query: {query}")


class FakeSyncDriver:
    def __init__(self, graph: FakeGraph) -> None:
        self.graph = graph

    @contextmanager
    def session(self, database: str | None = None):
        graph = self.graph

        class _Session:
            def run(self, query, **params):
                return graph.run(query, **params)

            def execute_write(self, fn):
                return fn(self)

        yield _Session()


def _production_rows() -> list[dict]:
    """195 rows with an MA and a Spec; 193 Specs carry a reference (2 rows have
    an empty Spec ID, so no relationship was written). The MA join value was
    built from two columns ("<code>; CANTABNEW-<n>"); the Spec references one
    of them."""
    rows = []
    for i in range(193):
        code, ident = f"{300000 + i}", f"CANTABNEW-{i:04d}"
        ref = code if i % 2 else ident
        rows.append(_row(i, ref.lower(), [_ma(i, f"{code}; {ident}")]))
    return rows


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(relink, "with_neo4j_retry_sync", lambda fn: fn())


def test_run_relink_production_shape_then_idempotent() -> None:
    graph = FakeGraph(_production_rows())
    driver = FakeSyncDriver(graph)

    first = run_relink(driver, tenant_id=TENANT, database="neo4j")
    assert first["relinked"] == 193
    assert first["ambiguous"] == 0
    assert first["unmatched"] == 0
    assert first["shell_relationships"] == 193
    assert first["shells_deleted"] == 193
    assert first["by_relationship"] == {
        "RegulatoryProductSpecification-[SPECIFICATION_FOR]->MarketingAuthorisationRecord": {
            "relinked": 193,
            "shell_relationships": 193,
        }
    }
    assert {(m["src_uid"], m["sib_uid"]) for m in graph.moved} == {
        (f"spec{i}", f"ma{i}") for i in range(193)
    }

    second = run_relink(driver, tenant_id=TENANT, database="neo4j")
    assert (second["relinked"], second["shell_relationships"], second["shells_deleted"]) == (0, 0, 0)


def test_run_relink_moves_in_bounded_batches(monkeypatch) -> None:
    monkeypatch.setattr(relink, "DELETE_BATCH_ROWS", 50)
    graph = FakeGraph(_production_rows())
    run_relink(FakeSyncDriver(graph), tenant_id=TENANT, database="neo4j")
    batches = [len(p["rows"]) for q, p in graph.queries if "DELETE r_old" in q]
    assert batches == [50, 50, 50, 43]


def test_run_relink_dry_run_writes_nothing() -> None:
    graph = FakeGraph(_production_rows())
    result = run_relink(FakeSyncDriver(graph), tenant_id=TENANT, database="neo4j", dry_run=True)
    assert result["dry_run"] is True
    assert result["relinked"] == 193 and result["shells_deleted"] == 193
    assert graph.moved == [] and graph.deleted_shells == set()
    assert not any("DELETE" in q for q, _ in graph.queries)


def test_run_relink_skips_unsafe_relationship_types() -> None:
    bad = Declaration("S", "BAD`) DETACH DELETE (n", "T", "f", {"f": "g"})
    graph = FakeGraph([], declarations=[bad])
    result = run_relink(FakeSyncDriver(graph), tenant_id=TENANT, database="neo4j")
    assert result["relinked"] == 0
    assert not any("BAD`" in q for q, _ in graph.queries)


def test_shell_query_is_tenant_scoped_and_excludes_instance_relationship_hops() -> None:
    q = relink._shell_edges_query("SPECIFICATION_FOR")
    assert "src.tenant_id = $tenant_id" in q and "tgt.tenant_id = $tenant_id" in q
    assert "er.tenant_id = $tenant_id" in q and "sib.tenant_id = $tenant_id" in q
    assert "none(x IN rels WHERE type(x) IN $irel_types)" in q
    move = relink._move_query("SPECIFICATION_FOR")
    assert "src.tenant_id = $tenant_id" in move and "sib.tenant_id = $tenant_id" in move
