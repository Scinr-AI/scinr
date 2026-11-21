"""
tests/equivalence/test_read_equivalence.py -- The SQL navigator reads what Neo4j reads.

Each golden scenario that leaves navigable data runs twice through the public API, as in
``test_write_equivalence.py``: on Neo4j + MongoDB and on a SQL engine. After each run the
suite opens a navigator on that engine and makes **every call** of ``_calls.build_calls``
(every method of ``GraphNavigator`` with real anchors and the scope / filter variants). The
Pydantic results (``raw`` removed), or the type of the exception, must be identical;
``search_info_units`` is compared by the set of ``uid`` (the score and the order of a
full-text search are the engine's); ``execute_raw`` has tests of its own
(``test_navigator_sql.py``).

Run (both container sets up)::

    SCINR_GOLDEN_ALLOW_WIPE=1 PYTHONHASHSEED=0 SCINR_SQL_ALLOW_WIPE=1 \\
        .venv/bin/python -m pytest -m integration tests/equivalence/test_read_equivalence.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from scinr.newton.navigation import GraphNavigator, graph_navigator
from tests.equivalence._calls import LIFECYCLE, NOT_COMPARED, Call, build_calls, covered_methods
from tests.golden import _runner
from tests.golden._dump import _normalize, collect_object_ids, sort_lists
from tests.golden._scenarios import SCENARIOS

pytestmark = pytest.mark.integration

#: Scenarios that leave navigable data (documents, annotation, extraction, tabular, restore,
#: several tenants).
READ_SCENARIOS = (
    "s1_ingest_versions",
    "s3_annotation",
    "s4_extraction",
    "s5_tabular",
    "s7_freeze_restore",
    "s8_tenants",
)

#: ``method -> (why, which calls)``: calls that cannot give the same result because of a real
#: difference between the engines. Only with the owner's agreement.
KNOWN_DIFFERENCES: dict[str, tuple[str, Callable[[Call], bool]]] = {
    "get_graph_summary": (
        "Neo4j fails with a CypherSyntaxError when the summary is filtered by user or job: "
        "the alias ``x`` of the scoped count is also the variable of the ``any(x IN ...)`` "
        "predicate of the array provenance. The SQL backend answers.",
        lambda call: "created_by_user_id" in call.kwargs or "job_id" in call.kwargs,
    ),
    "search_nodes": (
        "A page that skips results (``skip`` > 0) depends on the engine's own full-text ranking of "
        "the info units (Lucene in Neo4j, the SQL text search elsewhere): each engine truncates its "
        "candidates to ``skip + limit`` in its own order, so the same page can differ. The calls "
        "without ``skip`` are equal.",
        lambda call: bool(call.kwargs.get("skip")),
    ),
}

#: Methods whose result is compared as a set of ``uid`` (the engine's own full-text ranking).
SET_COMPARED = frozenset({"search_info_units"})

#: ``method -> why``: methods whose Cypher has no total ``ORDER BY`` (no ``ORDER BY``, or ties
#: it does not break), so Neo4j returns the rows in an arbitrary order that changes from run to
#: run. They are compared with every list sorted; the SQL ones do have a total order.
ORDER_UNSPECIFIED: dict[str, str] = {
    "get_unannotated_nodes": "ORDER BY appearance_order only (ties)",
    "get_structure_subtree": "children come in the order of the path rows",
    "get_document_model_decisions": "ORDER BY appearance_order only (ties)",
    "get_document_extraction_results": "ORDER BY appearance_order only (ties)",
    "get_model_instance_subtree": "children come in the order of the path rows",
    "get_model_instance_relationships": "UNION without ORDER BY",
    "get_entity_relationships": "UNION without ORDER BY",
    "get_triples": "no ORDER BY",
    "list_catalog_models": "collect() of fields and themes has no order",
    "get_catalog_graph": "collect() of fields and themes has no order",
    "get_catalog_graph_in_use": "collect() of fields and themes has no order",
    "neighbors": "RETURN DISTINCT without ORDER BY",
    "subgraph": "collect(DISTINCT) without ORDER BY",
    "get_model_instances_by_class": "ORDER BY <order_by> only (ties)",
    "list_entity_labels": "ORDER BY count only (ties)",
    "list_node_roles": "ORDER BY count only (ties)",
    "list_model_instance_relationship_types": "ORDER BY count only (ties)",
    "list_root_documents": "ORDER BY path, version (the same path in several tenants ties)",
    "get_documents": "ORDER BY path, version (the same path in several tenants ties)",
    "summarize_sources": "ordered by date then path (the same path in several tenants ties)",
}

#: Keys of the scope arguments (a call without any of them is not scoped).
_SCOPE_KEYS = ("tenant_id", "include_public", "created_by_user_id", "job_id")

Outcome = tuple[str, Any]


def _plain(value: Any) -> Any:
    """JSON-able copy of a result: models dumped without their ``raw``, recursively."""
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k != "raw"}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


async def _make_call(nav: GraphNavigator, call: Call) -> Outcome:
    try:
        return "ok", _plain(await getattr(nav, call.method)(**call.kwargs))
    except Exception as exc:  # noqa: BLE001 — the type of the error is part of the contract
        return "error", type(exc).__name__


async def _outcomes(calls: list[Call]) -> list[Outcome]:
    """Make every call on the navigator of the engine configured right now."""
    async with graph_navigator() as nav:
        results = [await _make_call(nav, call) for call in calls]
    pairs = [[kind, value] for kind, value in results]
    oids = collect_object_ids(pairs)
    return [(kind, _normalize(value, oids)) for kind, value in pairs]


def _is_known_difference(call: Call) -> bool:
    known = KNOWN_DIFFERENCES.get(call.method)
    return known is not None and known[1](call)


def _comparable(call: Call, outcome: Outcome) -> Any:
    kind, value = outcome
    if kind != "ok":
        return outcome
    if call.method in SET_COMPARED:
        return ("ok", sorted(item["uid"] for item in value))
    if call.method == "subgraph" and not any(k in call.kwargs for k in _SCOPE_KEYS):
        # Unscoped Neo4j subgraph: with APOC the relationships carry ``elementId`` ends
        # (the engine's internal ids) and without it only those of the first node come back
        # (see the report); the nodes are the common part.
        value = {"nodes": value["nodes"]}
    if call.method == "get_graph_summary" and not any(k in call.kwargs for k in _SCOPE_KEYS):
        # Without APOC Neo4j reports no relationship types (``relTypeCount`` is 0 in its
        # fallback); the SQL backend counts them.
        value = {k: v for k, v in value.items() if k != "total_relationship_types"}
    if call.method in ORDER_UNSPECIFIED:
        value = sort_lists(value)
    return (kind, value)


#: Per scenario: the calls and the outcomes of the Neo4j run, computed once.
_NEO4J: dict[str, tuple[list[Call], list[Outcome]]] = {}


def _drop_unused_labels(calls: list[Call], outcomes: list[Outcome]) -> list[Outcome]:
    """Neo4j keeps a label in ``db.labels()`` and in the counts of ``apoc.meta.stats`` after
    the last node that had it is gone (the harness wipes the graph between scenarios but
    not the label tokens), so the labels without nodes are left out of the Neo4j side: the
    SQL backend reports the labels in use."""
    summary = next(
        (
            out
            for call, out in zip(calls, outcomes, strict=True)
            if call.method == "get_graph_summary" and not call.kwargs
        ),
        None,
    )
    if summary is None or summary[0] != "ok":
        return outcomes
    unused = {k for k, v in summary[1]["node_counts"].items() if not v}
    fixed: list[Outcome] = []
    for call, (kind, value) in zip(calls, outcomes, strict=True):
        if kind == "ok" and call.method == "list_node_labels":
            value = [label for label in value if label not in unused]
        elif kind == "ok" and call.method == "get_graph_summary" and not call.kwargs:
            value = {**value, "node_counts": {k: v for k, v in value["node_counts"].items() if v}}
        fixed.append((kind, value))
    return fixed


async def _neo4j_side(scenario: str) -> tuple[list[Call], list[Outcome]]:
    if scenario not in _NEO4J:
        _, graph = await _runner.run_scenario(scenario, with_trace=False)
        calls = build_calls(graph)
        _NEO4J[scenario] = (calls, _drop_unused_labels(calls, await _outcomes(calls)))
    return _NEO4J[scenario]


def _first_difference(want: Any, got: Any, path: str = "$") -> str:
    """Where two JSON-able values first differ, and what each has there."""
    if isinstance(want, dict) and isinstance(got, dict):
        for key in sorted(set(want) | set(got)):
            if key not in want or key not in got or want[key] != got[key]:
                if key in want and key in got:
                    return _first_difference(want[key], got[key], f"{path}.{key}")
                return f"{path}.{key}: neo4j={want.get(key, '<absent>')!r} sql={got.get(key, '<absent>')!r}"
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(want) != len(got):
            return f"{path}: neo4j has {len(want)} items, sql has {len(got)}"
        for index, (a, b) in enumerate(zip(want, got, strict=True)):
            if a != b:
                return _first_difference(a, b, f"{path}[{index}]")
    return f"{path}: neo4j={want!r} sql={got!r}"


def _describe(call: Call, neo4j: Any, sql: Any) -> str:
    args = {k: (v.model_dump() if isinstance(v, BaseModel) else v) for k, v in call.kwargs.items()}
    where = _first_difference(_comparable(call, neo4j), _comparable(call, sql))
    return f"{call.method}({json.dumps(args, default=str, sort_keys=True)})\n  {where[:700]}"


def test_every_method_has_calls() -> None:
    """``build_calls`` covers each method of ``GraphNavigator`` (also on an empty graph)."""
    expected = set(GraphNavigator.__abstractmethods__) - LIFECYCLE - NOT_COMPARED
    covered = covered_methods(build_calls({"nodes": [], "relationships": []}))
    assert expected - covered == set(), f"methods without calls: {sorted(expected - covered)}"
    assert covered - expected == set(), f"calls to unknown methods: {sorted(covered - expected)}"


def test_scenario_list_is_known() -> None:
    assert set(READ_SCENARIOS) <= set(SCENARIOS)


@pytest.mark.parametrize("scenario", READ_SCENARIOS)
async def test_navigation_matches_neo4j(scenario, neo4j_available, sql_engine):
    calls, expected = await _neo4j_side(scenario)
    _, graph = await _runner.run_scenario(
        scenario, settings=sql_engine.settings, dump=sql_engine.dump, with_trace=False
    )
    assert [c.method for c in build_calls(graph)] == [c.method for c in calls], (
        "the engines left different graphs: run the write-equivalence suite first"
    )
    actual = await _outcomes(calls)
    failures: dict[str, list[str]] = {}
    for call, want, got in zip(calls, expected, actual, strict=True):
        if (
            not _is_known_difference(call)
            and call.method not in NOT_COMPARED
            and _comparable(call, want) != _comparable(call, got)
        ):
            failures.setdefault(call.method, []).append(_describe(call, want, got))
    total = sum(len(v) for v in failures.values())
    summary = "\n".join(f"{method}: {len(items)} calls" for method, items in failures.items())
    examples = "\n".join(items[0] for items in failures.values())
    assert not failures, f"{total} of {len(calls)} calls differ:\n{summary}\n\n{examples}"
