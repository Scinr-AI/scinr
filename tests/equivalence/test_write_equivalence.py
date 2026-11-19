"""
tests/equivalence/test_write_equivalence.py -- The SQL store writes the same graph as Neo4j.

Each golden scenario (``tests/golden/_scenarios.py``, ``s1`` to ``s9``) runs twice through the
public API with the scripted LLM: on Neo4j + MongoDB and on a SQL engine with graph, storage
and freeze in SQL. The neutral dumps (nodes, relationships and the scenario results, with
the volatile values normalized the same way) must be identical; the Cypher trace is not
compared. After the SQL run the invariants of ``tests/sql/_invariants.py`` must hold.

No scenario reads the Cypher trace or the Neo4j driver directly, so none is Neo4j-only.

Run (both container sets up)::

    SCINR_GOLDEN_ALLOW_WIPE=1 PYTHONHASHSEED=0 SCINR_SQL_ALLOW_WIPE=1 \\
        .venv/bin/python -m pytest -m integration tests/equivalence
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.contract._helpers import diff, same
from tests.golden import _runner
from tests.golden._scenarios import SCENARIOS

pytestmark = pytest.mark.integration

#: Neo4j result of each scenario, computed once per session (both SQL engines compare to it).
_NEO4J: dict[str, dict[str, Any]] = {}


async def _neo4j_graph(scenario: str) -> dict[str, Any]:
    if scenario not in _NEO4J:
        _, graph = await _runner.run_scenario(scenario, with_trace=False)
        _NEO4J[scenario] = graph
    return _NEO4J[scenario]


def _extras_diff(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    exp, act = expected.get("extras") or {}, actual.get("extras") or {}
    keys = sorted(k for k in set(exp) | set(act) if exp.get(k) != act.get(k))
    return "\n".join(
        f"  {k}: neo4j={exp.get(k)!r}\n  {' ' * len(k)}  sql={act.get(k)!r}" for k in keys
    )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_scenario_writes_the_same_graph(scenario, neo4j_available, sql_engine):
    from tests.sql._invariants import check_all

    expected = await _neo4j_graph(scenario)
    _, actual = await _runner.run_scenario(
        scenario, settings=sql_engine.settings, dump=sql_engine.dump, with_trace=False
    )
    graph_part = {k: v for k, v in actual.items() if k != "extras"}
    expected_part = {k: v for k, v in expected.items() if k != "extras"}
    assert same(expected_part, graph_part), diff(expected_part, graph_part)
    assert (expected.get("extras") or {}) == (actual.get("extras") or {}), _extras_diff(
        expected, actual
    )
    settings = sql_engine.settings
    check_all(
        settings["sql_url"], settings["sql_schema"] if sql_engine.name == "postgres" else None
    )
