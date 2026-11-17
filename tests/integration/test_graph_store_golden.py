"""
tests/integration/test_graph_store_golden.py — Golden harness of the graph writes.

Runs the scenarios of ``tests/golden/_scenarios.py`` against real Neo4j and MongoDB and
compares, with the artifacts stored in ``tests/golden/<scenario>/``:

1. the final graph (and the scenario results) -- identical;
2. the multiset of units of work sent to Neo4j (statements, parameters and transaction
   boundaries) -- identical;
3. the order of the units issued through the sync driver -- identical.

It is the safety net of the ``GraphStore`` refactor (plans/graph-store-interface-plan.md):
moving Cypher around must not change a single statement. Do NOT regenerate the artifacts
to make it pass (``scripts/golden_regen.py`` is for establishing a baseline only).

THE HARNESS WIPES its databases before every scenario, so it only runs against the
disposable containers (never ``neo4j-test`` / ``mongodb-test`` or any database with data)::

    docker run -d --rm --name scinr-golden-neo4j -p 7698:7687 \\
        -e NEO4J_AUTH=neo4j/scratchpass1 neo4j:2026.08.1-community
    docker run -d --rm --name scinr-golden-mongo -p 27028:27017 mongo:7.0

    SCINR_GOLDEN_ALLOW_WIPE=1 PYTHONHASHSEED=0 \\
        .venv/bin/python -m pytest -m integration tests/integration/test_graph_store_golden.py

It is skipped unless ``SCINR_GOLDEN_ALLOW_WIPE=1``, ``PYTHONHASHSEED=0`` and the servers are
those on Neo4j 7698 / MongoDB 27028.
"""

from __future__ import annotations

import pytest

from tests.golden import _runner
from tests.golden._scenarios import SCENARIOS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_scenario_matches_golden(scenario: str) -> None:
    try:
        _runner.guard()
        trace, graph = await _runner.run_scenario(scenario)
    except _runner.GoldenUnavailable as exc:
        pytest.skip(str(exc))

    if not (_runner.GOLDEN_DIR / scenario / "trace.jsonl").exists():
        pytest.fail("no golden artifacts: run scripts/golden_regen.py to establish the baseline")
    expected_trace, expected_graph = _runner.read_artifacts(scenario)

    graph_diff = _runner.diff_graph(expected_graph, graph)
    assert graph_diff is None, graph_diff
    trace_diff = _runner.diff_trace(expected_trace, trace)
    assert trace_diff is None, trace_diff
