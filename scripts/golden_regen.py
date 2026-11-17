"""Regenerate the golden artifacts of the graph-write harness (tests/golden/).

Usage (disposable containers only, see plans/graph-store-interface-plan.md §3.2)::

    SCINR_GOLDEN_ALLOW_WIPE=1 PYTHONHASHSEED=0 .venv/bin/python scripts/golden_regen.py [scenario ...]

Writes ``tests/golden/<scenario>/trace.jsonl`` and ``graph.json`` for every scenario (or the
named ones) and ``tests/golden/BASELINE`` with the hash of the commit the code was at.

This is only for establishing or deliberately changing the baseline. A work package of the
refactor must pass ``tests/integration/test_graph_store_golden.py`` WITHOUT regenerating:
needing to regenerate means the behaviour changed.

The graph and MongoDB database of the harness are WIPED before every scenario; the script
refuses to run unless ``SCINR_GOLDEN_ALLOW_WIPE=1`` and the servers are the containers on
Neo4j 7698 / MongoDB 27028. By default it also refuses to run with uncommitted changes under
``src/`` (the baseline must describe a commit); pass ``--allow-dirty`` to override.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.golden import _runner  # noqa: E402
from tests.golden._scenarios import SCENARIOS  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="*", choices=[*SCENARIOS, []], default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    try:
        _runner.guard()
    except _runner.GoldenUnavailable as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    if _git("status", "--porcelain", "--", "src") and not args.allow_dirty:
        print("refusing to run: uncommitted changes under src/ (--allow-dirty)", file=sys.stderr)
        return 2

    names = args.scenarios or list(SCENARIOS)
    for name in names:
        trace, graph = asyncio.run(_runner.run_scenario(name))
        _runner.write_artifacts(name, trace, graph)
        print(f"{name}: {len(trace)} trace entries, {len(graph['nodes'])} nodes")
    if not args.scenarios:
        (ROOT / "tests" / "golden" / "BASELINE").write_text(
            _git("rev-parse", "HEAD") + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
