"""Acceptance checks over the ``results.json`` files of ``run.py`` (WP7.2, rule of decision P7).

    # postgres against neo4j at the same size (run it once for 1M and once for 10M)
    python scripts/sql_bench/compare.py accept results/neo4j-1m.json results/postgres-1m.json

    # growth of one engine from 1M to 10M, per operation
    python scripts/sql_bench/compare.py scale results/postgres-1m.json results/postgres-10m.json

``accept BASELINE CANDIDATE`` (baseline = neo4j, candidate = postgres) fails (exit 1) when

* any navigation operation (p50 or p95) of the candidate is more than 3x the baseline's, or
* the peak memory of the ingest is more than 20 % above the baseline's.

The other operations (ingest rate, delete, GC, freeze, restore) are printed with their ratio but
are not part of the rule. ``scale SMALL LARGE`` prints the large/small ratio of every metric,
so the gap between engines can be compared across sizes; it never fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MAX_NAVIGATION_RATIO = 3.0
MAX_INGEST_MEMORY_RATIO = 1.2
NAVIGATION_OPS = (
    "get_structure_tree",
    "get_info_units",
    "search_info_units",
    "get_model_instance_by_key",
    "subgraph",
    "shortest_path",
)
NAVIGATION_METRICS = ("p50_ms", "p95_ms")


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def ratio(base: float, other: float) -> float:
    if base == 0:
        return float("inf") if other else 1.0
    return other / base


def describe(result: dict[str, Any]) -> str:
    meta = result["meta"]
    return f"{meta['backend']} {meta['nodes_actual']:,} nodes ({meta['memory_mode']} memory)"


def accept(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Print the comparison and return the list of broken rules."""
    base, cand = baseline["ops"], candidate["ops"]
    print(f"baseline : {describe(baseline)}")
    print(f"candidate: {describe(candidate)}")
    failures: list[str] = []
    if baseline["meta"]["nodes_actual"] != candidate["meta"]["nodes_actual"]:
        failures.append("the two results are not of the same size")
    if baseline["meta"]["memory_mode"] != candidate["meta"]["memory_mode"]:
        failures.append("the two results measured memory differently")

    print(f"\n{'operation':<28}{'metric':<12}{'baseline':>14}{'candidate':>14}{'ratio':>8}  rule")
    for op in NAVIGATION_OPS:
        for metric in NAVIGATION_METRICS:
            r = ratio(base[op][metric], cand[op][metric])
            ok = r <= MAX_NAVIGATION_RATIO
            print(
                f"{op:<28}{metric:<12}{base[op][metric]:>14.2f}{cand[op][metric]:>14.2f}"
                f"{r:>8.2f}  {'ok' if ok else f'FAIL (> {MAX_NAVIGATION_RATIO:g}x)'}"
            )
            if not ok:
                failures.append(f"{op} {metric} is {r:.2f}x slower")
    r = ratio(base["ingest"]["peak_mib"], cand["ingest"]["peak_mib"])
    ok = r <= MAX_INGEST_MEMORY_RATIO
    print(
        f"{'ingest':<28}{'peak_mib':<12}{base['ingest']['peak_mib']:>14.2f}"
        f"{cand['ingest']['peak_mib']:>14.2f}{r:>8.2f}  "
        f"{'ok' if ok else f'FAIL (> {MAX_INGEST_MEMORY_RATIO:g}x)'}"
    )
    if not ok:
        failures.append(f"ingest peak memory is {r:.2f}x the baseline")

    print("\ninformative (not part of the rule)")
    for op in base:
        if op in NAVIGATION_OPS:
            continue
        for metric, value in base[op].items():
            if metric in ("nodes", "n") or metric not in cand[op]:
                continue
            if op == "ingest" and metric == "peak_mib":
                continue
            r = ratio(value, cand[op][metric])
            print(f"{op:<28}{metric:<12}{value:>14.2f}{cand[op][metric]:>14.2f}{r:>8.2f}")
    return failures


def scale(small: dict[str, Any], large: dict[str, Any]) -> None:
    print(f"small: {describe(small)}\nlarge: {describe(large)}")
    factor = large["meta"]["nodes_actual"] / small["meta"]["nodes_actual"]
    print(f"size factor: {factor:.1f}x (a ratio near it means linear growth)\n")
    print(f"{'operation':<28}{'metric':<12}{'small':>14}{'large':>14}{'large/small':>13}")
    for op, metrics in small["ops"].items():
        for metric, value in metrics.items():
            if metric in ("nodes", "n"):
                continue
            other = large["ops"][op][metric]
            print(f"{op:<28}{metric:<12}{value:>14.2f}{other:>14.2f}{ratio(value, other):>13.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("accept", "scale"):
        cmd = sub.add_parser(name)
        cmd.add_argument("first")
        cmd.add_argument("second")
    args = parser.parse_args()
    if args.command == "scale":
        scale(load(args.first), load(args.second))
        return 0
    failures = accept(load(args.first), load(args.second))
    print()
    if failures:
        print("ACCEPTANCE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ACCEPTANCE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
