"""Peak LIVE memory (tracemalloc; allocator independent) of the graph WRITE paths.

Usage (the databases are WIPED: disposable containers only, see below):

    SCINR_GOLDEN_ALLOW_WIPE=1 PYTHONHASHSEED=0 .venv/bin/python scripts/memory_bench/bench_graph.py make
    ... bench_graph.py run <case>                one case, every size/variant, prints its peaks
    ... bench_graph.py all [--runs 3] [--cases a,b] [--write-baseline]
                                                 every case, each run in its own process; prints the
                                                 table  case | size | baseline | now | delta | ok

Plan: plans/graph-store-interface-plan.md §3.5. The golden harness (tests/golden) proves that the
Cypher does not change; this bench proves that the memory peak of the write paths does not grow
(a method returning a ``list`` where an iterator is consumed today, a sink that accumulates rows, an
export callback that buffers the file, a ``sync_scope`` that retains results, a second driver...).
The read/convert benches (bench_csv, bench_pdf, bench_media, bench_fanout) do not cover these paths.

Setup
-----
Same disposable containers and the same guard as the golden harness (tests/golden/_runner.py):
Neo4j on localhost:7698, MongoDB on localhost:27028, ``SCINR_GOLDEN_ALLOW_WIPE=1``,
``PYTHONHASHSEED=0``. Both are emptied before every measurement. The LLM is simulated in process
(tests/golden/_llm.py) and the golden recorder is NOT active. Only what the library allocates inside
the measured call counts: inputs and setup are prepared before ``tracemalloc.start()``.

Cases (two sizes each, small and ~4x)
-------------------------------------
ingest_doc       run_ingestion of one extraction JSON (explicit transaction of the loader).
tabular_rows     run_tabular_pipeline of a 12-column CSV (batches of 500 rows, model extraction per row).
annotate_extract run_annotation_agent + run_entity_extraction_agent of a document with many nodes,
                 variants: c1 (llm/neo4j concurrency 1) and cdef (library defaults).
freeze_export    freeze_document of the ingest_doc document and of the tabular document.
restore          restore_document of both snapshots, variants c1 / c4 (concurrency).
delete_gc        delete_document of both documents and collect_orphans().

Criteria (per case, size and variant)
-------------------------------------
* peak <= baseline * 1.05 + 2 MiB (median of 3 runs), and
* tabular_rows, freeze_export, restore: peak(large) <= peak(small) * 1.10 (bounded memory). If the
  baseline already violates it, graph_baseline.json says so and the criterion becomes "not worse
  than the baseline".

Deviations from the plan's orientative sizes: tabular is 2 000 / 8 000 rows (plan: 50 000 / 200 000;
the per-row writes of the library take ~15 ms and a 3-run baseline of 200 000 rows is hours). The
memory shape (500-row batches) is the same: 4 vs 16 batches.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.golden import _runner  # noqa: E402
from tests.golden._llm import Script, decision, set_script  # noqa: E402

DIR = Path(tempfile.gettempdir()) / "scinr_bench_graph"
BASELINE = Path(__file__).with_name("graph_baseline.json")
MODELS = Path(__file__).with_name("graph_models")
MIB = 1024 * 1024
TENANT = "bench"
TOLERANCE_FACTOR = 1.05
TOLERANCE_BYTES = 2 * MIB
GROWTH_FACTOR = 1.10
DOC_NAME = "bench_doc.txt"
ANNOT_NAME = "bench_annot.txt"

SIZES = ("small", "large")
# "warm" is a tiny input run (unmeasured) before the sizes, so that the one-time costs of the
# first call in a process (lazy imports, theme registry, schemas...) do not land in "small".
WARM = "warm"
DOC_NODES = {WARM: 200, "small": 5_000, "large": 20_000}
CSV_ROWS = {WARM: 100, "small": 2_000, "large": 8_000}
ANNOTATE_NODES = {WARM: 30, "small": 500, "large": 2_000}
CSV_COLUMNS = 12
CHILDREN_PER_SECTION = 49

# case -> variants measured inside one process (each one with fresh databases).
CASES: dict[str, tuple[str, ...]] = {
    "ingest_doc": ("-",),
    "tabular_rows": ("-",),
    "annotate_extract": ("c1", "cdef"),
    "freeze_export": ("-",),
    "restore": ("c1", "c4"),
    "delete_gc": ("-",),
}
# Cases whose peak must not grow with the input (large <= small + 10 %).
BOUNDED = ("tabular_rows", "freeze_export", "restore")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _node(node_id: str, order: int, children: list[dict], text: str) -> dict:
    return {
        "node_id": node_id,
        "title": f"Section {node_id}",
        "role": "section",
        "appearance_order": order,
        "theme": "default",
        "info_units": [
            {"title": f"{node_id} fact {k}", "order": k, "description": f"{text} {node_id} #{k}"}
            for k in range(2)
        ],
        "children": children,
    }


def _document(name: str, nodes: int) -> dict:
    """A document with *nodes* StructureNodes: sections of 49 children each."""
    text = "Synthetic technical description of the item, with units and conditions. " * 4
    structure: list[dict] = []
    made = 0
    section = 0
    while made < nodes:
        children = []
        for k in range(min(CHILDREN_PER_SECTION, nodes - made - 1)):
            children.append(_node(f"n{made + k + 1:06d}", k + 1, [], text))
        structure.append(_node(f"s{section:04d}", section + 1, children, text))
        made += 1 + len(children)
        section += 1
    return {
        "document_name": name,
        "doc_path": name,
        "raw_file_id": "",
        "document_structure": structure,
    }


def _node_ids(nodes: int) -> list[str]:
    """Ids of the nodes of ``_document(nodes)`` (same loop)."""
    ids: list[str] = []
    made = 0
    section = 0
    while made < nodes:
        count = min(CHILDREN_PER_SECTION, nodes - made - 1)
        ids.append(f"s{section:04d}")
        ids.extend(f"n{made + k + 1:06d}" for k in range(count))
        made += 1 + count
        section += 1
    return ids


def make() -> None:
    DIR.mkdir(exist_ok=True)
    for size in (WARM, *SIZES):
        for prefix, name, nodes in (
            ("doc", DOC_NAME, DOC_NODES[size]),
            ("annot", ANNOT_NAME, ANNOTATE_NODES[size]),
        ):
            folder = DIR / f"{prefix}_{size}"
            folder.mkdir(exist_ok=True)
            (folder / f"extract-{name}.json").write_text(
                json.dumps(_document(name, nodes)), encoding="utf-8"
            )
        tab = DIR / f"tab_{size}"
        tab.mkdir(exist_ok=True)
        with (tab / f"rows_{size}.csv").open("w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(f"c{c:02d}" for c in range(CSV_COLUMNS)) + "\n")
            for r in range(CSV_ROWS[size]):
                fh.write(",".join(f"v{r}_{c}" if c % 3 else str(r * c) for c in range(CSV_COLUMNS)))
                fh.write("\n")
    print("written to", DIR)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _fresh(*, concurrency: int | None = None, script: Script | None = None) -> None:
    """Empty both databases and configure the library (concurrency 1, or the defaults)."""
    _runner.guard()
    _runner.wipe_databases()
    overrides: dict = {"extra_models_paths": [MODELS], "normalization_enabled": False}
    if concurrency is None:
        overrides.update(llm_concurrency=4, neo4j_concurrency=10, neo4j_sync_concurrency=8)
    else:
        overrides.update(
            llm_concurrency=concurrency,
            neo4j_concurrency=concurrency,
            neo4j_sync_concurrency=concurrency,
        )
    _runner.configure_harness(**overrides)
    set_script(script or Script(tabular={"rows_": decision("BenchWide")}))


def _doc_file(prefix: str, size: str) -> Path:
    name = DOC_NAME if prefix == "doc" else ANNOT_NAME
    return DIR / f"{prefix}_{size}" / f"extract-{name}.json"


async def _ingest_doc(size: str, prefix: str = "doc") -> None:
    from scinr.newton.stages import run_ingestion

    result = await run_ingestion(files=[_doc_file(prefix, size)], tenant_id=TENANT)
    assert result.success, [d.errors for d in result.documents]


async def _ingest_rows(size: str) -> None:
    from scinr.newton.stages import run_tabular_pipeline

    result = await run_tabular_pipeline(input_raw=str(DIR / f"tab_{size}"), tenant_id=TENANT)
    assert result.success, [d.errors for d in result.documents]


async def _measure(call) -> int:
    """Peak live bytes allocated while *call* (a coroutine function) runs."""
    gc.collect()
    tracemalloc.start()
    try:
        await call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


# ---------------------------------------------------------------------------
# Cases: each returns {variant: peak bytes} for one size
# ---------------------------------------------------------------------------


async def case_ingest_doc(size: str) -> dict[str, int]:
    from scinr.newton.stages import run_ingestion

    _fresh(concurrency=1)
    path = _doc_file("doc", size)

    async def call() -> None:
        result = await run_ingestion(files=[path], tenant_id=TENANT)
        assert result.success, [d.errors for d in result.documents]

    return {"-": await _measure(call)}


async def case_tabular_rows(size: str) -> dict[str, int]:
    from scinr.newton.stages import run_tabular_pipeline

    _fresh(concurrency=1)

    async def call() -> None:
        result = await run_tabular_pipeline(
            input_raw=str(DIR / f"tab_{size}"), tenant_id=TENANT
        )
        assert result.success, [d.errors for d in result.documents]

    return {"-": await _measure(call)}


async def case_annotate_extract(size: str) -> dict[str, int]:
    from scinr.newton.annotation.agent import run_annotation_agent
    from scinr.newton.entity_extraction.agent import run_entity_extraction_agent

    ids = _node_ids(ANNOTATE_NODES[size])
    script = Script(
        annotations={i: decision("BenchItem") for i in ids},
        extractions={
            i: {"bench_item": {"name": f"item-{i}", "note": f"note {i}", "tags": ["a", "b"]}}
            for i in ids
        },
    )
    out: dict[str, int] = {}
    for variant, concurrency in (("c1", 1), ("cdef", None)):
        _fresh(concurrency=concurrency, script=script)
        await _ingest_doc(size, "annot")

        async def call() -> None:
            ann = await run_annotation_agent(
                ANNOT_NAME, tenant_id=TENANT, doc_path=ANNOT_NAME
            )
            assert not ann["errors"], ann["errors"][:3]
            ext = await run_entity_extraction_agent(
                ANNOT_NAME, tenant_id=TENANT, doc_path=ANNOT_NAME
            )
            assert not ext["errors"], ext["errors"][:3]

        out[variant] = await _measure(call)
    return out


async def case_freeze_export(size: str) -> dict[str, int]:
    from scinr.newton.ingest.freeze import freeze_document

    _fresh(concurrency=1)
    await _ingest_doc(size)
    await _ingest_rows(size)

    async def call() -> None:
        for path in (DOC_NAME, f"rows_{size}.csv"):
            result = await freeze_document(path, tenant_id=TENANT)
            assert result.found and result.frozen_blob_id, result

    return {"-": await _measure(call)}


async def case_restore(size: str) -> dict[str, int]:
    from scinr.newton.ingest.freeze import freeze_document
    from scinr.newton.ingest.restore import restore_document

    _fresh(concurrency=1)
    await _ingest_doc(size)
    await _ingest_rows(size)
    paths = (DOC_NAME, f"rows_{size}.csv")
    out: dict[str, int] = {}
    for variant, concurrency in (("c1", 1), ("c4", 4)):
        for path in paths:
            await freeze_document(path, tenant_id=TENANT)

        async def call(concurrency: int = concurrency) -> None:
            for path in paths:
                result = await restore_document(path, tenant_id=TENANT, concurrency=concurrency)
                assert result.found and result.documents_restored, result

        out[variant] = await _measure(call)
    return out


async def case_delete_gc(size: str) -> dict[str, int]:
    from scinr.newton.ingest.deletion import collect_orphans, delete_document

    _fresh(concurrency=1)
    await _ingest_doc(size)
    await _ingest_rows(size)

    async def call() -> None:
        for path in (DOC_NAME, f"rows_{size}.csv"):
            result = await delete_document(path, tenant_id=TENANT)
            assert result.found, result
        await collect_orphans(tenant_id=TENANT)

    return {"-": await _measure(call)}


RUNNERS = {
    "ingest_doc": case_ingest_doc,
    "tabular_rows": case_tabular_rows,
    "annotate_extract": case_annotate_extract,
    "freeze_export": case_freeze_export,
    "restore": case_restore,
    "delete_gc": case_delete_gc,
}


def run_case(case: str) -> dict[str, dict[str, int]]:
    """Warm-up, then every size of *case* (one asyncio.run each); ``{size: {variant: peak}}``."""
    asyncio.run(RUNNERS[case](WARM))
    results: dict[str, dict[str, int]] = {}
    for size in SIZES:
        results[size] = asyncio.run(RUNNERS[case](size))
        for variant, peak in results[size].items():
            label = case if variant == "-" else f"{case}[{variant}]"
            print(f"{label} {size}: peak={peak / MIB:.1f}MiB", flush=True)
    return results


# ---------------------------------------------------------------------------
# Baseline and table
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def _meta(runs: int) -> dict:
    import neo4j

    return {
        "commit": _git("rev-parse", "HEAD"),
        "src_dirty": bool(_git("status", "--porcelain", "--", "src")),
        "python": platform.python_version(),
        "neo4j_driver": neo4j.__version__,
        "machine": f"{platform.platform()} ({platform.machine()})",
        "runs": runs,
        "unit": "bytes, tracemalloc peak, median of runs",
        "sizes": {
            "ingest_doc / freeze_export / restore / delete_gc (nodes)": DOC_NODES,
            "tabular_rows (rows x 12 columns)": CSV_ROWS,
            "annotate_extract (nodes)": ANNOTATE_NODES,
        },
    }


def _launch(case: str) -> dict[str, dict[str, int]]:
    proc = subprocess.run(
        [sys.executable, __file__, "run", case],
        env={**os.environ, "SCINR_GOLDEN_ALLOW_WIPE": "1", "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"case {case} failed:\n{proc.stdout}\n{proc.stderr[-3000:]}")
    results: dict[str, dict[str, int]] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            results = json.loads(line[len("RESULT ") :])
    return results


def _median(runs: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for size in SIZES:
        out[size] = {
            v: int(statistics.median(r[size][v] for r in runs)) for v in runs[0][size]
        }
    return out


def _own_ok(case: str, peaks: dict[str, dict[str, int]]) -> dict[str, bool]:
    """Per variant: does ``large <= small * 1.10`` hold (always true for unbounded cases)."""
    if case not in BOUNDED:
        return {v: True for v in peaks["small"]}
    return {v: peaks["large"][v] <= peaks["small"][v] * GROWTH_FACTOR for v in peaks["small"]}


def _print_table(now: dict[str, dict], baseline: dict | None) -> bool:
    print(f"{'case':<22}{'size':<7}{'baseline':>10}{'now':>9}{'delta':>9}  ok")
    all_ok = True
    for case, peaks in now.items():
        own = _own_ok(case, peaks)
        base_peaks = baseline["peaks_bytes"].get(case) if baseline else None
        base_own = baseline["own_criteria"].get(case, {}) if baseline else {}
        for size in SIZES:
            for variant, peak in peaks[size].items():
                label = case if variant == "-" else f"{case}[{variant}]"
                base = base_peaks[size][variant] if base_peaks else None
                ok = True
                if base is not None:
                    ok = peak <= base * TOLERANCE_FACTOR + TOLERANCE_BYTES
                if size == "large" and not own[variant]:
                    ok = ok and base_own.get(variant, {}).get("baseline_violates", False)
                all_ok &= ok
                delta = f"{(peak - base) / MIB:+.1f}" if base is not None else "-"
                base_s = f"{base / MIB:.1f}" if base is not None else "-"
                print(
                    f"{label:<22}{size:<7}{base_s:>10}{peak / MIB:>9.1f}{delta:>9}  "
                    f"{'ok' if ok else 'FAIL'}"
                )
    return all_ok


def run_all(cases: list[str], runs: int, write_baseline: bool) -> int:
    now: dict[str, dict] = {}
    raw: dict[str, list] = {}
    for case in cases:
        collected = []
        for n in range(runs):
            print(f"[{case}] run {n + 1}/{runs}", flush=True)
            collected.append(_launch(case))
        raw[case] = collected
        now[case] = _median(collected)
    if write_baseline:
        document = {
            "meta": _meta(runs),
            "peaks_bytes": now,
            "runs_bytes": raw,
            "own_criteria": {
                case: {
                    v: {"large_le_small_x1.10": ok, "baseline_violates": not ok}
                    for v, ok in _own_ok(case, peaks).items()
                }
                for case, peaks in now.items()
            },
        }
        BASELINE.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("baseline written to", BASELINE)
        baseline = document
    else:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    return 0 if _print_table(now, baseline) else 1


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "make":
        make()
        return 0
    if command == "run" and len(sys.argv) == 3 and sys.argv[2] in RUNNERS:
        if not DIR.exists():
            raise SystemExit("run `bench_graph.py make` first")
        print("RESULT " + json.dumps(run_case(sys.argv[2])))
        return 0
    if command == "all":
        args = sys.argv[2:]
        runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 3
        cases = args[args.index("--cases") + 1].split(",") if "--cases" in args else list(CASES)
        return run_all(cases, runs, "--write-baseline" in args)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
