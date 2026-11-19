"""Benchmark of the graph backends (neo4j, postgres, sqlite) on the synthetic corpus (WP7.2).

    SCINR_SQL_ALLOW_WIPE=1 .venv/bin/python scripts/sql_bench/run.py \\
        --backend postgres --nodes 1000000 --out results/postgres-1m.json

The target database is WIPED before every repetition. The script refuses to start unless
``SCINR_SQL_ALLOW_WIPE=1`` and the URL points to a benchmark port (neo4j 7699, postgres 55433)
or to a SQLite file. See README.md for the containers, the run matrix and ``compare.py``.

Each repetition runs in its own subprocess (wipe, ingest, reads, deletes, GC, freeze/restore),
so the process-level memory peak of one repetition does not leak into the next. The result is
the median of the repetitions, per operation. Memory is the peak RSS of the process above its
level at the start of the measured step (``--tracemalloc`` switches to the peak of live Python
allocations, as ``scripts/memory_bench`` does; time is then inflated and the file says so).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus  # noqa: E402

BENCH_PORTS = {"neo4j": 7699, "postgres": 55433}
SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
MIB = 1024 * 1024
SAMPLE_INTERVAL_S = 0.05
TREE_SAMPLE = 100
LOOKUP_SAMPLE = 100
SEARCH_TERMS = 50
DELETE_DOCS = 20
DEFAULT_PG_URL = "postgresql://newton_admin:newton_admin@127.0.0.1:55433/newton_bench"
DEFAULT_NEO4J_URI = "bolt://localhost:7699"
DEFAULT_SQLITE = Path(tempfile.gettempdir()) / "scinr_sql_bench" / "bench.db"
PG_SCHEMA = "newton_bench"
INSTANCE_RELATIONS = (corpus.SUPPLIED_BY, corpus.RELATED_TO)


# ---------------------------------------------------------------------------
# Guard and wipe
# ---------------------------------------------------------------------------


def _sqlite_path(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        raise SystemExit(f"refusing to start: not a sqlite file URL: {url!r}")
    raw = url[len("sqlite:///") :]
    if not raw or raw.startswith(":memory:") or not raw.endswith(SQLITE_SUFFIXES):
        raise SystemExit(
            f"refusing to start: the sqlite URL must point to a *.db / *.sqlite file: {url!r}"
        )
    return Path(raw)


def guard(args: argparse.Namespace) -> None:
    """Refuse to start unless wiping the target is explicitly allowed and it is a bench target."""
    if os.environ.get("SCINR_SQL_ALLOW_WIPE") != "1":
        raise SystemExit(
            "refusing to start: the benchmark wipes its database; set SCINR_SQL_ALLOW_WIPE=1"
        )
    if args.backend == "sqlite":
        _sqlite_path(args.sql_url)
        return
    url = args.neo4j_uri if args.backend == "neo4j" else args.sql_url
    port = urlsplit(url).port
    allowed = {BENCH_PORTS[args.backend]}
    if args.smoke_allow_port:
        allowed.add(args.smoke_allow_port)
    if port not in allowed:
        raise SystemExit(
            f"refusing to start: port {port} of the {args.backend} URL is not the benchmark "
            f"port {BENCH_PORTS[args.backend]}"
        )


def wipe(args: argparse.Namespace) -> None:
    if args.backend == "sqlite":
        db = _sqlite_path(args.sql_url)
        db.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db}{suffix}").unlink(missing_ok=True)
    elif args.backend == "postgres":
        import psycopg
        from psycopg import sql

        with psycopg.connect(args.sql_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(PG_SCHEMA))
            )
    else:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
        try:
            with driver.session(database=args.neo4j_database) as session:
                session.run(
                    "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
                ).consume()
        finally:
            driver.close()


def configure_library(args: argparse.Namespace) -> None:
    from scinr.newton.config import configure

    if args.backend == "neo4j":
        # Neo4j has no SQL storage: snapshots (freeze) go to a throwaway SQLite file.
        freeze_db = Path(tempfile.gettempdir()) / "scinr_sql_bench" / "neo4j_freeze.db"
        freeze_db.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(f"{freeze_db}{suffix}").unlink(missing_ok=True)
        configure(
            graph_backend="neo4j",
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            storage_backend="none",
            freeze_backend="sqlite",
            sql_url=f"sqlite:///{freeze_db}",
            log_level="ERROR",
        )
        return
    configure(
        graph_backend=args.backend,
        storage_backend=args.backend,
        freeze_backend=args.backend,
        sql_url=args.sql_url,
        sql_schema=PG_SCHEMA,
        log_level="ERROR",
    )


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def _current_rss() -> int:
    """Resident set size of this process, in bytes."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) * 1024


class PeakMemory:
    """Context manager: peak memory of the block, in MiB (``.peak_mib``).

    RSS mode samples the process RSS every 50 ms and reports the peak above the level at the
    start. ``tracemalloc`` mode reports the peak of live Python allocations.
    """

    def __init__(self, use_tracemalloc: bool) -> None:
        self._trace = use_tracemalloc
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._peak = 0
        self._base = 0
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL_S):
            self._peak = max(self._peak, _current_rss())

    def __enter__(self) -> PeakMemory:
        if self._trace:
            tracemalloc.start()
            tracemalloc.reset_peak()
            return self
        self._base = self._peak = _current_rss()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._trace:
            self.peak_mib = tracemalloc.get_traced_memory()[1] / MIB
            tracemalloc.stop()
            return
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._peak = max(self._peak, _current_rss())
        self.peak_mib = (self._peak - self._base) / MIB


def log(message: str) -> None:
    print(f"  {time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def percentile(sorted_ms: list[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    return sorted_ms[min(len(sorted_ms) - 1, max(0, round(q * len(sorted_ms) + 0.5) - 1))]


async def latency(calls: list[Callable[[], Awaitable[object]]]) -> dict[str, float]:
    """Run *calls* one after the other (after one unmeasured warm-up) and summarise in ms."""
    await calls[0]()
    samples: list[float] = []
    for call in calls:
        start = time.perf_counter()
        await call()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "n": len(samples),
    }


# ---------------------------------------------------------------------------
# One repetition
# ---------------------------------------------------------------------------


async def ingest(store: object, docs: int, workers: int) -> None:
    """Write documents 0..docs-1 through ``GraphStore``, *workers* at a time."""
    indexes = iter(range(docs))

    async def lane() -> None:
        for index in indexes:
            spec = corpus.build_document(index)
            await asyncio.to_thread(corpus.write_structure, store, spec)
            await corpus.write_extractions(store, spec)

    await asyncio.gather(*(lane() for _ in range(workers)))


async def repetition(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    from scinr.newton import collect_orphans, delete_document, freeze_document, restore_document
    from scinr.newton.graph import get_graph_store
    from scinr.newton.navigation.factory import get_graph_navigator
    from scinr.newton.navigation.models import NodeSelector

    guard(args)
    log("wipe + schema")
    wipe(args)
    configure_library(args)
    store = get_graph_store()
    store.schema.setup()
    await corpus.write_catalog(store)
    docs = corpus.documents_for(args.nodes)
    ops: dict[str, dict[str, float]] = {}

    log(f"ingest {docs} documents")
    with PeakMemory(args.tracemalloc) as memory:
        start = time.perf_counter()
        await ingest(store, docs, args.workers)
        seconds = time.perf_counter() - start
    nodes = docs * corpus.nodes_per_document()
    ops["ingest"] = {
        "seconds": seconds,
        "nodes": nodes,
        "nodes_per_s": nodes / seconds,
        "peak_mib": memory.peak_mib,
    }

    log("large document")
    large = corpus.build_document(docs, args.large_scale)
    corpus.write_structure(store, large)
    await corpus.write_extractions(store, large)

    rng = random.Random(corpus.SEED + 1)

    def pick(count: int) -> list[corpus.DocSpec]:
        return [corpus.build_document(i) for i in rng.sample(range(docs), min(count, docs))]

    def call(fn: Callable[..., Awaitable[object]], *a: object, **kw: object):
        return lambda: fn(*a, **kw)

    def instance(spec: corpus.DocSpec, k: int) -> NodeSelector:
        return NodeSelector(
            type="ModelInstance", key="uid", value=spec.instance_uid(k % spec.instances)
        )

    nav = await get_graph_navigator()
    try:
        specs = pick(TREE_SAMPLE)
        log("get_structure_tree")
        ops["get_structure_tree"] = await latency(
            [
                call(
                    nav.get_structure_subtree,
                    s.root_id,
                    include_info_units=True,
                    tenant_id=s.stored_tenant,
                )
                for s in specs
            ]
        )
        log("get_info_units")
        ops["get_info_units"] = await latency(
            [
                call(
                    nav.get_info_units,
                    s.node_full_id(rng.choice(s.leaf_paths)),
                    tenant_id=s.stored_tenant,
                )
                for s in specs
            ]
        )
        log("search_info_units")
        terms = corpus.search_terms(SEARCH_TERMS, corpus.SEED + 2)
        ops["search_info_units"] = await latency(
            [
                call(nav.search_info_units, t, tenant_id=rng.choice(specs).stored_tenant, limit=25)
                for t in terms
            ]
        )
        specs = pick(LOOKUP_SAMPLE)
        log("get_model_instance_by_key")
        ops["get_model_instance_by_key"] = await latency(
            [
                call(
                    nav.get_model_instance_by_key,
                    corpus.MODEL_CLASS,
                    {"code": s.instance_code(rng.randrange(s.instances))},
                    tenant_id=s.stored_tenant,
                )
                for s in specs
            ]
        )
        log("subgraph")
        ops["subgraph"] = await latency(
            [
                call(
                    nav.subgraph,
                    instance(s, rng.randrange(s.instances)),
                    depth=2,
                    tenant_id=s.stored_tenant,
                )
                for s in specs
            ]
        )
        log("shortest_path")
        pairs = [(s, rng.randrange(s.instances), rng.randint(2, 14)) for s in specs]
        ops["shortest_path"] = await latency(
            [
                call(
                    nav.shortest_path,
                    instance(s, k),
                    instance(s, k + d),
                    edge_types=INSTANCE_RELATIONS,
                    tenant_id=s.stored_tenant,
                )
                for s, k, d in pairs
            ]
        )
    finally:
        await nav.close()

    log("delete_document")
    doomed = pick(DELETE_DOCS)
    start = time.perf_counter()
    for spec in doomed:
        result = await delete_document(spec.doc_path, tenant_id=spec.tenant)
        if not result.found:
            raise RuntimeError(f"delete_document did not find {spec.doc_path}")
    ops["delete_document"] = {
        "mean_s": (time.perf_counter() - start) / len(doomed),
        "n": len(doomed),
    }

    log("collect_orphans")
    start = time.perf_counter()
    for tenant in (*corpus.TENANTS, None):
        await collect_orphans(tenant_id=tenant)
    ops["collect_orphans"] = {"seconds": time.perf_counter() - start}

    large_nodes = corpus.nodes_per_document(args.large_scale)
    for name, call in (
        ("freeze", freeze_document),
        ("restore", restore_document),
    ):
        log(name)
        with PeakMemory(args.tracemalloc) as memory:
            start = time.perf_counter()
            result = await call(large.doc_path, tenant_id=large.tenant)
            seconds = time.perf_counter() - start
        if not result.found:
            raise RuntimeError(f"{name}_document did not find {large.doc_path}")
        ops[name] = {"seconds": seconds, "peak_mib": memory.peak_mib, "nodes": large_nodes}
    return ops


def worker(args: argparse.Namespace) -> None:
    ops = asyncio.run(repetition(args))
    Path(args.worker_out).write_text(json.dumps(ops))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def median_of(runs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        op: {metric: statistics.median(run[op][metric] for run in runs) for metric in runs[0][op]}
        for op in runs[0]
    }


def machine() -> dict[str, object]:
    try:
        memory_gib = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (ValueError, OSError):
        memory_gib = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "memory_gib": memory_gib,
        "python": platform.python_version(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--backend", choices=("neo4j", "postgres", "sqlite"), required=True)
    parser.add_argument("--nodes", type=int, required=True, help="target number of graph nodes")
    parser.add_argument("--out", required=True, help="results.json to write")
    parser.add_argument("--reps", type=int, default=3, help="repetitions (median); default 3")
    parser.add_argument("--workers", type=int, default=4, help="documents ingested at once")
    parser.add_argument("--large-scale", type=int, default=20, help="size factor of the freeze doc")
    parser.add_argument("--tracemalloc", action="store_true", help="memory = live Python peak")
    parser.add_argument("--sql-url", help="postgres or sqlite URL (defaults per backend)")
    parser.add_argument(
        "--neo4j-uri", default=os.environ.get("SCINR_BENCH_NEO4J_URI", DEFAULT_NEO4J_URI)
    )
    parser.add_argument("--neo4j-user", default=os.environ.get("SCINR_BENCH_NEO4J_USER", "neo4j"))
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("SCINR_BENCH_NEO4J_PASSWORD", "benchmark-password"),
    )
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument(
        "--smoke-allow-port",
        type=int,
        help="FOR SMOKE TESTS ONLY: also accept this port as the benchmark port of the backend "
        "(e.g. a disposable test container). Never use it for a real run.",
    )
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sql_url is None:
        args.sql_url = {
            "postgres": os.environ.get("SCINR_BENCH_PG_URL", DEFAULT_PG_URL),
            "sqlite": f"sqlite:///{DEFAULT_SQLITE}",
            "neo4j": "",
        }[args.backend]
    return args


def main() -> None:
    args = parse_args()
    if args.worker_out:
        worker(args)
        return
    guard(args)
    runs: list[dict[str, dict[str, float]]] = []
    argv = [a for a in sys.argv[1:]]
    for rep in range(args.reps):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rep.json"
            print(f"[{args.backend}] repetition {rep + 1}/{args.reps} ...", flush=True)
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *argv, "--worker-out", str(out)],
                check=True,
            )
            runs.append(json.loads(out.read_text()))
    docs = corpus.documents_for(args.nodes)
    result = {
        "meta": {
            "backend": args.backend,
            "nodes_target": args.nodes,
            "nodes_actual": docs * corpus.nodes_per_document(),
            "documents": docs,
            "reps": args.reps,
            "workers": args.workers,
            "large_scale": args.large_scale,
            "memory_mode": "tracemalloc" if args.tracemalloc else "rss",
            "seed": corpus.SEED,
            "machine": machine(),
            "smoke_port_override": args.smoke_allow_port,
        },
        "ops": median_of(runs),
        "runs": runs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
