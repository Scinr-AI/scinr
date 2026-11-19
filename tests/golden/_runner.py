"""
tests/golden/_runner.py — Runs a golden scenario and stores / compares its artifacts.

Artifacts of each scenario, in ``tests/golden/<scenario>/``:

- ``trace.jsonl``: every statement sent to Neo4j (see ``_recorder.py``), normalized;
- ``graph.json``: the final graph (see ``_dump.py``) plus the scenario ``extras``.

The harness **wipes** the Neo4j graph and every MongoDB collection of its database before
each scenario. It therefore refuses to run unless ``SCINR_GOLDEN_ALLOW_WIPE=1`` and both
servers are the disposable containers of the plan (Neo4j on port 7698, MongoDB on 27028,
on localhost); see ``guard()``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ._dump import dump_graph, normalize_artifacts, sort_lists
from ._llm import ScriptedLLM, deterministic_uuid4
from ._recorder import recording
from ._scenarios import SCENARIOS, Env, prepare_script

GOLDEN_DIR = Path(__file__).parent
MODELS_DIR = GOLDEN_DIR / "models"

ALLOW_ENV = "SCINR_GOLDEN_ALLOW_WIPE"
NEO4J_PORT = 7698
MONGO_PORT = 27028
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

NEO4J_URI = os.getenv("SCINR_GOLDEN_NEO4J_URI", f"bolt://localhost:{NEO4J_PORT}")
NEO4J_USER = os.getenv("SCINR_GOLDEN_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("SCINR_GOLDEN_NEO4J_PASSWORD", "scratchpass1")
NEO4J_DATABASE = "neo4j"
MONGO_URI = os.getenv("SCINR_GOLDEN_MONGODB_URI", f"mongodb://localhost:{MONGO_PORT}")
MONGO_DATABASE = "scinr_golden"


class GoldenUnavailable(Exception):
    """The harness cannot (or must not) run here; the message says why."""


def guard() -> None:
    """Raise :class:`GoldenUnavailable` unless it is safe to wipe the databases."""
    if os.getenv(ALLOW_ENV) != "1":
        raise GoldenUnavailable(f"{ALLOW_ENV}=1 not set (the harness wipes its databases)")
    for label, uri, port in (
        ("Neo4j", NEO4J_URI, NEO4J_PORT),
        ("MongoDB", MONGO_URI, MONGO_PORT),
    ):
        parsed = urlparse(uri)
        if parsed.hostname not in LOCAL_HOSTS or parsed.port != port:
            raise GoldenUnavailable(
                f"{label} URI {uri!r} is not the disposable container "
                f"(localhost:{port}); refusing to wipe it"
            )
    if os.getenv("PYTHONHASHSEED") != "0":
        raise GoldenUnavailable("PYTHONHASHSEED=0 is required (set iteration order is hashed)")


def wipe_databases() -> None:
    """Empty the graph and every MongoDB collection (indexes and schema are kept).

    Re-checks :func:`guard` itself: no caller can wipe anything else by mistake.
    """
    guard()
    import pymongo
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        raise GoldenUnavailable(f"Neo4j not reachable at {NEO4J_URI}: {exc}") from exc
    with driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    try:
        db = client[MONGO_DATABASE]
        for name in db.list_collection_names():
            db[name].delete_many({})
    except Exception as exc:  # noqa: BLE001
        raise GoldenUnavailable(f"MongoDB not reachable at {MONGO_URI}: {exc}") from exc
    finally:
        client.close()


def configure_harness(**overrides: Any) -> None:
    """``configure()`` the library for the harness databases and a scripted LLM.

    *overrides* replace any of the default arguments (the memory bench changes the models
    folder and the concurrency). With a SQL ``graph_backend`` in *overrides*, the Neo4j
    connection arguments, the MongoDB ones and the default ``storage_backend`` are left out
    (the SQL engine comes with its own ``storage_backend`` / ``freeze_backend`` / ``sql_url``);
    the concurrency settings stay, so the scenarios run the same way on every engine.
    """
    from scinr.newton.config import (
        configure,
        reset_llm_semaphore,
        reset_neo4j_semaphore,
        reset_neo4j_sync_semaphore,
    )

    # The semaphores bind to the event loop of their first use; every scenario has its own.
    reset_llm_semaphore()
    reset_neo4j_semaphore()
    reset_neo4j_sync_semaphore()
    settings: dict[str, Any] = dict(
        llm=ScriptedLLM(),
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_database=NEO4J_DATABASE,
        storage_backend="mongodb",
        mongodb_uri=MONGO_URI,
        mongodb_database=MONGO_DATABASE,
        extra_models_paths=[MODELS_DIR],
        llm_concurrency=1,
        neo4j_concurrency=1,
        neo4j_sync_concurrency=1,
        normalization_enabled=True,
        log_level="WARNING",
    )
    if overrides.get("graph_backend", "neo4j") != "neo4j":
        for name in _ENGINE_CONNECTION_SETTINGS:
            settings.pop(name)
    configure(**{**settings, **overrides})


#: Defaults of :func:`configure_harness` that only make sense for Neo4j + MongoDB.
_ENGINE_CONNECTION_SETTINGS = (
    "neo4j_uri",
    "neo4j_user",
    "neo4j_password",
    "neo4j_database",
    "storage_backend",
    "mongodb_uri",
    "mongodb_database",
)


async def run_scenario(
    name: str,
    *,
    settings: dict[str, Any] | None = None,
    dump: Callable[[], dict[str, Any]] | None = None,
    with_trace: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run scenario *name* and return its normalized ``(trace, graph)``.

    By default on the harness Neo4j + MongoDB (wiped first). With *settings* (the
    ``configure_harness()`` arguments of another engine, already emptied by the caller) and
    *dump* (its neutral dump), on that engine instead; the trace then only holds what was
    sent to Neo4j (nothing, for a SQL engine). Without *with_trace* the trace is dropped
    before the normalization (the ObjectId placeholders are then numbered from the graph and
    the extras only, so runs on different engines are comparable).
    """
    if settings is None:
        guard()
        wipe_databases()
        configure_harness()  # also resets the catalog memoization and the async driver
    else:
        configure_harness(**settings)
    prepare_script()
    with tempfile.TemporaryDirectory(prefix="scinr_golden_") as tmp:
        with deterministic_uuid4(), recording() as rec:
            extras = await SCENARIOS[name](Env(tmp=Path(tmp)))
    if dump is None:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            graph = dump_graph(driver, NEO4J_DATABASE)
        finally:
            driver.close()
    else:
        graph = dump()
    trace, graph = normalize_artifacts(rec.entries if with_trace else [], graph, extras)
    # Same types as when read back from the artifacts (tuples -> lists, ...).
    return json.loads(json.dumps(trace)), json.loads(json.dumps(graph))


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_artifacts(name: str, trace: list[dict[str, Any]], graph: dict[str, Any]) -> None:
    out = GOLDEN_DIR / name
    out.mkdir(exist_ok=True)
    lines = (json.dumps(e, sort_keys=True, ensure_ascii=False) for e in trace)
    (out / "trace.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "graph.json").write_text(
        json.dumps(graph, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_artifacts(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = GOLDEN_DIR / name
    trace = [
        json.loads(line)
        for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    graph = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    return trace, graph


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

Unit = tuple[str, str, tuple[str, ...]]  # (kind, driver, events as JSON)


def units(trace: list[dict[str, Any]]) -> list[tuple[int, Unit]]:
    """Group the trace by unit of work, in order of unit start.

    ``lane`` is ignored and the lists inside the parameters are compared sorted (see
    ``_dump.sort_lists``).
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in trace:
        grouped.setdefault(entry["unit"], []).append(entry)
    out: list[tuple[int, Unit]] = []
    for unit_id in sorted(grouped):
        entries = grouped[unit_id]
        events = tuple(
            json.dumps(
                {k: sort_lists(e[k]) for k in ("op", "query", "params") if k in e},
                sort_keys=True,
                ensure_ascii=False,
            )
            for e in entries
        )
        out.append((unit_id, (entries[0]["kind"], entries[0]["driver"], events)))
    return out


def _describe(unit: Unit) -> str:
    kind, driver, events = unit
    return f"[{driver} {kind}]\n    " + "\n    ".join(e[:400] for e in events)


def diff_graph(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    for section in ("nodes", "relationships", "extras"):
        if expected.get(section) == actual.get(section):
            continue
        exp, act = expected.get(section), actual.get(section)
        if isinstance(exp, list) and isinstance(act, list):
            exp_s = Counter(json.dumps(x, sort_keys=True) for x in exp)
            act_s = Counter(json.dumps(x, sort_keys=True) for x in act)
            missing = list((exp_s - act_s).elements())[:3]
            extra = list((act_s - exp_s).elements())[:3]
            return f"graph {section} differ.\n  missing: {missing}\n  unexpected: {extra}"
        return (
            f"graph {section} differ.\n  expected: {json.dumps(exp, sort_keys=True)[:1500]}"
            f"\n  actual:   {json.dumps(act, sort_keys=True)[:1500]}"
        )
    return None


def diff_trace(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str | None:
    """First difference between two traces, or ``None`` when they are equivalent.

    1. The multiset of units of work (statements, parameters and transaction boundaries
       inside each unit, in order) must be equal.
    2. The order of the units issued through the **sync** driver must be equal: those
       paths (ingestion, deletion, GC, freeze) are sequential.
    """
    exp_units, act_units = units(expected), units(actual)
    exp_count = Counter(u for _, u in exp_units)
    act_count = Counter(u for _, u in act_units)
    if exp_count != act_count:
        missing = list((exp_count - act_count).elements())
        extra = list((act_count - exp_count).elements())
        message = (
            f"unit multiset differs: {sum(exp_count.values())} expected, "
            f"{sum(act_count.values())} actual."
        )
        if missing:
            message += f"\n  missing (first of {len(missing)}):\n  {_describe(missing[0])}"
        if extra:
            message += f"\n  unexpected (first of {len(extra)}):\n  {_describe(extra[0])}"
        return message
    exp_sync = [u for _, u in exp_units if u[1] == "sync"]
    act_sync = [u for _, u in act_units if u[1] == "sync"]
    for position, (e, a) in enumerate(zip(exp_sync, act_sync, strict=True)):
        if e != a:
            return (
                f"sync unit #{position} out of order.\n  expected:\n  {_describe(e)}"
                f"\n  actual:\n  {_describe(a)}"
            )
    return None
