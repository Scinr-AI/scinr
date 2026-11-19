"""
tests/golden/_dump.py — Neutral dump of the graph and normalization of volatile values.

The dump is the observation tool of the golden harness (``tests/integration/
test_graph_store_golden.py``) and, later, of the ``GraphStore`` contract suite
(WP9): it reads the graph with plain Cypher through a driver that the caller
owns, and returns a JSON-able structure identical for any backend that can
produce it.

Shape::

    {"nodes": [{"key": "<primary label>|<business key>", "labels": [...], "props": {...}}],
     "relationships": [{"type": "...", "from": "<key>", "to": "<key>", "props": {...}}]}

Nodes are identified by their business key (``ingest.freeze.node_identity``) and both
lists are sorted, so the dump does not depend on the internal ids of the engine.
Nodes whose labels have no known business key (they should not exist) are identified
by their labels and properties.

Volatile values are normalized by :func:`normalize_artifacts`, applied to the dump and
to the recorded trace together so MongoDB ObjectIds get the same number in both:

- ISO 8601 date-times -> ``<ts>``;
- Neo4j ``elementId`` values -> ``<eid>`` (they are internal and change on every run; the
  statements that carry them keep their shape and the length of their lists);
- MongoDB ObjectIds (24 hex) -> ``<oid:n>``, numbered by ascending value, i.e. by
  creation order, which does not depend on the order in which they appear.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from scinr.newton.exceptions import FreezeError
from scinr.newton.ingest.freeze import node_identity

TIMESTAMP_PLACEHOLDER = "<ts>"

_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
# Internal ids of Neo4j (``elementId``), e.g. ``4:707f9ad3-d321-4302-9e20-ccbef767e5ea:187``:
# they change from run to run and only the OrphanCollector sends them back to the server.
_ELEMENT_ID_RE = re.compile(r"\d+:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:\d+")
ELEMENT_ID_PLACEHOLDER = "<eid>"
_OBJECT_ID_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{24}(?![0-9a-fA-F])")
_VOLATILE_KEY_RE = re.compile(r"duration|elapsed|seconds", re.IGNORECASE)


def jsonable(value: Any) -> Any:
    """Deep copy of *value* made of JSON types only (stable under later mutation)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    to_native = getattr(value, "to_native", None)  # neo4j.time.DateTime / Date / Time
    if callable(to_native):
        return jsonable(to_native())
    if hasattr(value, "value") and isinstance(getattr(value, "value", None), (str, int)):
        return value.value  # Enum
    return repr(value)


# ---------------------------------------------------------------------------
# Graph dump
# ---------------------------------------------------------------------------


def node_key(labels: list[str], props: dict[str, Any]) -> str:
    try:
        primary, fields = node_identity(labels)
    except FreezeError:
        return "UNKNOWN|" + json.dumps(
            [sorted(labels), props], sort_keys=True, default=str, ensure_ascii=False
        )
    key = {f: props.get(f) for f in fields}
    return f"{primary}|{json.dumps(key, sort_keys=True, default=str, ensure_ascii=False)}"


def build_dump(
    nodes: Iterable[tuple[Sequence[str], dict[str, Any]]],
    rels: Iterable[tuple[str, str, str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build the neutral dump from ``(labels, props)`` nodes and ``(type, from, to, props)``.

    The relationship ends are node keys already computed with :func:`node_key`. Any backend
    that can list its nodes and relationships produces the same structure.
    """
    out_nodes: list[dict[str, Any]] = []
    for labels, raw_props in nodes:
        labels = sorted(labels)
        props = jsonable(raw_props)
        out_nodes.append({"key": node_key(labels, props), "labels": labels, "props": props})
    relationships = [
        {"type": rel_type, "from": src, "to": dst, "props": jsonable(props)}
        for rel_type, src, dst, props in rels
    ]
    return {"nodes": _sorted_nodes(out_nodes), "relationships": _sorted_rels(relationships)}


def dump_graph(driver: Any, database: str) -> dict[str, Any]:
    """Dump the whole graph of *database* with a sync Neo4j driver."""
    with driver.session(database=database) as session:
        raw_nodes = session.run(
            "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props"
        ).data()
        raw_rels = session.run(
            "MATCH (a)-[r]->(b) RETURN type(r) AS type, elementId(a) AS a, elementId(b) AS b, "
            "properties(r) AS props"
        ).data()

    keys = {row["id"]: node_key(sorted(row["labels"]), jsonable(row["props"])) for row in raw_nodes}
    return build_dump(
        [(row["labels"], row["props"]) for row in raw_nodes],
        [(row["type"], keys[row["a"]], keys[row["b"]], row["props"]) for row in raw_rels],
    )


def _sorted_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(nodes, key=lambda n: (n["key"], json.dumps(n["props"], sort_keys=True)))


def _sorted_rels(rels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rels, key=lambda r: json.dumps([r["type"], r["from"], r["to"], r["props"]], sort_keys=True)
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_strings(v)


def collect_object_ids(*structures: Any) -> dict[str, str]:
    """Map every ObjectId found in *structures* to ``<oid:n>`` (ascending value order)."""
    found: set[str] = set()
    for structure in structures:
        for text in _walk_strings(structure):
            found.update(_OBJECT_ID_RE.findall(_TIMESTAMP_RE.sub("", text)))
    return {oid: f"<oid:{n}>" for n, oid in enumerate(sorted(found))}


def _normalize_text(text: str, oids: dict[str, str]) -> str:
    text = _TIMESTAMP_RE.sub(TIMESTAMP_PLACEHOLDER, text)
    text = _ELEMENT_ID_RE.sub(ELEMENT_ID_PLACEHOLDER, text)
    return _OBJECT_ID_RE.sub(lambda m: oids.get(m.group(0), m.group(0)), text)


def _normalize(value: Any, oids: dict[str, str], *, timings: bool = False) -> Any:
    """Normalize strings; with *timings*, numbers under duration-like keys become ``<ts>``."""
    if isinstance(value, str):
        return _normalize_text(value, oids)
    if isinstance(value, dict):
        return {
            _normalize_text(str(k), oids): (
                TIMESTAMP_PLACEHOLDER
                if timings and _VOLATILE_KEY_RE.search(str(k)) and isinstance(v, (int, float))
                else _normalize(v, oids, timings=timings)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_normalize(v, oids, timings=timings) for v in value]
    return value


def sort_lists(value: Any) -> Any:
    """*value* with every list recursively sorted (by JSON text).

    Neo4j does not specify the order of the rows of a query without ``ORDER BY`` and, after
    the harness wipes the graph, it changes from run to run (the node ids it hands out
    again change). Lists that come from such a result (the keys of the documents to delete,
    the documents of a result object...) are therefore compared as sets-with-multiplicity.
    """
    if isinstance(value, dict):
        return {k: sort_lists(v) for k, v in value.items()}
    if isinstance(value, list):
        return sorted(
            (sort_lists(v) for v in value),
            key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False),
        )
    return value


def normalize_artifacts(
    trace: list[dict[str, Any]], graph: dict[str, Any], extras: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize the trace and the graph dump (plus scenario *extras*) together.

    Returns ``(trace, graph_with_extras)``; the graph dump comes back re-sorted because
    business keys may have contained volatile values.
    """
    oids = collect_object_ids(trace, graph, extras)
    norm_trace = _normalize(trace, oids)
    nodes = _sorted_nodes(_normalize(graph["nodes"], oids))
    rels = _sorted_rels(_normalize(graph["relationships"], oids))
    out: dict[str, Any] = {"nodes": nodes, "relationships": rels}
    if extras:
        out["extras"] = sort_lists(_normalize(extras, oids, timings=True))
    return norm_trace, out
