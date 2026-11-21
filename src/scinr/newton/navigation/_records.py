"""
navigation/_records.py — What counts as a *field* of a ``:ModelInstance`` (engine-neutral).

The SQL backend aggregates the memory summaries and the record search in Python over the node
dictionaries (``sql.rows.row_to_node``); the Neo4j backend does the same in Cypher. Both must
agree on the definitions below, which are the ones of ``GraphNavigator.summarize_model_classes``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal

from scinr.newton.navigation.models import (
    ModelClassSummary,
    ModelLinkSummary,
    SourceContribution,
    SourceSummary,
)

__all__ = [
    "RecordKind",
    "build_sources",
    "cap",
    "class_summaries",
    "field_keys",
    "link_summaries",
    "record_fields",
    "record_kind",
    "record_kind_of",
    "search_hits",
    "value_text",
]

#: Keys of a node that are not fields: the identity pair (``ModelInstanceRef`` carries them
#: apart from ``properties``) and the engine-internal ones the mapping drops.
NON_FIELD_KEYS = frozenset({"uid", "model_class", "_labels", "_elementId", "_id"})

RecordKind = Literal["empty", "placeholder", "real"]


def record_fields(node: Mapping[str, Any], ignore_keys: Collection[str] = ()) -> list[str]:
    """Sorted keys of *node* that are fields: not identity, not ignored, value not null."""
    return sorted(
        key
        for key, value in node.items()
        if key not in NON_FIELD_KEYS and key not in ignore_keys and value is not None
    )


def record_kind(n_fields: int, extracted: bool) -> RecordKind:
    """``empty`` (no fields), ``placeholder`` (one field, not extracted) or ``real``.

    *extracted*: some ``:ExtractionResult`` has a relationship to the record.
    """
    if n_fields == 0:
        return "empty"
    if n_fields == 1 and not extracted:
        return "placeholder"
    return "real"


def value_text(value: Any) -> str | None:
    """The text of a scalar value like Cypher's ``toStringOrNull`` (lists and maps: ``None``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return None


def cap(items: list[Any], limit: int | None) -> list[Any]:
    """The first *limit* items (all of them for ``None`` or a negative limit)."""
    return items[: int(limit)] if limit is not None and limit >= 0 else items


def record_kind_of(
    node: Mapping[str, Any], extracted: bool, ignore_keys: Collection[str]
) -> RecordKind:
    return record_kind(len(record_fields(node, ignore_keys)), extracted)


def class_summaries(
    records: Iterable[tuple[Mapping[str, Any], bool]],
    selectable: Mapping[str, bool | None],
    ignore_keys: Collection[str],
    limit: int | None,
) -> list[ModelClassSummary]:
    """``summarize_model_classes`` from ``(record node, extracted)`` pairs."""
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"real": 0, "empty": 0, "placeholder": 0}
    )
    for node, extracted in records:
        totals[node["model_class"]][record_kind_of(node, extracted, ignore_keys)] += 1
    out = [
        ModelClassSummary(
            model_class=model_class,
            records=counts["real"],
            empty=counts["empty"],
            placeholders=counts["placeholder"],
            selectable=selectable.get(model_class),
        )
        for model_class, counts in totals.items()
    ]
    out.sort(key=lambda s: (-s.records, s.model_class))
    return cap(out, limit)


def link_summaries(
    links: Iterable[tuple[str, str, str, Mapping[str, Any], bool]],
    ignore_keys: Collection[str],
    limit: int | None,
) -> list[ModelLinkSummary]:
    """``summarize_model_instance_links`` from ``(from uid, from class, rel type, to node,
    to extracted)`` tuples (one per relationship)."""
    count: dict[tuple[str, str, str], int] = defaultdict(int)
    to_empty: dict[tuple[str, str, str], int] = defaultdict(int)
    sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for from_uid, from_class, rel_type, target, extracted in links:
        group = (from_class, rel_type, target["model_class"])
        count[group] += 1
        if record_kind_of(target, extracted, ignore_keys) != "real":
            to_empty[group] += 1
            sources[group].add(from_uid)
    out = [
        ModelLinkSummary(
            from_class=g[0],
            rel_type=g[1],
            to_class=g[2],
            links=n,
            to_empty=to_empty[g],
            from_to_empty=len(sources[g]),
        )
        for g, n in count.items()
    ]
    out.sort(key=lambda s: (-s.links, s.from_class, s.rel_type, s.to_class))
    return cap(out, limit)


def record_jobs(node: Mapping[str, Any]) -> list[str]:
    """The job ids of a record (``job_ids``, or the scalar ``job_id``), without nulls."""
    jobs = node.get("job_ids")
    return [j for j in (jobs if jobs is not None else [node.get("job_id")]) if j is not None]


def build_sources(
    latest: Sequence[Mapping[str, Any]],
    versions: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    records: Iterable[tuple[Mapping[str, Any], bool]],
    ignore_keys: Collection[str],
    limit: int | None,
) -> list[SourceSummary]:
    """``summarize_sources`` from the latest documents, all the versions by ``(tenant, path)``
    and ``(record node, extracted)`` pairs."""
    groups: dict[tuple[tuple[str, ...], str], int] = defaultdict(int)
    for node, extracted in records:
        jobs = record_jobs(node)
        if jobs and record_kind_of(node, extracted, ignore_keys) == "real":
            groups[(tuple(jobs), node["model_class"])] += 1
    out: list[SourceSummary] = []
    for node in latest:
        history = sorted(
            versions.get((node.get("tenant_id"), node["path"]), ()),
            key=lambda v: v.get("version") or 0,
            reverse=True,
        )
        dated = sorted(
            (v for v in history if v.get("load_date") is not None),
            key=lambda v: str(v["load_date"]),
            reverse=True,
        )
        ordered = dated + [v for v in history if v.get("load_date") is None]
        job_ids = list(
            dict.fromkeys(j for j in [node.get("job_id"), *(v.get("job_id") for v in ordered)] if j)
        )
        wanted = set(job_ids)
        contributions = sorted(
            (
                SourceContribution(job_ids=list(jobs), model_class=model_class, records=n)
                for (jobs, model_class), n in groups.items()
                if wanted.intersection(jobs)
            ),
            key=lambda c: (c.model_class, c.job_ids),
        )
        load_date = node.get("load_date")
        out.append(
            SourceSummary(
                raw={},
                path=node["path"],
                name=node.get("name") or node["path"],
                job_id=node.get("job_id"),
                job_ids=job_ids,
                added_at=None if load_date is None else str(load_date),
                added_by_id=node.get("created_by_user_id"),
                contributions=contributions,
            )
        )
    out.sort(key=lambda s: s.path)
    dated_out = sorted((s for s in out if s.added_at is not None), key=lambda s: s.added_at or "", reverse=True)
    out = dated_out + [s for s in out if s.added_at is None]
    return cap(out, limit)


def search_hits(
    nodes: Iterable[Mapping[str, Any]], text: str, ignore_keys: Collection[str], limit: int | None
) -> list[tuple[Mapping[str, Any], str]]:
    """``(record node, field)`` of the records matching *text*: uid hits first, then by uid."""
    needle = text.lower()
    found: list[tuple[int, str, Mapping[str, Any], str]] = []
    for node in nodes:
        if node["uid"] == text:
            found.append((0, node["uid"], node, "uid"))
            continue
        for key in record_fields(node, ignore_keys):
            value = value_text(node[key])
            if value is not None and needle in value.lower():
                found.append((1, node["uid"], node, key))
                break
    found.sort(key=lambda item: (item[0], item[1]))
    return [(item[2], item[3]) for item in cap(found, limit)]


def field_keys(nodes: Iterable[Mapping[str, Any]], ignore_keys: Collection[str]) -> list[str]:
    """Sorted field keys with a value (text values must not be blank)."""
    keys: set[str] = set()
    for node in nodes:
        for key in record_fields(node, ignore_keys):
            value = node[key]
            if not (isinstance(value, str) and not value.strip()):
                keys.add(key)
    return sorted(keys)
