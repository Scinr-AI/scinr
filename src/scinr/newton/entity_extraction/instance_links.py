"""
entity_extraction/instance_links.py

Resolution rules for ``instance_relationships`` between ModelInstances that
come from the **same** :ExtractionResult (the same table row, the same
StructureNode extraction), and static checks of the declarations themselves.

Same-extraction instances are *composition*, not cross-document references:
when a model declares ``instance_relationships`` to a ``target_model`` that was
extracted next to it, the relationship is resolved against that sibling first,
with no key lookup. Only when no sibling matches does the writer fall back to
the key-addressed shell (``make_instance_uid(target, join keys, tenant)``) that
a later extraction of the target fills by MERGE.

Everything here is pure (no Neo4j, no I/O), shared by the writer
(:mod:`~scinr.newton.entity_extraction.graph_mapper`) and the repair utility
(:mod:`~scinr.newton.ingest.relink`), so both apply exactly the same rule.

Matching rule (:func:`match_sibling`)
-------------------------------------
The candidates are the instances of ``target_model`` in the same extraction,
minus the source instance itself. Each candidate is compared with the join
key the relationship would address (``{remote_field: normalized value}``),
field by field, after :func:`~scinr.newton.utils.uid.normalize_key`:

- ``equal``     — same normalized value (for a list field: same set of items);
- ``contains``  — one side's parts are a subset of the other's, where a value
  is split into parts on :data:`JOIN_VALUE_SEPARATOR` (the separator the
  tabular pipeline uses when it joins several columns into one field) and a
  list field contributes each item;
- ``missing``   — the candidate has no value for that field;
- ``different`` — anything else.

Candidates are ranked in tiers; the first non-empty tier decides:

1. **exact**      — every join field ``equal``;
2. **compatible** — no field ``different`` and at least one ``equal``/``contains``;
3. **unknown**    — every join field ``missing`` on the candidate.

One candidate in the deciding tier → link it. Several → ``ambiguous``: no
sibling is picked and the writer falls back to the shell (and warns).

If no tier has a candidate (every candidate contradicts the key) the
relationship is linked to a **sole** sibling only when all of these hold:
there is exactly one candidate, the relationship produced exactly one target
(not a fan-out list), and the candidate is key-less. A key-less instance can
never be reached by key, so a same-extraction sibling is the only thing the
reference can mean; a *keyed* candidate whose key contradicts the reference
is a different record, so the reference is a genuine cross-document link and
the shell is kept.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

from scinr.newton.utils.uid import normalize_key

log = logging.getLogger(__name__)

JOIN_VALUE_SEPARATOR = ";"
"""Separator between distinct values packed into one field.

``tabular.neo4j_ops._merge_and_join`` joins the values of several columns
mapped to the same ``str`` field with ``"; "``; the matcher splits on it so a
reference to one of those values still finds the instance that holds them all.
"""

MatchOutcome = Literal["exact", "compatible", "unknown", "sole", "ambiguous", "none"]

_LINKED_OUTCOMES: frozenset[str] = frozenset({"exact", "compatible", "unknown", "sole"})


class SiblingCandidate(NamedTuple):
    """One ModelInstance written under the same :ExtractionResult."""

    uid: str
    model_class: str
    values: Mapping[str, Any]
    """Field values of the instance (raw, not normalized). Only the join
    fields of the relationship being resolved are read."""
    keyed: bool
    """True when the instance's uid is derived from its ``instance_key``
    fields (MERGE-deduplicated), False for a key-less (random uid) instance."""


class SiblingMatch(NamedTuple):
    """Result of :func:`match_sibling`."""

    outcome: MatchOutcome
    uid: str | None = None
    """uid of the sibling to link, or None when no sibling is linked."""
    candidates: int = 0
    """Number of instances of the target class considered."""

    @property
    def linked(self) -> bool:
        return self.outcome in _LINKED_OUTCOMES


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _parts(value: Any) -> set[str]:
    """Normalized parts of a value: split on JOIN_VALUE_SEPARATOR; a list
    contributes every item. Empty parts are dropped."""
    items = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    parts: set[str] = set()
    for item in items:
        if item is None:
            continue
        for piece in str(item).split(JOIN_VALUE_SEPARATOR):
            norm = normalize_key(piece)
            if norm:
                parts.add(norm)
    return parts


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_is_empty(v) for v in value)
    return str(value).strip() == ""


def compare_join_value(expected: str, candidate_value: Any) -> str:
    """Compare one join value (already normalized) with a candidate's raw value.

    Returns ``"equal"``, ``"contains"``, ``"missing"`` or ``"different"``
    (see the module docstring).
    """
    if _is_empty(candidate_value):
        return "missing"
    if not isinstance(candidate_value, (list, tuple, set, frozenset)):
        if normalize_key(str(candidate_value)) == expected:
            return "equal"
    expected_parts = _parts(expected)
    candidate_parts = _parts(candidate_value)
    if not expected_parts or not candidate_parts:
        return "different"
    if expected_parts == candidate_parts:
        return "equal"
    if expected_parts <= candidate_parts or candidate_parts <= expected_parts:
        return "contains"
    return "different"


def match_sibling(
    target_model: str,
    target_key: Mapping[str, str],
    candidates: Iterable[SiblingCandidate],
    *,
    exclude_uid: str | None = None,
    single_target: bool = True,
) -> SiblingMatch:
    """Pick the same-extraction sibling an ``instance_relationships`` entry
    refers to, if any.

    Parameters
    ----------
    target_model:
        ``target_model`` of the relationship.
    target_key:
        The key the relationship would address: ``{remote_field: normalized
        value}`` (exactly what is hashed into the shell uid).
    candidates:
        Every ModelInstance of the same :ExtractionResult (any class: only
        those of ``target_model`` are considered).
    exclude_uid:
        uid of the source instance, never its own target.
    single_target:
        True when the relationship produced one target (a scalar field, or a
        one-item list). The *sole sibling* fallback only applies then.

    Returns
    -------
    SiblingMatch
        ``outcome`` is one of ``exact`` / ``compatible`` / ``unknown`` /
        ``sole`` (a sibling is linked, ``uid`` set), ``ambiguous`` (several
        equally good siblings, none linked) or ``none`` (no usable sibling).
    """
    # One candidate per node: a keyed instance repeated in the extraction is
    # MERGE'd onto a single uid, so it must not count as two siblings.
    pool = list(
        {
            c.uid: c
            for c in reversed(list(candidates))
            if c.model_class == target_model and c.uid != exclude_uid
        }.values()
    )[::-1]
    if not pool:
        return SiblingMatch("none", None, 0)

    tiers: dict[str, list[SiblingCandidate]] = {"exact": [], "compatible": [], "unknown": []}
    for cand in pool:
        results = [
            compare_join_value(expected, cand.values.get(field))
            for field, expected in target_key.items()
        ]
        if "different" in results:
            continue
        if results and all(r == "equal" for r in results):
            tiers["exact"].append(cand)
        elif all(r == "missing" for r in results):
            tiers["unknown"].append(cand)
        else:
            tiers["compatible"].append(cand)

    for tier in ("exact", "compatible", "unknown"):
        members = tiers[tier]
        if len(members) == 1:
            return SiblingMatch(tier, members[0].uid, len(pool))  # type: ignore[arg-type]
        if len(members) > 1:
            return SiblingMatch("ambiguous", None, len(pool))

    if single_target and len(pool) == 1 and not pool[0].keyed:
        return SiblingMatch("sole", pool[0].uid, 1)
    return SiblingMatch("none", None, len(pool))


# ---------------------------------------------------------------------------
# Declaration analysis
# ---------------------------------------------------------------------------


def _extra(field_info: Any) -> dict:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    return extra if isinstance(extra, dict) else {}


def instance_key_fields(cls: type) -> set[str]:
    """Names of the fields of *cls* marked ``instance_key: True``."""
    fields = getattr(cls, "model_fields", None) or {}
    return {name for name, info in fields.items() if _extra(info).get("instance_key")}


def iter_instance_relationships(cls: type) -> Iterable[tuple[str, dict]]:
    """Yield ``(field_name, rel_def)`` for every ``instance_relationships``
    entry declared on *cls*."""
    fields = getattr(cls, "model_fields", None) or {}
    for field_name, info in fields.items():
        for rel_def in _extra(info).get("instance_relationships", None) or []:
            if isinstance(rel_def, dict):
                yield field_name, rel_def


def join_key_fields(cls: type, models: Iterable[type] = ()) -> set[str]:
    """Fields of *cls* whose value is used as an identity or join key.

    - its ``instance_key`` fields;
    - its own ``instance_relationships`` fields and the local fields of their
      ``join_via``;
    - its fields named as remote fields in the ``join_via`` of any
      relationship (declared on any class of *models*, or on *cls* itself)
      whose ``target_model`` is *cls*.

    Packing several source values into one of these fields changes the key it
    is compared on, which is why the tabular pipeline warns about it.
    """
    keys = set(instance_key_fields(cls))
    for field_name, rel_def in iter_instance_relationships(cls):
        keys.add(field_name)
        keys.update((rel_def.get("join_via") or {}).keys())
    name = getattr(cls, "__name__", "")
    for model in {cls, *models}:
        for _field_name, rel_def in iter_instance_relationships(model):
            if rel_def.get("target_model") == name:
                keys.update((rel_def.get("join_via") or {}).values())
    fields = getattr(cls, "model_fields", None) or {}
    return {k for k in keys if k in fields}


def check_instance_relationships(models: Mapping[str, type]) -> list[str]:
    """Static checks of every ``instance_relationships`` declaration.

    Returns one human-readable message per problem (empty when all is well):

    - the ``target_model`` is not a known model;
    - a ``join_via`` local field does not exist on the source model;
    - a ``join_via`` remote field does not exist on the target model;
    - a ``join_via`` remote field is not ``instance_key`` on the target model:
      the shell the relationship creates for a cross-document reference can
      then never merge with the real instance (a same-extraction sibling is
      still linked, see :func:`match_sibling`).
    """
    problems: list[str] = []
    for src_name, cls in models.items():
        src_fields = getattr(cls, "model_fields", None) or {}
        for field_name, rel_def in iter_instance_relationships(cls):
            where = f"{src_name}.{field_name}"
            target = rel_def.get("target_model")
            join_via = rel_def.get("join_via") or {}
            if not target or not join_via:
                problems.append(f"{where}: instance_relationships entry without target_model/join_via")
                continue
            for local in join_via:
                if local not in src_fields:
                    problems.append(f"{where}: join_via local field {local!r} not on {src_name}")
            tgt_cls = models.get(target)
            if tgt_cls is None:
                problems.append(f"{where}: target_model {target!r} is not a known model")
                continue
            tgt_fields = getattr(tgt_cls, "model_fields", None) or {}
            tgt_keys = instance_key_fields(tgt_cls)
            for remote in join_via.values():
                if remote not in tgt_fields:
                    problems.append(
                        f"{where}: join_via remote field {remote!r} not on {target}"
                    )
                elif remote not in tgt_keys:
                    problems.append(
                        f"{where}: join_via remote field {target}.{remote} is not "
                        f"instance_key — cross-document references to {target} create "
                        f"shells that never merge with the real instance"
                    )
    return problems


def warn_instance_relationships(models: Mapping[str, type]) -> list[str]:
    """Log :func:`check_instance_relationships` problems as warnings (once per
    call) and return them."""
    problems = check_instance_relationships(models)
    for problem in problems:
        log.warning("instance_relationships: %s", problem)
    return problems


def multi_column_join_fields(
    mappings: Sequence[Any],
    classes: Mapping[str, type],
    models: Iterable[type] = (),
) -> dict[tuple[str, str], list[str]]:
    """Join-key fields that a tabular column mapping fills from more than one column.

    Parameters
    ----------
    mappings:
        ``ColumnMapping.mappings`` (objects with ``column_name``,
        ``model_field_name``, ``target_model`` and ``confidence``). Entries
        the row router skips (``low`` confidence, ``__extra__``) are ignored.
    classes:
        ``{target_model value: class}`` — ``"primary"`` and every
        complementary class name.
    models:
        Every model whose relationships may target these classes (see
        :func:`join_key_fields`).

    Returns
    -------
    dict
        ``{(target_model, field_name): [column, ...]}`` for each join-key
        field that receives two or more columns.
    """
    models = list(models)
    columns: dict[tuple[str, str], list[str]] = {}
    for m in mappings:
        if m.confidence == "low" or m.model_field_name == "__extra__":
            continue
        cols = columns.setdefault((m.target_model, m.model_field_name), [])
        if m.column_name not in cols:
            cols.append(m.column_name)
    key_cache: dict[str, set[str]] = {}
    result: dict[tuple[str, str], list[str]] = {}
    for (target, field_name), cols in columns.items():
        if len(cols) < 2 or target not in classes:
            continue
        if target not in key_cache:
            key_cache[target] = join_key_fields(classes[target], models)
        if field_name in key_cache[target]:
            result[(target, field_name)] = cols
    return result
