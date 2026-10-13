"""
entity_extraction/key_fields.py

Which fields of a model carry identity, and static checks of the
``instance_relationships`` declarations that depend on them.

A ModelInstance with ``instance_key`` fields gets a deterministic uid
(``make_instance_uid(model_class, {key: normalize_key(value)}, tenant)``), so
an ``instance_relationships`` reference that carries the same values computes
the same uid and lands on the same node, whatever the load order. Two things
silently break that:

- the target's join field is not ``instance_key``: the real instance gets a
  random uid and the reference's shell never merges with it
  (:func:`check_instance_relationships`, warned at catalog setup);
- a key field holds several values packed into one (the tabular pipeline
  used to join several mapped columns with ``"; "``): the uid is computed
  from the packed string, which no reference carries (:func:`key_fields`,
  used by the tabular row router to keep a single value).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

log = logging.getLogger(__name__)


def _extra(field_info: Any) -> dict:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    return extra if isinstance(extra, dict) else {}


def _fields(cls: type) -> dict:
    return getattr(cls, "model_fields", None) or {}


def instance_key_fields(cls: type) -> set[str]:
    """Fields of *cls* marked ``instance_key: True``."""
    return {name for name, info in _fields(cls).items() if _extra(info).get("instance_key")}


def iter_instance_relationships(cls: type) -> Iterable[tuple[str, dict]]:
    """``(field_name, rel_def)`` for every ``instance_relationships`` entry of *cls*."""
    for field_name, info in _fields(cls).items():
        for rel_def in _extra(info).get("instance_relationships") or []:
            if isinstance(rel_def, dict):
                yield field_name, rel_def


def key_fields(cls: type) -> set[str]:
    """Fields of *cls* whose value is (part of) a key: its ``instance_key``
    fields and the local fields of its ``instance_relationships`` (the field
    carrying the declaration and every ``join_via`` key). Each must hold a
    single value — never several source values joined into one string."""
    keys = instance_key_fields(cls)
    for field_name, rel_def in iter_instance_relationships(cls):
        keys.add(field_name)
        keys.update((rel_def.get("join_via") or {}).keys())
    return keys & set(_fields(cls))


def check_instance_relationships(models: Mapping[str, type]) -> list[str]:
    """One message per problem in the ``instance_relationships`` declarations
    of *models* (``{class_name: class}``); empty when all is well:

    - incomplete entry (no ``target_model`` / ``join_via``);
    - ``join_via`` local field missing on the source model;
    - unknown ``target_model``;
    - ``join_via`` remote field missing on the target model;
    - ``join_via`` remote field not ``instance_key`` on the target model —
      the reference's shell can then never merge with the real instance.
    """
    problems: list[str] = []
    for src_name, cls in models.items():
        src_fields = _fields(cls)
        for field_name, rel_def in iter_instance_relationships(cls):
            where = f"{src_name}.{field_name}"
            target = rel_def.get("target_model")
            join_via = rel_def.get("join_via") or {}
            if not target or not join_via:
                problems.append(f"{where}: entry without target_model/join_via")
                continue
            for local in join_via:
                if local not in src_fields:
                    problems.append(f"{where}: join_via field {local!r} not on {src_name}")
            tgt_cls = models.get(target)
            if tgt_cls is None:
                problems.append(f"{where}: target_model {target!r} is not a known model")
                continue
            tgt_keys = instance_key_fields(tgt_cls)
            for remote in join_via.values():
                if remote not in _fields(tgt_cls):
                    problems.append(f"{where}: join_via target field {remote!r} not on {target}")
                elif remote not in tgt_keys:
                    problems.append(
                        f"{where}: join_via target field {target}.{remote} is not "
                        f"instance_key — {target} instances get a random uid, so the "
                        f"shell this relationship points at never merges with them "
                        f"(mark {target}.{remote} with \"instance_key\": True)"
                    )
    return problems


def warn_instance_relationships(models: Mapping[str, type]) -> list[str]:
    """Log every :func:`check_instance_relationships` problem as a warning."""
    problems = check_instance_relationships(models)
    for problem in problems:
        log.warning("instance_relationships: %s", problem)
    return problems
