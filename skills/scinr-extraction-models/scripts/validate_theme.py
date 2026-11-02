#!/usr/bin/env python3
"""Static checks for scinr.newton extraction-model themes.

Usage:
    python validate_theme.py <models_root> [--tabular] [--registry]

<models_root> is the directory you pass to configure(extra_models_paths=[...]):
every folder below it that holds a catalog.py is a theme.

  --tabular   the models will also be used with the tabular (CSV/XLSX) pipeline;
              enables the tabular-only rules.
  --registry  after the static checks, build scinr's real ThemeRegistry against
              <models_root> and print the theme paths it discovers.

Exit code 0 when there are no ERROR lines (WARN lines do not fail the run).
Needs only pydantic and the theme's own imports (usually scinr) to be installed.
"""

from __future__ import annotations

import argparse
import enum
import importlib
import importlib.util
import inspect
import re
import sys
import types
import typing
from pathlib import Path

from pydantic import BaseModel

UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")
SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules"}

errors: list[str] = []
warnings: list[str] = []


def err(where: str, msg: str) -> None:
    errors.append(f"ERROR {where}: {msg}")


def warn(where: str, msg: str) -> None:
    warnings.append(f"WARN  {where}: {msg}")


# ── discovery / import (mirrors ThemeRegistry._load_catalog_from_path) ────────


def find_catalogs(root: Path) -> list[Path]:
    found: list[Path] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name.startswith(("_", ".")) or item.name in SKIP_DIRS:
            continue
        if (item / "catalog.py").exists():
            found.append(item / "catalog.py")
        found.extend(find_catalogs(item))
    return found


def check_init_files(root: Path) -> None:
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        if any(part.startswith((".", "_")) or part in SKIP_DIRS for part in d.relative_to(root).parts):
            continue
        if any(d.glob("*.py")) and not (d / "__init__.py").exists():
            err(str(d), "folder contains .py files but no __init__.py — create an empty one")


def import_catalog(catalog: Path, theme_path: str) -> types.ModuleType | None:
    parts = ["catalog"]
    current = catalog.parent
    while (current / "__init__.py").exists():
        parts.append(current.name)
        current = current.parent
    try:
        if len(parts) > 1:  # package layout
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return importlib.import_module(".".join(reversed(parts)))
        name = f"_scinr_validate.{theme_path.replace('/', '.')}.catalog"
        spec = importlib.util.spec_from_file_location(name, catalog)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[name] = module
        sys.path.insert(0, str(catalog.parent))
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            sys.path.remove(str(catalog.parent))
        return module
    except Exception as exc:  # noqa: BLE001
        err(str(catalog), f"import failed: {type(exc).__name__}: {exc}")
        return None


# ── type helpers ─────────────────────────────────────────────────────────────


def unwrap_optional(tp: typing.Any) -> typing.Any:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def is_model(tp: typing.Any) -> bool:
    return inspect.isclass(tp) and issubclass(tp, BaseModel)


def list_item(tp: typing.Any) -> typing.Any | None:
    tp = unwrap_optional(tp)
    if typing.get_origin(tp) is list:
        args = typing.get_args(tp)
        return args[0] if args else typing.Any
    return None


def nested_models(tp: typing.Any) -> list[type]:
    tp = unwrap_optional(tp)
    item = list_item(tp)
    if item is not None:
        tp = unwrap_optional(item)
    if is_model(tp):
        return [tp]
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        return [a for a in typing.get_args(tp) if is_model(a)]
    return []


def enum_types(tp: typing.Any) -> list[type]:
    tp = unwrap_optional(tp)
    item = list_item(tp)
    if item is not None:
        tp = unwrap_optional(item)
    return [tp] if inspect.isclass(tp) and issubclass(tp, enum.Enum) else []


def extra(info: typing.Any) -> dict:
    e = getattr(info, "json_schema_extra", None) or {}
    return e if isinstance(e, dict) else {}


def words(text: str | None) -> int:
    return len((text or "").split())


def is_list_wrapper(cls: type) -> type | None:
    fields = cls.model_fields
    if len(fields) != 1:
        return None
    item = list_item(next(iter(fields.values())).annotation)
    item = unwrap_optional(item) if item is not None else None
    return item if is_model(item) else None


# ── per-model checks ─────────────────────────────────────────────────────────


def check_model(cls: type, tabular: bool, top_level: bool) -> None:
    name = cls.__name__
    doc = inspect.getdoc(cls) if cls.__doc__ else None
    if not doc:
        err(name, "class has no docstring — the annotation LLM reads it to pick the model")
    else:
        first = doc.strip().splitlines()[0]
        if words(first) > 15:
            warn(name, f"docstring first line is {words(first)} words (max 15); move details to later lines")

    cfg = getattr(cls, "model_config", {}) or {}
    if cfg.get("extra") != "forbid":
        warn(name, "model_config lacks extra='forbid' — inherit from scinr.newton.models.base.ExtractionModel")

    if tabular and top_level:
        for attr in ("__pydantic_decorators__",):
            decs = getattr(cls, attr, None)
            if decs and (decs.field_validators or decs.model_validators):
                warn(name, "has validators, but the tabular pipeline builds rows with model_construct() — "
                           "they never run there; use a normalization_model submodel instead")

    fields = cls.model_fields
    for fname, info in fields.items():
        where = f"{name}.{fname}"
        ann = info.annotation
        x = extra(info)

        if not info.description:
            err(where, "no description= — the extraction LLM's primary signal")
        elif words(info.description) < 15:
            warn(where, f"description is {words(info.description)} words (aim for ≥15: what, format with examples, when None)")

        if list_item(ann) is not None and isinstance(info.default, list):
            err(where, "list field uses default=[...]; use default_factory=list")

        for et in enum_types(ann):
            if not issubclass(et, str):
                err(where, f"enum {et.__name__} must subclass (str, Enum)")
            elif not inspect.getdoc(et) or "Values:" not in (et.__doc__ or ""):
                warn(where, f"enum {et.__name__} docstring should have a 'Values:' section documenting each member")

        if x.get("entity_label") and (list_item(ann) is None) and unwrap_optional(ann) in (bool, int, float):
            warn(where, "entity_label on a bool/number — dedup is meaningless; keep entity_label for identifiers")

        if x.get("instance_key") and list_item(ann) is not None:
            err(where, "instance_key on a list field — a key must hold exactly one value")

        for fr in x.get("field_relationships") or []:
            to = fr.get("to_field")
            if to not in fields:
                err(where, f"field_relationships to_field {to!r} is not a field of {name}")
            else:
                if not x.get("entity_label"):
                    err(where, "has field_relationships but no entity_label of its own")
                if not extra(fields[to]).get("entity_label"):
                    err(where, f"field_relationships target {to!r} has no entity_label — edge is silently skipped")
            if not UPPER_SNAKE.match(fr.get("rel_type") or ""):
                err(where, f"rel_type {fr.get('rel_type')!r} must be UPPER_SNAKE_CASE")

        for ir in x.get("instance_relationships") or []:
            if not UPPER_SNAKE.match(ir.get("rel_type") or ""):
                err(where, f"rel_type {ir.get('rel_type')!r} must be UPPER_SNAKE_CASE")

        if x.get("normalization_model"):
            if not nested_models(ann):
                err(where, "normalization_model: True must be on a nested-submodel field")
            src = x.get("normalization_source_fields")
            if not src:
                err(where, "normalization_model without explicit normalization_source_fields — "
                           "engine would silently send ALL other scalar fields to the LLM")
            else:
                for s in src:
                    if s not in fields:
                        err(where, f"normalization_source_fields entry {s!r} is not a field of {name}")
        elif tabular and top_level:
            base = unwrap_optional(ann)
            item = list_item(ann)
            ok = base is str or (item is not None and unwrap_optional(item) is str)
            if not ok:
                err(where, f"tabular models need str or list[str] fields (got {ann!r}); wrap typed values in a "
                           "normalization_model submodel")


def check_instance_relationships(models: dict[str, type]) -> None:
    for src_name, cls in models.items():
        fields = cls.model_fields
        for fname, info in fields.items():
            for rel in extra(info).get("instance_relationships") or []:
                where = f"{src_name}.{fname}"
                target, join_via = rel.get("target_model"), rel.get("join_via") or {}
                if not target or not join_via:
                    err(where, "instance_relationships entry needs target_model and join_via")
                    continue
                for local in join_via:
                    if local not in fields:
                        err(where, f"join_via local field {local!r} is not a field of {src_name} (exact Python name)")
                tgt = models.get(target)
                if tgt is None:
                    err(where, f"target_model {target!r} is not a model in this models root "
                               f"(known: {', '.join(sorted(models))})")
                    continue
                for remote in join_via.values():
                    if remote not in tgt.model_fields:
                        err(where, f"join_via target field {remote!r} is not a field of {target}")
                    elif not extra(tgt.model_fields[remote]).get("instance_key"):
                        err(where, f"{target}.{remote} must be marked instance_key: True or shells never merge")


def collect(cls: type, seen: dict[str, type]) -> None:
    if cls.__name__ in seen:
        return
    seen[cls.__name__] = cls
    for info in cls.model_fields.values():
        for sub in nested_models(info.annotation):
            collect(sub, seen)


# ── main ─────────────────────────────────────────────────────────────────────


REGISTRY_PROBE = """
import logging, sys
logging.disable(logging.INFO)
from scinr.newton.config import configure
from scinr.newton.utils.theme_registry import get_theme_registry
configure(neo4j_user="x", neo4j_password="x", neo4j_database="x", extra_models_paths=[sys.argv[1]])
reg = get_theme_registry()
for path in reg.get_all_theme_paths():
    print(f"{path}\\t{len(reg.find_best_theme(path).models)}")
"""


def run_registry(root: Path, expected: set[str]) -> None:
    """Build the real ThemeRegistry in a fresh interpreter (same cwd / PYTHONPATH as
    the caller, none of this script's sys.path tweaks), so an un-importable package
    root shows up the way it would in the user's pipeline run."""
    import subprocess

    proc = subprocess.run([sys.executable, "-c", REGISTRY_PROBE, str(root)], capture_output=True, text=True)
    if proc.returncode != 0:
        err("--registry", f"probe failed:\n{proc.stderr.strip()}")
        return
    found = dict(line.split("\t") for line in proc.stdout.strip().splitlines() if "\t" in line)
    for theme in sorted(expected):
        n = int(found.get(theme, "0"))
        if theme not in found:
            err("--registry", f"theme {theme!r} not discovered by ThemeRegistry")
        elif n == 0:
            err("--registry", f"theme {theme!r} discovered but loaded 0 models — its catalog failed to import in "
                              "a fresh process. For package layouts the folder ABOVE the top __init__.py must be "
                              "on sys.path (run from there or set PYTHONPATH). Probe stderr:\n" + proc.stderr.strip())
        else:
            print(f"ThemeRegistry OK: {theme!r} → {n} selectable model(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models_root", type=Path)
    ap.add_argument("--tabular", action="store_true")
    ap.add_argument("--registry", action="store_true")
    args = ap.parse_args()
    root = args.models_root.resolve()
    if not root.is_dir():
        print(f"ERROR {root}: not a directory", file=sys.stderr)
        return 2

    check_init_files(root)
    catalogs = find_catalogs(root)
    if not catalogs:
        err(str(root), "no catalog.py found below this folder — each theme folder needs one")

    all_models: dict[str, type] = {}
    top_level: set[str] = set()
    for catalog in catalogs:
        theme = catalog.parent.relative_to(root).as_posix()
        module = import_catalog(catalog, theme)
        if module is None:
            continue
        desc = getattr(module, "THEME_DESCRIPTION", None)
        if not isinstance(desc, str) or not desc.strip():
            err(theme, "catalog.py has no THEME_DESCRIPTION string")
        elif words(desc) < 20:
            warn(theme, f"THEME_DESCRIPTION is {words(desc)} words — name document types, standards, "
                        "example entities, and what it is distinct from")
        selectable = getattr(module, "SELECTABLE_MODELS", None)
        if not isinstance(selectable, list) or not selectable:
            err(theme, "catalog.py needs a non-empty SELECTABLE_MODELS list")
            continue
        for cls in selectable:
            if not is_model(cls):
                err(theme, f"SELECTABLE_MODELS item {cls!r} is not a pydantic BaseModel subclass")
                continue
            top_level.add(cls.__name__)
            collect(cls, all_models)
            item = is_list_wrapper(cls)
            if item is not None and item not in selectable:
                warn(theme, f"{cls.__name__} wraps {item.__name__} but {item.__name__} is not in SELECTABLE_MODELS")

    for name, cls in all_models.items():
        row_model = name in top_level and is_list_wrapper(cls) is None
        check_model(cls, tabular=args.tabular, top_level=row_model)
    check_instance_relationships(all_models)
    if args.registry and not errors:
        run_registry(root, {c.parent.relative_to(root).as_posix() for c in catalogs})

    for line in errors + warnings:
        print(line)
    print(f"\n{len(catalogs)} theme(s), {len(all_models)} model(s): {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
