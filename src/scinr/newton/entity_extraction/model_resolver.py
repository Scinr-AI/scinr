"""
entity_extraction/model_resolver.py

Builds a flat {class_name: class} registry of every Pydantic ExtractionModel
reachable from any theme registered in models/. Discovery is fully automatic:
  1. All SELECTABLE_MODELS from every registered theme are seeded via ThemeRegistry.
  2. BFS expands all nested Pydantic sub-models transitively.

No manual registry files are needed. Adding a new theme only requires creating
models/<theme>/catalog.py with SELECTABLE_MODELS.

Usage:
    from entity_extraction.model_resolver import resolve_model_class
    cls = resolve_model_class("DrugProductComposition")
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type] | None = None


def _build_registry() -> dict[str, type]:
    """
    Discover all Pydantic BaseModel subclasses reachable from any theme:
      1. Seed with all SELECTABLE_MODELS from every theme in ThemeRegistry
      2. BFS-expand to include all nested Pydantic sub-models transitively

    Returns a flat dict {class_name: class}.
    """
    from scinr.newton.utils.theme_registry import get_theme_registry

    registry: dict[str, type] = {}
    theme_registry = get_theme_registry()

    for theme_path, theme_node in theme_registry._themes.items():
        for cls in theme_node.models:
            if isinstance(cls, type) and issubclass(cls, BaseModel):
                name = cls.__name__
                if name in registry and registry[name] is not cls:
                    log.warning(
                        "model_resolver: duplicate class name '%s' — "
                        "found in theme '%s' but already registered from a different theme. "
                        "Last definition wins.",
                        name, theme_path,
                    )
                registry[name] = cls
                log.debug("model_resolver: registered %s from theme '%s'", name, theme_path)

    # BFS to include all nested Pydantic models referenced from field annotations
    queue = list(registry.values())
    visited: set[str] = set(registry.keys())
    while queue:
        cls = queue.pop()
        if not hasattr(cls, "model_fields"):
            continue
        for field_info in cls.model_fields.values():
            for ref_cls in _extract_model_classes(field_info.annotation):
                ref_name = ref_cls.__name__
                if ref_name not in visited:
                    visited.add(ref_name)
                    registry[ref_name] = ref_cls
                    queue.append(ref_cls)

    log.info("model_resolver: registry built with %d classes", len(registry))
    return registry


def _extract_model_classes(annotation: object) -> list[type]:
    """Recursively extract all Pydantic BaseModel subclasses from a type annotation."""
    import types as _builtin_types
    import typing

    results: list[type] = []
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        return _extract_model_classes(args[0]) if args else []

    if origin is list:
        return _extract_model_classes(args[0]) if args else []

    is_union = origin is typing.Union
    if not is_union and hasattr(_builtin_types, "UnionType"):
        is_union = isinstance(annotation, _builtin_types.UnionType)
    if is_union:
        for arg in args:
            if arg is not type(None):
                results.extend(_extract_model_classes(arg))
        return results

    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
    except TypeError:
        pass

    return results


def _get_registry() -> dict[str, type]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def reset_model_resolver() -> None:
    """Force registry rebuild. Called after configure() or in tests."""
    global _REGISTRY
    _REGISTRY = None


def resolve_model_class(class_name: str) -> type:
    """
    Return the Pydantic class for *class_name*.

    Raises
    ------
    ModelError
        If the class is not found in the registry.
    """
    from scinr.newton.exceptions import ModelError

    registry = _get_registry()
    if class_name not in registry:
        raise ModelError(
            f"model_resolver: class '{class_name}' not found in registry. "
            f"Available: {sorted(registry.keys())}"
        )
    return registry[class_name]
