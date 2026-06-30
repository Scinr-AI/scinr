"""
ThemeRegistry — dynamic discovery and management of annotation themes.

Scans the models/ directory for theme folders (those containing a catalog.py),
builds a tree of ThemeNode objects, and provides catalog lookup and Neo4j export.

Convention for catalog.py files:
    THEME_DESCRIPTION: str         — one-line description for LLM classification
    SELECTABLE_MODELS: list[type]  — Pydantic classes usable as annotation targets

Usage:
    from utils.theme_registry import get_theme_registry
    registry = get_theme_registry()
    theme = registry.find_best_theme("structural_specs")
    catalog_block = registry.build_catalog_block(theme)

Custom user models:
    Pass extra_models_paths to configure() to scan additional directories:
        configure(llm=..., extra_models_paths=["/path/to/my/models"])
    User themes override built-in themes with the same name (warning is emitted).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ThemeNode:
    """Represents one thematic domain discovered from a models/ subfolder."""

    path: str                              # e.g. "pharmaceutical" or "structural_specs/nta"
    name: str                              # folder name, e.g. "pharmaceutical"
    description: str                       # from THEME_DESCRIPTION in catalog.py
    models: list[type]                     # from SELECTABLE_MODELS in catalog.py
    children: dict[str, ThemeNode] = field(default_factory=dict)

    def __repr__(self) -> str:
        model_names = [m.__name__ for m in self.models]
        return f"ThemeNode(path={self.path!r}, models={model_names})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ThemeRegistry:
    """
    Scans models/ recursively to discover theme folders and build a theme tree.

    A folder is a theme iff it contains a catalog.py.
    Nested theme folders (sub-themes) are supported to any depth.
    """

    def __init__(
        self,
        models_root: Path,
        extra_models_roots: list[Path] | None = None,
        enabled_base_themes: list[str] | None = None,
        enabled_user_themes: list[str] | None = None,
    ) -> None:
        self.models_root = models_root
        self._extra_roots: list[Path] = list(extra_models_roots or [])
        # Flat map: theme path → ThemeNode  (e.g. "structural_specs" → ThemeNode)
        self._themes: dict[str, ThemeNode] = {}
        # Track which theme paths came from user roots (for split filtering)
        self._user_theme_paths: set[str] = set()
        # Top-level theme name → ThemeNode
        self._root_themes: dict[str, ThemeNode] = self._scan(models_root, prefix="")


        if enabled_base_themes is not None:
            self._apply_enabled_filter(enabled_base_themes, scope="builtin")

        self._discover_external_packages()
        self._scan_extra_roots()
        
        if enabled_user_themes is not None:
            self._apply_enabled_filter(enabled_user_themes, scope="user")

        log.info(
            "ThemeRegistry: discovered %d themes: %s",
            len(self._themes),
            sorted(self._themes.keys()),
        )
    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self, directory: Path, prefix: str) -> dict[str, ThemeNode]:
        """
        Recursively scan *directory* for theme folders.

        Returns a dict of {folder_name: ThemeNode} for the immediate children
        of *directory* that are themes (or whose descendants are themes).
        """
        result: dict[str, ThemeNode] = {}

        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            log.debug(
                "ThemeRegistry: permission denied scanning '%s' — skipping.", directory
            )
            return result
        except FileNotFoundError:
            if directory == self.models_root:
                from scinr.newton.exceptions import ConfigurationError
                raise ConfigurationError(
                    f"Models directory '{directory}' does not exist. "
                    f"Ensure the package is installed correctly or create the directory."
                )
            return result

        for item in entries:
            if not item.is_dir():
                continue
            # Skip Python internals and hidden directories
            if item.name.startswith("_") or item.name.startswith("."):
                continue
            # Skip virtual environment directories
            if item.name in (".venv", "venv", "__pycache__", "node_modules"):
                continue

            theme_path = f"{prefix}{item.name}" if prefix else item.name
            catalog_file = item / "catalog.py"

            # Recurse into children regardless of whether this folder has catalog.py
            sub_prefix = f"{theme_path}/"
            children = self._scan(item, sub_prefix)

            if catalog_file.exists():
                # This folder IS a theme
                models, description = self._load_catalog(theme_path)
                node = ThemeNode(
                    path=theme_path,
                    name=item.name,
                    description=description,
                    models=models,
                    children=children,
                )
                result[item.name] = node
                self._themes[theme_path] = node
                log.debug(
                    "ThemeRegistry: registered theme '%s' with %d models",
                    theme_path,
                    len(models),
                )
            else:
                # Not a theme itself, but propagate discovered children upward
                result.update(children)

        return result

    def _discover_external_packages(self) -> None:
        """Discover model packages registered via Python entry points."""
        try:
            import importlib.metadata
            eps = importlib.metadata.entry_points(group="scinr.newton.models")
        except Exception:
            return
        for ep in eps:
            try:
                import importlib as _importlib
                module = _importlib.import_module(ep.value)
                pkg_root = Path(module.__file__).parent
                external_themes = self._scan(pkg_root, prefix="")
                for name, node in external_themes.items():
                    if name in self._themes:
                        log.warning(
                            "ThemeRegistry: external package '%s' defines theme '%s' "
                            "which already exists in built-in models. "
                            "Built-in theme takes precedence. "
                            "Rename the external theme to avoid conflicts.",
                            ep.name, name,
                        )
                    else:
                        self._themes[name] = node
            except Exception as exc:
                log.warning(
                    "ThemeRegistry: failed to load external model package '%s': %s",
                    ep.name, exc,
                )

    def _scan_extra_roots(self) -> None:
        """Scan user-supplied extra model directories.

        User themes take precedence over built-in themes with the same name,
        but a warning is emitted so the user is aware of the override.
        """
        for extra_root in self._extra_roots:
            if not extra_root.exists():
                log.warning(
                    "ThemeRegistry: extra_models_path '%s' does not exist — skipping.",
                    extra_root,
                )
                continue
            extra_themes = self._scan_external(extra_root, prefix="")
            for theme_path, node in extra_themes.items():
                if theme_path in self._themes and theme_path not in self._user_theme_paths:
                    log.warning(
                        "ThemeRegistry: user theme '%s' from '%s' overrides a built-in theme "
                        "with the same name. To use the built-in instead, remove '%s' from "
                        "extra_models_paths or rename your theme folder.",
                        theme_path, extra_root, theme_path,
                    )
                self._themes[theme_path] = node
                self._user_theme_paths.add(theme_path)
                log.debug(
                    "ThemeRegistry: registered user theme '%s' from '%s'",
                    theme_path, extra_root,
                )

    def _scan_external(self, directory: Path, prefix: str) -> dict[str, ThemeNode]:
        """
        Recursively scan *directory* for theme folders, loading catalogs via
        importlib.util.spec_from_file_location (works for any filesystem path,
        not just installed packages).
        """
        result: dict[str, ThemeNode] = {}

        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, FileNotFoundError):
            return result

        for item in entries:
            if not item.is_dir():
                continue
            if item.name.startswith("_") or item.name.startswith("."):
                continue
            if item.name in (".venv", "venv", "__pycache__", "node_modules"):
                continue

            theme_path = f"{prefix}{item.name}" if prefix else item.name
            catalog_file = item / "catalog.py"

            sub_prefix = f"{theme_path}/"
            children = self._scan_external(item, sub_prefix)

            if catalog_file.exists():
                models, description = self._load_catalog_from_path(theme_path, catalog_file)
                node = ThemeNode(
                    path=theme_path,
                    name=item.name,
                    description=description,
                    models=models,
                    children=children,
                )
                result[item.name] = node
                self._themes[theme_path] = node
            else:
                result.update(children)

        return result

    def _load_catalog_from_path(self, theme_path: str, catalog_file: Path) -> tuple[list[type], str]:
        """
        Load catalog.py from an arbitrary filesystem path.

        Two layouts are supported:

        **Package layout** (directory tree has ``__init__.py`` files):
            The catalog is imported via ``importlib.import_module`` using its
            fully-qualified dotted name (e.g.
            ``own_models.pharma_regulatory.variation_guidelines.catalog``).
            Relative imports inside the catalog and its siblings (``from .models
            import ...``, ``from ..baseModels import ...``) resolve correctly
            because Python's normal package machinery handles them.
            The package root directory must already be on ``sys.path`` — this is
            guaranteed because the user passed it (or a parent of it) as
            ``extra_models_paths`` to ``configure()``.

        **Standalone layout** (no ``__init__.py``):
            The catalog is loaded via ``importlib.util.spec_from_file_location``.
            The catalog's own directory is temporarily added to ``sys.path`` so
            that bare sibling imports (``from sibling import ...``) resolve.
        """
        import importlib
        import importlib.util
        import sys
        _log = logging.getLogger(__name__)

        # ------------------------------------------------------------------
        # Detect layout: walk up from catalog_file looking for __init__.py
        # ------------------------------------------------------------------
        def _find_package_info(path: Path) -> tuple[Path, str] | None:
            """
            Return (package_root, dotted_module_name) if *path* lives inside a
            package (i.e. its parent directory has __init__.py), else None.
            """
            parts: list[str] = ["catalog"]
            current = path.parent
            while (current / "__init__.py").exists():
                parts.append(current.name)
                current = current.parent
            if len(parts) == 1:
                return None  # no __init__.py found → standalone layout
            parts.reverse()
            dotted = ".".join(parts)
            return current, dotted  # current = first dir without __init__.py

        layout = _find_package_info(catalog_file)

        try:
            if layout is not None:
                # ── Package layout ────────────────────────────────────────
                _package_root, module_name = layout
                if module_name in sys.modules:
                    module = importlib.reload(sys.modules[module_name])
                else:
                    module = importlib.import_module(module_name)
            else:
                # ── Standalone layout ─────────────────────────────────────
                module_name = f"_scinr_user_models.{theme_path.replace('/', '.')}.catalog"
                catalog_dir = str(catalog_file.parent)
                spec = importlib.util.spec_from_file_location(module_name, catalog_file)
                if spec is None or spec.loader is None:
                    _log.warning(
                        "ThemeRegistry: cannot create module spec for '%s'. "
                        "Theme '%s' will not be available.",
                        catalog_file, theme_path,
                    )
                    return [], theme_path
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                sys.path.insert(0, catalog_dir)
                try:
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
                finally:
                    if catalog_dir in sys.path:
                        sys.path.remove(catalog_dir)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            _log.warning(
                "ThemeRegistry: failed to load catalog '%s': %s. "
                "Theme '%s' will not be available.",
                catalog_file, exc, theme_path,
            )
            return [], theme_path

        # ------------------------------------------------------------------
        # Extract SELECTABLE_MODELS
        # ------------------------------------------------------------------
        raw_models = getattr(module, "SELECTABLE_MODELS", None)
        if raw_models is None:
            _log.warning(
                "ThemeRegistry: '%s' has no SELECTABLE_MODELS. "
                "Theme '%s' will be discoverable but no models can be selected for it.",
                catalog_file, theme_path,
            )
            raw_models = []
        elif not isinstance(raw_models, list):
            _log.warning(
                "ThemeRegistry: SELECTABLE_MODELS in '%s' is not a list (got %s). "
                "Treating as empty.",
                catalog_file, type(raw_models).__name__,
            )
            raw_models = []

        from pydantic import BaseModel as PydanticBaseModel

        from scinr.newton.exceptions import ConfigurationError
        models: list[type] = []
        for cls in raw_models:
            if not isinstance(cls, type):
                _log.warning(
                    "ThemeRegistry: item %r in SELECTABLE_MODELS of '%s' is not a class. Skipping.",
                    cls, catalog_file,
                )
                continue
            if not issubclass(cls, PydanticBaseModel):
                raise ConfigurationError(
                    f"ThemeRegistry: '{cls.__name__}' in SELECTABLE_MODELS of '{catalog_file}' "
                    f"is not a subclass of pydantic.BaseModel."
                )
            models.append(cls)

        # ------------------------------------------------------------------
        # Extract THEME_DESCRIPTION
        # ------------------------------------------------------------------
        description = getattr(module, "THEME_DESCRIPTION", None)
        if not description or not isinstance(description, str) or not description.strip():
            _log.warning(
                "ThemeRegistry: '%s' has no valid THEME_DESCRIPTION. Using theme path as description.",
                catalog_file,
            )
            description = theme_path

        return models, description

    def _apply_enabled_filter(
        self, enabled: list[str], scope: Literal["builtin", "user"]
    ) -> None:
        """Apply a whitelist filter to either built-in or user themes.

        Parameters
        ----------
        enabled:
            Whitelist of theme paths to keep within the given scope.
        scope:
            ``"builtin"`` filters only themes NOT in ``_user_theme_paths``.
            ``"user"`` filters only themes IN ``_user_theme_paths``.
        """
        from scinr.newton.exceptions import ConfigurationError

        if not enabled:
            param = "enabled_base_themes" if scope == "builtin" else "enabled_user_themes"
            raise ConfigurationError(
                f"{param} cannot be empty. Pass None to activate all themes in this group, "
                f"or include at least one valid theme path.\n"
                f"Example: configure({param}=['default'])"
            )

        # Determine which paths belong to this scope
        if scope == "builtin":
            scope_paths = {p for p in self._themes if p not in self._user_theme_paths}
        else:
            scope_paths = self._user_theme_paths.copy()

        unknown = [t for t in enabled if t not in scope_paths]
        if unknown:
            available = sorted(scope_paths)
            param = "enabled_base_themes" if scope == "builtin" else "enabled_user_themes"
            raise ConfigurationError(
                f"Unknown theme(s) in {param}: {unknown}.\n"
                f"Available {scope} themes: {available}"
            )

        # Remove paths in this scope that are not whitelisted
        for path in list(self._themes.keys()):
            if path in scope_paths and path not in enabled:
                del self._themes[path]
                self._user_theme_paths.discard(path)

    def _load_catalog(self, theme_path: str) -> tuple[list[type], str]:
        """
        Import catalog.py for *theme_path* via importlib and extract SELECTABLE_MODELS
        and THEME_DESCRIPTION.

        The module path is derived by replacing '/' with '.' and prepending 'scinr.newton.models.':
            "pharmaceutical"           → "scinr.newton.models.pharmaceutical.catalog"
            "structural_specs/nta"     → "scinr.newton.models.structural_specs.nta.catalog"
        """
        import importlib as _importlib
        _log = logging.getLogger(__name__)

        module_path = "scinr.newton.models." + theme_path.replace("/", ".") + ".catalog"
        try:
            module = _importlib.import_module(module_path)
        except ImportError as exc:
            msg = str(exc)
            if f"scinr.newton.models.{theme_path.replace('/', '.')}" in msg:
                _log.warning(
                    "ThemeRegistry: cannot import '%s' — is __init__.py missing? (%s). "
                    "Theme '%s' will not be available.",
                    module_path, exc, theme_path,
                )
            else:
                _log.warning(
                    "ThemeRegistry: import error in '%s': %s. "
                    "Theme '%s' will not be available. "
                    "Check the imports in catalog.py.",
                    module_path, exc, theme_path,
                )
            return [], theme_path

        # SELECTABLE_MODELS
        raw_models = getattr(module, "SELECTABLE_MODELS", None)
        if raw_models is None:
            _log.warning(
                "ThemeRegistry: '%s' has no SELECTABLE_MODELS. "
                "Theme '%s' will be discoverable but no models can be selected for it. "
                "Add SELECTABLE_MODELS = [YourModel, ...] to catalog.py.",
                module_path, theme_path,
            )
            raw_models = []
        elif not isinstance(raw_models, list):
            _log.warning(
                "ThemeRegistry: SELECTABLE_MODELS in '%s' is not a list (got %s). "
                "Treating as empty.",
                module_path, type(raw_models).__name__,
            )
            raw_models = []

        # Validate each class
        from pydantic import BaseModel as PydanticBaseModel

        from scinr.newton.exceptions import ConfigurationError
        models: list[type] = []
        for cls in raw_models:
            if not isinstance(cls, type):
                _log.warning(
                    "ThemeRegistry: item %r in SELECTABLE_MODELS of '%s' is not a class. "
                    "Skipping.",
                    cls, module_path,
                )
                continue
            if not issubclass(cls, PydanticBaseModel):
                raise ConfigurationError(
                    f"ThemeRegistry: '{cls.__name__}' in SELECTABLE_MODELS of '{module_path}' "
                    f"is not a subclass of pydantic.BaseModel. "
                    f"All models in SELECTABLE_MODELS must inherit from BaseModel."
                )
            models.append(cls)

        # THEME_DESCRIPTION
        description = getattr(module, "THEME_DESCRIPTION", None)
        if description is None:
            _log.warning(
                "ThemeRegistry: '%s' has no THEME_DESCRIPTION. "
                "Using theme path '%s' as description. "
                "Add THEME_DESCRIPTION = '...' to catalog.py for better LLM classification.",
                module_path, theme_path,
            )
            description = theme_path
        elif not isinstance(description, str) or not description.strip():
            _log.warning(
                "ThemeRegistry: THEME_DESCRIPTION in '%s' is empty or not a string. "
                "Using theme path '%s' as description.",
                module_path, theme_path,
            )
            description = theme_path

        return models, description

    # ------------------------------------------------------------------
    # Theme lookup
    # ------------------------------------------------------------------

    def find_best_theme(self, detected_path: str | None) -> ThemeNode:
        """
        Return the most specific available ThemeNode for *detected_path*.

        Resolution order (most to least specific):
            1. Exact match for the full detected path
            2. Progressively shorter path prefixes
            3. "default" theme (if available)
            4. First registered theme (emergency fallback)

        Examples:
            detected_path="structural_specs/pharmaceutical/ema"
            → tries "structural_specs/pharmaceutical/ema" → not found
            → tries "structural_specs/pharmaceutical"      → not found
            → tries "structural_specs"                     → FOUND, returns it

            detected_path="unknown_theme"
            → tries "unknown_theme"  → not found
            → falls back to "default"
        """
        if detected_path:
            path = detected_path.strip().strip("/")
            parts = [p for p in path.split("/") if p]
            for end in range(len(parts), 0, -1):
                candidate = "/".join(parts[:end])
                if candidate in self._themes:
                    log.debug(
                        "find_best_theme: '%s' resolved to '%s'",
                        detected_path,
                        candidate,
                    )
                    return self._themes[candidate]

        return self._get_default()

    def _get_default(self) -> ThemeNode:
        """Return the 'default' theme, or an emergency fallback if it's missing."""
        if "default" in self._themes:
            return self._themes["default"]
        if self._themes:
            fallback = next(iter(self._themes.values()))
            log.warning(
                "ThemeRegistry: 'default' theme not found, falling back to '%s'",
                fallback.path,
            )
            return fallback
        raise RuntimeError(
            "ThemeRegistry: no themes found in models directory. "
            "Ensure at least one folder with a catalog.py exists."
        )

    # ------------------------------------------------------------------
    # Catalog building
    # ------------------------------------------------------------------

    def build_catalog_block(self, theme: ThemeNode) -> str:
        """
        Build the plain-text model catalog block to inject into LLM decision prompts.

        Format:
            Available annotation models:

            1. ClassName — First line of docstring
               Fields: field1: str, field2: int | None

            2. ContainerClass [list container] — Description
               Fields: items: list[ItemClass]
               Each item (ItemClass): field1: str, field2: str

        This format is readable by all LLM families (Claude, OpenAI, Kimi, GLM, etc.)
        and replaces the previous XML <model_catalog> format.
        """
        lines = ["Available annotation models:", ""]
        for idx, cls in enumerate(theme.models, start=1):
            if self._is_list_container(cls):
                item_cls = self._get_list_item_class(cls)
                summary = self._get_docstring_summary(cls)
                fields = self._get_field_names(cls)
                lines.append(f"{idx}. {cls.__name__} [list container] — {summary}")
                lines.append(f"   Fields: {', '.join(fields)}")
                if item_cls is not None:
                    item_fields = self._get_field_names(item_cls)
                    lines.append(f"   Each item ({item_cls.__name__}): {', '.join(item_fields)}")
            else:
                summary = self._get_docstring_summary(cls)
                fields = self._get_field_names(cls)
                lines.append(f"{idx}. {cls.__name__} — {summary}")
                lines.append(f"   Fields: {', '.join(fields)}")
            lines.append("")  # blank line between models
        # Remove trailing blank line if present
        if lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def get_theme_list_for_prompt(self) -> str:
        """
        Build a formatted list of available themes for injection into the
        classification prompt.

        Example output:
            - default: Generic fallback for content that does not fit a specific domain
            - pharmaceutical: Pharmaceutical drug development documents following ICH CTD Module 3
            - structural_specs: Documents that prescribe how other documents must be structured
        """
        lines: list[str] = []
        for path in sorted(self._themes.keys()):
            theme = self._themes[path]
            indent = "  " * path.count("/")
            lines.append(f"{indent}- {path}: {theme.description}")
        return "\n".join(lines)

    def get_all_theme_paths(self) -> list[str]:
        """Returns all registered theme paths sorted alphabetically.

        Used by the extraction pipeline to build a dynamic Literal type
        for structured output validation.
        """
        return sorted(self._themes.keys())

    def build_theme_section_for_extraction_prompt(self) -> str:
        """Builds the <theme_classification> XML block to inject into the extraction prompt.

        Called once at extraction startup; the result is static for the duration
        of the run (themes do not change at runtime).
        """
        lines = ["<theme_classification>"]
        lines.append(
            "Each StructureNode must include a `theme` field that identifies its thematic domain."
        )
        lines.append(
            "Choose exactly one of the following values for each node:"
        )
        lines.append("")
        for path in sorted(self._themes.keys()):
            desc = self._themes[path].description
            lines.append(f'- "{path}": {desc}')
        lines.append("")
        lines.append("Classification rules:")
        lines.append("1. Use the node title, role, and content to determine the theme.")
        lines.append('2. Use "default" only when no other theme clearly applies.')
        lines.append("3. Sibling nodes in the same structural section typically share the same theme.")
        lines.append("4. The theme value MUST exactly match one of the listed values above.")
        lines.append("</theme_classification>")
        return "\n".join(lines)

    def get_valid_model_classes(self, theme: ThemeNode) -> list[str]:
        """Return list of valid model class names for the repair prompt (current theme only)."""
        return [cls.__name__ for cls in theme.models]

    def get_all_valid_model_classes(self) -> list[str]:
        """Return sorted list of ALL valid model class names across ALL themes."""
        all_classes: set[str] = set()
        for theme in self._themes.values():
            for cls in theme.models:
                all_classes.add(cls.__name__)
        return sorted(all_classes)

    # ------------------------------------------------------------------
    # Neo4j export
    # ------------------------------------------------------------------

    def get_neo4j_theme_structure(self) -> list[dict[str, Any]]:
        """
        Return the full theme tree as a list of dicts for Neo4j ingestion.

        Each dict has:
            path        — str, e.g. "structural_specs/nta"
            name        — str, e.g. "nta"
            parent_path — str | None (parent theme path, or None if top-level)
            model_names — list[str] of selectable class names
        """
        result: list[dict[str, Any]] = []
        for path, theme in self._themes.items():
            # Compute parent path
            if "/" in path:
                parent_path = path.rsplit("/", 1)[0]
                # Only record parent if it is also a theme
                if parent_path not in self._themes:
                    parent_path = None
            else:
                parent_path = None

            result.append({
                "path": path,
                "name": theme.name,
                "parent_path": parent_path,
                "model_names": [cls.__name__ for cls in theme.models],
            })
        return result

    # ------------------------------------------------------------------
    # Static helpers (also used by neo4j_ops)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_list_container(cls: type) -> bool:
        """
        Return True if *cls* is a list_container model.

        A model is a list_container iff it has exactly one Pydantic field whose
        resolved annotation is list[SomePydanticModel] (where SomePydanticModel
        is a subclass of BaseModel).

        Uses typing.get_type_hints() to resolve forward references that arise
        from ``from __future__ import annotations``.
        """
        import typing

        from pydantic import BaseModel as PydanticBaseModel

        if not hasattr(cls, "model_fields"):
            return False
        if len(cls.model_fields) != 1:
            return False
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            return False
        if len(hints) != 1:
            return False
        annotation = next(iter(hints.values()))
        origin = typing.get_origin(annotation)
        if origin is not list:
            return False
        args = typing.get_args(annotation)
        if not args:
            return False
        item_type = args[0]
        try:
            return isinstance(item_type, type) and issubclass(item_type, PydanticBaseModel)
        except TypeError:
            return False

    @staticmethod
    def _get_list_item_class(cls: type) -> type | None:
        """
        Return the Pydantic model class that is the item type of the single
        list field in a list_container model.

        Uses typing.get_type_hints() to resolve forward references that arise
        from ``from __future__ import annotations``.

        Returns None if the class is not a valid list_container.
        """
        import typing

        if not hasattr(cls, "model_fields"):
            return None
        if len(cls.model_fields) != 1:
            return None
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            return None
        if len(hints) != 1:
            return None
        annotation = next(iter(hints.values()))
        origin = typing.get_origin(annotation)
        if origin is not list:
            return None
        args = typing.get_args(annotation)
        if not args:
            return None
        return args[0]

    @staticmethod
    def _get_docstring_summary(cls: type) -> str:
        """Extract the first non-empty line of a class docstring."""
        doc = cls.__doc__ or ""
        for line in doc.strip().splitlines():
            line = line.strip()
            if line:
                return line
        return f"Model for {cls.__name__}"

    @staticmethod
    def _annotation_to_str(annotation) -> str:
        """Convert a type annotation to a compact readable string."""
        import typing
        if annotation is None:
            return "Any"
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is typing.Annotated:
            return ThemeRegistry._annotation_to_str(args[0]) if args else "Any"
        if origin is list:
            inner = ThemeRegistry._annotation_to_str(args[0]) if args else "Any"
            return f"list[{inner}]"
        if origin is not None:
            # Union / Optional — render each arg by name
            rendered = [ThemeRegistry._annotation_to_str(a) for a in args]
            return " | ".join(rendered)
        if annotation is type(None):
            return "None"
        return getattr(annotation, "__name__", str(annotation))

    @staticmethod
    def _get_field_names(cls: type) -> list[str]:
        """Return list of 'field_name: TypeStr' strings for Pydantic model fields.

        Uses typing.get_type_hints() to resolve forward references that arise
        from ``from __future__ import annotations``, falling back to the raw
        field_info.annotation when hint resolution fails.
        """
        import typing

        if not hasattr(cls, "model_fields"):
            return []
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {}
        return [
            f"{name}: {ThemeRegistry._annotation_to_str(hints.get(name, field_info.annotation))}"
            for name, field_info in cls.model_fields.items()
        ]


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_BUILTIN_MODELS_ROOT = Path(__file__).parent.parent / "models"
_registry_instance: ThemeRegistry | None = None


def get_theme_registry() -> ThemeRegistry:
    """Return the global ThemeRegistry, initializing it on first call."""
    global _registry_instance
    if _registry_instance is None:
        from scinr.newton.config import get_config
        cfg = get_config()
        _registry_instance = ThemeRegistry(
            models_root=_BUILTIN_MODELS_ROOT,
            extra_models_roots=cfg.extra_models_paths if cfg.extra_models_paths else None,
            enabled_base_themes=cfg.enabled_base_themes,
            enabled_user_themes=cfg.enabled_user_themes,
        )
    return _registry_instance


def reset_theme_registry() -> None:
    """Force registry rebuild. Called after configure() or in tests."""
    global _registry_instance
    _registry_instance = None
    # Also reset model resolver
    try:
        from scinr.newton.entity_extraction.model_resolver import reset_model_resolver
        reset_model_resolver()
    except Exception:
        pass
    # Also reset extraction cache
    try:
        from scinr.newton.extraction.extraction import reset_extraction_cache
        reset_extraction_cache()
    except Exception:
        pass
