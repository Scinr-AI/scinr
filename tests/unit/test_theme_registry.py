"""
tests/unit/test_theme_registry.py — Unit tests for scinr.newton.utils.theme_registry

Imports directly from the submodule to avoid triggering the CLI import chain.
"""
from __future__ import annotations

import pytest

# Import directly from the submodule — NOT from scinr.newton (top-level)
import scinr.newton.utils.theme_registry as tr_module
from scinr.newton.utils.theme_registry import ThemeNode, ThemeRegistry, reset_theme_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_theme_node(path: str, name: str | None = None) -> ThemeNode:
    """Build a minimal ThemeNode for testing."""
    return ThemeNode(
        path=path,
        name=name or path.split("/")[-1],
        description=f"Description for {path}",
        models=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThemeRegistrySingleton:
    def test_registry_is_singleton(self, tmp_path):
        """get_theme_registry() called twice returns the same object when singleton is set."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Directly instantiate and set the singleton
        registry1 = ThemeRegistry(models_root=models_dir)
        tr_module._registry_instance = registry1

        from scinr.newton.utils.theme_registry import get_theme_registry
        registry2 = get_theme_registry()
        assert registry1 is registry2

    def test_reset_theme_registry_clears_singleton(self, tmp_path):
        """reset_theme_registry() sets _registry_instance to None."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        tr_module._registry_instance = registry

        reset_theme_registry()
        assert tr_module._registry_instance is None


class TestThemeRegistryLazyInit:
    def test_importing_does_not_raise_without_models_dir(self):
        """Importing theme_registry does not raise even if models/ is missing."""
        # The import already happened; just verify the module is importable
        import scinr.newton.utils.theme_registry  # noqa: F401
        # No exception means success

    def test_instantiate_with_missing_models_root_raises_config_error(self, tmp_path):
        """ThemeRegistry raises ConfigurationError when models_root does not exist."""
        from scinr.newton.exceptions import ConfigurationError

        missing = tmp_path / "nonexistent_models"
        with pytest.raises(ConfigurationError, match="does not exist"):
            ThemeRegistry(models_root=missing)

    def test_instantiate_with_empty_models_dir(self, tmp_path):
        """ThemeRegistry with an empty models dir initializes without error."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        assert registry._themes == {}


class TestEnabledThemesFilter:
    def test_enabled_themes_filter(self, tmp_path):
        """Only themes in enabled_themes are kept after filtering."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)

        # Manually inject fake themes (bypassing catalog loading)
        registry._themes = {
            "foo": _make_fake_theme_node("foo"),
            "bar": _make_fake_theme_node("bar"),
        }

        # Apply filter
        registry._apply_enabled_filter(["foo"])
        assert "foo" in registry._themes
        assert "bar" not in registry._themes

    def test_enabled_themes_filter_unknown_raises(self, tmp_path):
        """_apply_enabled_filter raises ConfigurationError for unknown theme names."""
        from scinr.newton.exceptions import ConfigurationError

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        registry._themes = {
            "foo": _make_fake_theme_node("foo"),
        }

        with pytest.raises(ConfigurationError, match="Unknown theme"):
            registry._apply_enabled_filter(["nonexistent"])

    def test_enabled_themes_empty_list_raises(self, tmp_path):
        """_apply_enabled_filter raises ConfigurationError for empty list."""
        from scinr.newton.exceptions import ConfigurationError

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        registry._themes = {"foo": _make_fake_theme_node("foo")}

        with pytest.raises(ConfigurationError, match="cannot be empty"):
            registry._apply_enabled_filter([])


class TestThemeRegistryLookup:
    def test_find_best_theme_exact_match(self, tmp_path):
        """find_best_theme returns exact match when available."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        node = _make_fake_theme_node("pharmaceutical")
        registry._themes = {"pharmaceutical": node}

        result = registry.find_best_theme("pharmaceutical")
        assert result is node

    def test_find_best_theme_prefix_fallback(self, tmp_path):
        """find_best_theme falls back to shorter prefix when exact path not found."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        parent_node = _make_fake_theme_node("structural_specs")
        registry._themes = {"structural_specs": parent_node}

        # Request a more specific path that doesn't exist
        result = registry.find_best_theme("structural_specs/nta/subsection")
        assert result is parent_node

    def test_find_best_theme_default_fallback(self, tmp_path):
        """find_best_theme falls back to 'default' theme when path not found."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        default_node = _make_fake_theme_node("default")
        registry._themes = {"default": default_node}

        result = registry.find_best_theme("unknown_theme")
        assert result is default_node

    def test_find_best_theme_no_themes_raises(self, tmp_path):
        """find_best_theme raises RuntimeError when no themes are registered."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        registry._themes = {}

        with pytest.raises(RuntimeError, match="no themes found"):
            registry.find_best_theme("anything")

    def test_find_best_theme_none_path(self, tmp_path):
        """find_best_theme with None path falls back to default."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        default_node = _make_fake_theme_node("default")
        registry._themes = {"default": default_node}

        result = registry.find_best_theme(None)
        assert result is default_node


class TestThemeRegistryHelpers:
    def test_get_all_theme_paths_sorted(self, tmp_path):
        """get_all_theme_paths returns sorted list of theme paths."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        registry._themes = {
            "zzz": _make_fake_theme_node("zzz"),
            "aaa": _make_fake_theme_node("aaa"),
            "mmm": _make_fake_theme_node("mmm"),
        }

        paths = registry.get_all_theme_paths()
        assert paths == ["aaa", "mmm", "zzz"]

    def test_get_theme_list_for_prompt(self, tmp_path):
        """get_theme_list_for_prompt returns a non-empty string."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        registry._themes = {
            "foo": _make_fake_theme_node("foo"),
        }

        result = registry.get_theme_list_for_prompt()
        assert "foo" in result
        assert "Description for foo" in result

    def test_get_all_theme_paths_empty(self, tmp_path):
        """get_all_theme_paths returns empty list when no themes registered."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        registry = ThemeRegistry(models_root=models_dir)
        assert registry.get_all_theme_paths() == []
