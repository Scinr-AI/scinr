"""
tests/unit/test_converters_registry.py — Unit tests for scinr.newton.converters.registry

Imports directly from submodules to avoid triggering the CLI import chain.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import scinr.newton.converters.registry as registry_module

# Import directly from submodules — NOT from scinr.newton (top-level)
from scinr.newton.converters.base import BaseConverter, IntermediateDocument
from scinr.newton.converters.registry import (
    apply_converter_overrides,
    get_converter,
    list_supported_extensions,
)
from scinr.newton.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Fixture: restore registry state after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_registry():
    """Capture and restore the converter registry dict after each test."""
    # Force the registry to be built before we capture it
    original = dict(registry_module._get_registry())
    yield
    # Restore the registry to its original state
    reg = registry_module._get_registry()
    reg.clear()
    reg.update(original)


# ---------------------------------------------------------------------------
# Dummy converters for testing
# ---------------------------------------------------------------------------


class MyPdfConverter(BaseConverter):
    """Custom PDF converter for testing."""
    supported_extensions: frozenset[str] = frozenset({"pdf"})

    def convert(self, source: Path) -> IntermediateDocument:
        return IntermediateDocument(pages=[])


class EpubConverter(BaseConverter):
    """Custom EPUB converter for testing."""
    supported_extensions: frozenset[str] = frozenset({"epub"})

    def convert(self, source: Path) -> IntermediateDocument:
        return IntermediateDocument(pages=[])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyConverterOverrides:
    def test_override_existing_converter(self):
        """Overriding an existing extension replaces the converter in the registry."""
        apply_converter_overrides({"pdf": MyPdfConverter})

        converter = get_converter(Path("document.pdf"))
        assert isinstance(converter, MyPdfConverter)

    def test_register_new_extension(self):
        """Registering a new extension makes it available in the registry."""
        apply_converter_overrides({"epub": EpubConverter})

        assert "epub" in list_supported_extensions()
        converter = get_converter(Path("book.epub"))
        assert isinstance(converter, EpubConverter)

    def test_invalid_converter_raises_config_error(self):
        """Passing a non-BaseConverter class raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            apply_converter_overrides({"pdf": int})

    def test_invalid_converter_non_class_raises(self):
        """Passing a non-class value raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            apply_converter_overrides({"pdf": "not_a_class"})  # type: ignore[arg-type]

    def test_extension_normalisation_uppercase(self):
        """Extension keys are normalised to lowercase."""
        apply_converter_overrides({"PDF": MyPdfConverter})

        converter = get_converter(Path("document.pdf"))
        assert isinstance(converter, MyPdfConverter)

    def test_extension_normalisation_with_dot(self):
        """Extension keys with leading dot are normalised (dot stripped)."""
        apply_converter_overrides({".pdf": MyPdfConverter})

        converter = get_converter(Path("document.pdf"))
        assert isinstance(converter, MyPdfConverter)

    def test_extension_normalisation_uppercase_with_dot(self):
        """Extension keys like '.PDF' are normalised to 'pdf'."""
        apply_converter_overrides({".PDF": MyPdfConverter})

        converter = get_converter(Path("document.pdf"))
        assert isinstance(converter, MyPdfConverter)

    def test_multiple_overrides_in_one_call(self):
        """Multiple overrides can be applied in a single call."""
        apply_converter_overrides({
            "pdf": MyPdfConverter,
            "epub": EpubConverter,
        })

        assert isinstance(get_converter(Path("doc.pdf")), MyPdfConverter)
        assert isinstance(get_converter(Path("book.epub")), EpubConverter)

    def test_override_does_not_affect_other_extensions(self):
        """Overriding 'pdf' does not change converters for other extensions."""
        original_docx_cls = type(get_converter(Path("doc.docx")))
        apply_converter_overrides({"pdf": MyPdfConverter})
        assert isinstance(get_converter(Path("doc.docx")), original_docx_cls)


class TestGetConverter:
    def test_get_converter_returns_instance(self):
        """get_converter returns an instance of BaseConverter."""
        converter = get_converter(Path("document.docx"))
        assert isinstance(converter, BaseConverter)

    def test_get_converter_case_insensitive_extension(self):
        """get_converter handles uppercase file extensions."""
        converter = get_converter(Path("document.DOCX"))
        assert isinstance(converter, BaseConverter)

    def test_get_converter_unsupported_raises(self):
        """get_converter raises UnsupportedFormatError for unknown extensions."""
        from scinr.newton.converters.base import UnsupportedFormatError

        with pytest.raises(UnsupportedFormatError):
            get_converter(Path("file.xyz_unknown_format"))


class TestListSupportedExtensions:
    def test_list_supported_extensions_returns_sorted_list(self):
        """list_supported_extensions returns a sorted list of strings."""
        exts = list_supported_extensions()
        assert isinstance(exts, list)
        assert exts == sorted(exts)
        assert all(isinstance(e, str) for e in exts)

    def test_list_supported_extensions_includes_common_formats(self):
        """Common formats are in the supported extensions list."""
        exts = list_supported_extensions()
        for expected in ("docx", "pdf", "txt", "html"):
            assert expected in exts, f"Expected '{expected}' in supported extensions"
