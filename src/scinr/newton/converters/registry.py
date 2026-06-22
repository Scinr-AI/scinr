"""
converters/registry.py — Extension-to-converter registry.

Provides ``get_converter(path)`` which returns the appropriate
``BaseConverter`` instance for a given file path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from scinr.newton.converters.base import BaseConverter, UnsupportedFormatError

logger = logging.getLogger(__name__)

# Populated lazily at first access to avoid circular import issues
# and to defer heavy imports (e.g. pptx, docx) until actually needed.
_REGISTRY: dict[str, type[BaseConverter]] | None = None


def _build_registry() -> dict[str, type[BaseConverter]]:
    """Import all converters and build the extension map."""
    # Local imports so that missing optional dependencies raise only
    # when the specific converter is actually requested.
    from scinr.newton.converters.csv import CsvConverter
    from scinr.newton.converters.docx import DocxConverter
    from scinr.newton.converters.html import HtmlConverter
    from scinr.newton.converters.pdf import PdfConverter
    from scinr.newton.converters.pptx import PptxConverter
    from scinr.newton.converters.text import TextConverter
    from scinr.newton.converters.xlsx import XlsxConverter

    registry: dict[str, type[BaseConverter]] = {}
    for cls in (
        TextConverter,
        CsvConverter,
        HtmlConverter,
        PdfConverter,
        DocxConverter,
        XlsxConverter,
        PptxConverter,
    ):
        for ext in cls.supported_extensions:
            registry[ext] = cls

    return registry


def _get_registry() -> dict[str, type[BaseConverter]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_converter(path: Path) -> BaseConverter:
    """Return the appropriate converter instance for *path*.

    Parameters
    ----------
    path:
        File to be converted.  The extension (case-insensitive, without
        leading dot) is used to look up the converter.

    Returns
    -------
    BaseConverter
        A freshly instantiated converter for the file's format.

    Raises
    ------
    UnsupportedFormatError
        If no converter is registered for the file's extension.
    """
    ext = path.suffix.lower().lstrip(".")
    registry = _get_registry()
    cls = registry.get(ext)
    if cls is None:
        supported = sorted(registry)
        raise UnsupportedFormatError(
            f"No converter registered for extension: {ext!r}. "
            f"Supported extensions: {supported}"
        )
    logger.debug("Resolved converter %s for extension %r", cls.__name__, ext)
    return cls()


def list_supported_extensions() -> list[str]:
    """Return all registered file extensions in sorted order.

    Returns
    -------
    list[str]
        Sorted list of supported extensions (without leading dots).
    """
    return sorted(_get_registry())


def apply_converter_overrides(overrides: dict[str, type]) -> None:
    """
    Apply converter overrides/additions to the registry.

    Parameters
    ----------
    overrides:
        Dict mapping file extensions (without leading dot, lowercase) to
        BaseConverter subclasses. Existing extensions are overridden.
        New extensions are registered.

    Raises
    ------
    ConfigurationError
        If any converter class is not a subclass of BaseConverter.

    Example
    -------
    configure(extra_converters={
        "pdf": MyCustomPdfConverter,   # override built-in PDF converter
        "epub": EpubConverter,         # add new format
    })
    """
    from scinr.newton.exceptions import ConfigurationError
    registry = _get_registry()
    for ext, cls in overrides.items():
        ext_lower = ext.lower().lstrip(".")
        if not (isinstance(cls, type) and issubclass(cls, BaseConverter)):
            raise ConfigurationError(
                f"Converter for extension '{ext_lower}' must be a subclass of BaseConverter. "
                f"Received: {cls!r}"
            )
        if ext_lower in registry:
            logger.info(
                "converters: overriding built-in converter for '.%s' "
                "(%s → %s)",
                ext_lower, registry[ext_lower].__name__, cls.__name__,
            )
        else:
            logger.info(
                "converters: registering new converter for '.%s' (%s)",
                ext_lower, cls.__name__,
            )
        registry[ext_lower] = cls
