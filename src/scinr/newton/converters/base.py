"""
converters/base.py — Core dataclasses and abstract base converter.

Defines the intermediate document format that all converters produce,
and the BaseConverter abstract class that all format-specific converters
must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from scinr.newton.exceptions import ScinrError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConverterError(Exception):
    """Base class for converter errors (kept for backward compatibility)."""


class UnsupportedFormatError(ConverterError):
    """Raised when no converter is registered for the given file extension."""


class ConversionError(ConverterError, ScinrError):
    """Raised when a converter fails to process a specific file."""


# ---------------------------------------------------------------------------
# Intermediate format dataclasses (Pydantic v2)
# ---------------------------------------------------------------------------


class PageDimensions(BaseModel):
    """Physical dimensions of a page."""

    dpi: int | None = None
    height: int | None = None
    width: int | None = None


class PageImage(BaseModel):
    """An image extracted from a page.

    Parameters
    ----------
    index:
        Zero-based position of the image within the page.
    base64:
        Base64-encoded image bytes.
    media_type:
        MIME type of the image (e.g. ``"image/png"``).
    description:
        Optional textual description. Left empty by converters;
        filled by Stage 1 using LLM vision.
    """

    index: int
    base64: str
    media_type: str = "image/png"
    description: str = ""


class IntermediatePage(BaseModel):
    """A single page of a document in the intermediate format.

    This schema mirrors the Mistral OCR output consumed by Stage 1.
    """

    index: int
    markdown: str
    images: list[PageImage] = Field(default_factory=list)
    dimensions: PageDimensions = Field(default_factory=PageDimensions)
    tables: list = Field(default_factory=list)
    hyperlinks: list = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None
    confidence_scores: None = None
    page_id: str | None = None  # MongoDB ObjectId of the ConvertedPageRecord stored by Stage 0


class IntermediateDocument(BaseModel):
    """A complete document in the intermediate format.

    This is the root object serialised to JSON and written to
    ``data/input/`` or ``data/input-pruebas/``.
    """

    pages: list[IntermediatePage]
    folder_path: str | None = None  # relative path of parent folder from input root (None for root files)
    raw_file_id: str | None = None  # MongoDB ObjectId of the RawFileRecord stored by Stage 0
    context_instructions: str | None = None  # Free-text user-provided ingestion context. Injected via CLI --context.
    document_name: str | None = None  # Stem of the original source file. Injected by convert_folder() / convert_single_file().
    missing_page_ranges: list[tuple[int, int]] | None = None
    # Rangos [start, end) de páginas del documento ORIGINAL que no pudieron
    # convertirse y fueron omitidas en modo best-effort (mistral_ocr_error_strategy).
    # None cuando la conversión fue completa o no aplica (converters no-PDF).

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string.

        Parameters
        ----------
        indent:
            JSON indentation level.

        Returns
        -------
        str
            JSON representation of this document.
        """
        return self.model_dump_json(indent=indent)


# ---------------------------------------------------------------------------
# Abstract base converter
# ---------------------------------------------------------------------------


class BaseConverter(ABC):
    """Abstract base class for all format-specific converters.

    Subclasses must:
    1. Define ``supported_extensions`` (frozenset of lowercase extensions
       without leading dot, e.g. ``frozenset({"pdf"})``).
    2. Implement ``convert(source)``.

    Subclasses whose ``convert()`` is implemented as a coroutine (``async
    def``) rather than a regular blocking method — e.g. converters that
    perform genuine async network I/O, like ``PdfConverter`` calling the
    Mistral OCR API — must set the class attribute ``is_async = True``.
    ``ABC`` cannot verify at runtime whether a subclass's ``convert()`` is a
    coroutine function, so this flag is purely declarative: callers (see
    ``scinr.newton.converters.main._run_convert()``) read it to decide
    whether to ``await converter.convert(source)`` directly on the event
    loop, or to dispatch the (blocking, sync) call to a worker thread via
    ``asyncio.to_thread()``.

    Parameters
    ----------
    supported_extensions : frozenset[str]
        Class attribute listing the file extensions this converter handles.
    is_async : bool
        Class attribute declaring whether ``convert()`` is a coroutine
        function. Defaults to ``False`` (regular sync method).
    """

    supported_extensions: frozenset[str] = frozenset()
    is_async: bool = False

    @abstractmethod
    def convert(self, source: Path) -> IntermediateDocument:
        """Convert *source* to the intermediate document format.

        Parameters
        ----------
        source:
            Path to the source file to convert.

        Returns
        -------
        IntermediateDocument
            The converted document with one or more pages.

        Raises
        ------
        ConversionError
            If conversion fails for any reason.
        FileNotFoundError
            If *source* does not exist.
        """

    def convert_and_write(self, source: Path, output_dir: Path) -> Path:
        """Convert *source* and write the result to *output_dir*.

        The output file is named ``{source.stem}.json``. If a file with
        that name already exists, a numeric suffix is appended to avoid
        silent overwriting (e.g. ``doc_1.json``).

        Parameters
        ----------
        source:
            Path to the source file to convert.
        output_dir:
            Directory where the JSON output will be written.
            Created automatically if it does not exist.

        Returns
        -------
        Path
            Path to the written output file.

        Raises
        ------
        ConversionError
            If conversion fails, or if this converter declares
            ``is_async = True`` (async converters are not supported by
            this method).
        """
        if self.is_async:
            raise ConversionError(
                f"{type(self).__name__}.convert_and_write() does not support "
                "async converters (is_async=True). Use 'await converter.convert(source)' "
                "directly, or scinr.newton.converters.main._run_convert(), instead."
            )

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._resolve_output_path(source, output_dir)

        logger.info("Converting %s → %s", source.name, output_path.name)
        try:
            doc = self.convert(source)
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Failed to convert {source}: {exc}") from exc

        output_path.write_text(doc.to_json(), encoding="utf-8")
        logger.info("Written: %s (%d page(s))", output_path, len(doc.pages))
        return output_path

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _make_page(
        self,
        index: int,
        markdown: str,
        images: list[PageImage] | None = None,
        dimensions: PageDimensions | None = None,
    ) -> IntermediatePage:
        """Build an ``IntermediatePage`` with sensible defaults.

        Parameters
        ----------
        index:
            Zero-based page number.
        markdown:
            Page content as Markdown text.
        images:
            Optional list of images extracted from this page.
        dimensions:
            Optional page dimensions.

        Returns
        -------
        IntermediatePage
        """
        return IntermediatePage(
            index=index,
            markdown=markdown,
            images=images or [],
            dimensions=dimensions or PageDimensions(),
        )

    @staticmethod
    def _resolve_output_path(source: Path, output_dir: Path) -> Path:
        """Return a unique output path, appending suffix if needed.

        Parameters
        ----------
        source:
            The source file whose stem is used for the output name.
        output_dir:
            Target directory.

        Returns
        -------
        Path
            A path inside *output_dir* that does not yet exist.
        """
        candidate = output_dir / f"{source.stem}.json"
        if not candidate.exists():
            return candidate
        counter = 1
        while True:
            candidate = output_dir / f"{source.stem}_{counter}.json"
            if not candidate.exists():
                logger.warning(
                    "Output collision: renamed to %s", candidate.name
                )
                return candidate
            counter += 1
