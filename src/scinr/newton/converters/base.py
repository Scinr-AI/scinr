"""
converters/base.py — Core dataclasses and abstract base converter.

Defines the intermediate document format that all converters produce,
and the BaseConverter abstract class that all format-specific converters
must implement.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from scinr.newton.exceptions import ScinrError
from scinr.newton.models.file_metadata import FileMetadata
from scinr.newton.models.transcript import TranscriptCue

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
        Base64-encoded image bytes. Empty when the image is loaded lazily
        (see :meth:`load_bytes`) or once it has been released after the
        embedded-image enrichment of Stage 0.
    media_type:
        MIME type of the image (e.g. ``"image/png"``).
    description:
        Textual description. Left empty by converters; filled by the
        embedded-image enrichment of Stage 0 (``describe_embedded_images``,
        only with a ``vision_llm`` configured).
    image_id:
        Id of the stored :class:`~storage.models.PageImageRecord`, when the
        described image was stored (``store_described_images`` with a
        persistent storage backend whose ``stores_images`` is ``True``).
        Serialised in the intermediate JSON, like ``page_id``.
    ref:
        Text that references the image in the page markdown as
        ``![ref](ref)``. When present, the enrichment replaces that reference
        with the description block; otherwise the block goes at the end of
        the page.
    width, height:
        Size in pixels, when the converter knows it without decoding.

    A converter may attach a private loader instead of filling ``base64``
    (``PageImage(...)._loader = callable``, see :meth:`with_loader`) so the
    bytes are only read while the image is described. The loader is never
    serialised: an intermediate JSON never carries lazy blobs.
    """

    index: int
    base64: str = ""
    media_type: str = "image/png"
    description: str = ""
    image_id: str | None = None
    ref: str | None = None
    width: int | None = None
    height: int | None = None
    _loader: Callable[[], bytes] | None = PrivateAttr(default=None)

    @classmethod
    def with_loader(cls, loader: Callable[[], bytes], **fields: Any) -> PageImage:
        """Build an image whose bytes are read lazily by *loader*."""
        image = cls(**fields)
        image._loader = loader
        return image

    @property
    def has_content(self) -> bool:
        """``True`` while the image still holds bytes (base64 or a loader)."""
        return bool(self.base64) or self._loader is not None

    def load_bytes(self) -> bytes:
        """Return the image bytes: from the loader if set, else decoded from ``base64``."""
        if self._loader is not None:
            return self._loader()
        return base64.b64decode(self.base64)

    def release(self) -> None:
        """Drop the image bytes (``base64`` and loader) once they are no longer needed."""
        self.base64 = ""
        self._loader = None


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
    # Time range [start_ms, end_ms) of the page in the source medium (audio /
    # video segments). None for documents without a time axis.
    start_ms: int | None = None
    end_ms: int | None = None
    # Transcribed sentences of the page with their absolute times (audio /
    # video; decision D7). The markdown is the cues' text, one per line; it
    # does not change. None for documents without a time axis. Serialised in
    # the intermediate JSON, like ``page_id``.
    transcript: list[TranscriptCue] | None = None


class IntermediateDocument(BaseModel):
    """A complete document in the intermediate format.

    This is the root object serialised to JSON and written to the converter
    output directory.
    """

    pages: list[IntermediatePage]
    folder_path: str | None = None  # relative path of parent folder from input root (None for root files)
    raw_file_id: str | None = None  # MongoDB ObjectId of the RawFileRecord stored by Stage 0
    context_instructions: str | None = None  # Free-text user-provided ingestion context. Set via ``context_instructions=``.
    document_name: str | None = None  # Name of the original source file, extension included (``informe.pdf``). Injected by convert_folder() / convert_single_file().
    # Owner of the upload, stamped by convert_one() / convert_single_file() and
    # written on the stored raw file and pages. Public-API form: None = public.
    tenant_id: str | None = None
    created_by_user_id: str | None = None
    job_id: str | None = None
    missing_page_ranges: list[tuple[int, int]] | None = None
    # Rangos [start, end) de páginas del documento ORIGINAL que no pudieron
    # convertirse y fueron omitidas en modo best-effort (mistral_ocr_error_strategy).
    # None cuando la conversión fue completa o no aplica (converters no-PDF).
    file_metadata: FileMetadata | None = None
    # Metadatos del fichero fuente (imagen, audio, vídeo); llegan al nodo Document.
    missing_segments: list[tuple[int, int]] | None = None
    # Tramos [start_ms, end_ms) de audio o vídeo omitidos en best_effort
    # (media_error_strategy). None cuando no se omitió ninguno.
    missing_images: list[tuple[int, int]] | None = None
    # Imágenes (page_index, image_index) que no pudieron describirse y se
    # omitieron en best_effort. None cuando no se omitió ninguna.

    def omissions(self) -> dict[str, list[tuple[int, int]]]:
        """The non-empty ``missing_*`` fields, by name (empty when nothing was skipped)."""
        fields = {
            "missing_page_ranges": self.missing_page_ranges,
            "missing_segments": self.missing_segments,
            "missing_images": self.missing_images,
        }
        return {name: list(value) for name, value in fields.items() if value}

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

        The output file is named ``{source.name}.json`` (the source file name
        with its extension, e.g. ``doc.pdf.json``). If a file with that name
        already exists, a numeric suffix is appended to avoid silent
        overwriting (e.g. ``doc.pdf_1.json``).

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
            The source file whose name (with extension) is used for the output name.
        output_dir:
            Target directory.

        Returns
        -------
        Path
            A path inside *output_dir* that does not yet exist.
        """
        candidate = output_dir / f"{source.name}.json"
        if not candidate.exists():
            return candidate
        counter = 1
        while True:
            candidate = output_dir / f"{source.name}_{counter}.json"
            if not candidate.exists():
                logger.warning(
                    "Output collision: renamed to %s", candidate.name
                )
                return candidate
            counter += 1
