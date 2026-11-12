"""
converters/pdf.py — PDF to Markdown converter via Mistral OCR API.

Submits the PDF as a base64 data URL to the Mistral OCR endpoint
(``POST https://api.mistral.ai/v1/ocr``).  The JSON body is streamed: the
base64 is encoded block by block while it is sent (``_OcrRequestBody``), so
a request needs a few MB of memory whatever the size of the PDF.  The API
response already matches the intermediate document format, so each page is
mapped directly to an ``IntermediatePage``.

At most ``mistral_ocr_concurrency`` PDFs are converted at the same time per
event loop, independently of the caller's ``parallel_docs``.

Para PDFs que exceden los límites de la API de Mistral OCR (máx. 1000
páginas / 50 MB por solicitud), el documento se divide automáticamente
en chunks contiguos (ver ``pdf_splitter.py``), cada uno se envía por
separado, y los resultados se reúnen de forma transparente preservando
el índice de página absoluto del documento original. La división es
perezosa: los chunks se generan de uno en uno directamente desde el
fichero (nunca se lee el PDF completo en memoria), por lo que el pico de
memoria depende del tamaño de un chunk, no del del archivo. El manejo de
errores por chunk es configurable vía ``mistral_ocr_error_strategy``
(``"fail_fast"`` por defecto, o ``"best_effort"``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
    PageDimensions,
    PageImage,
)
from scinr.newton.converters.pdf_splitter import iter_pdf_chunks, probe_pdf
from scinr.newton.utils.loop_semaphore import LoopSemaphore

logger = logging.getLogger(__name__)

load_dotenv()

_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
_MISTRAL_OCR_MODEL = "mistral-ocr-latest"

# Defaults usados cuando get_config() no está disponible (configure() no
# ha sido llamado) y no se pasó override explícito al constructor.
_DEFAULT_SAFE_MAX_PAGES = 900
_DEFAULT_SAFE_MAX_BYTES = 45 * 1024 * 1024
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
_DEFAULT_ERROR_STRATEGY = "fail_fast"
_DEFAULT_OCR_CONCURRENCY = 2

# PDF bytes encoded per step of the streamed body. A multiple of 3, so every
# block but the last encodes to base64 without padding and the blocks can be
# concatenated into one valid base64 string.
_BASE64_BLOCK_BYTES = 3 * 1024 * 1024

# The OCR request body is this prefix, the PDF in base64 and this suffix: the same
# JSON the API received when the body was built with json=payload.
_OCR_BODY_PREFIX = (
    '{"model": ' + json.dumps(_MISTRAL_OCR_MODEL) + ', "document": '
    '{"type": "document_url", "document_url": "data:application/pdf;base64,'
).encode("ascii")
_OCR_BODY_SUFFIX = b'"}}'


def _ocr_body_suffix(image_options: dict | None) -> bytes:
    """Suffix of the body: closes the document and adds the image options, if any.

    Without image options it is exactly :data:`_OCR_BODY_SUFFIX`, so the body is
    byte for byte the one sent before images could be requested.
    """
    if not image_options:
        return _OCR_BODY_SUFFIX
    extra = "".join(f", {json.dumps(k)}: {json.dumps(v)}" for k, v in image_options.items())
    return ('"}' + extra + "}").encode("ascii")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_BEST_EFFORT_HINT = (
    "Sugerencia: si prefiere continuar la conversión omitiendo únicamente "
    "las páginas que fallen (modo best-effort) en lugar de abortar el "
    "documento completo, configure mistral_ocr_error_strategy='best_effort' "
    "(o la variable de entorno MISTRAL_OCR_ERROR_STRATEGY=best_effort)."
)


@dataclass(frozen=True)
class _OcrLimits:
    """Límites y comportamiento resueltos para una llamada a convert()."""

    safe_max_pages: int
    safe_max_bytes: int
    max_retries: int
    retry_backoff_seconds: float
    error_strategy: str
    concurrency: int
    # Embedded images requested (describe_embedded_images + vision_llm).
    include_images: bool = False
    image_max_pages: int = 50
    image_min_size: int = 0
    max_images_per_page: int = 10

    def image_options(self, page_count: int) -> dict | None:
        """OCR request options that ask for the images of a *page_count*-page chunk."""
        if not self.include_images:
            return None
        options: dict = {
            "include_image_base64": True,
            # Upper bound for the chunk; the per-page cap is applied locally.
            "image_limit": self.max_images_per_page * max(1, page_count),
        }
        if self.image_min_size > 0:
            options["image_min_size"] = self.image_min_size
        return options


class _OcrRequestBody:
    """JSON body of one OCR request, produced block by block while httpx sends it.

    Only one block of the PDF and its base64 are alive at a time. ``document`` is
    either the path of the PDF (read from disk during the send) or the bytes of a
    chunk already serialised by the splitter. Iterable more than once, so every
    retry streams it again from the start.
    """

    def __init__(self, document: Path | bytes, image_options: dict | None = None) -> None:
        self._document = document
        self._suffix = _ocr_body_suffix(image_options)
        if isinstance(document, Path):
            try:
                self._size = document.stat().st_size
            except OSError as exc:
                raise ConversionError(f"Cannot read PDF file {document}: {exc}") from exc
        else:
            self._size = len(document)
        self.content_length = (
            len(_OCR_BODY_PREFIX) + 4 * ((self._size + 2) // 3) + len(self._suffix)
        )

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _OCR_BODY_PREFIX
        if isinstance(self._document, Path):
            async for block in self._file_blocks(self._document):
                yield base64.b64encode(block)
        else:
            view = memoryview(self._document)
            for start in range(0, self._size, _BASE64_BLOCK_BYTES):
                yield base64.b64encode(view[start : start + _BASE64_BLOCK_BYTES])
        yield self._suffix

    async def _file_blocks(self, path: Path) -> AsyncIterator[bytes]:
        # Reads exactly the size measured for Content-Length: a file that changed in
        # between is an error, not a request that disagrees with its own header.
        remaining = self._size
        try:
            with path.open("rb") as fh:
                while remaining > 0:
                    block = await asyncio.to_thread(fh.read, min(_BASE64_BLOCK_BYTES, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    yield block
        except OSError as exc:
            raise ConversionError(f"Cannot read PDF file {path}: {exc}") from exc
        if remaining:
            raise ConversionError(
                f"PDF file {path} changed size while it was being sent to Mistral OCR"
            )


# One semaphore per event loop (see utils/loop_semaphore.py).
_ocr_semaphores = LoopSemaphore()


def _get_ocr_semaphore(limit: int) -> asyncio.Semaphore:
    """Semaphore bounding the PDFs in OCR conversion on the running event loop.

    A different *limit* (``configure(mistral_ocr_concurrency=N)`` between runs)
    replaces it for the conversions that start afterwards.
    """
    return _ocr_semaphores.get(limit)


def _split_data_url(value: str) -> tuple[str, str]:
    """``data:image/jpeg;base64,XXXX`` → ``("image/jpeg", "XXXX")``; plain base64 → PNG."""
    if value.startswith("data:"):
        header, sep, payload = value.partition(",")
        if sep:
            media_type = header[5:].split(";", 1)[0] or "image/png"
            return media_type, payload
    return "image/png", value


def _bbox_side(img: dict, start_key: str, end_key: str) -> int | None:
    """Side of the image bounding box (rendered page pixels), when the response has it."""
    start, end = img.get(start_key), img.get(end_key)
    if isinstance(start, int | float) and isinstance(end, int | float) and end > start:
        return int(end - start)
    return None


class _EnrichmentRun:
    """Embedded-image enrichment across the chunks of one PDF."""

    def __init__(self) -> None:
        from scinr.newton.converters.enrichment import EnrichmentState

        self._state = EnrichmentState()
        self.missing: list[tuple[int, int]] | None = None

    async def enrich(self, pages: list[IntermediatePage]) -> None:
        from scinr.newton.converters.enrichment import enrich_pages

        missing = await enrich_pages(pages, self._state)
        if missing:
            self.missing = (self.missing or []) + missing


class PdfConverter(BaseConverter):
    """Convert ``.pdf`` files to the intermediate format via Mistral OCR.

    Sends the PDF to the Mistral OCR API and maps the response to an
    :class:`~converters.base.IntermediateDocument`.  Each page returned
    by Mistral becomes one :class:`~converters.base.IntermediatePage`.

    Si el PDF excede los límites configurados de páginas o bytes, se
    divide en chunks contiguos (ver :mod:`pdf_splitter`), cada uno se
    envía por separado a la API, y las páginas resultantes se reúnen
    preservando el índice absoluto del documento original. Los chunks se
    generan de uno en uno directamente desde el fichero (``iter_pdf_chunks``),
    de modo que solo hay un chunk en memoria a la vez.

    Parameters
    ----------
    api_key:
        Mistral API key.  If ``None``, the value of the environment
        variable ``MISTRAL_API_KEY`` is used at conversion time.
    safe_max_pages:
        Override explícito del máximo de páginas por chunk. Si ``None``,
        se resuelve vía ``get_config()`` o el default del módulo.
    safe_max_bytes:
        Override explícito del máximo de bytes por chunk. Si ``None``,
        se resuelve vía ``get_config()`` o el default del módulo.
    max_retries:
        Override explícito del número máximo de intentos por chunk.
    retry_backoff_seconds:
        Override explícito de la base del backoff exponencial entre
        reintentos.
    error_strategy:
        Override explícito de la estrategia de manejo de errores por
        chunk: ``"fail_fast"`` o ``"best_effort"``.

        Nota: esta estrategia solo aplica a fallos de red/API por chunk
        (reintentos agotados, errores HTTP no reintentables) sobre chunks
        ya generados por ``iter_pdf_chunks()``. NO cubre el caso en que la
        propia partición falla estructuralmente
        (:class:`PdfSplitError`, una página individual que excede
        ``safe_max_bytes`` incluso aislada) — en ese caso el documento
        aborta siempre, independientemente del ``error_strategy``
        configurado. Como los chunks se generan bajo demanda, ese error
        puede detectarse tras haber enviado ya chunks anteriores a la API.
    concurrency:
        Override explícito del máximo de PDFs en conversión a la vez en el
        event loop. Si ``None``, se resuelve vía ``get_config()``
        (``mistral_ocr_concurrency``) o el default del módulo.
    """

    supported_extensions: frozenset[str] = frozenset({"pdf"})
    is_async: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        safe_max_pages: int | None = None,
        safe_max_bytes: int | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        error_strategy: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        self._api_key: str | None = api_key
        self._safe_max_pages = safe_max_pages
        self._safe_max_bytes = safe_max_bytes
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._error_strategy = error_strategy
        self._concurrency = concurrency

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def convert(self, source: Path) -> IntermediateDocument:
        """Convert a PDF file to the intermediate format.

        This is a native ``async def`` coroutine (not a sync method
        dispatched to a thread): the actual conversion work is genuine
        network I/O against the Mistral OCR API (``httpx.AsyncClient``,
        with ``asyncio.sleep()`` backoff between retries), so it can be
        awaited directly on the event loop without blocking it. See
        ``is_async = True`` on this class and
        ``scinr.newton.converters.main._run_convert()`` for how callers
        dispatch based on this contract.

        Parameters
        ----------
        source:
            Path to the ``.pdf`` file.

        Returns
        -------
        IntermediateDocument
            Document with one :class:`~converters.base.IntermediatePage`
            per PDF page recognised by Mistral OCR. If el documento fue
            dividido en chunks y alguno falló en modo ``best_effort``,
            ``missing_page_ranges`` contendrá los rangos omitidos.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist.
        ConversionError
            If the Mistral API key is not available, the HTTP request
            fails, or the response is in an unexpected format.
        """
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ConversionError(
                "httpx is required for PDF conversion. "
                "Install it with: uv add httpx"
            ) from exc

        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        # Try to get from scinr_config first (supports configure(mistral_api_key=...))
        api_key = self._api_key
        if not api_key:
            try:
                from scinr.newton.config import get_config
                api_key = get_config().mistral_api_key
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ConversionError(
                "MISTRAL_API_KEY is not configured. This key is required to convert PDF files.\n"
                "Get a key at https://console.mistral.ai/ and either:\n"
                "  - Add MISTRAL_API_KEY=your_key to your .env file, or\n"
                "  - Pass it to configure(mistral_api_key='your_key')"
            )

        limits = self._resolve_limits()

        # Each conversion holds at most one chunk and Mistral's response for it; the
        # semaphore bounds how many coincide, whatever the caller's parallel_docs.
        async with _get_ocr_semaphore(limits.concurrency):
            return await self._convert_pdf(source, api_key, limits)

    async def _convert_pdf(
        self, source: Path, api_key: str, limits: _OcrLimits
    ) -> IntermediateDocument:
        """Body of :meth:`convert`, run while holding the OCR semaphore."""
        logger.info("Reading PDF: %s", source.name)
        # Only the size and the page index are read here — never the whole file.
        size_bytes, total_pages_original = probe_pdf(source)

        # With images the response is far heavier than text only: smaller chunks.
        max_pages = (
            min(limits.safe_max_pages, limits.image_max_pages)
            if limits.include_images
            else limits.safe_max_pages
        )
        must_split = size_bytes > limits.safe_max_bytes or total_pages_original > max_pages

        # One state for the whole document: an image repeated across chunks
        # (a logo on every page) is described once.
        enrichment = _EnrichmentRun() if limits.include_images else None

        if not must_split:
            # Within the API limits (<= safe_max_bytes): streamed from disk as is.
            pages = await self._convert_chunk(
                source, api_key, source.name, page_offset=0, limits=limits,
                page_count=total_pages_original,
            )
            if enrichment is not None:
                await enrichment.enrich(pages)
            logger.info("Converted %d page(s) from %s", len(pages), source.name)
            return IntermediateDocument(
                pages=pages, missing_images=enrichment.missing if enrichment else None
            )

        logger.info(
            "PDF %s excede los límites seguros (%d páginas, %d bytes); se procesará por "
            "chunks generados de uno en uno.",
            source.name,
            total_pages_original,
            size_bytes,
        )

        all_pages: list[IntermediatePage] = []
        missing_page_ranges: list[tuple[int, int]] = []

        # PdfSplitError (ya es una ConversionError) se deja propagar tal cual desde
        # el generador; puede aparecer tras haber enviado chunks anteriores.
        for i, chunk in enumerate(
            iter_pdf_chunks(source, max_pages, limits.safe_max_bytes, source_name=source.name),
            start=1,
        ):
            start_page, end_page = chunk.start_page, chunk.end_page
            etiqueta = (
                f"chunk {i} (páginas originales "
                f"{start_page}-{end_page - 1} de {source.name})"
            )
            logger.info(
                "Enviando %s: %d página(s), %d bytes",
                etiqueta,
                chunk.page_count,
                len(chunk.pdf_bytes),
            )
            try:
                paginas_chunk = await self._convert_chunk(
                    chunk.pdf_bytes,
                    api_key,
                    etiqueta,
                    page_offset=start_page,
                    limits=limits,
                    error_label=etiqueta,
                    page_count=chunk.page_count,
                )
            except ConversionError as exc:
                if limits.error_strategy == "best_effort":
                    logger.warning(
                        "%s falló y será omitido en modo best-effort. Error: %s",
                        etiqueta,
                        exc,
                    )
                    missing_page_ranges.append((start_page, end_page))
                    continue
                raise ConversionError(
                    f"Fallo al convertir {etiqueta}. Rango de páginas afectado: "
                    f"[{start_page}, {end_page}) de {total_pages_original} "
                    f"páginas totales en {source.name}. Error original: {exc}\n"
                    f"{_BEST_EFFORT_HINT}"
                ) from exc
            finally:
                # Drop the chunk before the generator serialises the next one.
                del chunk
            # Outside the try: an image failure (fail_fast) is not a chunk failure.
            # The chunk's images are described and released before the next chunk.
            if enrichment is not None:
                await enrichment.enrich(paginas_chunk)
            all_pages.extend(paginas_chunk)
            del paginas_chunk

        missing_images = enrichment.missing if enrichment else None
        if not missing_page_ranges:
            logger.info("Converted %d page(s) from %s", len(all_pages), source.name)
            return IntermediateDocument(pages=all_pages, missing_images=missing_images)

        missing_count = sum(end - start for start, end in missing_page_ranges)
        logger.warning(
            "Documento convertido en modo best-effort: %d página(s) omitidas en "
            "%d rango(s): %s",
            missing_count,
            len(missing_page_ranges),
            missing_page_ranges,
        )
        return IntermediateDocument(
            pages=all_pages,
            missing_page_ranges=missing_page_ranges,
            missing_images=missing_images,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_limits(self) -> _OcrLimits:
        """Resuelve los límites/comportamiento de OCR con prioridad:
        override del constructor > get_config() > default del módulo.

        Raises
        ------
        ConversionError
            Si el valor resuelto de ``error_strategy`` (de cualquier
            fuente) no es ``"fail_fast"`` ni ``"best_effort"``. Mismo
            nivel de estrictez que ``configure()`` (que lanza
            ``ConfigurationError`` en el mismo caso), para no ser más
            permisivo aquí con un override explícito al constructor.
        """
        cfg = None
        try:
            from scinr.newton.config import get_config
            cfg = get_config()
        except Exception:
            cfg = None

        safe_max_pages = self._safe_max_pages
        if safe_max_pages is None:
            safe_max_pages = getattr(cfg, "mistral_ocr_safe_max_pages", None)
        if safe_max_pages is None:
            safe_max_pages = _DEFAULT_SAFE_MAX_PAGES

        safe_max_bytes = self._safe_max_bytes
        if safe_max_bytes is None:
            safe_max_bytes = getattr(cfg, "mistral_ocr_safe_max_bytes", None)
        if safe_max_bytes is None:
            safe_max_bytes = _DEFAULT_SAFE_MAX_BYTES

        max_retries = self._max_retries
        if max_retries is None:
            max_retries = getattr(cfg, "mistral_ocr_max_retries", None)
        if max_retries is None:
            max_retries = _DEFAULT_MAX_RETRIES

        retry_backoff_seconds = self._retry_backoff_seconds
        if retry_backoff_seconds is None:
            retry_backoff_seconds = getattr(cfg, "mistral_ocr_retry_backoff_seconds", None)
        if retry_backoff_seconds is None:
            retry_backoff_seconds = _DEFAULT_RETRY_BACKOFF_SECONDS

        error_strategy = self._error_strategy
        if error_strategy is None:
            error_strategy = getattr(cfg, "mistral_ocr_error_strategy", None)
        if error_strategy is None:
            error_strategy = _DEFAULT_ERROR_STRATEGY

        concurrency = self._concurrency
        if concurrency is None:
            concurrency = getattr(cfg, "mistral_ocr_concurrency", None)
        if concurrency is None:
            concurrency = _DEFAULT_OCR_CONCURRENCY
        if concurrency < 1:
            raise ConversionError(
                f"mistral_ocr_concurrency debe ser >= 1. Recibido: {concurrency!r}."
            )

        if error_strategy not in ("fail_fast", "best_effort"):
            raise ConversionError(
                f"Valor inválido para mistral_ocr_error_strategy: {error_strategy!r}. "
                f"Valores válidos: 'fail_fast', 'best_effort'."
            )

        from scinr.newton.config import embedded_images_enabled

        include_images = embedded_images_enabled()
        return _OcrLimits(
            safe_max_pages=safe_max_pages,
            safe_max_bytes=safe_max_bytes,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            error_strategy=error_strategy,
            concurrency=concurrency,
            include_images=include_images,
            image_max_pages=getattr(cfg, "mistral_ocr_image_max_pages", 50),
            image_min_size=getattr(cfg, "image_min_side_px", 0),
            max_images_per_page=getattr(cfg, "max_images_per_page", 10),
        )

    async def _convert_chunk(
        self,
        document: Path | bytes,
        api_key: str,
        chunk_label: str,
        page_offset: int,
        limits: _OcrLimits,
        error_label: str | None = None,
        page_count: int = 1,
    ) -> list[IntermediatePage]:
        """Envía un chunk de PDF a Mistral OCR y mapea su respuesta.

        Parameters
        ----------
        document:
            PDF autocontenido a enviar: la ruta del fichero (caso sin split, se
            lee por bloques al enviarlo) o los bytes del chunk (caso con split).
        api_key:
            Mistral API key.
        chunk_label:
            Etiqueta descriptiva usada en logs y en el mensaje de error
            de red. Para el caso sin split, es simplemente el nombre
            del archivo.
        page_offset:
            Desplazamiento de página absoluto del documento original,
            sumado al ``index`` reportado por Mistral para este chunk.
        limits:
            Límites/comportamiento resueltos para esta conversión.
        error_label:
            Etiqueta usada específicamente en el mensaje de error HTTP
            (``response.is_error``). ``None`` (el caso sin split)
            preserva el formato de mensaje EXACTO previo a esta
            funcionalidad: ``"Mistral OCR API returned HTTP {status}: {text}"``.
            Cuando se pasa (caso con split), el mensaje incluye la
            etiqueta del chunk para trazabilidad.
        page_count:
            Páginas del chunk; acota ``image_limit`` cuando se piden imágenes.

        Returns
        -------
        list[IntermediatePage]
        """
        body = _OcrRequestBody(document, limits.image_options(page_count))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # An explicit Content-Length makes httpx send the streamed body as a plain
            # request, not with Transfer-Encoding: chunked.
            "Content-Length": str(body.content_length),
        }

        logger.info("Calling Mistral OCR API for %s", chunk_label)
        data = await self._post_to_mistral_with_retry(
            body, headers, limits, chunk_label, error_label=error_label
        )

        if isinstance(data, bytes | bytearray):
            # Parsed page by page from the raw body: neither the decoded text nor
            # the whole parsed tree of a response with images is ever built.
            pages = self._map_pages_streamed(data, page_offset)
            del data
            return pages

        if "pages" not in data:
            raise ConversionError(
                f"Unexpected Mistral OCR API response (no 'pages' key): {data}"
            )

        pages = [
            self._map_page(page_data, index_offset=page_offset) for page_data in data["pages"]
        ]
        # The response (with its images) is not kept beyond the mapping.
        del data
        return pages

    def _map_pages_streamed(self, body: bytes, page_offset: int) -> list[IntermediatePage]:
        """Map the ``pages`` of a raw OCR response body one page at a time (ijson)."""
        import io

        import ijson

        pages: list[IntermediatePage] = []
        try:
            for page_data in ijson.items(io.BytesIO(body), "pages.item", use_float=True):
                pages.append(self._map_page(page_data, index_offset=page_offset))
        except ijson.JSONError as exc:
            raise ConversionError(
                f"Cannot parse Mistral OCR API response as JSON: {exc}"
            ) from exc
        if not pages:
            # Small body: an empty page list or an unexpected shape.
            try:
                parsed = json.loads(body)
            except ValueError as exc:
                raise ConversionError(
                    f"Cannot parse Mistral OCR API response as JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict) or "pages" not in parsed:
                raise ConversionError(
                    f"Unexpected Mistral OCR API response (no 'pages' key): {parsed}"
                )
        return pages

    async def _post_to_mistral_with_retry(
        self,
        body: _OcrRequestBody,
        headers: dict,
        limits: _OcrLimits,
        chunk_label: str,
        error_label: str | None = None,
    ) -> bytes | dict:
        """Ejecuta el POST a Mistral OCR con reintento/backoff exponencial
        ante errores de red o códigos HTTP reintentables.

        Parameters
        ----------
        body:
            Cuerpo JSON de la solicitud, generado por bloques en cada intento.
        headers:
            Cabeceras HTTP (incluye Authorization).
        limits:
            Límites/comportamiento resueltos (max_retries, backoff).
        chunk_label:
            Etiqueta usada en logs y en el mensaje de error de red.
        error_label:
            Etiqueta usada específicamente en el mensaje de error HTTP.
            ``None`` preserva el formato de mensaje exacto previo a
            esta funcionalidad (sin sufijo "for X"), usado en el caso
            sin split para no romper la compatibilidad del mensaje.

        Returns
        -------
        bytes | dict
            Cuerpo de la respuesta exitosa: los bytes tal cual (se parsean página a
            página en :meth:`_map_pages_streamed`) o, si la respuesta no los expone,
            el JSON ya parseado.

        Raises
        ------
        ConversionError
            Si se agotan los reintentos o el error no es reintentable.
        """
        import httpx

        last_request_exc: httpx.RequestError | None = None
        for attempt in range(1, limits.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    response = await client.post(
                        _MISTRAL_OCR_URL,
                        content=body,
                        headers=headers,
                    )
            except httpx.RequestError as exc:
                last_request_exc = exc
                if attempt < limits.max_retries:
                    # Backoff exponencial con límite de 5 minutos (300s) entre retries
                    backoff = min(
                        limits.retry_backoff_seconds * (2 ** (attempt - 1)),
                        300.0  # 5 minutos máximo entre retries sucesivos
                    )
                    logger.warning(
                        "Network error calling Mistral OCR API for %s (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        chunk_label,
                        attempt,
                        limits.max_retries,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise ConversionError(
                    f"Network error calling Mistral OCR API for {chunk_label} "
                    f"after {limits.max_retries} attempt(s): {exc}"
                ) from exc

            if response.is_error:
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < limits.max_retries:
                    # Backoff exponencial con límite de 5 minutos (300s) entre retries
                    backoff = min(
                        limits.retry_backoff_seconds * (2 ** (attempt - 1)),
                        300.0  # 5 minutos máximo entre retries sucesivos
                    )
                    logger.warning(
                        "Mistral OCR API returned HTTP %d for %s (attempt %d/%d). "
                        "Retrying in %.1fs...",
                        response.status_code,
                        chunk_label,
                        attempt,
                        limits.max_retries,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                # error_label=None preserva el formato de mensaje EXACTO
                # previo a esta funcionalidad (caso sin split); con
                # error_label se añade la etiqueta del chunk para
                # trazabilidad (caso con split).
                if error_label:
                    raise ConversionError(
                        f"Mistral OCR API returned HTTP {response.status_code} for "
                        f"{error_label}: {response.text}"
                    )
                raise ConversionError(
                    f"Mistral OCR API returned HTTP {response.status_code}: {response.text}"
                )

            content = getattr(response, "content", None)
            if isinstance(content, bytes | bytearray):
                # Raw body; parsed page by page by the caller.
                return content
            try:
                return response.json()
            except Exception as exc:
                raise ConversionError(
                    f"Cannot parse Mistral OCR API response as JSON: {exc}"
                ) from exc

        # Inalcanzable en la práctica (el loop siempre retorna o lanza),
        # pero se cubre por completitud/tipo de retorno.
        raise ConversionError(
            f"Mistral OCR API call for {chunk_label} exhausted retries: {last_request_exc}"
        )

    def _map_page(self, page_data: dict, index_offset: int = 0) -> IntermediatePage:
        """Map a single Mistral OCR page dict to an :class:`IntermediatePage`.

        Parameters
        ----------
        page_data:
            A single element from the ``pages`` list in the Mistral OCR
            response.
        index_offset:
            Desplazamiento de página absoluto del documento original,
            sumado al ``index`` reportado por Mistral. ``0`` preserva el
            comportamiento previo (sin chunking).

        Returns
        -------
        IntermediatePage
        """
        index: int = index_offset + page_data.get("index", 0)
        markdown: str = page_data.get("markdown", "")

        # Map images
        images: list[PageImage] = []
        for img_idx, img in enumerate(page_data.get("images") or []):
            media_type, raw_b64 = _split_data_url(img.get("image_base64") or "")
            images.append(
                PageImage(
                    index=img_idx,
                    base64=raw_b64,
                    media_type=media_type,
                    ref=img.get("id") or None,
                    width=_bbox_side(img, "top_left_x", "bottom_right_x"),
                    height=_bbox_side(img, "top_left_y", "bottom_right_y"),
                )
            )

        # Map dimensions
        dims_data: dict = page_data.get("dimensions") or {}
        dimensions = PageDimensions(
            dpi=dims_data.get("dpi"),
            height=dims_data.get("height"),
            width=dims_data.get("width"),
        )

        return IntermediatePage(
            index=index,
            markdown=markdown,
            images=images,
            dimensions=dimensions,
        )
