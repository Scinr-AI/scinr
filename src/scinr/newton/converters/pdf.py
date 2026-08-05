"""
converters/pdf.py — PDF to Markdown converter via Mistral OCR API.

Encodes the PDF as base64 and submits it to the Mistral OCR endpoint
(``POST https://api.mistral.ai/v1/ocr``).  The API response already
matches the intermediate document format, so each page is mapped
directly to an ``IntermediatePage``.

Para PDFs que exceden los límites de la API de Mistral OCR (máx. 1000
páginas / 50 MB por solicitud), el documento se divide automáticamente
en chunks contiguos (ver ``pdf_splitter.py``), cada uno se envía por
separado, y los resultados se reúnen de forma transparente preservando
el índice de página absoluto del documento original. El manejo de
errores por chunk es configurable vía ``mistral_ocr_error_strategy``
(``"fail_fast"`` por defecto, o ``"best_effort"``).
"""

from __future__ import annotations

import base64
import logging
import os
import time
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
from scinr.newton.converters.pdf_splitter import PdfSplitError, needs_splitting, split_pdf

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


class PdfConverter(BaseConverter):
    """Convert ``.pdf`` files to the intermediate format via Mistral OCR.

    Sends the PDF to the Mistral OCR API and maps the response to an
    :class:`~converters.base.IntermediateDocument`.  Each page returned
    by Mistral becomes one :class:`~converters.base.IntermediatePage`.

    Si el PDF excede los límites configurados de páginas o bytes, se
    divide en chunks contiguos (ver :mod:`pdf_splitter`), cada uno se
    envía por separado a la API, y las páginas resultantes se reúnen
    preservando el índice absoluto del documento original.

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
        ya generados por ``split_pdf()``. NO cubre el caso en que la
        propia partición inicial falla estructuralmente
        (:class:`PdfSplitError`, una página individual que excede
        ``safe_max_bytes`` incluso aislada) — en ese caso el documento
        aborta siempre, independientemente del ``error_strategy``
        configurado.
    """

    supported_extensions: frozenset[str] = frozenset({"pdf"})

    def __init__(
        self,
        api_key: str | None = None,
        safe_max_pages: int | None = None,
        safe_max_bytes: int | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        error_strategy: str | None = None,
    ) -> None:
        self._api_key: str | None = api_key
        self._safe_max_pages = safe_max_pages
        self._safe_max_bytes = safe_max_bytes
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._error_strategy = error_strategy

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def convert(self, source: Path) -> IntermediateDocument:
        """Convert a PDF file to the intermediate format.

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

        logger.info("Reading PDF: %s", source.name)
        try:
            pdf_bytes = source.read_bytes()
        except OSError as exc:
            raise ConversionError(f"Cannot read PDF file {source}: {exc}") from exc

        limits = self._resolve_limits()

        must_split = needs_splitting(pdf_bytes, limits.safe_max_pages, limits.safe_max_bytes)

        if not must_split:
            pages = self._convert_chunk(
                pdf_bytes, api_key, source.name, page_offset=0, limits=limits
            )
            logger.info("Converted %d page(s) from %s", len(pages), source.name)
            return IntermediateDocument(pages=pages)

        try:
            chunks = split_pdf(
                pdf_bytes, limits.safe_max_pages, limits.safe_max_bytes, source_name=source.name
            )
        except PdfSplitError:
            # PdfSplitError ya es una ConversionError; se deja propagar tal cual.
            raise
        logger.info(
            "PDF %s excede los límites seguros; dividido en %d chunk(s): %s",
            source.name,
            len(chunks),
            ", ".join(f"[{c.start_page}-{c.end_page - 1}]" for c in chunks),
        )

        total_pages_original = chunks[-1].end_page if chunks else 0
        all_pages: list[IntermediatePage] = []
        missing_page_ranges: list[tuple[int, int]] = []

        for i, chunk in enumerate(chunks):
            etiqueta = (
                f"chunk {i + 1}/{len(chunks)} (páginas originales "
                f"{chunk.start_page}-{chunk.end_page - 1} de {source.name})"
            )
            logger.info(
                "Enviando %s: %d página(s), %d bytes",
                etiqueta,
                chunk.page_count,
                len(chunk.pdf_bytes),
            )
            try:
                paginas_chunk = self._convert_chunk(
                    chunk.pdf_bytes,
                    api_key,
                    etiqueta,
                    page_offset=chunk.start_page,
                    limits=limits,
                    error_label=etiqueta,
                )
            except ConversionError as exc:
                if limits.error_strategy == "best_effort":
                    logger.warning(
                        "%s falló y será omitido en modo best-effort. Error: %s",
                        etiqueta,
                        exc,
                    )
                    missing_page_ranges.append((chunk.start_page, chunk.end_page))
                    continue
                raise ConversionError(
                    f"Fallo al convertir {etiqueta}. Rango de páginas afectado: "
                    f"[{chunk.start_page}, {chunk.end_page}) de {total_pages_original} "
                    f"páginas totales en {source.name}. Error original: {exc}\n"
                    f"{_BEST_EFFORT_HINT}"
                ) from exc
            all_pages.extend(paginas_chunk)

        if not missing_page_ranges:
            logger.info("Converted %d page(s) from %s", len(all_pages), source.name)
            return IntermediateDocument(pages=all_pages)

        missing_count = sum(end - start for start, end in missing_page_ranges)
        logger.warning(
            "Documento convertido en modo best-effort: %d página(s) omitidas en "
            "%d rango(s): %s",
            missing_count,
            len(missing_page_ranges),
            missing_page_ranges,
        )
        return IntermediateDocument(pages=all_pages, missing_page_ranges=missing_page_ranges)

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

        if error_strategy not in ("fail_fast", "best_effort"):
            raise ConversionError(
                f"Valor inválido para mistral_ocr_error_strategy: {error_strategy!r}. "
                f"Valores válidos: 'fail_fast', 'best_effort'."
            )

        return _OcrLimits(
            safe_max_pages=safe_max_pages,
            safe_max_bytes=safe_max_bytes,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            error_strategy=error_strategy,
        )

    def _convert_chunk(
        self,
        pdf_bytes: bytes,
        api_key: str,
        chunk_label: str,
        page_offset: int,
        limits: _OcrLimits,
        error_label: str | None = None,
    ) -> list[IntermediatePage]:
        """Envía un chunk de PDF a Mistral OCR y mapea su respuesta.

        Parameters
        ----------
        pdf_bytes:
            Bytes del chunk (PDF autocontenido) a enviar.
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

        Returns
        -------
        list[IntermediatePage]
        """
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

        payload = {
            "model": _MISTRAL_OCR_MODEL,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        logger.info("Calling Mistral OCR API for %s", chunk_label)
        data = self._post_to_mistral_with_retry(
            payload, headers, limits, chunk_label, error_label=error_label
        )

        if "pages" not in data:
            raise ConversionError(
                f"Unexpected Mistral OCR API response (no 'pages' key): {data}"
            )

        return [
            self._map_page(page_data, index_offset=page_offset) for page_data in data["pages"]
        ]

    def _post_to_mistral_with_retry(
        self,
        payload: dict,
        headers: dict,
        limits: _OcrLimits,
        chunk_label: str,
        error_label: str | None = None,
    ) -> dict:
        """Ejecuta el POST a Mistral OCR con reintento/backoff exponencial
        ante errores de red o códigos HTTP reintentables.

        Parameters
        ----------
        payload:
            Cuerpo JSON de la solicitud.
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
        dict
            Cuerpo JSON de la respuesta exitosa.

        Raises
        ------
        ConversionError
            Si se agotan los reintentos o el error no es reintentable.
        """
        import httpx

        last_request_exc: httpx.RequestError | None = None
        for attempt in range(1, limits.max_retries + 1):
            try:
                response = httpx.post(
                    _MISTRAL_OCR_URL,
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(300.0),
                )
            except httpx.RequestError as exc:
                last_request_exc = exc
                if attempt < limits.max_retries:
                    backoff = limits.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Network error calling Mistral OCR API for %s (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        chunk_label,
                        attempt,
                        limits.max_retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise ConversionError(
                    f"Network error calling Mistral OCR API for {chunk_label} "
                    f"after {limits.max_retries} attempt(s): {exc}"
                ) from exc

            if response.is_error:
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < limits.max_retries:
                    backoff = limits.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Mistral OCR API returned HTTP %d for %s (attempt %d/%d). "
                        "Retrying in %.1fs...",
                        response.status_code,
                        chunk_label,
                        attempt,
                        limits.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
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
        for img_idx, img in enumerate(page_data.get("images", [])):
            raw_b64 = img.get("image_base64") or ""
            images.append(
                PageImage(
                    index=img_idx,
                    base64=raw_b64,
                    media_type="image/png",
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
