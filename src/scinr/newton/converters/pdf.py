"""
converters/pdf.py — PDF to Markdown converter via Mistral OCR API.

Encodes the PDF as base64 and submits it to the Mistral OCR endpoint
(``POST https://api.mistral.ai/v1/ocr``).  The API response already
matches the intermediate document format, so each page is mapped
directly to an ``IntermediatePage``.
"""

from __future__ import annotations

import base64
import logging
import os
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

logger = logging.getLogger(__name__)

load_dotenv()

_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
_MISTRAL_OCR_MODEL = "mistral-ocr-latest"


class PdfConverter(BaseConverter):
    """Convert ``.pdf`` files to the intermediate format via Mistral OCR.

    Sends the PDF to the Mistral OCR API and maps the response to an
    :class:`~converters.base.IntermediateDocument`.  Each page returned
    by Mistral becomes one :class:`~converters.base.IntermediatePage`.

    Parameters
    ----------
    api_key:
        Mistral API key.  If ``None``, the value of the environment
        variable ``MISTRAL_API_KEY`` is used at conversion time.
    """

    supported_extensions: frozenset[str] = frozenset({"pdf"})

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key: str | None = api_key

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
            per PDF page recognised by Mistral OCR.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist.
        ConversionError
            If the Mistral API key is not available, the HTTP request
            fails, or the response is in an unexpected format.
        """
        try:
            import httpx
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

        logger.info("Calling Mistral OCR API for %s", source.name)
        try:
            response = httpx.post(
                _MISTRAL_OCR_URL,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(300.0),
            )
        except httpx.RequestError as exc:
            raise ConversionError(
                f"Network error calling Mistral OCR API: {exc}"
            ) from exc

        if response.is_error:
            raise ConversionError(
                f"Mistral OCR API returned HTTP {response.status_code}: "
                f"{response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ConversionError(
                f"Cannot parse Mistral OCR API response as JSON: {exc}"
            ) from exc

        if "pages" not in data:
            raise ConversionError(
                f"Unexpected Mistral OCR API response (no 'pages' key): {data}"
            )

        pages = [self._map_page(page_data) for page_data in data["pages"]]
        logger.info("Converted %d page(s) from %s", len(pages), source.name)
        return IntermediateDocument(pages=pages)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_page(self, page_data: dict) -> IntermediatePage:
        """Map a single Mistral OCR page dict to an :class:`IntermediatePage`.

        Parameters
        ----------
        page_data:
            A single element from the ``pages`` list in the Mistral OCR
            response.

        Returns
        -------
        IntermediatePage
        """
        index: int = page_data.get("index", 0)
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
