"""
converters — Multi-format document converter for scinr-ingest.

Converts PDF, DOCX, XLSX, CSV, PPTX, HTML, TXT/MD, and API data
to the intermediate JSON format consumed by the scinr-ingest Stage 1 pipeline.

Intermediate format
-------------------
{
    "pages": [
        {
            "index": 0,
            "markdown": "## Título\n\nTexto...",
            "images": [{"index": 0, "base64": "...", "media_type": "image/png", "description": ""}],
            "dimensions": {"dpi": null, "height": null, "width": null},
            "tables": [],
            "hyperlinks": [],
            "header": null,
            "footer": null,
            "confidence_scores": null
        }
    ]
}
"""

from __future__ import annotations

__version__ = "0.1.0"

from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    ConverterError,
    IntermediateDocument,
    IntermediatePage,
    PageDimensions,
    PageImage,
    UnsupportedFormatError,
)
from scinr.newton.converters.registry import get_converter, list_supported_extensions

__all__ = [
    "BaseConverter",
    "ConversionError",
    "ConverterError",
    "IntermediateDocument",
    "IntermediatePage",
    "PageDimensions",
    "PageImage",
    "UnsupportedFormatError",
    "get_converter",
    "list_supported_extensions",
]
