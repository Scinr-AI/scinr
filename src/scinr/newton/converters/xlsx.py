"""
converters/xlsx.py — XLSX/XLS files are handled by the tabular ingestion pipeline.

XLSX/XLS files do not require conversion to an intermediate format.
Use: python main.py --stage tabular --input-raw <carpeta>
"""

from __future__ import annotations

from pathlib import Path

from scinr.newton.converters.base import BaseConverter, ConversionError, IntermediateDocument


class XlsxConverter(BaseConverter):
    """Redirect handler for ``.xlsx`` and ``.xls`` files.

    XLSX/XLS files are processed directly and efficiently by the tabular ingestion
    pipeline. This converter raises a :class:`ConversionError` with a clear
    redirect message so that callers know to use the correct pipeline stage.
    """

    supported_extensions: frozenset[str] = frozenset({"xlsx", "xls"})

    def convert(self, source: Path) -> IntermediateDocument:
        raise ConversionError(
            f"El fichero XLSX/XLS '{source.name}' es totalmente compatible con el pipeline de "
            "ingesta tabular, que lo procesa de forma directa y eficiente sin conversión previa. "
            "Usa: python main.py --stage tabular --input-raw <carpeta>. "
            "No utilices el módulo converters para ficheros XLSX/XLS."
        )

