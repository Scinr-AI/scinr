"""
converters/csv.py — CSV files are handled by the tabular ingestion pipeline.

CSV files do not require conversion to an intermediate format.
Use: python main.py --stage tabular --input-raw <carpeta>
"""

from __future__ import annotations

from pathlib import Path

from scinr.newton.converters.base import BaseConverter, ConversionError, IntermediateDocument


class CsvConverter(BaseConverter):
    """Redirect handler for ``.csv`` files.

    CSV files are processed directly and efficiently by the tabular ingestion
    pipeline. This converter raises a :class:`ConversionError` with a clear
    redirect message so that callers know to use the correct pipeline stage.
    """

    supported_extensions: frozenset[str] = frozenset({"csv"})

    def convert(self, source: Path) -> IntermediateDocument:
        raise ConversionError(
            f"El fichero CSV '{source.name}' es totalmente compatible con el pipeline de "
            "ingesta tabular, que lo procesa de forma directa y eficiente sin conversión previa. "
            "Usa: python main.py --stage tabular --input-raw <carpeta>. "
            "No utilices el módulo converters para ficheros CSV."
        )

