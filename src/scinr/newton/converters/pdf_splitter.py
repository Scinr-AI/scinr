"""
converters/pdf_splitter.py — PDF chunking helpers (no network I/O).

Divide un PDF en sub-rangos de páginas contiguos que respetan un límite
máximo de páginas y de tamaño en bytes por chunk, para poder enviarlos
por separado a APIs con límites (p.ej. Mistral OCR: máx. 1000 páginas /
50 MB por solicitud). Toda la lógica aquí es pura y local — no hace
llamadas de red.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader, PdfWriter

from scinr.newton.converters.base import ConversionError


class PdfSplitError(ConversionError):
    """Un PDF no puede dividirse por debajo de max_bytes ni a nivel de
    1 sola página. El mensaje incluye el número de página absoluta
    0-based del documento original y su tamaño serializado en bytes."""


@dataclass(frozen=True)
class PdfChunk:
    """Sub-rango contiguo de páginas del PDF original, ya serializado.

    Parameters
    ----------
    start_page:
        Índice 0-based inclusive relativo al documento ORIGINAL.
    end_page:
        Índice 0-based exclusivo relativo al documento ORIGINAL.
    pdf_bytes:
        PDF válido y autocontenido con solo esas páginas.
    """

    start_page: int
    end_page: int
    pdf_bytes: bytes

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page


def count_pdf_pages(pdf_bytes: bytes) -> int:
    """Return the number of pages in *pdf_bytes*.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file contents.

    Returns
    -------
    int
        Number of pages.

    Raises
    ------
    ConversionError
        If pypdf cannot open the document (corrupt or encrypted PDF).
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception as exc:
        raise ConversionError(f"Cannot read PDF to count pages: {exc}") from exc


def needs_splitting(pdf_bytes: bytes, max_pages: int, max_bytes: int) -> bool:
    """Return True if *pdf_bytes* exceeds either *max_pages* or *max_bytes*.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file contents.
    max_pages:
        Maximum number of pages allowed per chunk.
    max_bytes:
        Maximum number of bytes allowed per chunk.

    Returns
    -------
    bool
    """
    if len(pdf_bytes) > max_bytes:
        return True
    return count_pdf_pages(pdf_bytes) > max_pages


def split_pdf(
    pdf_bytes: bytes,
    max_pages: int,
    max_bytes: int,
    *,
    source_name: str = "<document>",
) -> list[PdfChunk]:
    """Split *pdf_bytes* into contiguous chunks satisfying both limits.

    Algoritmo: ventaneo inicial por páginas de tamaño ``max_pages``
    recorriendo todo el documento; cada ventana se serializa y se mide
    su tamaño real; si excede ``max_bytes``, se bisecciona
    recursivamente la ventana en dos mitades y se reintenta cada mitad
    independientemente, hasta que cada resultado cumpla ``max_bytes``.
    Si una ventana de exactamente 1 página ya excede ``max_bytes`` tras
    serializarse sola, se lanza :class:`PdfSplitError` (no se puede
    dividir más).

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file contents to split.
    max_pages:
        Maximum number of pages allowed per chunk.
    max_bytes:
        Maximum number of bytes allowed per chunk (serialized size).
    source_name:
        Human-readable name of the source document, used in error
        messages.

    Returns
    -------
    list[PdfChunk]
        Chunks ordenados ascendentemente, cubriendo ``[0, N)`` sin
        huecos ni superposiciones. Longitud 1 si el documento ya
        cumple ambos límites.

    Raises
    ------
    PdfSplitError
        Si una sola página excede ``max_bytes`` tras serializarse.
    ConversionError
        Si pypdf falla al abrir el documento, al contar sus páginas, o al
        acceder/serializar el contenido de alguna página (p.ej. un PDF
        cifrado que abre correctamente pero lanza una excepción nativa de
        pypdf más adelante al leer sus páginas).
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        chunks: list[PdfChunk] = []
        inicio = 0
        while inicio < total_pages:
            fin_tentativo = min(inicio + max_pages, total_pages)
            chunks.extend(_bisect_window(reader, inicio, fin_tentativo, max_bytes, source_name))
            inicio = fin_tentativo
    except PdfSplitError:
        # Intencional — no envolver de nuevo.
        raise
    except Exception as exc:
        raise ConversionError(
            f"Cannot split PDF {source_name}: {exc}"
        ) from exc
    return chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _bisect_window(
    reader: PdfReader,
    start: int,
    end: int,
    max_bytes: int,
    source_name: str,
) -> list[PdfChunk]:
    """Serializa la ventana [start, end); si excede max_bytes, la bisecciona
    recursivamente hasta que cada mitad cumpla el límite o quede en 1
    página (en cuyo caso, si aún excede, se lanza PdfSplitError)."""
    pdf_bytes = _serialize_page_range(reader, start, end)
    if len(pdf_bytes) <= max_bytes:
        return [PdfChunk(start, end, pdf_bytes)]
    if end - start == 1:
        raise PdfSplitError(
            f"La página {start} de {source_name} pesa {len(pdf_bytes)} bytes "
            f"tras serializarse sola, lo cual excede el límite de {max_bytes} "
            f"bytes. No puede subdividirse más."
        )
    medio = start + (end - start) // 2
    return _bisect_window(reader, start, medio, max_bytes, source_name) + _bisect_window(
        reader, medio, end, max_bytes, source_name
    )


def _serialize_page_range(reader: PdfReader, start: int, end: int) -> bytes:
    """Serializa las páginas [start, end) del reader a un PDF autocontenido."""
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()
