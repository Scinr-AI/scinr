"""
converters/pdf_splitter.py — PDF chunking helpers (no network I/O).

Divide un PDF en sub-rangos de páginas contiguos que respetan un límite
máximo de páginas y de tamaño en bytes por chunk, para poder enviarlos
por separado a APIs con límites (p.ej. Mistral OCR: máx. 1000 páginas /
50 MB por solicitud). Toda la lógica aquí es pura y local — no hace
llamadas de red.

Hay dos APIs:

* ``split_pdf(bytes, ...)`` — basada en bytes; materializa **todos** los
  chunks en una lista. Útil para PDFs pequeños o ya en memoria.
* ``probe_pdf(path)`` + ``iter_pdf_chunks(path, ...)`` — basada en ruta y
  perezosa: produce **un chunk cada vez** y nunca lee el fichero completo,
  de modo que el pico de memoria depende del tamaño de un chunk y no del
  del archivo (un PDF de varios GB usa la misma memoria que uno de 100 MB).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

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


def probe_pdf(path: Path) -> tuple[int, int]:
    """Return ``(size_bytes, total_pages)`` of the PDF at *path* without reading it whole.

    Only the file size (``stat``) and the page index (via a ``PdfReader`` over
    the file handle, which reads the cross-reference data, not the page
    contents) are needed to decide whether the document must be split.

    Raises
    ------
    ConversionError
        If the file cannot be read, or pypdf cannot open it (corrupt or
        encrypted PDF).
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ConversionError(f"Cannot read PDF file {path}: {exc}") from exc
    try:
        with path.open("rb") as fh:
            total_pages = len(PdfReader(fh).pages)
    except Exception as exc:
        raise ConversionError(f"Cannot read PDF to count pages: {exc}") from exc
    return size, total_pages


def initial_window_pages(size_bytes: int, total_pages: int, max_pages: int, max_bytes: int) -> int:
    """First-guess pages per window, from the document's average bytes per page.

    Aims at ~80 % of *max_bytes* per window so that, in the common case, a
    window already fits and is never serialised twice. Capped by *max_pages*.
    A window that still exceeds *max_bytes* once serialised is bisected.
    """
    if total_pages <= 0 or size_bytes <= 0:
        return max(1, max_pages)
    avg_page_bytes = size_bytes / total_pages
    return max(1, min(max_pages, int(0.8 * max_bytes / avg_page_bytes)))


def iter_pdf_chunks(
    path: Path,
    max_pages: int,
    max_bytes: int,
    *,
    source_name: str = "<document>",
) -> Iterator[PdfChunk]:
    """Lazily yield the chunks of the PDF at *path*, one at a time.

    Same contract as :func:`split_pdf` (contiguous chunks covering ``[0, N)``
    without gaps or overlaps, each within *max_pages* and *max_bytes*), but:

    * the file is never read whole: a fresh ``PdfReader`` is opened on the
      file for each window of pages and closed when the window is done
      (pypdf caches the content of every page it reads, so reusing one reader
      would make memory grow with the file);
    * chunks are produced on demand — the consumer must send a chunk and drop
      it before asking for the next one;
    * the first window is sized from the average bytes per page (see
      :func:`initial_window_pages`) instead of *max_pages*, so a window is not
      serialised just to be discarded and bisected.

    Parameters
    ----------
    path:
        PDF file to split.
    max_pages, max_bytes, source_name:
        As in :func:`split_pdf`.

    Raises
    ------
    PdfSplitError
        If a single page exceeds ``max_bytes`` once serialised. Unlike
        :func:`split_pdf`, this can be raised *after* earlier chunks were
        already yielded.
    ConversionError
        If pypdf fails to open the document or to read/serialise any page
        (e.g. an encrypted PDF that opens but fails when its pages are read).
    """
    try:
        size_bytes, total_pages = _probe_raw(path)
        window = initial_window_pages(size_bytes, total_pages, max_pages, max_bytes)
        for start in range(0, total_pages, window):
            end = min(start + window, total_pages)
            with path.open("rb") as fh:
                reader = PdfReader(fh)
                yield from _iter_bisect_window(reader, start, end, max_bytes, source_name)
    except PdfSplitError:
        # Intencional — no envolver de nuevo.
        raise
    except Exception as exc:
        raise ConversionError(f"Cannot split PDF {source_name}: {exc}") from exc


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


def _probe_raw(path: Path) -> tuple[int, int]:
    """``(size_bytes, total_pages)`` letting native errors propagate (the
    caller wraps them with its own message)."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        return size, len(PdfReader(fh).pages)


def _iter_bisect_window(
    reader: PdfReader,
    start: int,
    end: int,
    max_bytes: int,
    source_name: str,
) -> Iterator[PdfChunk]:
    """Lazy version of :func:`_bisect_window`: serialise [start, end); if it
    exceeds max_bytes, bisect it and yield each half in order."""
    pdf_bytes = _serialize_page_range(reader, start, end)
    if len(pdf_bytes) <= max_bytes:
        yield PdfChunk(start, end, pdf_bytes)
        return
    if end - start == 1:
        raise PdfSplitError(
            f"La página {start} de {source_name} pesa {len(pdf_bytes)} bytes "
            f"tras serializarse sola, lo cual excede el límite de {max_bytes} "
            f"bytes. No puede subdividirse más."
        )
    del pdf_bytes  # too big: drop it before serialising the halves
    medio = start + (end - start) // 2
    yield from _iter_bisect_window(reader, start, medio, max_bytes, source_name)
    yield from _iter_bisect_window(reader, medio, end, max_bytes, source_name)


def _serialize_page_range(reader: PdfReader, start: int, end: int) -> bytes:
    """Serializa las páginas [start, end) del reader a un PDF autocontenido."""
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()
