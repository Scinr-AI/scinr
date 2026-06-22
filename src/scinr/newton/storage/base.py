"""
storage/base.py — Abstract repository interfaces.

All storage backends must implement these ABCs so that the rest of the
pipeline can remain backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scinr.newton.storage.models import ConvertedPageRecord


class RawFileRepository(ABC):
    """Almacena el fichero original tal cual fue recibido (binario)."""

    @abstractmethod
    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None,
    ) -> str:
        """Persiste el fichero y devuelve su raw_file_id (str).

        Parameters
        ----------
        filename:
            Nombre del fichero original (p.ej. ``"3.2.P.1.pdf"``).
        content:
            Contenido binario del fichero.
        content_type:
            MIME type del fichero (p.ej. ``"application/pdf"``).
        folder_path:
            Ruta relativa de la carpeta contenedora desde la raíz de ingesta,
            o ``None`` si el fichero está en la raíz.

        Returns
        -------
        str
            El ``raw_file_id``: representación en cadena del identificador
            único asignado por el backend de almacenamiento.
        """


class PageRepository(ABC):
    """Almacena las páginas convertidas (markdown) de un documento."""

    @abstractmethod
    async def store_page(
        self,
        raw_file_id: str,
        filename: str,
        folder_path: str | None,
        page_index: int,
        markdown: str,
    ) -> str:
        """Persiste una página y devuelve su page_id (str).

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` al que pertenece
            esta página.
        filename:
            Stem del fichero sin extensión (p.ej. ``"3.2.P.1"``).
        folder_path:
            Ruta relativa de la carpeta contenedora, o ``None``.
        page_index:
            Índice 0-based de la página, idéntico a
            :attr:`~converters.base.IntermediatePage.index`.
        markdown:
            Texto completo de la página en formato Markdown.

        Returns
        -------
        str
            El ``page_id``: identificador único de la página persistida.
        """

    @abstractmethod
    async def get_pages(self, raw_file_id: str) -> list[ConvertedPageRecord]:
        """Recupera todas las páginas de un fichero por su raw_file_id.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` cuyas páginas se
            quieren recuperar.

        Returns
        -------
        list[ConvertedPageRecord]
            Lista ordenada por ``page_index`` ascendente.
            Puede estar vacía si aún no se han almacenado páginas.
        """
