"""
storage/base.py — Abstract repository interfaces.

All storage backends must implement these ABCs so that the rest of the
pipeline can remain backend-agnostic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

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

        Notes
        -----
        Los backends propios pueden sobrescribir además :meth:`store_file`
        para subir el fichero en streaming sin materializarlo en memoria.
        """

    async def store_file(
        self,
        path: Path,
        filename: str,
        content_type: str,
        folder_path: str | None,
    ) -> str:
        """Persiste el fichero situado en *path* y devuelve su raw_file_id.

        Implementación por defecto: lee el fichero completo y delega en
        :meth:`store`. Los backends capaces de hacer streaming (p.ej. MongoDB/GridFS)
        la sobrescriben para no mantener el fichero entero en memoria. No es
        abstracto para no romper backends propios que solo implementen :meth:`store`.

        Parameters
        ----------
        path:
            Ruta del fichero a persistir.
        filename, content_type, folder_path:
            Igual que en :meth:`store`.

        Returns
        -------
        str
            El ``raw_file_id`` asignado por el backend.
        """
        content = await asyncio.to_thread(path.read_bytes)
        return await self.store(
            filename=filename,
            content=content,
            content_type=content_type,
            folder_path=folder_path,
        )

    @abstractmethod
    async def delete(self, raw_file_id: str) -> None:
        """Borra el fichero binario (y su metadata) identificado por *raw_file_id*.

        Debe ser idempotente: si *raw_file_id* no existe (ya borrado, ID
        inválido, o llamada repetida), no debe lanzar excepción — solo
        loggear y no hacer nada.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` a borrar.
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

    @abstractmethod
    async def delete_pages(self, raw_file_id: str) -> int:
        """Borra todas las páginas asociadas a *raw_file_id*.

        No es un error que no existan páginas para ese ``raw_file_id``
        (devuelve ``0`` en ese caso, sin lanzar excepción).

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` cuyas páginas se
            quieren borrar.

        Returns
        -------
        int
            Número de páginas borradas (``0`` si no había ninguna).
        """
