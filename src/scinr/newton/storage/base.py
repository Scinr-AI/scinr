"""
storage/base.py — Abstract repository interfaces.

All storage backends must implement these ABCs so that the rest of the
pipeline can remain backend-agnostic.

Multi-tenancy
-------------
Every record carries the same tenant / provenance fields as the graph:

- ``tenant_id`` — **stored** value (:func:`~scinr.newton.utils.tenancy.tenant_key`):
  the tenant, or ``"__public__"`` for a public upload — never ``null``.
- ``created_by_user_id`` / ``job_id`` — provenance of the upload (scalars).

**Writes** take ``tenant_id`` in its public-API form: ``None`` and
``"__public__"`` both mean public and are stored as ``"__public__"``.

**Reads and deletes** take the same four keyword-only filters as the graph
navigation API (see :mod:`scinr.newton.utils.scope`)::

    tenant_id: str | None = None                            # None = all tenants
    include_public: bool = False                            # add the public records
    created_by_user_id: str | Sequence[str] | None = None   # IN
    job_id: str | Sequence[str] | None = None               # IN

The library does not impose a restriction — ``tenant_id=None`` reads across
every tenant (legacy records without a tenant included). A multi-tenant API
layer should always pass its tenant. A record outside the requested scope
behaves exactly as a record that does not exist.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from scinr.newton.storage.models import ConvertedPageRecord, RawFileRecord


class RawFileRepository(ABC):
    """Almacena el fichero original tal cual fue recibido (binario)."""

    @abstractmethod
    async def store(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        folder_path: str | None,
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
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
        tenant_id:
            Tenant propietario (``None`` o ``"__public__"`` = público). Se
            guarda como :func:`~scinr.newton.utils.tenancy.tenant_key`.
        created_by_user_id, job_id:
            Procedencia de la subida (se guardan tal cual; ``None`` = sin dato).

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
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
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
        filename, content_type, folder_path, tenant_id, created_by_user_id, job_id:
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
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )

    @abstractmethod
    async def get(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> RawFileRecord | None:
        """Devuelve la metadata de *raw_file_id*, o ``None`` si no existe o
        queda fuera del scope pedido (ambos casos son indistinguibles).

        Un ``raw_file_id`` inválido para el backend también devuelve ``None``.

        Raises
        ------
        ScopeError
            Si los filtros son inválidos (``tenant_id=""``, lista vacía).
        """

    @abstractmethod
    async def open(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> AsyncIterator[bytes] | None:
        """Abre el binario original de *raw_file_id* para leerlo en streaming.

        Devuelve un iterador asíncrono de bloques de bytes (``async for chunk
        in stream``), o ``None`` si el registro no existe o queda fuera del
        scope (mismas reglas que :meth:`get`). El backend debe localizar el
        binario a partir del registro ya filtrado, nunca a partir de un id
        recibido del llamador.

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        StorageError
            Si el backend no guarda binarios.
        """

    @abstractmethod
    async def open_with_record(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[RawFileRecord, AsyncIterator[bytes]] | None:
        """Como :meth:`open`, pero devuelve también la metadata del registro:
        ``(record, stream)`` con una sola lectura del registro, o ``None`` si
        no existe o queda fuera del scope.

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        StorageError
            Si el backend no guarda binarios, o el binario del registro falta.
        """

    @abstractmethod
    async def list_raw_files(
        self,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
        folder_path: str | None = None,
        filename: str | None = None,
    ) -> list[RawFileRecord]:
        """Inventario de ficheros originales dentro del scope, ordenado por
        ``stored_at`` ascendente.

        *folder_path* / *filename* filtran por igualdad exacta (``None`` = sin
        filtro).

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        """

    @abstractmethod
    async def delete(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> None:
        """Borra el fichero binario (y su metadata) identificado por *raw_file_id*.

        Solo borra si el registro está dentro del scope pedido; fuera de él
        se comporta como si no existiera.

        Debe ser idempotente: si *raw_file_id* no existe (ya borrado, ID
        inválido, fuera de scope, o llamada repetida), no debe lanzar
        excepción — solo loggear y no hacer nada.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` a borrar.
        tenant_id, include_public, created_by_user_id, job_id:
            Filtros de scope (ver el docstring del módulo).

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
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
        *,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
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
        tenant_id, created_by_user_id, job_id:
            Igual que en :meth:`RawFileRepository.store` (deben coincidir con
            los del fichero original).

        Returns
        -------
        str
            El ``page_id``: identificador único de la página persistida.
        """

    @abstractmethod
    async def get_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        """Recupera las páginas de un fichero por su raw_file_id, dentro del scope.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` cuyas páginas se
            quieren recuperar.
        tenant_id, include_public, created_by_user_id, job_id:
            Filtros de scope (ver el docstring del módulo).

        Returns
        -------
        list[ConvertedPageRecord]
            Lista ordenada por ``page_index`` ascendente.
            Puede estar vacía si aún no se han almacenado páginas o si quedan
            fuera del scope.

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        """

    @abstractmethod
    async def get_pages_by_ids(
        self,
        page_ids: Sequence[str],
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[ConvertedPageRecord]:
        """Recupera las páginas cuyos ``page_id`` están en *page_ids*, dentro
        del scope.

        Es la lectura directa del texto de un nodo de estructura (sus
        ``source_page_ids``), sin cargar el resto de páginas del documento.

        Parameters
        ----------
        page_ids:
            ``page_id`` devueltos por :meth:`store_page`. Los que no existen,
            quedan fuera del scope o no son válidos para el backend se ignoran.
            Con una lista vacía devuelve ``[]``.
        tenant_id, include_public, created_by_user_id, job_id:
            Filtros de scope (ver el docstring del módulo).

        Returns
        -------
        list[ConvertedPageRecord]
            Las páginas encontradas, ordenadas por ``page_index`` ascendente.

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        """

    @abstractmethod
    async def delete_pages(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Borra las páginas asociadas a *raw_file_id* dentro del scope.

        No es un error que no existan páginas para ese ``raw_file_id``
        (devuelve ``0`` en ese caso, sin lanzar excepción).

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` cuyas páginas se
            quieren borrar.
        tenant_id, include_public, created_by_user_id, job_id:
            Filtros de scope (ver el docstring del módulo).

        Returns
        -------
        int
            Número de páginas borradas (``0`` si no había ninguna).

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        """
