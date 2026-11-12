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

from scinr.newton.storage.models import ConvertedPageRecord, PageImageRecord, RawFileRecord


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

    # ------------------------------------------------------------------
    # Imágenes de página (opcionales, D4)
    # ------------------------------------------------------------------
    #
    # Los métodos de este bloque no son abstractos: un backend propio que no
    # los implemente sigue funcionando, no guarda imágenes y la ingesta avisa
    # una vez por proceso de que las imágenes descritas no se conservan.

    @property
    def stores_images(self) -> bool:
        """``True`` si el backend guarda imágenes de página.

        Si es ``False``, la ingesta no llama a los demás métodos de escritura
        de imágenes y avisa una vez por proceso de que las imágenes descritas
        no se conservan.
        """
        return False

    async def store_image(
        self,
        raw_file_id: str,
        page_index: int,
        image_index: int,
        content: bytes,
        *,
        kind: str,
        content_type: str,
        description: str,
        filename: str,
        folder_path: str | None,
        width: int | None = None,
        height: int | None = None,
        time_ms: int | None = None,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Guarda el original de una imagen descrita y devuelve su image_id.

        La escritura guarda el tenant como
        :func:`~scinr.newton.utils.tenancy.tenant_key` (``None`` y
        ``"__public__"`` = público), igual que :meth:`store`.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` al que pertenece la
            imagen.
        page_index:
            Índice 0-based de la página, idéntico a
            :attr:`~converters.base.IntermediatePage.index` (el índice del
            tramo, en vídeo).
        image_index:
            Índice de la imagen dentro de su página, es decir el ``index``
            del bloque ``<image index="N">`` de esa página.
        content:
            Bytes originales de la imagen (sin reescalar).
        kind:
            ``"embedded"`` para una imagen incrustada en el documento,
            ``"frame"`` para un fotograma de vídeo.
        content_type:
            MIME type de la imagen (p.ej. ``"image/png"``, ``"image/jpeg"``).
        description:
            La misma descripción que aparece en el bloque del markdown.
        filename:
            Nombre del documento fuente, con su extensión.
        folder_path:
            Ruta relativa de la carpeta contenedora, o ``None``.
        width, height:
            Dimensiones de la imagen en píxeles, o ``None``.
        time_ms:
            Solo fotogramas: instante del fotograma en el vídeo.
        tenant_id, created_by_user_id, job_id:
            Igual que en :meth:`store`.

        Returns
        -------
        str
            El ``image_id``: representación en cadena del identificador
            único asignado por el backend.

        Notes
        -----
        Los backends capaces de hacer streaming pueden sobrescribir además
        :meth:`store_image_file` para subir el fichero sin materializarlo en
        memoria, igual que :meth:`store_file` respecto de :meth:`store`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not store page images "
            "(RawFileRepository.stores_images is False)"
        )

    async def store_image_file(
        self,
        path: Path,
        raw_file_id: str,
        page_index: int,
        image_index: int,
        *,
        kind: str,
        content_type: str,
        description: str,
        filename: str,
        folder_path: str | None,
        width: int | None = None,
        height: int | None = None,
        time_ms: int | None = None,
        tenant_id: str | None = None,
        created_by_user_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Guarda la imagen situada en *path* y devuelve su image_id.

        Implementación por defecto: lee el fichero completo y delega en
        :meth:`store_image`. Un backend con streaming la sobrescribe para no
        mantener la imagen entera en memoria, igual que :meth:`store_file`
        respecto de :meth:`store`.

        Parameters
        ----------
        path:
            Ruta del fichero de imagen a guardar.
        raw_file_id, page_index, image_index, kind, content_type, description,
        filename, folder_path, width, height, time_ms, tenant_id,
        created_by_user_id, job_id:
            Igual que en :meth:`store_image`.

        Returns
        -------
        str
            El ``image_id`` asignado por el backend.
        """
        content = await asyncio.to_thread(path.read_bytes)
        return await self.store_image(
            raw_file_id,
            page_index,
            image_index,
            content,
            kind=kind,
            content_type=content_type,
            description=description,
            filename=filename,
            folder_path=folder_path,
            width=width,
            height=height,
            time_ms=time_ms,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )

    async def list_images(
        self,
        raw_file_id: str,
        *,
        page_indexes: Sequence[int] | None = None,
        kind: str | None = None,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> list[PageImageRecord]:
        """Inventario de las imágenes guardadas de *raw_file_id*, dentro del
        scope, ordenadas por ``(page_index, image_index)``.

        *page_indexes* filtra por esos índices de página (``None`` = todas) y
        *kind* por igualdad exacta sobre el tipo (``None`` = sin filtro).

        La implementación por defecto no guarda nada y devuelve ``[]``.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` cuyas imágenes se
            quieren listar.
        page_indexes, kind:
            Filtros de contenido (ver arriba).
        tenant_id, include_public, created_by_user_id, job_id:
            Filtros de scope (ver el docstring del módulo).

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        """
        return []

    async def open_image(
        self,
        image_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> tuple[PageImageRecord, AsyncIterator[bytes]] | None:
        """Abre el binario de la imagen *image_id*, dentro del scope.

        Igual que :meth:`open_with_record`: devuelve ``(record, stream)`` con
        una sola lectura del registro, o ``None`` si no existe o queda fuera
        del scope (ambos casos son indistinguibles). El backend debe localizar
        el binario a partir del registro ya filtrado, nunca a partir de un id
        recibido del llamador.

        La implementación por defecto no guarda nada y devuelve ``None``.

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        StorageError
            Si el registro existe pero falta el binario.
        """
        return None

    async def delete_images(
        self,
        raw_file_id: str,
        *,
        tenant_id: str | None = None,
        include_public: bool = False,
        created_by_user_id: str | Sequence[str] | None = None,
        job_id: str | Sequence[str] | None = None,
    ) -> int:
        """Borra las imágenes guardadas de *raw_file_id*, dentro del scope.

        Debe ser idempotente: si no hay ninguna imagen en el scope (ya
        borradas, fuera de scope o llamada repetida), devuelve ``0`` sin
        lanzar excepción. La implementación por defecto no guarda nada y
        devuelve ``0``.

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` cuyas imágenes se
            quieren borrar.
        tenant_id, include_public, created_by_user_id, job_id:
            Filtros de scope (ver el docstring del módulo).

        Returns
        -------
        int
            Número de imágenes borradas (``0`` si no había ninguna).

        Raises
        ------
        ScopeError
            Si los filtros son inválidos.
        """
        return 0


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
        start_ms: int | None = None,
        end_ms: int | None = None,
        transcript: list[dict] | None = None,
    ) -> str:
        """Persiste una página y devuelve su page_id (str).

        Parameters
        ----------
        raw_file_id:
            ID del :class:`~storage.models.RawFileRecord` al que pertenece
            esta página.
        filename:
            Nombre del fichero fuente con su extensión (p.ej. ``"3.2.P.1.pdf"``).
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
        start_ms, end_ms:
            Rango ``[start_ms, end_ms)`` de la página en el medio original
            (tramos de audio y vídeo), o ``None``. El conversor solo los pasa
            cuando no son ``None``: un repositorio propio con la firma
            anterior a 0.5.0 sigue funcionando para todo lo que no sea audio
            o vídeo, pero para ingerir audio y vídeo debe aceptarlos.
        transcript:
            Frases transcritas de la página con sus tiempos absolutos
            (audio y vídeo; decisión D7), como lista de dicts
            ``{start_ms, end_ms, text}``, o ``None``. El conversor solo la
            pasa cuando no es ``None``: para ingerir audio y vídeo, el
            repositorio debe aceptar ``start_ms``, ``end_ms`` y
            ``transcript``.

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
