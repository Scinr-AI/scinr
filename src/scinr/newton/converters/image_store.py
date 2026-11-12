"""Image storage context of Stage 0 (WP4 of
``plans/stored-images-implementation-plan.md``).

The described embedded images (and, from WP5, the described video frames)
are stored in the document base, linked to their page, **inside the same
call that describes them** — so only what got a description is ever written
(decision D1) and no ``raw_file_id`` / tenancy has to be threaded through
every converter signature (external converters included): the context
travels in a :class:`~contextvars.ContextVar`, which reaches both the tasks
of an :func:`asyncio.gather` and the worker threads of
:func:`asyncio.to_thread` (both copy the current context).

- :func:`image_sink` installs the context for one document conversion and
  returns its :class:`ImageSink`, or ``None`` — no context installed — when
  there is nothing to store to: an empty ``raw_file_id`` (storage
  ``none``), both switches (``store_described_images`` /
  ``store_video_frames``) off, or a backend whose
  :attr:`~storage.base.RawFileRepository.stores_images` is ``False`` (the
  last case warns once per process, D4).
- :func:`store_embedded` stores the original bytes of an image that just
  got a non-empty description and returns its ``image_id`` (``None`` when
  there is no sink, or the ``embedded`` switch is off).
- :func:`store_frame` is the same for a video frame, streamed from disk
  through :meth:`~storage.base.RawFileRepository.store_image_file`.
- :func:`discard_stored` deletes what was stored of the ``raw_file_id``
  when the conversion or the page saving fails afterwards (D5: a described
  image is never kept without its document).

Memory: :func:`store_embedded` reads the original bytes and uploads them
**inside the same lane as the decoding** (``vision.image_bytes_lane()``,
``image_decode_concurrency`` per event loop), so the number of originals in
memory at a time has the same cap as the images being decoded, even with
``llm_concurrency`` description calls in flight.

Errors: any storage failure is wrapped in
:class:`~scinr.newton.exceptions.ImageStorageError`, which the enrichment
always re-raises after its ``gather`` (D5), whatever the
``media_error_strategy``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from scinr.newton.config import get_config
from scinr.newton.converters import vision
from scinr.newton.converters.base import PageImage
from scinr.newton.exceptions import ConfigurationError, ImageStorageError, StorageError
from scinr.newton.storage.base import RawFileRepository

logger = logging.getLogger(__name__)

__all__ = ["ImageSink", "image_sink", "store_embedded", "store_frame", "discard_stored"]


@dataclass
class ImageSink:
    """Storage context of one document conversion (see :func:`image_sink`).

    Attributes
    ----------
    repo:
        Repository the images are stored to (its ``stores_images`` is
        ``True``).
    raw_file_id:
        The document's ``raw_file_id`` the stored images are linked to.
    filename, folder_path, tenant_id, created_by_user_id, job_id:
        Stamped on every stored image, exactly as on the raw file.
    embedded:
        ``store_described_images`` switch: embedded images are stored.
    frames:
        ``store_video_frames`` switch: described video frames are stored.
    stored:
        How many images have been stored so far in this context. Read by
        the caller to decide whether a failure needs :func:`discard_stored`.
    """

    repo: RawFileRepository
    raw_file_id: str
    filename: str
    folder_path: str | None
    tenant_id: str | None
    created_by_user_id: str | None
    job_id: str | None
    embedded: bool  # cfg.store_described_images
    frames: bool  # cfg.store_video_frames
    stored: int = 0


_current: ContextVar[ImageSink | None] = ContextVar("scinr_image_sink", default=None)

#: Per-process flag of the "backend does not store images" warning (D4).
_warned_no_image_storage = False


@contextmanager
def image_sink(
    repo: RawFileRepository,
    raw_file_id: str,
    *,
    filename: str,
    folder_path: str | None = None,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> Iterator[ImageSink | None]:
    """Install the image-storage context of one document conversion.

    Returns the installed :class:`ImageSink`, or ``None`` — installing no
    context at all — when there is nothing to store to: an empty
    *raw_file_id* (storage ``none``), both switches
    (``store_described_images`` / ``store_video_frames``) off, or a *repo*
    whose ``stores_images`` is ``False``. The last case warns once per
    process (D4); a custom backend written before 0.5.0 keeps working,
    simply without keeping the described images.

    The context is a :class:`~contextvars.ContextVar`, so it reaches the
    tasks of an :func:`asyncio.gather` and the worker threads of
    :func:`asyncio.to_thread` (both copy the current context): converters
    do not change signature.
    """
    global _warned_no_image_storage
    try:
        cfg = get_config()
    except ConfigurationError:
        yield None  # unconfigured: no switches to read, nothing to store
        return
    if not raw_file_id:
        yield None  # storage ``none``: no raw_file_id to link the images to
        return
    if not (cfg.store_described_images or cfg.store_video_frames):
        yield None
        return
    # ``getattr``: a duck-typed pre-0.5.0 repo may not even declare the
    # property; a RawFileRepository defaults it to False.
    if not getattr(repo, "stores_images", False):
        if not _warned_no_image_storage:
            _warned_no_image_storage = True
            logger.warning(
                "The configured storage backend does not store images "
                "(RawFileRepository.stores_images is False); described images and "
                "frames are not kept."
            )
        yield None
        return
    sink = ImageSink(
        repo=repo,
        raw_file_id=raw_file_id,
        filename=filename,
        folder_path=folder_path,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
        embedded=cfg.store_described_images,
        frames=cfg.store_video_frames,
    )
    token = _current.set(sink)
    try:
        yield sink
    finally:
        _current.reset(token)


async def store_embedded(page_index: int, image: PageImage, description: str) -> str | None:
    """Store the original bytes of *image*, just described with *description*.

    ``None`` (nothing stored) when there is no :func:`image_sink` context,
    or its ``embedded`` switch (``store_described_images``) is off;
    otherwise the ``image_id`` assigned by the backend.

    The original bytes are re-read (``image.load_bytes()``) and uploaded
    inside the same lane as the decoding
    (:func:`~scinr.newton.converters.vision.image_bytes_lane`,
    ``image_decode_concurrency`` per event loop): the number of originals
    in memory at a time has the same cap as the images being decoded, even
    with ``llm_concurrency`` description calls in flight.

    Raises
    ------
    ImageStorageError
        The image could not be stored. Always fails the document (D5),
        whatever the ``media_error_strategy``.
    """
    sink = _current.get()
    if sink is None or not sink.embedded:
        return None
    try:
        async with vision.image_bytes_lane():
            content = await asyncio.to_thread(image.load_bytes)
            image_id = await sink.repo.store_image(
                sink.raw_file_id,
                page_index,
                image.index,
                content,
                kind="embedded",
                content_type=image.media_type,
                description=description,
                filename=sink.filename,
                folder_path=sink.folder_path,
                width=image.width,
                height=image.height,
                tenant_id=sink.tenant_id,
                created_by_user_id=sink.created_by_user_id,
                job_id=sink.job_id,
            )
    except StorageError:
        raise
    except Exception as exc:
        raise ImageStorageError(
            f"Image {image.index} of page {page_index} of {sink.filename} "
            f"could not be stored: {exc}"
        ) from exc
    sink.stored += 1
    return image_id


async def store_frame(
    page_index: int,
    frame_index: int,
    path: Path,
    *,
    time_ms: int,
    description: str,
) -> str | None:
    """Store the described video frame at *path* (streamed from disk).

    ``None`` (nothing stored) when there is no :func:`image_sink` context,
    or its ``frames`` switch (``store_video_frames``) is off; otherwise the
    ``image_id`` assigned by the backend. The upload goes through
    :meth:`~storage.base.RawFileRepository.store_image_file`, so a backend
    with streaming never materialises the frame in memory; the frame's
    ``width`` / ``height`` are read from the JPEG header
    (:func:`~scinr.newton.converters.vision.image_size`), without decoding.

    Raises
    ------
    ImageStorageError
        The frame could not be stored. Always fails the document (D5),
        whatever the ``media_error_strategy``.
    """
    sink = _current.get()
    if sink is None or not sink.frames:
        return None
    try:
        width, height = await asyncio.to_thread(vision.image_size, path)
        image_id = await sink.repo.store_image_file(
            path,
            sink.raw_file_id,
            page_index,
            frame_index,
            kind="frame",
            content_type="image/jpeg",
            description=description,
            filename=sink.filename,
            folder_path=sink.folder_path,
            width=width,
            height=height,
            time_ms=time_ms,
            tenant_id=sink.tenant_id,
            created_by_user_id=sink.created_by_user_id,
            job_id=sink.job_id,
        )
    except StorageError:
        raise
    except Exception as exc:
        raise ImageStorageError(
            f"Frame {frame_index} of segment {page_index} of {sink.filename} "
            f"could not be stored: {exc}"
        ) from exc
    sink.stored += 1
    return image_id


async def discard_stored(sink: ImageSink) -> None:
    """Delete the images stored so far of *sink*'s ``raw_file_id``.

    Called when the conversion or the page saving fails after some image
    was already stored (D5: a described image is never left without its
    document). A cleanup failure does not mask the original error: it is
    logged with the ``raw_file_id`` and swallowed.
    """
    try:
        deleted = await sink.repo.delete_images(sink.raw_file_id)
    except Exception as exc:
        logger.warning(
            "Images of raw_file_id %s could not be cleaned up after the failure: %s",
            sink.raw_file_id,
            exc,
        )
        return
    if deleted:
        logger.info(
            "Cleaned up %d stored image(s) of raw_file_id %s after the failure.",
            deleted,
            sink.raw_file_id,
        )
