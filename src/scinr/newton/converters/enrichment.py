"""Embedded-image enrichment of Stage 0.

Describes the images of each :class:`~converters.base.IntermediatePage` with
the ``vision_llm`` and writes the description into the page markdown as::

    <image index="2">
    …description…
    </image>

The block replaces the image reference ``![…](ref)`` when the page has one,
or goes at the end of the page otherwise. Only images that still hold bytes
and have no description are treated; every treated image is released
(``base64`` and loader dropped), whatever the outcome.

Called only when :func:`~scinr.newton.config.embedded_images_enabled` is
``True`` (by ``converters/main.py`` after conversion, and by the PDF converter
chunk by chunk).

Memory: one page at a time. Inside a page the filter reads each image's
bytes one at a time and drops them before the next; the images to describe
are then read and decoded **one at a time** (per event loop, see
:func:`~converters.vision.prepare_for_vision`), and only their downscaled
JPEGs (``image_max_side_px``) wait for the model, at most
``llm_concurrency`` calls in flight.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
from dataclasses import dataclass, field

from scinr.newton.config import get_config
from scinr.newton.converters.base import ConversionError, IntermediatePage, PageImage
from scinr.newton.converters.image_store import store_embedded
from scinr.newton.converters.vision import describe_image, image_size
from scinr.newton.exceptions import ImageStorageError

logger = logging.getLogger(__name__)

__all__ = ["EnrichmentState", "enrich_pages", "render_image_block"]


@dataclass
class EnrichmentState:
    """Per-document state shared by every page (and every PDF chunk).

    Attributes
    ----------
    seen:
        SHA-1 of the images already seen in the document: a repeated image
        (a logo on every page) is described only the first time.
    """

    seen: set[str] = field(default_factory=set)


def render_image_block(index: int, description: str, **attrs: str) -> str:
    """Render the description block of image *index* (extra attributes, e.g. ``time``)."""
    attributes = "".join(
        f' {name}="{html.escape(str(value), quote=True)}"' for name, value in attrs.items()
    )
    return f'<image index="{index}"{attributes}>\n{description.strip()}\n</image>'


def _ref_pattern(ref: str) -> re.Pattern[str]:
    return re.compile(r"!\[[^\]]*\]\(" + re.escape(ref) + r"\)")


def _remove_ref(markdown: str, ref: str | None) -> str:
    if not ref:
        return markdown
    return _ref_pattern(ref).sub("", markdown, count=1)


@dataclass
class _Candidate:
    image: PageImage
    area: int


def _select(
    page: IntermediatePage, state: EnrichmentState, *, min_side: int, max_images: int
) -> tuple[list[PageImage], list[PageImage], list[PageImage]]:
    """Split the page's pending images into ``(to_describe, discarded, unreadable)``.

    Reads each image's bytes once, one image at a time, for the size check
    (only when the converter did not give ``width`` / ``height``) and the
    repetition check.
    """
    candidates: list[_Candidate] = []
    discarded: list[PageImage] = []
    unreadable: list[PageImage] = []
    for image in page.images:
        if image.description or not image.has_content:
            continue
        try:
            data = image.load_bytes()
            if image.width and image.height:
                width, height = image.width, image.height
            else:
                width, height = image_size(data)
            digest = hashlib.sha1(data, usedforsecurity=False).hexdigest()
        except Exception as exc:
            logger.warning(
                "Embedded image %d of page %d cannot be read: %s", image.index, page.index, exc
            )
            unreadable.append(image)
            continue
        finally:
            data = None  # noqa: F841 - drop the bytes before the next image
        if min(width, height) < min_side:
            discarded.append(image)
            continue
        if digest in state.seen:
            discarded.append(image)
            continue
        state.seen.add(digest)
        candidates.append(_Candidate(image, width * height))
    if len(candidates) > max_images:
        candidates.sort(key=lambda c: c.area, reverse=True)
        discarded.extend(c.image for c in candidates[max_images:])
        candidates = candidates[:max_images]
    to_describe = sorted((c.image for c in candidates), key=lambda i: i.index)
    return to_describe, discarded, unreadable


async def _describe(image: PageImage, context: str, page_index: int) -> str:
    # The loader is called only when it is this image's turn to be decoded
    # (one at a time, see vision.prepare_for_vision): the images of a page
    # waiting for their turn are not read in the meantime. Storing the
    # described original re-reads them inside that same lane (see
    # image_store.store_embedded).
    try:
        description = await describe_image(image.load_bytes, mode="embedded", context=context)
        if description:
            image.image_id = await store_embedded(page_index, image, description)
        return description
    finally:
        image.release()


async def enrich_pages(
    pages: list[IntermediatePage], state: EnrichmentState
) -> list[tuple[int, int]]:
    """Describe the embedded images of *pages*, in place.

    Returns the ``(page_index, image_index)`` pairs that failed and were
    skipped (``media_error_strategy="best_effort"``).

    Raises
    ------
    ConversionError
        An image failed and ``media_error_strategy`` is ``"fail_fast"``.
    ImageStorageError
        A described image could not be stored. Always re-raised, whatever
        the ``media_error_strategy`` (D5: a described image is never kept
        without its original).
    """
    cfg = get_config()
    fail_fast = cfg.media_error_strategy == "fail_fast"
    missing: list[tuple[int, int]] = []
    for page in pages:
        to_describe, discarded, unreadable = _select(
            page, state, min_side=cfg.image_min_side_px, max_images=cfg.max_images_per_page
        )
        if not (to_describe or discarded or unreadable):
            continue
        markdown = page.markdown
        for image in discarded:
            image.release()
            markdown = _remove_ref(markdown, image.ref)
        for image in unreadable:
            image.release()
            markdown = _remove_ref(markdown, image.ref)
            missing.append((page.index, image.index))

        context = page.markdown
        results = await asyncio.gather(
            *(_describe(image, context, page.index) for image in to_describe),
            return_exceptions=True,
        )
        # A storage failure of a described image is never best-effort (D5).
        for result in results:
            if isinstance(result, ImageStorageError):
                raise result
        trailing: list[str] = []
        first_error: BaseException | None = None
        for image, result in zip(to_describe, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result  # CancelledError, KeyboardInterrupt…
                logger.warning(
                    "Embedded image %d of page %d could not be described: %s",
                    image.index, page.index, result,
                )
                first_error = first_error or result
                markdown = _remove_ref(markdown, image.ref)
                missing.append((page.index, image.index))
                continue
            image.description = result
            if not result:
                markdown = _remove_ref(markdown, image.ref)
                continue
            block = render_image_block(image.index, result)
            if image.ref and _ref_pattern(image.ref).search(markdown):
                markdown = _ref_pattern(image.ref).sub(
                    lambda _m, b=block: b, markdown, count=1
                )
            else:
                trailing.append(block)
        if trailing:
            markdown = markdown.rstrip("\n") + "\n\n" + "\n\n".join(trailing) + "\n"
        page.markdown = markdown
        if fail_fast and (first_error is not None or unreadable):
            reason = first_error or "unreadable image"
            raise ConversionError(
                f"Embedded image of page {page.index} could not be described: {reason}"
            )
    return missing
