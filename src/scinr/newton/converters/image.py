"""converters/image.py — image files (photos, scans, screenshots) to text.

The whole image becomes one page: the ``vision_llm`` transcribes its visible
text and describes it (``standalone`` mode). File metadata (size, capture
time, device and, only with ``strip_location=False``, GPS) is read with
Pillow and reaches the ``:Document`` node as ``file_*`` properties.

Requirements are checked at conversion time: without a ``vision_llm`` the
conversion fails with a :class:`ConversionError` that says how to configure
it, before the file is read. HEIC/HEIF need ``pillow-heif``
(``pip install "scinr[media]"``).

Memory: one image (the file size) plus its downscaled JPEG copy while it is
described.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import logging
from pathlib import Path

from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
)
from scinr.newton.exceptions import ConfigurationError
from scinr.newton.models.file_metadata import FileMetadata

logger = logging.getLogger(__name__)

__all__ = ["ImageConverter", "read_image_metadata"]

_HEIF_EXTENSIONS = frozenset({"heic", "heif"})

# EXIF tags
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_DATETIME = 0x0132
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825
_TAG_DATETIME_ORIGINAL = 0x9003
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4


class ImageConverter(BaseConverter):
    """Convert an image file into a one-page document with the ``vision_llm``."""

    supported_extensions: frozenset[str] = frozenset(
        {"jpg", "jpeg", "png", "webp", "tif", "tiff", "heic", "heif"}
    )
    is_async: bool = True

    async def convert(self, source: Path) -> IntermediateDocument:
        """Describe *source* and return it as a one-page document.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist.
        ConversionError
            No ``vision_llm`` configured, HEIC without ``pillow-heif``, an
            unreadable image or a failed model call (an image has nothing
            partial to keep, so ``media_error_strategy`` does not apply).
        """
        from scinr.newton.config import get_config, get_vision_llm
        from scinr.newton.converters.vision import describe_image

        try:
            get_vision_llm()
        except ConfigurationError as exc:
            raise ConversionError(
                f"Cannot convert image {source.name}: no vision_llm is configured.\n{exc}"
            ) from exc
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")
        if source.suffix.lower().lstrip(".") in _HEIF_EXTENSIONS and not _heif_available():
            raise ConversionError(
                f"Cannot convert {source.name}: HEIC/HEIF images need pillow-heif. "
                'Install it with: pip install "scinr[media]"'
            )

        data = await asyncio.to_thread(source.read_bytes)
        metadata = await asyncio.to_thread(
            read_image_metadata, data, strip_location=get_config().strip_location
        )
        try:
            markdown = await describe_image(data, mode="standalone")
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Failed to describe image {source.name}: {exc}") from exc
        finally:
            del data
        logger.info("Image %s converted (%d characters).", source.name, len(markdown))
        return IntermediateDocument(
            pages=[IntermediatePage(index=0, markdown=markdown)],
            file_metadata=metadata,
        )


def _heif_available() -> bool:
    return importlib.util.find_spec("pillow_heif") is not None


def _exif_datetime(value: object) -> str | None:
    """``"2024:05:01 10:30:00"`` → ``"2024-05-01T10:30:00"`` (no time zone)."""
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("\x00")
    date, _, time = text.partition(" ")
    parts = date.split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    iso = "-".join(parts)
    return f"{iso}T{time}" if time else iso


def _text(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, str):
        return None
    value = value.strip().rstrip("\x00").strip()
    return value or None


def _degrees(dms: object, ref: object) -> float | None:
    """EXIF ``(deg, min, sec)`` rationals and ``N/S/E/W`` → signed decimal degrees."""
    try:
        deg, minutes, seconds = (float(x) for x in dms)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    value = deg + minutes / 60 + seconds / 3600
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", errors="ignore")
    if isinstance(ref, str) and ref.strip().upper() in ("S", "W"):
        value = -value
    return round(value, 7)


def read_image_metadata(data: bytes, *, strip_location: bool = True) -> FileMetadata:
    """Read size, capture time, device and (optionally) GPS from an encoded image.

    The GPS IFD is not even read when *strip_location* is ``True``. Size is
    reported after EXIF orientation (a portrait photo is taller than wide).
    """
    from PIL import Image

    from scinr.newton.converters.vision import _register_heif

    _register_heif()
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            exif = image.getexif()
    except Exception as exc:
        raise ConversionError(f"Cannot read image: {exc}") from exc

    orientation = exif.get(0x0112)
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    exif_ifd = exif.get_ifd(_IFD_EXIF)
    fields: dict = {
        "width": width,
        "height": height,
        "captured_at": _exif_datetime(exif_ifd.get(_TAG_DATETIME_ORIGINAL))
        or _exif_datetime(exif.get(_TAG_DATETIME)),
        "device_make": _text(exif.get(_TAG_MAKE)),
        "device_model": _text(exif.get(_TAG_MODEL)),
    }
    if not strip_location:
        gps = exif.get_ifd(_IFD_GPS)
        if gps:
            fields["gps_latitude"] = _degrees(gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF))
            fields["gps_longitude"] = _degrees(gps.get(_GPS_LON), gps.get(_GPS_LON_REF))
    return FileMetadata(**fields)
