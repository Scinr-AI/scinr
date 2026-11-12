"""File-level metadata of a source file (images, audio, video).

Closed model on purpose: the ingestion writes every key of
:data:`FILE_METADATA_KEYS` on the ``:Document`` node (``file_<key>``) on every
ingestion, ``null`` included, so ``update_mode`` never leaves stale
properties behind, and no property name ever comes from data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["FileMetadata", "FILE_METADATA_KEYS"]


class FileMetadata(BaseModel):
    """Metadata read from a source file. Every field is optional.

    Attributes
    ----------
    captured_at:
        Capture time in ISO 8601 (EXIF ``DateTimeOriginal``), as written by the
        device (no time zone conversion).
    device_make, device_model:
        Camera or device make and model (EXIF).
    gps_latitude, gps_longitude:
        Location in decimal degrees (EXIF GPS). Never set when the
        ``strip_location`` option is ``True`` (the default).
    width, height:
        Size in pixels (images and video).
    duration_ms:
        Duration in milliseconds (audio and video).
    container, audio_codec, video_codec:
        Container format and codecs, as reported by ``ffprobe``.
    frame_rate:
        Frames per second (video).
    """

    model_config = ConfigDict(extra="forbid")

    captured_at: str | None = None
    device_make: str | None = None
    device_model: str | None = None
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    container: str | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    frame_rate: float | None = None


#: The field names of :class:`FileMetadata`: the only list the ingestion uses
#: to build the ``file_*`` properties of the ``:Document`` node.
FILE_METADATA_KEYS: tuple[str, ...] = tuple(FileMetadata.model_fields)
