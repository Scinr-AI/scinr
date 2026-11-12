"""Image preparation and description with the configured ``vision_llm``.

Memory: decoding is the expensive step (a 12 MP photo is ~36 MB of pixels,
whatever its file size), so :func:`prepare_for_vision` reads, decodes and
downscales at most ``image_decode_concurrency`` images at a time per event
loop (default ``1``), whatever the number of pages, documents
(``parallel_docs``) or LLM calls in flight. What waits for
the model is only the downscaled JPEG (``image_max_side_px``, a few hundred
KB) and its base64; the number of calls in flight is bounded by the global
LLM semaphore (``llm_concurrency``).

The largest single decode is bounded by Pillow's ``Image.MAX_IMAGE_PIXELS``
(about 89 MP): a larger image fails as a decompression bomb. JPEGs are
decoded already reduced (``draft``), so only lossless formats (PNG, TIFF,
WebP lossless) pay their full size.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import warnings
from collections.abc import Callable

from scinr.newton.config import get_config, get_llm_semaphore, get_vision_llm
from scinr.newton.converters.base import ConversionError
from scinr.newton.converters.vision_prompts import VisionMode, system_prompt, user_prompt
from scinr.newton.utils.llm_retry import with_llm_retry
from scinr.newton.utils.loop_semaphore import LoopSemaphore

logger = logging.getLogger(__name__)

__all__ = [
    "image_size",
    "prepare_image",
    "prepare_for_vision",
    "describe_image",
    "CONTEXT_MAX_CHARS",
]

#: Characters of the page markdown sent as context in ``embedded`` mode.
CONTEXT_MAX_CHARS = 1500

_JPEG_QUALITY = 85

#: Images read and decoded at once per event loop: ``image_decode_concurrency``.
_decode_semaphores = LoopSemaphore()

ImageSource = bytes | Callable[[], bytes]


def _open(data: bytes):
    """Open *data* with Pillow, turning a decompression bomb into ConversionError."""
    from PIL import Image

    _register_heif()
    try:
        with warnings.catch_warnings():
            # Pillow warns (and later raises) on very large images: treat both as an error.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return Image.open(io.BytesIO(data))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ConversionError(f"Image too large to process safely: {exc}") from exc
    except Exception as exc:
        raise ConversionError(f"Cannot read image: {exc}") from exc


_heif_registered = False


def _register_heif() -> None:
    """Register the HEIC/HEIF opener when ``pillow-heif`` is installed (``scinr[media]``)."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif  # type: ignore[import-not-found]
    except ImportError:
        return
    pillow_heif.register_heif_opener()
    _heif_registered = True


def image_size(data: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` of the encoded image *data*, reading only its header."""
    with _open(data) as image:
        return image.size


def prepare_image(data: bytes, max_side: int) -> bytes:
    """Return *data* as a JPEG whose longer side is at most *max_side* pixels.

    Applies the EXIF orientation, converts to RGB (flattening transparency on
    white) and downscales with :meth:`PIL.Image.Image.thumbnail`. Synchronous
    and CPU-bound: call it through :func:`asyncio.to_thread`.
    """
    from PIL import Image, ImageOps

    try:
        with _open(data) as image:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image.draft("RGB", (max_side, max_side))  # cheap JPEG downscale on decode
                oriented = ImageOps.exif_transpose(image)
            if oriented.mode in ("RGBA", "LA", "P"):
                rgba = oriented.convert("RGBA")
                flat = Image.new("RGB", rgba.size, "white")
                flat.paste(rgba, mask=rgba.split()[-1])
                rgb = flat
            else:
                rgb = oriented.convert("RGB")
            rgb.thumbnail((max_side, max_side))
            out = io.BytesIO()
            rgb.save(out, format="JPEG", quality=_JPEG_QUALITY)
            return out.getvalue()
    except ConversionError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ConversionError(f"Image too large to process safely: {exc}") from exc
    except Exception as exc:
        raise ConversionError(f"Cannot prepare image: {exc}") from exc


async def prepare_for_vision(image: ImageSource, max_side: int) -> bytes:
    """Read (when *image* is a loader), decode and downscale one image.

    Runs under a per-loop semaphore of size ``image_decode_concurrency``
    (default ``1``), so the encoded bytes and decoded pixels of at most that
    many images are in memory at a time; the bytes are dropped as soon as the
    downscaled JPEG exists.
    """
    async with _decode_semaphores.get(get_config().image_decode_concurrency):
        data = await asyncio.to_thread(image) if callable(image) else image
        try:
            return await asyncio.to_thread(prepare_image, data, max_side)
        finally:
            del data


def _response_text(response) -> str:
    """Text of a chat model response; ``content`` may be a string or a list of blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


async def describe_image(
    image: ImageSource,
    *,
    mode: VisionMode,
    context: str | None = None,
) -> str:
    """Describe the image *data* with ``vision_llm`` and return Markdown text.

    The image is downscaled to ``image_max_side_px`` and sent as a JPEG in a
    standard LangChain image block (translated by each provider integration:
    OpenAI-compatible, Bedrock Converse, Ollama…). The call runs under the
    global LLM semaphore with the usual retries on throttling.

    Parameters
    ----------
    image:
        Encoded image bytes (any format Pillow can read), or a callable
        that returns them: a loader is only called once it is this image's
        turn to be decoded, so the bytes of images still waiting are not
        read.
    mode:
        ``"embedded"``, ``"standalone"`` or ``"frame"`` (see :mod:`vision_prompts`).
    context:
        ``embedded`` mode: the page markdown (truncated to
        :data:`CONTEXT_MAX_CHARS`).

    Returns
    -------
    str
        The description; an empty string means the image carries no information.

    Raises
    ------
    ConfigurationError
        No ``vision_llm`` configured.
    ConversionError
        The image cannot be read.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    vision_llm = get_vision_llm()
    max_side = get_config().image_max_side_px
    prepared = await prepare_for_vision(image, max_side)
    del image  # the caller may hold it, but this frame no longer does
    encoded = base64.b64encode(prepared).decode("ascii")
    del prepared
    if context and len(context) > CONTEXT_MAX_CHARS:
        context = context[:CONTEXT_MAX_CHARS]
    messages = [
        SystemMessage(content=system_prompt(mode)),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt(mode, context)},
                {"type": "image", "base64": encoded, "mime_type": "image/jpeg"},
            ]
        ),
    ]
    del encoded
    try:
        async with get_llm_semaphore():
            response = await with_llm_retry(lambda: vision_llm.ainvoke(messages))
    finally:
        messages.clear()
    return _response_text(response)
