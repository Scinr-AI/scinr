"""Peak LIVE memory (tracemalloc; allocator independent) of the multimodal ingestion paths.

Usage:
    python scripts/memory_bench/bench_media.py make          # synthetic inputs (needs ffmpeg)
    python scripts/memory_bench/bench_media.py run <case>    # one case, prints its peak
    python scripts/memory_bench/bench_media.py all           # every case, each in its own process

Only the library is measured: the vision model, the transcriber and the Mistral OCR server are
simulated in-process (the OCR server is an httpx.MockTransport that answers each chunk with one
unique ~150 KB JPEG per page when images are requested). ffmpeg runs as a subprocess and is not
counted. Conversions go through converters.main.convert_one (so the enrichment step runs), with
no storage and no JSON written.

Cases and criteria (plan WP13)
------------------------------
pdf_text / pdf_images : 300-page PDF without / with image description (chunks of 50 pages).
                        Criterion: pdf_images <= pdf_text + one 50-page chunk response.
docx_20 / docx_100    : DOCX with 20 / 100 images, description on; *_off = description off.
pptx_20 / pptx_100      Criterion: (on - off) does not grow with the number of images.
audio_30 / audio_120  : 30 min / 2 h audio. Criterion: 2 h not appreciably above 30 min.
video_small / video_big: 1 min / ~10 min high-bitrate video (hundreds of MB).
                        Criterion: does not depend on the file size.
batch8                : 8 audios with parallel_docs=4. Criterion: at most media_concurrency (1)
                        media conversions in flight.
*_stored              : docx_100 / pdf_images / video_small with a raw-file repository that
                        stores every described image / frame (it consumes and drops the bytes).
                        Criterion (stored-images plan WP9): *_stored - * <= the largest stored
                        original + 2 MiB.

Measured on 2026-11-12 (stored images):
    docx_100 34.4 | docx_100_stored 34.6     (+0.2; 100 stored, largest original 261 KiB)
    pdf_images 23.0 | pdf_images_stored 23.2 (+0.2; 300 stored, largest crop 261 KiB)
    video_small 1.7 | video_small_stored 2.2 (+0.5; 2 frames stored, streamed from disk)
The original of an embedded image is re-read inside the decode lane
(image_decode_concurrency), so stored originals in memory never exceed that count.

Measured on 2026-11-11 (macOS arm64, Python 3.14, tracemalloc peak in MiB):
    pdf_text 1.9 | pdf_images 23.9   one 50-page chunk response with images = 17.0 MiB of JSON;
        the excess (22.0) is that chunk's mapped base64 plus the images being described
        (llm_concurrency=4). The response is parsed page by page with ijson from the raw body
        (before this, response.json() held body text + parsed tree: 39.4). The simulated
        server's body is built before measuring; with a real server httpx also holds the raw
        body (+17 MiB) while that chunk is parsed, released before the images are described.
    docx_20_off 11.6 | docx_20 14.6 | docx_100_off 32.1 | docx_100 36.2   (on - off: 3.0 / 4.1)
    pptx_20_off 13.4 | pptx_20 17.1 | pptx_100_off 34.0 | pptx_100 37.7   (on - off: 3.7 / 3.7)
    audio_30 1.5 | audio_120 1.5     (10.7 MB / 42.8 MB files; 6 / 24 segments)
    video_small 5.2 | video_big 5.7  (7.8 MB / 363 MB files)
    batch8 1.4, max media conversions in flight: 1 (media_concurrency=1, parallel_docs=4)
The python-docx / python-pptx "off" peaks are the libraries holding the opened file (they grow
with the file, images included, with or without scinr); the cost scinr adds on top stays flat.
Do NOT use RSS (ps / ru_maxrss) for this: on macOS freed large blocks stay in RSS.

Blind spot: tracemalloc does not see Pillow's pixel buffers (allocated in C), so a decoded
image costs nothing here, and the synthetic images are small. Decoding is bounded by design
(one image at a time per event loop, converters/vision.py::prepare_for_vision), pinned by
tests/unit/test_best_effort_outcome.py. Measured apart with ru_maxrss on 2026-11-11: 8 PNG of
3000x2000 on one page peak at +65 MiB RSS, against +632 MiB when they were decoded at once.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

DIR = Path(tempfile.gettempdir()) / "scinr_bench_media"
MIB = 1024 * 1024


def _noise_jpeg(seed: int, size=(800, 600)) -> bytes:
    from PIL import Image

    rnd = os.urandom(size[0] * size[1] * 3) if seed < 0 else _seeded(seed, size[0] * size[1] * 3)
    image = Image.frombytes("RGB", size, rnd)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=70)
    return out.getvalue()


def _seeded(seed: int, n: int) -> bytes:
    import random

    return random.Random(seed).randbytes(n)


def _ffmpeg(*args: str) -> None:
    subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", *args, "-y"],
                   check=True)


def make() -> None:
    from pypdf import PdfWriter

    DIR.mkdir(exist_ok=True)
    writer = PdfWriter()
    for _ in range(300):
        writer.add_blank_page(612, 792)
    with (DIR / "doc300.pdf").open("wb") as fh:
        writer.write(fh)

    import docx
    import pptx
    from pptx.util import Inches

    for n in (20, 100):
        document = docx.Document()
        deck = pptx.Presentation()
        for i in range(n):
            img = DIR / f"img_{i}.jpg"
            img.write_bytes(_noise_jpeg(i))
            document.add_paragraph(f"Figura {i}")
            document.add_picture(str(img))
            slide = deck.slides.add_slide(deck.slide_layouts[6])
            slide.shapes.add_picture(str(img), Inches(1), Inches(1))
        document.save(DIR / f"doc_{n}.docx")
        deck.save(DIR / f"deck_{n}.pptx")
    for img in DIR.glob("img_*.jpg"):
        img.unlink()

    _ffmpeg("-f", "lavfi", "-i", "anoisesrc=d=1800:c=pink:a=0.1", "-ac", "1", "-b:a", "48k",
            str(DIR / "audio_30.m4a"))
    _ffmpeg("-f", "lavfi", "-i", "anoisesrc=d=7200:c=pink:a=0.1", "-ac", "1", "-b:a", "48k",
            str(DIR / "audio_120.m4a"))
    for name, seconds, rate in (("video_small", 60, "1M"), ("video_big", 600, "5M")):
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=25:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                "-shortest", "-c:v", "mpeg4", "-b:v", rate, "-c:a", "aac", str(DIR / f"{name}.mp4"))
    for i in range(8):
        shutil.copy(DIR / "audio_30.m4a", DIR / f"batch_{i}.m4a")
    print("written to", DIR)


# ---------------------------------------------------------------------------
# Simulated providers
# ---------------------------------------------------------------------------


def _fake_vision():
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class FakeVision(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "bench-vision"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage("A described figure."))])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            await asyncio.sleep(0.005)
            return self._generate(messages)

    return FakeVision()


def _fake_transcriber(in_flight: dict):
    from scinr.newton.media.transcriber import Transcriber, TranscriptSegment

    class FakeTranscriber(Transcriber):
        async def transcribe(self, path, language=None):
            await asyncio.sleep(0)
            return [TranscriptSegment(start_ms=0, end_ms=1000, text="palabras " * 300)]

    return FakeTranscriber()


def _fake_ocr(prebuild_pages: int = 0, chunk_pages: int = 50) -> int:
    """Install the simulated OCR server; return the size of one chunk response with images.

    Responses with images are built **before** tracemalloc starts (they live on Mistral's
    side in production), so only what the library allocates is measured: the response bytes
    httpx reads are still counted through ``response.json()``'s parse.
    """
    import httpx
    from pypdf import PdfReader

    prebuilt: list[bytes] = []
    for start in range(0, prebuild_pages, chunk_pages):
        out = []
        for i in range(min(chunk_pages, prebuild_pages - start)):
            images = [{
                "id": f"img-{i}.jpeg", "top_left_x": 0, "top_left_y": 0,
                "bottom_right_x": 800, "bottom_right_y": 600,
                "image_base64": "data:image/jpeg;base64,"
                + base64.b64encode(_noise_jpeg(start + i)).decode(),
            }]
            out.append({"index": i, "markdown": f"Page {i}\n\n![img-{i}.jpeg](img-{i}.jpeg)",
                        "images": images, "dimensions": {}})
        prebuilt.append(json.dumps({"pages": out}).encode())
    served = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        raw = await request.aread()
        payload = json.loads(raw)
        with_images = payload.get("include_image_base64", False)
        if with_images:
            body = prebuilt[served["n"]]
            served["n"] += 1
            return httpx.Response(
                200, content=body, headers={"content-type": "application/json"}
            )
        url = payload["document"]["document_url"]
        pages = len(PdfReader(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).pages)
        out = [{"index": i, "markdown": f"Page {i}\n\n![img-{i}.jpeg](img-{i}.jpeg)",
                "images": [], "dimensions": {}} for i in range(pages)]
        return httpx.Response(200, json={"pages": out})

    real_client = httpx.AsyncClient

    class _Client(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    httpx.AsyncClient = _Client
    return len(prebuilt[0]) if prebuilt else 0


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = {
    "pdf_text": ("doc300.pdf", False),
    "pdf_images": ("doc300.pdf", True),
    "docx_20_off": ("doc_20.docx", False),
    "docx_20": ("doc_20.docx", True),
    "docx_100_off": ("doc_100.docx", False),
    "docx_100": ("doc_100.docx", True),
    "pptx_20_off": ("deck_20.pptx", False),
    "pptx_20": ("deck_20.pptx", True),
    "pptx_100_off": ("deck_100.pptx", False),
    "pptx_100": ("deck_100.pptx", True),
    "audio_30": ("audio_30.m4a", True),
    "audio_120": ("audio_120.m4a", True),
    "video_small": ("video_small.mp4", True),
    "video_big": ("video_big.mp4", True),
    "batch8": (None, True),
    # Stored images / frames (stored-images plan WP9): same inputs, with a raw-file
    # repository that stores the described originals (and consumes and drops the bytes).
    "docx_100_stored": ("doc_100.docx", True),
    "pdf_images_stored": ("doc300.pdf", True),
    "video_small_stored": ("video_small.mp4", True),
}


class _DiscardingRawRepo:
    """Raw-file repository that **stores images** but keeps nothing: every upload
    is consumed (streamed from disk in 256 KiB reads for files) and dropped, so
    the measured peak is what the library itself holds to store them."""

    stores_images = True

    def __init__(self) -> None:
        self.images = 0
        self.largest = 0

    @staticmethod
    def _drain(path) -> int:
        size = 0
        with open(path, "rb") as fh:
            while chunk := fh.read(256 * 1024):
                size += len(chunk)
        return size

    async def store_file(self, path, **kwargs) -> str:
        self._drain(path)
        return "bench-raw-file"

    async def store_image(self, raw_file_id, page_index, image_index, content, **kwargs) -> str:
        self.images += 1
        self.largest = max(self.largest, len(content))
        return f"img-{self.images}"

    async def store_image_file(self, path, raw_file_id, page_index, image_index, **kwargs) -> str:
        self.images += 1
        self.largest = max(self.largest, self._drain(path))
        return f"img-{self.images}"

    async def delete_images(self, raw_file_id, **kwargs) -> int:
        return 0


async def _run(case: str) -> None:
    from scinr.newton.config import configure
    from scinr.newton.converters import main as converters_main

    source, describe = CASES[case]
    in_flight = {"now": 0, "max": 0}
    configure(
        neo4j_user="u", neo4j_password="p", neo4j_database="d",
        vision_llm=_fake_vision(), transcriber=_fake_transcriber(in_flight),
        describe_embedded_images=describe, mistral_api_key="bench", log_level="WARNING",
    )
    chunk_bytes = _fake_ocr(prebuild_pages=300 if case.startswith("pdf_images") else 0)

    if case == "batch8":
        from scinr.newton.converters import audio as audio_module

        original = audio_module.AudioConverter._convert

        async def _tracked(self, *args, **kwargs):
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
            try:
                return await original(self, *args, **kwargs)
            finally:
                in_flight["now"] -= 1

        audio_module.AudioConverter._convert = _tracked
        batch = DIR / "batch"
        batch.mkdir(exist_ok=True)
        for i in range(8):
            target = batch / f"a{i}.m4a"
            if not target.exists():
                target.symlink_to(DIR / f"batch_{i}.m4a")
        tracemalloc.start()
        t0 = time.time()
        written, failures = await converters_main.convert_folder(
            batch, DIR / "batch_out", parallel_docs=4
        )
        shutil.rmtree(DIR / "batch_out", ignore_errors=True)
        assert not failures, failures
        extra = f" max_in_flight={in_flight['max']}"
    else:
        path = DIR / source
        repo = _DiscardingRawRepo() if case.endswith("_stored") else None
        tracemalloc.start()
        t0 = time.time()
        written, failures = await converters_main.convert_one(path, None, raw_file_repo=repo)
        assert not failures, failures
        doc = written[0][2]
        assert all(not img.has_content for p in doc.pages for img in p.images)
        extra = f" pages={len(doc.pages)} file={path.stat().st_size / MIB:.1f}MiB"
        if repo is not None:
            assert repo.images, "nothing was stored"
            extra += f" stored={repo.images} largest_original={repo.largest / 1024:.0f}KiB"
        if chunk_bytes:
            extra += f" one_chunk_response={chunk_bytes / MIB:.1f}MiB"
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{case}: peak={peak / MIB:.1f}MiB time={time.time() - t0:.1f}s{extra}")


def main() -> None:
    if sys.argv[1] == "make":
        make()
    elif sys.argv[1] == "run":
        asyncio.run(_run(sys.argv[2]))
    elif sys.argv[1] == "all":
        for case in CASES:
            subprocess.run([sys.executable, __file__, "run", case], check=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
