"""0.5.0 review fixes: what a best-effort conversion reports, and the decode bound.

- A document whose every part failed in best-effort mode fails (any converter),
  instead of producing no ``:Document`` without an error.
- The parts skipped in best-effort mode reach ``DocumentResult`` /
  ``PipelineResult.documents_with_omissions`` and a closing warning.
- Embedded images are read and decoded one at a time, whatever
  ``llm_concurrency``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_converters_enrichment import FakeVision, _png  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.converters import main as converters_main  # noqa: E402
from scinr.newton.converters import vision as vision_module  # noqa: E402
from scinr.newton.converters.base import (  # noqa: E402
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    IntermediatePage,
    PageImage,
)
from scinr.newton.converters.enrichment import EnrichmentState, enrich_pages  # noqa: E402
from scinr.newton.media.transcriber import Transcriber  # noqa: E402
from scinr.newton.results import DocumentResult, PipelineResult, StageResult  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}


class _Fixed(BaseConverter):
    """Returns a prepared document, as any (external) converter could."""

    supported_extensions = frozenset({"x"})

    def __init__(self, doc: IntermediateDocument) -> None:
        self.doc = doc

    def convert(self, source: Path) -> IntermediateDocument:
        return self.doc


def _doc(*markdowns: str, **missing) -> IntermediateDocument:
    pages = [IntermediatePage(index=i, markdown=m) for i, m in enumerate(markdowns)]
    return IntermediateDocument(pages=pages, **missing)


# ---------------------------------------------------------------------------
# Nothing converted → the document fails, for every converter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc",
    [
        _doc(missing_page_ranges=[(0, 50), (50, 80)]),  # PDF: every chunk failed
        _doc(missing_segments=[(0, 300_000)]),  # audio: every segment failed
        _doc("", "  \n", missing_images=[(0, 0), (1, 0)]),  # images only, all failed
    ],
    ids=["pdf", "audio", "images-only"],
)
async def test_everything_skipped_fails_the_document(doc, tmp_path):
    configure(**_NEO)
    with pytest.raises(ConversionError, match=r"Nothing could be converted from f\.x"):
        await converters_main._run_convert(_Fixed(doc), tmp_path / "f.x")


@pytest.mark.parametrize(
    "doc",
    [
        _doc("texto", missing_page_ranges=[(1, 2)]),  # partial: kept
        _doc(),  # empty and nothing failed (a silent audio): not an error
        _doc(""),  # a blank page, nothing failed
    ],
    ids=["partial", "empty", "blank"],
)
async def test_partial_or_empty_documents_are_kept(doc, tmp_path):
    configure(**_NEO)
    assert await converters_main._run_convert(_Fixed(doc), tmp_path / "f.x") is doc


class _Rejecting(Transcriber):
    async def transcribe(self, path, language=None):
        raise RuntimeError("HTTP 401 invalid api key")


async def test_audio_with_every_segment_failed_is_a_failed_document(tmp_path, monkeypatch):
    from scinr.newton.media import ffmpeg
    from scinr.newton.stages.preprocess import run_preprocess

    monkeypatch.setattr(
        ffmpeg, "find_binaries", lambda a, b: ffmpeg.Binaries(ffmpeg="/ff", ffprobe="/fp")
    )

    async def _probe(ffprobe, path):
        return ffmpeg.MediaInfo(duration_ms=25_000, has_audio=True, has_video=False)

    async def _extract(ff, src, start_s, duration_s, dst):
        dst.write_bytes(b"flac")

    monkeypatch.setattr(ffmpeg, "probe", _probe)
    monkeypatch.setattr(ffmpeg, "extract_audio_segment", _extract)
    configure(**_NEO, transcriber=_Rejecting(), media_segment_seconds=10)
    (tmp_path / "charla.mp3").write_bytes(b"x")
    (tmp_path / "notas.txt").write_text("hola", encoding="utf-8")

    result, docs = await run_preprocess(input_raw=str(tmp_path))

    by_name = {d.document_name: d for d in result.documents}
    assert result.success is False
    assert by_name["charla.mp3"].nodes_failed == 1
    assert "Nothing could be converted from charla.mp3" in by_name["charla.mp3"].errors[0]
    assert "missing_segments=[(0, 10000), (10000, 20000), (20000, 25000)]" in (
        by_name["charla.mp3"].errors[0]
    )
    assert [d.document_name for d in docs] == ["notas.txt"]


# ---------------------------------------------------------------------------
# Omissions reach the results
# ---------------------------------------------------------------------------


async def test_preprocess_result_carries_the_omissions(tmp_path, monkeypatch):
    from scinr.newton.stages.preprocess import run_preprocess

    configure(**_NEO)
    partial = _doc("texto", missing_page_ranges=[(3, 5)], missing_images=[(0, 1)])
    monkeypatch.setattr(converters_main, "get_converter", lambda path: _Fixed(partial))
    (tmp_path / "informe.pdf").write_bytes(b"%PDF")

    result, _docs = await run_preprocess(input_raw=str(tmp_path))

    (doc_result,) = result.documents
    assert result.success is True  # a partial document is processed, not failed
    assert doc_result.nodes_processed == 1
    assert doc_result.missing_page_ranges == [(3, 5)]
    assert doc_result.missing_segments is None
    assert doc_result.missing_images == [(0, 1)]
    assert doc_result.has_omissions


def test_pipeline_result_lists_documents_with_omissions():
    complete = DocumentResult("a.pdf", 1, 0)
    partial = DocumentResult("b.mp3", 1, 0, missing_segments=[(0, 300_000)])
    stage = StageResult("preprocess", True, [complete, partial], 2, 0, 1.0)
    result = PipelineResult(True, 1.0, ["preprocess"], preprocess=stage)
    assert result.documents_with_omissions == [partial]
    assert PipelineResult(True, 1.0, []).documents_with_omissions == []
    # positional construction (pre-0.5.0 callers) is unchanged
    assert DocumentResult("c", 1, 0, []).has_omissions is False


def test_closing_warning_names_each_partial_document(caplog):
    from scinr.newton.pipeline import _log_omissions

    stage = StageResult(
        "preprocess", True,
        [DocumentResult("b.mp4", 1, 0, missing_segments=[(0, 300_000)],
                        missing_images=[(1, 0)])],
        1, 0, 1.0,
    )
    with caplog.at_level(logging.WARNING, logger="scinr.newton.pipeline"):
        _log_omissions(PipelineResult(True, 1.0, ["preprocess"], preprocess=stage))
    (record,) = [r for r in caplog.records if "best_effort" in r.getMessage()]
    message = record.getMessage()
    assert "b.mp4" in message
    assert "time ranges (ms) [(0, 300000)]" in message
    assert "images (page, image) [(1, 0)]" in message


# ---------------------------------------------------------------------------
# Decode bound
# ---------------------------------------------------------------------------


async def test_embedded_images_are_read_and_decoded_one_at_a_time(monkeypatch):
    configure(**_NEO, vision_llm=FakeVision(), llm_concurrency=4)
    events: list[str] = []
    real = vision_module.prepare_image

    def _recording_prepare(data, max_side):
        events.append("decode")
        try:
            return real(data, max_side)
        finally:
            events.append("done")

    monkeypatch.setattr(vision_module, "prepare_image", _recording_prepare)

    def _image(index: int) -> PageImage:
        data = _png((0, 0, 200 - index), size=(300 + index, 200))

        def _load() -> bytes:
            events.append("read")
            return data

        return PageImage.with_loader(_load, index=index)

    page = IntermediatePage(index=0, markdown="texto", images=[_image(i) for i in range(8)])
    await enrich_pages([page], EnrichmentState())

    # The filter reads each image once (header and hash, no decode); then each
    # image is read only on its turn and decoded before the next one is read,
    # although four model calls may be in flight.
    assert events == ["read"] * 8 + ["read", "decode", "done"] * 8
    assert page.markdown.count("<image") == 8


async def test_image_decode_concurrency_sets_the_number_of_decoding_lanes(monkeypatch):
    configure(**_NEO, vision_llm=FakeVision(), llm_concurrency=4, image_decode_concurrency=3)
    in_flight = peak = 0
    real = vision_module.prepare_image
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _slow_prepare(data, max_side):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if peak == 3:
            loop.call_soon_threadsafe(release.set)
        asyncio.run_coroutine_threadsafe(asyncio.wait_for(release.wait(), 5), loop).result()
        try:
            return real(data, max_side)
        finally:
            in_flight -= 1

    monkeypatch.setattr(vision_module, "prepare_image", _slow_prepare)
    images = [PageImage(index=i, base64=base64.b64encode(_png((0, 0, 200 - i), size=(300 + i, 200))).decode())
              for i in range(6)]
    await enrich_pages([IntermediatePage(index=0, markdown="t", images=images)], EnrichmentState())
    assert peak == 3


async def test_standalone_bytes_are_also_decoded_one_at_a_time(monkeypatch):
    configure(**_NEO, vision_llm=FakeVision(), llm_concurrency=4)
    in_flight = peak = 0
    real = vision_module.prepare_image

    def _counting_prepare(data, max_side):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            return real(data, max_side)
        finally:
            in_flight -= 1

    monkeypatch.setattr(vision_module, "prepare_image", _counting_prepare)
    images = [base64.b64decode(base64.b64encode(_png((10 * i, 90, 90)))) for i in range(6)]
    results = await asyncio.gather(
        *(vision_module.describe_image(data, mode="standalone") for data in images)
    )
    assert all(results)
    assert peak == 1
