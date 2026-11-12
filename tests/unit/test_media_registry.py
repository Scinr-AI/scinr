"""WP12 (0.5.0): image, audio and video converters are always registered.

A media file that used to be skipped is now a regular unit: without its
requirement it becomes a failed document whose error says what to configure,
and the rest of the batch is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scinr.newton.config import configure
from scinr.newton.converters.audio import AudioConverter
from scinr.newton.converters.image import ImageConverter
from scinr.newton.converters.registry import get_converter
from scinr.newton.converters.video import VideoConverter
from scinr.newton.pipeline_units import _discover_raw_inputs
from scinr.newton.stages.preprocess import run_preprocess

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}


@pytest.mark.parametrize(
    ("ext", "cls"),
    [
        *[(e, ImageConverter) for e in ("jpg", "jpeg", "png", "webp", "tif", "tiff", "heic", "heif")],
        *[(e, AudioConverter) for e in ("mp3", "m4a", "wav", "ogg", "flac")],
        *[(e, VideoConverter) for e in ("mp4", "mov", "mkv", "webm")],
    ],
)
def test_extension_is_registered(ext, cls):
    assert isinstance(get_converter(Path(f"x.{ext.upper()}")), cls)


def test_media_files_are_units_not_skipped(tmp_path):
    for name in ("a.txt", "b.mp3", "c.jpg", "d.mp4"):
        (tmp_path / name).write_bytes(b"x")
    inputs = _discover_raw_inputs(str(tmp_path), tabular_extensions=None)
    assert sorted(u.doc_path for u in inputs.units) == ["a.txt", "b.mp3", "c.jpg", "d.mp4"]
    assert inputs.skipped == []


async def test_missing_requirements_fail_only_their_documents(tmp_path):
    configure(**_NEO)  # no vision_llm, no transcriber
    (tmp_path / "notas.txt").write_text("hola", encoding="utf-8")
    (tmp_path / "charla.mp3").write_bytes(b"not audio")
    Image.new("RGB", (32, 32)).save(tmp_path / "foto.jpg")

    result, docs = await run_preprocess(input_raw=str(tmp_path))

    by_name = {d.document_name: d for d in result.documents}
    assert by_name["notas.txt"].nodes_failed == 0
    assert [d.document_name for d in docs] == ["notas.txt"]
    assert "transcriber" in by_name["charla.mp3"].errors[0]
    assert "vision_llm" in by_name["foto.jpg"].errors[0]
    assert result.total_failed == 2
