"""WP7 (0.5.0): image files converted with the ``vision_llm`` (standalone mode)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from test_converters_enrichment import GREEN, RED, FakeVision  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.converters import image as image_module  # noqa: E402
from scinr.newton.converters.base import ConversionError  # noqa: E402
from scinr.newton.converters.image import ImageConverter, read_image_metadata  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}


def _jpeg_with_exif(path: Path, *, orientation: int = 1) -> Path:
    image = Image.new("RGB", (400, 200), GREEN)
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "EOS R6"
    exif[0x0112] = orientation
    exif.get_ifd(0x8769)[0x9003] = "2024:05:01 10:30:00"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (40.0, 25.0, 1.2)
    gps[3] = "W"
    gps[4] = (3.0, 42.0, 9.0)
    image.save(path, format="JPEG", exif=exif.tobytes())
    return path


@pytest.fixture
def vision():
    model = FakeVision()
    model.calls = []
    return model


class TestMetadata:
    def test_exif_fields_without_location_by_default(self, tmp_path):
        data = _jpeg_with_exif(tmp_path / "a.jpg").read_bytes()
        meta = read_image_metadata(data)
        assert (meta.width, meta.height) == (400, 200)
        assert meta.captured_at == "2024-05-01T10:30:00"
        assert (meta.device_make, meta.device_model) == ("Canon", "EOS R6")
        assert meta.gps_latitude is None and meta.gps_longitude is None

    def test_location_only_when_not_stripped(self, tmp_path):
        data = _jpeg_with_exif(tmp_path / "a.jpg").read_bytes()
        meta = read_image_metadata(data, strip_location=False)
        assert meta.gps_latitude == pytest.approx(40.41700, abs=1e-4)
        assert meta.gps_longitude == pytest.approx(-3.70250, abs=1e-4)

    def test_orientation_swaps_the_reported_size(self, tmp_path):
        data = _jpeg_with_exif(tmp_path / "a.jpg", orientation=6).read_bytes()
        assert (read_image_metadata(data).width, read_image_metadata(data).height) == (200, 400)

    def test_png_without_exif(self, tmp_path):
        path = tmp_path / "b.png"
        Image.new("RGB", (64, 32), GREEN).save(path)
        meta = read_image_metadata(path.read_bytes())
        assert (meta.width, meta.height, meta.captured_at, meta.device_make) == (64, 32, None, None)


class TestConvert:
    async def test_one_page_with_description_and_metadata(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        source = _jpeg_with_exif(tmp_path / "foto.JPG", orientation=6)

        doc = await ImageConverter().convert(source)

        assert [p.index for p in doc.pages] == [0]
        assert doc.pages[0].markdown == "desc 0,3,0"
        assert doc.pages[0].images == []
        assert doc.file_metadata.device_model == "EOS R6"
        assert doc.file_metadata.gps_latitude is None
        (call,) = vision.calls
        assert call["size"] == (200, 400)  # orientation applied before sending
        assert "Convert this image" in call["text"]

    async def test_location_kept_when_configured(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision, strip_location=False)
        doc = await ImageConverter().convert(_jpeg_with_exif(tmp_path / "a.jpg"))
        assert doc.file_metadata.gps_latitude is not None

    async def test_without_vision_llm_fails_before_reading(self, tmp_path, monkeypatch):
        configure(**_NEO)
        source = tmp_path / "a.png"
        Image.new("RGB", (10, 10)).save(source)
        monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("file was read"))
        with pytest.raises(ConversionError, match="vision_llm"):
            await ImageConverter().convert(source)

    async def test_heic_without_pillow_heif(self, tmp_path, vision, monkeypatch):
        configure(**_NEO, vision_llm=vision)
        monkeypatch.setattr(image_module, "_heif_available", lambda: False)
        source = tmp_path / "a.heic"
        source.write_bytes(b"\x00")
        with pytest.raises(ConversionError, match=r"scinr\[media\]"):
            await ImageConverter().convert(source)

    async def test_model_failure_is_a_conversion_error(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        source = tmp_path / "red.png"
        Image.new("RGB", (100, 100), RED).save(source)
        with pytest.raises(ConversionError, match="exploded"):
            await ImageConverter().convert(source)
