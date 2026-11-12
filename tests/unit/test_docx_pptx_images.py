"""WP6 (0.5.0): embedded images of DOCX and PPTX.

Enabled: lazy ``PageImage`` objects (no base64 until described), described in
place by the enrichment step. Disabled: no ``PageImage`` at all and no access
to any image blob.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from test_converters_enrichment import GREEN, FakeVision  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.converters import main as converters_main  # noqa: E402
from scinr.newton.converters.docx import DocxConverter  # noqa: E402
from scinr.newton.converters.enrichment import render_image_block  # noqa: E402
from scinr.newton.converters.pptx import PptxConverter  # noqa: E402

docx = pytest.importorskip("docx")
pptx = pytest.importorskip("pptx")

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "neo4j_database": "neo4j"}
BLUE = (0, 0, 255)


def _png_file(tmp_path: Path, name: str, colour) -> Path:
    path = tmp_path / name
    Image.new("RGB", (240, 160), colour).save(path, format="PNG")
    return path


def _docx(tmp_path: Path) -> Path:
    document = docx.Document()
    document.add_paragraph("Antes de la figura")
    document.add_picture(str(_png_file(tmp_path, "g.png", GREEN)))
    document.add_paragraph("Después de la figura")
    path = tmp_path / "informe.docx"
    document.save(path)
    return path


def _pptx(tmp_path: Path) -> Path:
    from pptx.util import Inches

    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Resultados"
    slide.shapes.add_picture(str(_png_file(tmp_path, "g.png", GREEN)), Inches(1), Inches(2))
    slide.shapes.add_picture(str(_png_file(tmp_path, "b.png", BLUE)), Inches(5), Inches(2))
    path = tmp_path / "deck.pptx"
    presentation.save(path)
    return path


@pytest.fixture
def vision():
    model = FakeVision()
    model.calls = []
    return model


@pytest.fixture
def no_blob_access(monkeypatch):
    """Fail the test if the converters build an image or touch a picture's blob."""
    from pptx.shapes.picture import Picture

    from scinr.newton.converters import docx as docx_module
    from scinr.newton.converters.base import PageImage

    def _boom(*args, **kwargs):
        raise AssertionError("image extracted while images are disabled")

    monkeypatch.setattr(Picture, "image", property(_boom))
    monkeypatch.setattr(docx_module, "_inline_images", _boom)
    monkeypatch.setattr(PageImage, "with_loader", classmethod(_boom))


class TestDocx:
    def test_enabled_gives_lazy_images_with_reference(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        doc = DocxConverter().convert(_docx(tmp_path))
        (page,) = doc.pages
        (image,) = page.images
        assert image.ref == "docx-img-0" and image.index == 0
        assert image.base64 == "" and image.has_content
        assert image.media_type == "image/png"
        assert "![docx-img-0](docx-img-0)" in page.markdown
        assert image.load_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    async def test_block_lands_between_the_two_paragraphs(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        written, failures = await converters_main.convert_one(_docx(tmp_path), None)
        assert failures == []
        markdown = written[0][2].pages[0].markdown
        block = render_image_block(0, "desc 0,3,0")
        assert markdown == f"Antes de la figura\n\n{block}\n\nDespués de la figura"
        assert not written[0][2].pages[0].images[0].has_content

    def test_disabled_extracts_nothing(self, tmp_path, no_blob_access):
        configure(**_NEO)  # no vision_llm
        doc = DocxConverter().convert(_docx(tmp_path))
        assert all(page.images == [] for page in doc.pages)
        assert "docx-img" not in doc.pages[0].markdown


class TestPptx:
    async def test_two_images_described_at_the_end_of_the_slide(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        written, failures = await converters_main.convert_one(_pptx(tmp_path), None)
        assert failures == []
        page = written[0][2].pages[0]
        assert page.markdown.startswith("# Resultados")
        assert page.markdown.endswith(
            render_image_block(0, "desc 0,3,0") + "\n\n" + render_image_block(1, "desc 0,0,5") + "\n"
        )
        assert all(not img.has_content for img in page.images)
        assert len(vision.calls) == 2

    def test_pictures_inside_groups_are_extracted(self, tmp_path, vision):
        from pptx.util import Inches

        configure(**_NEO, vision_llm=vision)
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_picture(str(_png_file(tmp_path, "g.png", GREEN)), Inches(1), Inches(1))
        outer = slide.shapes.add_group_shape()
        outer.shapes.add_picture(str(_png_file(tmp_path, "b.png", BLUE)), Inches(4), Inches(1))
        inner = outer.shapes.add_group_shape()
        inner.shapes.add_picture(str(_png_file(tmp_path, "r.png", (200, 0, 200))), Inches(6), Inches(1))
        path = tmp_path / "grupos.pptx"
        presentation.save(path)

        images = PptxConverter().convert(path).pages[0].images
        assert [i.index for i in images] == [0, 1, 2]
        assert all(i.has_content for i in images)

    def test_enabled_images_are_lazy(self, tmp_path, vision):
        configure(**_NEO, vision_llm=vision)
        page = PptxConverter().convert(_pptx(tmp_path)).pages[0]
        assert [i.index for i in page.images] == [0, 1]
        assert all(i.base64 == "" and i.has_content and i.ref is None for i in page.images)

    def test_disabled_extracts_nothing(self, tmp_path, no_blob_access):
        configure(**_NEO, describe_embedded_images=False)
        doc = PptxConverter().convert(_pptx(tmp_path))
        assert doc.pages[0].images == []
        assert '"base64"' not in doc.to_json()


def test_png_helper_is_a_png():
    out = io.BytesIO()
    Image.new("RGB", (1, 1)).save(out, format="PNG")
    assert out.getvalue()[:4] == b"\x89PNG"
