"""Document name and ``doc_path`` keep the source file's extension (0.5.0).

Covers the rule in ``utils/naming.py`` end to end: the converter output, the
``doc_path`` predicted at discovery, and the ``doc_path`` the extraction stage
produces from the in-memory document and from the JSON written on disk must
all agree. JSON files written by earlier versions (``informe.json``) keep
their extension-less name.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scinr.newton.converters.main import convert_folder, convert_single_file
from scinr.newton.pipeline_units import _discover_raw_inputs
from scinr.newton.stages import extraction
from scinr.newton.utils.naming import source_doc_path, source_document_name


@pytest.fixture
def no_llm_extraction(monkeypatch):
    """Run the extraction entry points without an LLM: return the Document as built."""

    async def _fake_run_chunk_extraction(document, *args, **kwargs):
        return document

    monkeypatch.setattr(extraction, "_run_chunk_extraction", _fake_run_chunk_extraction)
    monkeypatch.setattr(extraction, "get_llm", lambda: None)
    monkeypatch.setattr(
        extraction, "get_config", lambda: SimpleNamespace(extraction_batch_size=10)
    )


def test_source_document_name_keeps_extension_case_and_dots():
    assert source_document_name(Path("Modulo/Informe.PDF")) == "Informe.PDF"
    assert source_document_name(Path("a.b.PDF")) == "a.b.PDF"
    assert source_doc_path(Path("x/a.b.PDF"), "Modulo/Sub") == "Modulo/Sub/a.b.PDF"
    assert source_doc_path(Path("a.b.PDF"), None) == "a.b.PDF"
    assert source_doc_path(Path("a.b.PDF"), "") == "a.b.PDF"


async def test_same_stem_different_extension_gives_two_documents(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "informe.txt").write_text("texto plano", encoding="utf-8")
    (src / "informe.html").write_text("<p>html</p>", encoding="utf-8")
    out = tmp_path / "out"

    written, failures = await convert_folder(src, out)

    assert failures == []
    assert sorted(j.name for _, j, _ in written) == ["informe.html.json", "informe.txt.json"]
    names = sorted(
        json.loads(j.read_text(encoding="utf-8"))["document_name"] for _, j, _ in written
    )
    assert names == ["informe.html", "informe.txt"]
    units = _discover_raw_inputs(str(src), tabular_extensions=None).units
    assert sorted(u.doc_path for u in units) == ["informe.html", "informe.txt"]


async def test_predicted_doc_path_matches_extraction_in_memory_and_from_disk(
    tmp_path, no_llm_extraction
):
    src = tmp_path / "in"
    nested = src / "Modulo" / "Sub"
    nested.mkdir(parents=True)
    (nested / "a.b.TXT").write_text("contenido", encoding="utf-8")
    out = tmp_path / "out"

    (unit,) = _discover_raw_inputs(str(src), tabular_extensions=None).units
    assert unit.doc_path == "Modulo/Sub/a.b.TXT"
    assert unit.document_name_hint == "a.b.TXT"

    written, failures = await convert_folder(src, out)
    assert failures == []
    ((_, json_path, _),) = written
    assert json_path.name == "a.b.TXT.json"

    from scinr.newton.converters.base import IntermediateDocument

    doc = IntermediateDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
    in_memory = await extraction.extract_one_intermediate(doc, output_path=None)
    from_disk = await extraction.extract_one_file(json_path, None, str(out))

    assert in_memory.doc_path == unit.doc_path
    assert from_disk.doc_path == unit.doc_path
    assert in_memory.document_name == from_disk.document_name == "a.b.TXT"


async def test_old_json_without_extension_keeps_old_name(tmp_path, no_llm_extraction):
    old = tmp_path / "informe.json"
    old.write_text(
        json.dumps(
            {
                "source_file": "informe.pdf",
                "source_format": "pdf",
                "document_name": "informe",
                "pages": [{"index": 0, "markdown": "hola"}],
            }
        ),
        encoding="utf-8",
    )

    doc = await extraction.extract_one_file(old, None, str(tmp_path))

    assert doc.document_name == "informe"
    assert doc.doc_path == "informe"


async def test_convert_single_file_names_output_with_extension(tmp_path):
    src = tmp_path / "Notas.Finales.txt"
    src.write_text("hola", encoding="utf-8")

    out = await convert_single_file(src, tmp_path / "out")

    assert out.name == "Notas.Finales.txt.json"
    assert json.loads(out.read_text(encoding="utf-8"))["document_name"] == "Notas.Finales.txt"
