"""
tests/unit/test_discover_documents.py — scinr.newton.discover_documents().

discover_documents() must report exactly what run_pipeline() discovers: the
same walk (``_discover_raw_inputs``) with the same default tabular extensions,
and the ``doc_path`` each file gets in run_tabular_pipeline() and in the
convertible-document stages. No conversion, no Neo4j.
"""

from __future__ import annotations

import pytest

from scinr.newton import DiscoveredDocuments, SkippedFile, discover_documents
from scinr.newton.pipeline import _DEFAULT_TABULAR_EXTENSIONS
from scinr.newton.pipeline_units import _discover_raw_inputs
from scinr.newton.utils.naming import tabular_doc_path


def _folder(tmp_path):
    (tmp_path / "informe.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "ventas.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "raro.xyz").write_text("?", encoding="utf-8")
    nested = tmp_path / "ModuloA" / "SubModulo"
    nested.mkdir(parents=True)
    (nested / "notas.txt").write_text("hola", encoding="utf-8")
    (nested / "datos.xlsx").write_bytes(b"PK")
    return tmp_path


def test_sorts_each_file_into_its_list(tmp_path):
    result = discover_documents(str(_folder(tmp_path)))

    assert isinstance(result, DiscoveredDocuments)
    assert result.tabular == ["ModuloA/SubModulo/datos.xlsx", "ventas.csv"]
    assert result.documents == ["ModuloA/SubModulo/notas.txt", "informe.pdf"]
    assert result.skipped == [
        SkippedFile(path=str(tmp_path / "raro.xyz"), reason="unsupported format: .xyz")
    ]
    assert result.count == 4


def test_matches_what_run_pipeline_discovers(tmp_path):
    """Same walk and same default extensions as run_pipeline(), and the
    tabular doc_path rule of run_tabular_pipeline()."""
    root = _folder(tmp_path)

    expected = _discover_raw_inputs(str(root), _DEFAULT_TABULAR_EXTENSIONS)
    result = discover_documents(str(root))

    assert result.documents == [unit.doc_path for unit in expected.units]
    assert result.tabular == [tabular_doc_path(f, root) for f in expected.tabular]
    assert len(result.skipped) == len(expected.skipped)


def test_custom_tabular_extensions_change_the_split(tmp_path):
    root = _folder(tmp_path)

    result = discover_documents(str(root), tabular_extensions={".csv"})

    assert result.tabular == ["ventas.csv"]
    assert "ModuloA/SubModulo/datos.xlsx" in result.documents
    assert result.count == 4


def test_empty_folder_counts_zero(tmp_path):
    result = discover_documents(str(tmp_path))

    assert result == DiscoveredDocuments(tabular=[], documents=[], skipped=[])
    assert result.count == 0


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_documents(str(tmp_path / "does-not-exist"))
