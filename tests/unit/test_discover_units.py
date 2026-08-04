"""
tests/unit/test_discover_units.py — Discovery-layer tests for
scinr.newton.pipeline_units._discover_units().

All filesystem-based branches (raw_file, extraction_json, ingestion_json) are
verified against a manually-replicated version of the legacy doc_path
derivation logic they must match exactly:

    raw_file        -> scinr.newton.converters.main.convert_one() (folder_path_str
                        from _relative_prefix) + stages/extraction.py
                        ._process_intermediate() (doc_path = folder_path/name).
    extraction_json -> scinr.newton.stages.extraction.run_extraction()
                        ._process_file() (doc_path built from the JSON's own
                        "folder_path" field).
    ingestion_json  -> scinr.newton.ingest.loader._read_doc_path().

The Neo4j-backed branch (document_names) is exercised with
resolve_leaf_document_names monkeypatched — no real Neo4j connection.
"""

from __future__ import annotations

import json

import pytest

from scinr.newton.pipeline_units import (
    DocumentUnit,
    _discover_extraction_json_units,
    _discover_ingestion_json_units,
    _discover_pre_ingested_units,
    _discover_raw_file_units,
    _discover_units,
    build_all_paths_for_versioning,
)

# ---------------------------------------------------------------------------
# raw_file
# ---------------------------------------------------------------------------


def test_discover_raw_file_units_nested_two_levels(tmp_path):
    """doc_path for a nested raw file must equal
    f"{relative_parent_dir}/{stem}" — mirroring convert_one()'s
    folder_path_str (from _relative_prefix) combined with
    stages/extraction.py's `f"{folder_path}/{name}" if folder_path else name`
    rule (see module docstrings cited above).
    """
    root_file = tmp_path / "root_doc.txt"
    root_file.write_text("hello", encoding="utf-8")

    nested_dir = tmp_path / "ModuloA" / "SubModulo"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "nested_doc.txt"
    nested_file.write_text("world", encoding="utf-8")

    units = _discover_raw_file_units(str(tmp_path), tabular_extensions=None)
    by_path = {u.source_path: u for u in units}

    root_unit = by_path[root_file]
    assert root_unit.kind == "raw_file"
    assert root_unit.doc_path == "root_doc"  # no folder prefix at root level
    assert root_unit.document_name_hint == "root_doc"

    nested_unit = by_path[nested_file]
    assert nested_unit.kind == "raw_file"
    assert nested_unit.doc_path == "ModuloA/SubModulo/nested_doc"
    assert nested_unit.document_name_hint == "nested_doc"


def test_discover_raw_file_units_excludes_tabular_extensions(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

    units = _discover_raw_file_units(str(tmp_path), tabular_extensions={".csv"})

    assert len(units) == 1
    assert units[0].document_name_hint == "doc"


def test_discover_raw_file_units_skips_unsupported_format(tmp_path):
    (tmp_path / "unsupported.xyz").write_text("binary-ish", encoding="utf-8")
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

    units = _discover_raw_file_units(str(tmp_path), tabular_extensions=None)

    assert len(units) == 1
    assert units[0].document_name_hint == "doc"


def test_discover_raw_file_units_missing_root_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        _discover_raw_file_units(str(missing), tabular_extensions=None)


# ---------------------------------------------------------------------------
# extraction_json
# ---------------------------------------------------------------------------


def test_discover_extraction_json_units_nested_two_levels(tmp_path):
    """doc_path must be built from the JSON's own "folder_path" field —
    mirroring stages/extraction.py::run_extraction()._process_file():
        name, ext = os.path.splitext(json_file); nombre = Path(name).name
        doc_path = f"{folder_path}/{nombre}" if folder_path else nombre
    NOT from the file's actual location on disk (deliberately placed in a
    mismatched directory below to prove this).
    """
    root_json = tmp_path / "root_doc.json"
    root_json.write_text(json.dumps({"pages": [], "folder_path": None}), encoding="utf-8")

    nested_dir = tmp_path / "ModuloA" / "SubModulo"
    nested_dir.mkdir(parents=True)
    nested_json = nested_dir / "nested_doc.json"
    # folder_path intentionally differs from the file's actual directory to
    # prove doc_path comes from the JSON content, not from disk location.
    nested_json.write_text(
        json.dumps({"pages": [], "folder_path": "ModuloA/SubModulo"}), encoding="utf-8"
    )

    units = _discover_extraction_json_units(str(tmp_path))
    by_path = {u.source_path: u for u in units}

    root_unit = by_path[root_json]
    assert root_unit.kind == "extraction_json"
    assert root_unit.doc_path == "root_doc"

    nested_unit = by_path[nested_json]
    assert nested_unit.kind == "extraction_json"
    assert nested_unit.doc_path == "ModuloA/SubModulo/nested_doc"
    assert nested_unit.document_name_hint == "nested_doc"


def test_discover_extraction_json_units_missing_dir_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        _discover_extraction_json_units(str(missing))


def test_discover_extraction_json_units_skips_unreadable_file(tmp_path):
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "good.json").write_text(
        json.dumps({"pages": [], "folder_path": None}), encoding="utf-8"
    )

    units = _discover_extraction_json_units(str(tmp_path))

    assert len(units) == 1
    assert units[0].document_name_hint == "good"


# ---------------------------------------------------------------------------
# ingestion_json
# ---------------------------------------------------------------------------


def test_discover_ingestion_json_units_nested_two_levels(tmp_path):
    """doc_path must be read via ingest.loader._read_doc_path(): prefer the
    file's own "doc_path" field; else fall back to
    f"{folder_path}/{name}" where name = stem with the "extract-" prefix
    stripped.
    """
    root_json = tmp_path / "extract-root_doc.json"
    root_json.write_text(json.dumps({"doc_path": "root_doc"}), encoding="utf-8")

    nested_dir = tmp_path / "ModuloA" / "SubModulo"
    nested_dir.mkdir(parents=True)
    nested_json = nested_dir / "extract-nested_doc.json"
    # No explicit doc_path -> _read_doc_path() falls back to
    # f"{folder_path}/{name}" using the JSON's own folder_path field.
    nested_json.write_text(json.dumps({"folder_path": "ModuloA/SubModulo"}), encoding="utf-8")

    units = _discover_ingestion_json_units(str(tmp_path))
    by_path = {u.source_path: u for u in units}

    root_unit = by_path[root_json]
    assert root_unit.kind == "ingestion_json"
    assert root_unit.doc_path == "root_doc"
    assert root_unit.document_name_hint == "root_doc"

    nested_unit = by_path[nested_json]
    assert nested_unit.kind == "ingestion_json"
    assert nested_unit.doc_path == "ModuloA/SubModulo/nested_doc"
    assert nested_unit.document_name_hint == "nested_doc"


def test_discover_ingestion_json_units_only_matches_extract_prefix(tmp_path):
    """Files not matching the extract-*.json glob must be ignored, mirroring
    ingest.loader._FILE_GLOB / load_folder().
    """
    (tmp_path / "not-an-extract-file.json").write_text(
        json.dumps({"doc_path": "irrelevant"}), encoding="utf-8"
    )
    (tmp_path / "extract-good.json").write_text(json.dumps({"doc_path": "good"}), encoding="utf-8")

    units = _discover_ingestion_json_units(str(tmp_path))

    assert len(units) == 1
    assert units[0].doc_path == "good"


def test_discover_ingestion_json_units_missing_dir_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        _discover_ingestion_json_units(str(missing))


# ---------------------------------------------------------------------------
# pre_ingested (document_names) — Neo4j mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_pre_ingested_units_resolves_and_dedupes(monkeypatch):
    """A folder name resolving to 3 leaves must yield 3 units; two input
    names that share a leaf must not produce a duplicate unit for it.
    """

    def fake_resolve(driver, document_name):
        if document_name == "FolderDoc":
            return ["LeafA", "LeafB", "LeafC"]
        if document_name == "OtherDoc":
            return ["LeafC", "LeafD"]  # LeafC overlaps with FolderDoc's result
        return [document_name]

    class _FakeDriver:
        def close(self):
            pass

    monkeypatch.setattr(
        "scinr.newton.utils.document_resolver.resolve_leaf_document_names",
        fake_resolve,
    )
    monkeypatch.setattr(
        "scinr.newton.ingest.config.get_driver",
        lambda: _FakeDriver(),
    )

    units = await _discover_pre_ingested_units(["FolderDoc", "OtherDoc"])

    doc_paths = [u.doc_path for u in units]
    assert doc_paths == ["LeafA", "LeafB", "LeafC", "LeafD"]  # dedup, order preserved
    assert all(u.kind == "pre_ingested" for u in units)
    assert all(u.source_path is None for u in units)


@pytest.mark.asyncio
async def test_discover_pre_ingested_units_closes_driver(monkeypatch):
    closed = {"value": False}

    class _FakeDriver:
        def close(self):
            closed["value"] = True

    monkeypatch.setattr(
        "scinr.newton.utils.document_resolver.resolve_leaf_document_names",
        lambda driver, name: [name],
    )
    monkeypatch.setattr(
        "scinr.newton.ingest.config.get_driver",
        lambda: _FakeDriver(),
    )

    await _discover_pre_ingested_units(["SomeDoc"])

    assert closed["value"] is True


# ---------------------------------------------------------------------------
# document_names_dir
# ---------------------------------------------------------------------------


def test_discover_units_document_names_dir_reads_names(tmp_path):
    (tmp_path / "extract-a.json").write_text(
        json.dumps({"document_name": "DocA"}), encoding="utf-8"
    )
    (tmp_path / "extract-b.json").write_text(
        json.dumps({"document_name": "DocB"}), encoding="utf-8"
    )

    import asyncio

    units = asyncio.run(_discover_units(document_names_dir=str(tmp_path)))

    doc_paths = sorted(u.doc_path for u in units)
    assert doc_paths == ["DocA", "DocB"]
    assert all(u.kind == "pre_ingested" for u in units)
    assert all(u.source_path is None for u in units)


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_units_rejects_multiple_sources(tmp_path):
    with pytest.raises(ValueError):
        await _discover_units(input_raw=str(tmp_path), extraction_input_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_discover_units_rejects_no_source():
    with pytest.raises(ValueError):
        await _discover_units()


@pytest.mark.asyncio
async def test_discover_units_dispatches_raw_file(tmp_path):
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

    units = await _discover_units(input_raw=str(tmp_path))

    assert len(units) == 1
    assert units[0].kind == "raw_file"


# ---------------------------------------------------------------------------
# build_all_paths_for_versioning
# ---------------------------------------------------------------------------


def test_build_all_paths_for_versioning_includes_ancestors():
    units = [
        DocumentUnit(
            kind="raw_file",
            source_path=None,
            doc_path="ModuloA/SubModulo/doc_a",
            relative_dir=None,
            document_name_hint="doc_a",
        ),
        DocumentUnit(
            kind="raw_file",
            source_path=None,
            doc_path="ModuloA/SubModulo/doc_b",
            relative_dir=None,
            document_name_hint="doc_b",
        ),
    ]

    all_paths = set(build_all_paths_for_versioning(units))

    assert all_paths == {
        "ModuloA/SubModulo/doc_a",
        "ModuloA/SubModulo/doc_b",
        "ModuloA/SubModulo",
        "ModuloA",
    }
