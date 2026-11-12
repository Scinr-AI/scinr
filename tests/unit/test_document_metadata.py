"""
tests/unit/test_document_metadata.py — Unit tests for the provenance metadata
(tenant_id / created_by_user_id / job_id) threaded through ingestion onto
:Document nodes.

Covers:
  * insert_document() always SETs the three properties (null when omitted).
  * insert_folder_document_hierarchy() SETs them on every folder-parent node.
  * insert_document_graph() reads them straight off the Document and forwards
    them to both of the above.
  * loader._apply_metadata_overrides() override-only-when-provided semantics.
"""

from __future__ import annotations

import pytest

from scinr.newton.ingest import nodes
from scinr.newton.ingest.loader import _apply_metadata_overrides
from scinr.newton.models.document_structure import Document


class _FakeTx:
    """Records every tx.run(query, **params) call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params) -> None:
        self.calls.append((query, params))

    def params_for(self, needle: str) -> dict:
        """Return the params of the first recorded query containing *needle*."""
        for query, params in self.calls:
            if needle in query:
                return params
        raise AssertionError(f"no recorded query contained {needle!r}")


# ---------------------------------------------------------------------------
# insert_document
# ---------------------------------------------------------------------------


class TestInsertDocumentMetadata:
    def test_values_are_written_when_provided(self):
        tx = _FakeTx()
        nodes.insert_document(
            tx,
            "doc",
            "folder/doc",
            3,
            "raw123",
            tenant_id="tenant-1",
            created_by_user_id="user-9",
            job_id="job-abc",
        )
        params = tx.params_for("MERGE (d:Document")
        assert params["tenant_id"] == "tenant-1"
        assert params["created_by_user_id"] == "user-9"
        assert params["job_id"] == "job-abc"
        # tenant_id is part of the MERGE key, not a SET property
        assert "MERGE (d:Document {tenant_id: $tenant_id, path: $path, version: $version})" in (
            tx.calls[0][0]
        )
        assert "d.created_by_user_id" in tx.calls[0][0]
        assert "d.job_id" in tx.calls[0][0]

    def test_tenant_is_required(self):
        with pytest.raises(TypeError):
            nodes.insert_document(_FakeTx(), "doc", "doc", 1)  # type: ignore[call-arg]

    def test_values_default_to_none_and_are_still_bound(self):
        tx = _FakeTx()
        nodes.insert_document(tx, "doc", "doc", 1, tenant_id="__public__")
        params = tx.params_for("MERGE (d:Document")
        assert params["tenant_id"] == "__public__"
        assert params["created_by_user_id"] is None
        assert params["job_id"] is None


# ---------------------------------------------------------------------------
# insert_folder_document_hierarchy
# ---------------------------------------------------------------------------


class TestInsertFolderHierarchyMetadata:
    def test_every_folder_node_gets_the_metadata(self):
        tx = _FakeTx()
        nodes.insert_folder_document_hierarchy(
            tx,
            "ModuloA/SubModulo",
            2,
            tenant_id="t1",
            created_by_user_id="u1",
            job_id="j1",
        )
        merge_calls = [
            params for query, params in tx.calls if "MERGE (f:Document" in query
        ]
        assert len(merge_calls) == 2  # ModuloA + ModuloA/SubModulo
        for params in merge_calls:
            assert params["tenant_id"] == "t1"
            assert params["created_by_user_id"] == "u1"
            assert params["job_id"] == "j1"


# ---------------------------------------------------------------------------
# insert_document_graph
# ---------------------------------------------------------------------------


def _doc(**meta) -> Document:
    return Document(
        document_name="doc_a",
        document_type="",
        document_structure=[],
        doc_path="ModuloA/doc_a",
        raw_file_id="raw-1",
        **meta,
    )


class TestInsertDocumentGraphForwardsMetadata:
    def test_metadata_reaches_leaf_and_folder_nodes(self):
        tx = _FakeTx()
        doc = _doc(tenant_id="tt", created_by_user_id="uu", job_id="jj")

        nodes.insert_document_graph(tx, doc, resolved_version=5)

        leaf = tx.params_for("MERGE (d:Document")
        assert (leaf["tenant_id"], leaf["created_by_user_id"], leaf["job_id"]) == (
            "tt",
            "uu",
            "jj",
        )
        folder = tx.params_for("MERGE (f:Document")
        assert (folder["tenant_id"], folder["created_by_user_id"], folder["job_id"]) == (
            "tt",
            "uu",
            "jj",
        )

    def test_absent_metadata_is_forwarded_as_none(self):
        """No tenant → the reserved public key (never null, see utils/tenancy.py);
        the other metadata is forwarded as None."""
        tx = _FakeTx()
        doc = _doc()

        nodes.insert_document_graph(tx, doc, resolved_version=1)

        leaf = tx.params_for("MERGE (d:Document")
        assert leaf["tenant_id"] == "__public__"
        assert doc.tenant_id is None  # the Document itself is not modified
        assert leaf["created_by_user_id"] is None
        assert leaf["job_id"] is None


# ---------------------------------------------------------------------------
# _apply_metadata_overrides
# ---------------------------------------------------------------------------


class TestApplyMetadataOverrides:
    def test_none_arguments_leave_existing_values_untouched(self):
        doc = _doc(tenant_id="orig-t", created_by_user_id="orig-u", job_id="orig-j")
        _apply_metadata_overrides(doc, None, None, None)
        assert doc.tenant_id == "orig-t"
        assert doc.created_by_user_id == "orig-u"
        assert doc.job_id == "orig-j"

    def test_provided_arguments_override(self):
        doc = _doc(tenant_id="orig-t", job_id="orig-j")
        _apply_metadata_overrides(doc, "new-t", "new-u", None)
        assert doc.tenant_id == "new-t"
        assert doc.created_by_user_id == "new-u"
        assert doc.job_id == "orig-j"  # untouched (arg was None)


# ---------------------------------------------------------------------------
# Document model serialization
# ---------------------------------------------------------------------------


def test_metadata_round_trips_through_document_json():
    doc = _doc(tenant_id="t", created_by_user_id="u", job_id="j")
    reloaded = Document.model_validate_json(doc.model_dump_json())
    assert reloaded.tenant_id == "t"
    assert reloaded.created_by_user_id == "u"
    assert reloaded.job_id == "j"


def test_metadata_defaults_to_none_on_document():
    doc = _doc()
    assert doc.tenant_id is None
    assert doc.created_by_user_id is None
    assert doc.job_id is None


# ---------------------------------------------------------------------------
# file_* properties (0.5.0, FileMetadata)
# ---------------------------------------------------------------------------


class TestFileMetadataProperties:
    def test_every_key_is_in_the_set_and_bound(self):
        from scinr.newton.models.file_metadata import FILE_METADATA_KEYS, FileMetadata

        tx = _FakeTx()
        nodes.insert_document(
            tx, "clip.mp4", "clip.mp4", 1, tenant_id="acme",
            file_metadata=FileMetadata(duration_ms=60_000, video_codec="h264"),
        )
        query, params = tx.calls[0]
        for key in FILE_METADATA_KEYS:
            assert f"d.file_{key} = $file_{key}" in query
            assert f"file_{key}" in params
        assert params["file_duration_ms"] == 60_000
        assert params["file_video_codec"] == "h264"
        assert params["file_width"] is None  # null erases a stale value

    def test_without_metadata_every_key_is_null(self):
        from scinr.newton.models.file_metadata import FILE_METADATA_KEYS

        tx = _FakeTx()
        nodes.insert_document(tx, "a.pdf", "a.pdf", 1, tenant_id="acme")
        _, params = tx.calls[0]
        assert all(params[f"file_{key}"] is None for key in FILE_METADATA_KEYS)

    def test_graph_forwards_the_document_metadata(self):
        from scinr.newton.models.file_metadata import FileMetadata

        tx = _FakeTx()
        doc = _doc(file_metadata=FileMetadata(width=10, height=20))
        nodes.insert_document_graph(tx, doc, resolved_version=1)
        leaf = tx.params_for("MERGE (d:Document")
        assert (leaf["file_width"], leaf["file_height"]) == (10, 20)
        folder_query = next(q for q, _ in tx.calls if "MERGE (f:Document" in q)
        assert "file_" not in folder_query  # folders carry no file metadata

    def test_old_extract_json_without_the_field_validates(self):
        old = _doc().model_dump(mode="json")
        old.pop("file_metadata")
        assert Document.model_validate(old).file_metadata is None


async def test_file_metadata_travels_from_intermediate_json_to_document(tmp_path, monkeypatch):
    """End to end on disk: intermediate JSON → extract_one_file → Document → SET."""
    from types import SimpleNamespace

    from scinr.newton.converters.base import IntermediateDocument, IntermediatePage
    from scinr.newton.models.file_metadata import FileMetadata
    from scinr.newton.stages import extraction

    async def _no_llm(document, *args, **kwargs):
        return document

    monkeypatch.setattr(extraction, "_run_chunk_extraction", _no_llm)
    monkeypatch.setattr(extraction, "get_llm", lambda: None)
    monkeypatch.setattr(extraction, "get_config", lambda: SimpleNamespace(extraction_batch_size=1))
    intermediate = IntermediateDocument(
        pages=[IntermediatePage(index=0, markdown="x", start_ms=0, end_ms=1000)],
        document_name="clip.mp3",
        file_metadata=FileMetadata(duration_ms=1000, audio_codec="mp3"),
    )
    path = tmp_path / "clip.mp3.json"
    path.write_text(intermediate.to_json(), encoding="utf-8")

    from_disk = await extraction.extract_one_file(path, None, str(tmp_path))
    in_memory = await extraction.extract_one_intermediate(intermediate, None)

    assert from_disk.file_metadata == in_memory.file_metadata == intermediate.file_metadata
    tx = _FakeTx()
    nodes.insert_document_graph(tx, from_disk, resolved_version=1)
    assert tx.params_for("MERGE (d:Document")["file_audio_codec"] == "mp3"
