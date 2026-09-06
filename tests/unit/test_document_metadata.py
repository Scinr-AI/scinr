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
        assert "d.tenant_id" in tx.calls[0][0]
        assert "d.created_by_user_id" in tx.calls[0][0]
        assert "d.job_id" in tx.calls[0][0]

    def test_values_default_to_none_and_are_still_bound(self):
        tx = _FakeTx()
        nodes.insert_document(tx, "doc", "doc", 1)
        params = tx.params_for("MERGE (d:Document")
        assert params["tenant_id"] is None
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
        tx = _FakeTx()
        doc = _doc()

        nodes.insert_document_graph(tx, doc, resolved_version=1)

        leaf = tx.params_for("MERGE (d:Document")
        assert leaf["tenant_id"] is None
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
