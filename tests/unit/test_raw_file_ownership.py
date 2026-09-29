"""
tests/unit/test_raw_file_ownership.py — A Document's raw_file_id must belong to its tenant.

Covers plans/multitenancy-document-storage-plan.md WP5 (D4 / D7): before a
Document with a non-empty ``raw_file_id`` reaches Neo4j, the stored raw file
must exist and carry the Document's (effective, stored) tenant. No real Neo4j
or MongoDB: the storage factory and the synchronous graph loaders are patched.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scinr.newton.config import configure
from scinr.newton.exceptions import IngestionError
from scinr.newton.ingest import loader
from scinr.newton.ingest.raw_file_check import verify_raw_file_owner
from scinr.newton.models.document_structure import Document
from scinr.newton.storage.models import RawFileRecord

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw"}


class _RawRepo:
    """Fake RawFileRepository.get honouring a single stored tenant filter."""

    def __init__(self, owners: dict[str, tuple[str, str | None]]) -> None:
        # raw_file_id -> (stored tenant, job_id of the upload)
        self.owners = owners
        self.calls: list[tuple[str, dict]] = []

    async def get(self, raw_file_id: str, **scope):
        self.calls.append((raw_file_id, scope))
        owner = self.owners.get(raw_file_id)
        if owner is None:
            return None
        tenant, job = owner
        if scope.get("tenant_id") is not None and scope["tenant_id"] != tenant:
            return None
        return RawFileRecord(
            id=raw_file_id, filename="f.pdf", folder_path=None, content_type="application/pdf",
            size_bytes=1, checksum_sha256="x", stored_at=datetime.now(UTC),
            tenant_id=tenant, job_id=job,
        )


@pytest.fixture
def storage(monkeypatch):
    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    repo = _RawRepo({"rf-acme": ("acme", "upload-job"), "rf-public": ("__public__", None)})
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (repo, MagicMock()))
    return repo


def _doc(raw_file_id: str, tenant_id: str | None, *, job_id: str | None = None) -> Document:
    return Document(
        document_name="doc",
        document_type="pdf",
        doc_path="F/doc",
        raw_file_id=raw_file_id,
        tenant_id=tenant_id,
        job_id=job_id,
        document_structure=[],
    )


class TestVerifyRawFileOwner:
    async def test_empty_raw_file_id_needs_no_storage(self):
        configure(storage_backend="none", **_NEO)
        await verify_raw_file_owner("", "acme", document="d")
        await verify_raw_file_owner(None, None, document="d")

    async def test_storage_none_with_raw_file_id_is_an_error(self):
        configure(storage_backend="none", **_NEO)
        with pytest.raises(IngestionError, match="storage_backend='none'"):
            await verify_raw_file_owner("rf-acme", "acme", document="d")

    async def test_same_tenant_is_accepted(self, storage):
        await verify_raw_file_owner("rf-acme", "acme", document="d")
        assert storage.calls == [("rf-acme", {"tenant_id": "acme"})]

    async def test_other_tenant_is_rejected(self, storage):
        with pytest.raises(IngestionError, match="does not belong"):
            await verify_raw_file_owner("rf-acme", "globex", document="d")

    async def test_public_document_passes_the_stored_key_not_none(self, storage):
        """A public document must only accept a public file: the lookup uses
        '__public__', never None (which would mean 'every tenant')."""
        with pytest.raises(IngestionError):
            await verify_raw_file_owner("rf-acme", None, document="d")
        assert storage.calls == [("rf-acme", {"tenant_id": "__public__"})]
        await verify_raw_file_owner("rf-public", None, document="d")

    async def test_a_tenant_cannot_claim_a_public_file(self, storage):
        with pytest.raises(IngestionError):
            await verify_raw_file_owner("rf-public", "acme", document="d")

    async def test_unknown_raw_file_id_is_rejected(self, storage):
        with pytest.raises(IngestionError):
            await verify_raw_file_owner("rf-missing", "acme", document="d")


class TestIngestOne:
    async def test_rejected_document_never_reaches_neo4j(self, storage, monkeypatch):
        load = MagicMock(return_value="doc")
        monkeypatch.setattr(loader, "_load_document_object", load)
        with pytest.raises(IngestionError):
            await loader.ingest_one(_doc("rf-acme", None), driver=object(), tenant_id="globex")
        load.assert_not_called()

    async def test_override_tenant_is_the_one_checked(self, storage, monkeypatch):
        """The run's tenant wins over the one baked into the Document."""
        load = MagicMock(return_value="doc")
        monkeypatch.setattr(loader, "_load_document_object", load)
        await loader.ingest_one(_doc("rf-acme", "globex"), driver=object(), tenant_id="acme")
        load.assert_called_once()

    async def test_same_tenant_other_job_is_ingested(self, storage, monkeypatch):
        load = MagicMock(return_value="doc")
        monkeypatch.setattr(loader, "_load_document_object", load)
        await loader.ingest_one(
            _doc("rf-acme", "acme"), driver=object(), job_id="ingest-job"
        )
        load.assert_called_once()

    async def test_forged_ingestion_json_fails_and_writes_nothing(
        self, storage, monkeypatch, tmp_path: Path
    ):
        """A globex extract-*.json carrying acme's raw_file_id is rejected."""
        path = tmp_path / "extract-doc.json"
        path.write_text(_doc("rf-acme", "globex").model_dump_json(), encoding="utf-8")
        load = MagicMock(return_value="doc")
        monkeypatch.setattr(loader, "load_file", load)
        with pytest.raises(IngestionError):
            await loader.ingest_one_from_path(path, driver=object())
        with pytest.raises(IngestionError):
            await loader.ingest_one_from_path(path, driver=object(), tenant_id="globex")
        load.assert_not_called()

    async def test_json_without_raw_file_id_skips_the_check(self, monkeypatch, tmp_path: Path):
        configure(storage_backend="none", **_NEO)
        path = tmp_path / "extract-doc.json"
        path.write_text(json.dumps({"raw_file_id": "", "tenant_id": "acme"}), encoding="utf-8")
        load = MagicMock(return_value="doc")
        monkeypatch.setattr(loader, "load_file", load)
        await loader.ingest_one_from_path(path, driver=object())
        load.assert_called_once()


class TestRunIngestion:
    async def test_rejects_only_the_forged_document(self, storage, monkeypatch):
        from scinr.newton.stages import ingestion

        driver = MagicMock()
        monkeypatch.setattr(ingestion, "get_driver", lambda: driver)
        monkeypatch.setattr(ingestion, "setup_schema", lambda d: None)
        seen: list[list[Document]] = []

        def _load_documents(docs, *a, **k):
            seen.append(list(docs))
            return [d.document_name for d in docs]

        monkeypatch.setattr(ingestion, "load_documents", _load_documents)

        good = _doc("rf-acme", None)
        good.document_name = "good"
        bad = _doc("rf-public", None)
        bad.document_name = "bad"
        result = await ingestion.run_ingestion(documents=[good, bad], tenant_id="acme")

        assert [d.document_name for d in seen[0]] == ["good"]
        by_name = {r.document_name: r for r in result.documents}
        assert by_name["good"].nodes_processed == 1
        assert by_name["bad"].nodes_failed == 1
        assert "does not belong" in by_name["bad"].errors[0]
        assert result.success is False

    async def test_folder_mode_filters_forged_files(self, storage, monkeypatch, tmp_path: Path):
        from scinr.newton.stages import ingestion

        monkeypatch.setattr(ingestion, "get_driver", lambda: MagicMock())
        monkeypatch.setattr(ingestion, "setup_schema", lambda d: None)
        loaded: list[list[Path]] = []

        def _load_files(files, *a, **k):
            loaded.append(list(files))
            return [f.stem.removeprefix("extract-") for f in files]

        monkeypatch.setattr(ingestion, "load_files", _load_files)
        (tmp_path / "extract-ok.json").write_text(
            json.dumps({"raw_file_id": "rf-acme", "tenant_id": "acme"}), encoding="utf-8"
        )
        (tmp_path / "extract-forged.json").write_text(
            json.dumps({"raw_file_id": "rf-acme", "tenant_id": "globex"}), encoding="utf-8"
        )

        result = await ingestion.run_ingestion(output_folder=str(tmp_path))

        assert [p.name for p in loaded[0]] == ["extract-ok.json"]
        by_name = {r.document_name: r for r in result.documents}
        assert by_name["ok"].nodes_processed == 1
        assert by_name["forged"].nodes_failed == 1
