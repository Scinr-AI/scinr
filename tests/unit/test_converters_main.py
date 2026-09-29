"""
tests/unit/test_converters_main.py — Unit tests for scinr.newton.converters.main.

Currently focused on the `parallel_docs` guard in `convert_folder()`: passing
`parallel_docs=0` (or negative) must raise `ValueError` immediately, rather
than creating an unacquirable `asyncio.Semaphore(0)` and hanging forever.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scinr.newton.converters.main import convert_folder

# Short guard timeout: if the `parallel_docs < 1` validation is ever removed
# or bypassed, convert_folder() would hang indefinitely on an unacquirable
# asyncio.Semaphore(0)/negative-sized semaphore. Wrapping the call in
# asyncio.wait_for() ensures this test fails fast with a clear TimeoutError
# instead of hanging the whole test suite.
_HANG_GUARD_TIMEOUT_SECONDS = 1.0


class TestConvertFolderParallelDocsGuard:
    @pytest.mark.parametrize("bad_value", [0, -1, -5])
    async def test_parallel_docs_below_one_raises_value_error(self, tmp_path: Path, bad_value: int):
        """`parallel_docs < 1` raises ValueError immediately (no hang).

        A source file is created so `convert_folder` actually schedules an
        entry and awaits the semaphore — with the guard removed,
        `parallel_docs=0` would hang forever on `asyncio.Semaphore(0)`
        (an empty *tmp_path* would never exercise the semaphore at all,
        masking the bug), and negative values raise asyncio's own generic
        ``ValueError`` instead of the clear, immediate one this guard adds.
        """
        (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="parallel_docs must be >= 1"):
            await asyncio.wait_for(
                convert_folder(tmp_path, tmp_path, parallel_docs=bad_value),
                timeout=_HANG_GUARD_TIMEOUT_SECONDS,
            )

    async def test_parallel_docs_one_does_not_raise(self, tmp_path: Path):
        """Sanity check: the default/valid `parallel_docs=1` still works."""
        result = await asyncio.wait_for(
            convert_folder(tmp_path, tmp_path, parallel_docs=1),
            timeout=_HANG_GUARD_TIMEOUT_SECONDS,
        )
        written, failures = result
        assert written == []
        assert failures == []


class TestConvertFolderUnsupportedFormat:
    async def test_unsupported_extension_is_skipped_silently(self, tmp_path: Path):
        """An unsupported-extension file must not appear in `failures`.

        `convert_one` catches `UnsupportedFormatError` internally and returns
        `([], [])` for that entry (a deliberate "skip silently" design — see
        its docstring), rather than surfacing it as a failure. Mixed with a
        valid file in the same directory, the valid file must still be
        converted (present in `written`) while the unsupported one is simply
        absent from both `written` and `failures`.
        """
        (tmp_path / "doc.txt").write_text("hello world", encoding="utf-8")
        (tmp_path / "unsupported.xyz123").write_text("ignored content", encoding="utf-8")

        written, failures = await asyncio.wait_for(
            convert_folder(tmp_path, tmp_path, parallel_docs=1),
            timeout=_HANG_GUARD_TIMEOUT_SECONDS,
        )

        assert len(written) == 1
        raw_source, _json_written, _doc = written[0]
        assert raw_source.name == "doc.txt"
        assert failures == []


class TestConvertOneStreamsRawFile:
    async def test_convert_one_uses_store_file_and_never_reads_bytes(
        self, tmp_path: Path, monkeypatch
    ):
        """The raw file goes to the repo via ``store_file(path=...)`` — the
        converter layer must never materialise it with ``Path.read_bytes``."""
        from scinr.newton.converters.main import convert_one

        src = tmp_path / "in"
        src.mkdir()
        doc = src / "doc.txt"
        doc.write_text("hello", encoding="utf-8")

        calls: list[dict] = []

        class _Repo:
            async def store_file(self, path, filename, content_type, folder_path, **owner):
                calls.append({"path": path, "filename": filename, **owner})
                return "raw-1"

            async def store(self, *a, **k):  # pragma: no cover - must not be used
                raise AssertionError("store() must not be called from convert_one")

        def _boom(self):
            raise AssertionError("Path.read_bytes must not be called")

        monkeypatch.setattr(Path, "read_bytes", _boom)

        written, failures = await convert_one(doc, tmp_path / "out", raw_file_repo=_Repo())

        assert failures == []
        assert calls == [
            {
                "path": doc,
                "filename": "doc.txt",
                "tenant_id": None,
                "created_by_user_id": None,
                "job_id": None,
            }
        ]
        assert written[0][2].raw_file_id == "raw-1"

    async def test_null_repo_never_opens_the_file(self, tmp_path: Path, monkeypatch):
        from scinr.newton.converters.main import convert_one
        from scinr.newton.storage.null import NullRawFileRepository

        doc = tmp_path / "doc.txt"
        doc.write_text("hello", encoding="utf-8")

        def _boom(self):
            raise AssertionError("Path.read_bytes must not be called")

        monkeypatch.setattr(Path, "read_bytes", _boom)

        written, failures = await convert_one(
            doc, tmp_path / "out", raw_file_repo=NullRawFileRepository()
        )

        assert failures == []
        assert written[0][2].raw_file_id == ""


class TestConvertOneInMemoryOnly:
    async def test_output_dir_none_writes_no_files_and_returns_none_path(
        self, tmp_path: Path, monkeypatch
    ):
        from scinr.newton.converters.main import convert_one

        doc = tmp_path / "doc.txt"
        doc.write_text("hello", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        before = set(tmp_path.rglob("*"))

        written, failures = await convert_one(doc, None)

        assert failures == []
        assert len(written) == 1
        raw_source, json_written, intermediate = written[0]
        assert raw_source == doc
        assert json_written is None
        assert intermediate.pages  # the converted document is returned in memory
        assert set(tmp_path.rglob("*")) == before  # nothing was created on disk

    async def test_output_dir_none_never_serialises_the_document(self, tmp_path: Path, monkeypatch):
        from scinr.newton.converters import base as converters_base
        from scinr.newton.converters.main import convert_one

        doc = tmp_path / "doc.txt"
        doc.write_text("hello", encoding="utf-8")

        def _boom(self, *a, **k):
            raise AssertionError("to_json must not be called when output_dir is None")

        monkeypatch.setattr(converters_base.IntermediateDocument, "to_json", _boom)

        written, failures = await convert_one(doc, None)

        assert failures == []
        assert len(written) == 1

    async def test_output_dir_none_is_rejected_for_a_directory_entry(self, tmp_path: Path):
        from scinr.newton.converters.main import convert_one

        with pytest.raises(ValueError, match="output_dir is required"):
            await convert_one(tmp_path, None)


class TestConvertStampsTheOwner:
    """convert_one / convert_folder write the tenant + provenance on the stored
    raw file and pages, and stamp them on the IntermediateDocument."""

    class _Raw:
        def __init__(self) -> None:
            self.owners: list[dict] = []

        async def store_file(self, path, filename, content_type, folder_path, **owner):
            self.owners.append(owner)
            return f"raw-{len(self.owners)}"

    class _Pages:
        def __init__(self) -> None:
            self.owners: list[dict] = []

        async def store_page(self, raw_file_id, filename, folder_path, page_index, markdown, **owner):
            self.owners.append(owner)
            return f"page-{len(self.owners)}"

    _OWNER = {"tenant_id": "acme", "created_by_user_id": "u1", "job_id": "j1"}

    async def test_convert_one(self, tmp_path: Path):
        from scinr.newton.converters.main import convert_one

        src = tmp_path / "doc.txt"
        src.write_text("hello", encoding="utf-8")
        raw, pages = self._Raw(), self._Pages()

        written, failures = await convert_one(
            src, None, raw_file_repo=raw, page_repo=pages, **self._OWNER
        )

        assert failures == []
        doc = written[0][2]
        assert (doc.tenant_id, doc.created_by_user_id, doc.job_id) == ("acme", "u1", "j1")
        assert raw.owners == [self._OWNER]
        assert pages.owners and all(o == self._OWNER for o in pages.owners)

    async def test_convert_folder_forwards_the_owner_through_recursion(self, tmp_path: Path):
        from scinr.newton.converters.main import convert_folder

        src = tmp_path / "in"
        (src / "sub" / "deeper").mkdir(parents=True)
        (src / "a.txt").write_text("a", encoding="utf-8")
        (src / "sub" / "b.txt").write_text("b", encoding="utf-8")
        (src / "sub" / "deeper" / "c.txt").write_text("c", encoding="utf-8")
        raw, pages = self._Raw(), self._Pages()

        written, failures = await convert_folder(
            src, tmp_path / "out", raw_file_repo=raw, page_repo=pages, **self._OWNER
        )

        assert failures == []
        assert len(written) == 3
        assert all(w[2].tenant_id == "acme" and w[2].job_id == "j1" for w in written)
        assert raw.owners == [self._OWNER] * 3
        # The owner is serialized into the intermediate JSON.
        import json

        for _src, json_path, _doc in written:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert (data["tenant_id"], data["created_by_user_id"], data["job_id"]) == (
                "acme", "u1", "j1",
            )

    async def test_default_is_public(self, tmp_path: Path):
        from scinr.newton.converters.main import convert_one

        src = tmp_path / "doc.txt"
        src.write_text("hello", encoding="utf-8")
        raw = self._Raw()
        written, _ = await convert_one(src, None, raw_file_repo=raw)
        assert written[0][2].tenant_id is None
        assert raw.owners == [{"tenant_id": None, "created_by_user_id": None, "job_id": None}]
