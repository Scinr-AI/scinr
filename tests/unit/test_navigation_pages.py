"""Unit tests for scinr.newton.navigation.pages (source-text bridge)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import make_fake_llm  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.exceptions import StorageError  # noqa: E402
from scinr.newton.navigation.models import DocumentRef, StructureNodeRef  # noqa: E402
from scinr.newton.navigation.pages import (  # noqa: E402
    get_node_source_page_ids,
    get_node_source_text,
)
from scinr.newton.storage.models import ConvertedPageRecord  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}


class _Nav:
    def __init__(self, node, doc) -> None:
        self._node = node
        self._doc = doc

    async def get_structure_node(self, node_id: str):
        return self._node

    async def get_document_of_node(self, node_id: str):
        return self._doc


def _page(pid: str, idx: int) -> ConvertedPageRecord:
    return ConvertedPageRecord(
        id=pid, raw_file_id="rf", filename="f", folder_path=None, page_index=idx,
        markdown=f"# page {idx}", converted_at=datetime.now(UTC),
    )


async def test_source_page_ids_needs_no_storage() -> None:
    node = StructureNodeRef(id="i", node_id="n", role="table", source_page_ids=["p1", "p2"])
    ids = await get_node_source_page_ids(_Nav(node, None), "i")
    assert ids == ["p1", "p2"]


async def test_get_node_source_text_filters_to_node_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    node = StructureNodeRef(id="i", node_id="n", role="table", source_page_ids=["p2"])
    doc = DocumentRef(path="d", name="d", version=1, latest=True, is_folder=False, raw_file_id="rf")
    repo = AsyncMock()
    repo.get_pages = AsyncMock(return_value=[_page("p1", 0), _page("p2", 1)])
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (AsyncMock(), repo))
    out = await get_node_source_text(_Nav(node, doc), "i")
    assert [p.page_id for p in out] == ["p2"]
    assert out[0].markdown == "# page 1"


async def test_get_node_source_text_raises_without_storage() -> None:
    configure(storage_backend="none", **_NEO)
    node = StructureNodeRef(id="i", node_id="n", role="table", source_page_ids=["p1"])
    doc = DocumentRef(path="d", name="d", version=1, latest=True, is_folder=False, raw_file_id="rf")
    with pytest.raises(StorageError):
        await get_node_source_text(_Nav(node, doc), "i")


async def test_get_node_source_text_empty_when_no_page_ids() -> None:
    configure(storage_backend="none", **_NEO)
    node = StructureNodeRef(id="i", node_id="n", role="table", source_page_ids=[])
    out = await get_node_source_text(_Nav(node, None), "i")
    assert out == []
