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
from scinr.newton.navigation.pages import get_structure_nodes_source_pages  # noqa: E402
from scinr.newton.storage.models import ConvertedPageRecord  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}


class _Nav:
    """Graph side for the batch: resolves the given nodes by id, in request order."""

    def __init__(self, *nodes: StructureNodeRef) -> None:
        self._nodes = {n.id: n for n in nodes}
        self.calls: list[tuple[list[str], dict]] = []

    async def get_structure_nodes_by_ids(self, node_ids, **scope):
        self.calls.append((list(node_ids), scope))
        return [self._nodes[i] for i in node_ids if i in self._nodes]

    async def get_document_of_node(self, node_id: str, **scope):
        raise AssertionError("node source text must not resolve the document")


def _node(
    page_ids: list[str], tenant_id: str | None = "acme", node_id: str = "i"
) -> StructureNodeRef:
    raw = {"tenant_id": tenant_id} if tenant_id is not None else {}
    return StructureNodeRef(
        raw=raw, id=node_id, node_id=node_id, role="table", source_page_ids=page_ids
    )


def _page(pid: str, idx: int) -> ConvertedPageRecord:
    return ConvertedPageRecord(
        id=pid, raw_file_id="rf", filename="f", folder_path="a/b", page_index=idx,
        markdown=f"# page {idx}", converted_at=datetime.now(UTC),
    )


def _tenant_storage(monkeypatch, pages: dict[str, tuple[str | None, int]]) -> AsyncMock:
    """Storage whose ``get_pages_by_ids`` filters *pages* (id -> (tenant, index))
    by the requested tenant, like the real repository (``None`` = unfiltered)."""
    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)

    async def _by_ids(ids, *, tenant_id=None):
        return [
            _page(p, pages[p][1])
            for p in ids
            if p in pages and (tenant_id is None or pages[p][0] == tenant_id)
        ]

    repo = AsyncMock()
    repo.get_pages_by_ids = AsyncMock(side_effect=_by_ids)
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (AsyncMock(), repo))
    return repo


async def test_batch_reads_only_the_node_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """The node's pages are read by id, ordered by page index; the document and
    its other pages are never loaded (_Nav.get_document_of_node fails the test)."""
    repo = _tenant_storage(monkeypatch, {"p2": ("acme", 1), "p3": ("acme", 2)})
    out = await get_structure_nodes_source_pages(_Nav(_node(["p3", "p2"])), ["i"])
    assert [(e.structure_node_id, e.page_ids) for e in out.nodes] == [("i", ["p2", "p3"])]
    assert out.pages["p2"].markdown == "# page 1"
    repo.get_pages_by_ids.assert_awaited_once_with(["p3", "p2"], tenant_id="acme")
    repo.get_pages.assert_not_called()


async def test_page_text_carries_the_file_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _tenant_storage(monkeypatch, {"p1": ("acme", 0)})
    out = await get_structure_nodes_source_pages(_Nav(_node(["p1"])), ["i"])
    page = out.pages["p1"]
    assert (page.page_id, page.index, page.raw_file_id, page.filename, page.folder_path) == (
        "p1", 0, "rf", "f", "a/b",
    )


async def test_shared_page_is_read_and_returned_once(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _tenant_storage(monkeypatch, {"p1": ("acme", 0), "p2": ("acme", 1)})
    nav = _Nav(_node(["p1", "p2"], node_id="a"), _node(["p2"], node_id="b"))
    out = await get_structure_nodes_source_pages(nav, ["a", "b"])
    assert [(e.structure_node_id, e.page_ids) for e in out.nodes] == [
        ("a", ["p1", "p2"]), ("b", ["p2"]),
    ]
    assert list(out.pages) == ["p1", "p2"]
    repo.get_pages_by_ids.assert_awaited_once_with(["p1", "p2"], tenant_id="acme")


async def test_status_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing / out-of-scope nodes, nodes without pages and missing pages are
    reported, not raised; duplicate ids are looked up once, in request order."""
    _tenant_storage(monkeypatch, {"p1": ("acme", 0)})
    nav = _Nav(_node(["p1", "gone"], node_id="a"), _node([], node_id="empty"))
    out = await get_structure_nodes_source_pages(nav, ["missing", "a", "empty", "a"])
    assert nav.calls[0][0] == ["missing", "a", "empty"]
    assert out.not_found_structure_nodes == ["missing"]
    assert out.structure_nodes_without_pages == ["empty"]
    assert [(e.structure_node_id, e.page_ids, e.not_found_page_ids) for e in out.nodes] == [
        ("a", ["p1"], ["gone"]),
    ]


async def test_include_public_reads_once_per_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    """One storage read per stored tenant of the nodes; a page of another
    tenant requested by a node counts as not found for it."""
    repo = _tenant_storage(monkeypatch, {"a1": ("acme", 0), "pub": ("__public__", 0)})
    nav = _Nav(
        _node(["a1", "pub"], "acme", node_id="mine"),
        _node(["pub"], "__public__", node_id="public"),
    )
    out = await get_structure_nodes_source_pages(
        nav, ["mine", "public"], tenant_id="acme", include_public=True
    )
    assert {c.kwargs["tenant_id"]: c.args[0] for c in repo.get_pages_by_ids.await_args_list} == {
        "acme": ["a1", "pub"], "__public__": ["pub"],
    }
    by_node = {e.structure_node_id: e for e in out.nodes}
    assert (by_node["mine"].page_ids, by_node["mine"].not_found_page_ids) == (["a1"], ["pub"])
    assert by_node["public"].page_ids == ["pub"]
    assert set(out.pages) == {"a1", "pub"}


@pytest.mark.parametrize("tenant", ["acme", "__public__", None])
async def test_node_pages_are_read_with_the_nodes_tenant(
    monkeypatch: pytest.MonkeyPatch, tenant
) -> None:
    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    repo = AsyncMock()
    repo.get_pages_by_ids = AsyncMock(return_value=[_page("p1", 0)])
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (AsyncMock(), repo))
    # The call's scope does not matter: the node's stored tenant is used.
    await get_structure_nodes_source_pages(_Nav(_node(["p1"], tenant)), ["i"], tenant_id="acme")
    # Stored tenant as is: a public node only reads public pages; a legacy
    # node (no tenant) reads unfiltered.
    repo.get_pages_by_ids.assert_awaited_once_with(["p1"], tenant_id=tenant)


async def test_batch_raises_without_storage_only_when_there_are_pages() -> None:
    configure(storage_backend="none", **_NEO)
    with pytest.raises(StorageError):
        await get_structure_nodes_source_pages(_Nav(_node(["p1"])), ["i"])
    # Nothing to read: no storage needed.
    out = await get_structure_nodes_source_pages(_Nav(_node([])), ["i", "x"])
    assert out.nodes == [] and out.pages == {}
    assert out.structure_nodes_without_pages == ["i"]
    assert out.not_found_structure_nodes == ["x"]
    empty = await get_structure_nodes_source_pages(_Nav(), [])
    assert empty.nodes == [] and empty.not_found_structure_nodes == []


# ---------------------------------------------------------------------------
# Multi-tenancy: the storage read is filtered by the node's / document's own tenant
# ---------------------------------------------------------------------------


def _doc_ref(tenant_id: str | None, raw_file_id: str = "rf") -> DocumentRef:
    return DocumentRef(
        path="d", name="d", version=1, latest=True, is_folder=False,
        raw_file_id=raw_file_id, tenant_id=tenant_id,
    )


_NO_SCOPE = {"tenant_id": None, "include_public": False, "created_by_user_id": None, "job_id": None}


class _DocNav:
    """Records the document lookups (path + scope kwargs)."""

    def __init__(self, doc) -> None:
        self._doc = doc
        self.calls: list[tuple] = []

    async def get_latest_version(self, path, **kwargs):
        self.calls.append(("latest", path, kwargs))
        return self._doc

    async def get_one_document(self, path, version, **kwargs):
        self.calls.append(("one", path, version, kwargs))
        return self._doc


async def test_document_source_text_by_path_and_by_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    from scinr.newton.navigation.pages import get_document_source_text

    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    repo = AsyncMock()
    repo.get_pages = AsyncMock(return_value=[_page("p2", 1), _page("p1", 0)])
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (AsyncMock(), repo))

    nav = _DocNav(_doc_ref("acme"))
    out = await get_document_source_text(nav, "d")
    assert [p.page_id for p in out] == ["p1", "p2"]
    assert nav.calls == [("latest", "d", _NO_SCOPE)]
    repo.get_pages.assert_awaited_with("rf", tenant_id="acme")

    # A DocumentRef selector carries its tenant into the graph lookup.
    await get_document_source_text(nav, _doc_ref("acme"), version=1)
    assert nav.calls[-1] == ("one", "d", 1, {**_NO_SCOPE, "tenant_id": "acme"})


class _RawRepo:
    def __init__(self, record) -> None:
        self.record = record
        self.calls: list[tuple] = []

    async def open_with_record(self, raw_file_id, **scope):
        self.calls.append(("open_with_record", raw_file_id, scope))
        if self.record is None:
            return None

        async def _chunks():
            yield b"ab"
            yield b"c"

        return self.record, _chunks()


async def test_get_document_original_streams_the_tenants_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scinr.newton.navigation import OriginalFile
    from scinr.newton.navigation.pages import get_document_original
    from scinr.newton.storage.models import RawFileRecord

    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    record = RawFileRecord(
        id="rf", filename="d.pdf", folder_path=None, content_type="application/pdf",
        size_bytes=3, checksum_sha256="x", stored_at=datetime.now(UTC), tenant_id="acme",
    )
    raw = _RawRepo(record)
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (raw, AsyncMock()))

    original = await get_document_original(_DocNav(_doc_ref("acme")), "d")

    assert isinstance(original, OriginalFile)
    assert original.filename == "d.pdf" and original.content_type == "application/pdf"
    assert await original.read() == b"abc"
    # a single lookup of the raw-file record, in the document's tenant
    assert raw.calls == [("open_with_record", "rf", {"tenant_id": "acme"})]


async def test_get_document_original_none_when_out_of_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw_file_id pointing at another tenant's upload resolves to nothing."""
    from scinr.newton.navigation.pages import get_document_original

    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    raw = _RawRepo(None)
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (raw, AsyncMock()))

    assert await get_document_original(_DocNav(_doc_ref("globex")), "d") is None
    assert [c[0] for c in raw.calls] == ["open_with_record"]
    assert await get_document_original(_DocNav(None), "d") is None
    assert await get_document_original(_DocNav(_doc_ref("acme", raw_file_id="")), "d") is None


async def test_get_document_original_needs_storage() -> None:
    from scinr.newton.navigation.pages import get_document_original

    configure(storage_backend="none", **_NEO)
    with pytest.raises(StorageError):
        await get_document_original(_DocNav(_doc_ref("acme")), "d")


# ---------------------------------------------------------------------------
# GraphNavigator methods (delegating to navigation.pages) and ScopedNavigator
# ---------------------------------------------------------------------------


class _RecordingNav:
    """Graph side of a navigator: records the scope of every lookup. The
    source-text methods are the real GraphNavigator ones (concrete delegations)."""

    from scinr.newton.navigation.base import GraphNavigator as _GN

    get_structure_nodes_source_pages = _GN.get_structure_nodes_source_pages
    get_info_unit_source_text = _GN.get_info_unit_source_text
    get_document_source_text = _GN.get_document_source_text
    get_document_original = _GN.get_document_original

    def __init__(self, node, doc) -> None:
        self._node, self._doc = node, doc
        self.calls: list[tuple[str, dict]] = []

    async def get_structure_nodes_by_ids(self, node_ids, **scope):
        self.calls.append(("nodes", scope))
        return [self._node] if self._node is not None and self._node.id in node_ids else []

    async def get_document_of_node(self, node_id, **scope):
        raise AssertionError("node source text must not resolve the document")

    async def get_node_for_info_unit(self, uid, **scope):
        self.calls.append(("node_for_iu", scope))
        return self._node

    async def get_latest_version(self, path, **scope):
        self.calls.append(("latest", scope))
        return self._doc


def _storage_with(monkeypatch, pages):
    configure(storage_backend="mongodb", mongodb_uri="mongodb://x", **_NEO)
    repo = AsyncMock()
    repo.get_pages = AsyncMock(return_value=pages)
    repo.get_pages_by_ids = AsyncMock(return_value=pages)
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (AsyncMock(), repo))
    return repo


async def test_navigator_method_forwards_the_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _storage_with(monkeypatch, [_page("p1", 0)])
    nav = _RecordingNav(_node(["p1"]), _doc_ref("acme"))

    out = await nav.get_structure_nodes_source_pages(["i"], tenant_id="acme", job_id="j1")

    assert out.nodes[0].page_ids == ["p1"]
    # a single graph lookup, with the full scope
    assert nav.calls == [(
        "nodes", {"tenant_id": "acme", "include_public": False, "created_by_user_id": None, "job_id": "j1"}
    )]
    repo.get_pages_by_ids.assert_awaited_once_with(["p1"], tenant_id="acme")


async def test_info_unit_source_text_reuses_the_owning_node(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _storage_with(monkeypatch, [_page("p1", 0)])
    nav = _RecordingNav(_node(["p1"]), None)

    out = await nav.get_info_unit_source_text("u", tenant_id="acme", created_by_user_id="u1")

    assert [p.page_id for p in out] == ["p1"]
    assert nav.calls == [(
        "node_for_iu",
        {"tenant_id": "acme", "include_public": False, "created_by_user_id": "u1", "job_id": None},
    )]
    repo.get_pages_by_ids.assert_awaited_once_with(["p1"], tenant_id="acme")


async def test_scoped_navigator_applies_its_scope_to_every_graph_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scinr.newton.exceptions import NavigationError
    from scinr.newton.navigation import ScopedNavigator

    _storage_with(monkeypatch, [_page("p1", 0)])
    node = StructureNodeRef(id="i", node_id="n", role="table", source_page_ids=["p1"])
    inner = _RecordingNav(node, _doc_ref("acme"))
    acme = ScopedNavigator(inner, tenant_id="acme", include_public=True)

    await acme.get_structure_nodes_source_pages(["i"])
    await acme.get_info_unit_source_text("u")
    await acme.get_document_source_text("d")

    for kind, scope in inner.calls:
        assert scope["tenant_id"] == "acme", kind
    assert ("latest", {"tenant_id": "acme", "include_public": True,
                       "created_by_user_id": None, "job_id": None}) in inner.calls
    with pytest.raises(NavigationError):
        await acme.get_structure_nodes_source_pages(["i"], tenant_id="globex")


async def test_document_ref_outside_the_scope_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from scinr.newton.exceptions import NavigationError

    _storage_with(monkeypatch, [])
    nav = _RecordingNav(None, _doc_ref("globex"))
    with pytest.raises(NavigationError):
        await nav.get_document_source_text(_doc_ref("globex"), tenant_id="acme")


async def test_describe_node_reads_source_text_from_the_resolved_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """describe_node(include_source_text=True) reuses its node: one lookup of
    the node, no document resolution, only the node's pages read."""
    from scinr.newton.navigation.neo4j._structure import _StructureMixin

    repo = _storage_with(monkeypatch, [_page("p1", 0)])
    lookups: list[dict] = []

    class _Nav(_StructureMixin):
        async def get_structure_node(self, node_id, **scope):
            lookups.append(scope)
            return _node(["p1"], "acme")

        async def get_document_of_node(self, node_id, **scope):
            raise AssertionError("describe_node must not resolve the document")

        async def get_node_ancestors(self, node_id, **scope):
            return []

        async def get_info_units(self, node_id, **scope):
            return []

        async def get_model_decision(self, node_id, **scope):
            return None

        async def get_extraction_result(self, node_id, **scope):
            return None

        async def _read_one(self, cypher, /, **params):
            return {"child_count": 0, "mi_count": 0}

    desc = await _Nav().describe_node("i", include_source_text=True, tenant_id="acme")

    assert [p.page_id for p in desc.source_text] == ["p1"]
    assert len(lookups) == 1
    repo.get_pages_by_ids.assert_awaited_once_with(["p1"], tenant_id="acme")
