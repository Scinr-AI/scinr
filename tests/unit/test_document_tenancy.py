"""
tests/unit/test_document_tenancy.py — The tenant is part of the document identity.

Covers plans/multitenancy-document-identity-plan.md §3.1 with fake drivers
that record the emitted Cypher (no real Neo4j):

  * ``tenant_key`` / ``tenant_from_key`` (None <-> "__public__").
  * ``insert_document_graph`` for two tenants and the same path: every
    ``:Document`` MATCH/MERGE carries the tenant, StructureNode ids differ.
  * Derived uids (InfoUnit, ModelDecision, ExtractionResult) differ per tenant.
  * Version resolution (``get_next_version``, batch) filters by tenant.
  * Annotation / extraction / resolver / tabular / ``replaces`` select by
    ``(tenant_id, path)``, never by name alone.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scinr.newton.ingest import loader, nodes
from scinr.newton.models.document_structure import Document, InfoUnit, StructureNode
from scinr.newton.utils.tenancy import PUBLIC_TENANT, tenant_key

# ---------------------------------------------------------------------------
# Recording fakes
# ---------------------------------------------------------------------------


class _Tx:
    """Sync tx/session: records (query, params); run() returns an empty result."""

    def __init__(self, single=None):
        self.calls: list[tuple[str, dict]] = []
        self._single = single

    def run(self, query, **params):
        self.calls.append((query, params))
        return SimpleNamespace(single=lambda: self._single, data=lambda: [])


class _AsyncResult:
    def __init__(self, single=None, data=None):
        self._single = single
        self._data = data or []

    async def single(self):
        return self._single

    async def data(self):
        return self._data


class _AsyncSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.driver.calls.append((query, params))
        return self.driver.respond(query, params)


class _AsyncDriver:
    """Async driver whose sessions record every query; *respond* builds the result."""

    def __init__(self, respond=None):
        self.calls: list[tuple[str, dict]] = []
        self.respond = respond or (lambda q, p: _AsyncResult())

    def session(self, **kwargs):
        return _AsyncSession(self)


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    cfg = SimpleNamespace(neo4j_database="neo4j")
    import scinr.newton.annotation.neo4j_ops as ann_ops
    import scinr.newton.entity_extraction.neo4j_ops as ee_ops
    import scinr.newton.stages.ingestion as ingestion_stage
    import scinr.newton.utils.document_resolver as resolver

    for mod in (ann_ops, ee_ops, ingestion_stage, resolver, loader):
        monkeypatch.setattr(mod, "get_config", lambda: cfg)


_DOC_PATTERN = re.compile(r"\(\w+:Document\s*\{([^}]*)\}\)")


def _document_patterns(calls) -> list[str]:
    """Property maps of every ``(x:Document {...})`` pattern in the recorded queries."""
    return [m for q, _ in calls for m in _DOC_PATTERN.findall(q)]


def _doc(tenant_id: str | None) -> Document:
    return Document(
        document_name="doc",
        document_type="pdf",
        doc_path="Folder/Sub/doc",
        raw_file_id="",
        tenant_id=tenant_id,
        document_structure=[
            StructureNode(
                node_id="1",
                title="Section 1",
                role="section",
                appearance_order=1,
                info_units=[InfoUnit(title="t", description="d")],
                children=[
                    StructureNode(
                        node_id="1_1",
                        title="Sub 1",
                        role="subsection",
                        appearance_order=2,
                        info_units=[InfoUnit(title="t2", description="d2")],
                    )
                ],
            )
        ],
    )


def _ingest(tenant_id: str | None, update_mode: bool = False) -> _Tx:
    tx = _Tx()
    nodes.insert_document_graph(tx, _doc(tenant_id), resolved_version=1, update_mode=update_mode)
    return tx


def _structure_ids(tx: _Tx) -> list[str]:
    return [p["id"] for q, p in tx.calls if "MERGE (n:StructureNode {id: $id})" in q]


def _info_unit_uids(tx: _Tx) -> list[str]:
    return [p["uid"] for q, p in tx.calls if "MERGE (u:InfoUnit {uid: $uid})" in q]


# ---------------------------------------------------------------------------
# tenant_key
# ---------------------------------------------------------------------------


class TestTenantKey:
    def test_none_and_sentinel_are_the_public_key(self):
        assert tenant_key(None) == PUBLIC_TENANT == "__public__"
        assert tenant_key("__public__") == PUBLIC_TENANT

    def test_real_tenant_is_unchanged(self):
        assert tenant_key("acme") == "acme"

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            tenant_key("")


# ---------------------------------------------------------------------------
# Ingestion: Document identity and StructureNode ids
# ---------------------------------------------------------------------------


class TestInsertDocumentGraphTenantIdentity:
    def test_every_document_pattern_carries_the_tenant(self):
        tx = _ingest("A")
        patterns = _document_patterns(tx.calls)
        # leaf + 2 folders MERGEd, versioning, IS_COMPOSED_OF links, HAS_STRUCTURE
        assert len(patterns) >= 8
        for props in patterns:
            assert "tenant_id: $tenant_id" in props, props
        for q, p in tx.calls:
            if ":Document" in q:
                assert p["tenant_id"] == "A"

    def test_public_document_uses_the_reserved_key(self):
        tx = _ingest(None)
        doc_params = [p for q, p in tx.calls if ":Document" in q]
        assert doc_params and all(p["tenant_id"] == PUBLIC_TENANT for p in doc_params)
        assert all(i.startswith("__public__::") for i in _structure_ids(tx))

    def test_same_path_two_tenants_get_disjoint_structure_ids(self):
        ids_a, ids_b = _structure_ids(_ingest("A")), _structure_ids(_ingest("B"))
        assert ids_a == ["A::Folder/Sub/doc::1::1", "A::Folder/Sub/doc::1::1/1_1"]
        assert ids_b == ["B::Folder/Sub/doc::1::1", "B::Folder/Sub/doc::1::1/1_1"]
        assert not set(ids_a) & set(ids_b)

    def test_update_mode_wipes_only_the_tenants_document(self):
        tx = _ingest("A", update_mode=True)
        wipes = [(q, p) for q, p in tx.calls if "DETACH DELETE" in q]
        assert wipes
        for q, p in wipes:
            assert "{tenant_id: $tenant_id, path: $path, version: $version}" in q
            assert p["tenant_id"] == "A"

    def test_public_sentinel_is_the_same_document_as_none(self):
        assert _structure_ids(_ingest("__public__")) == _structure_ids(_ingest(None))
        assert [p["tenant_id"] for _, p in _ingest("__public__").calls if "tenant_id" in p] == [
            p["tenant_id"] for _, p in _ingest(None).calls if "tenant_id" in p
        ]

    def test_invalid_tenant_is_rejected_before_any_write(self):
        tx = _Tx()
        with pytest.raises(ValueError):
            nodes.insert_document_graph(tx, _doc(""), resolved_version=1)
        assert tx.calls == []


# ---------------------------------------------------------------------------
# Derived uids inherit the tenant from the StructureNode id
# ---------------------------------------------------------------------------


class TestDerivedUidsDifferPerTenant:
    def test_info_unit_uids(self):
        a, b = _info_unit_uids(_ingest("A")), _info_unit_uids(_ingest("B"))
        assert len(a) == len(b) == 2
        assert not set(a) & set(b)

    async def test_model_decision_uids(self):
        from scinr.newton.annotation.neo4j_ops import write_manual_annotation

        async def _uids(tenant):
            node_ids = _structure_ids(_ingest(tenant))

            def respond(q, p):
                if "RETURN DISTINCT n.id AS full_node_id" in q:
                    return _AsyncResult(data=[{"full_node_id": i} for i in node_ids])
                return _AsyncResult()

            driver = _AsyncDriver(respond)
            await write_manual_annotation(
                driver, "doc", "SomeModel", tenant_id=tenant, doc_path="Folder/Sub/doc"
            )
            return [p["uid"] for q, p in driver.calls if "CREATE (md:ModelDecision" in q]

        a, b = await _uids("A"), await _uids("B")
        assert len(a) == len(b) == 2
        assert not set(a) & set(b)

    async def test_extraction_result_uids(self):
        from scinr.newton.entity_extraction import nodes as ee_nodes

        async def _uids(tenant):
            write = AsyncMock()
            with patch.object(ee_nodes, "write_triple_subgraph", write):
                for node_id in _structure_ids(_ingest(tenant)):
                    target = {"node_full_id": node_id, "model_class": None}
                    await ee_nodes._write_entities(object(), target, object(), "doc")
            return [c.kwargs["extraction_uid"] for c in write.await_args_list]

        a, b = await _uids("A"), await _uids("B")
        assert len(a) == len(b) == 2
        assert not set(a) & set(b)


# ---------------------------------------------------------------------------
# Version resolution is per tenant
# ---------------------------------------------------------------------------


class TestVersionResolutionPerTenant:
    def test_get_next_version_filters_by_tenant(self):
        session = _Tx(single={"max_version": 4})
        assert nodes.get_next_version(session, "Folder/doc", "A") == 5
        query, params = session.calls[0]
        assert "{tenant_id: $tenant_id, path: $path}" in query
        assert params == {"tenant_id": "A", "path": "Folder/doc"}

    def test_get_current_latest_version_filters_by_tenant(self):
        session = _Tx(single={"version": 2})
        assert nodes.get_current_latest_version(session, "Folder/doc", "A") == 2
        query, params = session.calls[0]
        assert "{tenant_id: $tenant_id, path: $path, latest: true}" in query
        assert params["tenant_id"] == "A"

    @pytest.mark.parametrize("update_mode", [False, True])
    def test_batch_resolution_filters_by_tenant(self, update_mode):
        session = _Tx(single={"max_version": 3})
        loader._resolve_batch_version(session, ["Folder", "Folder/doc"], update_mode, [None])
        query, params = session.calls[0]
        assert "d.tenant_id IN $tenant_ids" in query
        assert params["tenant_ids"] == [PUBLIC_TENANT]

    def test_batch_tenants_override_wins_over_baked_values(self):
        assert loader._batch_tenants("acme", ["x", None]) == ["acme"]
        assert loader._batch_tenants(None, ["x", None, "x"]) == ["x", None]

    def test_load_documents_resolves_version_within_the_override_tenant(self, monkeypatch):
        session = _Tx(single={"max_version": None})
        driver = SimpleNamespace(session=lambda **kw: _CtxSession(session))
        monkeypatch.setattr(loader, "_load_document_object", lambda doc, *a, **k: doc.document_name)

        loader.load_documents([_doc(None)], driver, tenant_id="acme")

        assert session.calls[0][1]["tenant_ids"] == ["acme"]


class _CtxSession:
    def __init__(self, inner):
        self.inner = inner

    def __enter__(self):
        return self.inner

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Stages select documents by (tenant, path), never by name alone
# ---------------------------------------------------------------------------


class TestStageSelectors:
    async def test_fetch_nodes_to_annotate(self):
        from scinr.newton.annotation.neo4j_ops import fetch_nodes_to_annotate

        driver = _AsyncDriver()
        await fetch_nodes_to_annotate(driver, tenant_id="A", doc_path="Folder/doc")
        query, params = driver.calls[0]
        assert "{tenant_id: $tenant_id, path: $doc_path, latest: true}" in query
        assert "name:" not in query
        assert params == {"tenant_id": "A", "doc_path": "Folder/doc"}

    async def test_fetch_document_context_instructions_public(self):
        from scinr.newton.annotation.neo4j_ops import fetch_document_context_instructions

        driver = _AsyncDriver()
        await fetch_document_context_instructions(driver, tenant_id=None, doc_path="doc")
        query, params = driver.calls[0]
        assert "{tenant_id: $tenant_id, path: $doc_path, latest: true}" in query
        assert params["tenant_id"] == PUBLIC_TENANT

    async def test_fetch_extraction_targets(self):
        from scinr.newton.entity_extraction.neo4j_ops import fetch_extraction_targets

        driver = _AsyncDriver()
        await fetch_extraction_targets(driver, tenant_id="A", doc_path="Folder/doc")
        query, params = driver.calls[0]
        assert "{tenant_id: $tenant_id, path: $doc_path, latest: true}" in query
        assert params == {"tenant_id": "A", "doc_path": "Folder/doc"}

    async def test_resolver_pins_root_and_leaves_to_the_tenant(self):
        from scinr.newton.utils.document_resolver import (
            LeafDocument,
            resolve_leaf_documents_async,
        )

        rows = [{"docs": [{"name": "a", "path": "F/a"}, {"name": "b", "path": "F/b"}]}]
        driver = _AsyncDriver(lambda q, p: _AsyncResult(data=rows))

        leaves = await resolve_leaf_documents_async(driver, tenant_id="A", doc_path="F")

        assert leaves == [LeafDocument("a", "F/a"), LeafDocument("b", "F/b")]
        query, params = driver.calls[0]
        assert "(root:Document {tenant_id: $tenant_id, path: $doc_selector, latest: true})" in query
        assert "(leaf:Document {tenant_id: $tenant_id, latest: true})" in query
        assert params == {"tenant_id": "A", "doc_selector": "F"}

    async def test_resolver_by_name_is_still_tenant_scoped_and_dedups_by_path(self):
        from scinr.newton.utils.document_resolver import resolve_leaf_documents_async

        # Two latest documents named "doc" in different folders of the tenant.
        rows = [
            {"docs": [{"name": "doc", "path": "X/doc"}]},
            {"docs": [{"name": "doc", "path": "Y/doc"}, {"name": "doc", "path": "X/doc"}]},
        ]
        driver = _AsyncDriver(lambda q, p: _AsyncResult(data=rows))

        leaves = await resolve_leaf_documents_async(driver, tenant_id=None, document_name="doc")

        assert [leaf.path for leaf in leaves] == ["X/doc", "Y/doc"]
        query, params = driver.calls[0]
        assert "{tenant_id: $tenant_id, name: $doc_selector, latest: true}" in query
        assert params["tenant_id"] == PUBLIC_TENANT


class TestTabularTenantScope:
    async def test_table_attaches_to_the_tenants_existing_document(self, monkeypatch):
        from scinr.newton.tabular import neo4j_ops as tab_ops

        driver = _AsyncDriver()
        tx_calls: list[tuple[str, dict]] = []

        class _Tx:
            async def run(self, q, **p):
                tx_calls.append((q, p))
                return _AsyncResult(single={"created": 0})

            async def commit(self):
                pass

            async def rollback(self):
                pass

        async def begin_transaction(self):
            return _Tx()

        monkeypatch.setattr(_AsyncSession, "begin_transaction", begin_transaction, raising=False)
        monkeypatch.setattr(tab_ops, "get_config", lambda: SimpleNamespace(neo4j_database="neo4j"))

        with pytest.raises(RuntimeError, match="Document not found"):
            await tab_ops.write_tabular_subgraph(
                driver=driver,
                doc_path="F/sheet",
                document_name="sheet",
                resolved_version=1,
                sheet={"sheet_name": "s", "headers": [], "total_rows": 0},
                row_batches=lambda n: iter(()),
                sheet_index=0,
                decision=SimpleNamespace(matched_model_class=None),
                mapping=None,
                tenant_id="A",
            )

        query, params = tx_calls[0]
        # MATCH, not MERGE: a missing Document must not be silently created.
        assert "MATCH (d:Document {tenant_id: $tenant_id, path: $doc_path, version: $version})" in query
        assert "MERGE (d:Document" not in query
        assert params["tenant_id"] == "A"
        assert params["id"] == "A::F/sheet::1::table_1"


class TestReplacesTenantScope:
    def test_preflight_only_sees_the_tenants_documents(self):
        from scinr.newton.stages.ingestion import preflight_check_replaces

        session = _Tx()
        session.run = lambda q, **p: (session.calls.append((q, p)), [{"path": "p", "version": 1}])[1]
        driver = SimpleNamespace(session=lambda **kw: _CtxSession(session))

        assert preflight_check_replaces(driver, "old", tenant_id="A") == {"path": "p", "version": 1}
        query, params = session.calls[0]
        assert "{tenant_id: $tenant_id, name: $name, latest: true}" in query
        assert params == {"tenant_id": "A", "name": "old"}

    def test_apply_replacement_links_within_the_tenant_by_path(self):
        from scinr.newton.stages.ingestion import apply_replacement

        session = _Tx()
        responses = iter([[{"path": "new", "version": 1, "name": "newdoc"}], []])
        session.run = lambda q, **p: (session.calls.append((q, p)), next(responses))[1]
        driver = SimpleNamespace(session=lambda **kw: _CtxSession(session))

        apply_replacement(driver, "old", ["newdoc"], tenant_id="A", replaced_path="Old/old")

        (q_roots, p_roots), (q_link, p_link) = session.calls
        assert "{tenant_id: $tenant_id, latest: true}" in q_roots
        assert p_roots["tenant_id"] == "A"
        assert "(old:Document {tenant_id: $tenant_id, path: $old_path, latest: true})" in q_link
        assert "(new:Document {tenant_id: $tenant_id, path: $new_path, version: $new_version})" in q_link
        assert "WHERE old <> new" in q_link
        assert p_link["tenant_id"] == "A" and p_link["old_path"] == "Old/old"
