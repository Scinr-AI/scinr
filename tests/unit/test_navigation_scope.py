"""
Tests for the read-side scope (tenant / include_public / user / job) of the
navigation layer — ``navigation/scope.py``, ``navigation/scoped.py`` and the
scope predicates the Neo4j backend emits. No real Neo4j: a recording fake driver.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import FakeAsyncDriver, make_fake_llm  # noqa: E402

from scinr.newton.config import configure  # noqa: E402
from scinr.newton.exceptions import NavigationError  # noqa: E402
from scinr.newton.navigation.base import GraphNavigator  # noqa: E402
from scinr.newton.navigation.models import DocumentRef, NodeSelector  # noqa: E402
from scinr.newton.navigation.neo4j.navigator import Neo4jGraphNavigator  # noqa: E402
from scinr.newton.navigation.scope import make_scope, resolve_selector  # noqa: E402

_NEO = {"neo4j_user": "neo4j", "neo4j_password": "pw", "llm": make_fake_llm()}

SCOPE_KWARGS = ("tenant_id", "include_public", "created_by_user_id", "job_id")
#: Global catalogue / lifecycle / raw methods: they take no scope.
EXEMPT = {
    "connect", "close", "ping", "execute_raw", "execute_raw_one",
    "list_catalog_models", "get_catalog_graph", "list_themes",
    "list_relationship_types", "list_node_labels",
}


def _mk(responder=None) -> tuple[Neo4jGraphNavigator, FakeAsyncDriver]:
    configure(**_NEO)
    drv = FakeAsyncDriver(responder or (lambda cypher, params: []))
    return Neo4jGraphNavigator(driver=drv), drv


def _public_methods(cls: type) -> dict[str, Any]:
    return {
        n: f
        for n, f in inspect.getmembers(cls, inspect.iscoroutinefunction)
        if not n.startswith("_")
    }


_ARG_BY_NAME: dict[str, Any] = {
    "path": "A/b", "document": "A/b", "node_id": "n1", "node_ids": ["n1"], "uid": "u1",
    "model_class": "M", "theme": "t", "text": "q", "value_or_uid": "v",
    "rel_type": "R", "key_fields": {"k": "v"}, "version": 1,
    "selector": NodeSelector(type="Document", key="path", value="x"),
    "from_selector": NodeSelector(type="Document", key="path", value="x"),
    "to_selector": NodeSelector(type="Document", key="path", value="y"),
}


def _required_args(fn: Any) -> list[Any]:
    out = []
    for name, p in inspect.signature(fn).parameters.items():
        if name == "self" or p.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        if p.default is inspect.Parameter.empty:
            out.append(_ARG_BY_NAME[name])
    return out


# ---------------------------------------------------------------------------
# make_scope / Scope
# ---------------------------------------------------------------------------


class TestMakeScope:
    def test_none_is_no_tenant_filter(self):
        sc = make_scope()
        assert sc.tenants is None and sc.is_unfiltered
        assert sc.clauses("n") == [] and sc.params() == {}

    def test_public_sentinel(self):
        assert make_scope("__public__").tenants == ("__public__",)
        assert make_scope("__public__", include_public=True).tenants == ("__public__",)

    def test_tenant_alone_and_with_public(self):
        assert make_scope("acme").tenants == ("acme",)
        assert make_scope("acme", True).tenants == ("acme", "__public__")

    def test_include_public_without_tenant_does_nothing(self):
        assert make_scope(None, True).tenants is None

    @pytest.mark.parametrize("bad", ["", 3])
    def test_invalid_tenant(self, bad):
        with pytest.raises(NavigationError):
            make_scope(bad)

    @pytest.mark.parametrize("kw", ["created_by_user_id", "job_id"])
    def test_empty_list_is_an_error(self, kw):
        with pytest.raises(NavigationError):
            make_scope(**{kw: []})
        with pytest.raises(NavigationError):
            make_scope(**{kw: [""]})

    def test_str_is_a_list_of_one(self):
        sc = make_scope(job_id="j1", created_by_user_id=["u1", "u2"])
        assert sc.job_ids == ("j1",) and sc.user_ids == ("u1", "u2")


class TestScopeClauses:
    def test_single_tenant_uses_equality(self):
        sc = make_scope("acme")
        assert sc.clauses("n") == ["n.tenant_id = $scope_tenant"]
        assert sc.params() == {"scope_tenant": "acme"}

    def test_public_uses_in(self):
        sc = make_scope("acme", True)
        assert sc.clauses("n") == ["n.tenant_id IN $scope_tenants"]
        assert sc.params() == {"scope_tenants": ["acme", "__public__"]}

    def test_scalar_provenance(self):
        sc = make_scope(job_id=["j1", "j2"], created_by_user_id="u")
        assert sc.clauses("n", "scalar") == [
            "n.created_by_user_id IN $scope_user_ids",
            "n.job_id IN $scope_job_ids",
        ]
        assert sc.params() == {"scope_user_ids": ["u"], "scope_job_ids": ["j1", "j2"]}

    def test_array_provenance_matches_any(self):
        sc = make_scope(job_id=["j1", "j2"])
        (clause,) = sc.clauses("mi", "array")
        assert clause == "any(x IN $scope_job_ids WHERE x IN coalesce(mi.job_ids, []))"

    def test_tenant_only_kind_skips_provenance(self):
        sc = make_scope("acme", job_id="j")
        assert sc.clauses("d", "tenant_only") == ["d.tenant_id = $scope_tenant"]

    def test_tenant_false_skips_tenant(self):
        sc = make_scope("acme", job_id="j")
        assert sc.clauses("n", tenant=False) == ["n.job_id IN $scope_job_ids"]
        assert "scope_tenant" not in sc.params(tenant=False)


class TestResolveSelector:
    def _ref(self, tenant):
        return DocumentRef(path="A/b", name="b", version=1, latest=True, is_folder=False, tenant_id=tenant)

    def test_string_passes_through(self):
        sc = make_scope("acme")
        assert resolve_selector("A/b", sc) == ("A/b", sc)

    def test_ref_pins_the_tenant(self):
        path, sc = resolve_selector(self._ref("acme"), make_scope())
        assert path == "A/b" and sc.tenants == ("acme",)

    def test_ref_public_pins_public(self):
        _, sc = resolve_selector(self._ref("__public__"), make_scope("acme", True))
        assert sc.tenants == ("__public__",)

    def test_ref_with_other_tenant_is_an_error(self):
        with pytest.raises(NavigationError):
            resolve_selector(self._ref("acme"), make_scope("globex"))

    def test_bad_selector_type(self):
        with pytest.raises(TypeError):
            resolve_selector(3, make_scope())


# ---------------------------------------------------------------------------
# Contract by signature
# ---------------------------------------------------------------------------


class TestSignatureContract:
    @pytest.mark.parametrize("cls", [GraphNavigator, Neo4jGraphNavigator])
    def test_every_non_exempt_method_takes_the_four_filters_keyword_only(self, cls):
        for name, fn in _public_methods(cls).items():
            params = inspect.signature(fn).parameters
            if name in EXEMPT:
                assert not any(k in params for k in SCOPE_KWARGS), name
                continue
            for k in SCOPE_KWARGS:
                assert k in params, f"{cls.__name__}.{name} lacks {k}"
                assert params[k].kind is inspect.Parameter.KEYWORD_ONLY, (name, k)
                assert params[k].default in (None, False), (name, k)


# ---------------------------------------------------------------------------
# Contract by Cypher
# ---------------------------------------------------------------------------

_NON_EXEMPT = sorted(n for n in _public_methods(GraphNavigator) if n not in EXEMPT)
#: The uid embeds the tenant, so this lookup cannot span "all tenants" and
#: resolves ``include_public`` by trying each tenant in turn (see its own tests).
_TENANT_KEYED = {"get_model_instance_by_key"}
_SPANNING = [n for n in _NON_EXEMPT if n not in _TENANT_KEYED]


def _uses(drv: FakeAsyncDriver, param: str) -> bool:
    return any(f"${param}" in q and param in p for q, p in drv.last_queries)


async def _call(name: str, **scope: Any) -> FakeAsyncDriver:
    nav, drv = _mk()
    fn = getattr(nav, name)
    await fn(*_required_args(fn), **scope)
    return drv


class TestCypherContract:
    @pytest.mark.parametrize("name", _NON_EXEMPT)
    async def test_tenant_predicate_is_emitted(self, name):
        drv = await _call(name, tenant_id="acme")
        assert _uses(drv, "scope_tenant"), name
        assert all(p["scope_tenant"] == "acme" for _, p in drv.last_queries if "scope_tenant" in p)

    @pytest.mark.parametrize("name", _NON_EXEMPT)
    async def test_public_predicate(self, name):
        drv = await _call(name, tenant_id="__public__")
        assert _uses(drv, "scope_tenant"), name
        assert all(p["scope_tenant"] == "__public__" for _, p in drv.last_queries if "scope_tenant" in p)

    @pytest.mark.parametrize("name", _SPANNING)
    async def test_include_public_predicate(self, name):
        drv = await _call(name, tenant_id="acme", include_public=True)
        assert _uses(drv, "scope_tenants"), name
        assert all(
            p["scope_tenants"] == ["acme", "__public__"]
            for _, p in drv.last_queries
            if "scope_tenants" in p
        )

    @pytest.mark.parametrize("name", _SPANNING)
    async def test_none_means_no_tenant_predicate(self, name):
        drv = await _call(name)
        assert not any("scope_tenant" in q or "scope_tenant" in p for q, p in drv.last_queries), name

    @pytest.mark.parametrize("name", _SPANNING)
    async def test_job_and_user_are_in_predicates(self, name):
        drv = await _call(name, job_id=["j1", "j2"], created_by_user_id="u1")
        assert _uses(drv, "scope_job_ids"), name
        assert _uses(drv, "scope_user_ids"), name
        assert any(p.get("scope_job_ids") == ["j1", "j2"] for _, p in drv.last_queries)
        assert any(p.get("scope_user_ids") == ["u1"] for _, p in drv.last_queries)

    @pytest.mark.parametrize("name", _NON_EXEMPT)
    async def test_empty_list_is_rejected_before_any_query(self, name):
        nav, drv = _mk()
        fn = getattr(nav, name)
        with pytest.raises(NavigationError):
            await fn(*_required_args(fn), job_id=[])
        assert drv.last_queries == []

    async def test_get_model_instance_by_key_needs_a_tenant(self):
        nav, drv = _mk()
        with pytest.raises(NavigationError, match="tenant"):
            await nav.get_model_instance_by_key("M", {"k": "v"})
        assert drv.last_queries == []


class TestProvenanceForms:
    async def test_merged_nodes_use_array_provenance(self):
        drv = await _call("get_model_instances_by_class", job_id="j")
        (q, _), = [x for x in drv.last_queries if "scope_job_ids" in x[0]]
        assert "coalesce(mi.job_ids, [])" in q

    async def test_scalar_nodes_use_scalar_provenance(self):
        drv = await _call("find_structure_nodes", job_id="j")
        (q, _), = [x for x in drv.last_queries if "scope_job_ids" in x[0]]
        assert "n.job_id IN $scope_job_ids" in q

    async def test_document_anchor_carries_the_tenant(self):
        drv = await _call("get_structure_nodes", tenant_id="acme")
        q, _ = next(x for x in drv.last_queries if "scope_tenant" in x[0])
        assert "doc.tenant_id = $scope_tenant" in q

    async def test_node_anchor_guards_by_tenant(self):
        drv = await _call("get_child_nodes", tenant_id="acme")
        q, _ = next(x for x in drv.last_queries if "scope_tenant" in x[0])
        assert "a.tenant_id = $scope_tenant" in q

    async def test_fulltext_search_filters_before_limit(self):
        drv = await _call("search_info_units", tenant_id="acme")
        q, _ = next(x for x in drv.last_queries if "fulltext" in x[0])
        assert q.index("node.tenant_id") < q.index("LIMIT $lim")

    async def test_where_tenant_and_argument_combine_by_and(self):
        nav, drv = _mk()
        await nav.get_documents(where={"tenant_id": "acme"}, tenant_id="acme")
        q, _ = drv.last_queries[-1]
        assert "d.tenant_id = $scope_tenant" in q
        assert "d.`tenant_id` = $w_tenant_id" in q


# ---------------------------------------------------------------------------
# Single-document ambiguity (rules A / B / D)
# ---------------------------------------------------------------------------


def _held(*tenants):
    """Responder: the tenant-listing query returns these tenants; others nothing."""

    def responder(cypher: str, params: dict):
        if "RETURN d.tenant_id AS tenant_id" in cypher:
            return [{"tenant_id": t} for t in tenants]
        return []

    return responder


class TestSingleDocumentAmbiguity:
    async def test_two_tenants_without_tenant_is_ambiguous(self):
        nav, _ = _mk(_held("A", "B"))
        with pytest.raises(NavigationError, match="several tenants"):
            await nav.get_document_tree("A/b")

    @pytest.mark.parametrize(
        "name",
        [
            "get_one_document", "get_latest_version", "get_document_tree",
            "get_document_parent", "get_document_ancestors", "get_document_stats",
            "get_document_model_profile", "get_annotation_coverage",
        ],
    )
    async def test_every_single_document_method_raises(self, name):
        nav, _ = _mk(_held("A", "B"))
        fn = getattr(nav, name)
        with pytest.raises(NavigationError, match="several tenants"):
            await fn(*_required_args(fn))

    async def test_lists_return_every_match(self):
        nav, drv = _mk(_held("A", "B"))
        await nav.list_document_versions("A/b")  # no error, no resolution query
        assert not any("RETURN d.tenant_id AS tenant_id" in q for q, _ in drv.last_queries)

    async def test_tenant_shadows_public_with_include_public(self):
        nav, drv = _mk(_held("acme", "__public__"))
        await nav.get_document_tree("A/b", tenant_id="acme", include_public=True)
        q, p = drv.last_queries[-1]
        assert "root.tenant_id = $scope_tenant" in q and p["scope_tenant"] == "acme"

    async def test_single_tenant_needs_no_resolution_query(self):
        nav, drv = _mk(_held("A", "B"))
        await nav.get_document_tree("A/b", tenant_id="A")
        assert not any("RETURN d.tenant_id AS tenant_id" in q for q, _ in drv.last_queries)

    async def test_public_and_tenant_without_include_public_is_ambiguous_for_all(self):
        nav, _ = _mk(_held("acme", "__public__"))
        with pytest.raises(NavigationError):
            await nav.get_latest_version("A/b")

    async def test_documentref_tenant_is_authoritative(self):
        nav, drv = _mk()
        ref = DocumentRef(path="A/b", name="b", version=1, latest=True, is_folder=False, tenant_id="acme")
        await nav.get_structure_nodes(ref)
        assert any(p.get("scope_tenant") == "acme" for _, p in drv.last_queries)

    async def test_documentref_conflicting_tenant_raises(self):
        nav, _ = _mk()
        ref = DocumentRef(path="A/b", name="b", version=1, latest=True, is_folder=False, tenant_id="acme")
        with pytest.raises(NavigationError):
            await nav.get_structure_nodes(ref, tenant_id="globex")


# ---------------------------------------------------------------------------
# Mapping & tools
# ---------------------------------------------------------------------------


class TestMapping:
    async def test_document_ref_exposes_the_stored_tenant(self):
        row = {"path": "p", "name": "n", "version": 1, "latest": True,
               "is_folder": False, "tenant_id": "__public__"}
        nav, _ = _mk(lambda q, p: [{"d": row}] if "RETURN d" in q else [])
        (ref,) = await nav.get_documents()
        assert ref.tenant_id == "__public__"


class TestGetModelInstanceByKey:
    async def test_include_public_tries_tenant_then_public(self):
        nav, drv = _mk()
        assert await nav.get_model_instance_by_key("M", {"k": "v"}, tenant_id="acme", include_public=True) is None
        uids = [p["uid"] for _, p in drv.last_queries if "uid" in p]
        assert len(uids) == 2 and uids[0] != uids[1]

    async def test_public_sentinel_equals_public_uid(self):
        from scinr.newton.utils.uid import make_instance_uid

        nav, drv = _mk()
        await nav.get_model_instance_by_key("M", {"k": "v"}, tenant_id="__public__")
        (uid,) = [p["uid"] for _, p in drv.last_queries if "uid" in p]
        assert uid == make_instance_uid("M", {"k": "v"}, None)


class TestPowerTools:
    async def test_neighbors_restricts_intermediates_and_ends(self):
        nav, drv = _mk()
        await nav.neighbors(NodeSelector(type="Document", key="path", value="x"), depth=3, tenant_id="acme")
        q, p = next(x for x in drv.last_queries if "scope_tenant" in x[0])
        assert "all(m IN nodes(p)[1..-1] WHERE m.tenant_id = $scope_tenant)" in q
        assert "$scope_catalog_labels" in q and "CatalogModel" in p["scope_catalog_labels"]

    async def test_neighbors_without_tenant_is_unrestricted(self):
        nav, drv = _mk()
        await nav.neighbors(NodeSelector(type="Document", key="path", value="x"), depth=3)
        assert not any("scope_" in q for q, _ in drv.last_queries)

    async def test_shortest_path_catalogue_only_as_endpoint(self):
        nav, drv = _mk()
        await nav.shortest_path(
            NodeSelector(type="Document", key="path", value="x"),
            NodeSelector(type="Document", key="path", value="y"),
            tenant_id="acme",
        )
        q, _ = next(x for x in drv.last_queries if "scope_tenant" in x[0])
        assert "m = a OR m = b OR m.tenant_id = $scope_tenant" in q

    async def test_scoped_subgraph_avoids_apoc(self):
        nav, drv = _mk()
        await nav.subgraph(NodeSelector(type="Document", key="path", value="x"), tenant_id="acme")
        assert not any("apoc" in q for q, _ in drv.last_queries)


class TestGraphSummary:
    async def test_scoped_summary_counts_by_label(self):
        def responder(cypher, params):
            if "MATCH (x:Document)" in cypher and "count(x)" in cypher:
                return [{"c": 2}]
            if "MATCH (d:Document)" in cypher:
                return [{"total": 2, "latest": 1}]
            return []

        nav, drv = _mk(responder)
        s = await nav.get_graph_summary(tenant_id="acme")
        assert s.node_counts == {"Document": 2}
        assert s.documents == 2 and s.latest_documents == 1
        scoped = [p for q, p in drv.last_queries if "x.tenant_id" in q]
        assert scoped and all(p["scope_tenant"] == "acme" for p in scoped)

    async def test_unscoped_summary_is_unchanged(self):
        nav, drv = _mk()
        await nav.get_graph_summary()
        assert any("apoc.meta.stats" in q for q, _ in drv.last_queries)


# ---------------------------------------------------------------------------
# ScopedNavigator
# ---------------------------------------------------------------------------


class TestScopedNavigator:
    async def test_fills_the_scope_into_every_call(self):
        nav, drv = _mk()
        s = nav.scoped(tenant_id="acme", include_public=True, job_id=["j1", "j2"])
        await s.get_documents()
        _, p = drv.last_queries[-1]
        assert p["scope_tenants"] == ["acme", "__public__"]
        assert p["scope_job_ids"] == ["j1", "j2"]

    async def test_rejects_another_tenant(self):
        nav, drv = _mk()
        s = nav.scoped(tenant_id="acme")
        with pytest.raises(NavigationError, match="outside"):
            await s.get_documents(tenant_id="globex")
        assert drv.last_queries == []

    async def test_can_narrow_to_public_inside_an_include_public_view(self):
        nav, drv = _mk()
        s = nav.scoped(tenant_id="acme", include_public=True)
        await s.get_documents(tenant_id="__public__")
        assert drv.last_queries[-1][1]["scope_tenant"] == "__public__"

    async def test_cannot_widen_with_include_public(self):
        nav, _ = _mk()
        s = nav.scoped(tenant_id="acme")
        with pytest.raises(NavigationError, match="widens"):
            await s.get_documents(include_public=True)

    async def test_job_ids_may_narrow_not_widen(self):
        nav, drv = _mk()
        s = nav.scoped(tenant_id="acme", job_id=["j1", "j2"])
        await s.get_documents(job_id="j1")
        assert drv.last_queries[-1][1]["scope_job_ids"] == ["j1"]
        with pytest.raises(NavigationError):
            await s.get_documents(job_id=["j1", "j3"])

    async def test_a_view_without_tenant_accepts_one_per_call(self):
        nav, drv = _mk()
        s = nav.scoped(job_id="j1")
        await s.get_documents(tenant_id="acme")
        assert drv.last_queries[-1][1]["scope_tenant"] == "acme"

    async def test_execute_raw_is_refused(self):
        nav, drv = _mk()
        s = nav.scoped(tenant_id="acme")
        with pytest.raises(NavigationError, match="scoped"):
            await s.execute_raw("MATCH (n) RETURN n")
        assert drv.last_queries == []

    async def test_catalogue_methods_pass_through(self):
        nav, drv = _mk()
        s = nav.scoped(tenant_id="acme")
        await s.list_catalog_models()
        assert not any("scope_" in q for q, _ in drv.last_queries)

    async def test_scoped_of_scoped_only_narrows(self):
        nav, _ = _mk()
        s = nav.scoped(tenant_id="acme").scoped(job_id="j1")
        with pytest.raises(NavigationError):
            await s.get_documents(tenant_id="globex")

    def test_invalid_scope_fails_at_construction(self):
        nav, _ = _mk()
        with pytest.raises(NavigationError):
            nav.scoped(tenant_id="")
