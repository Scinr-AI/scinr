"""
tests/sql_rls/test_rls.py -- A restricted role cannot read or write the tenants it was not granted.

Cases 1 to 17 of ``plans/sql-backends/06-rls-admin.md`` (case 18, the full contract suite as a
restricted role, is the ``postgres_rls`` backend of ``tests/contract``).
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg import sql

from scinr.newton import (
    grant_tenant_access,
    list_tenant_access,
    revoke_tenant_access,
    setup_sql_schema,
)
from scinr.newton.exceptions import (
    GraphConnectionError,
    NavigationError,
    TenantAccessError,
)
from scinr.newton.graph.types import DocumentSelector
from scinr.newton.sql.schema import ensure_schema, rls_enabled
from tests.contract._helpers import make_document
from tests.golden._dump import normalize_artifacts
from tests.sql._dump import dump_sql
from tests.sql.conftest import drop_pg_schema

from ._support import with_credentials

pytestmark = pytest.mark.sql_rls


def _admin_scalar(db, statement: str, *params):
    with psycopg.connect(db.admin_url) as connection:
        return connection.execute(
            sql.SQL(statement).format(sql.Identifier(db.schema)), params
        ).fetchone()[0]


def _doc_count(db, tenant: str) -> int:
    return _admin_scalar(db, "SELECT count(*) FROM {}.documents WHERE tenant_id = %s", tenant)


def _insert(store, name: str, tenant: str | None) -> None:
    store.documents.insert_document_graph(make_document(name, tenant=tenant), 1, False)


async def _paths(navigator, **scope) -> set[str]:
    return {d.path for d in await navigator.get_documents(**scope)}


# ── 1, 2: setup ──────────────────────────────────────────────────────────────


def test_setup_is_idempotent(rls_db) -> None:
    first = setup_sql_schema(
        rls_db.admin_url, enable_rls=True, runtime_role=rls_db.role_a.name, schema=rls_db.schema
    )
    second = setup_sql_schema(
        rls_db.admin_url, enable_rls=True, runtime_role=rls_db.role_a.name, schema=rls_db.schema
    )
    assert first == second
    assert first.rls_enabled and first.rls_version == 1 and first.version >= 2


def test_setup_without_enable_rls_does_not_turn_it_off(rls_db, caplog) -> None:
    status = setup_sql_schema(rls_db.admin_url, schema=rls_db.schema)
    assert status.rls_enabled
    assert "RLS mode" in caplog.text


def test_rls_needs_an_existing_runtime_role(rls_db) -> None:
    with pytest.raises(ValueError, match="runtime_role"):
        setup_sql_schema(rls_db.admin_url, enable_rls=True, schema=rls_db.schema)
    with pytest.raises(ValueError, match="create the role first: CREATE ROLE newton_rt_nobody"):
        setup_sql_schema(
            rls_db.admin_url,
            enable_rls=True,
            runtime_role="newton_rt_nobody",
            schema=rls_db.schema,
        )


def test_grants_need_rls_mode(pg_url, pg_schema) -> None:
    setup_sql_schema(pg_url, schema=pg_schema)
    try:
        with pytest.raises(ValueError, match="RLS is not enabled"):
            grant_tenant_access(pg_url, "newton_rt_x", "t1", schema=pg_schema)
    finally:
        drop_pg_schema(pg_url, pg_schema)


# ── 3, 4: writes ─────────────────────────────────────────────────────────────


async def test_granted_tenant_ingests_like_in_simple_mode(
    rls_db, store_as, pg_url, pg_schema
) -> None:
    grant_tenant_access(
        rls_db.admin_url, rls_db.role_a.name, "t1", write=True, schema=rls_db.schema
    )
    _insert(store_as(rls_db.role_a), "a.txt", "t1")
    rls_dump = normalize_artifacts([], dump_sql(rls_db.admin_url, rls_db.schema), {})[1]

    # Same document through a plain (simple mode) schema of the same container.
    from scinr.newton import configure
    from scinr.newton.graph import get_graph_store, reset_graph_store
    from scinr.newton.sql.engine import reset_engines

    reset_graph_store()
    reset_engines()
    drop_pg_schema(pg_url, pg_schema)
    configure(
        graph_backend="postgres",
        storage_backend="postgres",
        freeze_backend="postgres",
        sql_url=pg_url,
        sql_schema=pg_schema,
    )
    try:
        simple = get_graph_store()
        simple.schema.setup()
        _insert(simple, "a.txt", "t1")
        simple_dump = normalize_artifacts([], dump_sql(pg_url, pg_schema), {})[1]
        await simple.aclose()
    finally:
        reset_graph_store()
        reset_engines()
        drop_pg_schema(pg_url, pg_schema)
    assert rls_dump == simple_dump


async def test_write_into_a_tenant_not_granted_fails_and_writes_nothing(rls_db, store_as) -> None:
    grant_tenant_access(
        rls_db.admin_url, rls_db.role_a.name, "t1", write=True, schema=rls_db.schema
    )
    store = store_as(rls_db.role_a)
    with pytest.raises(TenantAccessError):
        _insert(store, "a.txt", "t2")
    assert _doc_count(rls_db, "t2") == 0
    assert _admin_scalar(rls_db, "SELECT count(*) FROM {}.structure_nodes") == 0


async def test_read_only_grant_cannot_write(rls_db, store_as) -> None:
    grant_tenant_access(rls_db.admin_url, rls_db.role_a.name, "t1", schema=rls_db.schema)
    with pytest.raises(TenantAccessError):
        _insert(store_as(rls_db.role_a), "a.txt", "t1")


# ── 5 to 8: reads ────────────────────────────────────────────────────────────


async def _seed(db, store_as, *tenants: str | None) -> None:
    for tenant in tenants:
        grant_tenant_access(db.admin_url, db.role_a.name, tenant, write=True, schema=db.schema)
    store = store_as(db.role_a)
    for tenant in tenants:
        _insert(store, f"{tenant or 'public'}.txt", tenant)


async def test_read_only_role_sees_the_same_as_the_writer(rls_db, store_as, navigator_as) -> None:
    await _seed(rls_db, store_as, "t1")
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t1", schema=rls_db.schema)
    as_a = await _paths(await navigator_as(rls_db.role_a), tenant_id="t1")
    as_b = await _paths(await navigator_as(rls_db.role_b), tenant_id="t1")
    assert as_a == as_b and "t1.txt" in as_b


async def test_read_only_role_cannot_delete(rls_db, store_as) -> None:
    await _seed(rls_db, store_as, "t1")
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t1", schema=rls_db.schema)
    store = store_as(rls_db.role_b)
    with pytest.raises(TenantAccessError, match="read-only access to tenant 't1'"):
        store.deletion.fetch_document_keys(DocumentSelector(tenant_id="t1", path="t1.txt"))
    assert _doc_count(rls_db, "t1") >= 1


async def test_writer_can_delete(rls_db, store_as) -> None:
    await _seed(rls_db, store_as, "t1")
    keys, _, _ = store_as(rls_db.role_a).deletion.fetch_document_keys(
        DocumentSelector(tenant_id="t1", path="t1.txt")
    )
    assert keys


async def test_a_tenant_not_granted_reads_as_empty(rls_db, store_as, navigator_as) -> None:
    await _seed(rls_db, store_as, "t1", "t2")
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t2", schema=rls_db.schema)
    assert await _paths(await navigator_as(rls_db.role_b), tenant_id="t1") == set()


async def test_no_tenant_means_only_the_granted_ones(rls_db, store_as, navigator_as) -> None:
    await _seed(rls_db, store_as, "t1", "t2")
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t2", schema=rls_db.schema)
    assert await _paths(await navigator_as(rls_db.role_b)) == {"t2.txt"}


# ── 9, 10: public data and revocation ────────────────────────────────────────


async def test_public_data_needs_an_explicit_grant(rls_db, store_as, navigator_as) -> None:
    await _seed(rls_db, store_as, "t1", None)
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t1", schema=rls_db.schema)
    navigator = await navigator_as(rls_db.role_b)
    assert await _paths(navigator) == {"t1.txt"}
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, None, schema=rls_db.schema)
    assert await _paths(navigator) == {"t1.txt", "public.txt"}
    assert [
        (g.tenant_id, g.can_write)
        for g in list_tenant_access(rls_db.admin_url, role=rls_db.role_b.name, schema=rls_db.schema)
    ] == [(None, False), ("t1", False)]


async def test_revoke_takes_effect_at_once(rls_db, store_as, navigator_as) -> None:
    await _seed(rls_db, store_as, "t1")
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t1", schema=rls_db.schema)
    navigator = await navigator_as(rls_db.role_b)
    assert await _paths(navigator, tenant_id="t1") == {"t1.txt"}
    revoke_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t1", schema=rls_db.schema)
    revoke_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t1", schema=rls_db.schema)  # no-op
    assert await _paths(navigator, tenant_id="t1") == set()


def test_grant_again_changes_the_write_flag(rls_db) -> None:
    for write in (False, True):
        grant_tenant_access(
            rls_db.admin_url, rls_db.role_b.name, "t1", write=write, schema=rls_db.schema
        )
        (grant,) = list_tenant_access(rls_db.admin_url, schema=rls_db.schema)
        assert grant.can_write is write and grant.role == rls_db.role_b.name


# ── 11, 12, 13: catalog and hot indexes ──────────────────────────────────────


async def test_catalog_is_shared(rls_db, store_as, navigator_as) -> None:
    async with store_as(rls_db.role_a).annotation.catalog_sink() as sink:
        await sink.merge_catalog_model("SharedModel", "visible to everybody", True)
    names = {m.name for m in await (await navigator_as(rls_db.role_b)).list_catalog_models()}
    assert "SharedModel" in names


def _index_definitions(db, schema: str) -> dict[str, str]:
    with psycopg.connect(db.admin_url) as connection:
        rows = connection.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = %s AND indexname LIKE 'model_instances_key_%%'",
            (schema,),
        ).fetchall()
    return {name: definition.replace(schema, "<schema>") for name, definition in rows}


async def test_hot_index_is_the_one_of_simple_mode(rls_db, store_as, pg_url, pg_schema) -> None:
    async with store_as(rls_db.role_a).annotation.catalog_sink() as sink:
        await sink.ensure_instance_key_indexes(["batch_no"])
    assert rls_enabled()
    created = _index_definitions(rls_db, rls_db.schema)
    assert set(created) == {"model_instances_key_batch_no"}

    from scinr.newton import configure
    from scinr.newton.graph import get_graph_store, reset_graph_store
    from scinr.newton.sql.engine import reset_engines

    reset_graph_store()
    reset_engines()
    drop_pg_schema(pg_url, pg_schema)
    configure(
        graph_backend="postgres",
        storage_backend="postgres",
        freeze_backend="postgres",
        sql_url=pg_url,
        sql_schema=pg_schema,
    )
    try:
        simple = get_graph_store()
        simple.schema.setup()
        async with simple.annotation.catalog_sink() as sink:
            await sink.ensure_instance_key_indexes(["batch_no"])
        assert not rls_enabled()
        expected = _index_definitions(rls_db, pg_schema)
        await simple.aclose()
    finally:
        reset_graph_store()
        reset_engines()
        drop_pg_schema(pg_url, pg_schema)
    assert created == expected


def test_the_index_function_refuses_anything_but_a_property_name(rls_db) -> None:
    before = _index_definitions(rls_db, rls_db.schema)
    with psycopg.connect(rls_db.url_for(rls_db.role_a)) as connection:
        with pytest.raises(psycopg.Error) as caught:
            connection.execute(
                sql.SQL("SELECT {}.newton_ensure_instance_key_index(%s)").format(
                    sql.Identifier(rls_db.schema)
                ),
                ("x; drop table documents",),
            )
    assert caught.value.sqlstate == "22023"
    assert _index_definitions(rls_db, rls_db.schema) == before
    assert _admin_scalar(rls_db, "SELECT count(*) FROM {}.documents") == 0  # table still there


def test_runtime_role_cannot_run_ddl_or_edit_grants(rls_db) -> None:
    with psycopg.connect(rls_db.url_for(rls_db.role_a), autocommit=True) as connection:
        for statement in (
            "CREATE INDEX rogue ON {}.model_instances (tenant_id)",
            "INSERT INTO {}.newton_tenant_access (role_name, tenant_id, can_write) "
            "VALUES (current_user, 't1', true)",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                connection.execute(sql.SQL(statement).format(sql.Identifier(rls_db.schema)))


# ── 14, 15: execute_raw ──────────────────────────────────────────────────────


async def test_execute_raw_sees_only_granted_rows(rls_db, store_as, navigator_as) -> None:
    await _seed(rls_db, store_as, "t1", "t2")
    grant_tenant_access(rls_db.admin_url, rls_db.role_b.name, "t2", schema=rls_db.schema)
    rows = await (await navigator_as(rls_db.role_b)).execute_raw("SELECT tenant_id FROM documents")
    assert {row["tenant_id"] for row in rows} == {"t2"}


async def test_execute_raw_cannot_write(rls_db, navigator_as) -> None:
    navigator = await navigator_as(rls_db.role_a)
    with pytest.raises(NavigationError, match="read-only"):
        await navigator.execute_raw(
            "INSERT INTO documents (_key, tenant_id, path, version) VALUES ('k', 't1', 'p', 1)"
        )


# ── 16, 17: schema version and secrets ───────────────────────────────────────


def test_rls_schema_behind_this_newton_is_not_migrated(rls_db, connect) -> None:
    with psycopg.connect(rls_db.admin_url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("UPDATE {}.newton_schema SET version = 1").format(sql.Identifier(rls_db.schema))
        )
    connect(rls_db.role_a)
    with pytest.raises(
        GraphConnectionError,
        match=r"schema version 1 < \d+; an administrator must run setup_sql_schema\(",
    ):
        ensure_schema()
    assert _admin_scalar(rls_db, "SELECT version FROM {}.newton_schema") == 1  # no DDL ran


def test_a_failed_admin_connection_never_shows_the_password(rls_db) -> None:
    secret = "Sup3r-Secret-Pa55"
    bad_url = with_credentials(rls_db.admin_url, "newton_admin", secret)
    for call in (
        lambda: setup_sql_schema(bad_url, schema=rls_db.schema),
        lambda: grant_tenant_access(bad_url, rls_db.role_a.name, "t1", schema=rls_db.schema),
        lambda: revoke_tenant_access(bad_url, rls_db.role_a.name, "t1", schema=rls_db.schema),
        lambda: list_tenant_access(bad_url, schema=rls_db.schema),
    ):
        with pytest.raises(GraphConnectionError) as caught:
            call()
        assert secret not in str(caught.value) and secret not in repr(caught.value)
        assert caught.value.__cause__ is None
