"""
tests/unit/test_rls_migration.py -- The RLS migration file is generated from ``sql/tables.py``.

Adding a table with a ``tenant_id`` without regenerating (``scripts/gen_rls_migration.py``)
would leave it without policies: this test fails first.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scinr.newton.sql.migrations import runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "gen_rls_migration", ROOT / "scripts" / "gen_rls_migration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_file_is_what_the_generator_produces(generator) -> None:
    assert generator.TARGET.read_text(encoding="utf-8") == generator.render(), (
        "run .venv/bin/python scripts/gen_rls_migration.py"
    )


def test_every_tenant_table_has_its_four_policies(generator) -> None:
    text = generator.TARGET.read_text(encoding="utf-8")
    tables = generator.tenant_tables()
    assert {"documents", "edges", "raw_file_chunks", "frozen_snapshots", "pages"} <= set(tables)
    assert not {"catalog_models", "catalog_edges", "newton_schema", "all_edges"} & set(tables)
    for table in tables:
        assert f"ALTER TABLE {{schema}}.{table} FORCE ROW LEVEL SECURITY" in text
        for policy in ("tenant_read", "tenant_ins", "tenant_upd", "tenant_del"):
            assert f"CREATE POLICY {policy} ON {{schema}}.{table} " in text


def test_the_rls_track_follows_the_migration_naming_rules() -> None:
    directory = runner.migrations_dir("postgres") / runner.RLS_FOLDER
    assert [n for n, _ in runner.list_migrations(directory)] == [1]
    # The normal track does not pick the RLS folder up.
    assert runner.latest_version("postgres") >= 2


def test_the_index_function_uses_the_name_and_expression_of_simple_mode(generator) -> None:
    from scinr.newton.sql.dialect import PostgresDialect

    simple = PostgresDialect().instance_key_index_sql("p", schema="s")
    assert '"model_instances_key_p"' in simple and "(tenant_id, (props->>'p'))" in simple
    assert "'model_instances_key_' || prop" in generator.INDEX_FUNCTION
    assert "(tenant_id, (props ->> %L))" in generator.INDEX_FUNCTION
