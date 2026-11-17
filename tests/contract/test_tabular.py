"""
tests/contract/test_tabular.py -- ``store.tabular``.

The sink is opened by ``write_tabular_subgraph``; the tests drive it through the tabular
pipeline (scripted LLM), which is its only caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scinr.newton.stages import run_tabular_pipeline
from tests.golden._scenarios import TENANT_A, TENANT_B, USER

from ._helpers import labels, rels, twice

CSV = "sku,item,qty,x_note\nS1,Bolt,10,a\nS2,Nut,20,b\n"
RAW_CSV = "alpha,beta\n1,2\n3,4\n5,6\n"


def write_csv(tmp: Path, name: str, text: str) -> Path:
    root = tmp / "tab"
    root.mkdir(exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")
    return root


async def ingest(root: Path, tenant: str | None, *, update_mode: bool) -> None:
    result = await run_tabular_pipeline(
        input_raw=str(root),
        update_mode=update_mode,
        tenant_id=tenant,
        created_by_user_id=USER,
        job_id="job-tab",
    )
    assert result.success, [d.errors for d in result.documents]


class TestTabularSink:
    async def test_writes_table_rows_and_is_idempotent_in_update_mode(
        self, graph_store, dump, tmp_path
    ):
        root = write_csv(tmp_path, "plain.csv", CSV)
        await ingest(root, TENANT_A, update_mode=False)
        first = dump()
        assert len(labels(first, "Table")) == 1
        assert len(labels(first, "Row")) == 2
        assert rels(first, "HAS_STRUCTURE")

        # Update mode wipes the sheet (delete_tabular_subgraph) and writes it again.
        async def run():
            await ingest(root, TENANT_A, update_mode=True)

        graph = await twice(dump, run, ignore=("ModelInstance",), deterministic=False)
        assert len(labels(graph, "Table")) == 1
        assert len(labels(graph, "Row")) == 2

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known leak, not in plan section 7: update_mode on a tabular file deletes the "
            "Table and Rows but leaves their ModelInstance nodes behind (orphans for the GC "
            "to collect); the golden baseline contains them. Fix in a separate change."
        ),
    )
    async def test_update_mode_leaves_no_orphan_model_instances(self, graph_store, dump, tmp_path):
        root = write_csv(tmp_path, "plain.csv", CSV)
        await ingest(root, TENANT_A, update_mode=False)
        before = len(labels(dump(), "ModelInstance"))
        await ingest(root, TENANT_A, update_mode=True)
        assert len(labels(dump(), "ModelInstance")) == before

    async def test_raw_rows_without_a_model(self, graph_store, dump, tmp_path):
        root = write_csv(tmp_path, "raw.csv", RAW_CSV)
        await ingest(root, TENANT_A, update_mode=False)

        async def run():
            await ingest(root, TENANT_A, update_mode=True)

        graph = await twice(dump, run, ignore=("ModelInstance",), deterministic=False)
        assert len(labels(graph, "Row")) == 3
        assert labels(graph, "ExtractionResult")


class TestDeleteTabularSubgraph:
    async def test_deletes_only_the_sheet_of_the_tenant_and_is_idempotent(
        self, graph_store, dump, tmp_path
    ):
        root = write_csv(tmp_path, "plain.csv", CSV)
        await ingest(root, TENANT_A, update_mode=False)
        await ingest(root, TENANT_B, update_mode=False)
        assert len(labels(dump(), "Table")) == 2

        async def run():
            await graph_store.tabular.delete_tabular_subgraph("plain.csv", 1, 0, tenant_id=TENANT_A)

        graph = await twice(dump, run, ignore=("ModelInstance",), deterministic=False)
        assert len(labels(graph, "Table")) == 1
        assert len(labels(graph, "Row")) == 2
        assert all(TENANT_B in n["key"] for n in labels(graph, "Table"))
        # The documents stay.
        assert {d["props"]["tenant_id"] for d in labels(graph, "Document")} == {TENANT_A, TENANT_B}
