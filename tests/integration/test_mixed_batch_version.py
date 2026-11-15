"""
tests/integration/test_mixed_batch_version.py — a mixed tabular + document run
uses ONE batch version.

Regression test for the split-version bug: ``run_pipeline`` used to resolve the
tabular version and the document version separately, so the ancestor folder
shared by both ended up as two ``:Document`` nodes (versions N and N + 1).

Real Neo4j, real ``run_pipeline`` and real ingestion. Only the LLM-bound pieces
are replaced: the tabular agent (no LLM mapping) and the document conversion /
extraction (no LLM extraction; ``extract_one_file`` returns a minimal
``Document``). Raw-file storage is disabled.

Run (use a scratch database: the test writes ``scinr_it_*`` documents and
deletes them at the end)::

    NEO4J_URI=bolt://localhost:7698 NEO4J_PASSWORD=... \\
        .venv/bin/python -m pytest -m integration tests/integration/test_mixed_batch_version.py
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.integration

TENANT = "scinr_it_mixed"


@pytest.fixture
def neo4j_driver():
    if not os.getenv("NEO4J_URI") or not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_URI / NEO4J_PASSWORD not set")
    from neo4j import GraphDatabase

    from scinr.newton import config as config_module
    from scinr.newton.config import configure

    settings = {
        "neo4j_uri": os.environ["NEO4J_URI"],
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ["NEO4J_PASSWORD"],
        "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }
    previous = config_module._config
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    configure(llm=FakeListChatModel(responses=["{}"]), **settings)
    driver = GraphDatabase.driver(
        settings["neo4j_uri"], auth=(settings["neo4j_user"], settings["neo4j_password"])
    )
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        driver.close()
        config_module._config = previous
        pytest.skip(f"Neo4j not reachable: {exc}")

    database = settings["neo4j_database"]

    def _cleanup() -> None:
        with driver.session(database=database) as session:
            session.run(
                "MATCH (d:Document {tenant_id: $t}) DETACH DELETE d", t=TENANT
            ).consume()

    _cleanup()
    try:
        yield driver, database
    finally:
        _cleanup()
        driver.close()
        config_module._config = previous


async def test_mixed_run_shares_one_version_and_one_folder(tmp_path, monkeypatch, neo4j_driver):
    from scinr.newton.models.document_structure import Document
    from scinr.newton.pipeline import run_pipeline

    driver, database = neo4j_driver

    folder = tmp_path / "A"
    folder.mkdir()
    (folder / "a.txt").write_text("hello", encoding="utf-8")
    (folder / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    async def _extract_one_intermediate(doc, output_path, fast_extraction=False, **kw):
        return Document(
            document_name="a.txt",
            doc_path="A/a.txt",
            raw_file_id="",
            tenant_id=kw.get("tenant_id"),
        )

    monkeypatch.setattr(
        "scinr.newton.stages.extraction.extract_one_intermediate",
        AsyncMock(side_effect=_extract_one_intermediate),
    )
    # The real tabular agent creates the :Document + folder hierarchy with the
    # batch version; only its LLM-bound LangGraph is replaced.
    fake_graph = type(
        "G",
        (),
        {"ainvoke": AsyncMock(return_value={"errors": [], "ingested_table_node_ids": ["t-1"]})},
    )()
    monkeypatch.setattr("scinr.newton.tabular.agent.tabular_graph", fake_graph)

    async def _run() -> None:
        res = await run_pipeline(
            stages=["preprocess", "extraction", "ingestion"],
            input_raw=str(tmp_path),
            tenant_id=TENANT,
        )
        assert res.ingestion.success, res.ingestion.documents[0].errors

    def _versions() -> dict[str, list[int]]:
        with driver.session(database=database) as session:
            rows = session.run(
                "MATCH (d:Document {tenant_id: $t}) RETURN d.path AS path, d.version AS version",
                t=TENANT,
            )
            out: dict[str, list[int]] = {}
            for r in rows:
                out.setdefault(r["path"], []).append(r["version"])
        return out

    await _run()
    first = _versions()
    assert first["A"] == [1], first  # one folder node, not one per half of the run
    assert first["A/b.csv"] == [1], first
    assert first["A/a.txt"] == [1], first

    await _run()
    second = _versions()
    assert sorted(second["A"]) == [1, 2], second
    assert sorted(second["A/b.csv"]) == [1, 2], second
    assert sorted(second["A/a.txt"]) == [1, 2], second
