"""
tests/integration/test_same_row_linking_neo4j.py — Same-row instance
linking and its repair utility against a real Neo4j.

1. ``write_extraction_subgraph`` links a same-row sibling in the database (the
   real Cypher of the sibling MERGE), and keeps the key-addressed shell for a
   cross-row reference.
2. ``run_relink`` on a graph written the old way (relationships to dead
   shells): moves them to the sibling, deletes the shells, leaves unmatched
   references and other tenants alone, and does nothing on a second run.

Run::

    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=... \\
        pytest -m integration tests/integration/test_same_row_linking_neo4j.py

Skipped when the environment variables are missing or the server is
unreachable. Everything it writes is under throw-away tenants and
``ScinrIt*`` catalog names, deleted at the end.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest
from pydantic import BaseModel, Field

pytestmark = pytest.mark.integration

RUN = uuid.uuid4().hex[:8]
TENANT = f"scinr_it_relink_{RUN}"
OTHER_TENANT = f"scinr_it_relink_other_{RUN}"
SRC, TGT, REL = "ScinrItSpec", "ScinrItRecord", "SCINR_IT_SPEC_FOR"


def _settings() -> dict[str, str]:
    if not os.getenv("NEO4J_URI") or not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_URI / NEO4J_PASSWORD not set")
    return {
        "neo4j_uri": os.environ["NEO4J_URI"],
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ["NEO4J_PASSWORD"],
        "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


@pytest.fixture(scope="module")
def neo4j_env():
    settings = _settings()
    from neo4j import GraphDatabase

    from scinr.newton import config as config_module
    from scinr.newton.config import configure

    previous = config_module._config
    configure(**settings)
    driver = GraphDatabase.driver(
        settings["neo4j_uri"], auth=(settings["neo4j_user"], settings["neo4j_password"])
    )
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        driver.close()
        config_module._config = previous
        pytest.skip(f"Neo4j not reachable: {exc}")
    from scinr.newton.ingest.schema import setup_schema

    setup_schema(driver)
    db = settings["neo4j_database"]
    try:
        yield driver, db, settings
    finally:
        with driver.session(database=db) as s:
            s.run(
                "MATCH (n) WHERE n.tenant_id IN $tenants DETACH DELETE n",
                tenants=[TENANT, OTHER_TENANT],
            ).consume()
            s.run(
                "MATCH (c:CatalogModel) WHERE c.name IN $names DETACH DELETE c",
                names=[SRC, TGT],
            ).consume()
        driver.close()
        config_module._config = previous


def _q(driver, db, query, **params):
    with driver.session(database=db) as s:
        return s.run(query, **params).data()


# ---------------------------------------------------------------------------
# 1. The writer
# ---------------------------------------------------------------------------


class ScinrItRecord(BaseModel):
    portfolio_record_id: str | None = None


class ScinrItSpec(BaseModel):
    portfolio_record_id: str | None = Field(
        default=None,
        json_schema_extra={
            "instance_relationships": [
                {
                    "target_model": TGT,
                    "join_via": {"portfolio_record_id": "portfolio_record_id"},
                    "rel_type": REL,
                }
            ]
        },
    )


def test_writer_links_the_same_row_sibling_in_the_database(neo4j_env) -> None:
    driver, db, settings = neo4j_env
    from neo4j import AsyncGraphDatabase

    from scinr.newton.entity_extraction.graph_mapper import write_extraction_subgraph
    from scinr.newton.entity_extraction.schema_composer import compose_extraction_schema

    node_id = f"{TENANT}::it/v1/table_1/row_1"
    _q(
        driver,
        db,
        "CREATE (:StructureNode {id: $id, tenant_id: $t, created_by_user_id: 'u', job_id: 'j'})",
        id=node_id,
        t=TENANT,
    )
    composite_cls = compose_extraction_schema(ScinrItRecord, [ScinrItSpec])
    composite = composite_cls.model_construct(
        scinr_it_record=ScinrItRecord(portfolio_record_id="312947; CANTABNEW-0081"),
        scinr_it_spec=ScinrItSpec(portfolio_record_id="312947"),
    )

    async def _write() -> None:
        adriver = AsyncGraphDatabase.driver(
            settings["neo4j_uri"], auth=(settings["neo4j_user"], settings["neo4j_password"])
        )
        try:
            await write_extraction_subgraph(
                driver=adriver,
                node_full_id=node_id,
                composite_instance=composite,
                primary_model_class=TGT,
                complementary_model_classes=[SRC],
                document_name="it",
                extraction_uid=f"it_er_{RUN}",
            )
        finally:
            await adriver.close()

    asyncio.run(_write())
    rows = _q(
        driver,
        db,
        f"""
        MATCH (er:ExtractionResult {{uid: $er}})-->(spec:ModelInstance {{model_class: $src}})
              -[r:`{REL}`]->(tgt:ModelInstance)
        RETURN tgt.model_class AS cls, r.resolution AS resolution,
               EXISTS {{ (er)-->(tgt) }} AS sibling,
               COUNT {{ (s:ModelInstance {{tenant_id: $t, model_class: $tgt}}) }} AS records
        """,
        er=f"it_er_{RUN}",
        src=SRC,
        tgt=TGT,
        t=TENANT,
    )
    assert rows == [{"cls": TGT, "resolution": "sibling", "sibling": True, "records": 1}]


# ---------------------------------------------------------------------------
# 2. The repair utility
# ---------------------------------------------------------------------------


def _seed_old_graph(driver, db, tenant: str) -> None:
    """Rows written before sibling resolution: Spec -[REL]-> dead shell.

    Rows 0-2 have a key-less Record sibling; row 3 has none (cross-document
    reference, must stay).
    """
    rows = [
        {"i": 0, "record": "312947; CANTABNEW-0081", "ref": "312947"},
        {"i": 1, "record": "400001; CANTABNEW-0099", "ref": "cantabnew-0099"},
        {"i": 2, "record": "500002; CANTABNEW-0100", "ref": "unrelated"},
        {"i": 3, "record": None, "ref": "elsewhere"},
    ]
    _q(
        driver,
        db,
        f"""
        UNWIND $rows AS row
        CREATE (er:ExtractionResult {{uid: $t + '_er' + row.i, tenant_id: $t}})
        CREATE (er)-[:HAS_SCINR_IT_SPEC]->(spec:ModelInstance {{
            uid: $t + '_spec' + row.i, tenant_id: $t, model_class: $src,
            portfolio_record_id: row.ref}})
        CREATE (spec)-[:`{REL}`]->(:ModelInstance {{
            uid: $t + '_shell' + row.i, tenant_id: $t, model_class: $tgt,
            portfolio_record_id: row.ref}})
        WITH er, row WHERE row.record IS NOT NULL
        CREATE (er)-[:HAS_SCINR_IT_RECORD]->(:ModelInstance {{
            uid: $t + '_record' + row.i, tenant_id: $t, model_class: $tgt,
            portfolio_record_id: row.record}})
        """,
        rows=rows,
        t=tenant,
        src=SRC,
        tgt=TGT,
    )


def test_run_relink_repairs_dead_shells_and_is_idempotent(neo4j_env) -> None:
    driver, db, _ = neo4j_env
    from scinr.newton.ingest.relink import run_relink

    _q(
        driver,
        db,
        f"""
        MERGE (s:CatalogModel {{name: $src}})
        MERGE (t:CatalogModel {{name: $tgt}})
        MERGE (s)-[r:`{REL}` {{via_field: 'portfolio_record_id', join_via: $jv}}]->(t)
        """,
        src=SRC,
        tgt=TGT,
        jv=json.dumps({"portfolio_record_id": "portfolio_record_id"}),
    )
    _seed_old_graph(driver, db, TENANT)
    _seed_old_graph(driver, db, OTHER_TENANT)

    dry = run_relink(driver, tenant_id=TENANT, database=db, dry_run=True)
    mine = dry["by_relationship"][f"{SRC}-[{REL}]->{TGT}"]
    assert mine == {"relinked": 3, "unmatched": 1, "shell_relationships": 4}
    assert dry["shells_deleted"] == 3

    first = run_relink(driver, tenant_id=TENANT, database=db)
    assert first["by_relationship"][f"{SRC}-[{REL}]->{TGT}"] == mine
    assert first["ambiguous"] == 0

    links = _q(
        driver,
        db,
        f"""
        MATCH (spec:ModelInstance {{tenant_id: $t, model_class: $src}})-[r:`{REL}`]->(tgt)
        RETURN spec.uid AS spec, tgt.uid AS tgt, r.resolution AS resolution ORDER BY spec
        """,
        t=TENANT,
        src=SRC,
    )
    assert links == [
        {"spec": f"{TENANT}_spec0", "tgt": f"{TENANT}_record0", "resolution": "sibling"},
        {"spec": f"{TENANT}_spec1", "tgt": f"{TENANT}_record1", "resolution": "sibling"},
        {"spec": f"{TENANT}_spec2", "tgt": f"{TENANT}_record2", "resolution": "sibling"},
        {"spec": f"{TENANT}_spec3", "tgt": f"{TENANT}_shell3", "resolution": None},
    ]
    shells = _q(
        driver,
        db,
        "MATCH (s:ModelInstance) WHERE s.tenant_id = $t AND s.uid CONTAINS '_shell' "
        "RETURN s.uid AS uid",
        t=TENANT,
    )
    assert shells == [{"uid": f"{TENANT}_shell3"}]

    # The other tenant is untouched.
    other = _q(
        driver,
        db,
        f"MATCH (:ModelInstance {{tenant_id: $t}})-[r:`{REL}`]->(s) "
        "WHERE s.uid CONTAINS '_shell' RETURN count(r) AS n",
        t=OTHER_TENANT,
    )
    assert other == [{"n": 4}]

    second = run_relink(driver, tenant_id=TENANT, database=db)
    assert (second["relinked"], second["shells_deleted"]) == (0, 0)
    assert second["by_relationship"][f"{SRC}-[{REL}]->{TGT}"] == {
        "unmatched": 1,
        "shell_relationships": 1,
    }
