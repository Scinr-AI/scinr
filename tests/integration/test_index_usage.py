"""
tests/integration/test_index_usage.py — The multi-tenant queries CAN use their indexes.

Not a performance test: it measures no times and needs no data volume. It
proves that the planner is able to answer each representative query from the
expected index, which is what stops a tenant-scoped read from scanning the
whole graph / collection.

- **Neo4j**: each query is run as ``EXPLAIN <query>`` with a
  ``USING INDEX <expected index>`` hint. A hint the planner cannot honour is an
  error, so the test fails when the index is missing or the query shape
  (missing label, missing predicate on a composite property…) prevents its use.
  The navigation queries are captured from the real navigator (its ``_read``
  is replaced by a spy), and the ``graph_mapper`` writes from the real
  ``_write_model_fields`` with a mocked session — no query text is duplicated
  here except the ingestion replacement lookup, which is inline in a sync
  driver call.
- **MongoDB**: a throwaway ``scinr_it_<uuid>`` database. ``get_storage()`` must
  have created the indexes; a few documents are inserted (on an empty
  collection ``explain`` answers ``EOF``, which proves nothing) and
  ``explain`` must show an ``IXSCAN`` on the expected index and no
  ``COLLSCAN``.

Run::

    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=... MONGODB_URI=mongodb://localhost:27017 \\
        pytest -m integration tests/integration/test_index_usage.py

Each backend is skipped when its environment variables are missing or the
server is unreachable. The Neo4j part runs ``setup_schema`` on the target
database (idempotent, the same DDL the pipeline runs) and creates one
``idx_mi_key_scinr_it_key`` index, dropped at the end.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

pytestmark = pytest.mark.integration

TENANT = "scinr_it_tenant"
KEY_PROP = "scinr_it_key"


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


def _neo4j_settings() -> dict[str, str]:
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
    """Configured scinr + a sync driver on a database with the full schema."""
    settings = _neo4j_settings()
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
    _create_key_index(settings)
    _await_indexes_online(driver, settings["neo4j_database"])
    try:
        yield driver, settings["neo4j_database"]
    finally:
        from scinr.newton.annotation.neo4j_ops import instance_key_index_name

        with driver.session(database=settings["neo4j_database"]) as session:
            session.run(f"DROP INDEX {instance_key_index_name(KEY_PROP)} IF EXISTS").consume()
        driver.close()
        config_module._config = previous


def _create_key_index(settings: dict[str, str]) -> None:
    """Create the instance_key index through the real WP3 helper."""
    import asyncio

    from neo4j import AsyncGraphDatabase

    from scinr.newton.annotation.neo4j_ops import _ensure_instance_key_indexes

    async def _run() -> None:
        driver = AsyncGraphDatabase.driver(
            settings["neo4j_uri"], auth=(settings["neo4j_user"], settings["neo4j_password"])
        )
        try:
            async with driver.session(database=settings["neo4j_database"]) as session:
                await _ensure_instance_key_indexes(session, {KEY_PROP})
        finally:
            await driver.close()

    asyncio.run(_run())


def _await_indexes_online(driver, database: str, timeout: float = 60.0) -> None:
    """Hints only accept ONLINE indexes; new ones start as POPULATING."""
    deadline = time.monotonic() + timeout
    while True:
        with driver.session(database=database) as session:
            pending = session.run(
                "SHOW INDEXES YIELD name, state WHERE state <> 'ONLINE' RETURN name, state"
            ).data()
        if not pending:
            return
        if time.monotonic() > deadline:
            pytest.fail(f"indexes not ONLINE after {timeout}s: {pending}")
        time.sleep(0.5)


_NODE_PATTERN = r"\({alias}:{label}\b[^)]*\)"


def _with_hint(cypher: str, alias: str, label: str, props: tuple[str, ...]) -> str:
    """Insert ``USING INDEX alias:Label(props)`` right after the first
    ``(alias:Label …)`` node pattern of *cypher*."""
    m = re.search(_NODE_PATTERN.format(alias=re.escape(alias), label=re.escape(label)), cypher)
    assert m, f"no ({alias}:{label} …) pattern in: {cypher}"
    rest = cypher[m.end():].lstrip()
    assert not rest.startswith(("-", "<")), "hint helper expects a single-node MATCH pattern"
    prop_list = ", ".join(f"`{p}`" for p in props)
    hint = f" USING INDEX {alias}:{label}({prop_list}) "
    return cypher[: m.end()] + hint + cypher[m.end():]


def _explain(driver, database: str, cypher: str, params: dict[str, Any]) -> None:
    """EXPLAIN *cypher* (never executed); a hint that cannot be used raises."""
    with driver.session(database=database) as session:
        session.run("EXPLAIN " + cypher, **params).consume()


class _CapturingNavigator:
    """Build the real navigator with ``_read`` replaced by a recorder."""

    def __new__(cls):
        from scinr.newton.navigation.neo4j import Neo4jGraphNavigator

        class _Spy(Neo4jGraphNavigator):
            def __init__(self) -> None:
                super().__init__(driver=object())
                self.captured: list[tuple[str, dict[str, Any]]] = []

            async def _read(self, cypher: str, /, **params: Any) -> list[dict[str, Any]]:
                self.captured.append((cypher, params))
                return []

        return _Spy()


async def _captured(method: str, **kwargs: Any) -> list[tuple[str, dict[str, Any]]]:
    nav = _CapturingNavigator()
    await getattr(nav, method)(**kwargs)
    assert nav.captured, f"{method} ran no query"
    return nav.captured


# (navigator method, kwargs, alias, label, index properties)
_NAVIGATION_CASES = [
    # WP2 — composite tenant indexes
    ("get_model_instances_by_class", {"model_class": "X", "tenant_id": TENANT},
     "mi", "ModelInstance", ("tenant_id", "model_class")),
    ("count_model_instances_by_class",
     {"model_class": "X", "tenant_id": TENANT, "include_public": True},
     "mi", "ModelInstance", ("tenant_id", "model_class")),
    ("get_documents", {"tenant_id": TENANT}, "d", "Document", ("tenant_id", "latest")),
    ("list_root_documents", {"tenant_id": TENANT}, "d", "Document", ("tenant_id", "latest")),
    ("count_root_documents", {"tenant_id": TENANT, "include_public": True},
     "d", "Document", ("tenant_id", "latest")),
    ("find_structure_nodes", {"role": "section", "tenant_id": TENANT},
     "n", "StructureNode", ("tenant_id", "role")),
    ("get_labeled_entities", {"label": "CountryName", "tenant_id": TENANT},
     "le", "LabeledEntity", ("tenant_id", "label")),
    # Both MATCHes filter tenant + class in their own WHERE (not after a WITH)
    ("find_shell_model_instances", {"model_class": "X", "tenant_id": TENANT},
     "mi", "ModelInstance", ("tenant_id", "model_class")),
    ("find_shell_model_instances", {"model_class": "X", "tenant_id": TENANT},
     "mi2", "ModelInstance", ("tenant_id", "model_class")),
    # WP3 — a search on part of a composite instance key
    ("get_model_instances_by_class",
     {"model_class": "X", "where": {KEY_PROP: "es"}, "tenant_id": TENANT},
     "mi", "ModelInstance", ("tenant_id", KEY_PROP)),
    # Lookups by id / uid (unique constraints)
    ("get_structure_node", {"node_id": "t::doc::1", "tenant_id": TENANT},
     "n", "StructureNode", ("id",)),
    ("get_structure_nodes_by_ids",
     {"node_ids": ["t::doc::1", "t::doc::2"], "tenant_id": TENANT, "include_public": True},
     "n", "StructureNode", ("id",)),
    ("get_model_instance", {"uid": "abc", "tenant_id": TENANT},
     "mi", "ModelInstance", ("uid",)),
    ("get_info_unit", {"uid": "abc", "tenant_id": TENANT}, "u", "InfoUnit", ("uid",)),
]


@pytest.mark.parametrize(
    ("method", "kwargs", "alias", "label", "props"),
    _NAVIGATION_CASES,
    ids=[f"{c[0]}-{c[2]}-{'_'.join(c[4])}" for c in _NAVIGATION_CASES],
)
async def test_navigation_query_can_use_index(neo4j_env, method, kwargs, alias, label, props):
    driver, database = neo4j_env
    cypher, params = (await _captured(method, **kwargs))[0]
    _explain(driver, database, _with_hint(cypher, alias, label, props), params)


def test_replacement_lookup_can_use_tenant_name_index(neo4j_env):
    """``stages/ingestion.apply_replacement`` resolves the new roots by name."""
    driver, database = neo4j_env
    cypher = (
        "MATCH (d:Document {tenant_id: $tenant_id, latest: true}) "
        "WHERE d.name IN $names AND NOT ()-[:IS_COMPOSED_OF]->(d) RETURN d.path"
    )
    _explain(
        driver,
        database,
        _with_hint(cypher, "d", "Document", ("tenant_id", "name")),
        {"tenant_id": TENANT, "names": ["a", "b"]},
    )


def test_hint_on_unusable_index_is_rejected(neo4j_env):
    """Control: the check is meaningful — a composite index is refused when the
    query leaves one of its properties unconstrained (planner rule 2)."""
    from neo4j.exceptions import Neo4jError

    driver, database = neo4j_env
    cypher = "MATCH (mi:ModelInstance) WHERE mi.tenant_id = $t RETURN mi"
    with pytest.raises(Neo4jError):
        _explain(
            driver,
            database,
            _with_hint(cypher, "mi", "ModelInstance", ("tenant_id", "model_class")),
            {"t": TENANT},
        )


class _ItCountry(BaseModel):
    code: str = Field(json_schema_extra={"instance_key": True})
    name: str = Field(json_schema_extra={"entity_label": "CountryName"})
    aliases: list[str] = Field(default_factory=list, json_schema_extra={"entity_label": "CountryName"})


class _ItRoot(BaseModel):
    country: _ItCountry
    others: list[_ItCountry] = Field(default_factory=list)
    note: str = "x"
    related: list[str] = Field(
        default_factory=list,
        json_schema_extra={"instance_relationships": [
            {"target_model": "_ItCountry", "join_via": {"related": "code"}, "rel_type": "RELATED"}
        ]},
    )


async def test_graph_mapper_writes_find_their_anchor_by_index(neo4j_env):
    """Every parent / relationship-source MATCH in ``_write_model_fields`` is a
    seek on the uid constraint of its label (WP4)."""
    from scinr.newton.entity_extraction.graph_mapper import _write_model_fields

    driver, database = neo4j_env
    session = AsyncMock()
    await _write_model_fields(
        session=session,
        instance=_ItRoot(
            country=_ItCountry(code="ES", name="Spain", aliases=["España"]),
            others=[_ItCountry(code="FR", name="France")],
            related=["PT"],
        ),
        parent_uid="er-uid",
        parent_label="ExtractionResult",
        field_path_prefix="",
        entity_nodes={},
        depth=0,
        tenant_id=TENANT,
    )
    anchor = re.compile(r"MATCH \((parent|src):(\w+) \{uid:")
    checked = 0
    for call in session.run.call_args_list:
        cypher = call.args[0]
        m = anchor.search(cypher)
        if not m:
            continue
        alias, label = m.group(1), m.group(2)
        _explain(driver, database, _with_hint(cypher, alias, label, ("uid",)), dict(call.kwargs))
        checked += 1
    # parent MATCHes (child MERGE/CREATE, list items, REFERENCES, batch SET) + src
    assert checked >= 6, f"only {checked} anchored writes captured"


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mongo_env():
    """Throwaway database, indexes created by get_storage(), a few documents."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        client.close()
        pytest.skip(f"MongoDB not reachable: {exc}")

    from scinr.newton import config as config_module
    from scinr.newton.config import configure

    previous = config_module._config
    db_name = f"scinr_it_{uuid.uuid4().hex[:12]}"
    cfg = configure(
        neo4j_user="neo4j",
        neo4j_password="unused",
        neo4j_database="unused",
        storage_backend="mongodb",
        mongodb_uri=uri,
        mongodb_database=db_name,
    )
    try:
        from scinr.newton.storage.factory import get_storage

        get_storage()  # WP1: must create the indexes by itself
        db = client[db_name]
        _seed(db, cfg)
        yield db, cfg
    finally:
        client.drop_database(db_name)
        client.close()
        config_module._config = previous


def _seed(db, cfg) -> None:
    raw_files = db[cfg.mongodb_raw_files_collection]
    pages = db[cfg.mongodb_pages_collection]
    for t in (TENANT, "__public__", "other_tenant"):
        for f in range(3):
            raw_id = f"{t}-raw-{f}"
            raw_files.insert_one({
                "_id": raw_id, "tenant_id": t, "created_by_user_id": f"user-{f % 2}",
                "job_id": f"job-{f}", "checksum_sha256": f"sha-{t}-{f}",
                "filename": f"f{f}.pdf", "folder_path": "", "stored_at": f,
            })
            pages.insert_many([
                {"tenant_id": t, "raw_file_id": raw_id, "page_index": i,
                 "filename": f"f{f}.pdf", "folder_path": ""}
                for i in range(4)
            ])


def _stages(plan: Any) -> list[dict[str, Any]]:
    """Every stage dict nested anywhere in an explain plan."""
    out: list[dict[str, Any]] = []
    if isinstance(plan, dict):
        if "stage" in plan:
            out.append(plan)
        for value in plan.values():
            out.extend(_stages(value))
    elif isinstance(plan, list):
        for value in plan:
            out.extend(_stages(value))
    return out


def _assert_index_scan(explain: dict[str, Any], index_name: str, *, allow_sort: bool = False):
    winning = explain["queryPlanner"]["winningPlan"]
    stages = _stages(winning)
    names = {s["stage"] for s in stages}
    assert "COLLSCAN" not in names, winning
    assert "EOF" not in names, "empty collection: explain proves nothing"
    assert any(s["stage"] == "IXSCAN" and s.get("indexName") == index_name for s in stages), (
        f"expected IXSCAN on {index_name}: {winning}"
    )
    if not allow_sort:
        assert "SORT" not in names, f"blocking SORT stage: {winning}"


def test_get_storage_created_the_indexes(mongo_env):
    from scinr.newton.storage.mongodb.client import _INDEX_SPECS

    db, cfg = mongo_env
    for coll_attr, keys, name in _INDEX_SPECS:
        info = db[getattr(cfg, coll_attr)].index_information()
        assert name in info, f"{name} missing on {getattr(cfg, coll_attr)}"
        assert info[name]["key"] == keys


@pytest.mark.parametrize("include_public", [False, True])
def test_pages_by_raw_file_sorted_use_index(mongo_env, include_public):
    from scinr.newton.storage.filters import mongo_scope_filter

    db, cfg = mongo_env
    query = {"raw_file_id": f"{TENANT}-raw-1", **mongo_scope_filter(TENANT, include_public)}
    explain = db[cfg.mongodb_pages_collection].find(query).sort("page_index", 1).explain()
    _assert_index_scan(explain, "pages_by_tenant_raw_file_and_index")


def test_pages_by_ids_use_id_index(mongo_env):
    from scinr.newton.storage.filters import mongo_scope_filter

    db, cfg = mongo_env
    ids = [d["_id"] for d in db[cfg.mongodb_pages_collection].find({"tenant_id": TENANT}).limit(3)]
    query = {"_id": {"$in": ids}, **mongo_scope_filter(TENANT)}
    explain = db[cfg.mongodb_pages_collection].find(query).sort("page_index", 1).explain()
    # A handful of pages sorted in memory after the _id seek: the sort is expected.
    _assert_index_scan(explain, "_id_", allow_sort=True)


@pytest.mark.parametrize(
    ("scope", "index_name"),
    [
        ({"created_by_user_id": "user-1"}, "raw_files_by_tenant_user"),
        ({"job_id": "job-2"}, "raw_files_by_tenant_job"),
    ],
)
def test_list_raw_files_uses_index(mongo_env, scope, index_name):
    from scinr.newton.storage.filters import mongo_scope_filter

    db, cfg = mongo_env
    query = mongo_scope_filter(TENANT, **scope)
    explain = db[cfg.mongodb_raw_files_collection].find(query).sort("stored_at", 1).explain()
    # list_raw_files orders by stored_at, which no index covers: the (small,
    # per user / per job) result is sorted in memory.
    _assert_index_scan(explain, index_name, allow_sort=True)


def test_delete_pages_uses_index(mongo_env):
    from scinr.newton.storage.filters import mongo_scope_filter

    db, cfg = mongo_env
    query = {"raw_file_id": f"{TENANT}-raw-2", **mongo_scope_filter(TENANT)}
    # explain on a delete plans it without deleting anything.
    explain = db.command(
        "explain",
        {"delete": cfg.mongodb_pages_collection, "deletes": [{"q": query, "limit": 0}]},
        verbosity="queryPlanner",
    )
    _assert_index_scan(explain, "pages_by_tenant_raw_file_and_index")
