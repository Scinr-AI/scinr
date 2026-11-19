"""
tests/sql/test_dump_and_invariants.py -- Neutral dump, invariant checks and schema behaviour.

Rows are written directly with the table objects (no store exists yet), read back through
``row_to_node`` by ``dump_sql`` and checked by the invariant functions, on both engines.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from scinr.newton.sql import tables
from scinr.newton.sql.dialect import get_dialect
from scinr.newton.sql.engine import get_sync_engine
from scinr.newton.sql.keys import encode_key
from scinr.newton.sql.names import KEY_COLUMN, LABEL_TABLE, edge_ident
from scinr.newton.sql.rows import props_to_row
from scinr.newton.sql.schema import ensure_schema
from tests.sql import _invariants as inv
from tests.sql._dump import dump_sql

TENANT = "acme"
DOC = {"tenant_id": TENANT, "path": "a/b.pdf", "version": 1}
DOC_KEY = encode_key("Document", DOC)
NODE = {"id": f"{TENANT}|a/b.pdf|1|n1", "tenant_id": TENANT, "role": "section", "title": "Intro"}


@pytest.fixture
def db(sql_backend):
    """The configured sync engine, schema created, and a helper to write nodes and edges."""
    ensure_schema()
    engine = get_sync_engine()
    dialect = get_dialect(sql_backend.url)

    class Db:
        target = sql_backend
        url = sql_backend.url
        schema = sql_backend.schema

        @staticmethod
        def node(label: str, props: dict, **aux) -> str:
            name = LABEL_TABLE[label]
            row = props_to_row(name, props)
            key = encode_key(label, props)
            if KEY_COLUMN[label] == "_key":
                row["_key"] = key
            row.update(aux)
            with engine.begin() as connection:
                connection.execute(dialect.insert(tables.metadata.tables[name]).values(**row))
            return key

        @staticmethod
        def edge(rel_type, src, dst, tenant=TENANT, **props) -> None:
            table = tables.edges if tenant is not None else tables.catalog_edges
            values = {
                "type": rel_type,
                "src_label": src[0],
                "src_key": src[1],
                "dst_label": dst[0],
                "dst_key": dst[1],
                "_ident": edge_ident(rel_type, props),
                "props": props,
            }
            if tenant is not None:
                values["tenant_id"] = tenant
            with engine.begin() as connection:
                connection.execute(dialect.insert(table).values(**values))

        @staticmethod
        def connection():
            return engine.begin()

    return Db


def _graph(db) -> None:
    db.node("Document", {**DOC, "name": "b.pdf", "latest": True, "frozen": False})
    db.node(
        "StructureNode",
        {**NODE, "source_page_ids": ["p1", "p2"], "appearance_order": 3},
        _document_key=DOC_KEY,
    )
    db.node(
        "InfoUnit",
        {"uid": "iu1", "tenant_id": TENANT, "title": "Hola Mundo", "extra": {"x": [1, 2]}},
        _document_key=DOC_KEY,
        _structure_node_id=NODE["id"],
    )
    db.node("CatalogModel", {"name": "Invoice", "selectable": True})
    db.node("ModelField", {"name": "total", "model": "Invoice", "type": "float"})
    db.node(
        "ModelInstance",
        {
            "uid": "mi1",
            "tenant_id": TENANT,
            "model_class": "Invoice",
            "created_by_user_ids": ["u1"],
            "job_ids": [],
            "total": 10.5,
        },
    )
    doc = ("Document", DOC_KEY)
    db.edge("HAS_STRUCTURE", doc, ("StructureNode", NODE["id"]))
    db.edge("HAS_INFO_UNIT", ("StructureNode", NODE["id"]), ("InfoUnit", "iu1"))
    db.edge("MATCHED_MODEL", ("InfoUnit", "iu1"), ("CatalogModel", "Invoice"))
    db.edge(
        "HAS_FIELD",
        ("CatalogModel", "Invoice"),
        ("ModelField", encode_key("ModelField", {"name": "total", "model": "Invoice"})),
        tenant=None,
        required=True,
    )


def test_dump_reads_rows_back_as_neo4j_would_return_them(db) -> None:
    _graph(db)
    dump = dump_sql(db.url, db.schema)

    nodes = {n["key"]: n for n in dump["nodes"]}
    assert len(nodes) == 6
    structure = nodes['StructureNode|{"id": "acme|a/b.pdf|1|n1"}']
    assert structure["labels"] == ["Section", "StructureNode"]
    assert structure["props"] == {
        **NODE,
        "source_page_ids": ["p1", "p2"],
        "appearance_order": 3,
    }
    assert nodes['InfoUnit|{"uid": "iu1"}']["props"]["extra"] == {"x": [1, 2]}
    instance = nodes['ModelInstance|{"uid": "mi1"}']["props"]
    assert instance["job_ids"] == [] and instance["total"] == 10.5
    assert "_document_key" not in structure["props"]
    document = nodes['Document|{"path": "a/b.pdf", "tenant_id": "acme", "version": 1}']
    assert document["props"]["latest"] is True and "_key" not in document["props"]

    rels = {(r["type"], r["from"], r["to"]) for r in dump["relationships"]}
    assert (
        "HAS_FIELD",
        'CatalogModel|{"name": "Invoice"}',
        'ModelField|{"model": "Invoice", "name": "total"}',
    ) in rels
    assert len(rels) == 4
    has_field = next(r for r in dump["relationships"] if r["type"] == "HAS_FIELD")
    assert has_field["props"] == {"required": True}


def test_invariants_pass_on_a_consistent_graph(db) -> None:
    _graph(db)
    inv.check_all(db.url, db.schema)


def test_dangling_edge_is_detected(db) -> None:
    _graph(db)
    db.edge("HAS_INFO_UNIT", ("StructureNode", NODE["id"]), ("InfoUnit", "ghost"))
    with pytest.raises(AssertionError, match="dangling"):
        inv.no_dangling_edges(db.url, db.schema)


def test_catalog_to_data_edge_is_detected(db) -> None:
    _graph(db)
    db.edge("BAD", ("CatalogModel", "Invoice"), ("ModelInstance", "mi1"), tenant=None)
    with pytest.raises(AssertionError, match="catalog -> data"):
        inv.no_catalog_to_data_edges(db.url, db.schema)


def test_edge_tenant_mismatch_is_detected(db) -> None:
    _graph(db)
    with db.connection() as connection:
        connection.execute(
            sa.update(tables.edges)
            .where(tables.edges.c.type == "HAS_INFO_UNIT")
            .values(tenant_id="x")
        )
    with pytest.raises(AssertionError, match="tenant"):
        inv.edge_tenants_consistent(db.url, db.schema)


def test_null_in_props_is_detected(db) -> None:
    _graph(db)
    with db.connection() as connection:
        connection.execute(
            sa.update(tables.info_units)
            .where(tables.info_units.c.uid == "iu1")
            .values(props={"deep": {"list": [1, None]}})
        )
    with pytest.raises(AssertionError, match="null"):
        inv.no_null_props(db.url, db.schema)


def test_wrong_document_key_is_detected(db) -> None:
    _graph(db)
    with db.connection() as connection:
        connection.execute(
            sa.update(tables.structure_nodes)
            .where(tables.structure_nodes.c.id == NODE["id"])
            .values(_document_key="nope")
        )
    with pytest.raises(AssertionError, match="_document_key"):
        inv.document_keys_consistent(db.url, db.schema)


def test_null_property_is_not_stored(db) -> None:
    key = db.node("Entity", {"uid": "e1", "tenant_id": TENANT, "value": "x", "gone": None})
    assert key == "e1"
    node = next(n for n in dump_sql(db.url, db.schema)["nodes"] if "e1" in n["key"])
    assert node["props"] == {"uid": "e1", "tenant_id": TENANT, "value": "x"}


def test_relationships_differing_in_identity_properties_coexist(db) -> None:
    _graph(db)
    src, dst = ("ModelInstance", "mi1"), ("Entity", "e1")
    db.node("Entity", {"uid": "e1", "tenant_id": TENANT})
    db.edge("REFERENCES", src, dst, field_name="a")
    db.edge("REFERENCES", src, dst, field_name="a", list_index=1)
    db.edge("REFERENCES", src, dst, field_name="b")
    with pytest.raises(sa.exc.IntegrityError):
        db.edge("REFERENCES", src, dst, field_name="b")
    inv.check_all(db.url, db.schema)


def test_fulltext_follows_inserts_updates_and_deletes(db) -> None:
    dialect = get_dialect(db.url)
    info = tables.info_units

    def matching(field: str, query: str) -> set[str]:
        with db.connection() as connection:
            stmt = sa.select(info.c.uid).where(dialect.fulltext_match(field, query))
            if db.target.backend == "sqlite":
                stmt = stmt.select_from(
                    info.join(
                        sa.table("info_units_fts"),
                        sa.text("info_units_fts.rowid = info_units.rowid"),
                    )
                )
            return {r[0] for r in connection.execute(stmt)}

    db.node(
        "InfoUnit", {"uid": "a", "tenant_id": TENANT, "title": "Cuenta anual", "description": "x"}
    )
    db.node("InfoUnit", {"uid": "b", "tenant_id": TENANT, "title": "otra", "description": "Cuenta"})
    assert matching("title", "cuenta") == {"a"}
    assert matching("description", "cuenta") == {"b"}
    with db.connection() as connection:
        connection.execute(sa.update(info).where(info.c.uid == "a").values(title="Balance"))
    assert matching("title", "cuenta") == set()
    assert matching("title", "balance") == {"a"}
    with db.connection() as connection:
        connection.execute(sa.delete(info).where(info.c.uid == "a"))
    assert matching("title", "balance") == set()


def test_chunk_rows_follow_their_owner_only_in_postgres(db) -> None:
    storage = tables.storage_tables(db.target.backend)
    if db.target.backend == "sqlite":
        assert storage.raw_file_chunks is None
        return
    dialect = get_dialect(db.url)
    with db.connection() as connection:
        connection.execute(
            dialect.insert(storage.raw_files).values(
                id="f1",
                tenant_id=TENANT,
                filename="a.pdf",
                content_type="application/pdf",
                size_bytes=3,
                checksum_sha256="0" * 64,
                stored_at="2026-01-01T00:00:00",
            )
        )
        connection.execute(
            dialect.insert(storage.raw_file_chunks).values(
                file_id="f1", n=0, tenant_id=TENANT, data=b"abc"
            )
        )
        connection.execute(sa.delete(storage.raw_files).where(storage.raw_files.c.id == "f1"))
        left = connection.execute(sa.select(sa.func.count()).select_from(storage.raw_file_chunks))
        assert left.scalar_one() == 0


def test_stored_at_round_trips_as_utc(db) -> None:
    from datetime import UTC, datetime, timedelta, timezone

    dialect = get_dialect(db.url)
    storage = tables.storage_tables(db.target.backend)
    location = {"_chunk_count": 0} if db.target.backend == "postgres" else {"_file_path": "x"}
    moment = datetime(2026, 3, 4, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    with db.connection() as connection:
        connection.execute(
            dialect.insert(storage.raw_files).values(
                id="f2",
                tenant_id=TENANT,
                filename="a",
                content_type="t",
                size_bytes=0,
                checksum_sha256="",
                stored_at=moment,
                **location,
            )
        )
        stored = connection.execute(sa.select(storage.raw_files.c.stored_at)).scalar_one()
    assert stored == moment and stored.utcoffset() == timedelta(0)
    assert stored.tzinfo is UTC or stored.utcoffset() == timedelta(0)
