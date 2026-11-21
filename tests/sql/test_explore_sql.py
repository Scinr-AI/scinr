"""
tests/sql/test_explore_sql.py -- Memory summaries, explore methods and ``get_node_property_keys``
of ``SqlGraphNavigator`` over a small hand-built graph (expected values written by hand).

Graph (tenant ``t``): two classes of records (Drug x3 with one empty and one placeholder,
Company x1), one extracted record, links between them, a document with two versions, a folder
and a second document.
"""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import NavigationError
from scinr.newton.navigation.sql.navigator import SqlGraphNavigator
from scinr.newton.sql import tables as T
from scinr.newton.sql.engine import get_sync_engine

TENANT = "t"


def _mi(uid, model_class, jobs, props=None, tenant=TENANT):
    return T.model_instances.insert().values(
        uid=uid,
        tenant_id=tenant,
        model_class=model_class,
        created_by_user_ids=["u1"],
        job_ids=jobs,
        props=props or {},
    )


def _edge(rel, src, dst, tenant=TENANT):
    return T.edges.insert().values(
        tenant_id=tenant,
        type=rel,
        src_label=src[0],
        src_key=src[1],
        dst_label=dst[0],
        dst_key=dst[1],
        _ident="",
        props={},
    )


def _doc(key, path, version, latest, job, load_date, folder=False):
    return T.documents.insert().values(
        _key=key,
        tenant_id=TENANT,
        path=path,
        version=version,
        name=path.rsplit("/", 1)[-1],
        latest=latest,
        is_folder=folder,
        job_id=job,
        created_by_user_id="u1",
        props={"load_date": load_date} if load_date else {},
    )


@pytest.fixture
async def nav(sql_backend):
    navigator = SqlGraphNavigator()
    await navigator.connect()
    with get_sync_engine().begin() as c:
        for stmt in (
            T.catalog_models.insert().values(name="Drug", props={"selectable": True}),
            _mi("d1", "Drug", ["j1"], {"name": "Aspirin", "dose": "100 mg"}),
            _mi("d2", "Drug", ["j1"], {}),  # empty
            _mi("d3", "Drug", ["j2"], {"name": "Placebo"}),  # placeholder (not extracted)
            _mi("c1", "Company", ["j1"], {"name": "Bayer", "country": "DE"}),
            _mi("x1", "Drug", ["j1"], {"name": "Other"}, tenant="other"),
            _edge("MADE_BY", ("ModelInstance", "d1"), ("ModelInstance", "c1")),
            _edge("MADE_BY", ("ModelInstance", "d3"), ("ModelInstance", "c1")),
            _edge("RELATED", ("ModelInstance", "c1"), ("ModelInstance", "d2")),
            _doc("t|a.pdf|1", "a.pdf", 1, False, "j0", "2026-01-01T00:00:00"),
            _doc("t|a.pdf|2", "a.pdf", 2, True, "j1", "2026-02-01T00:00:00"),
            _doc("t|dir", "dir", 1, True, None, None, folder=True),
            _doc("t|b.pdf|1", "b.pdf", 1, True, "j2", "2026-03-01T00:00:00"),
            T.structure_nodes.insert().values(
                id="sn1", tenant_id=TENANT, role="Section", _document_key="t|a.pdf|2", props={}
            ),
            _edge("HAS_STRUCTURE", ("Document", "t|a.pdf|2"), ("StructureNode", "sn1")),
            T.extraction_results.insert().values(
                uid="er1", tenant_id=TENANT, job_id="j1", props={"text": "Aspirin 100 mg"}
            ),
            _edge("HAS_EXTRACTION", ("StructureNode", "sn1"), ("ExtractionResult", "er1")),
            _edge("EXTRACTED", ("ExtractionResult", "er1"), ("ModelInstance", "d1")),
        ):
            c.execute(stmt)
    try:
        yield navigator
    finally:
        await navigator.close()


KW = {"tenant_id": TENANT}
BOOK = {"ignore_keys": ["tenant_id", "job_ids", "created_by_user_ids"]}


async def test_summarize_model_classes(nav) -> None:
    rows = await nav.summarize_model_classes(**BOOK, **KW)
    got = [(r.model_class, r.records, r.empty, r.placeholders, r.selectable) for r in rows]
    assert got == [("Company", 1, 0, 0, None), ("Drug", 1, 1, 1, True)]


async def test_summarize_model_instance_links(nav) -> None:
    rows = await nav.summarize_model_instance_links(**BOOK, **KW)
    got = [(r.from_class, r.rel_type, r.to_class, r.links, r.to_empty, r.from_to_empty) for r in rows]
    assert got == [
        ("Drug", "MADE_BY", "Company", 2, 0, 0),
        ("Company", "RELATED", "Drug", 1, 1, 1),
    ]


async def test_summarize_sources(nav) -> None:
    rows = await nav.summarize_sources(**BOOK, **KW)
    assert [r.path for r in rows] == ["b.pdf", "a.pdf"]  # newest first, the folder is absent
    a = rows[1]
    assert a.job_ids == ["j1", "j0"]
    assert [(c.model_class, c.job_ids, c.records) for c in a.contributions] == [
        ("Company", ["j1"], 1),
        ("Drug", ["j1"], 1),
    ]


async def test_get_model_instance_links(nav) -> None:
    rows = await nav.get_model_instance_links(["d1"], **KW)
    assert [(r.source.uid, r.rel_type, r.target.uid) for r in rows] == [("d1", "MADE_BY", "c1")]
    assert await nav.get_model_instance_links([], **KW) == []
    both = await nav.get_model_instance_links(["c1"], **KW)
    assert [(r.source.uid, r.target.uid) for r in both] == [
        ("c1", "d2"),
        ("d1", "c1"),
        ("d3", "c1"),
    ]
    assert len(await nav.get_model_instance_links(["c1"], limit=1, **KW)) == 1


async def test_extraction_links_and_keys(nav) -> None:
    rows = await nav.get_extraction_links(**KW)
    assert [(r.source.properties.get("uid") or r.source.uid, r.rel_type, r.target.uid) for r in rows] == [
        ("er1", "EXTRACTED", "d1")
    ]
    assert "embedding" not in rows[0].source.properties
    assert len(await nav.get_extraction_links(field_contains=[("name", "Asp")], **KW)) == 1
    assert await nav.get_extraction_links(field_contains=[("name", "zzz")], **KW) == []
    assert await nav.get_extraction_links(document_paths=["b.pdf"], **KW) == []
    keys = await nav.list_extraction_field_keys(**KW)
    assert {"text", "name"} <= set(keys)
    with pytest.raises(NavigationError):
        await nav.get_extraction_links(document_paths=[], **KW)


async def test_get_model_instances_in_documents(nav) -> None:
    got = await nav.get_model_instances_in_documents("Drug", **KW)
    assert [r.uid for r in got] == ["d1"]
    only = await nav.get_model_instances_in_documents("Company", document_paths=["a.pdf"], **KW)
    assert [r.uid for r in only] == ["c1"]


async def test_search_model_instances(nav) -> None:
    hits = await nav.search_model_instances("aspir", **BOOK, **KW)
    assert [(h.instance.uid, h.field) for h in hits] == [("d1", "name")]
    assert [(h.instance.uid, h.field) for h in await nav.search_model_instances("c1", **BOOK, **KW)] == [
        ("c1", "uid")
    ]
    assert await nav.search_model_instances("aspir", ignore_keys=["name", "tenant_id", "job_ids", "created_by_user_ids"], **KW) == []
    with pytest.raises(NavigationError):
        await nav.search_model_instances("", **KW)


async def test_list_model_instance_field_keys(nav) -> None:
    assert await nav.list_model_instance_field_keys("Drug", **BOOK, **KW) == ["dose", "name"]
    assert await nav.list_model_instance_field_keys("Drug", ignore_keys=["dose", "tenant_id", "job_ids", "created_by_user_ids"], **KW) == ["name"]
    assert await nav.list_model_instance_field_keys("Drug", sample_limit=0, **KW) == []


async def test_get_node_property_keys(nav) -> None:
    assert "name" in await nav.get_node_property_keys("ModelInstance", **KW)
    assert await nav.get_node_property_keys("Entity", **KW) == []
    assert "text" in await nav.get_node_property_keys("ExtractionResult", **KW)
    assert "path" in await nav.get_node_property_keys("Document", **KW)
    with pytest.raises(NavigationError):
        await nav.get_node_property_keys("Nope", **KW)
