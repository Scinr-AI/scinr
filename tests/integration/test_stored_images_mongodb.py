"""
tests/integration/test_stored_images_mongodb.py — Stored page images against
a real MongoDB (WP2 of plans/stored-images-implementation-plan.md).

A round trip of the ``MongoDBRawFileRepository`` image methods on a
throwaway ``scinr_it_<uuid>`` database (indexes are created by
``get_storage()`` itself): an image and a frame are stored, listed, opened
(the binary round-trips byte for byte) and deleted, the metadata reads are
filtered by the tenant scope, and the GridFS blobs are gone afterwards.

Run::

    MONGODB_URI=mongodb://localhost:27017 \\
        pytest -m integration tests/integration/test_stored_images_mongodb.py

(With the ``mongodb-test`` container, which requires authentication:
``MONGODB_URI=mongodb://mongodb:mongodbmongodb@localhost:27017``.)

Skipped when ``MONGODB_URI`` is not set or the server is unreachable (the
``mongodb-test`` container is not running).
"""

from __future__ import annotations

import os
import uuid

import pytest
from bson.objectid import ObjectId

pytestmark = pytest.mark.integration

from scinr.newton import config as config_module  # noqa: E402


@pytest.fixture
async def mongo_repo():
    """A ``MongoDBRawFileRepository`` on a throwaway database (dropped at the
    end), with the indexes already ensured by ``get_storage()``."""
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

    from scinr.newton.config import configure

    previous = config_module._config
    db_name = f"scinr_it_{uuid.uuid4().hex[:12]}"
    configure(
        neo4j_user="neo4j",
        neo4j_password="unused",
        neo4j_database="unused",
        storage_backend="mongodb",
        mongodb_uri=uri,
        mongodb_database=db_name,
    )
    try:
        from scinr.newton.storage.factory import get_storage

        raw_repo, _ = get_storage()  # creates the indexes by itself
        yield raw_repo
    finally:
        client.drop_database(db_name)
        client.close()
        config_module._config = previous


async def _store_one(repo, raw_file_id, **overrides):
    kwargs = {
        "page_index": 0,
        "image_index": 0,
        "content": b"png-bytes",
        "kind": "embedded",
        "content_type": "image/png",
        "description": "a logo",
        "filename": "doc.pdf",
        "folder_path": None,
        "width": 10,
        "height": 20,
        "tenant_id": "it-tenant",
    }
    kwargs.update(overrides)
    page_index = kwargs.pop("page_index")
    image_index = kwargs.pop("image_index")
    content = kwargs.pop("content")
    return await repo.store_image(raw_file_id, page_index, image_index, content, **kwargs)


async def test_image_round_trip_and_scope(mongo_repo):
    from scinr.newton.storage.mongodb.client import get_db

    repo = mongo_repo
    raw_file_id = str(ObjectId())  # images of a (here nonexistent) raw file

    # Store two images of the same raw file and one of another tenant.
    image_id = await _store_one(repo, raw_file_id)
    frame_id = await _store_one(
        repo,
        raw_file_id,
        page_index=1,
        kind="frame",
        content_type="image/jpeg",
        description="a frame",
        time_ms=12_000,
    )
    other_tenant_id = await _store_one(repo, raw_file_id, tenant_id="other")

    # list_images: both records of the tenant, ordered by (page, image); the
    # other tenant only sees its own; page/kind filters apply.
    records = await repo.list_images(raw_file_id, tenant_id="it-tenant")
    assert [(r.page_index, r.image_index) for r in records] == [(0, 0), (1, 0)]
    assert records[0].id == image_id
    assert records[0].description == "a logo"
    assert (records[0].width, records[0].height) == (10, 20)
    assert records[1].kind == "frame"
    assert records[1].time_ms == 12_000
    other_records = await repo.list_images(raw_file_id, tenant_id="other")
    assert [r.id for r in other_records] == [other_tenant_id]
    only_frames = await repo.list_images(raw_file_id, tenant_id="it-tenant", kind="frame")
    assert [r.id for r in only_frames] == [frame_id]
    only_page_0 = await repo.list_images(raw_file_id, tenant_id="it-tenant", page_indexes=[0])
    assert [r.id for r in only_page_0] == [image_id]

    # open_image: the binary round-trips; another tenant cannot open it.
    opened = await repo.open_image(image_id, tenant_id="it-tenant")
    assert opened is not None
    record, stream = opened
    assert record.id == image_id
    assert b"".join([chunk async for chunk in stream]) == b"png-bytes"
    assert await repo.open_image(image_id, tenant_id="other") is None
    assert await repo.open_image(other_tenant_id, tenant_id="it-tenant") is None

    # delete_images: scoped; a repeated call is a no-op 0.
    assert await repo.delete_images(raw_file_id, tenant_id="other") == 1
    assert await repo.delete_images(raw_file_id, tenant_id="it-tenant") == 2
    assert await repo.delete_images(raw_file_id, tenant_id="it-tenant") == 0

    # The GridFS blobs and the metadata are gone.
    db = get_db()
    from scinr.newton.config import get_config

    cfg = get_config()
    assert await db[cfg.mongodb_images_collection].count_documents({}) == 0
    assert await db[f"{cfg.mongodb_images_gridfs_bucket}.files"].count_documents({}) == 0


async def test_duplicate_image_is_rejected(mongo_repo):
    """The unique index is the safety net: re-storing the same
    (tenant, raw_file, page, image) is an error, not a silent overwrite."""
    import pymongo

    repo = mongo_repo
    raw_file_id = str(ObjectId())
    await _store_one(repo, raw_file_id)
    with pytest.raises(pymongo.errors.DuplicateKeyError):
        await _store_one(repo, raw_file_id)
