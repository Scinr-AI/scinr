"""
tests/unit/test_storage_mongodb_indexes.py — MongoDB readiness in get_storage().

get_storage() runs on every source-text read, so the connection ping and the
index bootstrap must happen once per process and MongoDB target, index
creation must not block storage access, and configure() must re-arm both.
"""

from __future__ import annotations

import pytest
from pymongo.errors import OperationFailure

from scinr.newton import config as config_module
from scinr.newton.config import ScinrConfig
from scinr.newton.storage import factory
from scinr.newton.storage.mongodb import client as mongo_client


class _FakeCollection:
    def __init__(self, name: str, log: list, existing: set[str] | None = None, fail: bool = False):
        self.name = name
        self._log = log
        self._existing = existing or set()
        self._fail = fail

    def index_information(self):
        return {n: {} for n in self._existing}

    def drop_index(self, name):
        self._log.append(("drop", self.name, name))

    def create_index(self, keys, name):
        if self._fail:
            raise OperationFailure("not authorized to execute command createIndexes", code=13)
        self._log.append(("create", self.name, name, tuple(keys)))


class _FakeClient:
    """Stand-in for pymongo.MongoClient; records every instance and call."""

    instances: list[_FakeClient] = []
    fail_indexes = False
    existing: set[str] = set()

    def __init__(self, uri, **kwargs):
        self.uri = uri
        self.log: list = []
        self.closed = False
        self.admin = self
        _FakeClient.instances.append(self)

    def command(self, name):
        self.log.append(("command", name))

    def __getitem__(self, db_name):
        client = self

        class _Db:
            def __getitem__(self, coll):
                return _FakeCollection(coll, client.log, _FakeClient.existing, _FakeClient.fail_indexes)

        return _Db()

    def close(self):
        self.closed = True


@pytest.fixture
def mongo_cfg(monkeypatch):
    _FakeClient.instances = []
    _FakeClient.fail_indexes = False
    _FakeClient.existing = set()
    monkeypatch.setattr("pymongo.MongoClient", _FakeClient)
    cfg = ScinrConfig(storage_backend="mongodb", mongodb_uri="mongodb://h:27017")
    monkeypatch.setattr(config_module, "_config", cfg)
    factory.reset_mongodb_readiness()
    yield cfg
    factory.reset_mongodb_readiness()


def _creates(client: _FakeClient) -> list[str]:
    return [entry[2] for entry in client.log if entry[0] == "create"]


def test_get_storage_pings_and_indexes_once_per_process(mongo_cfg):
    for _ in range(3):
        factory.get_storage()
    assert len(_FakeClient.instances) == 1
    client = _FakeClient.instances[0]
    assert client.log[0] == ("command", "ping")
    assert sorted(_creates(client)) == sorted(name for _, _, name in mongo_client._INDEX_SPECS)
    # The same client serves ping and indexes, and is closed afterwards.
    assert client.closed


def test_every_index_is_prefixed_by_tenant():
    for _, keys, name in mongo_client._INDEX_SPECS:
        assert keys[0] == ("tenant_id", 1), name


def test_indexes_skipped_when_disabled(mongo_cfg):
    mongo_cfg.mongodb_ensure_indexes = False
    factory.get_storage()
    client = _FakeClient.instances[0]
    assert client.log == [("command", "ping")]


def test_index_permission_error_warns_and_continues(mongo_cfg, caplog):
    _FakeClient.fail_indexes = True
    with caplog.at_level("WARNING", logger="scinr.newton.storage.factory"):
        factory.get_storage()
        factory.get_storage()
    assert "Could not create MongoDB indexes" in caplog.text
    # Not retried (nor re-warned) on every call.
    assert len(_FakeClient.instances) == 1
    assert caplog.text.count("Could not create MongoDB indexes") == 1


def test_obsolete_indexes_are_dropped_only_when_present(mongo_cfg):
    _FakeClient.existing = {"raw_files_by_checksum"}
    factory.get_storage()
    drops = [e for e in _FakeClient.instances[0].log if e[0] == "drop"]
    assert drops == [("drop", "raw_files", "raw_files_by_checksum")]


def test_reset_client_rearms_the_check(mongo_cfg):
    factory.get_storage()
    mongo_client.reset_client()
    factory.get_storage()
    assert len(_FakeClient.instances) == 2


def test_new_target_is_checked_again(mongo_cfg):
    factory.get_storage()
    mongo_cfg.mongodb_database = "other"
    factory.get_storage()
    assert len(_FakeClient.instances) == 2


def test_configure_reads_mongodb_ensure_indexes_env(monkeypatch):
    creds = {"neo4j_user": "neo4j", "neo4j_password": "pw"}
    monkeypatch.setenv("MONGODB_ENSURE_INDEXES", "false")
    assert config_module.configure(**creds).mongodb_ensure_indexes is False
    monkeypatch.delenv("MONGODB_ENSURE_INDEXES")
    assert config_module.configure(**creds).mongodb_ensure_indexes is True
    assert config_module.configure(**creds, mongodb_ensure_indexes=False).mongodb_ensure_indexes is False
