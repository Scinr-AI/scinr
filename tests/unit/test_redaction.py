"""
tests/unit/test_redaction.py — Credentials never leak through error messages,
result objects, config reprs or log lines.
"""
from __future__ import annotations

import logging
import sys

import pytest

from scinr.newton import config as config_module
from scinr.newton.config import ScinrConfig
from scinr.newton.exceptions import StorageError
from scinr.newton.pipeline_units import UnitResult
from scinr.newton.results import DocumentResult, StageResult
from scinr.newton.utils.logging_config import RedactingFormatter, redact_handlers
from scinr.newton.utils.redaction import redact_secrets, redact_uri

_MONGO_PW = "Sup3rS3cret!"
_MONGO_URI = f"mongodb://app_user:{_MONGO_PW}@db1.internal:27017,db2.internal:27017/?authSource=admin"
_NEO4J_PW = "n3o4j-Pa55word"
_MISTRAL_KEY = "mistral-key-0123456789"


@pytest.fixture
def configured(monkeypatch):
    """Install a config carrying real-looking secrets."""
    cfg = ScinrConfig(
        neo4j_user="neo4j",
        neo4j_password=_NEO4J_PW,
        storage_backend="mongodb",
        mongodb_uri=_MONGO_URI,
        mistral_api_key=_MISTRAL_KEY,
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    return cfg


# ---------------------------------------------------------------------------
# redact_uri
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("mongodb://u:p4ss@host:27017/db", "mongodb://u:***@host:27017/db"),
        (_MONGO_URI, "mongodb://app_user:***@db1.internal:27017,db2.internal:27017/?authSource=admin"),
        ("mongodb+srv://u:p4ss@cluster0.x.mongodb.net/?retryWrites=true", "mongodb+srv://u:***@cluster0.x.mongodb.net/?retryWrites=true"),
        # unescaped reserved characters in the password are still fully masked
        ("mongodb://u:pa/ss@word@host:27017", "mongodb://u:***@host:27017"),
        ("mongodb://localhost:27017", "mongodb://localhost:27017"),
        ("mongodb://only_user@host", "mongodb://only_user@host"),
        ("", ""),
        (None, None),
    ],
)
def test_redact_uri(uri, expected):
    assert redact_uri(uri) == expected


# ---------------------------------------------------------------------------
# redact_secrets
# ---------------------------------------------------------------------------


def test_redact_secrets_masks_uris_in_free_text_without_config(monkeypatch):
    monkeypatch.setattr(config_module, "_config", None)
    text = "Cannot connect to 'mongodb://u:hunter2@h:1/' nor 'https://bob:xyz@api.example.com/x'"
    out = redact_secrets(text)
    assert "hunter2" not in out and "xyz" not in out
    assert "mongodb://u:***@h:1/" in out
    assert "https://bob:***@api.example.com/x" in out


def test_redact_secrets_masks_configured_literals(configured):
    text = f"auth failed with {_MONGO_PW} / {_NEO4J_PW} / key={_MISTRAL_KEY}"
    out = redact_secrets(text)
    for secret in (_MONGO_PW, _NEO4J_PW, _MISTRAL_KEY):
        assert secret not in out


def test_redact_secrets_masks_url_decoded_mongo_password(monkeypatch):
    cfg = ScinrConfig(mongodb_uri="mongodb://u:p%40ss%2Fw0rd@h:27017")
    monkeypatch.setattr(config_module, "_config", cfg)
    assert redact_secrets("server said p@ss/w0rd") == "server said ***"


def test_redact_secrets_leaves_short_secrets_alone(monkeypatch):
    # Too short to replace literally without mangling unrelated words.
    monkeypatch.setattr(config_module, "_config", ScinrConfig(neo4j_password="neo"))
    assert redact_secrets("neo4j is down") == "neo4j is down"


@pytest.mark.parametrize("text", ["", None])
def test_redact_secrets_empty(text):
    assert redact_secrets(text) == text


# ---------------------------------------------------------------------------
# Storage factory — the reported leak
# ---------------------------------------------------------------------------


def test_mongodb_connection_error_does_not_leak_uri_password(configured, monkeypatch):
    from pymongo.errors import ServerSelectionTimeoutError

    from scinr.newton.storage import factory

    class _FailingClient:
        def __init__(self, uri, **kwargs):
            self.admin = self

        def command(self, *_a, **_kw):
            raise ServerSelectionTimeoutError(f"cannot reach {_MONGO_URI}")

        def close(self):
            pass

    monkeypatch.setattr("pymongo.MongoClient", _FailingClient)
    with pytest.raises(StorageError) as excinfo:
        factory._check_mongodb_connection(configured)
    message = str(excinfo.value)
    assert _MONGO_PW not in message
    assert "mongodb://app_user:***@db1.internal:27017" in message


# ---------------------------------------------------------------------------
# Config repr
# ---------------------------------------------------------------------------


def test_config_repr_hides_secrets(configured):
    text = repr(configured)
    for secret in (_MONGO_PW, _NEO4J_PW, _MISTRAL_KEY):
        assert secret not in text
    assert "mongodb_uri='mongodb://app_user:***@db1.internal" in text
    assert "neo4j_user='neo4j'" in text


def test_config_repr_does_not_mask_empty_secrets():
    text = repr(ScinrConfig())
    assert "neo4j_password=''" in text
    assert "mistral_api_key=None" in text


# ---------------------------------------------------------------------------
# Results returned to API callers
# ---------------------------------------------------------------------------


def test_result_errors_are_redacted(configured):
    leak = f"Cannot connect to MongoDB at '{_MONGO_URI}'"
    doc = DocumentResult("d", 0, 1, [leak])
    stage = StageResult("preprocess", False, [doc], 0, 1, 0.1, [leak])
    unit = UnitResult("d", {"ingestion": doc}, "ingestion", leak)
    for text in (*doc.errors, *stage.errors, unit.fatal_error):
        assert _MONGO_PW not in text
        assert "app_user:***@" in text


def test_unit_result_without_fatal_error():
    assert UnitResult("d", {}, None, None).fatal_error is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _record(msg, *args, exc_info=None):
    return logging.LogRecord("t", logging.ERROR, __file__, 1, msg, args, exc_info)


def test_redacting_formatter_masks_message_args_and_traceback(configured):
    try:
        raise RuntimeError(f"driver failed for {_MONGO_URI}")
    except RuntimeError:
        exc_info = sys.exc_info()
    out = RedactingFormatter("%(message)s").format(
        _record("connecting to %s", _MONGO_URI, exc_info=exc_info)
    )
    assert _MONGO_PW not in out
    assert "RuntimeError" in out  # traceback still present


def test_redact_handlers_wraps_existing_formatter(configured):
    logger = logging.getLogger("test_redaction.redact_handlers")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("PREFIX %(message)s"))
    logger.addHandler(handler)
    try:
        redact_handlers(logger)
        redact_handlers(logger)  # idempotent
        assert isinstance(handler.formatter, RedactingFormatter)
        assert not isinstance(handler.formatter._wrapped, RedactingFormatter)
        out = handler.format(_record("uri=%s", _MONGO_URI))
        assert out.startswith("PREFIX uri=mongodb://app_user:***@")
    finally:
        logger.removeHandler(handler)
