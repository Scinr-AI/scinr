"""
tests/unit/test_sql_keys.py -- ``sql.keys``: the business key of a node encoded as text.
"""

from __future__ import annotations

import pytest

from scinr.newton.exceptions import FreezeError
from scinr.newton.ingest.freeze import CATALOG_KEY_FIELDS, UID_LABELS
from scinr.newton.sql.keys import decode_key, encode_key

DOCUMENT = {"tenant_id": "acme", "path": "carpeta/a.pdf", "version": 1}


def test_document_key_is_a_json_array_that_keeps_the_integer() -> None:
    assert encode_key("Document", DOCUMENT) == '["acme","carpeta/a.pdf",1]'
    decoded = decode_key("Document", '["acme","carpeta/a.pdf",1]')
    assert decoded == DOCUMENT and isinstance(decoded["version"], int)


def test_model_field_key_follows_the_catalog_field_order() -> None:
    assert CATALOG_KEY_FIELDS["ModelField"] == ("name", "model")
    key = encode_key("ModelField", {"model": "Invoice", "name": "total"})
    assert key == '["total","Invoice"]'
    assert decode_key("ModelField", key) == {"name": "total", "model": "Invoice"}


@pytest.mark.parametrize(
    "label, field",
    [("StructureNode", "id"), *[(label, "uid") for label in UID_LABELS]]
    + [(label, fields[0]) for label, fields in CATALOG_KEY_FIELDS.items() if len(fields) == 1],
)
def test_single_field_keys_are_the_value_itself(label: str, field: str) -> None:
    assert encode_key(label, {field: "abc|def"}) == "abc|def"
    assert decode_key(label, "abc|def") == {field: "abc|def"}


@pytest.mark.parametrize(
    "path",
    ["año/ñandú.pdf", 'say "hi".pdf', "a\\b/c.pdf", "x,y]/[z.pdf", "日本語/文書.pdf", ""],
)
def test_round_trip_with_unusual_characters(path: str) -> None:
    props = {"tenant_id": "ténant", "path": path, "version": 12}
    key = encode_key("Document", props)
    assert decode_key("Document", key) == props
    assert "\\u" not in key  # non-ASCII text is kept readable


def test_a_missing_key_field_is_a_caller_error() -> None:
    with pytest.raises(KeyError):
        encode_key("Document", {"tenant_id": "acme", "path": "a"})


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(FreezeError):
        encode_key("Nope", {"uid": "1"})


def test_malformed_composite_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        decode_key("Document", '["acme"]')
