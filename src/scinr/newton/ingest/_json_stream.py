"""
ingest/_json_stream.py — Incremental writers for document snapshots.

A snapshot (see ``docs/user-guides/document-freezing.md``) has the shape::

    {"schema_version": 1, "tenant_id": ..., "frozen_at": ..., "keep_flags": {...},
     "documents": [
        {"document": {...}, "nodes": [...], "relationships": [...]},
        ...
     ]}

:class:`JsonSnapshotWriter` writes it element by element to a text stream
(``json.dumps`` per element, commas between them), so the subtree is never
materialised in memory; :class:`DictSnapshotWriter` builds the same content
as a Python ``dict`` for ``export_document_snapshot(destination="dict")``.

Per document entry the writers expect every node first and then every
relationship — the order in which ``ingest/freeze.py`` produces them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date, datetime, time
from typing import Any, TextIO


def _json_default(value: Any) -> Any:
    """Serialise the non-JSON values a Neo4j record can hold.

    Temporal values (``neo4j.time`` types expose ``iso_format()``; Python
    ``datetime``/``date``/``time`` expose ``isoformat()``) become ISO-8601
    strings. The pipeline itself only stores ISO strings, so this only
    matters for properties written by hand.
    """
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any) -> str:
    """``json.dumps`` with the snapshot conventions (UTF-8 kept, temporal → ISO)."""
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def to_plain(value: Any) -> Any:
    """Return *value* as the plain JSON value the file writer would emit."""
    return json.loads(dumps(value))


class SnapshotWriter(ABC):
    """Receives a snapshot element by element."""

    @abstractmethod
    def begin(self, header: dict[str, Any]) -> None:
        """Open the snapshot with its header fields (everything but ``documents``)."""

    @abstractmethod
    def begin_document(self, document: dict[str, Any]) -> None:
        """Open a ``documents[]`` entry with the :Document properties."""

    @abstractmethod
    def write_node(self, node: dict[str, Any]) -> None:
        """Append to the current entry's ``nodes``."""

    @abstractmethod
    def write_relationship(self, relationship: dict[str, Any]) -> None:
        """Append to the current entry's ``relationships`` (after all its nodes)."""

    @abstractmethod
    def end_document(self) -> None:
        """Close the current ``documents[]`` entry."""

    @abstractmethod
    def end(self) -> None:
        """Close the snapshot."""


class JsonSnapshotWriter(SnapshotWriter):
    """Streams the snapshot as JSON text to *stream* (one element per line)."""

    def __init__(self, stream: TextIO) -> None:
        self._out = stream
        self._first_document = True
        self._section: str | None = None  # "nodes" | "relationships" inside an entry
        self._first_item = True

    def begin(self, header: dict[str, Any]) -> None:
        self._out.write("{")
        for name, value in header.items():
            self._out.write(f"{dumps(name)}: {dumps(value)}, ")
        self._out.write('"documents": [')

    def begin_document(self, document: dict[str, Any]) -> None:
        if self._section is not None:
            raise RuntimeError("begin_document() called before end_document().")
        self._out.write("\n" if self._first_document else ",\n")
        self._first_document = False
        self._out.write(f'{{"document": {dumps(document)},\n"nodes": [')
        self._section = "nodes"
        self._first_item = True

    def _write_item(self, item: dict[str, Any]) -> None:
        self._out.write("\n" if self._first_item else ",\n")
        self._first_item = False
        self._out.write(dumps(item))

    def _open_relationships(self) -> None:
        self._out.write('],\n"relationships": [')
        self._section = "relationships"
        self._first_item = True

    def write_node(self, node: dict[str, Any]) -> None:
        if self._section != "nodes":
            raise RuntimeError("write_node() is only valid before the entry's relationships.")
        self._write_item(node)

    def write_relationship(self, relationship: dict[str, Any]) -> None:
        if self._section is None:
            raise RuntimeError("write_relationship() called outside a document entry.")
        if self._section == "nodes":
            self._open_relationships()
        self._write_item(relationship)

    def end_document(self) -> None:
        if self._section is None:
            raise RuntimeError("end_document() called outside a document entry.")
        if self._section == "nodes":
            self._open_relationships()
        self._out.write("]}")
        self._section = None

    def end(self) -> None:
        if self._section is not None:
            raise RuntimeError("end() called inside a document entry.")
        self._out.write("\n]}\n")


class DictSnapshotWriter(SnapshotWriter):
    """Accumulates the snapshot in :attr:`snapshot` (same content as the JSON file)."""

    def __init__(self) -> None:
        self.snapshot: dict[str, Any] = {}
        self._entry: dict[str, Any] | None = None

    def begin(self, header: dict[str, Any]) -> None:
        self.snapshot = {**to_plain(header), "documents": []}

    def begin_document(self, document: dict[str, Any]) -> None:
        self._entry = {"document": to_plain(document), "nodes": [], "relationships": []}
        self.snapshot["documents"].append(self._entry)

    def write_node(self, node: dict[str, Any]) -> None:
        self._entry["nodes"].append(to_plain(node))

    def write_relationship(self, relationship: dict[str, Any]) -> None:
        self._entry["relationships"].append(to_plain(relationship))

    def end_document(self) -> None:
        self._entry = None

    def end(self) -> None:
        pass
