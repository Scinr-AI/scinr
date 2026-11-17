"""
graph/types.py — Domain types that cross the :class:`~scinr.newton.graph.base.GraphStore` interface.

Nothing here depends on a graph engine. The types that already exist elsewhere
(``DocumentInfo``, ``LeafDocument``, ``ExtractionTarget``, ``KeepFlags``) move
here in the work package that first needs them across the interface; the
module they come from re-exports them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

GC_MAX_PASSES = 7
"""Maximum number of iterations the tenant-wide orphan sweep runs for each pass."""

__all__ = [
    "GC_MAX_PASSES",
    "DocumentInfo",
    "DocumentSelector",
    "ExportReader",
    "KeepFlags",
    "LeafDocument",
    "Provenance",
    "RawNode",
    "RawProvenance",
    "RawRelationship",
    "RestoreCounters",
    "RowData",
    "TableNode",
]


@dataclass(frozen=True)
class DocumentSelector:
    """Which :Document nodes an operation targets.

    A single class for the two selection styles of the pipeline:

    * **Filters** (delete / freeze / restore): ``tenant_id`` (the *stored* tenant
      key, never ``None``; ``"__public__"`` selects public documents) plus any
      of ``path``, ``version``, ``job_id`` and ``created_by_user_id``. Only the
      fields that are set restrict the selection. ``job_id`` and
      ``created_by_user_id`` accept one value or several (any-of).
    * **Latest** (annotation / extraction): ``latest=True`` selects the
      ``latest`` version of the document at ``path``, or, when only
      ``document_name`` is given, of every document with that name.
      ``tenant_id`` is ``None`` for the public documents.

    The domain code builds the selector; each backend translates it to its own
    query language.
    """

    tenant_id: str | None = None
    path: str | None = None
    version: int | None = None
    job_id: str | Sequence[str] | None = None
    created_by_user_id: str | Sequence[str] | None = None
    document_name: str | None = None
    latest: bool = False


# (keep_structure_nodes, keep_annotations, keep_extraction_results)
KeepFlags = tuple[bool, bool, bool]


@dataclass(frozen=True)
class DocumentInfo:
    """One :Document of a resolved freeze / restore set (matched or descendant)."""

    path: str
    version: int
    tenant_id: str
    is_seed: bool
    properties: dict[str, Any]

    @property
    def frozen(self) -> bool:
        return bool(self.properties.get("frozen"))

    @property
    def frozen_blob_id(self) -> str | None:
        return self.properties.get("frozen_blob_id")

    @property
    def cleanup_pending(self) -> bool:
        """Marked frozen by a freeze that did not get to delete the whole subtree."""
        return self.frozen and bool(self.properties.get("frozen_cleanup_pending"))

    @property
    def keep_flags(self) -> KeepFlags:
        """The ``keep_*`` flags the stub was frozen with."""
        return (
            bool(self.properties.get("frozen_keep_structure_nodes")),
            bool(self.properties.get("frozen_keep_annotations")),
            bool(self.properties.get("frozen_keep_extraction_results")),
        )

    @property
    def key(self) -> dict[str, Any]:
        return {"path": self.path, "version": self.version}


class LeafDocument(NamedTuple):
    """A leaf document to process: its display ``name`` and its ``path`` (the
    selector, together with the tenant, of its ``latest`` :Document)."""

    name: str
    path: str


@dataclass(frozen=True)
class RestoreCounters:
    """What a restore batch created (the engine's write counters)."""

    nodes_created: int = 0
    relationships_created: int = 0


@dataclass(frozen=True)
class RawProvenance:
    """Provenance of a StructureNode exactly as stored (``tenant_id`` may be ``None``)."""

    tenant_id: str | None
    created_by_user_id: str | None
    job_id: str | None


@dataclass(frozen=True)
class Provenance:
    """Validated provenance copied onto the nodes an extraction writes.

    ``tenant_id`` is the stored value, already validated.
    """

    tenant_id: str
    created_by_user_id: str | None
    job_id: str | None


@dataclass(frozen=True)
class TableNode:
    """Fields of a ``:Table`` node written by the tabular sink.

    ``doc_key`` is ``(tenant_id, path, version)`` of the owning ``:Document``
    (the stored tenant key).
    """

    id: str
    node_id: str
    title: str
    order: int
    theme: str
    col_count: int
    row_count: int
    doc_key: tuple[str, str, int]
    page_id: str


@dataclass(frozen=True)
class RowData:
    """Fields of a ``:Row`` node written by the tabular sink."""

    id: str
    node_id: str
    row_index: int
    appearance_order: int
    info_uid: str
    description: str


@dataclass(frozen=True)
class RawNode:
    """A node as read for a snapshot export: its labels and properties."""

    labels: tuple[str, ...]
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawRelationship:
    """A relationship as read for a snapshot export.

    The ends are identified by their labels and business key, never by an
    engine-internal id.
    """

    type: str
    from_labels: tuple[str, ...]
    from_key: dict[str, Any]
    to_labels: tuple[str, ...]
    to_key: dict[str, Any]
    properties: dict[str, Any] = field(default_factory=dict)


class ExportReader(Protocol):
    """Read side of a snapshot export, valid inside one ``export_snapshot`` attempt.

    Each method streams: callers iterate it once and must not assume the
    backend materialised the result.
    """

    def subtree_nodes(self, doc: Any) -> Iterator[RawNode]:
        """Nodes of the subtree of *doc* (a ``DocumentInfo``)."""
        ...

    def document_relationships(self, doc: Any) -> Iterator[RawRelationship]:
        """Relationships that leave the :Document node of *doc*."""
        ...

    def subtree_relationships(self, doc: Any) -> Iterator[RawRelationship]:
        """Relationships inside the subtree of *doc*."""
        ...
