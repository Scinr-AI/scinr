"""
graph/base.py — The :class:`GraphStore` interface (write path of the graph).

A ``GraphStore`` is a facade with one sub-interface per group of operations
(``schema``, ``documents``, ``annotation``, ``extraction``, ``tabular``,
``deletion``, ``gc``, ``freeze``, ``restore``). Rules for every method:

1. **No engine objects in the signature**: no driver, session, transaction,
   runner or query text.
2. **Each method owns its sessions and transactions.** What shares a
   transaction today is one method.
3. **Sync or async as the code is today.** ``S`` methods are synchronous (they
   run in worker threads), ``A`` methods are coroutines.
4. **Tenant:** each method documents whether it takes the *stored* tenant key
   (``"__public__"`` for public documents) or the API value (``None`` = public).
5. **Retries live inside** the implementation and wrap exactly what they wrap
   today.
6. **Semaphores stay with the caller.**
7. **Only :mod:`scinr.newton.exceptions` errors leave a method** when the
   engine fails (see :class:`~scinr.newton.exceptions.GraphError`). Errors that
   do not come from the engine (validation, ``FreezeError``, errors raised by a
   caller's callback) pass through untouched.

Every group has its final signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from scinr.newton.graph.types import (
    DocumentInfo,
    DocumentSelector,
    ExportReader,
    KeepFlags,
    LeafDocument,
    RestoreCounters,
)

if TYPE_CHECKING:
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.entity_extraction.state import ExtractionTarget
    from scinr.newton.graph.sinks import CatalogSink, ExtractionSink, TabularSink
    from scinr.newton.models.document_structure import Document
    from scinr.newton.results import DeletedDocument

T = TypeVar("T")

__all__ = [
    "AnnotationOps",
    "DeletionOps",
    "DocumentOps",
    "ExtractionOps",
    "FreezeOps",
    "GcOps",
    "GcPassKind",
    "GraphStore",
    "OrphanCollector",
    "RestoreOps",
    "RestoreWriter",
    "SchemaOps",
    "TabularOps",
]


class SchemaOps(ABC):
    """Constraints and indexes."""

    @abstractmethod
    def setup(self) -> None:
        """Create the constraints and indexes (idempotent), checking the engine's minimum version. (S)

        Raises:
            ConfigurationError: If the engine is older than the minimum version.
            GraphConfigurationError: If the credentials are not configured.
            GraphError: If the engine fails.
        """

    @abstractmethod
    def ensure_indexes(self, names: Iterable[str]) -> int:
        """Create the named indexes if they are missing, waiting until they are online. (S)

        Best effort: a failure to create them is logged, not raised.

        Returns:
            The number of indexes created.

        Raises:
            ValueError: If a name is not a known index.
            GraphConfigurationError: If the credentials are not configured.
        """


class DocumentOps(ABC):
    """:Document nodes: versions, loading, replacement, leaf resolution.

    Unless a method says otherwise, tenants are the *stored* key
    (``utils.tenancy.tenant_key()`` applied, ``"__public__"`` for public
    documents). Selectors carry the public-API value (``None`` = public).
    """

    @abstractmethod
    def next_version(self, path: str, tenant: str) -> int:
        """Next version number for *path* within *tenant* (1 if none). (S)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def current_latest_version(self, path: str, tenant: str) -> int | None:
        """Version flagged ``latest`` for *path* within *tenant*, if any. (S)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def resolve_batch_version(
        self, paths: Sequence[str], tenants: Iterable[str | None], update_mode: bool
    ) -> int:
        """The single version shared by every document of one run. (S)

        *tenants* are in public-API form (``None`` = public). *paths* are the
        leaves plus their ancestor folders.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def insert_document_graph(self, doc: Document, version: int, update_mode: bool) -> None:
        """Load one document in one transaction, with retry. (S)

        Raises:
            DocumentVersionConflict: The version was already ingested.
            GraphError: If the engine fails.
        """

    @abstractmethod
    def create_tabular_document(
        self,
        *,
        document_name: str,
        doc_path: str,
        version: int,
        raw_file_id: str,
        tenant: str,
        created_by_user_id: str | None,
        job_id: str | None,
    ) -> None:
        """Hierarchy, :Document, versioning and link of a tabular file, in one transaction. (S)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def preflight_replaces(self, name: str, tenant: str) -> list[dict]:
        """``{"path", "version"}`` of each ``latest`` document of *tenant* named *name*. (S)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def apply_replacement(
        self,
        replaces_name: str,
        new_root_doc_names: Sequence[str],
        *,
        tenant: str,
        replaced_path: str | None,
    ) -> None:
        """Link the replaced document to the new root document(s). (S)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def resolve_leaf_documents(self, selector: DocumentSelector) -> list[LeafDocument]:
        """Leaf documents under the roots a ``latest`` *selector* selects; empty if none. (S)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def resolve_leaf_documents_async(self, selector: DocumentSelector) -> list[LeafDocument]:
        """Leaf documents under the roots a ``latest`` *selector* selects; empty if none. (A)

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def count_latest_documents(self, selector: DocumentSelector) -> int:
        """How many ``latest`` documents *selector* selects. (A)

        Raises:
            GraphError: If the engine fails.
        """


class AnnotationOps(ABC):
    """Catalog, themes and the annotation of StructureNodes (all A).

    Tenants are the API value (``None`` = public) in the readers and in
    ``write_manual_annotation``; ``write_annotation`` addresses StructureNodes
    by their composite id, which already carries the stored tenant.
    """

    @abstractmethod
    def catalog_sink(self) -> AbstractAsyncContextManager[CatalogSink]:
        """Open the sink that writes the model catalog and the themes (one engine session).

        Used as ``async with store.annotation.catalog_sink() as sink``; the
        domain traversal in :mod:`scinr.newton.annotation.catalog` calls it
        once per setup (catalog, themes). Calls are autocommit. Idempotent
        writes. Engine failures leave the sink methods as ``GraphError``.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def count_annotated_nodes(self, selector: DocumentSelector) -> int:
        """How many StructureNodes of the ``latest`` documents *selector* selects carry a model decision.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def fetch_nodes_to_annotate(
        self, *, tenant_id: str | None, doc_path: str, only_unannotated: bool = False
    ) -> list[dict]:
        """StructureNodes with information units of the ``latest`` document at *doc_path*.

        Each dict has ``full_id``, ``node_id``, ``title``, ``role``,
        ``appearance_order`` and ``theme``. With *only_unannotated*, nodes that
        already carry a model decision are left out.

        Args:
            tenant_id: API value (``None`` = public).

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def fetch_node_context(
        self,
        full_node_id: str,
        node_id: str,
        title: str | None = None,
        role: str | None = None,
        depth: int = 0,
        max_depth: int = 3,
        visited: set[str] | None = None,
    ) -> dict:
        """Context of a StructureNode: its information units and, recursively, its qualifying children.

        One session serves the whole traversal. The dict has ``node_id``,
        ``full_id``, ``title``, ``role``, ``info_units``, ``children``,
        ``depth`` and ``depth_limit_reached``.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def fetch_document_context_instructions(
        self, *, tenant_id: str | None, doc_path: str
    ) -> str | None:
        """``context_instructions`` of the ``latest`` document at *doc_path*, or ``None``.

        Args:
            tenant_id: API value (``None`` = public).

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def write_annotation(
        self, full_node_id: str, decision: AnnotationDecision, document_name: str
    ) -> None:
        """Write one automatic annotation (autocommit: each statement commits on its own).

        Replaces the stale decision subtree of the node; some statements are
        retried on transient errors.

        Args:
            full_node_id: Composite StructureNode id (carries the stored tenant).

        Raises:
            RuntimeError: If the StructureNode does not exist.
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def write_manual_annotation(
        self,
        document_name: str,
        matched_model_class: str,
        *,
        tenant_id: str | None,
        doc_path: str,
    ) -> int:
        """Assign a manual decision to every StructureNode with information units of a document (autocommit).

        Replaces each node's extraction result and decision.

        Args:
            tenant_id: API value (``None`` = public).

        Returns:
            The number of StructureNodes updated (0 if none qualifies).

        Raises:
            GraphError: If the engine fails.
        """


class ExtractionOps(ABC):
    """Entity extraction targets and subgraphs (all A)."""

    @abstractmethod
    async def fetch_targets(
        self,
        *,
        tenant_id: str | None,
        doc_path: str,
        only_unextracted: bool = False,
    ) -> list[ExtractionTarget]:
        """Extraction targets of the ``latest`` document at *doc_path*, in document order.

        A target is a StructureNode with a model decision and at least one
        information unit (their order ascending); nodes of a raw tabular
        extraction are excluded, and so are the already extracted ones when
        *only_unextracted* is set. ``model_class`` is ``None`` for a node with
        no matched model (Triple fallback). Falls back to an unsorted query
        if the engine lacks the APOC sort function.

        Args:
            tenant_id: API value (``None`` = public).

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def mark_info_units_extracted(self, node_full_id: str) -> None:
        """Flag as extracted (timestamp) the information units of a node that are not yet.

        Idempotent.

        Args:
            node_full_id: Composite StructureNode id.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def sink(self) -> AbstractAsyncContextManager[ExtractionSink]:
        """Open the sink that writes an extraction subgraph (one engine session).

        Used as ``async with store.extraction.sink() as sink`` by
        ``write_extraction_subgraph`` and ``write_triple_subgraph`` in
        :mod:`scinr.newton.entity_extraction.graph_mapper`, which walk the
        extracted instance and call the sink once per statement. Calls are
        autocommit. Engine failures leave the sink methods as ``GraphError``.

        Raises:
            GraphError: If the engine fails.
        """


class TabularOps(ABC):
    """Tabular subgraphs (all A)."""

    @abstractmethod
    def sink(self) -> AbstractAsyncContextManager[TabularSink]:
        """Open the sink that writes a sheet's Table and Rows.

        Used as ``async with store.tabular.sink() as sink`` by
        ``write_tabular_subgraph`` in :mod:`scinr.newton.tabular.graph_writer`,
        which walks the sheet and calls the sink once per statement. Engine
        failures leave the sink methods as ``GraphError``.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    async def delete_tabular_subgraph(
        self,
        doc_path: str,
        resolved_version: int,
        sheet_index: int,
        *,
        tenant_id: str,
    ) -> None:
        """Delete the Table of one sheet and its Row descendants (explicit transaction).

        Args:
            doc_path: Path of the owning document.
            resolved_version: Document version the table belongs to.
            sheet_index: Index of the sheet whose table is deleted.
            tenant_id: Stored tenant key of the owning :Document; it prefixes the
                Table id, so only that tenant's table is touched.

        Raises:
            GraphError: If the engine fails.
        """


GcPassKind = Literal["entity_model_instance", "labeled_entity"]
"""The two tenant-wide collection passes of ``GcOps.run_pass``."""


class OrphanCollector(Protocol):
    """Collects the orphans one operation leaves, from its candidates.

    ``gather()`` reads the candidates of a set of documents **before** their
    extraction results are deleted; ``chunk_done()`` tells the collector they are
    gone; ``collect()`` deletes the candidates that ended up orphaned, and then
    the orphans among what those pointed at, until a round deletes nothing.

    The candidates are opaque handles the collector keeps (in Neo4j, element
    ids); the pipeline never inspects them. They only live in the process, so an
    operation that dies between deleting the extraction results and ``collect()``
    loses them: its caller must leave a mark in the graph first and run
    ``GcOps.sweep_tenant`` instead when it finds the mark.

    Each method documents ``Raises:`` :class:`~scinr.newton.exceptions.GraphError`.
    """

    def gather(self, keys: Sequence[dict[str, Any]]) -> None:
        """Read and keep the orphan candidates of the documents *keys*.

        Raises:
            GraphError: If the engine fails.
        """
        ...

    def chunk_done(self) -> None:
        """Tell the collector the extraction results gathered so far are deleted."""
        ...

    def collect(self) -> None:
        """Delete the candidates that ended up orphaned (and what they pointed at).

        Raises:
            GraphError: If the engine fails.
        """
        ...

    def counts(self) -> dict[str, int]:
        """The ``gc_*`` counters shared by ``DeletionResult`` and ``FreezeResult``."""
        ...


class DeletionOps(ABC):
    """Document deletion (all S). Tenants are the *stored* key; selectors carry it too."""

    @abstractmethod
    def fetch_document_keys(
        self, selector: DocumentSelector
    ) -> tuple[list[dict[str, Any]], list[DeletedDocument], bool]:
        """``(keys, deleted_documents, interrupted)`` of the selected documents and their
        descendants: ``{"path", "version"}`` keys (descendants first), the same documents
        as ``DeletedDocument`` and whether an earlier delete or freeze was interrupted.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def fetch_existing_versions(self, selector: DocumentSelector) -> list[int]:
        """Sorted versions that exist for the selected documents.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def fetch_raw_file_ids(self, selector: DocumentSelector) -> list[tuple[str, str | None]]:
        """Distinct ``(raw_file_id, stored tenant)`` pairs referenced by the selected
        documents and their descendants.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def mark_deletion_pending(self, tenant: str, keys: Sequence[dict[str, Any]]) -> None:
        """Mark documents as half-deleted before anything is removed (in batches).

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def cascade_delete(
        self,
        tenant: str,
        keys: Sequence[dict[str, Any]],
        collector: OrphanCollector | None,
        *,
        caller: str,
    ) -> dict[str, int]:
        """Delete the subtree of the documents (not the documents) in resumable batches,
        feeding *collector* (from ``GcOps.collector``) with the nodes it can orphan.
        Returns the summed counters; *caller* is only the log prefix.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def delete_documents(self, tenant: str, keys: Sequence[dict[str, Any]]) -> int:
        """Delete the documents themselves, in batches and in the order given.

        Raises:
            GraphError: If the engine fails.
        """


class GcOps(ABC):
    """Orphan collection (all S)."""

    @abstractmethod
    def collector(self, tenant: str, caller: str) -> OrphanCollector:
        """A collector for one operation; it stores opaque handles the pipeline never
        inspects.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def sweep_tenant(self, tenant_id: str, caller: str) -> dict[str, int]:
        """Collect every orphan of one tenant; returns the ``gc_*`` counters.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def run_pass(self, kind: GcPassKind, tenant_id: str, caller: str) -> tuple[int, int]:
        """One tenant-wide pass: ``(nodes deleted, iterations run)``.

        Raises:
            GraphError: If the engine fails.
        """


class FreezeOps(ABC):
    """Freeze and snapshot export (all S). Tenants are the *stored* key.

    Every method may raise :class:`~scinr.newton.exceptions.GraphError` when the
    engine fails.
    """

    @abstractmethod
    def fetch_document_set(self, selector: DocumentSelector) -> list[DocumentInfo]:
        """The selected :Document nodes and their ``IS_COMPOSED_OF*`` descendants, each once.

        Args:
            selector: Filters selector; ``tenant_id`` is the *stored* key.

        Raises:
            ValueError: If the selector carries no tenant.
            GraphError: If the engine fails.
        """

    @abstractmethod
    def export_snapshot(
        self, docs: Sequence[DocumentInfo], tenant: str, attempt: Callable[[ExportReader], T]
    ) -> T:
        """Run *attempt* with a reader inside one session, retrying it whole.

        A retry starts *attempt* again from scratch (it opens the output file
        inside). The reader streams: *attempt* must consume each iterator it
        asks for before asking for the next, and write as it reads. Exceptions
        raised by *attempt* that do not come from the engine pass through
        untouched; engine errors raised while iterating the reader are retried
        and translated on the way out.

        Args:
            docs: The documents to export (the reader is asked about each).
            tenant: Stored tenant key of the snapshot.
            attempt: Reads from the reader and writes the snapshot; its return
                value is returned.

        Raises:
            GraphError: If the engine fails after the retries.
        """

    @abstractmethod
    def mark_frozen(
        self,
        tenant: str,
        docs: Sequence[DocumentInfo],
        *,
        blob_id: str,
        frozen_at: str,
        keep_flags: KeepFlags,
    ) -> None:
        """Mark the stubs ``frozen`` in one transaction, before anything is deleted.

        Raises:
            FreezeError: If the documents changed since they were read (one is
                missing or already frozen); nothing is marked.
            GraphError: If the engine fails.
        """

    @abstractmethod
    def mark_backup(
        self, tenant: str, docs: Sequence[DocumentInfo], blob_id: str, backup_at: str
    ) -> None:
        """Record the last backup (``blob_id``, ``backup_at``) on the documents.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def reduce_to_stubs(
        self,
        tenant: str,
        docs: Sequence[DocumentInfo],
        keep_flags: KeepFlags,
        collector: OrphanCollector | None,
        *,
        caller: str,
    ) -> dict[str, int]:
        """Delete the subtree of *docs* that *keep_flags* do not keep, in bounded batches.

        Resumable: it only deletes what is still there. *collector* (from
        ``GcOps.collector``) is handed what the ExtractionResults about to be
        deleted point at. Returns the summed subtree counters.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def clear_cleanup_pending(self, tenant: str, docs: Sequence[DocumentInfo]) -> None:
        """Remove the pending-cleanup marks from *docs*, in batches (the last step of a freeze).

        Raises:
            GraphError: If the engine fails.
        """


class RestoreWriter(Protocol):
    """Batch writer of a restore (see ``RestoreOps.batch_writer``).

    ``lane`` and the writer's concurrency are suggestions a backend may ignore.
    The contract: ``drain()`` waits for everything submitted and re-raises the
    first error, and two batches of the same ``lane`` never run at once. Every
    method may raise :class:`~scinr.newton.exceptions.GraphError`.
    """

    async def delete_stale(
        self,
        lane: Hashable,
        node_ids: Sequence[str],
        *,
        model_decisions: bool,
        extraction_results: bool,
    ) -> None:
        """Delete the stale ModelDecision / ExtractionResult subtrees hanging from StructureNodes.

        Raises:
            GraphError: If the engine fails.
        """
        ...

    async def merge_nodes(
        self,
        lane: Hashable,
        labels: tuple[str, ...],
        rows: Sequence[dict[str, Any]],
        on_done: Callable[[RestoreCounters], None],
    ) -> None:
        """Merge a batch of nodes sharing *labels*; each row is ``{"key", "props"}``.

        Raises:
            GraphError: If the engine fails.
        """
        ...

    async def create_relationships(
        self,
        lane: Hashable,
        group: tuple[str, str, str],
        rows: Sequence[dict[str, Any]],
        on_done: Callable[[RestoreCounters], None],
    ) -> None:
        """Create a batch of relationships of *group* ``(type, source label, target label)``
        unless an identical one exists; each row is ``{"from", "to", "props"}`` (business keys).

        Raises:
            GraphError: If the engine fails.
        """
        ...

    async def drain(self) -> None:
        """Wait for every batch in flight and re-raise the first failure.

        Raises:
            GraphError: If the engine fails.
        """
        ...

    async def abort(self) -> None:
        """Cancel every batch in flight.

        Raises:
            GraphError: If the engine fails.
        """
        ...


class RestoreOps(ABC):
    """Restore from stubs or snapshots (sync methods are S). Tenants are the *stored* key.

    Every method may raise :class:`~scinr.newton.exceptions.GraphError` when the
    engine fails.
    """

    @abstractmethod
    def batch_writer(self, concurrency: int) -> RestoreWriter:
        """A writer shared by the stale-subtree deletion and the rebuild. (Its methods are A)

        Must be called from the event loop that will use it.

        Raises:
            GraphConfigurationError: If the credentials are not configured.
        """

    @abstractmethod
    def existing_documents(self, tenant: str, keys: Sequence[dict[str, Any]]) -> list[DocumentInfo]:
        """The :Document nodes among *keys* (``{"path", "version"}``) that exist.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def frozen_stubs_of_blob(self, tenant: str, blob_id: str) -> list[DocumentInfo]:
        """The documents of *tenant* frozen with the snapshot *blob_id*.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def finish_restore(
        self, tenant: str, docs: Sequence[DocumentInfo], recreated_paths: Sequence[str]
    ) -> None:
        """Drop the temporary links and unmark the stubs, in bounded batches, then
        recompute ``latest`` for *recreated_paths*.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def count_blob_references(self, tenant: str, blob_id: str) -> int:
        """How many documents of *tenant* still reference the snapshot *blob_id*.

        Raises:
            GraphError: If the engine fails.
        """

    @abstractmethod
    def has_stale_extraction_results(self, node_id_groups: Iterable[Sequence[str]]) -> bool:
        """Whether any of the StructureNodes (one id list per snapshot entry) still carries
        an ExtractionResult the restore is about to delete.

        Raises:
            GraphError: If the engine fails.
        """


class GraphStore(ABC):
    """Facade over every write-path operation of the graph.

    Obtain the process-wide instance with
    :func:`scinr.newton.graph.get_graph_store`.
    """

    @property
    @abstractmethod
    def schema(self) -> SchemaOps: ...

    @property
    @abstractmethod
    def documents(self) -> DocumentOps: ...

    @property
    @abstractmethod
    def annotation(self) -> AnnotationOps: ...

    @property
    @abstractmethod
    def extraction(self) -> ExtractionOps: ...

    @property
    @abstractmethod
    def tabular(self) -> TabularOps: ...

    @property
    @abstractmethod
    def deletion(self) -> DeletionOps: ...

    @property
    @abstractmethod
    def gc(self) -> GcOps: ...

    @property
    @abstractmethod
    def freeze(self) -> FreezeOps: ...

    @property
    @abstractmethod
    def restore(self) -> RestoreOps: ...

    @abstractmethod
    def sync_scope(self) -> AbstractContextManager[None]:
        """Share one synchronous connection among the calls made inside the block.

        Worker threads started with ``asyncio.to_thread`` inside the block see
        the same connection; other concurrent runs do not. Without a scope,
        each synchronous method opens and closes its own. Scopes nest: an inner
        one reuses the outer connection.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the synchronous resources the store owns."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the asynchronous resources the store owns."""
