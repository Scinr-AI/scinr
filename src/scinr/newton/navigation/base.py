"""
navigation/base.py — ``GraphNavigator``, the engine-agnostic navigation interface.

A :class:`GraphNavigator` is a **read-only**, fully ``async`` view over the
knowledge graph produced by ``scinr.newton``. Concrete backends translate each
call to their native query language; the only backend today is
``navigation.neo4j.Neo4jGraphNavigator``.

Nothing on this interface mutates the graph. The single non-portable seam is
:meth:`GraphNavigator.execute_raw` (and its one-row sibling): an optional escape
hatch whose query string is written in the backend's own dialect.

Conventions
-----------
* Every method is ``async``.
* Return types come from ``navigation.models`` and are engine-neutral.
* A *single-get* method returns ``SomeRef | None`` **only** when its arguments
  form a full unique key. Any looser selector returns a list.
* Every method that walks a variable-length path takes ``depth: int | None``.
  ``depth=None`` = "no explicit limit" → the backend applies
  :data:`DEFAULT_MAX_DEPTH` as a runaway-traversal guard (not a hard ceiling).
  ``depth=1`` = direct only. An explicit ``n`` is used verbatim.
* Filter values passed via ``where=`` / key selectors are used **verbatim** —
  the caller is responsible for any normalisation (instance-key values, for
  instance, are stored lower-cased and accent-stripped by ingestion).
* List methods take ``limit: int | None = None`` and ``skip: int = 0`` and order
  deterministically. Dynamic filters are only applied when supplied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeAlias

from scinr.newton.exceptions import UnsupportedOperationError
from scinr.newton.navigation.models import (
    AnnotationCoverage,
    CatalogGraph,
    CatalogModelRef,
    DocumentModelProfile,
    DocumentRef,
    DocumentStats,
    DocumentTree,
    EntityLabelStat,
    EntityRelation,
    ExtractionResultRef,
    ExtractionResultWithNode,
    GraphNode,
    GraphSummary,
    InfoUnitRef,
    LabeledEntityRef,
    ModelClassStat,
    ModelDecisionRef,
    ModelDecisionWithNode,
    ModelInstanceRef,
    ModelInstanceRelation,
    ModelInstanceTree,
    NodeDescription,
    NodePath,
    NodeSelector,
    PathResult,
    ProposedModelRef,
    RelTypeStat,
    RoleStat,
    ScoredInfoUnit,
    StructureNodeRef,
    StructureTree,
    Subgraph,
    ThemeRef,
    Triple,
)

#: Fallback depth applied only when ``depth=None``. Not a hard ceiling — pass an
#: explicit ``depth`` to traverse deeper.
DEFAULT_MAX_DEPTH: int = 10

#: Fallback depth for ``:ExtractionResult``→``HAS_*``→``:ModelInstance``
#: containment walks (nested models can nest a few levels deep).
INSTANCE_CONTAINMENT_DEPTH: int = 12

_Where: TypeAlias = Mapping[str, Any] | None
_Selector: TypeAlias = "str | DocumentRef"


class GraphNavigator(ABC):
    """Engine-agnostic, read-only navigation over the ``scinr.newton`` graph.

    Obtain an instance with
    :func:`scinr.newton.navigation.get_graph_navigator` or the
    :func:`scinr.newton.navigation.graph_navigator` async context manager rather
    than constructing a backend directly.
    """

    #: Native query language of this backend: ``"cypher"``, ``"gremlin"``, … or
    #: ``"none"`` when the backend exposes no raw-query path.
    dialect: ClassVar[str] = "none"

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Acquire the backend connection and verify it is reachable.

        Raises:
            GraphConnectionError: If the engine cannot be reached.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources this navigator owns.

        Never closes a connection handed in from outside (e.g. a shared driver).
        """

    @abstractmethod
    async def ping(self) -> bool:
        """Return ``True`` if the backend answers a trivial read.

        Raises:
            GraphConnectionError: If the engine cannot be reached.
        """

    async def __aenter__(self) -> GraphNavigator:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- raw escape hatch (optional capability) ---------------------------

    async def execute_raw(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        dialect: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a raw, engine-native **read** query and return raw records.

        NON-PORTABLE: *query* is written in :attr:`dialect`. Prefer a typed
        method; reach for this only for one-off queries the typed API cannot
        express.

        Args:
            query: An engine-native read query.
            params: Query parameters (values are always sent parameterised).
            dialect: If given and it does not equal :attr:`dialect`, the call
                fails immediately.

        Returns:
            A list of plain dict records (never ``*Ref`` models).

        Raises:
            UnsupportedOperationError: If this backend has no raw-query path.
            NavigationError: If *query* contains a write clause, or *dialect*
                does not match.
        """
        raise UnsupportedOperationError(
            f"{type(self).__name__} (dialect={self.dialect!r}) has no execute_raw"
        )

    async def execute_raw_one(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        *,
        dialect: str | None = None,
    ) -> dict[str, Any] | None:
        """Like :meth:`execute_raw` but return the first record or ``None``."""
        rows = await self.execute_raw(query, params, dialect=dialect)
        return rows[0] if rows else None

    # ===================================================================
    # A. Documents & folder hierarchy
    # ===================================================================

    @abstractmethod
    async def list_root_documents(
        self,
        *,
        latest_only: bool = True,
        only_folders: bool = False,
        only_leaves: bool = False,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[DocumentRef]:
        """List "parent" documents — those with no incoming ``IS_COMPOSED_OF``.

        Args:
            latest_only: Keep only ``latest=true`` documents (default).
            only_folders: Restrict to folder-parents (``is_folder = true``).
            only_leaves: Restrict to leaf documents (``is_folder = false``).
                Mutually exclusive with *only_folders*.
        """

    @abstractmethod
    async def count_root_documents(
        self, *, latest_only: bool = True, only_folders: bool = False, only_leaves: bool = False
    ) -> int:
        """Count the documents :meth:`list_root_documents` would return."""

    @abstractmethod
    async def get_one_document(self, path: str, version: int) -> DocumentRef | None:
        """Return the document with exactly this ``(path, version)`` composite key.

        Both arguments are mandatory — this is the unique key. No "latest"
        resolution happens here.

        Returns:
            The document, or ``None`` if that exact pair does not exist.
        """

    @abstractmethod
    async def get_documents(
        self,
        *,
        path: str | None = None,
        name_contains: str | None = None,
        version: int | None = None,
        latest_only: bool = True,
        is_folder: bool | None = None,
        path_prefix: str | None = None,
        where: _Where = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[DocumentRef]:
        """Find documents by any combination of filters. **Always** a list.

        Only the filters you pass are applied. Anything looser than the full
        ``(path, version)`` key can match more than one node.

        Args:
            where: Property filters on the ``:Document`` node, as
                ``{property_name: value | Op}``. A bare value means equality; the
                operator objects (``Eq``, ``In``, ``Gte``, ``Contains``,
                ``IsNotNull``, …) cover the rest. Values are matched verbatim and
                always parameterised, and are ANDed with the other arguments and
                with ``latest_only``.

        Examples:
            A bare value is an equality test::

                await nav.get_documents(where={"tenant_id": "acme"})

            Operator objects for anything else::

                from scinr.newton.navigation import In, IsNotNull, StartsWith

                await nav.get_documents(where={
                    "job_id": In(["job-1", "job-2"]),
                    "created_by_user_id": IsNotNull(),
                    "raw_file_id": StartsWith("s3://bucket/"),
                })

        See Also:
            :mod:`scinr.newton.navigation.filters` — the full operator set and
            the ``where=`` contract (property-name rules, verbatim values).
        """

    @abstractmethod
    async def document_exists(self, path: str, *, version: int | None = None) -> bool:
        """Return whether a document exists at *path* (any version, or a specific one)."""

    @abstractmethod
    async def get_child_documents(
        self,
        path: str,
        *,
        depth: int | None = 1,
        version: int | None = None,
        is_folder: bool | None = None,
        limit: int | None = None,
    ) -> list[DocumentRef]:
        """Walk ``IS_COMPOSED_OF`` downward from *path* — **child documents only**.

        Flat, deduplicated. ``depth=1`` returns the immediate children.
        """

    @abstractmethod
    async def get_document_tree(
        self, path: str, *, depth: int | None = None, version: int | None = None
    ) -> DocumentTree | None:
        """Return the nested ``IS_COMPOSED_OF`` subtree rooted at *path*."""

    @abstractmethod
    async def get_document_parent(
        self, path: str, *, version: int | None = None
    ) -> DocumentRef | None:
        """Return the immediate folder-parent of *path*, or ``None`` for a root."""

    @abstractmethod
    async def get_document_ancestors(
        self, path: str, *, version: int | None = None, depth: int | None = None
    ) -> DocumentTree | None:
        """Return the ancestor lineage of *path* as a single-spine tree.

        The result is the root folder-parent, its ``children`` a chain leading
        down to (but not including) *path*. Flatten it for a plain list; keep it
        to render the hierarchy. ``None`` when *path* is itself a root.
        """

    @abstractmethod
    async def get_document_leaves(
        self, path: str, *, version: int | None = None, depth: int | None = None
    ) -> list[DocumentRef]:
        """Return descendants of *path* with no outgoing ``IS_COMPOSED_OF``."""

    @abstractmethod
    async def list_document_versions(self, path: str) -> list[DocumentRef]:
        """Return every version at *path*, ascending by ``version``."""

    @abstractmethod
    async def get_latest_version(self, path: str) -> DocumentRef | None:
        """Return the ``latest=true`` document at *path*, or ``None``."""

    @abstractmethod
    async def get_version_chain(self, path: str) -> list[DocumentRef]:
        """Return the versions at *path* ordered by the ``HAS_NEWER_VERSION`` chain."""

    @abstractmethod
    async def get_document_stats(
        self, path: str, *, version: int | None = None
    ) -> DocumentStats | None:
        """Return aggregate counts (nodes by role, instances by class, …) for a document."""

    # ===================================================================
    # B. StructureNodes & the document tree
    # ===================================================================

    @abstractmethod
    async def get_structure_nodes(
        self,
        document: _Selector,
        *,
        version: int | None = None,
        roles: Sequence[str] | None = None,
        title_contains: str | None = None,
        theme: str | None = None,
        where: _Where = None,
        depth: int | None = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[StructureNodeRef]:
        """Return structure nodes of *document*, flat, ordered by appearance.

        ``depth`` bounds how deep into the ``HAS_CHILD`` tree to descend
        (``None`` = whole tree). ``title_contains`` is a convenience substring
        filter on ``title``; ``where=`` covers the rest — a
        ``{property_name: value | Op}`` mapping on the node, e.g.
        ``where={"appearance_order": Gte(3)}`` (see
        :mod:`scinr.newton.navigation.filters`).
        """

    @abstractmethod
    async def count_structure_nodes(
        self,
        document: _Selector,
        *,
        version: int | None = None,
        roles: Sequence[str] | None = None,
        depth: int | None = None,
    ) -> int:
        """Count the structure nodes :meth:`get_structure_nodes` would return."""

    @abstractmethod
    async def get_root_structure_nodes(
        self, document: _Selector, *, version: int | None = None
    ) -> list[StructureNodeRef]:
        """Return only the ``HAS_STRUCTURE`` (top-level) nodes of *document*."""

    @abstractmethod
    async def get_structure_node(self, node_id: str) -> StructureNodeRef | None:
        """Return the structure node with this composite ``id``, or ``None``."""

    @abstractmethod
    async def get_child_nodes(
        self,
        node_id: str,
        *,
        depth: int | None = 1,
        roles: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[StructureNodeRef]:
        """Walk ``HAS_CHILD`` downward from *node_id*. Flat list."""

    @abstractmethod
    async def get_structure_subtree(
        self, node_id: str, *, depth: int | None = None, include_info_units: bool = False
    ) -> StructureTree | None:
        """Return the nested ``HAS_CHILD`` subtree rooted at *node_id*."""

    @abstractmethod
    async def get_parent_node(self, node_id: str) -> StructureNodeRef | None:
        """Return the ``HAS_CHILD`` parent of *node_id*, or ``None`` for a root node."""

    @abstractmethod
    async def get_node_ancestors(
        self, node_id: str, *, depth: int | None = None
    ) -> list[StructureNodeRef]:
        """Return the ancestors of *node_id*, ordered root → immediate parent."""

    @abstractmethod
    async def get_node_path(self, node_id: str) -> NodePath | None:
        """Return the document plus the node chain from its root down to *node_id*."""

    @abstractmethod
    async def get_document_of_node(self, node_id: str) -> DocumentRef | None:
        """Return the document that owns *node_id* (by traversal, not id-parsing)."""

    @abstractmethod
    async def get_sibling_nodes(
        self, node_id: str, *, include_self: bool = False
    ) -> list[StructureNodeRef]:
        """Return the nodes sharing a parent with *node_id*."""

    @abstractmethod
    async def find_structure_nodes(
        self,
        *,
        title_contains: str | None = None,
        node_id: str | None = None,
        role: str | None = None,
        theme: str | None = None,
        document: _Selector | None = None,
        where: _Where = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[StructureNodeRef]:
        """Search structure nodes across all documents.

        ``where=`` takes ``{property_name: value | Op}`` filters on the node —
        e.g. ``where={"role": In(["section", "subsection"])}``. See
        :mod:`scinr.newton.navigation.filters`.
        """

    @abstractmethod
    async def get_nodes_by_theme(
        self, theme: str, *, document: _Selector | None = None, limit: int | None = None
    ) -> list[StructureNodeRef]:
        """Return structure nodes carrying ``theme``."""

    @abstractmethod
    async def describe_node(
        self, node_id: str, *, include_source_text: bool = False
    ) -> NodeDescription | None:
        """Return an aggregate view of *node_id*: info units, decision, extraction, …."""

    # ===================================================================
    # C. InfoUnits
    # ===================================================================

    @abstractmethod
    async def get_info_units(
        self, node_id: str, *, order_by: str = "order"
    ) -> list[InfoUnitRef]:
        """Return the info units of one structure node."""

    @abstractmethod
    async def count_info_units(
        self, document: _Selector, *, version: int | None = None, depth: int | None = None
    ) -> int:
        """Count the info units of *document*."""

    @abstractmethod
    async def search_info_units(
        self,
        text: str,
        *,
        field: Literal["title", "description", "both"] = "both",
        document: _Selector | None = None,
        limit: int = 25,
    ) -> list[ScoredInfoUnit]:
        """Relevance-search info units by ``title`` / ``description``."""

    @abstractmethod
    async def get_info_unit(self, uid: str) -> InfoUnitRef | None:
        """Return the info unit with this ``uid``, or ``None``."""

    @abstractmethod
    async def get_node_for_info_unit(self, uid: str) -> StructureNodeRef | None:
        """Return the structure node that owns info unit *uid*."""

    # ===================================================================
    # D. Annotation (ModelDecision)
    # ===================================================================

    @abstractmethod
    async def get_model_decision(self, node_id: str) -> ModelDecisionRef | None:
        """Return the model decision for *node_id*, or ``None`` if unannotated."""

    @abstractmethod
    async def get_document_model_decisions(
        self,
        document: _Selector,
        *,
        version: int | None = None,
        matched_only: bool | None = None,
        depth: int | None = None,
    ) -> list[ModelDecisionWithNode]:
        """Return the model decisions of *document*, each carrying its node."""

    @abstractmethod
    async def get_document_model_profile(
        self, document: _Selector, *, version: int | None = None, depth: int | None = None
    ) -> DocumentModelProfile | None:
        """Return how *document* was semantically catalogued — a roll-up of the
        ``matched`` and ``complementary`` model classes across all its
        decisions, with per-class counts, **without** the individual decisions.
        """

    @abstractmethod
    async def get_nodes_by_annotated_model(
        self, model_class: str, *, document: _Selector | None = None
    ) -> list[StructureNodeRef]:
        """Return structure nodes whose decision matched ``model_class``."""

    @abstractmethod
    async def get_unannotated_nodes(
        self, document: _Selector, *, version: int | None = None, depth: int | None = None
    ) -> list[StructureNodeRef]:
        """Return structure nodes of *document* with no ``HAS_MODEL_DECISION``."""

    @abstractmethod
    async def get_proposed_models(
        self, *, document: _Selector | None = None
    ) -> list[ProposedModelRef]:
        """Return proposed (new) models with their fields and source node."""

    @abstractmethod
    async def get_annotation_coverage(
        self, document: _Selector, *, version: int | None = None, depth: int | None = None
    ) -> AnnotationCoverage | None:
        """Return annotated / unannotated / matched / proposed counts and ratio."""

    # ===================================================================
    # E. Extraction & model instances
    # ===================================================================

    @abstractmethod
    async def get_extraction_result(self, node_id: str) -> ExtractionResultRef | None:
        """Return the extraction result for *node_id*, or ``None``."""

    @abstractmethod
    async def get_document_extraction_results(
        self,
        document: _Selector,
        *,
        version: int | None = None,
        model_class: str | None = None,
        depth: int | None = None,
        limit: int | None = None,
    ) -> list[ExtractionResultWithNode]:
        """Return the extraction results of *document*, each carrying its node."""

    @abstractmethod
    async def get_node_model_instances(
        self,
        node_id: str,
        *,
        model_class: str | None = None,
        where: _Where = None,
        depth: int | None = None,
        direct_only: bool = False,
    ) -> list[ModelInstanceRef]:
        """Return the ``:ModelInstance`` nodes extracted at *node_id*.

        Reached via ``HAS_EXTRACTION`` then ``HAS_*`` containment edges — this is
        the "belongs to this node" set, not arbitrary cross-references.

        ``where=`` filters the instances by property
        (``{property_name: value | Op}``, values matched verbatim); see
        :mod:`scinr.newton.navigation.filters`.
        """

    @abstractmethod
    async def get_document_model_instances(
        self,
        document: _Selector,
        *,
        version: int | None = None,
        model_class: str | None = None,
        where: _Where = None,
        depth: int | None = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[ModelInstanceRef]:
        """Return every ``:ModelInstance`` extracted anywhere in *document* (deduped).

        ``where=`` filters the instances by property
        (``{property_name: value | Op}``, values matched verbatim) — e.g.
        ``where={"status": "active"}``. See
        :mod:`scinr.newton.navigation.filters`.
        """

    @abstractmethod
    async def count_document_model_instances(
        self,
        document: _Selector,
        *,
        version: int | None = None,
        model_class: str | None = None,
        where: _Where = None,
        depth: int | None = None,
    ) -> int:
        """Count the instances :meth:`get_document_model_instances` would return.

        Accepts the same ``where=`` property filter (see
        :mod:`scinr.newton.navigation.filters`).
        """

    @abstractmethod
    async def get_model_instances_by_class(
        self,
        model_class: str,
        *,
        where: _Where = None,
        document: _Selector | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[ModelInstanceRef]:
        """Return ``:ModelInstance`` nodes of ``model_class``, filtered by ``where=``.

        Args:
            where: Property filters on the instance, as
                ``{property_name: value | Op}``. A bare value means equality.
                Values are matched **verbatim** — normalise them yourself
                (instance-key / entity values are stored lower-cased &
                accent-stripped by ingestion; see
                :func:`scinr.newton.utils.uid.normalize_key`).
            order_by: Instance property to sort by (validated as an identifier);
                defaults to ``uid``.

        Examples:
            ::

                from scinr.newton.navigation import In, Gte

                await nav.get_model_instances_by_class(
                    "VariationCodeModel",
                    where={"procedure_type": In(["ia", "ib"]), "confidence": Gte(0.8)},
                    order_by="procedure_type",
                    limit=50,
                )

            Use :meth:`get_model_properties` to discover which property names a
            class actually carries.

        See Also:
            :mod:`scinr.newton.navigation.filters` — the full operator set and
            the ``where=`` contract.
        """

    @abstractmethod
    async def count_model_instances_by_class(
        self, model_class: str, *, where: _Where = None, document: _Selector | None = None
    ) -> int:
        """Count the instances :meth:`get_model_instances_by_class` would return.

        Accepts the same ``where=`` property filter (see
        :mod:`scinr.newton.navigation.filters`).
        """

    @abstractmethod
    async def get_model_instance(self, uid: str) -> ModelInstanceRef | None:
        """Return the ``:ModelInstance`` with this ``uid``, or ``None``."""

    @abstractmethod
    async def get_model_instance_by_key(
        self, model_class: str, key_fields: Mapping[str, str]
    ) -> ModelInstanceRef | None:
        """Return the instance whose deterministic ``instance_key`` uid matches.

        *key_fields* maps each ``instance_key`` field name to its value. Values
        are normalised (NFKD, accent-stripped, lower-cased, whitespace-collapsed)
        before the uid is rebuilt — same as ingestion.
        """

    @abstractmethod
    async def get_structure_nodes_for_model_instance(self, uid: str) -> list[StructureNodeRef]:
        """Return the structure node(s) that own instance *uid* via containment.

        Always a list: a deduplicated ``instance_key`` instance can belong to
        several nodes; a shell instance belongs to none (``[]``).
        """

    @abstractmethod
    async def get_documents_for_model_instance(self, uid: str) -> list[DocumentRef]:
        """Return the document(s) that contain instance *uid*. Always a list."""

    @abstractmethod
    async def get_extraction_results_for_model_instance(
        self, uid: str
    ) -> list[ExtractionResultRef]:
        """Return the extraction result(s) that reach instance *uid*. Always a list."""

    @abstractmethod
    async def get_incoming_model_instances(
        self,
        uid: str,
        *,
        rel_type: str | None = None,
        depth: int | None = 1,
        limit: int | None = None,
    ) -> list[ModelInstanceRef]:
        """Return ``:ModelInstance`` nodes with an edge **into** *uid* (any rel type).

        Each carries ``via_rel`` and ``direction="in"``.
        """

    @abstractmethod
    async def get_outgoing_model_instances(
        self,
        uid: str,
        *,
        rel_type: str | None = None,
        depth: int | None = 1,
        limit: int | None = None,
    ) -> list[ModelInstanceRef]:
        """Return ``:ModelInstance`` nodes reached by an edge **out of** *uid* (any rel type).

        Each carries ``via_rel`` and ``direction="out"``.
        """

    @abstractmethod
    async def get_model_instance_subtree(
        self, uid: str, *, depth: int | None = None
    ) -> ModelInstanceTree | None:
        """Return the outgoing-edge subtree rooted at instance *uid*."""

    @abstractmethod
    async def get_model_instance_relationships(
        self,
        uid: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        rel_type: str | None = None,
    ) -> list[ModelInstanceRelation]:
        """Return every edge between *uid* and another ``:ModelInstance``.

        No relationship-type filtering by default — containment (``HAS_*``) and
        typed cross-references alike, each with its ``rel_type`` and
        ``direction``.
        """

    @abstractmethod
    async def get_related_model_instances(
        self, uid: str, rel_type: str, *, direction: Literal["out", "in"] = "out"
    ) -> list[ModelInstanceRef]:
        """Return instances linked to *uid* by ``rel_type`` in ``direction``."""

    @abstractmethod
    async def find_shell_model_instances(
        self, *, model_class: str | None = None, limit: int | None = None
    ) -> list[ModelInstanceRef]:
        """Return likely "shell" instances (only key properties populated)."""

    @abstractmethod
    async def list_model_instance_relationship_types(
        self, *, document: _Selector | None = None
    ) -> list[RelTypeStat]:
        """Return distinct ``(source model, rel_type, target model, count)`` triples
        for **non-containment** edges between model instances.
        """

    # ===================================================================
    # F. Entities (LabeledEntity, Entity, triples)
    # ===================================================================

    @abstractmethod
    async def get_model_instance_entities(
        self, uid: str, *, label: str | None = None
    ) -> list[LabeledEntityRef]:
        """Return labeled entities that instance *uid* ``REFERENCES``."""

    @abstractmethod
    async def get_node_entities(
        self, node_id: str, *, label: str | None = None, depth: int | None = None
    ) -> list[LabeledEntityRef]:
        """Return labeled entities referenced by any model instance under *node_id*.

        ``REFERENCES`` always originates from a ``:ModelInstance``; the walk is
        ``node → HAS_EXTRACTION → (HAS_* )* → ModelInstance → REFERENCES``.
        """

    @abstractmethod
    async def get_document_entities(
        self,
        document: _Selector,
        *,
        label: str | None = None,
        version: int | None = None,
        depth: int | None = None,
        limit: int | None = None,
    ) -> list[LabeledEntityRef]:
        """Return labeled entities referenced anywhere in *document*."""

    @abstractmethod
    async def list_entity_labels(self) -> list[EntityLabelStat]:
        """Return each distinct ``:LabeledEntity`` label with its node count."""

    @abstractmethod
    async def get_labeled_entities(
        self,
        *,
        label: str | None = None,
        value: str | None = None,
        normalized_value: str | None = None,
        where: _Where = None,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[LabeledEntityRef]:
        """Find labeled entities by label / value / normalised value / ``where=``.

        ``where=`` takes ``{property_name: value | Op}`` filters on the
        ``:LabeledEntity`` node — e.g.
        ``where={"label": In(["Country", "ProcedureType"])}``. See
        :mod:`scinr.newton.navigation.filters`.
        """

    @abstractmethod
    async def get_labeled_entity(self, uid: str) -> LabeledEntityRef | None:
        """Return the labeled entity with this ``uid``, or ``None``."""

    @abstractmethod
    async def get_model_instances_referencing_entity(
        self, uid: str, *, model_class: str | None = None, limit: int | None = None
    ) -> list[ModelInstanceRef]:
        """Return instances that ``REFERENCES`` labeled entity *uid* (reverse lookup)."""

    @abstractmethod
    async def get_nodes_referencing_entity(
        self, uid: str, *, depth: int | None = None, limit: int | None = None
    ) -> list[StructureNodeRef]:
        """Return structure nodes whose model instances reference entity *uid*.

        Walk: ``LabeledEntity ← REFERENCES ← ModelInstance ← (HAS_*)* ←
        ExtractionResult ← HAS_EXTRACTION ← StructureNode``.
        """

    @abstractmethod
    async def get_entity_relationships(
        self,
        uid: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        rel_type: str | None = None,
    ) -> list[EntityRelation]:
        """Return Level-2 ``field_relationships`` edges of labeled entity *uid*.

        Discriminated by both endpoints being ``:LabeledEntity`` — some of these
        types happen to start with ``HAS_``.
        """

    @abstractmethod
    async def get_related_entities(
        self, uid: str, rel_type: str, *, direction: Literal["out", "in"] = "out"
    ) -> list[LabeledEntityRef]:
        """Return labeled entities linked to *uid* by ``rel_type``."""

    @abstractmethod
    async def get_triples(self, node_id: str) -> list[Triple]:
        """Return subject–predicate–object triples extracted from *node_id*.

        The predicate edge is optional: a subject entity with no predicate edge
        to an object of the same extraction result yields a partial ``Triple``
        (``predicate``/``object`` ``None``).
        """

    @abstractmethod
    async def get_entity_triples(
        self, value_or_uid: str, *, direction: Literal["out", "in", "both"] = "both"
    ) -> list[Triple]:
        """Return triples touching the ``:Entity`` identified by value or uid."""

    # ===================================================================
    # G. Catalogue / schema introspection
    # ===================================================================

    @abstractmethod
    async def list_catalog_models(self, *, include_fields: bool = False) -> list[CatalogModelRef]:
        """Return the registered ``:CatalogModel`` nodes."""

    @abstractmethod
    async def get_catalog_graph(
        self, *, include_fields: bool = True, include_relationships: bool = True
    ) -> CatalogGraph:
        """Return the whole model catalogue — ``:CatalogModel`` / ``:EntityLabel``
        nodes plus the declared relationships between them (``AGGREGATES`` and the
        domain relationship declarations with their ``join_via`` / ``via_field``
        metadata).
        """

    @abstractmethod
    async def list_model_classes_in_use(
        self, *, document: _Selector | None = None
    ) -> list[ModelClassStat]:
        """Return each distinct ``ModelInstance.model_class`` with its count."""

    @abstractmethod
    async def get_model_properties(
        self, model_class: str, *, document: _Selector | None = None
    ) -> dict[str, list[str]]:
        """Return ``{"declared": [...], "observed": [...]}`` property names for
        ``model_class`` — the catalog-declared ``:ModelField`` names and the
        names actually seen on a sample of its instances.
        """

    @abstractmethod
    async def list_node_roles(self, *, document: _Selector | None = None) -> list[RoleStat]:
        """Return each distinct ``StructureNode.role`` with its count."""

    @abstractmethod
    async def list_themes(self) -> list[ThemeRef]:
        """Return the themes present in the graph."""

    @abstractmethod
    async def list_relationship_types(self, *, structural_only: bool = True) -> list[str]:
        """Return relationship-type names.

        ``structural_only=True`` (default) returns the curated set the pipeline
        writes structurally; ``False`` returns every type in the graph (can be
        thousands — mostly unique normalised ``Triple`` predicates).
        """

    @abstractmethod
    async def list_node_labels(self) -> list[str]:
        """Return the engine-native node-type names in the graph."""

    @abstractmethod
    async def get_graph_summary(self) -> GraphSummary:
        """Return whole-graph counts by node type and (structural) relationship type."""

    # ===================================================================
    # H. Generic navigation / power tools
    # ===================================================================

    @abstractmethod
    async def neighbors(
        self,
        selector: NodeSelector,
        *,
        edge_types: Sequence[str] | None = None,
        direction: Literal["out", "in", "both"] = "both",
        target_types: Sequence[str] | None = None,
        depth: int | None = 1,
        limit: int | None = None,
    ) -> list[GraphNode]:
        """Return nodes adjacent to *selector* along the given edges."""

    @abstractmethod
    async def shortest_path(
        self,
        from_selector: NodeSelector,
        to_selector: NodeSelector,
        *,
        max_hops: int = 6,
        edge_types: Sequence[str] | None = None,
    ) -> PathResult | None:
        """Return a shortest path between two nodes, or ``None``."""

    @abstractmethod
    async def subgraph(
        self,
        selector: NodeSelector,
        *,
        depth: int = 2,
        edge_types: Sequence[str] | None = None,
        max_nodes: int = 500,
    ) -> Subgraph:
        """Return a bounded neighbourhood of *selector* as ``{nodes, edges}``."""
