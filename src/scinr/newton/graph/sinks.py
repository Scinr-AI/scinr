"""
graph/sinks.py — Protocols of the use-case sinks (work package 8).

A sink is the write side of one traversal that lives in the domain code: the
domain walks the Pydantic model (or the rows) and calls one sink method per
statement it emits today. The three sinks are::

    ExtractionSink   entity extraction and triples (``entity_extraction/graph_mapper.py``)
    TabularSink      tabular subgraphs (``tabular/graph_writer.py``)
    CatalogSink      catalog models and themes (``annotation/catalog.py``)

Work package 8 fills the protocols one at a time (one method per query of
today, in the order of emission). ``CatalogSink`` is done and returned by
``store.annotation.catalog_sink()``; ``ExtractionSink`` is done and returned by
``store.extraction.sink()``; ``TabularSink`` is done and returned by
``store.tabular.sink()``.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal, Protocol

from scinr.newton.graph.types import Provenance, RawProvenance, RowData, TableNode

__all__ = ["CatalogSink", "ExtractionSink", "TabularSink"]


class ExtractionSink(Protocol):
    """Write side of ``write_extraction_subgraph`` / ``write_triple_subgraph``.

    One sink is one engine session; every call is one autocommit statement of
    the writers of today. The domain (``entity_extraction/graph_mapper.py``)
    emits, for a model extraction: ``read_node_provenance``,
    ``delete_stale_extraction_result``, ``create_extraction_result``,
    ``link_catalog_model`` (primary, then one per complementary model), then
    the fields of each instance (``merge_keyed_instance`` / ``create_instance``
    for sub-models, ``merge_labeled_entity`` + ``link_reference`` for entity
    fields, at the end of each instance ``merge_instance_shell`` +
    ``link_instance_relationship`` per target and ``set_instance_properties``)
    and, last, ``link_field_relationship``. For a triple extraction:
    ``read_node_provenance``, ``delete_triple_extraction_result``,
    ``create_triple_extraction_result`` and, per complete triple,
    ``merge_entity`` (subject, object), ``link_triple_entity`` (subject,
    object) and ``merge_predicate``.

    ``parent_label`` is ``"ExtractionResult"`` or ``"ModelInstance"`` (checked
    again by the backend). ``list_index`` is ``None`` outside lists and picks
    the variant of the statement with the index. Relationship types are already
    validated by the domain. Engine failures leave as
    :class:`~scinr.newton.exceptions.GraphError`.
    """

    async def read_node_provenance(self, node_full_id: str) -> RawProvenance | None:
        """Provenance stored on the StructureNode, or ``None`` if it does not exist."""

    async def delete_stale_extraction_result(self, node_full_id: str) -> None:
        """Delete the node's ExtractionResult with every ModelInstance reachable from it."""

    async def create_extraction_result(
        self,
        node_full_id: str,
        uid: str,
        document_name: str,
        model_class: str,
        timestamp: str,
        prov: Provenance,
    ) -> None:
        """Create the ExtractionResult of a model extraction and link it to the node."""

    async def link_catalog_model(self, er_uid: str, name: str, *, primary: bool) -> None:
        """Link the ExtractionResult to a CatalogModel (primary or complementary).

        Retried on transient errors.
        """

    async def merge_keyed_instance(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        model_class: str,
        key_fields: dict[str, str],
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        """Merge the ModelInstance identified by its key and link it to its parent."""

    async def create_instance(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        model_class: str,
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        """Create a keyless ModelInstance and link it to its parent."""

    async def merge_labeled_entity(
        self, uid: str, label: str, value: str, normalized_value: str, prov: Provenance
    ) -> None:
        """Merge a LabeledEntity. Retried on transient errors."""

    async def link_reference(
        self,
        parent_label: str,
        parent_uid: str,
        le_uid: str,
        field_name: str,
        list_index: int | None,
    ) -> None:
        """Link a parent to a LabeledEntity through the field that holds the value."""

    async def merge_instance_shell(
        self, tgt_uid: str, model_class: str, key_fields: dict[str, str], prov: Provenance
    ) -> None:
        """Merge the target ModelInstance of an instance relationship. Retried on transient errors."""

    async def link_instance_relationship(
        self, src_label: str, src_uid: str, tgt_uid: str, rel_type: str
    ) -> None:
        """Link two ModelInstance nodes. Retried on transient errors."""

    async def set_instance_properties(
        self, parent_label: str, parent_uid: str, props: dict[str, Any]
    ) -> None:
        """Set the scalar properties of a parent node, in the order of *props*."""

    async def link_field_relationship(self, src_uid: str, tgt_uid: str, rel_type: str) -> None:
        """Link two LabeledEntity nodes through a declared field relationship."""

    async def delete_triple_extraction_result(self, node_full_id: str) -> None:
        """Delete the node's ExtractionResult (the ModelInstance nodes are not purged)."""

    async def create_triple_extraction_result(
        self, node_full_id: str, uid: str, document_name: str, timestamp: str, prov: Provenance
    ) -> None:
        """Create the ExtractionResult of a triple extraction and link it to the node."""

    async def merge_entity(
        self, uid: str, value: str, normalized_value: str, prov: Provenance
    ) -> None:
        """Merge an Entity. Retried on transient errors."""

    async def link_triple_entity(
        self, er_uid: str, entity_uid: str, role: Literal["subject", "object"]
    ) -> None:
        """Link the ExtractionResult to an Entity with the given role."""

    async def merge_predicate(
        self, subj_uid: str, obj_uid: str, rel_type: str, predicate_raw: str
    ) -> None:
        """Link the subject and object Entity nodes with the predicate relationship."""


class TabularSink(Protocol):
    """Write side of ``write_tabular_subgraph`` (``tabular/graph_writer.py``).

    Unlike the other sinks it holds no session: each method opens the session
    (and transaction) the writer of today used for the same statement. The
    domain emits, per sheet: ``create_table``; per batch of rows
    ``write_row_batch`` (the batch is written and dropped: the sink keeps
    nothing between calls); then, per row of the batch and under a local
    concurrency bound, either the extraction-subgraph writer of the domain
    (rows with a model) or ``replace_raw_extraction_result`` followed by one
    ``link_raw_column_value`` per non-empty column (rows without a model).
    Relationship types are already sanitized by the domain. Engine failures
    leave as :class:`~scinr.newton.exceptions.GraphError`.
    """

    async def create_table(self, table: TableNode) -> None:
        """Create the Table node under the tenant's Document (explicit transaction).

        ``table.doc_key`` is ``(tenant_id, path, version)`` of the owning
        Document.

        Raises:
            RuntimeError: If the Document does not exist (the transaction is
                rolled back).
        """

    async def write_row_batch(
        self, table_id: str, rows: list[RowData], md_uid: str, theme: str, page_id: str
    ) -> None:
        """Create the Row nodes, their InfoUnit and their link to the ModelDecision.

        One explicit transaction in its own session.
        """

    async def replace_raw_extraction_result(
        self, row_id: str, extraction_uid: str, document_name: str, timestamp: str
    ) -> Provenance | None:
        """Replace the Row's ExtractionResult by a fresh ``GenericTabularRow`` one.

        Autocommit, one session for its two statements. Returns the Row's
        validated provenance, or ``None`` (nothing created) if the Row does not
        exist.
        """

    async def link_raw_column_value(
        self,
        er_uid: str,
        rel_type: str,
        entity_uid: str,
        value: str,
        normalized_value: str,
        provenance: Provenance,
    ) -> None:
        """Merge the Entity of a column value and link the ExtractionResult to it.

        Autocommit, own session; retried on transient errors.
        """


class CatalogSink(Protocol):
    """Write side of the catalog and theme setup (``annotation/catalog.py``).

    One sink is one engine session: every call is written on its own (autocommit)
    and the sink keeps nothing between calls. The domain emits in this order
    (each loop walks the models and their fields in the same order as before):

    * catalog: ``merge_catalog_model`` per model, ``merge_model_field`` per
      field, ``merge_aggregates`` per model reference, ``set_field_metadata``
      per annotated field, one ``ensure_instance_key_indexes``,
      ``link_produces_entity``, ``merge_field_relationship`` and
      ``merge_instance_relationship``;
    * themes: ``merge_theme`` per theme, ``link_subtopic`` per nested theme,
      ``link_model_to_theme`` per model of each theme.

    Relationship types reaching ``merge_field_relationship`` and
    ``merge_instance_relationship`` are already validated by the domain
    (``[A-Z][A-Z0-9_]*``). Engine failures leave as
    :class:`~scinr.newton.exceptions.GraphError`, except
    ``ensure_instance_key_indexes``, which is best effort.
    """

    async def merge_catalog_model(self, name: str, description: str, selectable: bool) -> None:
        """Create or update the ``CatalogModel`` called *name*."""

    async def merge_model_field(
        self, model_name: str, field_name: str, field_type: str, description: str, required: bool
    ) -> None:
        """Create the ``ModelField`` and link it to its model (``HAS_FIELD``)."""

    async def merge_aggregates(
        self, src_name: str, tgt_name: str, field_name: str, is_list: bool, required: bool
    ) -> None:
        """Link two ``CatalogModel`` nodes where a field of *src_name* references *tgt_name*."""

    async def set_field_metadata(
        self, model_name: str, field_name: str, is_instance_key: bool, entity_label: str | None
    ) -> None:
        """Record on the ``ModelField`` whether it is an instance key and its entity label."""

    async def ensure_instance_key_indexes(self, props: Collection[str]) -> None:
        """Create one ``(tenant_id, <prop>)`` index per property, in alphabetical order.

        Best effort: a failed index is logged and does not raise.
        """

    async def link_produces_entity(self, model_name: str, field_name: str, entity_label: str) -> None:
        """Link the model to the ``EntityLabel`` its field produces."""

    async def merge_field_relationship(
        self,
        rel_type: str,
        source_label: str,
        target_label: str,
        via_model: str,
        from_field: str,
        to_field: str,
    ) -> None:
        """Link two ``EntityLabel`` nodes with a relationship of type *rel_type*."""

    async def merge_instance_relationship(
        self,
        rel_type: str,
        src_name: str,
        tgt_name: str,
        via_field: str,
        join_via_json: str,
        is_composite: bool,
    ) -> None:
        """Link two ``CatalogModel`` nodes with a relationship of type *rel_type*."""

    async def merge_theme(self, path: str, name: str) -> None:
        """Create or update the ``Theme`` at *path*."""

    async def link_subtopic(self, parent_path: str, child_path: str) -> None:
        """Link a theme to its nested theme (``HAS_SUBTOPIC``)."""

    async def link_model_to_theme(self, model_name: str, path: str) -> None:
        """Link a ``CatalogModel`` to its theme (``BELONGS_TO_THEME``)."""
