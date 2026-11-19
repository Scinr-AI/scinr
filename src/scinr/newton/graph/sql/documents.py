"""
graph/sql/documents.py — ``store.documents`` over SQL: versions, loading, replacement, leaves.

Translation of ``graph/neo4j/documents.py`` (the specification): the same statements in the
same order, inside the same transactions. Every ``MERGE`` is an ``INSERT … ON CONFLICT``
(see ``graph/sql/_common.py``); a property written as ``null`` in Cypher is removed.

Every node below a :Document carries its ``_document_key`` (the :Document ``_key``) and every
:InfoUnit its ``_structure_node_id``; they are auxiliary columns, never properties.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from scinr.newton.graph.base import DocumentOps
from scinr.newton.graph.sql import _common as c
from scinr.newton.graph.sql._traversal import reachable
from scinr.newton.graph.types import DELETE_BATCH_ROWS, DocumentSelector, LeafDocument
from scinr.newton.ingest.nodes import _make_uid, make_structure_node_id
from scinr.newton.models.document_structure import Document, InfoUnit, StructureNode
from scinr.newton.models.file_metadata import FILE_METADATA_KEYS, FileMetadata
from scinr.newton.sql import tables as t
from scinr.newton.sql.errors import async_sql_errors, sql_errors
from scinr.newton.utils.tenancy import tenant_key

__all__ = [
    "LEAF_MAX_DEPTH",
    "STRUCTURE_MAX_DEPTH",
    "SqlDocumentOps",
    "delete_document_content",
    "handle_versioning",
    "insert_document",
    "insert_folder_document_hierarchy",
    "latest_documents",
    "link_leaf_to_folder",
]

logger = logging.getLogger(__name__)

D = t.documents

#: Bound of the ``IS_COMPOSED_OF*1..`` walk from a root to its leaves (folder depth).
LEAF_MAX_DEPTH = 100
#: Bound of the ``HAS_STRUCTURE|HAS_CHILD*1..`` walk of a document's tree (tree depth).
STRUCTURE_MAX_DEPTH = 1000

_VERSION_CONFLICT = (
    "Version conflict: version {version} was already ingested by a concurrent process. "
    "Retry the ingestion."
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_true(column: Any) -> Any:
    return column == sa.true()


def _doc(tenant: str, path: str, version: int) -> Any:
    return sa.and_(D.c.tenant_id == tenant, D.c.path == path, D.c.version == version)


def _document_exists(connection: Connection, tenant: str, path: str, version: int) -> bool:
    return (
        connection.execute(sa.select(D.c._key).where(_doc(tenant, path, version))).first()
        is not None
    )


# ---------------------------------------------------------------------------
# Writers (they run inside the caller's transaction)
# ---------------------------------------------------------------------------


def delete_document_content(
    connection: Connection, doc_path: str, version: int, tenant: str
) -> None:
    """Delete the structure and annotation data of one document version, leaf first.

    Cypher: from ``(d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)``, DETACH DELETE the
    InfoUnits, ProposedFields, ProposedModels, SupplementaryFields, ComplementaryMatches,
    ModelDecisions, the LabeledEntities REFERENCED by the ExtractionResults, the
    ExtractionResults and the StructureNodes, in that order (ModelInstances are left).
    """
    start = sa.select(sa.literal("Document", sa.Text).label("label"), D.c._key.label("key")).where(
        _doc(tenant, doc_path, version)
    )
    tree = reachable(
        start,
        max_hops=STRUCTURE_MAX_DEPTH,
        rel_types={"HAS_STRUCTURE", "HAS_CHILD"},
        name="tree",
    )
    node_keys = c.column_values(
        connection.execute(sa.select(tree.c.key).where(tree.c.label == "StructureNode").distinct())
    )
    if not node_keys:
        logger.info(
            "delete_document_content: wiped structure for tenant=%s path=%s version=%d",
            tenant,
            doc_path,
            version,
        )
        return
    md = ("HAS_MODEL_DECISION", "ModelDecision")
    steps: list[list[c.Step]] = [
        [("HAS_INFO_UNIT", "InfoUnit")],
        [md, ("HAS_PROPOSED_MODEL", "ProposedModel"), ("HAS_PROPOSED_FIELD", None)],
        [md, ("HAS_PROPOSED_MODEL", "ProposedModel")],
        [md, ("HAS_SUPPLEMENTARY_FIELD", None)],
        [md, ("HAS_COMPLEMENTARY_MATCH", None)],
        [md],
        [("HAS_EXTRACTION", "ExtractionResult"), ("REFERENCES", "LabeledEntity")],
        [("HAS_EXTRACTION", "ExtractionResult")],
    ]
    for batch in c.batched(node_keys, DELETE_BATCH_ROWS):
        for path in steps:
            found = connection.execute(c.follow(c.nodes_with_keys("StructureNode", batch), path))
            for label, keys in c.group_by_label(found).items():
                for statement in c.detach_delete(label, keys):
                    connection.execute(statement)
    for batch in c.batched(node_keys, DELETE_BATCH_ROWS):
        for statement in c.detach_delete("StructureNode", batch):
            connection.execute(statement)
    logger.info(
        "delete_document_content: wiped structure for tenant=%s path=%s version=%d",
        tenant,
        doc_path,
        version,
    )


def _merge_document(
    connection: Connection, *, tenant: str, path: str, version: int, values: dict[str, Any]
) -> None:
    key = {"tenant_id": tenant, "path": path, "version": version}
    connection.execute(c.merge_node("Document", key, on_create=values, on_match=values))


def insert_document(
    connection: Connection,
    document_name: str,
    doc_path: str,
    version: int,
    raw_file_id: str = "",
    is_folder: bool = False,
    context_instructions: str | None = None,
    *,
    tenant_id: str,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
    file_metadata: FileMetadata | None = None,
) -> None:
    """MERGE the :Document keyed by (tenant_id, path, version) and SET its properties.

    Cypher: SET name, load_date, is_folder, raw_file_id, context_instructions,
    created_by_user_id, job_id, latest = true and every ``file_<key>`` (``null`` removes).
    """
    file_values = file_metadata.model_dump() if file_metadata is not None else {}
    values: dict[str, Any] = {
        "name": document_name,
        "load_date": _now(),
        "is_folder": is_folder,
        "raw_file_id": raw_file_id,
        "context_instructions": context_instructions,
        "created_by_user_id": created_by_user_id,
        "job_id": job_id,
        "latest": True,
    }
    values.update({f"file_{key}": file_values.get(key) for key in FILE_METADATA_KEYS})
    _merge_document(connection, tenant=tenant_id, path=doc_path, version=version, values=values)


def handle_versioning(connection: Connection, doc_path: str, new_version: int, tenant: str) -> None:
    """Flag the previous latest version ``latest = false`` and link it to the new one.

    Cypher: MATCH old {tenant, path, latest: true} WHERE old.version <> $new, MATCH new;
    SET old.latest = false, MERGE (old)-[:HAS_NEWER_VERSION]->(new).
    """
    if not _document_exists(connection, tenant, doc_path, new_version):
        return
    olds = c.column_values(
        connection.execute(
            sa.select(D.c._key).where(
                D.c.tenant_id == tenant,
                D.c.path == doc_path,
                _is_true(D.c.latest),
                D.c.version != new_version,
            )
        )
    )
    if not olds:
        return
    new_key = c.document_key(tenant, doc_path, new_version)
    connection.execute(sa.update(D).where(D.c._key.in_(olds)).values(latest=False))
    for old_key in olds:
        connection.execute(
            c.merge_edge(
                "HAS_NEWER_VERSION", "Document", old_key, "Document", new_key, tenant=tenant
            )
        )


def _link_composed(
    connection: Connection, tenant: str, parent: str, child: str, version: int
) -> None:
    """MATCH parent, MATCH child (same tenant and version), MERGE (parent)-[:IS_COMPOSED_OF]->(child)."""
    if not (
        _document_exists(connection, tenant, parent, version)
        and _document_exists(connection, tenant, child, version)
    ):
        return
    connection.execute(
        c.merge_edge(
            "IS_COMPOSED_OF",
            "Document",
            c.document_key(tenant, parent, version),
            "Document",
            c.document_key(tenant, child, version),
            tenant=tenant,
        )
    )


def insert_folder_document_hierarchy(
    connection: Connection,
    folder_path: str,
    version: int,
    *,
    tenant_id: str,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """Every ancestor folder :Document (MERGE + SET), its versioning and its parent link.

    Cypher: per prefix, MERGE folder SET name, load_date, is_folder = true,
    created_by_user_id, job_id, latest = true; handle_versioning; MERGE IS_COMPOSED_OF.
    """
    load_date = _now()
    parts = folder_path.split("/")
    for depth, folder_name in enumerate(parts):
        current_path = "/".join(parts[: depth + 1])
        values = {
            "name": folder_name,
            "load_date": load_date,
            "is_folder": True,
            "created_by_user_id": created_by_user_id,
            "job_id": job_id,
            "latest": True,
        }
        _merge_document(
            connection, tenant=tenant_id, path=current_path, version=version, values=values
        )
        handle_versioning(connection, current_path, version, tenant_id)
        if depth > 0:
            _link_composed(connection, tenant_id, "/".join(parts[:depth]), current_path, version)


def link_leaf_to_folder(
    connection: Connection, doc_path: str, version: int, tenant_id: str
) -> None:
    """MERGE (folder)-[:IS_COMPOSED_OF]->(leaf) when the leaf has a parent folder."""
    if "/" not in doc_path:
        return
    _link_composed(connection, tenant_id, doc_path.rsplit("/", 1)[0], doc_path, version)


def _insert_info_unit(
    connection: Connection,
    info_unit: InfoUnit,
    parent_node_id: str,
    order: int,
    *,
    tenant: str,
    created_by_user_id: str | None,
    job_id: str | None,
    document_key: str,
) -> None:
    """MERGE the :InfoUnit (uid from the parent id) SET its fields and provenance, and link it.

    Cypher: MATCH (n {id}) MERGE (u:InfoUnit {uid}) SET info_unit_id, title, description,
    order, tenant_id/created_by_user_id/job_id = n's; MERGE (n)-[:HAS_INFO_UNIT]->(u). The
    parent was written by the same transaction with these provenance values.
    """
    uid = _make_uid(parent_node_id, info_unit.title, info_unit.description)
    values = {
        "info_unit_id": uid,
        "title": info_unit.title,
        "description": info_unit.description,
        "order": order,
        "tenant_id": tenant,
        "created_by_user_id": created_by_user_id,
        "job_id": job_id,
    }
    aux = {"_document_key": document_key, "_structure_node_id": parent_node_id}
    connection.execute(
        c.merge_node(
            "InfoUnit", {"uid": uid}, on_create=values, on_match=values, aux=aux, aux_on_match=aux
        )
    )
    connection.execute(
        c.merge_edge(
            "HAS_INFO_UNIT", "StructureNode", parent_node_id, "InfoUnit", uid, tenant=tenant
        )
    )


def _insert_structure_node(
    connection: Connection,
    node: StructureNode,
    doc_path: str,
    version: int,
    parent_id: str | None,
    node_path: str,
    *,
    tenant: str,
    created_by_user_id: str | None,
    job_id: str | None,
    document_key: str,
) -> None:
    """MERGE the :StructureNode (on id) SET its fields, link it, then its InfoUnits and children."""
    role = node.role if isinstance(node.role, str) else node.role.value
    current_node_path = f"{node_path}/{node.node_id}" if node_path else node.node_id
    composite_id = make_structure_node_id(tenant, doc_path, version, current_node_path)
    values = {
        "node_id": node.node_id,
        "role": role,
        "appearance_order": node.appearance_order,
        "title": node.title,
        "theme": getattr(node, "theme", "default"),
        "source_page_ids": node.source_page_ids,
        "tenant_id": tenant,
        "created_by_user_id": created_by_user_id,
        "job_id": job_id,
    }
    aux = {"_document_key": document_key}
    connection.execute(
        c.merge_node(
            "StructureNode",
            {"id": composite_id},
            on_create=values,
            on_match=values,
            aux=aux,
            aux_on_match=aux,
        )
    )
    if parent_id is None:
        if _document_exists(connection, tenant, doc_path, version):
            connection.execute(
                c.merge_edge(
                    "HAS_STRUCTURE",
                    "Document",
                    c.document_key(tenant, doc_path, version),
                    "StructureNode",
                    composite_id,
                    tenant=tenant,
                )
            )
    else:
        connection.execute(
            c.merge_edge(
                "HAS_CHILD",
                "StructureNode",
                parent_id,
                "StructureNode",
                composite_id,
                tenant=tenant,
            )
        )
    for index, info_unit in enumerate(node.info_units):
        _insert_info_unit(
            connection,
            info_unit,
            composite_id,
            index,
            tenant=tenant,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
            document_key=document_key,
        )
    for child in node.children:
        _insert_structure_node(
            connection,
            child,
            doc_path,
            version,
            composite_id,
            current_node_path,
            tenant=tenant,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
            document_key=document_key,
        )


def insert_document_graph(
    connection: Connection, doc: Document, version: int, update_mode: bool
) -> None:
    """Steps 1-7 of ``graph/neo4j/documents.py::insert_document_graph`` on one connection."""
    doc_path = doc.doc_path if doc.doc_path else doc.document_name
    tenant = tenant_key(doc.tenant_id)
    logger.info(
        "Inserting document graph for '%s' (tenant=%s, path=%s, version=%d, update=%s, %d root nodes)",
        doc.document_name,
        tenant,
        doc_path,
        version,
        update_mode,
        len(doc.document_structure),
    )
    if update_mode:
        delete_document_content(connection, doc_path, version, tenant)
    if "/" in doc_path:
        insert_folder_document_hierarchy(
            connection,
            doc_path.rsplit("/", 1)[0],
            version,
            tenant_id=tenant,
            created_by_user_id=doc.created_by_user_id,
            job_id=doc.job_id,
        )
    insert_document(
        connection,
        doc.document_name,
        doc_path,
        version,
        doc.raw_file_id,
        is_folder=False,
        context_instructions=doc.context_instructions,
        tenant_id=tenant,
        created_by_user_id=doc.created_by_user_id,
        job_id=doc.job_id,
        file_metadata=doc.file_metadata,
    )
    if not update_mode:
        handle_versioning(connection, doc_path, version, tenant)
    link_leaf_to_folder(connection, doc_path, version, tenant)
    document_key = c.document_key(tenant, doc_path, version)
    for root in doc.document_structure:
        _insert_structure_node(
            connection,
            root,
            doc_path,
            version,
            None,
            "",
            tenant=tenant,
            created_by_user_id=doc.created_by_user_id,
            job_id=doc.job_id,
            document_key=document_key,
        )
    logger.info(
        "Document graph insertion complete for '%s' (path=%s, version=%d).",
        doc.document_name,
        doc_path,
        version,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def latest_documents(selector: DocumentSelector) -> Any:
    """``WHERE`` of the ``latest`` :Document(s) a selector selects (path, else name).

    Raises:
        ValueError: If the selector is not a ``latest`` one, neither ``path`` nor
            ``document_name`` is given, or ``tenant_id`` is invalid.
    """
    if not selector.latest:
        raise ValueError("A latest-document selector (latest=True) is required.")
    if selector.path is None and selector.document_name is None:
        raise ValueError("Either doc_path or document_name is required.")
    tenant = tenant_key(selector.tenant_id)
    match = (
        D.c.path == selector.path
        if selector.path is not None
        else D.c.name == selector.document_name
    )
    return sa.and_(D.c.tenant_id == tenant, match, _is_true(D.c.latest))


def _roots_query(selector: DocumentSelector) -> sa.Select:
    return (
        sa.select(D.c._key, D.c.tenant_id, D.c.name, D.c.path)
        .where(latest_documents(selector))
        .order_by(c.dialect().order_text(D.c.path), D.c.version)
    )


def _leaves_query(root_key: str, tenant: str) -> sa.Select:
    """Latest documents of *tenant* reached by ``IS_COMPOSED_OF*1..`` from *root_key* that
    compose nothing themselves (by path)."""
    start = sa.select(
        sa.literal("Document", sa.Text).label("label"), sa.literal(root_key, sa.Text).label("key")
    )
    below = reachable(start, max_hops=LEAF_MAX_DEPTH, rel_types={"IS_COMPOSED_OF"}, name="below")
    leaf = D.alias("leaf")
    has_children = sa.exists().where(
        c.edges.c.src_label == "Document",
        c.edges.c.src_key == leaf.c._key,
        c.edges.c.type == "IS_COMPOSED_OF",
        c.edges.c.dst_label == "Document",
    )
    return (
        sa.select(leaf.c.name, leaf.c.path)
        .where(
            leaf.c._key.in_(sa.select(below.c.key).where(below.c.label == "Document")),
            leaf.c.tenant_id == tenant,
            _is_true(leaf.c.latest),
            ~has_children,
        )
        .order_by(c.dialect().order_text(leaf.c.path))
    )


def _collect_leaves(groups: Iterable[list[tuple[str, str]]]) -> list[LeafDocument]:
    """Leaves of every matched root, deduplicated by path (first occurrence wins)."""
    leaves: dict[str, LeafDocument] = {}
    for docs in groups:
        for name, path in docs:
            leaves.setdefault(path, LeafDocument(name, path))
    return list(leaves.values())


def _leaves_of(roots: list[Any], found: list[list[tuple[str, str]]]) -> list[LeafDocument]:
    groups = [
        leaves if leaves else [(root.name, root.path)]
        for root, leaves in zip(roots, found, strict=True)
    ]
    return _collect_leaves(groups)


class SqlDocumentOps(DocumentOps):
    """``store.documents`` over SQL. Synchronous methods use the sync engine, asynchronous
    ones the async engine; every connection is its own (``sync_scope()`` shares nothing)."""

    @sql_errors()
    def next_version(self, path: str, tenant: str) -> int:
        """``max(version) + 1`` of *path* within *tenant*, or 1.

        Cypher: MATCH (d:Document {tenant_id, path}) RETURN max(d.version).
        """

        def read(connection: Connection) -> int:
            top = connection.execute(
                sa.select(sa.func.max(D.c.version)).where(D.c.tenant_id == tenant, D.c.path == path)
            ).scalar()
            return top + 1 if top is not None else 1

        return c.sync_read(read, retry=False)

    @sql_errors()
    def current_latest_version(self, path: str, tenant: str) -> int | None:
        """Version of the ``latest`` :Document at *path* within *tenant*, or ``None``.

        Cypher: MATCH (d:Document {tenant_id, path, latest: true}) RETURN d.version (single).
        """

        def read(connection: Connection) -> int | None:
            row = connection.execute(
                sa.select(D.c.version)
                .where(D.c.tenant_id == tenant, D.c.path == path, _is_true(D.c.latest))
                .order_by(D.c.version)
                .limit(1)
            ).first()
            return row[0] if row else None

        return c.sync_read(read, retry=False)

    @sql_errors()
    def resolve_batch_version(
        self, paths: Sequence[str], tenants: Iterable[str | None], update_mode: bool
    ) -> int:
        """The version shared by one batch (same logic as ``_resolve_batch_version`` of Neo4j).

        Cypher: max(d.version) over the batch's tenants and paths (``latest: true`` only in
        update mode); normal mode returns it + 1, update mode as is; 1 when there is none.
        """
        if not paths:
            return 1
        stored = sorted({tenant_key(tenant) for tenant in tenants})
        conditions = [D.c.tenant_id.in_(stored), D.c.path.in_(list(paths))]
        if update_mode:
            conditions.append(_is_true(D.c.latest))

        def read(connection: Connection) -> int | None:
            return connection.execute(
                sa.select(sa.func.max(D.c.version)).where(*conditions)
            ).scalar()

        top = c.sync_read(read, retry=False)
        if update_mode:
            return top if top is not None else 1
        return top + 1 if top is not None else 1

    def insert_document_graph(self, doc: Document, version: int, update_mode: bool) -> None:
        """Load one document in one transaction, retried whole on transient errors.

        Cypher: ``insert_document_graph`` steps 1-7 in one explicit transaction. A unique
        violation about the version becomes ``DocumentVersionConflict`` (Neo4j's message).

        Raises:
            DocumentVersionConflict: The version was already ingested by a concurrent process.
            GraphError: If the engine fails.
            ValueError: If ``doc.tenant_id`` is empty.
        """
        with sql_errors(version_conflict=_VERSION_CONFLICT.format(version=version)):
            c.sync_write(
                lambda connection: insert_document_graph(connection, doc, version, update_mode)
            )
        logger.info("Transaction committed for document: %s (v%d)", doc.document_name, version)

    @sql_errors()
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
        """Hierarchy, leaf :Document, versioning and folder link in one transaction, no retry.

        Cypher: insert_folder_document_hierarchy, insert_document (no context, no file
        metadata), handle_versioning, link_leaf_to_folder.
        """

        def write(connection: Connection) -> None:
            if "/" in doc_path:
                insert_folder_document_hierarchy(
                    connection,
                    doc_path.rsplit("/", 1)[0],
                    version,
                    tenant_id=tenant,
                    created_by_user_id=created_by_user_id,
                    job_id=job_id,
                )
            insert_document(
                connection,
                document_name,
                doc_path,
                version,
                raw_file_id,
                is_folder=False,
                context_instructions=None,
                tenant_id=tenant,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )
            handle_versioning(connection, doc_path, version, tenant)
            link_leaf_to_folder(connection, doc_path, version, tenant)

        try:
            c.sync_write(write, retry=False)
        except Exception as exc:
            logger.error("tabular agent: Document creation failed for '%s': %s", doc_path, exc)
            raise
        logger.info("tabular agent: Document node created for '%s' (v%d)", doc_path, version)

    @sql_errors()
    def preflight_replaces(self, name: str, tenant: str) -> list[dict]:
        """``{"path", "version"}`` of every ``latest`` :Document of *tenant* named *name*.

        Cypher: MATCH (d:Document {tenant_id, name, latest: true}) RETURN d.path, d.version.
        """

        def read(connection: Connection) -> list[dict]:
            rows = connection.execute(
                sa.select(D.c.path, D.c.version)
                .where(D.c.tenant_id == tenant, D.c.name == name, _is_true(D.c.latest))
                .order_by(c.dialect().order_text(D.c.path), D.c.version)
            )
            return [{"path": row.path, "version": row.version} for row in rows]

        return c.sync_read(read, retry=False)

    @sql_errors()
    def apply_replacement(
        self,
        replaces_name: str,
        new_root_doc_names: Sequence[str],
        *,
        tenant: str,
        replaced_path: str | None,
    ) -> None:
        """Link the replaced :Document to each new root with ``HAS_NEWER_VERSION``.

        Cypher: new roots = latest documents named in *new_root_doc_names* with no incoming
        IS_COMPOSED_OF (warning and nothing written if none); then, per root (autocommit),
        old (by path, else by name; latest; old <> new): SET latest = false, MERGE the link.
        """

        def read_roots(connection: Connection) -> list[Any]:
            incoming = sa.exists().where(
                c.edges.c.type == "IS_COMPOSED_OF",
                c.edges.c.dst_label == "Document",
                c.edges.c.dst_key == D.c._key,
            )
            return list(
                connection.execute(
                    sa.select(D.c.path, D.c.version, D.c.name)
                    .where(
                        D.c.tenant_id == tenant,
                        _is_true(D.c.latest),
                        D.c.name.in_(list(new_root_doc_names)),
                        ~incoming,
                    )
                    .order_by(c.dialect().order_text(D.c.path), D.c.version)
                )
            )

        new_roots = c.sync_read(read_roots, retry=False)
        if not new_roots:
            logger.warning(
                "apply_replacement: could not find any root documents among %s; skipping.",
                new_root_doc_names,
            )
            return
        old_match = (
            D.c.path == replaced_path if replaced_path is not None else D.c.name == replaces_name
        )
        for root in new_roots:
            new_key = c.document_key(tenant, root.path, root.version)

            def link(connection: Connection, new_key: str = new_key, root: Any = root) -> None:
                if not _document_exists(connection, tenant, root.path, root.version):
                    return
                olds = c.column_values(
                    connection.execute(
                        sa.select(D.c._key).where(
                            D.c.tenant_id == tenant,
                            old_match,
                            _is_true(D.c.latest),
                            D.c._key != new_key,
                        )
                    )
                )
                if not olds:
                    return
                connection.execute(sa.update(D).where(D.c._key.in_(olds)).values(latest=False))
                for old_key in olds:
                    connection.execute(
                        c.merge_edge(
                            "HAS_NEWER_VERSION",
                            "Document",
                            old_key,
                            "Document",
                            new_key,
                            tenant=tenant,
                        )
                    )

            c.sync_write(link, retry=False)
            logger.info(
                "apply_replacement: linked '%s' (latest=false) -[:HAS_NEWER_VERSION]-> '%s' (path=%s, v=%s)",
                replaces_name,
                root.name,
                root.path,
                root.version,
            )

    @sql_errors()
    def resolve_leaf_documents(self, selector: DocumentSelector) -> list[LeafDocument]:
        """Leaves under the selected ``latest`` root(s), deduplicated by path, by root path.

        Cypher: OPTIONAL MATCH (root)-[:IS_COMPOSED_OF*1..]->(leaf {tenant, latest}) WHERE
        NOT (leaf)-[:IS_COMPOSED_OF]->(); a root without leaves is its own leaf.
        """
        roots_query = _roots_query(selector)
        tenant = tenant_key(selector.tenant_id)

        def read(connection: Connection) -> list[LeafDocument]:
            roots = list(connection.execute(roots_query))
            found = [
                [
                    (row.name, row.path)
                    for row in connection.execute(_leaves_query(root._key, tenant))
                ]
                for root in roots
            ]
            return _leaves_of(roots, found)

        return c.sync_read(read, retry=False)

    async def resolve_leaf_documents_async(self, selector: DocumentSelector) -> list[LeafDocument]:
        """Async version of :meth:`resolve_leaf_documents`."""
        roots_query = _roots_query(selector)
        tenant = tenant_key(selector.tenant_id)
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                roots = list(await connection.execute(roots_query))
                found = []
                for root in roots:
                    rows = await connection.execute(_leaves_query(root._key, tenant))
                    found.append([(row.name, row.path) for row in rows])
        return _leaves_of(roots, found)

    async def count_latest_documents(self, selector: DocumentSelector) -> int:
        """How many ``latest`` :Document nodes *selector* selects.

        Cypher: MATCH <latest pattern> RETURN count(d).
        """
        condition = latest_documents(selector)
        async with async_sql_errors():
            async with c.async_read_engine().connect() as connection:
                return (
                    await connection.execute(
                        sa.select(sa.func.count()).select_from(D).where(condition)
                    )
                ).scalar_one()
