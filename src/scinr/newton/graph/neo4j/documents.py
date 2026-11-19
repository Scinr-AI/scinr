"""
graph/neo4j/documents.py — :Document nodes over Neo4j: versions, loading, replacement, leaf resolution.

:class:`Neo4jDocumentOps` (``store.documents``) owns the sessions, transactions
and retries of these operations. The functions at the top of the module are the
Cypher that used to live in ``ingest/nodes.py``: they receive an open
:class:`neo4j.Transaction` (or session) and one or more Pydantic model
instances. The insertion strategy is:

  MERGE — nodes with a unique composite key (Document, StructureNode) or a
          deterministic uid (InfoUnit). Prevents duplicates on re-ingestion.

The tenant is part of every document's identity: a :Document is keyed by
``(tenant_id, path, version)``, and every ``MATCH``/``MERGE`` of a :Document
here includes the tenant. Two tenants ingesting the same path get fully
independent documents, folders, versions and structure. All functions in this
module take the **stored** tenant key (``utils.tenancy.tenant_key()`` already
applied — ``"__public__"`` for a public document); ``insert_document_graph()``
is the one that normalizes ``doc.tenant_id``.

Relationships are established with MATCH … MERGE to avoid duplicates while
being tolerant of re-runs.

Child nodes (InfoUnit) are only created when their parent node already exists
(MATCH parent first), preventing orphan nodes when a parent is missing.

``insert_document_graph`` receives the resolved integer version externally
(via *resolved_version*) rather than reading it from ``doc.version``.

StructureNode composite id:  ``{tenant_key}::{doc_path}::{version}::{ancestor_path/node_id}``
                             (see :func:`~scinr.newton.ingest.nodes.make_structure_node_id`)
InfoUnit uid:                ``info_unit_id`` field (16-char SHA-256 hex), derived
                             from the parent StructureNode id — so it is tenant-scoped too
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC

from neo4j.exceptions import ConstraintError

from scinr.newton.config import get_config
from scinr.newton.graph.base import DocumentOps
from scinr.newton.graph.neo4j import driver as _driver
from scinr.newton.graph.neo4j._scope import borrow_sync_driver
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.retry import with_neo4j_retry_sync
from scinr.newton.graph.types import DocumentSelector, LeafDocument
from scinr.newton.ingest.nodes import ROLE_TO_LABEL, _make_uid, make_structure_node_id
from scinr.newton.models.document_structure import (
    Document,
    InfoUnit,
    StructureNode,
)
from scinr.newton.models.file_metadata import FILE_METADATA_KEYS, FileMetadata
from scinr.newton.utils.tenancy import tenant_key

# ``d.file_<key> = $file_<key>`` for every FileMetadata key, built from the
# constant (never from data).
_FILE_METADATA_SET = ",\n".join(
    f"            d.file_{key} = $file_{key}" for key in FILE_METADATA_KEYS
) + "\n"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version resolution helpers  (require a Session, not a Transaction)
# ---------------------------------------------------------------------------


def get_next_version(session, doc_path: str, tenant_id: str) -> int:
    """Query Neo4j for the highest existing version of *doc_path* within
    *tenant_id* and return next. Versions are numbered per tenant.

    Returns 1 if the tenant has no document with that path yet.

    Args:
        session: An open Neo4j session (not a transaction).
        doc_path: Relative path of the document.
        tenant_id: Stored tenant key.
    """
    result = session.run(
        "MATCH (d:Document {tenant_id: $tenant_id, path: $path}) "
        "RETURN max(d.version) AS max_version",
        tenant_id=tenant_id,
        path=doc_path,
    )
    record = result.single()
    max_version = record["max_version"] if record else None
    return (max_version + 1) if max_version is not None else 1


def get_current_latest_version(session, doc_path: str, tenant_id: str) -> int | None:
    """Return the version of *tenant_id*'s current ``latest=true`` Document at *doc_path*.

    Returns ``None`` if the tenant has no document with that path.

    Args:
        session: An open Neo4j session (not a transaction).
        doc_path: Relative path of the document.
        tenant_id: Stored tenant key.
    """
    result = session.run(
        "MATCH (d:Document {tenant_id: $tenant_id, path: $path, latest: true}) "
        "RETURN d.version AS version",
        tenant_id=tenant_id,
        path=doc_path,
    )
    record = result.single()
    return record["version"] if record else None


def delete_document_content(tx, doc_path: str, version: int, tenant_id: str) -> None:
    """Delete all structure and annotation data for a specific document version.

    Used by the ``update_mode=True`` flow to wipe the existing content before re-inserting.
    Deletes in leaf-first order to avoid orphan nodes:

    1. InfoUnits
    2. ProposedFields → ProposedModels → SupplementaryFields
       → ComplementaryMatches → ModelDecisions
    3. LabeledEntities → ExtractionResults
    4. StructureNodes

    Args:
        tx: An open Neo4j transaction.
        doc_path: Relative path of the document.
        version: Integer version number of the document to wipe.
        tenant_id: Stored tenant key of the document to wipe.
    """
    _base = (
        "MATCH (d:Document {tenant_id: $tenant_id, path: $path, version: $version})"
        "-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)"
    )
    params = {"tenant_id": tenant_id, "path": doc_path, "version": version}

    # 1. InfoUnits
    tx.run(f"{_base}-[:HAS_INFO_UNIT]->(iu:InfoUnit) DETACH DELETE iu", **params)
    # 2a. ProposedFields
    tx.run(f"{_base}-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(:ProposedModel)-[:HAS_PROPOSED_FIELD]->(pf) DETACH DELETE pf", **params)
    # 2b. ProposedModels
    tx.run(f"{_base}-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(pm:ProposedModel) DETACH DELETE pm", **params)
    # 2c. SupplementaryFields
    tx.run(f"{_base}-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_SUPPLEMENTARY_FIELD]->(sf) DETACH DELETE sf", **params)
    # 2d. ComplementaryMatches
    tx.run(f"{_base}-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_COMPLEMENTARY_MATCH]->(cm) DETACH DELETE cm", **params)
    # 2e. ModelDecisions
    tx.run(f"{_base}-[:HAS_MODEL_DECISION]->(md:ModelDecision) DETACH DELETE md", **params)
    # 3a. LabeledEntities
    tx.run(f"{_base}-[:HAS_EXTRACTION]->(:ExtractionResult)-[:REFERENCES]->(le:LabeledEntity) DETACH DELETE le", **params)
    # 3b. ExtractionResults
    tx.run(f"{_base}-[:HAS_EXTRACTION]->(er:ExtractionResult) DETACH DELETE er", **params)
    # 4. StructureNodes
    tx.run(f"{_base} DETACH DELETE n", **params)

    logger.info(
        "delete_document_content: wiped structure for tenant=%s path=%s version=%d",
        tenant_id,
        doc_path,
        version,
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def insert_document(
    tx,
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
    """MERGE a :Document node keyed by (tenant_id, path, version).

    Sets name, path, version, load_date, latest, is_folder,
    context_instructions, tenant_id, created_by_user_id, job_id and one
    ``file_<key>`` property per key of
    :data:`~scinr.newton.models.file_metadata.FILE_METADATA_KEYS`.
    The 'latest' flag management (setting old version to latest=False and
    creating HAS_NEWER_VERSION) is handled separately by handle_versioning().

    Args:
        tx: An open Neo4j transaction.
        document_name: Display name for the document (source file name with extension, or folder name).
        doc_path: Relative path from the pipeline input root (unique per document location).
        version: Integer version number.
        is_folder: True for folder-parent documents; False for leaf documents.
        context_instructions: Optional free-text user-provided ingestion context. When None, the
            property is stored as null in Neo4j (i.e. effectively absent).
        tenant_id: Stored tenant key — part of the node's identity (MERGE key).
        created_by_user_id: Optional caller-supplied id of the user that launched the
            ingestion. Always written (stored as null when None).
        job_id: Optional caller-supplied ingestion job/run id. Always written
            (stored as null when None).
        file_metadata: Metadata of the source file (images, audio, video). Every
            ``file_*`` key is written on every ingestion, ``null`` included, so
            ``update_mode`` never leaves stale values; the property names come
            from the constant, never from data.
    """
    from datetime import datetime
    load_date = datetime.now(UTC).isoformat()
    file_values = file_metadata.model_dump() if file_metadata is not None else {}
    file_params = {f"file_{key}": file_values.get(key) for key in FILE_METADATA_KEYS}

    tx.run(
        """
        MERGE (d:Document {tenant_id: $tenant_id, path: $path, version: $version})
        SET d.name                 = $name,
            d.load_date            = $load_date,
            d.is_folder            = $is_folder,
            d.raw_file_id          = $raw_file_id,
            d.context_instructions = $context_instructions,
            d.created_by_user_id   = $created_by_user_id,
            d.job_id               = $job_id,
            d.latest               = true,
        """
        + _FILE_METADATA_SET,
        path=doc_path,
        version=version,
        name=document_name,
        load_date=load_date,
        is_folder=is_folder,
        raw_file_id=raw_file_id,
        context_instructions=context_instructions,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
        **file_params,
    )
    logger.debug(
        "Merged Document node: tenant=%s path=%s version=%s", tenant_id, doc_path, version
    )


def handle_versioning(tx, doc_path: str, new_version: int, tenant_id: str) -> None:
    """Find the previous latest version of a document and link it to the new version.

    If *tenant_id* has a Document with the same path and latest=True (and a
    different version), it is marked latest=False and linked via
    HAS_NEWER_VERSION to the newly created version. Another tenant's document
    at the same path is never touched.

    Args:
        tx: An open Neo4j transaction.
        doc_path: Relative path identifying this document location.
        new_version: The integer version number of the newly created Document node.
        tenant_id: Stored tenant key.
    """
    tx.run(
        """
        MATCH (old:Document {tenant_id: $tenant_id, path: $path, latest: true})
        WHERE old.version <> $new_version
        MATCH (new:Document {tenant_id: $tenant_id, path: $path, version: $new_version})
        SET old.latest = false
        MERGE (old)-[:HAS_NEWER_VERSION]->(new)
        """,
        tenant_id=tenant_id,
        path=doc_path,
        new_version=new_version,
    )
    logger.debug(
        "handle_versioning: checked for previous latest at path=%s (new_version=%s)",
        doc_path,
        new_version,
    )


def insert_folder_document_hierarchy(
    tx,
    folder_path: str,
    version: int,
    *,
    tenant_id: str,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """Create all ancestor folder-parent Document nodes for a given path.

    Folders belong to one tenant: each tenant gets its own folder nodes (keyed
    by ``(tenant_id, path, version)``) even when two tenants use the same path.

    Given a path like "ModuloA/SubModulo/doc", creates::

      (:Document {path:"ModuloA", name:"ModuloA", is_folder:True})
      (:Document {path:"ModuloA/SubModulo", name:"SubModulo", is_folder:True})

    and links them::

      (ModuloA)-[:IS_COMPOSED_OF]->(SubModulo)

    Does NOT create the leaf document itself (that is done by insert_document).

    Args:
        tx: An open Neo4j transaction.
        folder_path: Relative folder path (e.g. "ModuloA/SubModulo"). Use the doc_path's
            parent: e.g. if doc_path="ModuloA/SubModulo/doc", pass "ModuloA/SubModulo".
        version: Integer version number shared across this ingestion run.
        tenant_id: Stored tenant key — part of every folder node's identity.
        created_by_user_id: Optional caller-supplied id of the user that launched the
            ingestion. Always written (stored as null when None).
        job_id: Optional caller-supplied ingestion job/run id. Always written
            (stored as null when None).
    """
    from datetime import datetime
    load_date = datetime.now(UTC).isoformat()

    parts = folder_path.split("/")
    for depth, _ in enumerate(parts):
        current_path = "/".join(parts[: depth + 1])
        folder_name = parts[depth]

        tx.run(
            """
            MERGE (f:Document {tenant_id: $tenant_id, path: $path, version: $version})
            SET f.name               = $name,
                f.load_date          = $load_date,
                f.is_folder          = true,
                f.created_by_user_id = $created_by_user_id,
                f.job_id             = $job_id,
                f.latest             = true
            """,
            path=current_path,
            version=version,
            name=folder_name,
            load_date=load_date,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )
        logger.debug("Merged folder Document: path=%s version=%s", current_path, version)

        # Handle versioning for this folder level too
        handle_versioning(tx, current_path, version, tenant_id)

        # Link parent → child if depth > 0
        if depth > 0:
            parent_path = "/".join(parts[:depth])
            tx.run(
                """
                MATCH (parent:Document {tenant_id: $tenant_id, path: $parent_path, version: $version})
                MATCH (child:Document {tenant_id: $tenant_id, path: $child_path, version: $version})
                MERGE (parent)-[:IS_COMPOSED_OF]->(child)
                """,
                tenant_id=tenant_id,
                parent_path=parent_path,
                child_path=current_path,
                version=version,
            )
            logger.debug(
                "Linked folder %s -[:IS_COMPOSED_OF]-> %s",
                parent_path,
                current_path,
            )


def link_leaf_to_folder(
    tx,
    doc_path: str,
    version: int,
    tenant_id: str,
) -> None:
    """Link a leaf document to its immediate folder parent (if any) of the same tenant.

    Args:
        tx: An open Neo4j transaction.
        doc_path: Full relative path of the leaf document (e.g. "ModuloA/SubModulo/doc_a").
        version: Integer version number.
        tenant_id: Stored tenant key.
    """
    # folder_path is everything before the last "/"
    if "/" not in doc_path:
        return  # No parent folder, this is a root document

    folder_path = doc_path.rsplit("/", 1)[0]

    tx.run(
        """
        MATCH (parent:Document {tenant_id: $tenant_id, path: $folder_path, version: $version})
        MATCH (leaf:Document {tenant_id: $tenant_id, path: $doc_path, version: $version})
        MERGE (parent)-[:IS_COMPOSED_OF]->(leaf)
        """,
        tenant_id=tenant_id,
        folder_path=folder_path,
        doc_path=doc_path,
        version=version,
    )
    logger.debug(
        "Linked folder %s -[:IS_COMPOSED_OF]-> leaf %s",
        folder_path,
        doc_path,
    )


# ---------------------------------------------------------------------------
# InfoUnit  (MERGE by uid derived from info_unit_id)
# ---------------------------------------------------------------------------


def insert_info_unit(
    tx,
    info_unit: InfoUnit,
    parent_node_id: str,
    order: int = 0,
) -> None:
    """MERGE an :InfoUnit node and link it to its parent :StructureNode.

    Uses the pre-computed ``info_unit_id`` as the Neo4j uid to MERGE,
    preventing duplicate nodes on re-ingestion. The node is only created
    when the parent :StructureNode already exists — no orphan nodes are
    produced if the parent is missing.

    Args:
        tx: An open Neo4j transaction.
        info_unit: The :class:`~scinr.newton.models.document_structure.InfoUnit`
            to persist.
        parent_node_id: The composite Neo4j ``id`` of the parent :StructureNode
            (``"{tenant_key}::{doc_path}::{version}::{ancestor_path/node_id}"``).
            For example ``"acme::Amox 500 mg/0000/m3::2::5_3/5_3_1"`` (nested) or
            ``"__public__::Amox 500 mg/0000/m3::2::5_3"`` (root-level node of a
            public document). The InfoUnit uid is hashed from it, so it is
            tenant-scoped as well.
    """
    uid = _make_uid(parent_node_id, info_unit.title, info_unit.description)
    tx.run(
        """
        MATCH (n:StructureNode {id: $node_id})
        MERGE (u:InfoUnit {uid: $uid})
        SET  u.info_unit_id       = $info_unit_id,
             u.title              = $title,
             u.description        = $description,
             u.order              = $order,
             u.tenant_id          = n.tenant_id,
             u.created_by_user_id = n.created_by_user_id,
             u.job_id             = n.job_id
        MERGE (n)-[:HAS_INFO_UNIT]->(u)
        """,
        node_id=parent_node_id,
        uid=uid,
        info_unit_id=uid,
        title=info_unit.title,
        description=info_unit.description,
        order=order,
    )
    logger.debug(
        "Merged InfoUnit %s under StructureNode %s",
        uid,
        parent_node_id,
    )


# ---------------------------------------------------------------------------
# StructureNode  (recursive)
# ---------------------------------------------------------------------------


def insert_structure_node(
    tx,
    node: StructureNode,
    doc_path: str,
    version: int,
    parent_id: str | None = None,
    node_path: str = "",
    *,
    tenant_id: str,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """MERGE a :StructureNode (with its role label), attach it to its parent,
    then recursively insert children and info units.

    Each :StructureNode receives **two** labels: the generic ``:StructureNode``
    and the role-specific label derived from :data:`ROLE_TO_LABEL`
    (e.g. ``:Section``, ``:Table``).

    The composite ``id`` is
    ``"{tenant_key}::{doc_path}::{version}::{ancestor_path/node_id}"`` (see
    :func:`make_structure_node_id`). For root-level nodes (no ancestor path) it
    reduces to ``"{tenant_key}::{doc_path}::{version}::{node_id}"``.

    Examples: ``"acme::Amox 500 mg/0000/m3::2::5_3/5_3_1"`` (nested),
    ``"__public__::Amox 500 mg/0000/m3::2::5_3"`` (root-level node, public document).

    Relationship rules:

    - ``parent_id is None``  → ``(:Document {tenant_id, path: doc_path, version})-[:HAS_STRUCTURE]->(:StructureNode)``
    - ``parent_id is not None`` → ``(:StructureNode {id: parent_id})-[:HAS_CHILD]->(:StructureNode)``

    Args:
        tx: An open Neo4j transaction.
        node: The :class:`~scinr.newton.models.document_structure.StructureNode`
            to persist.
        doc_path: Document path used to build all composite ids.
        version: Integer version number used to build all composite ids and look up the Document node.
        parent_id: The composite Neo4j ``id`` of the parent :StructureNode, or ``None``
            when this node is a direct child of the :Document node.
        node_path: Slash-separated ancestor ``node_id`` values leading up to (but NOT
            including) the current node.  Empty string for root-level nodes.
        tenant_id: Stored tenant key of the owning :Document. Prefixes the
            composite id, selects the :Document to attach to, and is written
            onto the node.
        tenant_id, created_by_user_id, job_id: Denormalized copies of the owning
            :Document's provenance, written verbatim onto every :StructureNode
            (scalar — a StructureNode belongs to exactly one Document, never
            merged across documents). Lets navigation filter structure nodes by
            tenant/user/job without a traversal back up to :Document.
    """
    role_val = node.role if isinstance(node.role, str) else node.role.value
    role_label = ROLE_TO_LABEL[role_val]
    current_node_path = f"{node_path}/{node.node_id}" if node_path else node.node_id
    composite_id = make_structure_node_id(tenant_id, doc_path, version, current_node_path)

    # MERGE on id only, then SET the role label separately.  Including
    # role_label in the MERGE pattern would cause a constraint violation when
    # the same id is re-encountered with a different label during upserts.
    # role_label is derived from a controlled enum, never from user input.
    tx.run(
        f"""
        MERGE (n:StructureNode {{id: $id}})
        SET   n:{role_label},
              n.node_id            = $node_id,
              n.role               = $role,
              n.appearance_order   = $appearance_order,
              n.title              = $title,
              n.theme              = $theme,
              n.source_page_ids    = $source_page_ids,
              n.source_paragraphs           = $source_paragraphs,
              n.paragraph_segmenter_version = $paragraph_segmenter_version,
              n.info_units_mode             = $info_units_mode,
              n.tenant_id          = $tenant_id,
              n.created_by_user_id = $created_by_user_id,
              n.job_id             = $job_id
        """,
        id=composite_id,
        node_id=node.node_id,
        title=node.title,
        role=role_val,
        appearance_order=node.appearance_order,
        theme=getattr(node, "theme", "default"),
        source_page_ids=node.source_page_ids,
        source_paragraphs=node.source_paragraphs,
        paragraph_segmenter_version=node.paragraph_segmenter_version,
        info_units_mode=node.info_units_mode,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
    )
    logger.debug("Merged StructureNode %s (role=%s)", composite_id, role_val)

    # Create relationship to parent
    if parent_id is None:
        tx.run(
            """
            MATCH (d:Document {tenant_id: $tenant_id, path: $doc_path, version: $version})
            MATCH (n:StructureNode {id: $node_id})
            MERGE (d)-[:HAS_STRUCTURE]->(n)
            """,
            tenant_id=tenant_id,
            doc_path=doc_path,
            version=version,
            node_id=composite_id,
        )
        logger.debug(
            "Linked Document '%s' (v=%s) -[:HAS_STRUCTURE]-> %s",
            doc_path,
            version,
            composite_id,
        )
    else:
        tx.run(
            """
            MATCH (p:StructureNode {id: $parent_id})
            MATCH (n:StructureNode {id: $node_id})
            MERGE (p)-[:HAS_CHILD]->(n)
            """,
            parent_id=parent_id,
            node_id=composite_id,
        )
        logger.debug(
            "Linked StructureNode %s -[:HAS_CHILD]-> %s", parent_id, composite_id
        )

    # InfoUnits attached to this node
    for idx, info_unit in enumerate(node.info_units):
        insert_info_unit(tx, info_unit, composite_id, order=idx)

    # Recursively process children, passing this node's composite id as parent
    # and this node's full path so children can build their own composite ids.
    for child in node.children:
        insert_structure_node(
            tx,
            child,
            doc_path=doc_path,
            version=version,
            parent_id=composite_id,
            node_path=current_node_path,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def insert_document_graph(
    tx,
    doc: Document,
    resolved_version: int,
    update_mode: bool = False,
) -> None:
    """Insert (or update) the complete document graph in Neo4j.

    The version to use is always supplied externally via *resolved_version*
    (computed by the loader before opening the write transaction). The
    ``doc.version`` field is ignored.

    Steps:
    1. Resolve doc_path (falls back to document_name if not set).
    2. If *update_mode*, delete all existing structure for this version first.
    3. Create all ancestor folder-parent Document nodes (if in a subfolder).
    4. Handle versioning: mark previous latest as old, create HAS_NEWER_VERSION
       (skipped in update_mode since the version does not change).
    5. MERGE the leaf Document node with the resolved version.
    6. Link leaf to its immediate folder parent.
    7. Recursively insert all StructureNodes.

    Args:
        tx: An open Neo4j transaction.
        doc: The fully validated Document to persist.
        resolved_version: The integer version to assign. Always provided by the caller;
            never derived from ``doc.version``.
        update_mode: If True, existing structure for this version is deleted before
            re-insertion. No new version is created; no HAS_NEWER_VERSION link.

    ``doc.tenant_id`` is normalized here, once, with
    :func:`~scinr.newton.utils.tenancy.tenant_key` (``None`` → ``"__public__"``)
    and the stored key selects/creates every :Document of this call (leaf and
    folder parents), prefixes every StructureNode id and is copied onto every
    node below. *doc* itself is not modified. ``doc.created_by_user_id`` and
    ``doc.job_id`` are written verbatim (null when None), mirroring
    ``doc.context_instructions``.

    Raises:
        ValueError: If ``doc.tenant_id`` is empty.
    """
    doc_path = doc.doc_path if doc.doc_path else doc.document_name
    tenant = tenant_key(doc.tenant_id)

    logger.info(
        "Inserting document graph for '%s' (tenant=%s, path=%s, version=%d, update=%s, %d root nodes)",
        doc.document_name,
        tenant,
        doc_path,
        resolved_version,
        update_mode,
        len(doc.document_structure),
    )

    # 1. Wipe existing structure when updating in-place
    if update_mode:
        delete_document_content(tx, doc_path, resolved_version, tenant)

    # 2. Create all ancestor folder-parent nodes (if any)
    if "/" in doc_path:
        folder_path = doc_path.rsplit("/", 1)[0]
        insert_folder_document_hierarchy(
            tx,
            folder_path,
            resolved_version,
            tenant_id=tenant,
            created_by_user_id=doc.created_by_user_id,
            job_id=doc.job_id,
        )

    # 3. Insert (or update) the leaf Document node itself
    insert_document(
        tx,
        doc.document_name,
        doc_path,
        resolved_version,
        doc.raw_file_id,
        is_folder=False,
        context_instructions=doc.context_instructions,
        tenant_id=tenant,
        created_by_user_id=doc.created_by_user_id,
        job_id=doc.job_id,
        file_metadata=doc.file_metadata,
    )

    # 4. Handle versioning (only for normal loads, not updates)
    if not update_mode:
        handle_versioning(tx, doc_path, resolved_version, tenant)

    # 5. Link leaf to its immediate folder parent
    link_leaf_to_folder(tx, doc_path, resolved_version, tenant)

    # 6. Insert all StructureNodes
    for root_node in doc.document_structure:
        insert_structure_node(
            tx,
            root_node,
            doc_path=doc_path,
            version=resolved_version,
            parent_id=None,
            tenant_id=tenant,
            created_by_user_id=doc.created_by_user_id,
            job_id=doc.job_id,
        )

    logger.info(
        "Document graph insertion complete for '%s' (path=%s, version=%d).",
        doc.document_name,
        doc_path,
        resolved_version,
    )


# ---------------------------------------------------------------------------
# Batch version
# ---------------------------------------------------------------------------


def _resolve_batch_version(
    session,
    all_paths: Sequence[str],
    update_mode: bool,
    tenant_ids: Iterable[str | None],
) -> int:
    """Compute a single shared integer version for a batch of documents.

    Versions are numbered per tenant, so only the batch's own tenants'
    documents are considered — another tenant's documents at the same paths
    never bump (or, in update mode, select) this batch's version.

    In normal mode:  max existing version across all paths + 1  (or 1 if none).
    In update mode:  max current latest version across all paths (or 1 if none).

    Args:
        session: An open Neo4j session.
        all_paths: All document paths in the batch (leaves + ancestor folders).
        update_mode: True → return the current latest version (no increment);
            False → return the next version (increment).
        tenant_ids: Tenant(s) the batch's documents will be written under, in
            public-API form (``None`` = public). Usually one; several only when
            documents carry different tenants baked into their
            ``extract-*.json``.
    """
    if not all_paths:
        return 1

    stored_tenants = sorted({tenant_key(t) for t in tenant_ids})
    if update_mode:
        result = session.run(
            "MATCH (d:Document {latest: true}) "
            "WHERE d.tenant_id IN $tenant_ids AND d.path IN $paths "
            "RETURN max(d.version) AS max_version",
            tenant_ids=stored_tenants,
            paths=all_paths,
        )
    else:
        result = session.run(
            "MATCH (d:Document) WHERE d.tenant_id IN $tenant_ids AND d.path IN $paths "
            "RETURN max(d.version) AS max_version",
            tenant_ids=stored_tenants,
            paths=all_paths,
        )

    record = result.single()
    max_version = record["max_version"] if record else None

    if update_mode:
        return max_version if max_version is not None else 1
    else:
        return (max_version + 1) if max_version is not None else 1


# ---------------------------------------------------------------------------
# Document selection (the "latest" selector)
# ---------------------------------------------------------------------------


def latest_document_pattern(alias: str, selector: DocumentSelector) -> tuple[str, dict]:
    """Cypher node pattern (and its parameters) matching the ``latest``
    :Document(s) that *selector* selects.

    The selector's ``tenant_id`` is in public-API form (``None`` = public); its
    ``path`` is the preferred key, or — when only ``document_name`` is given —
    every document with that name.

    Returns e.g. ``("(d:Document {tenant_id: $tenant_id, path: $doc_selector,
    latest: true})", {"tenant_id": "acme", "doc_selector": "A/b"})``.

    Raises:
        ValueError: If the selector is not a ``latest`` one, neither ``path``
            nor ``document_name`` is given, or ``tenant_id`` is invalid (see
            ``utils.tenancy.tenant_key``).
    """
    if not selector.latest:
        raise ValueError("A latest-document selector (latest=True) is required.")
    if selector.path is None and selector.document_name is None:
        raise ValueError("Either doc_path or document_name is required.")
    # Hard-coded property name, never user input.
    prop, value = (
        ("path", selector.path) if selector.path is not None else ("name", selector.document_name)
    )
    pattern = f"({alias}:Document {{tenant_id: $tenant_id, {prop}: $doc_selector, latest: true}})"
    return pattern, {"tenant_id": tenant_key(selector.tenant_id), "doc_selector": value}


# Root and leaves are both pinned to the tenant: IS_COMPOSED_OF never crosses
# tenants by construction (each tenant has its own folders), the leaf filter
# is defence in depth.
_LEAVES_QUERY = """
MATCH {root}
OPTIONAL MATCH (root)-[:IS_COMPOSED_OF*1..]->(leaf:Document {{tenant_id: $tenant_id, latest: true}})
WHERE NOT (leaf)-[:IS_COMPOSED_OF]->(:Document)
WITH root, collect(DISTINCT leaf) AS leaves
RETURN CASE
  WHEN size(leaves) > 0 THEN [l IN leaves | {{name: l.name, path: l.path}}]
  ELSE [{{name: root.name, path: root.path}}]
END AS docs
ORDER BY root.path
"""


def _leaves_query_and_params(selector: DocumentSelector) -> tuple[str, dict]:
    root, params = latest_document_pattern("root", selector)
    return _LEAVES_QUERY.format(root=root), params


def _collect_leaves(records: list[list[dict]]) -> list[LeafDocument]:
    """Leaves of every matched root, deduplicated by path (first occurrence wins)."""
    leaves: dict[str, LeafDocument] = {}
    for docs in records:
        for d in docs:
            leaves.setdefault(d["path"], LeafDocument(d["name"], d["path"]))
    return list(leaves.values())


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

_PREFLIGHT_REPLACES_QUERY = """
            MATCH (d:Document {tenant_id: $tenant_id, name: $name, latest: true})
            RETURN d.path AS path, d.version AS version
            """


class Neo4jDocumentOps(DocumentOps):
    """:Document operations over Neo4j.

    The synchronous methods use the driver of the current ``sync_scope()`` or,
    outside one, open and close their own. The asynchronous ones use the shared
    async driver.
    """

    @neo4j_errors()
    def next_version(self, path: str, tenant: str) -> int:
        """Next version number of *path* within *tenant* (1 if the tenant has none).

        Args:
            path: Relative path of the document.
            tenant: Stored tenant key.

        Raises:
            GraphError: If the engine fails.
        """
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                return get_next_version(session, path, tenant)

    @neo4j_errors()
    def current_latest_version(self, path: str, tenant: str) -> int | None:
        """Version of *tenant*'s ``latest`` document at *path*, or ``None`` if there is none.

        Args:
            path: Relative path of the document.
            tenant: Stored tenant key.

        Raises:
            GraphError: If the engine fails.
        """
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                return get_current_latest_version(session, path, tenant)

    @neo4j_errors()
    def resolve_batch_version(
        self, paths: Sequence[str], tenants: Iterable[str | None], update_mode: bool
    ) -> int:
        """The single version shared by every document of one batch.

        Versions are numbered per tenant: only the batch's own tenants'
        documents are considered. Normal mode returns the highest existing
        version across *paths* + 1 (1 if none); *update_mode* returns the
        highest current ``latest`` version (1 if none).

        Args:
            paths: All document paths of the batch (leaves + ancestor folders).
            tenants: Tenant(s) the batch will be written under, in public-API
                form (``None`` = public).
            update_mode: Return the current latest version instead of the next.

        Raises:
            GraphError: If the engine fails.
        """
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                return _resolve_batch_version(session, paths, update_mode, tenants)

    def insert_document_graph(self, doc: Document, version: int, update_mode: bool) -> None:
        """Insert (or update) the complete document graph in one explicit transaction, with retry.

        The retry wraps the whole transaction. Commit on success, rollback on
        any failure.

        Args:
            doc: The validated document. ``doc.tenant_id`` is normalized here.
            version: The resolved version; ``doc.version`` is ignored.
            update_mode: Wipe the existing structure of *version* first.

        Raises:
            DocumentVersionConflict: The version was already ingested by a
                concurrent process.
            GraphError: If the engine fails.
            ValueError: If ``doc.tenant_id`` is empty.
        """

        def _do_insert(driver) -> None:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                with session.begin_transaction() as tx:
                    try:
                        insert_document_graph(tx, doc, version, update_mode=update_mode)
                        tx.commit()
                        logger.info(
                            "Transaction committed for document: %s (v%d)",
                            doc.document_name,
                            version,
                        )
                    except ConstraintError:
                        tx.rollback()
                        raise
                    except Exception:
                        tx.rollback()
                        logger.exception(
                            "Transaction rolled back for document: %s", doc.document_name
                        )
                        raise

        with neo4j_errors(
            version_conflict=(
                f"Version conflict: version {version} was already "
                f"ingested by a concurrent process. Retry the ingestion."
            )
        ):
            with borrow_sync_driver() as driver:
                with_neo4j_retry_sync(lambda: _do_insert(driver))

    @neo4j_errors()
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
        """Create the folder hierarchy, the leaf :Document, its versioning and its folder link.

        One explicit transaction, no retry (as the tabular agent has always
        done): commit on success, rollback on any failure.

        Args:
            document_name: Display name (source file name with extension).
            doc_path: Relative path, the document key.
            version: The batch version.
            raw_file_id: Id of the stored raw file, or ``""``.
            tenant: Stored tenant key.
            created_by_user_id: Provenance written on the document and its folders.
            job_id: Provenance written on the document and its folders.

        Raises:
            GraphError: If the engine fails.
        """
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                with session.begin_transaction() as tx:
                    try:
                        if "/" in doc_path:
                            folder_path = doc_path.rsplit("/", 1)[0]
                            insert_folder_document_hierarchy(
                                tx,
                                folder_path,
                                version,
                                tenant_id=tenant,
                                created_by_user_id=created_by_user_id,
                                job_id=job_id,
                            )
                        insert_document(
                            tx,
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
                        handle_versioning(tx, doc_path, version, tenant)
                        link_leaf_to_folder(tx, doc_path, version, tenant)
                        tx.commit()
                        logger.info(
                            "tabular agent: Document node created for '%s' (v%d)",
                            doc_path,
                            version,
                        )
                    except Exception as exc:
                        tx.rollback()
                        logger.error(
                            "tabular agent: Document creation failed for '%s': %s",
                            doc_path,
                            exc,
                        )
                        raise

    @neo4j_errors()
    def preflight_replaces(self, name: str, tenant: str) -> list[dict]:
        """``{"path", "version"}`` of every ``latest`` document of *tenant* named *name*.

        Another tenant's document with the same name is never considered.

        Args:
            name: The ``name`` property of the document to replace.
            tenant: Stored tenant key.

        Raises:
            GraphError: If the engine fails.
        """
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                result = session.run(_PREFLIGHT_REPLACES_QUERY, tenant_id=tenant, name=name)
                return [dict(r) for r in result]

    @neo4j_errors()
    def apply_replacement(
        self,
        replaces_name: str,
        new_root_doc_names: Sequence[str],
        *,
        tenant: str,
        replaced_path: str | None,
    ) -> None:
        """Link the replaced document to the new root document(s) with ``HAS_NEWER_VERSION``.

        Finds the new roots (``latest`` documents among *new_root_doc_names*
        with no ``IS_COMPOSED_OF`` parent), then, for each, sets the old
        document ``latest = false`` and links it. Old and new are looked up
        within *tenant* only. A warning is logged and nothing is written when
        no new root is found.

        Args:
            replaces_name: The ``name`` of the old document.
            new_root_doc_names: Names of the newly ingested documents.
            tenant: Stored tenant key of the old and the new documents.
            replaced_path: Path of the old document. When given, the old
                document is selected by path instead of by name.

        Raises:
            GraphError: If the engine fails.
        """
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                # Find the new root documents (those just ingested with no IS_COMPOSED_OF parent)
                result = session.run(
                    """
                    MATCH (d:Document {tenant_id: $tenant_id, latest: true})
                    WHERE d.name IN $names AND NOT ()-[:IS_COMPOSED_OF]->(d)
                    RETURN d.path AS path, d.version AS version, d.name AS name
                    """,
                    tenant_id=tenant,
                    names=new_root_doc_names,
                )
                new_roots = [dict(r) for r in result]

            if not new_roots:
                logger.warning(
                    "apply_replacement: could not find any root documents among %s; skipping.",
                    new_root_doc_names,
                )
                return
            # Select the old document by path when known (a name is not unique within
            # a tenant); never the new document itself (same name, or same path when
            # ingestion already versioned it).
            old_filter = "path: $old_path" if replaced_path is not None else "name: $old_name"
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                for new_root in new_roots:
                    session.run(
                        f"""
                        MATCH (old:Document {{tenant_id: $tenant_id, {old_filter}, latest: true}})
                        MATCH (new:Document {{tenant_id: $tenant_id, path: $new_path, version: $new_version}})
                        WHERE old <> new
                        SET old.latest = false
                        MERGE (old)-[:HAS_NEWER_VERSION]->(new)
                        """,
                        tenant_id=tenant,
                        old_path=replaced_path,
                        old_name=replaces_name,
                        new_path=new_root["path"],
                        new_version=new_root["version"],
                    )
                    logger.info(
                        "apply_replacement: linked '%s' (latest=false) -[:HAS_NEWER_VERSION]-> '%s' (path=%s, v=%s)",
                        replaces_name,
                        new_root["name"],
                        new_root["path"],
                        new_root["version"],
                    )

    @neo4j_errors()
    def resolve_leaf_documents(self, selector: DocumentSelector) -> list[LeafDocument]:
        """Leaf documents reachable via ``IS_COMPOSED_OF`` from the selected ``latest`` root(s).

        The traversal runs entirely in Cypher (variable-length path match). A
        root with no children is its own leaf. Another tenant's documents are
        never returned. Leaves are deduplicated by path and ordered by root
        path; the list is empty when no root matches.

        Args:
            selector: A ``latest`` selector (``tenant_id`` in public-API form,
                ``path`` preferred, else ``document_name``).

        Raises:
            ValueError: If the selector is invalid (see :func:`latest_document_pattern`).
            GraphError: If the engine fails.
        """
        query, params = _leaves_query_and_params(selector)
        with borrow_sync_driver() as driver:
            cfg = get_config()
            with driver.session(database=cfg.neo4j_database) as session:
                result = session.run(query, **params)
                records = [row["docs"] for row in result.data()]
        return _collect_leaves(records)

    @neo4j_errors()
    async def resolve_leaf_documents_async(self, selector: DocumentSelector) -> list[LeafDocument]:
        """Async version of :meth:`resolve_leaf_documents` (shared async driver).

        Raises:
            ValueError: If the selector is invalid (see :func:`latest_document_pattern`).
            GraphError: If the engine fails.
        """
        query, params = _leaves_query_and_params(selector)
        driver = _driver.get_async_driver()
        cfg = get_config()
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(query, **params)
            records = [row["docs"] for row in await result.data()]
        return _collect_leaves(records)

    @neo4j_errors()
    async def count_latest_documents(self, selector: DocumentSelector) -> int:
        """How many ``latest`` documents *selector* selects (pre-check of the agents).

        Raises:
            ValueError: If the selector is invalid (see :func:`latest_document_pattern`).
            GraphError: If the engine fails.
        """
        pattern, params = latest_document_pattern("d", selector)
        driver = _driver.get_async_driver()
        cfg = get_config()
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(f"MATCH {pattern} RETURN count(d) AS n", **params)
            row = await result.single()
        return row["n"]
