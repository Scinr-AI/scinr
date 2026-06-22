"""
ingest/nodes.py — Node and relationship insertion functions for Neo4j.

All functions receive an open :class:`neo4j.Transaction` and one or more
Pydantic model instances. The insertion strategy is:

  MERGE — nodes with a unique composite key (Document, StructureNode) or a
          deterministic uid (InfoUnit). Prevents duplicates on re-ingestion.

Relationships are established with MATCH … MERGE to avoid duplicates while
being tolerant of re-runs.

Child nodes (InfoUnit) are only created when their parent node already exists
(MATCH parent first), preventing orphan nodes when a parent is missing.

``insert_document_graph`` receives the resolved integer version externally
(via *resolved_version*) rather than reading it from ``doc.version``.

StructureNode composite id:  ``{doc_path}::{version}::{ancestor_path/node_id}``
InfoUnit uid:                ``info_unit_id`` field (16-char SHA-256 hex)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC

from scinr.newton.models.document_structure import (
    Document,
    InfoUnit,
    NodeRole,
    StructureNode,
)

logger = logging.getLogger(__name__)


def _make_uid(*parts: str) -> str:
    """Compute a deterministic 16-char SHA-256 hex digest from one or more string parts."""
    payload = "||".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Role → Neo4j extra label mapping
# The f-string approach is safe because the value always comes from a
# validated NodeRole enum — never from user-supplied strings.
# ---------------------------------------------------------------------------
ROLE_TO_LABEL: dict[str, str] = {
    "section": "Section",
    "subsection": "Subsection",
    "table": "Table",
    "freeform_block": "FreeformBlock",
    "field_group": "FieldGroup",
    "appendix": "Appendix",
    "row": "Row",
}

assert set(ROLE_TO_LABEL.keys()) == {e.value for e in NodeRole}, (
    f"ROLE_TO_LABEL is out of sync with NodeRole enum. "
    f"Missing: {set(e.value for e in NodeRole) - set(ROLE_TO_LABEL.keys())}"
)


# ---------------------------------------------------------------------------
# Version resolution helpers  (require a Session, not a Transaction)
# ---------------------------------------------------------------------------


def get_next_version(session, doc_path: str) -> int:
    """Query Neo4j for the highest existing version of *doc_path* and return next.

    Returns 1 if no document with that path exists yet.

    Parameters
    ----------
    session:
        An open Neo4j session (not a transaction).
    doc_path:
        Relative path of the document.
    """
    result = session.run(
        "MATCH (d:Document {path: $path}) RETURN max(d.version) AS max_version",
        path=doc_path,
    )
    record = result.single()
    max_version = record["max_version"] if record else None
    return (max_version + 1) if max_version is not None else 1


def get_current_latest_version(session, doc_path: str) -> int | None:
    """Return the version of the current ``latest=true`` Document at *doc_path*.

    Returns ``None`` if no document with that path exists.

    Parameters
    ----------
    session:
        An open Neo4j session (not a transaction).
    doc_path:
        Relative path of the document.
    """
    result = session.run(
        "MATCH (d:Document {path: $path, latest: true}) RETURN d.version AS version",
        path=doc_path,
    )
    record = result.single()
    return record["version"] if record else None


def delete_document_content(tx, doc_path: str, version: int) -> None:
    """Delete all structure and annotation data for a specific document version.

    Used by the ``--update`` flow to wipe the existing content before re-inserting.
    Deletes in leaf-first order to avoid orphan nodes:

    1. InfoUnits
    2. ProposedFields → ProposedModels → SupplementaryFields
       → ComplementaryMatches → ModelDecisions
    3. LabeledEntities → ExtractionResults
    4. StructureNodes

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    doc_path:
        Relative path of the document.
    version:
        Integer version number of the document to wipe.
    """
    _base = "MATCH (d:Document {path: $path, version: $version})-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)"
    params = {"path": doc_path, "version": version}

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
        "delete_document_content: wiped structure for path=%s version=%d",
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
) -> None:
    """MERGE a :Document node keyed by (path, version).

    Sets name, path, version, load_date, latest, is_folder, and
    context_instructions.
    The 'latest' flag management (setting old version to latest=False and
    creating HAS_NEWER_VERSION) is handled separately by handle_versioning().

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    document_name:
        Display name for the document (file stem or folder name).
    doc_path:
        Relative path from the pipeline input root (unique per document location).
    version:
        Integer version number.
    is_folder:
        True for folder-parent documents; False for leaf documents.
    context_instructions:
        Optional free-text user-provided ingestion context. When None, the
        property is stored as null in Neo4j (i.e. effectively absent).
    """
    from datetime import datetime
    load_date = datetime.now(UTC).isoformat()

    tx.run(
        """
        MERGE (d:Document {path: $path, version: $version})
        SET d.name                 = $name,
            d.load_date            = $load_date,
            d.is_folder            = $is_folder,
            d.raw_file_id          = $raw_file_id,
            d.context_instructions = $context_instructions,
            d.latest               = true
        """,
        path=doc_path,
        version=version,
        name=document_name,
        load_date=load_date,
        is_folder=is_folder,
        raw_file_id=raw_file_id,
        context_instructions=context_instructions,
    )
    logger.debug("Merged Document node: path=%s version=%s", doc_path, version)


def handle_versioning(tx, doc_path: str, new_version: int) -> None:
    """Find the previous latest version of a document and link it to the new version.

    If a Document with the same path exists with latest=True (and a different version),
    it is marked latest=False and linked via HAS_NEWER_VERSION to the newly created version.

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    doc_path:
        Relative path identifying this document location.
    new_version:
        The integer version number of the newly created Document node.
    """
    tx.run(
        """
        MATCH (old:Document {path: $path, latest: true})
        WHERE old.version <> $new_version
        MATCH (new:Document {path: $path, version: $new_version})
        SET old.latest = false
        MERGE (old)-[:HAS_NEWER_VERSION]->(new)
        """,
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
) -> None:
    """Create all ancestor folder-parent Document nodes for a given path.

    Given a path like "ModuloA/SubModulo/doc", creates::

      (:Document {path:"ModuloA", name:"ModuloA", is_folder:True})
      (:Document {path:"ModuloA/SubModulo", name:"SubModulo", is_folder:True})

    and links them::

      (ModuloA)-[:IS_COMPOSED_OF]->(SubModulo)

    Does NOT create the leaf document itself (that is done by insert_document).

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    folder_path:
        Relative folder path (e.g. "ModuloA/SubModulo"). Use the doc_path's
        parent: e.g. if doc_path="ModuloA/SubModulo/doc", pass "ModuloA/SubModulo".
    version:
        Integer version number shared across this ingestion run.
    """
    from datetime import datetime
    load_date = datetime.now(UTC).isoformat()

    parts = folder_path.split("/")
    for depth, _ in enumerate(parts):
        current_path = "/".join(parts[: depth + 1])
        folder_name = parts[depth]

        tx.run(
            """
            MERGE (f:Document {path: $path, version: $version})
            SET f.name      = $name,
                f.load_date = $load_date,
                f.is_folder = true,
                f.latest    = true
            """,
            path=current_path,
            version=version,
            name=folder_name,
            load_date=load_date,
        )
        logger.debug("Merged folder Document: path=%s version=%s", current_path, version)

        # Handle versioning for this folder level too
        handle_versioning(tx, current_path, version)

        # Link parent → child if depth > 0
        if depth > 0:
            parent_path = "/".join(parts[:depth])
            tx.run(
                """
                MATCH (parent:Document {path: $parent_path, version: $version})
                MATCH (child:Document {path: $child_path, version: $version})
                MERGE (parent)-[:IS_COMPOSED_OF]->(child)
                """,
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
) -> None:
    """Link a leaf document to its immediate folder parent (if any).

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    doc_path:
        Full relative path of the leaf document (e.g. "ModuloA/SubModulo/doc_a").
    version:
        Integer version number.
    """
    # folder_path is everything before the last "/"
    if "/" not in doc_path:
        return  # No parent folder, this is a root document

    folder_path = doc_path.rsplit("/", 1)[0]

    tx.run(
        """
        MATCH (parent:Document {path: $folder_path, version: $version})
        MATCH (leaf:Document {path: $doc_path, version: $version})
        MERGE (parent)-[:IS_COMPOSED_OF]->(leaf)
        """,
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

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    info_unit:
        The :class:`~scinr.newton.models.document_structure.InfoUnit`
        to persist.
    parent_node_id:
        The composite Neo4j ``id`` of the parent :StructureNode
        (``"{doc_path}::{version}::{ancestor_path/node_id}"``).
        For example ``"Amox 500 mg/0000/m3::2::5_3/5_3_1"`` (nested) or
        ``"Amox 500 mg/0000/m3::2::5_3"`` (root-level node).
    """
    uid = _make_uid(parent_node_id, info_unit.title, info_unit.description)
    tx.run(
        """
        MATCH (n:StructureNode {id: $node_id})
        MERGE (u:InfoUnit {uid: $uid})
        SET  u.info_unit_id = $info_unit_id,
                      u.title        = $title,
                      u.description  = $description,
                      u.order        = $order
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
) -> None:
    """MERGE a :StructureNode (with its role label), attach it to its parent,
    then recursively insert children and info units.

    Each :StructureNode receives **two** labels: the generic ``:StructureNode``
    and the role-specific label derived from :data:`ROLE_TO_LABEL`
    (e.g. ``:Section``, ``:Table``).

    The composite ``id`` is ``"{doc_path}::{version}::{ancestor_path/node_id}"``.
    For root-level nodes (no ancestor path) it reduces to
    ``"{doc_path}::{version}::{node_id}"``.

    Examples: ``"Amox 500 mg/0000/m3::2::5_3/5_3_1"`` (nested),
    ``"Amox 500 mg/0000/m3::2::5_3"`` (root-level node).

    Relationship rules:

    - ``parent_id is None``  → ``(:Document {path: doc_path, version: version})-[:HAS_STRUCTURE]->(:StructureNode)``
    - ``parent_id is not None`` → ``(:StructureNode {id: parent_id})-[:HAS_CHILD]->(:StructureNode)``

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    node:
        The :class:`~scinr.newton.models.document_structure.StructureNode`
        to persist.
    doc_path:
        Document path used to build all composite ids.
    version:
        Integer version number used to build all composite ids and look up the Document node.
    parent_id:
        The composite Neo4j ``id`` of the parent :StructureNode, or ``None``
        when this node is a direct child of the :Document node.
    node_path:
        Slash-separated ancestor ``node_id`` values leading up to (but NOT
        including) the current node.  Empty string for root-level nodes.
    """
    role_val = node.role if isinstance(node.role, str) else node.role.value
    role_label = ROLE_TO_LABEL[role_val]
    current_node_path = f"{node_path}/{node.node_id}" if node_path else node.node_id
    composite_id = f"{doc_path}::{version}::{current_node_path}"

    # MERGE on id only, then SET the role label separately.  Including
    # role_label in the MERGE pattern would cause a constraint violation when
    # the same id is re-encountered with a different label during upserts.
    # role_label is derived from a controlled enum, never from user input.
    tx.run(
        f"""
        MERGE (n:StructureNode {{id: $id}})
        SET   n:{role_label},
              n.node_id          = $node_id,
              n.role             = $role,
              n.appearance_order = $appearance_order,
              n.title            = $title,
              n.theme            = $theme,
              n.source_page_ids  = $source_page_ids
        """,
        id=composite_id,
        node_id=node.node_id,
        title=node.title,
        role=role_val,
        appearance_order=node.appearance_order,
        theme=getattr(node, "theme", "default"),
        source_page_ids=node.source_page_ids,
    )
    logger.debug("Merged StructureNode %s (role=%s)", composite_id, role_val)

    # Create relationship to parent
    if parent_id is None:
        tx.run(
            """
            MATCH (d:Document {path: $doc_path, version: $version})
            MATCH (n:StructureNode {id: $node_id})
            MERGE (d)-[:HAS_STRUCTURE]->(n)
            """,
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

    Parameters
    ----------
    tx:
        An open Neo4j transaction.
    doc:
        The fully validated Document to persist.
    resolved_version:
        The integer version to assign. Always provided by the caller;
        never derived from ``doc.version``.
    update_mode:
        If True, existing structure for this version is deleted before
        re-insertion. No new version is created; no HAS_NEWER_VERSION link.
    """
    doc_path = doc.doc_path if doc.doc_path else doc.document_name

    logger.info(
        "Inserting document graph for '%s' (path=%s, version=%d, update=%s, %d root nodes)",
        doc.document_name,
        doc_path,
        resolved_version,
        update_mode,
        len(doc.document_structure),
    )

    # 1. Wipe existing structure when updating in-place
    if update_mode:
        delete_document_content(tx, doc_path, resolved_version)

    # 2. Create all ancestor folder-parent nodes (if any)
    if "/" in doc_path:
        folder_path = doc_path.rsplit("/", 1)[0]
        insert_folder_document_hierarchy(tx, folder_path, resolved_version)

    # 3. Insert (or update) the leaf Document node itself
    insert_document(
        tx,
        doc.document_name,
        doc_path,
        resolved_version,
        doc.raw_file_id,
        is_folder=False,
        context_instructions=doc.context_instructions,
    )

    # 4. Handle versioning (only for normal loads, not updates)
    if not update_mode:
        handle_versioning(tx, doc_path, resolved_version)

    # 5. Link leaf to its immediate folder parent
    link_leaf_to_folder(tx, doc_path, resolved_version)

    # 6. Insert all StructureNodes
    for root_node in doc.document_structure:
        insert_structure_node(
            tx,
            root_node,
            doc_path=doc_path,
            version=resolved_version,
            parent_id=None,
        )

    logger.info(
        "Document graph insertion complete for '%s' (path=%s, version=%d).",
        doc.document_name,
        doc_path,
        resolved_version,
    )
