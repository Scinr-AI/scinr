from __future__ import annotations

import json
import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============================================================
# BASE
# ============================================================


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )


# ============================================================
# ENUMS
# ============================================================


class NodeRole(str, Enum):
    """
    Structural role of a document node. Assign using the decision tree in the extraction prompt
    (evaluate top-to-bottom, stop at first match):
    appendix → table → field_group → section/subsection (ordered identifier present)
    → section (top-level, no identifier) → freeform_block (catch-all).
    Row is reserved for the tabular ingestion pipeline and is never assigned by the extraction LLM.
    """

    SECTION = "section"
    SUBSECTION = "subsection"
    TABLE = "table"
    APPENDIX = "appendix"
    FIELD_GROUP = "field_group"
    FREEFORM_BLOCK = "freeform_block"
    ROW = "row"  # A single data row within a tabular (CSV/XLSX) Table node


# ============================================================
# SEMANTIC LAYER
# ============================================================


class InfoUnit(StrictModel):
    """
    Smallest reusable unit of information capturing one semantically distinct concept
    extracted from the document. The description field is the sole representation of
    this concept available to downstream agents — preserve all technical details.
    """

    title: str = Field(
        description=(
            "Short label for the semantic concept expressed in this InfoUnit. "
            "3 to 8 words, written as a noun phrase or imperative phrase. "
            "Examples: 'Maximum load capacity', 'File submission deadline'."
        ),
    )
    order: int = Field(
        default=0,
        description=(
            "0-based position of this InfoUnit within its parent StructureNode. "
            "Reflects the order of appearance in the source document text. "
            "InfoUnit at index 0 is the first concept extracted from this node."
        ),
    )
    description: str = Field(
        description=(
            "Self-contained technical note preserving all quantitative values, named entities, "
            "conditions, restrictions, and qualifiers from the source passage. "
            "Written as synthesised prose — not a verbatim copy and not a one-line topic label. "
            "When the concept is a sub-entry whose meaning depends on its parent section "
            "or grouping, incorporate the parent identifier and scope so the description "
            "is independently interpretable — downstream agents access no other "
            "representation of this content, including parent node titles and body text. "
            "Grounded exclusively in CURRENT_PAGE content; reflect source ambiguity as-is."
        ),
    )


# ============================================================
# STRUCTURAL LAYER
# ============================================================


class StructureNode(StrictModel):
    """
    One structural division of the document (section, subsection, table, etc.).
    Nodes are hierarchical; children nest within their parent node.
    """

    node_id: str = Field(
        description=(
            "Unique identifier for this node.\n"
            "FORMAT A (numbered/coded heading): extract the numeric or letter code, replace dots "
            "and spaces with underscores, lowercase — e.g. '2.1 Scope' → '2_1', "
            "'Supplement A' → 'supplement_a'. FORMAT B (unnumbered heading): "
            "'{appearance_order}-{slug}' where slug is 1–3 lowercase words from the title — "
            "e.g. appearance_order=1, title 'General Information' → '1-general'. "
            "Must be unique across the full output tree. See extraction prompt for full FORMAT A/B rules."
        ),
    )
    title: str | None = Field(
        default=None,
        description=(
            "Exact heading text as it appears in the source document, including any numbering prefix "
            "(e.g. '2.1 Scope'). Never paraphrase, abbreviate, clean up, or translate the heading text. "
            "Set to null only for continuation freeform_block nodes that have no heading of their own."
        ),
    )
    role: NodeRole = Field(
        description=(
            "Structural role of this node. One of: "
            "section, subsection, table, freeform_block, field_group, appendix. "
            "Apply the decision tree from the extraction prompt to assign this field."
        ),
    )
    appearance_order: int = Field(
        description=(
            "1-based position among siblings at the same hierarchy level. "
            "For numbered top-level headings use the numeric section value (e.g. '3.' → 3). "
            "Child nodes restart at 1 within each parent independently."
        ),
    )
    parent_id: str | None = Field(
        default=None,
        description=(
            "node_id of this node's parent when the parent heading is not visible on CURRENT_PAGE "
            "but appears in active_hierarchy. Set to null when the parent is already visible on "
            "this page or cannot be identified. Null is always safer than an incorrect parent_id."
        ),
    )
    theme: str = "default"
    source_page_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Internal infrastructure field. Always output as an empty list []. "
            "Set by the pipeline after extraction — must not be populated by the LLM."
        ),
    )
    info_units: list[InfoUnit] = Field(
        default_factory=list,
        description=(
            "Ordered list of semantic information units extracted from this structural node. "
            "When the node's title carries semantic content — entity names, numeric values, "
            "conditions, or qualifiers — represent it as the first InfoUnit (order=0). "
            "When this node is a sub-entry whose meaning depends on its parent section "
            "or grouping, the order=0 description incorporates the parent identifier and "
            "scope (e.g. 'Section 4.2 item (b): ...') — downstream agents have no "
            "access to parent node titles or any surrounding context. "
            "Body-text InfoUnits follow at order=1 and above. "
            "Omit the heading InfoUnit only for pure structural labels with no independently "
            "useful content (e.g. 'Introduction', 'Scope', 'Overview'). "
            "InfoUnits are the sole content representation available to downstream agents."
        ),
    )
    children: list[StructureNode] = Field(
        default_factory=list,
        description=(
            "Ordered list of nested structural nodes that are direct children of this node. "
            "A child that begins near the bottom of the current page and clearly continues beyond "
            "it must be OMITTED from this list."
        ),
    )


# Required for forward reference in `children` field
StructureNode.model_rebuild()


# ============================================================
# DOCUMENT LAYER
# ============================================================


class DocumentStructure(StrictModel):
    """
    LLM output per chunk — the list of StructureNodes extracted from one sliding-window chunk.
    Used as the structured output schema for each LLM call. Results are merged across chunks
    into a final Document.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_nodes_from_string(cls, values):
        """Coerce nodes from JSON string to list if the LLM serialized it incorrectly."""
        if isinstance(values, dict) and isinstance(values.get("nodes"), str):
            values = dict(values)
            values["nodes"] = json.loads(values["nodes"])
        return values

    nodes: list[StructureNode] = Field(
        default_factory=list,
        description=(
            "Ordered list of structural nodes extracted from the current document chunk. "
            "Includes top-level nodes and any nested children discovered within this chunk."
        ),
    )


class Document(StrictModel):
    """
    Final merged document tree produced after all chunks have been processed and merged.
    Represents the complete structured extraction of one source document.
    """

    @field_validator("doc_path", mode="before")
    @classmethod
    def _normalize_doc_path_separators(cls, value):
        """Normalize Windows-style backslashes to forward slashes in doc_path."""
        if isinstance(value, str) and value:
            return value.replace("\\", "/")
        return value

    document_name: str = Field(
        default="",
        description=(
            "Official or working title of the document as stated in its header, "
            "title page, or main heading. "
            "Intentionally left empty — this field is never auto-populated by the pipeline."
        ),
    )
    document_type: str = Field(
        default="",
        description=(
            "Category or classification of the document "
            "(e.g. standard, regulation, guideline, template, specification, policy). "
            "Intentionally left empty — this field is never auto-populated by the pipeline."
        ),
    )
    doc_path: str | None = Field(
        default=None,
        description=(
            "Relative path of this document from the pipeline input root. "
            "For leaf documents: e.g. 'ModuloA/SubModulo/doc_a'. "
            "For root documents (no subfolder): just the document stem. "
            "Set by the pipeline orchestrator during Stage 1 extraction. "
            "Never populated by the LLM."
        ),
    )
    version: int = Field(
        default=1,
        description=(
            "Version number of this document in Neo4j. "
            "Informational only — the actual persisted version is always resolved "
            "from Neo4j at ingest time and may differ from this default. "
            "Never populated by the LLM."
        ),
    )
    raw_file_id: str = Field(
        description=("Raw file identifier saved in the DB (not graph)")
    )
    context_instructions: str | None = Field(
        default=None,
        description=(
            "Free-text user-provided context injected at ingestion time via --context. "
            "Used by LLM stages to focus extraction and annotation."
        ),
    )
    tenant_id: str | None = Field(
        default=None,
        description=(
            "Multi-tenant owner id, supplied by the caller at ingestion time and "
            "persisted verbatim on the :Document node. Never populated by the LLM."
        ),
    )
    created_by_user_id: str | None = Field(
        default=None,
        description=(
            "Id of the user that launched this ingestion, supplied by the caller and "
            "persisted verbatim on the :Document node. Never populated by the LLM."
        ),
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Ingestion job/run id this document belongs to, supplied by the caller and "
            "persisted verbatim on the :Document node. Used as a bulk-delete selector. "
            "Never populated by the LLM."
        ),
    )
    document_structure: list[StructureNode] = Field(
        default_factory=list,
        description=(
            "Ordered list of top-level structural nodes representing "
            "the document's main divisions."
        ),
    )
# No tiene sentido que sea siempre que se crea un documento, ya que cuando se crea manualmente también sale este warning y no procede.
    # @model_validator(mode="after")
    # def warn_if_empty_structure(self) -> Document:
    #     """Emit a warning when the document has no top-level StructureNodes."""
    #     if not self.document_structure:
    #         logging.getLogger(__name__).warning(
    #             "Document '%s' has an empty document_structure. "
    #             "It will be ingested without any StructureNodes.",
    #             self.document_name,
    #         )
    #     return self
