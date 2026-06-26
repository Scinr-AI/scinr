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
    Assign the most precise role using the following decision tree (evaluate top-to-bottom,
    stop at the first match):

      1. Is supplementary, ancillary, or reference material to the main document body?
         This includes sections explicitly or implicitly labelled as Annex, Appendix,
         Attachment, Supplement, Addendum, or any equivalent term in any language.
         These sections typically appear after the main numbered content and contain
         supporting material referenced from the main body.
         → appendix

      2. Has a rows-and-columns tabular layout?
         → table

      3. Is a group of named data fields (form-like structure)?
         → field_group  (numbering is irrelevant for this category)

      4. Carries an ordered identifier traceable as a sequence across the document?
         Identifiers may be numeric, alphabetic, Roman numeral, or any mixed combination
         (e.g. "1.1", "A.b", "I.iv", "A.b.3.I").
         → section  (if top-level in the document hierarchy)
         → subsection  (if nested under another section or subsection)

      5. Is top-level in the document hierarchy and has NO identifier?
         → section

      6. Everything else — unnumbered, nested, prose continuation, or any block that does
         not match steps 1–5:
         → freeform_block

    Role definitions:
      section        — Top-level heading. May carry an ordered identifier or be unnumbered
                       when it is genuinely the highest-level structural division.
      subsection     — Nested heading that carries an ordered identifier (numeric, alphabetic,
                       Roman numeral, or mixed). A heading WITHOUT any ordered identifier is
                       NEVER a subsection — assign freeform_block instead.
      table          — Tabular layout with rows and columns.
      appendix       — Any supplementary, ancillary, or reference section that is not
                       part of the main document body. Covers sections labelled as
                       Annex, Appendix, Attachment, Supplement, Addendum, or equivalent
                       terms in any language. May be numbered (Annex A, Appendix 1) or
                       unnumbered. Typically appears after the main content.
      field_group    — Group of named data fields (form-like structure). Numbering is
                       irrelevant — a form block is always field_group regardless of whether
                       it has an identifier.
       freeform_block — Catch-all for any block that does not qualify as section, subsection,
                        table, appendix, or field_group. Typical cases: unnumbered
                        headings nested inside a section, prose continuation blocks, mixed
                        content without a clear structural role.
       row            — A single data row within a tabular (CSV/XLSX) Table node. Used
                        exclusively by the tabular ingestion pipeline; never assigned by
                        the LLM document-extraction pipeline.
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
            "Example: 'Storage temperature requirement', 'Batch release criteria'."
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
            "Comprehensive technical note capturing ALL semantically significant details "
            "expressed in this InfoUnit. This field is the sole representation of the "
            "concept available to downstream agents — no access to the original document "
            "text exists at that stage.\n\n"
            "CONTENT REQUIREMENTS — the description MUST preserve:\n"
            "  • All quantitative values: numbers, percentages, concentrations, temperatures,\n"
            "    ranges, tolerances, limits, thresholds (e.g. '2–8°C', '≤0.1%', '24 months').\n"
            "  • All named entities: substance names, method names, standard references,\n"
            "    regulatory codes, equipment identifiers, personnel roles.\n"
            "  • All conditions and qualifiers: 'only if', 'unless', 'provided that',\n"
            "    'when stored correctly', 'subject to ongoing litigation'.\n"
            "  • All restrictions and prohibitions: 'do not freeze', 'must not exceed',\n"
            "    'rejected without review'.\n"
            "  • All format specifications, identifier patterns, and enumerated lists.\n\n"
            "WRITING STYLE:\n"
            "  • Synthesise — do NOT copy the source text verbatim sentence-by-sentence.\n"
            "  • Do NOT reduce to a one-line topic label (e.g. 'Storage requirements').\n"
            "  • Write as a self-contained technical note: a reader with no access to the\n"
            "    source document must be able to extract precise structured data from this\n"
            "    description alone.\n"
            "  • Prefer active, declarative sentences. Preserve all original numeric\n"
            "    values and units exactly as they appear in the source.\n\n"
            "GROUNDING CONSTRAINT:\n"
            "  • Include ONLY information present in CURRENT_PAGE.\n"
            "  • Do NOT infer, extrapolate, or add background knowledge.\n"
            "  • If the source text is ambiguous, reflect the ambiguity — do not resolve it.\n\n"
            "BAD (too superficial): 'Storage requirements for the product.'\n"
            "BAD (verbatim copy): 'The product must be stored at 2°C to 8°C, protected "
            "from light and moisture. Do not freeze. Shelf life is 24 months when stored correctly.'\n"
            "GOOD (synthesised, detail-preserving): 'The product requires refrigerated storage "
            "at 2–8°C, protected from light and moisture; freezing is prohibited. Shelf life "
            "is 24 months under these conditions.'"
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
            "Unique identifier for this node. Two formats depending on the heading:\n\n"
            "FORMAT A — Numeric/code heading (has a section number, supplement letter, or table number):\n"
            "  Extract the code directly. Replace dots and spaces with underscores. Lowercase.\n"
            "  Drop the descriptive title entirely — the code alone is the id.\n"
            "  Examples:\n"
            "    '1 Introduction'                       → 1\n"
            "    '2.1 Scope'                            → 2_1\n"
            "    'Supplement A'                          → supplement_a\n"
            "    'Table 1 – Comparability'               → table_1\n"
            "    'Section 1 – Introduction'              → section_1\n\n"
            "FORMAT B — Unnumbered heading (no numeric or letter code in the title):\n"
            "  Use: {appearance_order}-{1-to-3-word-slug} where slug is lowercase words from the title.\n"
            "  Examples:\n"
            "    'General Information' (order=1)         → 1-general\n"
            "    'Purification and Modification' (order=3)→ 3-purification\n"
            "    'Manufacturing Process' (order=2)       → 2-manufacturing-process\n\n"
            "Rules:\n"
            "  - Allowed characters for FORMAT A: [a-z0-9_] only (underscores).\n"
            "  - Allowed characters for FORMAT B: [a-z0-9-] only (hyphens as separator).\n"
            "  - Must be unique across the entire output tree.\n"
            "  - ORPHANED numbered subsections MUST use FORMAT A with the full numeric path\n"
            "    (e.g. '2.1.3 Background' → 2_1_3). Post-processing uses this prefix\n"
            "    for re-nesting. FORMAT B orphans rely solely on parent_id."
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
            "Structural role of this node. Must be one of: "
            "section, subsection, table, freeform_block, field_group, appendix. "
            "See NodeRole docstring for the full role assignment guide."
        ),
    )
    appearance_order: int = Field(
        description=(
            "1-based integer indicating the sequential position of this node among its siblings at "
            "the same hierarchical level.\n"
            "  - Top-level nodes: if headings are numbered (e.g. '3.', '4.'), use the numeric "
            "value of the section number as the integer (e.g. '3.' → 3, '4.' → 4, '10.' → 10). "
            "Otherwise continue from the last top-level section visible on the previous page; "
            "if none visible, start at 1.\n"
            "  - Child nodes: restart at 1 within each parent independently."
        ),
    )
    parent_id: str | None = Field(
        default=None,
        description=(
            "node_id of this node's parent when the parent is not visible on the current page. "
            "Set when: the parent's node_id is listed in the active hierarchy AND the parent "
            "heading is not present on the current page. "
            "Must be an exact copy of a node_id from the active hierarchy. "
            "Set to null when: the parent is already visible on the current page (node is nested "
            "normally inside it), or the parent cannot be found in the active hierarchy. "
            "Null is always safer than an incorrect parent_id."
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
            "Ordered list of semantic information units extracted from the body content "
            "of this structural node. Create one InfoUnit per semantically distinct concept "
            "found in this node's body text. Do NOT create InfoUnits for the heading itself."
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
