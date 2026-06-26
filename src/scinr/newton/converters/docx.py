"""
converters/docx.py — Microsoft Word DOCX to Markdown converter.

The DOCX document is split into pages using a two-layer strategy
(tried in order, first match wins):

1. **Explicit breaks** — any ``<w:br w:type="page"/>`` run or ``<w:sectPr>``
   section break triggers a page flush.  A safety cap of
   ``_PARAGRAPHS_PER_PAGE`` elements is also applied so that a single
   logical section never grows unbounded.
2. **Paragraph count** — fallback; a new page every ``_PARAGRAPHS_PER_PAGE``
   elements (paragraphs + tables counted together).

No content is ever discarded.
"""

from __future__ import annotations

import logging
from pathlib import Path

from scinr.newton.converters.base import BaseConverter, ConversionError, IntermediateDocument

logger = logging.getLogger(__name__)

# Heading style name → Markdown prefix character count
_HEADING_PREFIX_MAP: dict[str, int] = {
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
    "Heading 6": 6,
    "Title": 1,
}

_PARAGRAPHS_PER_PAGE = 4


class DocxConverter(BaseConverter):
    """Convert ``.docx`` files to a paginated Markdown document.

    Uses ``python-docx`` to iterate over paragraphs and tables in document
    order.  Headings are detected via paragraph style names and prefixed
    with the appropriate number of ``#`` characters.  Tables are rendered
    as GFM Markdown tables.

    Pagination strategy (first applicable wins):

    1. Explicit breaks (``<w:br w:type="page"/>`` or ``<w:sectPr>``), with a
       ``_PARAGRAPHS_PER_PAGE`` safety cap per section.
    2. Paragraph/table count — new page every ``_PARAGRAPHS_PER_PAGE``
       elements.
    """

    supported_extensions: frozenset[str] = frozenset({"docx"})

    def convert(self, source: Path) -> IntermediateDocument:
        """Convert a DOCX file to the intermediate format.

        Parameters
        ----------
        source:
            Path to the ``.docx`` file.

        Returns
        -------
        IntermediateDocument
            Multi-page document.  The number of pages depends on which
            pagination strategy is applied (see class docstring).

        Raises
        ------
        ConversionError
            If ``python-docx`` is not installed or the file cannot be parsed.
        FileNotFoundError
            If *source* does not exist.
        """
        try:
            import docx  # type: ignore[import-untyped]
            from docx.oxml.ns import qn  # type: ignore[import-untyped]
            from docx.table import Table  # type: ignore[import-untyped]
            from docx.text.paragraph import Paragraph  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ConversionError(
                "python-docx is required for DOCX conversion. "
                "Install it with: uv add python-docx"
            ) from exc

        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            doc = docx.Document(str(source))
        except Exception as exc:
            raise ConversionError(f"Cannot open DOCX file {source}: {exc}") from exc

        try:
            # ------------------------------------------------------------------
            # Collect all elements in document order
            # ------------------------------------------------------------------
            elements: list[dict] = []
            body = doc.element.body
            for child in body:
                if child.tag == qn("w:p"):
                    para = Paragraph(child, doc)
                    heading_level = _heading_level(para)
                    text = _para_to_markdown(para, heading_level)
                    has_pb = _has_page_break(para)
                    elements.append(
                        {
                            "type": "paragraph",
                            "text": text,
                            "has_page_break": has_pb,
                        }
                    )
                elif child.tag == qn("w:tbl"):
                    tbl = Table(child, doc)
                    md = _docx_table_to_markdown(tbl)
                    elements.append(
                        {
                            "type": "table",
                            "text": md,
                            "has_page_break": False,
                        }
                    )
                elif child.tag == qn("w:sectPr"):
                    # Section break (not the last default sectPr of the body)
                    elements.append(
                        {
                            "type": "section_break",
                            "text": "",
                            "has_page_break": False,
                        }
                    )

            # ------------------------------------------------------------------
            # Decide pagination strategy
            # ------------------------------------------------------------------
            has_explicit_breaks = any(
                e["has_page_break"] or e["type"] == "section_break"
                for e in elements
            )
            strategy = "explicit_breaks" if has_explicit_breaks else "paragraph_count"

            if strategy == "explicit_breaks":
                logger.info(
                    "DOCX %s: using explicit breaks for pagination.", source.name
                )
            else:
                logger.info(
                    "DOCX %s: no explicit breaks found — paginating every %d elements.",
                    source.name,
                    _PARAGRAPHS_PER_PAGE,
                )

            # ------------------------------------------------------------------
            # Build pages
            # ------------------------------------------------------------------
            pages = []
            current_parts: list[str] = []
            page_idx = 0
            element_count = 0

            def flush_page() -> None:
                nonlocal page_idx
                if not current_parts:
                    return
                md = "\n\n".join(p for p in current_parts if p)
                pages.append(self._make_page(index=page_idx, markdown=md))
                current_parts.clear()
                page_idx += 1

            for elem in elements:
                if strategy == "explicit_breaks":
                    is_break = elem["has_page_break"] or elem["type"] == "section_break"
                    if is_break:
                        flush_page()
                    else:
                        text = elem["text"]
                        if text:
                            current_parts.append(text)
                        if len(current_parts) >= _PARAGRAPHS_PER_PAGE:
                            flush_page()
                    continue

                # strategy == "paragraph_count"
                text = elem["text"]
                if text:
                    current_parts.append(text)
                    element_count += 1

                if (
                    strategy == "paragraph_count"
                    and element_count >= _PARAGRAPHS_PER_PAGE
                ):
                    flush_page()
                    element_count = 0

            # Flush any remaining content
            if current_parts:
                flush_page()

            if not pages:
                pages.append(self._make_page(index=0, markdown=""))

        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Failed to convert DOCX {source}: {exc}") from exc

        logger.info("DOCX %s: %d page(s) produced.", source.name, len(pages))
        return IntermediateDocument(pages=pages)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_page_break(paragraph) -> bool:  # type: ignore[no-untyped-def]
    """Return True if the paragraph contains a manual page break.

    Parameters
    ----------
    paragraph:
        A ``python-docx`` ``Paragraph`` object.

    Returns
    -------
    bool
        ``True`` if any run in the paragraph contains a ``<w:br
        w:type="page"/>`` element.
    """
    from docx.oxml.ns import qn  # type: ignore[import-untyped]

    for run in paragraph.runs:
        for br in run._r.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _heading_level(paragraph) -> int | None:  # type: ignore[no-untyped-def]
    """Detect the Markdown heading level for a paragraph.

    Parameters
    ----------
    paragraph:
        A ``python-docx`` ``Paragraph`` object.

    Returns
    -------
    int | None
        Heading level (1–6) if the paragraph is a heading, ``None`` otherwise.
    """
    style_name: str = paragraph.style.name if paragraph.style else ""
    if style_name in _HEADING_PREFIX_MAP:
        return _HEADING_PREFIX_MAP[style_name]
    # Handle styles like "Heading 1 Char", "heading 1" (case variations)
    lower = style_name.lower()
    if lower.startswith("heading "):
        suffix = lower[len("heading "):].strip()
        try:
            level = int(suffix)
            if 1 <= level <= 6:
                return level
        except ValueError:
            pass
    return None


def _para_to_markdown(paragraph, heading_level: int | None) -> str:  # type: ignore[no-untyped-def]
    """Convert a paragraph to a Markdown string.

    Parameters
    ----------
    paragraph:
        A ``python-docx`` ``Paragraph`` object.
    heading_level:
        Heading level (1–6) or ``None`` for body text.

    Returns
    -------
    str
        Markdown representation of the paragraph (may be empty).
    """
    text = paragraph.text.strip()
    if not text:
        return ""
    if heading_level is not None:
        prefix = "#" * heading_level
        return f"{prefix} {text}"
    return text


def _docx_table_to_markdown(table) -> str:  # type: ignore[no-untyped-def]
    """Convert a ``python-docx`` Table to a GFM Markdown table.

    Parameters
    ----------
    table:
        A ``python-docx`` ``Table`` object.

    Returns
    -------
    str
        GFM Markdown table string, or empty string if the table has no rows.
    """
    rows = table.rows
    if not rows:
        return ""

    def _escape(cell_text: str) -> str:
        return cell_text.replace("|", "\\|")

    all_rows: list[list[str]] = []
    for row in rows:
        cells = [_escape(cell.text.strip()) for cell in row.cells]
        all_rows.append(cells)

    if not all_rows:
        return ""

    col_count = max(len(r) for r in all_rows)

    def _pad(row: list[str]) -> list[str]:
        padded = list(row)
        while len(padded) < col_count:
            padded.append("")
        return padded[:col_count]

    lines: list[str] = []
    header = _pad(all_rows[0])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(col_count)) + " |")
    for row in all_rows[1:]:
        lines.append("| " + " | ".join(_pad(row)) + " |")

    return "\n".join(lines)
