"""
converters/html.py — HTML/HTM to Markdown converter.

Extracts structured text from HTML using BeautifulSoup4 + lxml,
converts headings/paragraphs/lists/tables to GFM Markdown, and
removes noise (scripts, styles, nav, footer).

The document is split into pages at H1 headings (``# …`` in Markdown).
Each H1 and its following content form one page.  If the HTML has no H1
headings the entire document is a single page.  No content is discarded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from scinr.newton.converters.base import BaseConverter, ConversionError, IntermediateDocument

logger = logging.getLogger(__name__)

# Tags to remove entirely (including their children)
_NOISE_TAGS = {"script", "style", "nav", "footer", "head", "noscript", "iframe"}

# Tags that map to Markdown heading levels
_HEADING_MAP = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}


class HtmlConverter(BaseConverter):
    """Convert ``.html`` and ``.htm`` files to the intermediate format.

    Parses the HTML with BeautifulSoup4 (lxml backend) and converts the
    document structure to GFM Markdown.  The output is split into one page
    per H1 heading (``# …``).  Documents without H1 headings produce a
    single page.
    """

    supported_extensions: frozenset[str] = frozenset({"html", "htm"})

    def convert(self, source: Path) -> IntermediateDocument:
        """Convert an HTML file.

        Args:
            source: Path to the ``.html`` or ``.htm`` file.

        Returns:
            Multi-page document split at H1 headings (one page per H1
            section).  Produces a single page when no H1 headings are
            present.

        Raises:
            ConversionError: If BeautifulSoup4 or lxml is not installed, or the file
                cannot be parsed.
        """
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ConversionError(
                "beautifulsoup4 is required for HTML conversion. "
                "Install it with: uv add beautifulsoup4 lxml"
            ) from exc

        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            raw = source.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ConversionError(f"Cannot read {source}: {exc}") from exc

        try:
            soup = BeautifulSoup(raw, "lxml")
        except Exception as exc:
            raise ConversionError(f"HTML parse error in {source}: {exc}") from exc

        # Remove noise tags
        for tag in _NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        markdown = _soup_to_markdown(soup)

        # Split at H1 headings using a lookahead so the heading stays
        # at the start of each section
        sections = re.split(r"(?m)(?=^# )", markdown)
        sections = [s.strip() for s in sections if s.strip()]

        if not sections:
            sections = [markdown]  # fallback: single page even if empty

        if len(sections) > 1:
            logger.info(
                "Converting %s: %d H1 section(s) → %d page(s)",
                source.name,
                len(sections),
                len(sections),
            )

        pages = [
            self._make_page(index=i, markdown=section)
            for i, section in enumerate(sections)
        ]
        return IntermediateDocument(pages=pages)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _soup_to_markdown(soup) -> str:  # type: ignore[no-untyped-def]
    """Walk the BeautifulSoup tree and produce GFM Markdown.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        GFM Markdown string.
    """
    from bs4 import Tag  # type: ignore[import-untyped]

    parts: list[str] = []

    body = soup.find("body") or soup

    for element in body.descendants:
        if not isinstance(element, Tag):
            continue

        tag_name = element.name.lower() if element.name else ""

        if tag_name in _HEADING_MAP:
            prefix = _HEADING_MAP[tag_name]
            text = element.get_text(separator=" ", strip=True)
            if text:
                parts.append(f"{prefix} {text}")

        elif tag_name == "p":
            # Only process direct <p> (not nested inside headings etc.)
            if not any(
                isinstance(parent, Tag) and parent.name in _HEADING_MAP
                for parent in element.parents
            ):
                text = element.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)

        elif tag_name == "table":
            md_table = _html_table_to_markdown(element)
            if md_table:
                parts.append(md_table)

        elif tag_name in {"ul", "ol"}:
            md_list = _html_list_to_markdown(element, ordered=(tag_name == "ol"))
            if md_list:
                parts.append(md_list)

    # Collapse excessive blank lines
    text = "\n\n".join(p for p in parts if p)
    return re.sub(r"\n{3,}", "\n\n", text)


def _html_table_to_markdown(table_tag) -> str:  # type: ignore[no-untyped-def]
    """Convert an HTML ``<table>`` tag to a GFM Markdown table.

    Args:
        table_tag: BeautifulSoup Tag for the ``<table>`` element.

    Returns:
        GFM Markdown table, or empty string if the table has no rows.
    """

    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            td.get_text(separator=" ", strip=True)
            for td in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)

    def _pad(row: list[str]) -> list[str]:
        padded = list(row)
        while len(padded) < col_count:
            padded.append("")
        return padded[:col_count]

    def _escape(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines: list[str] = []
    header = _pad(rows[0])
    lines.append("| " + " | ".join(_escape(c) for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(col_count)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(_escape(c) for c in _pad(row)) + " |")

    return "\n".join(lines)


def _html_list_to_markdown(list_tag, ordered: bool = False, depth: int = 0) -> str:  # type: ignore[no-untyped-def]
    """Convert an HTML list to Markdown.

    Args:
        list_tag: BeautifulSoup Tag for ``<ul>`` or ``<ol>``.
        ordered: Whether this is an ordered list.
        depth: Nesting depth (for indentation).

    Returns:
        Markdown list string.
    """
    from bs4 import Tag  # type: ignore[import-untyped]

    lines: list[str] = []
    indent = "  " * depth
    counter = 1
    for li in list_tag.find_all("li", recursive=False):
        # Extract direct text (not from nested lists)
        text_parts: list[str] = []
        nested_md = ""
        for child in li.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                nested_md = _html_list_to_markdown(
                    child,
                    ordered=(child.name == "ol"),
                    depth=depth + 1,
                )
            elif isinstance(child, Tag):
                text_parts.append(child.get_text(separator=" ", strip=True))
            else:
                stripped = str(child).strip()
                if stripped:
                    text_parts.append(stripped)

        item_text = " ".join(text_parts).strip()
        prefix = f"{counter}." if ordered else "-"
        if item_text:
            lines.append(f"{indent}{prefix} {item_text}")
        if nested_md:
            lines.append(nested_md)
        counter += 1

    return "\n".join(lines)
