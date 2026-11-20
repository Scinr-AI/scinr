"""Paragraph segmentation, anchors and verbatim InfoUnit helpers.

Pure module: no configuration, no I/O and no LLM calls. Pages are split into paragraphs with
a deterministic rule set (``PARAGRAPH_SEGMENTER_VERSION``); every paragraph of a document
gets a global 1-based number that the extraction LLM cites as an ``[[n]]`` anchor.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from scinr.newton.models.document_structure import (
    Document,
    ExtractionInfoUnitsMode,
    InfoUnit,
    ParagraphRange,
    StructureNode,
)

logger = logging.getLogger(__name__)

PARAGRAPH_SEGMENTER_VERSION = 1

_HEADING = re.compile(r"^ {0,3}#{1,6}(\s|$)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_TABLE = re.compile(r"^\s*\|")
_LIST_ITEM = re.compile(r"^( {0,3})([-*+]|\d+[.)])\s+\S")
_IMAGE_ONLY = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")

MARKER_RE = re.compile(r"\[\[\d+\]\]")

_KIND_TEXT = "text"
_KIND_TABLE = "table"
_KIND_LIST = "list"
_KIND_CODE = "code"

_TITLE_MAX_WORDS = 8
_TITLE_SEPARATOR_CHARS = frozenset("#|-*+_>`~:=")
_TITLE_EDGE_CHARS = "*_`"
_TITLE_ELLIPSIS = "…"
_MIN_TABLE_LINES = 3


# ============================================================
# SEGMENTATION
# ============================================================


def _line_content(line: str) -> str:
    """*line* without its terminator (\\n, \\r\\n or \\r)."""
    return line.rstrip("\r\n")


def segment_page_spans(markdown: str) -> list[tuple[int, int]]:
    """(start, end) offsets of every paragraph of *markdown*, in order (rules v1)."""
    spans: list[tuple[int, int]] = []
    start = 0
    end = 0
    kind: str | None = None
    list_indent = 0
    fence = ""

    def close() -> None:
        nonlocal kind
        if kind is not None and markdown[start:end].strip():
            spans.append((start, end))
        kind = None

    offset = 0
    for line in markdown.splitlines(keepends=True):
        content = _line_content(line)
        line_start = offset
        line_end = offset + len(content)
        offset += len(line)

        if kind == _KIND_CODE:
            end = line_end
            m = _FENCE.match(content)
            if m:
                run = m.group(1)
                if run[0] == fence[0] and len(run) >= len(fence):
                    close()
            continue

        if content.strip() == "":
            close()
            continue

        m = _FENCE.match(content)
        if m:
            close()
            start, end, kind, fence = line_start, line_end, _KIND_CODE, m.group(1)
            continue

        if _HEADING.match(content) or _IMAGE_ONLY.match(content):
            close()
            start, end, kind = line_start, line_end, _KIND_TEXT
            close()
            continue

        if _TABLE.match(content):
            if kind == _KIND_TABLE:
                end = line_end
            else:
                close()
                start, end, kind = line_start, line_end, _KIND_TABLE
            continue

        m = _LIST_ITEM.match(content)
        if m:
            close()
            start, end, kind = line_start, line_end, _KIND_LIST
            list_indent = len(m.group(1))
            continue

        indent = len(content) - len(content.lstrip(" "))
        if (kind == _KIND_LIST and indent > list_indent) or kind == _KIND_TEXT:
            end = line_end
        else:
            close()
            start, end, kind = line_start, line_end, _KIND_TEXT

    close()
    return spans


def segment_page(markdown: str) -> list[str]:
    return [markdown[s:e] for s, e in segment_page_spans(markdown)]


# ============================================================
# REFERENCES AND INDEX
# ============================================================


@dataclass(frozen=True)
class ParagraphRef:
    global_number: int
    page_position: int
    page_index: int
    page_id: str | None
    local_number: int
    text: str


@dataclass
class ParagraphStats:
    """Counters of one document's extraction, shared by all its chunks."""

    discarded_ranges: int = 0


class ParagraphIndex:
    """Global paragraph numbering over a sequence of pages (kept by reference, not copied)."""

    def __init__(
        self,
        pages: Sequence[str],
        page_indices: Sequence[int],
        page_ids_by_index: Mapping[int, str],
    ) -> None:
        self._pages = pages
        self._page_indices = page_indices
        self._page_ids_by_index = page_ids_by_index
        self._spans = [segment_page_spans(p) for p in pages]
        self._first: list[int] = []
        count = 0
        for spans in self._spans:
            self._first.append(count + 1)
            count += len(spans)
        self.total = count

    def number_range(self, positions: Sequence[int]) -> tuple[int, int] | None:
        """(lo, hi) paragraph numbers covered by the consecutive *positions*, or None."""
        if not positions:
            return None
        lo = self._first[positions[0]]
        hi = self._first[positions[-1]] + len(self._spans[positions[-1]]) - 1
        if hi < lo:
            return None
        return lo, hi

    def _position(self, number: int) -> int:
        if not 1 <= number <= self.total:
            raise ValueError(f"Paragraph number {number} out of range 1..{self.total}")
        return bisect.bisect_right(self._first, number) - 1

    def locate(self, number: int) -> ParagraphRef:
        pos = self._position(number)
        local = number - self._first[pos] + 1
        start, end = self._spans[pos][local - 1]
        page_index = self._page_indices[pos]
        return ParagraphRef(
            global_number=number,
            page_position=pos,
            page_index=page_index,
            page_id=self._page_ids_by_index.get(page_index),
            local_number=local,
            text=self._pages[pos][start:end],
        )

    def text(self, number: int) -> str:
        return self.locate(number).text

    def marked_page(self, position: int) -> str:
        """The page's paragraphs, each preceded by its ``[[n]]`` marker on its own line."""
        page = self._pages[position]
        first = self._first[position]
        return "\n\n".join(
            f"[[{first + i}]]\n{page[s:e]}" for i, (s, e) in enumerate(self._spans[position])
        )

    def _valid_numbers(self, ranges: Sequence[ParagraphRange]) -> Iterable[Iterable[int]]:
        for r in ranges:
            yield [n for n in range(r.start, r.end + 1) if 1 <= n <= self.total]

    def page_ids_for_ranges(self, ranges: Sequence[ParagraphRange]) -> list[str]:
        """Ids of the pages touched by *ranges*, unique, in page order, without None."""
        positions = sorted(
            {self._position(n) for nums in self._valid_numbers(ranges) for n in nums}
        )
        ids: list[str] = []
        for pos in positions:
            page_id = self._page_ids_by_index.get(self._page_indices[pos])
            if page_id is not None and page_id not in ids:
                ids.append(page_id)
        return ids

    def format_source_paragraphs(self, ranges: Sequence[ParagraphRange]) -> list[str]:
        """One ``"<page>:<from>-<to>"`` (1-based) item per page span of every range (D-5)."""
        result: list[str] = []
        for nums in self._valid_numbers(ranges):
            run: tuple[int, int, int] | None = None  # (page_index, from_local, to_local)
            for n in nums:
                ref = self.locate(n)
                if run and run[0] == ref.page_index and run[2] + 1 == ref.local_number:
                    run = (run[0], run[1], ref.local_number)
                    continue
                if run:
                    result.append(f"{run[0] + 1}:{run[1]}-{run[2]}")
                run = (ref.page_index, ref.local_number, ref.local_number)
            if run:
                result.append(f"{run[0] + 1}:{run[1]}-{run[2]}")
        return result


# ============================================================
# RANGES AND MARKERS
# ============================================================


def strip_markers(text: str) -> str:
    """*text* without any [[n]] marker, stripped."""
    return MARKER_RE.sub("", text).strip()


def merge_ranges(ranges: Iterable[ParagraphRange]) -> list[ParagraphRange]:
    """Sorted by (start, end); overlapping or contiguous ranges merged (D-4)."""
    merged: list[ParagraphRange] = []
    for r in sorted(ranges, key=lambda x: (x.start, x.end)):
        if merged and r.start <= merged[-1].end + 1:
            merged[-1] = ParagraphRange(start=merged[-1].start, end=max(merged[-1].end, r.end))
        else:
            merged.append(ParagraphRange(start=r.start, end=r.end))
    return merged


# ============================================================
# VERBATIM INFOUNITS
# ============================================================


def verbatim_title(text: str, fallback: str) -> str:
    tokens = [t for t in text.split() if not set(t) <= _TITLE_SEPARATOR_CHARS]
    tokens = [t.strip(_TITLE_EDGE_CHARS) for t in tokens]
    tokens = [t for t in tokens if t]
    if not tokens:
        return fallback
    title = " ".join(tokens[:_TITLE_MAX_WORDS])
    if len(tokens) > _TITLE_MAX_WORDS:
        title += _TITLE_ELLIPSIS
    return title


def _break_line(line: str, limit: int) -> list[str]:
    """Pieces of *line* no longer than *limit*, split at the last space before the limit."""
    pieces: list[str] = []
    while len(line) > limit:
        cut = line.rfind(" ", 0, limit)
        if cut <= 0:
            pieces.append(line[:limit])
            line = line[limit:]
        else:
            pieces.append(line[:cut])
            line = line[cut + 1 :]
    pieces.append(line)
    return [p for p in pieces if p]


def split_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    header = ""
    non_empty = [ln for ln in lines if ln.strip()]
    if (
        len(lines) >= _MIN_TABLE_LINES
        and non_empty
        and all(_TABLE.match(ln) for ln in non_empty)
        and len("\n".join(lines[:2])) < max_chars // 2
    ):
        header = "\n".join(lines[:2])
        lines = lines[2:]
    available = max_chars - len(header) - 1 if header else max_chars

    pieces: list[str] = []
    for line in lines:
        pieces.extend(_break_line(line, available))

    groups: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        added = len(piece) + (1 if current else 0)
        if current and current_len + added > available:
            groups.append("\n".join(current))
            current, current_len, added = [], 0, len(piece)
        current.append(piece)
        current_len += added
    if current:
        groups.append("\n".join(current))

    groups = [g for g in groups if g.strip()]
    if header:
        return [f"{header}\n{g}" for g in groups]
    return groups


def build_verbatim_info_units(
    ranges: Sequence[ParagraphRange],
    index: ParagraphIndex,
    *,
    node_title: str,
    max_chars: int,
) -> list[InfoUnit]:
    numbers: list[int] = []
    seen: set[int] = set()
    for r in merge_ranges(ranges):
        for n in range(r.start, r.end + 1):
            if 1 <= n <= index.total and n not in seen:
                seen.add(n)
                numbers.append(n)

    units: list[InfoUnit] = []
    for n in numbers:
        ref = index.locate(n)
        for piece in split_paragraph(ref.text, max_chars):
            units.append(
                InfoUnit(
                    title=verbatim_title(piece, f"{node_title} · p. {ref.page_index + 1}"),
                    order=len(units),
                    description=piece,
                )
            )
    return units


# ============================================================
# FINALIZATION (step F)
# ============================================================


def _walk_nodes(nodes: Iterable[StructureNode]) -> Iterable[StructureNode]:
    for node in nodes:
        yield node
        yield from _walk_nodes(node.children)


def finalize_paragraph_references(
    document: Document,
    index: ParagraphIndex,
    *,
    info_units_mode: ExtractionInfoUnitsMode,
    max_chars: int,
    stats: ParagraphStats,
) -> None:
    """Derive page ids, source paragraphs and verbatim InfoUnits from the node ranges.

    Visits every node of ``document.document_structure`` depth-first.
    """
    cited: set[int] = set()
    nodes_without_ranges = 0

    for node in _walk_nodes(document.document_structure):
        node.paragraph_segmenter_version = PARAGRAPH_SEGMENTER_VERSION
        if info_units_mode != "auto":
            # auto: keep the LLM's per-node decision; resolve_auto_info_units writes the units
            node.info_units_mode = info_units_mode
        if node.paragraph_ranges:
            node.source_page_ids = index.page_ids_for_ranges(node.paragraph_ranges)
            node.source_paragraphs = index.format_source_paragraphs(node.paragraph_ranges)
            for nums in index._valid_numbers(node.paragraph_ranges):
                cited.update(nums)
        else:
            node.source_paragraphs = []
            nodes_without_ranges += 1
        if info_units_mode == "verbatim":
            node.info_units = build_verbatim_info_units(
                node.paragraph_ranges,
                index,
                node_title=node.title or node.node_id,
                max_chars=max_chars,
            )

    cited_count = len(cited)
    pct = 100.0 * cited_count / index.total if index.total else 0.0
    logger.info(
        "Paragraph anchors for %r: %d paragraphs, %d cited (%.1f%%), %d ranges discarded, "
        "%d nodes without valid ranges, info_units_mode=%s",
        document.document_name,
        index.total,
        cited_count,
        pct,
        stats.discarded_ranges,
        nodes_without_ranges,
        info_units_mode,
    )
