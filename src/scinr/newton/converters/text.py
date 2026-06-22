"""
converters/text.py — Plain text and Markdown converter.

The file is paginated into chunks of ``_LINES_PER_PAGE`` lines.  Each
chunk becomes a separate ``IntermediatePage``.  ``.md`` files are passed
through unchanged (already Markdown); ``.txt`` files are used directly.
No lines are ever discarded.
"""

from __future__ import annotations

import logging
from pathlib import Path

from scinr.newton.converters.base import BaseConverter, ConversionError, IntermediateDocument

logger = logging.getLogger(__name__)

_LINES_PER_PAGE = 100


class TextConverter(BaseConverter):
    """Convert ``.txt`` and ``.md`` files to the intermediate format.

    The file content is split into pages of ``_LINES_PER_PAGE`` lines.
    Markdown files are passed through as-is; plain text files are used
    directly.  An empty file produces a single empty page.
    """

    supported_extensions: frozenset[str] = frozenset({"txt", "md"})

    def convert(self, source: Path) -> IntermediateDocument:
        """Convert a text or Markdown file.

        Parameters
        ----------
        source:
            Path to the ``.txt`` or ``.md`` file.

        Returns
        -------
        IntermediateDocument
            Multi-page document (one page per ``_LINES_PER_PAGE`` lines).
            An empty file produces a single empty page.

        Raises
        ------
        ConversionError
            If the file cannot be read.
        """
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            content = source.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ConversionError(f"Cannot read {source}: {exc}") from exc

        lines = content.splitlines()

        if not lines:
            return IntermediateDocument(pages=[self._make_page(index=0, markdown="")])

        chunks = [
            lines[i : i + _LINES_PER_PAGE]
            for i in range(0, len(lines), _LINES_PER_PAGE)
        ]

        if len(chunks) > 1:
            logger.info(
                "Converting %s: %d lines → %d page(s)",
                source.name,
                len(lines),
                len(chunks),
            )

        pages = [
            self._make_page(index=i, markdown="\n".join(chunk))
            for i, chunk in enumerate(chunks)
        ]
        return IntermediateDocument(pages=pages)
