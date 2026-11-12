"""Naming of documents derived from source files.

A source file's document name is its full file name, extension included and
case preserved: ``Modulo/Informe.PDF`` → ``"Informe.PDF"``. The ``doc_path`` of
the document is ``{folder_path}/{name}`` (or just ``name`` at the root), so two
files that only differ by extension (``informe.txt`` and ``informe.html``)
never collide.

Every place that derives a document name or ``doc_path`` from a *source* file
uses :func:`source_document_name`. Names derived from an intermediate or
extraction JSON file (``informe.pdf.json`` → ``informe.pdf``) keep using the
JSON file's stem: that is what keeps JSON files written by earlier versions
(``informe.json`` → ``informe``) ingestible under their old name.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["source_document_name", "source_doc_path"]


def source_document_name(path: Path) -> str:
    """Return the document name of the source file *path*.

    The file name with its extension, unchanged: ``a.b.PDF`` → ``"a.b.PDF"``.
    """
    return path.name


def source_doc_path(path: Path, folder_path: str | None) -> str:
    """Return the ``doc_path`` of the source file *path* inside *folder_path*.

    *folder_path* is the POSIX-style folder relative to the input root, or
    ``None`` / ``""`` for a file at the root.
    """
    name = source_document_name(path)
    return f"{folder_path}/{name}" if folder_path else name
