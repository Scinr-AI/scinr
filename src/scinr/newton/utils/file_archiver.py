"""utils/file_archiver.py — Move processed files to a ``processed/`` sub-folder.

After all pipeline stages complete successfully, this module is used to move
source files out of their original directories and into a ``processed/``
sub-folder that lives alongside them.  Files that have already been moved
(name collision) are renamed with a numeric suffix before being moved, so
existing ``processed/`` contents are never silently overwritten.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_archive_path(source: Path) -> Path:
    """Return a unique destination path inside ``source.parent / 'processed'``.

    The destination directory is ``source.parent / 'processed'``.  If a file
    with the same name already exists there, a numeric suffix is appended to
    the stem before the extension: ``doc.pdf`` → ``doc_1.pdf`` → ``doc_2.pdf``
    and so on, following the same collision-resolution convention used by
    :meth:`converters.base.BaseConverter._resolve_output_path`.

    Parameters
    ----------
    source:
        File to be archived.  Must exist.

    Returns
    -------
    Path
        A path inside ``source.parent / 'processed'`` that does not yet exist.
    """
    dest_dir = source.parent / "processed"
    candidate = dest_dir / source.name
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        new_name = f"{source.stem}_{counter}{source.suffix}"
        candidate = dest_dir / new_name
        if not candidate.exists():
            logger.warning(
                "Archive collision: %s already exists — renaming to %s",
                dest_dir / source.name,
                new_name,
            )
            return candidate
        counter += 1


def archive_processed_files(
    files: list[Path],
    label: str = "files",
) -> dict[Path, Path]:
    """Move each file in *files* to a ``processed/`` sub-folder beside it.

    For every path in *files*:

    1. Compute ``dest = source.parent / "processed" / source.name`` (with
       numeric-suffix collision resolution via :func:`_resolve_archive_path`).
    2. Create the ``processed/`` directory if it does not exist.
    3. Move the file using :func:`shutil.move` (string paths for
       Windows/WSL compatibility).
    4. Log the move at ``INFO`` level.  Errors are logged at ``WARNING``
       level and do **not** abort processing of the remaining files.

    Parameters
    ----------
    files:
        List of source :class:`~pathlib.Path` objects to archive.
    label:
        Human-readable label used in log messages to identify the batch
        (e.g. ``"intermediate JSONs"`` or ``"raw files"``).

    Returns
    -------
    dict[Path, Path]
        Mapping of ``source → destination`` for every file that was
        successfully moved.  Files that failed to move are omitted.
    """
    archived: dict[Path, Path] = {}

    logger.info("Archiving %d %s …", len(files), label)

    for source in files:
        try:
            dest = _resolve_archive_path(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            logger.info("Archived %s → %s", source, dest)
            archived[source] = dest
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not archive %s: %s",
                source,
                exc,
            )

    logger.info(
        "Archive complete for %s: %d/%d moved.",
        label,
        len(archived),
        len(files),
    )
    return archived