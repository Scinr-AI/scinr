"""
converters/config.py — Path constants and output directory resolution.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root-relative path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent

#: Folder where raw source files (PDF, DOCX, etc.) are placed by the user.
DEFAULT_SOURCE_DIR: Path = _PROJECT_ROOT / "files"

#: Output folder for production mode.
DEFAULT_OUTPUT_PROD: Path = _PROJECT_ROOT / "data" / "input"

#: Output folder for development / testing mode.
DEFAULT_OUTPUT_DEV: Path = _PROJECT_ROOT / "data" / "input-pruebas"


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def resolve_output_dir(
    dev: bool = False,
    output_override: str | None = None,
) -> Path:
    """Resolve and create the output directory.

    Priority: *output_override* > ``--dev`` flag > production default.

    Parameters
    ----------
    dev:
        If ``True`` and *output_override* is ``None``, use
        ``DEFAULT_OUTPUT_DEV``.
    output_override:
        Explicit output path provided by the user.  When set, both
        *dev* and defaults are ignored.

    Returns
    -------
    Path
        Resolved output directory (created if it did not exist).
    """
    if output_override is not None:
        path = Path(output_override)
    elif dev:
        path = DEFAULT_OUTPUT_DEV
    else:
        path = DEFAULT_OUTPUT_PROD

    path.mkdir(parents=True, exist_ok=True)
    return path
