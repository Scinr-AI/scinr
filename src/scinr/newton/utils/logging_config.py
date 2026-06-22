"""
utils/logging_config.py — Logging setup for scinr-ingest.

Library mode (default)
----------------------
When called without arguments, only a console handler is added — no files
are created, no directories are touched. This is the correct behaviour for
a library: the application that *uses* scinr-ingest is responsible for
configuring file logging if it wants it.

CLI mode
--------
When ``log_dir`` is provided (the CLI passes ``Path("logs")``), two
rotating daily-folder file handlers are added in addition to the console
handler:

    <log_dir>/
    └── YYYY-MM-DD/
        ├── scinr.log          ← INFO+, retained 30 days
        └── scinr.errors.log   ← ERROR+ only, retained 90 days

Usage::

    # Library — console only, no files:
    from scinr.newton.utils.logging_config import setup_logging
    setup_logging()

    # CLI — console + daily file rotation under ./logs/:
    setup_logging(log_dir=Path("logs"))
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure logging for scinr-ingest.

    Parameters
    ----------
    log_dir:
        Directory under which dated sub-folders and log files are created.
        When *None* (the default), only a console handler is configured —
        no files or directories are created.  Pass an explicit path (e.g.
        ``Path("logs")``) to enable file logging.
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — always present
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is None:
        # Library mode: done — let the caller manage file logging.
        return

    # CLI / explicit mode: create dated sub-folder and add file handlers.
    today_str = datetime.now().strftime("%Y-%m-%d")
    dated_dir = Path(log_dir) / today_str
    dated_dir.mkdir(parents=True, exist_ok=True)

    all_fh = logging.FileHandler(dated_dir / "scinr.log", encoding="utf-8")
    all_fh.setLevel(logging.INFO)
    all_fh.setFormatter(fmt)

    err_fh = logging.FileHandler(dated_dir / "scinr.errors.log", encoding="utf-8")
    err_fh.setLevel(logging.ERROR)
    err_fh.setFormatter(fmt)

    root.addHandler(all_fh)
    root.addHandler(err_fh)

    _cleanup_old_logs(Path(log_dir), all_days=30, errors_days=90)


def _cleanup_old_logs(logs_root: Path, all_days: int, errors_days: int) -> None:
    """Delete log files older than their retention period; remove empty dirs."""
    if not logs_root.exists():
        return
    cutoff_all = datetime.now() - timedelta(days=all_days)
    cutoff_errors = datetime.now() - timedelta(days=errors_days)
    for entry in sorted(logs_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            dir_date = datetime.strptime(entry.name, "%Y-%m-%d")
        except ValueError:
            continue
        if dir_date < cutoff_all:
            all_log = entry / "scinr.log"
            if all_log.exists():
                all_log.unlink()
        if dir_date < cutoff_errors:
            err_log = entry / "scinr.errors.log"
            if err_log.exists():
                err_log.unlink()
        if not any(entry.iterdir()):
            entry.rmdir()
