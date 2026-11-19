"""
tests/unit/test_sql_boundaries.py -- Architectural guard for the SQL layer.

Outside ``sql/``, ``graph/sql/``, ``navigation/sql/``, ``storage/sql/`` and ``freeze/sql/``
the library must not import ``sqlalchemy``, ``psycopg``, ``aiosqlite`` or ``sqlite3``, and
``import scinr.newton`` must work without the SQL extras installed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scinr.newton as _newton

SRC_ROOT = Path(_newton.__file__).parent
ALLOWED_DIRS = tuple(
    SRC_ROOT / package / "sql" if package else SRC_ROOT / "sql"
    for package in ("", "graph", "navigation", "storage", "freeze")
)
FORBIDDEN = ("sqlalchemy", "psycopg", "aiosqlite", "sqlite3")


def _guarded_files() -> list[Path]:
    return sorted(
        p for p in SRC_ROOT.rglob("*.py") if not any(d in p.parents for d in ALLOWED_DIRS)
    )


def _violations(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in FORBIDDEN:
                found.append((node.lineno, name))
    return found


def test_guard_scans_the_library() -> None:
    files = _guarded_files()
    assert len(files) > 50
    assert not any("newton/sql/" in str(p) for p in files)


def test_no_sql_driver_imports_outside_the_sql_packages() -> None:
    offenders = [
        f"{p.relative_to(SRC_ROOT)}:{line}: import {name}"
        for p in _guarded_files()
        for line, name in _violations(ast.parse(p.read_text(encoding="utf-8"), filename=str(p)))
    ]
    assert not offenders, "SQL drivers imported outside the SQL packages:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "snippet, expected",
    [
        ("import sqlalchemy", 1),
        ("import sqlalchemy.dialects.postgresql", 1),
        ("from sqlalchemy import select", 1),
        ("from psycopg import sql", 1),
        ("import aiosqlite, os", 1),
        ("def f():\n    import sqlite3\n", 1),
        ("from . import sqlalchemy_helpers", 0),
        ("import sqlalchemy_utils_not_ours", 0),
    ],
)
def test_detector_self_check(snippet: str, expected: int) -> None:
    assert len(_violations(ast.parse(snippet))) == expected


def test_import_and_configure_work_without_sqlalchemy() -> None:
    """With the SQL libraries blocked, the package imports and accepts a SQL configuration."""
    code = textwrap.dedent(
        """
        import sys
        for name in ("sqlalchemy", "psycopg", "aiosqlite"):
            sys.modules[name] = None  # any import of them raises ImportError

        import scinr.newton
        from scinr.newton import configure

        cfg = configure(graph_backend="sqlite", sql_url="sqlite:///x.db")
        assert cfg.graph_backend == "sqlite"

        from scinr.newton.sql.engine import get_sync_engine
        from scinr.newton.exceptions import ConfigurationError
        try:
            get_sync_engine()
        except ConfigurationError as exc:
            assert "pip install scinr[postgres]" in str(exc), exc
        else:
            raise AssertionError("expected ConfigurationError")
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
