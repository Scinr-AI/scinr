"""
tests/unit/test_graph_boundaries.py — Architectural guard for the graph layer.

Outside ``graph/neo4j/`` and ``navigation/neo4j/`` the library must not import
the ``neo4j`` driver, must not catch or import ``neo4j.exceptions`` and must
not contain Cypher text (including constant fragments of f-strings).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import scinr.newton as _newton

SRC_ROOT = Path(_newton.__file__).parent
ALLOWED_DIRS = (
    SRC_ROOT / "graph" / "neo4j",
    SRC_ROOT / "navigation" / "neo4j",
)

CYPHER_RE = re.compile(
    r"\b(MATCH|MERGE|UNWIND|DETACH DELETE|OPTIONAL MATCH)\b(\s|\(|\{|$)"
)


def _guarded_files() -> list[Path]:
    return sorted(
        p
        for p in SRC_ROOT.rglob("*.py")
        if not any(d in p.parents for d in ALLOWED_DIRS)
    )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of the Constant nodes that are docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _driver_import_violations(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "neo4j" or alias.name.startswith("neo4j."):
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and (mod == "neo4j" or mod.startswith("neo4j.")):
                found.append((node.lineno, f"from {mod} import ..."))
    return found


def _cypher_violations(tree: ast.AST) -> list[tuple[int, str]]:
    docstrings = _docstring_nodes(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and CYPHER_RE.search(node.value)
        ):
            found.append((node.lineno, node.value.strip()[:60]))
    return found


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC_ROOT))


def test_guard_scans_the_library() -> None:
    files = _guarded_files()
    assert len(files) > 50
    assert not any("graph/neo4j" in _rel(p) for p in files)
    assert not any("navigation/neo4j" in _rel(p) for p in files)


def test_no_neo4j_driver_imports_outside_backend() -> None:
    offenders = [
        f"{_rel(p)}:{line}: {text}"
        for p in _guarded_files()
        for line, text in _driver_import_violations(_parse(p))
    ]
    assert not offenders, "neo4j imported outside the backend:\n" + "\n".join(offenders)


def test_no_cypher_text_outside_backend() -> None:
    offenders = [
        f"{_rel(p)}:{line}: {text!r}"
        for p in _guarded_files()
        for line, text in _cypher_violations(_parse(p))
    ]
    assert not offenders, "Cypher outside the backend:\n" + "\n".join(offenders)


def test_no_neo4j_exceptions_referenced_outside_backend() -> None:
    """Nothing imports or catches ``neo4j.exceptions`` by attribute path."""
    offenders: list[str] = []
    for p in _guarded_files():
        for node in ast.walk(_parse(p)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "neo4j" and node.attr == "exceptions":
                    offenders.append(f"{_rel(p)}:{node.lineno}")
    assert not offenders, "neo4j.exceptions used outside the backend:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize(
    "snippet, expected",
    [
        ('q = f"MATCH {pat} RETURN d"', 1),
        ('q = "UNWIND $rows AS r"', 1),
        ('q = "OPTIONAL MATCH (a)"', 1),
        ('q = "DETACH DELETE n"', 1),
        ('q = "MERGE(n)"', 1),
        ('q = "MATCHING rows"', 0),
        ('q = "match everything"', 0),
        ('def f():\n    """MATCH (a) in docs"""\n', 0),
    ],
)
def test_cypher_detector_self_check(snippet: str, expected: int) -> None:
    assert len(_cypher_violations(ast.parse(snippet))) == expected
