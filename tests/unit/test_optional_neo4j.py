"""
tests/unit/test_optional_neo4j.py — ``neo4j`` is an optional extra (``scinr[neo4j]``).

Each case runs in a fresh interpreter with a meta-path finder that makes ``import neo4j``
fail, so the check does not depend on whether the driver is installed in the test
environment: the library must import and run on SQLite without it, and asking for
``graph_backend='neo4j'`` must fail with the install hint.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

BLOCK_NEO4J = textwrap.dedent(
    """
    import importlib.abc, sys

    class _BlockNeo4j(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "neo4j" or name.startswith("neo4j."):
                raise ImportError("neo4j is blocked by the test")
            return None

    sys.meta_path.insert(0, _BlockNeo4j())
    """
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_NEO4J + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_scinr_newton_without_neo4j() -> None:
    result = _run(
        """
        import scinr.newton
        assert "neo4j" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_sqlite_graph_write_without_neo4j(tmp_path: Path) -> None:
    db = tmp_path / "newton.db"
    result = _run(
        f"""
        from scinr.newton import configure
        from scinr.newton.graph import get_graph_store
        from scinr.newton.models.document_structure import Document

        configure(
            graph_backend="sqlite",
            storage_backend="sqlite",
            sql_url="sqlite:///{db}",
        )
        store = get_graph_store()
        store.schema.setup()
        document = Document(
            document_name="a.txt",
            doc_path="a.txt",
            raw_file_id="raw-a.txt",
            document_structure=[],
        )
        store.documents.insert_document_graph(document, 1, False)
        assert store.documents.current_latest_version("a.txt", "__public__") == 1
        assert "neo4j" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("entry", ["graph_store", "navigator"])
def test_neo4j_backend_without_extra_explains_install(entry: str) -> None:
    call = (
        "create_graph_store()"
        if entry == "graph_store"
        else "asyncio.run(get_graph_navigator())"
    )
    result = _run(
        f"""
        import asyncio
        from scinr.newton import configure, GraphConnectionError
        from scinr.newton.graph.factory import create_graph_store
        from scinr.newton.navigation.factory import get_graph_navigator

        configure(
            graph_backend="neo4j",
            neo4j_user="u",
            neo4j_password="p",
            neo4j_database="d",
        )
        try:
            {call}
        except GraphConnectionError as exc:
            assert "pip install 'scinr[neo4j]'" in str(exc), str(exc)
        else:
            raise SystemExit("no error raised")
        """
    )
    assert result.returncode == 0, result.stderr
