"""
tests/unit/test_sql_graph_schema.py -- ``graph/sql/schema.py`` knows the same index names as Neo4j.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from scinr.newton.graph.neo4j.schema import _REGULAR_INDEXES  # noqa: E402
from scinr.newton.graph.sql.schema import KNOWN_INDEXES  # noqa: E402


def test_known_indexes_are_the_neo4j_ones() -> None:
    assert set(dict(_REGULAR_INDEXES)) == KNOWN_INDEXES
