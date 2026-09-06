"""
navigation/neo4j/navigator.py — ``Neo4jGraphNavigator``.

The default (and, today, only) :class:`~scinr.newton.navigation.GraphNavigator`
backend. It is assembled from one mixin per method group (``_documents``,
``_structure``, …) over the shared ``_Neo4jRuntime`` (driver lifecycle + read
helpers). All Cypher lives in the mixins; this module only wires them together.
"""

from __future__ import annotations

from scinr.newton.navigation.base import GraphNavigator
from scinr.newton.navigation.neo4j._annotation import _AnnotationMixin
from scinr.newton.navigation.neo4j._documents import _DocumentsMixin
from scinr.newton.navigation.neo4j._entities import _EntitiesMixin
from scinr.newton.navigation.neo4j._info_units import _InfoUnitsMixin
from scinr.newton.navigation.neo4j._instances import _ModelInstancesMixin
from scinr.newton.navigation.neo4j._introspection import _IntrospectionMixin
from scinr.newton.navigation.neo4j._power import _PowerMixin
from scinr.newton.navigation.neo4j._structure import _StructureMixin


class Neo4jGraphNavigator(
    _DocumentsMixin,
    _StructureMixin,
    _InfoUnitsMixin,
    _AnnotationMixin,
    _ModelInstancesMixin,
    _EntitiesMixin,
    _IntrospectionMixin,
    _PowerMixin,
    GraphNavigator,
):
    """Cypher implementation of :class:`GraphNavigator`.

    Construct via :func:`scinr.newton.navigation.get_graph_navigator` /
    :func:`scinr.newton.navigation.graph_navigator` rather than directly. When
    ``driver`` is not supplied it reuses the shared async driver from
    ``ingest.config.get_async_driver()`` and never closes it; pass an explicit
    ``driver`` to own the connection lifecycle.

    Args:
        driver: An open ``neo4j.AsyncDriver``. Optional.
        database: Neo4j database name. Defaults to ``config.neo4j_database``.
    """

    dialect = "cypher"
