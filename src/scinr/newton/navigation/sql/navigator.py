"""
navigation/sql/navigator.py — ``SqlGraphNavigator``.

The Postgres / SQLite :class:`~scinr.newton.navigation.GraphNavigator`. It is assembled from
one mixin per method group (``_documents``, ``_structure``, …) over the shared
``_SqlRuntime`` (connection check + read helpers), the same partition as the Neo4j
navigator. All SQL lives in the mixins; this module only wires them together.
"""

from __future__ import annotations

from scinr.newton.navigation.base import GraphNavigator
from scinr.newton.navigation.sql._annotation import _AnnotationMixin
from scinr.newton.navigation.sql._documents import _DocumentsMixin
from scinr.newton.navigation.sql._entities import _EntitiesMixin
from scinr.newton.navigation.sql._explore import _ExploreMixin
from scinr.newton.navigation.sql._info_units import _InfoUnitsMixin
from scinr.newton.navigation.sql._instances import _ModelInstancesMixin
from scinr.newton.navigation.sql._introspection import _IntrospectionMixin
from scinr.newton.navigation.sql._power import _PowerMixin
from scinr.newton.navigation.sql._structure import _StructureMixin

__all__ = ["SqlGraphNavigator"]


class SqlGraphNavigator(
    _DocumentsMixin,
    _StructureMixin,
    _InfoUnitsMixin,
    _AnnotationMixin,
    _ModelInstancesMixin,
    _EntitiesMixin,
    _IntrospectionMixin,
    _PowerMixin,
    _ExploreMixin,
    GraphNavigator,
):
    """SQLAlchemy Core implementation of :class:`GraphNavigator`.

    Construct via :func:`scinr.newton.navigation.get_graph_navigator` /
    :func:`scinr.newton.navigation.graph_navigator`. It uses the process-wide engines of
    :mod:`scinr.newton.sql.engine` and never disposes them. ``dialect`` is ``"postgres"`` or
    ``"sqlite"`` once connected.
    """
