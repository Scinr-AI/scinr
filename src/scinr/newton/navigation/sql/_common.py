"""
navigation/sql/_common.py — Shared runtime for the SQL backend mixins.

``_SqlRuntime`` owns the connection check and the low-level read helpers (``_read`` /
``_read_one``) and the building blocks every mixin shares: scope predicates, the document
anchor, the node dictionaries the neutral mapping (``navigation/_map.py``) consumes, and the
``is_shell`` heuristic. It is the SQL twin of ``navigation/neo4j/_common.py``.

The engines are the process-wide ones of :mod:`scinr.newton.sql.engine`; ``close()`` never
disposes them (they belong to the process, like the shared Neo4j driver). Reads use the
``sqlite_deferred`` execution option, so a SQLite read does not queue behind the single writer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.expression import Select

from scinr.newton.exceptions import GraphConnectionError, NavigationError
from scinr.newton.navigation._safe import resolve_depth
from scinr.newton.navigation.base import DEFAULT_MAX_DEPTH, INSTANCE_CONTAINMENT_DEPTH
from scinr.newton.navigation.scope import Scope, ScopeKind
from scinr.newton.navigation.sql._translate import Property, prop_equals
from scinr.newton.navigation.sql._walk import UNBOUNDED_HOPS, reached, walk
from scinr.newton.sql import tables as T
from scinr.newton.sql.dialect import SQLITE_DEFERRED_OPTION, Dialect
from scinr.newton.sql.engine import get_async_engine, get_dialect_for_config
from scinr.newton.sql.errors import async_sql_errors
from scinr.newton.sql.retry import with_sql_retry
from scinr.newton.sql.rows import row_to_node
from scinr.newton.sql.scope import scope_conditions
from scinr.newton.utils.redaction import redact_secrets
from scinr.newton.utils.tenancy import PUBLIC_TENANT

__all__ = ["_SqlRuntime"]


class _SqlRuntime:
    """Connection check + read helpers shared by every Group mixin."""

    dialect = "sql"
    """``"postgres"`` or ``"sqlite"`` once connected (``execute_raw`` checks it)."""

    def __init__(self) -> None:
        self._dialect_obj: Dialect | None = None
        self._key_field_counts: dict[str, int | None] = {}

    # -- lifecycle ------------------------------------------------------------

    @property
    def _dl(self) -> Dialect:
        """The dialect of the configured ``sql_url``."""
        if self._dialect_obj is None:
            self._dialect_obj = get_dialect_for_config()
        return self._dialect_obj

    async def connect(self) -> None:
        """Create / verify the schema (like ``get_storage()``) and check the engine answers.

        The shared engines are created on first use.
        """
        from scinr.newton.sql.schema import ensure_schema

        self.dialect = self._dl.name
        async with async_sql_errors():
            await asyncio.to_thread(ensure_schema)
        await self.ping()

    async def close(self) -> None:
        """Nothing to close: the engines are shared by the whole process."""
        return None

    async def ping(self) -> bool:
        try:
            rows = await self._read(sa.select(sa.literal(1).label("ok")))
            return bool(rows and rows[0]["ok"] == 1)
        except GraphConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalise any driver error
            raise GraphConnectionError(
                f"The SQL database is not reachable: {redact_secrets(str(exc))}"
            ) from exc

    # -- read helpers ------------------------------------------------------------

    async def _read(self, statement: Any) -> list[dict[str, Any]]:
        """Run *statement* in a read transaction (with retry) and return dict rows."""
        engine = get_async_engine().execution_options(**{SQLITE_DEFERRED_OPTION: True})

        async def _run() -> list[dict[str, Any]]:
            async with engine.connect() as connection:
                result = await connection.execute(statement)
                return [dict(row._mapping) for row in result]

        async with async_sql_errors():
            return await with_sql_retry(_run)

    async def _read_one(self, statement: Any) -> dict[str, Any] | None:
        rows = await self._read(statement)
        return rows[0] if rows else None

    # -- nodes ---------------------------------------------------------------------

    @staticmethod
    def _node(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
        """The property dict (plus ``_labels``) of a node row, as ``_map`` consumes it."""
        labels, props = row_to_node(table, row)
        return {**props, "_labels": list(labels)}

    @staticmethod
    def _limit(stmt: Select, limit: int | None, skip: int = 0) -> Select:
        """``SKIP`` / ``LIMIT`` with the rules of ``_limit_clause`` (a falsy skip is none)."""
        if skip:
            stmt = stmt.offset(int(skip))
        if limit is not None and limit >= 0:
            stmt = stmt.limit(int(limit))
        return stmt

    # -- scope ---------------------------------------------------------------------

    def _scope(
        self,
        table: Any,
        scope: Scope,
        kind: ScopeKind = "scalar",
        *,
        tenant: bool = True,
    ) -> list[ColumnElement[bool]]:
        """Scope predicates on *table* (the SQL ``_scope_where``)."""
        return scope_conditions(scope, table, kind=kind, tenant=tenant, dialect=self._dl)

    def _doc_conditions(
        self,
        path: str,
        version: int | None,
        scope: Scope,
        kind: ScopeKind = "tenant_only",
    ) -> list[ColumnElement[bool]]:
        """Conditions resolving a document by path and version (default: the latest).

        Like ``_doc_match`` + ``_doc_where``: no tenant in the pattern; the scope adds it.
        """
        doc = T.documents
        conditions: list[ColumnElement[bool]] = [doc.c.path == path]
        if version is None:
            conditions.append(doc.c.latest == sa.true())
        else:
            conditions.append(doc.c.version == int(version))
        conditions.extend(self._scope(doc, scope, kind))
        return conditions

    def _doc_start(
        self, path: str, version: int | None, scope: Scope, kind: ScopeKind = "tenant_only"
    ) -> Select:
        """``(label, key)`` of the anchor documents: the start of a walk."""
        return sa.select(
            sa.literal("Document", sa.Text).label("label"), T.documents.c._key.label("key")
        ).where(*self._doc_conditions(path, version, scope, kind))

    @staticmethod
    def _node_start(label: str, key_column: Any, conditions: Sequence[Any]) -> Select:
        """``(label, key)`` of the nodes whose key column satisfies *conditions*."""
        return sa.select(sa.literal(label, sa.Text).label("label"), key_column.label("key")).where(
            *conditions
        )

    def _node_anchor(self, node_id: str, scope: Scope, kind: ScopeKind = "tenant_only") -> Select:
        """``(label, key)`` of the anchor structure node (``_doc_where`` of the Cypher)."""
        sn = T.structure_nodes
        return self._node_start(
            "StructureNode", sn.c.id, [sn.c.id == node_id, *self._scope(sn, scope, kind)]
        )

    @staticmethod
    def _in_document_path(path: str, key_column: Any) -> ColumnElement[bool]:
        """``EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path: $p}) }``.

        *key_column* holds the id of a structure node.
        """
        doc = T.documents
        start = sa.select(
            sa.literal("Document", sa.Text).label("label"), doc.c._key.label("key")
        ).where(doc.c.path == path)
        cte = walk(
            start,
            max_hops=UNBOUNDED_HOPS,
            types=("HAS_STRUCTURE", "HAS_CHILD"),
            name="in_doc",
        )
        return key_column.in_(sa.select(reached(cte, "StructureNode", name="in_doc_r").c.key))

    @staticmethod
    def _instances_from(documents_start: Select, depth: int, name: str) -> Select:
        """Keys of the ``ModelInstance`` nodes extracted from the given documents.

        ``(doc)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->
        (:ExtractionResult)-[HAS_*{1..depth}]->(mi:ModelInstance)``: *documents_start* is a
        ``(label, key)`` select of the documents.
        """
        edges = T.edges
        structure = reached(
            walk(
                documents_start,
                max_hops=UNBOUNDED_HOPS,
                types=("HAS_STRUCTURE", "HAS_CHILD"),
                name=f"{name}_sn",
            ),
            "StructureNode",
            name=f"{name}_snr",
        )
        results = sa.select(
            sa.literal("ExtractionResult", sa.Text).label("label"), edges.c.dst_key.label("key")
        ).where(
            edges.c.type == "HAS_EXTRACTION",
            edges.c.src_label == "StructureNode",
            edges.c.src_key.in_(sa.select(structure.c.key)),
            edges.c.dst_label == "ExtractionResult",
        )
        contained = reached(
            walk(results, max_hops=depth, prefix="HAS_", name=f"{name}_mi"),
            "ModelInstance",
            name=f"{name}_mir",
        )
        return sa.select(contained.c.key)

    def _instances_of_path(self, path: str, depth: int) -> Select:
        """:meth:`_instances_from` the documents at *path* (any tenant and version)."""
        start = sa.select(
            sa.literal("Document", sa.Text).label("label"), T.documents.c._key.label("key")
        ).where(T.documents.c.path == path)
        return self._instances_from(start, depth, "by_path")

    def _order_by_property(self, table: Any, field: str, *, desc: bool = False) -> list[Any]:
        """ORDER BY elements of ``n.<field>`` with the ordering of Cypher values.

        Strings sort before booleans before numbers, and ``null`` last; a typed column keeps
        its own type.
        """
        prop = Property(table, field, self._dl)
        dl = self._dl
        column = prop.column
        if column is not None:
            if isinstance(column.type, sa.Text):
                return [dl.order_text(column, desc=desc)]
            return [dl.order_value(column, desc=desc)]
        kind = dl.json_kind(table.c.props, field)
        rank = sa.case(
            (kind == "string", 0), (kind == "boolean", 1), (kind == "number", 2), else_=3
        )
        text = prop.values("string")
        number = prop.values("number")
        boolean = prop.values("boolean")
        assert text is not None and number is not None and boolean is not None
        return [
            dl.order_value(rank, desc=desc),
            dl.order_text(text, desc=desc),
            dl.order_value(boolean, desc=desc),
            dl.order_value(number, desc=desc),
        ]

    # -- small shared utilities ------------------------------------------------

    @staticmethod
    def _resolve_depth(depth: int | None, *, default: int = DEFAULT_MAX_DEPTH) -> int:
        if depth is None:
            return default
        return resolve_depth(depth)

    def _containment_depth(self, depth: int | None) -> int:
        return self._resolve_depth(depth, default=INSTANCE_CONTAINMENT_DEPTH)

    async def _single_document_scope(self, path: str, version: int | None, scope: Scope) -> Scope:
        """Pin *scope* to the one tenant a single-document method means.

        With a single tenant in scope nothing is queried. Otherwise the tenants that hold
        *path* (latest, or *version*) are listed: one or none → the scope as is; a tenant
        plus the public one under ``include_public`` → that tenant (it shadows the public
        document); anything else is ambiguous → :class:`NavigationError`.
        """
        if scope.tenants is not None and len(scope.tenants) == 1:
            return scope
        doc = T.documents
        version_condition = (
            doc.c.version == int(version) if version is not None else doc.c.latest == sa.true()
        )
        rows = await self._read(
            sa.select(doc.c.tenant_id).where(
                doc.c.path == path, version_condition, *self._scope(doc, scope, "tenant_only")
            )
        )
        held = [r["tenant_id"] for r in rows]
        if len(held) <= 1:
            return scope
        if scope.include_public and scope.tenants is not None:
            own = [t for t in held if t is not None and t != PUBLIC_TENANT]
            if len(own) == 1:
                return scope.with_tenant(own[0])
        raise NavigationError(
            f"document path {path!r} exists in several tenants "
            f"({sorted(str(t) for t in held)}); pass tenant_id to pick one"
        )

    async def _instance_key_field_count(self, model_class: str) -> int | None:
        """Number of ``instance_key`` fields declared for *model_class* (memoised).

        ``None`` when the model has no ``catalog_models`` entry at all.
        """
        if model_class in self._key_field_counts:
            return self._key_field_counts[model_class]
        exists = await self._read_one(
            sa.select(T.catalog_models.c.name).where(T.catalog_models.c.name == model_class)
        )
        value: int | None = None
        if exists is not None:
            fields = T.model_fields
            link = T.catalog_edges
            count = await self._read_one(
                sa.select(sa.func.count().label("n"))
                .select_from(link.join(fields, fields.c._key == link.c.dst_key))
                .where(
                    link.c.type == "HAS_FIELD",
                    link.c.src_label == "CatalogModel",
                    link.c.src_key == model_class,
                    link.c.dst_label == "ModelField",
                    prop_equals(fields, "is_instance_key", True, self._dl),
                )
            )
            value = int((count or {}).get("n", 0) or 0)
        self._key_field_counts[model_class] = value
        return value

    async def _is_shell(self, node_props: Mapping[str, Any]) -> bool | None:
        """Heuristic: a node with only ``uid`` + ``model_class`` + key fields."""
        mc = node_props.get("model_class")
        if not mc:
            return None
        n_keys = await self._instance_key_field_count(str(mc))
        if not n_keys:
            return None
        real_props = sum(1 for k in node_props if not k.startswith("_"))
        return real_props <= n_keys + 2

    @staticmethod
    def _roles_param(roles: Sequence[str] | None) -> list[str] | None:
        return list(roles) if roles else None
