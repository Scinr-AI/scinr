"""
navigation/neo4j/_common.py — Shared runtime for the Neo4j backend mixins.

``_Neo4jRuntime`` owns the driver/session lifecycle and the low-level read
helpers (``_read`` / ``_read_one`` / ``_stream``). Every Group mixin inherits it
so ``self._read(...)`` is available everywhere; the concrete
``Neo4jGraphNavigator`` composes all the mixins.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from scinr.newton.exceptions import GraphConnectionError, NavigationError
from scinr.newton.navigation.base import (
    DEFAULT_MAX_DEPTH,
    INSTANCE_CONTAINMENT_DEPTH,
)
from scinr.newton.navigation.neo4j._safe import resolve_depth
from scinr.newton.navigation.scope import Scope, ScopeKind
from scinr.newton.utils.redaction import redact_secrets
from scinr.newton.utils.tenancy import PUBLIC_TENANT

logger = logging.getLogger(__name__)

# Relationship types the pipeline writes *structurally* (everything that is not a
# one-off normalised Triple predicate). Used only as a documented reference /
# ordering hint; ``list_relationship_types`` computes the real set from the graph.
STRUCTURAL_REL_HINT: tuple[str, ...] = (
    "IS_COMPOSED_OF", "HAS_NEWER_VERSION",
    "HAS_STRUCTURE", "HAS_CHILD", "HAS_INFO_UNIT",
    "HAS_MODEL_DECISION", "MATCHED_MODEL", "HAS_COMPLEMENTARY_MATCH",
    "REFERS_TO_MODEL", "HAS_SUPPLEMENTARY_FIELD", "HAS_PROPOSED_MODEL",
    "HAS_PROPOSED_FIELD",
    "HAS_EXTRACTION", "USES_PRIMARY_MODEL", "USES_COMPLEMENTARY_MODEL",
    "REFERENCES", "HAS_ENTITY",
    "HAS_FIELD", "AGGREGATES", "BELONGS_TO_THEME", "PRODUCES_ENTITY",
    "HAS_SUBTOPIC",
)


class _Neo4jRuntime:
    """Driver lifecycle + read helpers shared by every Group mixin."""

    dialect = "cypher"

    def __init__(self, *, driver: Any = None, database: str | None = None) -> None:
        #: A driver handed in from outside — we never close it and never swap it.
        self._external_driver = driver is not None
        self._driver = driver
        #: True only when *this* instance created the driver (today: never — we
        #: always borrow the shared ``get_async_driver()`` singleton).
        self._owns_driver = False
        self._database_override = database
        self._key_field_counts: dict[str, int | None] = {}

    # -- lifecycle ------------------------------------------------------------

    async def connect(self) -> None:
        if self._driver is None:
            from scinr.newton.ingest.config import get_async_driver

            self._driver = get_async_driver()  # shared singleton — never closed here
            self._owns_driver = False
        await self.ping()

    async def close(self) -> None:
        if self._owns_driver and self._driver is not None:
            await self._driver.close()
        if not self._external_driver:
            self._driver = None

    async def ping(self) -> bool:
        try:
            rec = await self._read_one("RETURN 1 AS ok")
            return bool(rec and rec.get("ok") == 1)
        except GraphConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalise any driver error
            raise GraphConnectionError(f"Neo4j is not reachable: {redact_secrets(str(exc))}") from exc

    def _database(self) -> str | None:
        if self._database_override is not None:
            return self._database_override or None
        try:
            from scinr.newton.config import get_config

            return get_config().neo4j_database or None
        except Exception:  # noqa: BLE001
            return None

    def _live_driver(self) -> Any:
        if self._driver is None:
            raise GraphConnectionError("navigator is not connected; call connect() first")
        if not self._external_driver:
            # Re-fetch the shared singleton so a re-configure() is transparent.
            try:
                from scinr.newton.ingest.config import get_async_driver

                self._driver = get_async_driver()
            except Exception:  # noqa: BLE001 — keep the handle we already have
                pass
        return self._driver

    # -- read helpers ------------------------------------------------------------

    async def _read(self, cypher: str, /, **params: Any) -> list[dict[str, Any]]:
        """Run *cypher* in a READ transaction (with retry) and return dict rows."""
        from scinr.newton.utils.neo4j_retry import with_neo4j_retry

        driver = self._live_driver()

        async def _run() -> list[dict[str, Any]]:
            async with driver.session(database=self._database()) as session:
                async def _tx(tx: Any) -> list[dict[str, Any]]:
                    res = await tx.run(cypher, **params)
                    return [r.data() async for r in res]

                return await session.execute_read(_tx)

        try:
            return await with_neo4j_retry(_run)
        except GraphConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            from neo4j.exceptions import Neo4jError, ServiceUnavailable

            if isinstance(exc, (ServiceUnavailable,)):
                raise GraphConnectionError(redact_secrets(str(exc))) from exc
            if isinstance(exc, Neo4jError):
                raise
            raise

    async def _read_one(self, cypher: str, /, **params: Any) -> dict[str, Any] | None:
        rows = await self._read(cypher, **params)
        return rows[0] if rows else None

    async def _stream(self, cypher: str, /, **params: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield dict rows from a READ transaction without buffering the result."""
        driver = self._live_driver()
        async with driver.session(database=self._database()) as session:
            res = await session.run(cypher, **params)
            async for record in res:
                yield record.data()

    # -- small shared utilities ------------------------------------------------

    @staticmethod
    def _resolve_depth(depth: int | None, *, default: int = DEFAULT_MAX_DEPTH) -> int:
        if depth is None:
            return default
        return resolve_depth(depth)

    def _containment_depth(self, depth: int | None) -> int:
        return self._resolve_depth(depth, default=INSTANCE_CONTAINMENT_DEPTH)

    @staticmethod
    def _doc_match(alias: str, *, version: int | None) -> str:
        """Return a ``MATCH`` pattern fragment resolving a document by version.

        ``version`` given → pin it; omitted → ``latest = true``. The caller
        always passes ``path=$path`` and (when relevant) ``version=$version``.

        The pattern carries no tenant: a :Document is keyed by
        ``(tenant_id, path, version)``, so the same path can match one document
        per tenant. Callers add the tenant with :meth:`_scope_where` /
        :meth:`_doc_where`, and single-document methods first resolve which
        tenant they mean with :meth:`_single_document_scope`.
        """
        if version is None:
            return f"({alias}:Document {{path: $path, latest: true}})"
        return f"({alias}:Document {{path: $path, version: $version}})"

    @staticmethod
    def _scope_where(
        alias: str,
        clauses: list[str],
        params: dict[str, Any],
        scope: Scope,
        kind: ScopeKind = "scalar",
        *,
        tenant: bool = True,
    ) -> None:
        """Append the scope predicates on *alias* to *clauses* / *params* (in place).

        Plain property predicates on the denormalized tenant / provenance
        properties written at ingestion — never a traversal up to ``:Document``.
        *kind* tells how the node stores its provenance (see
        :data:`~scinr.newton.navigation.scope.ScopeKind`).
        """
        clauses.extend(scope.clauses(alias, kind, tenant=tenant))
        params.update(scope.params(tenant=tenant))

    @staticmethod
    def _doc_where(
        alias: str, params: dict[str, Any], scope: Scope, kind: ScopeKind = "tenant_only"
    ) -> str:
        """``WHERE`` text applying the scope to a matched anchor *alias*.

        The default (tenant only) pins a document to the scope's tenant; pass
        ``kind="scalar"`` to filter the anchor by user/job as well.
        """
        clauses = scope.clauses(alias, kind)
        params.update(scope.params())
        return f"WHERE {' AND '.join(clauses)} " if clauses else ""

    async def _single_document_scope(
        self, path: str, version: int | None, scope: Scope
    ) -> Scope:
        """Pin *scope* to the one tenant a single-document method means.

        With a single tenant in scope nothing is queried. Otherwise the
        tenants that hold *path* (latest, or *version*) are listed:
        one or none → the scope as is; a tenant plus the public one under
        ``include_public`` → that tenant (it shadows the public document);
        anything else is ambiguous → :class:`NavigationError`.
        """
        if scope.tenants is not None and len(scope.tenants) == 1:
            return scope
        clauses: list[str] = ["d.version = $version" if version is not None else "d.latest = true"]
        params: dict[str, Any] = {"path": path}
        if version is not None:
            params["version"] = int(version)
        self._scope_where("d", clauses, params, scope, "tenant_only")
        rows = await self._read(
            f"MATCH (d:Document {{path: $path}}) WHERE {' AND '.join(clauses)} "
            "RETURN d.tenant_id AS tenant_id",
            **params,
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

    @staticmethod
    def _limit_clause(limit: int | None, skip: int = 0) -> str:
        out = ""
        if skip:
            out += f" SKIP {int(skip)}"
        if limit is not None and limit >= 0:
            out += f" LIMIT {int(limit)}"
        return out

    async def _instance_key_field_count(self, model_class: str) -> int | None:
        """Number of ``instance_key`` fields declared for *model_class* (memoised).

        ``None`` when the model has no ``:CatalogModel`` entry at all.
        """
        if model_class in self._key_field_counts:
            return self._key_field_counts[model_class]
        rec = await self._read_one(
            """
            MATCH (cm:CatalogModel {name: $mc})
            OPTIONAL MATCH (cm)-[:HAS_FIELD]->(f:ModelField {is_instance_key: true})
            RETURN count(f) AS n
            """,
            mc=model_class,
        )
        val: int | None = None if rec is None else int(rec.get("n", 0) or 0)
        self._key_field_counts[model_class] = val
        return val

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
