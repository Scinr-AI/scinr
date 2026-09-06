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

from scinr.newton.exceptions import GraphConnectionError
from scinr.newton.navigation.base import (
    DEFAULT_MAX_DEPTH,
    INSTANCE_CONTAINMENT_DEPTH,
)
from scinr.newton.navigation.models import DocumentRef
from scinr.newton.navigation.neo4j._safe import resolve_depth

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


def selector_path(document: str | DocumentRef) -> str:
    """Return the ``path`` string of a document selector."""
    if isinstance(document, DocumentRef):
        return document.path
    if isinstance(document, str):
        return document
    raise TypeError(f"document selector must be a path str or DocumentRef, got {type(document).__name__}")


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
            raise GraphConnectionError(f"Neo4j is not reachable: {exc}") from exc

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
                raise GraphConnectionError(str(exc)) from exc
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
        """
        if version is None:
            return f"({alias}:Document {{path: $path, latest: true}})"
        return f"({alias}:Document {{path: $path, version: $version}})"

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
