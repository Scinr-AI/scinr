"""
graph/neo4j/extraction.py — Entity extraction over Neo4j.

:class:`Neo4jExtractionOps` (``store.extraction``) reads the extraction targets,
marks the information units as extracted and opens the extraction sink.
:class:`Neo4jExtractionSink` is the Cypher side of the subgraph writers: the
traversal of the Pydantic instance (uids, timestamp, which fields are written)
lives in :mod:`scinr.newton.entity_extraction.graph_mapper` and calls the sink
once per statement, in the order the statements were always emitted.

The sink uses **one session** for a whole write, with autocommit statements, and
keeps the retries that wrapped the same statements before the split.
Semaphores are the callers' business.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from scinr.newton.config import get_config
from scinr.newton.entity_extraction.state import ExtractionTarget
from scinr.newton.graph.base import ExtractionOps
from scinr.newton.graph.neo4j import driver as _driver
from scinr.newton.graph.neo4j._shared import (
    delete_stale_extraction_result,
    provenance_set_clause,
)
from scinr.newton.graph.neo4j.errors import neo4j_errors
from scinr.newton.graph.neo4j.retry import with_neo4j_retry
from scinr.newton.graph.types import Provenance, RawProvenance
from scinr.newton.utils.tenancy import tenant_key

if TYPE_CHECKING:
    from scinr.newton.graph.sinks import ExtractionSink

log = logging.getLogger(__name__)

# Interpolated into the Cypher: a closed set, never caller-controlled text.
_PARENT_LABELS = frozenset({"ExtractionResult", "ModelInstance"})


def _check_label(label: str) -> None:
    """Reject any parent label other than the two a model subgraph can hang from."""
    if label not in _PARENT_LABELS:
        raise ValueError(f"parent label must be one of {sorted(_PARENT_LABELS)}, got {label!r}")


def _prov_params(prov: Provenance) -> dict[str, Any]:
    """The three provenance parameters every merge/create statement binds."""
    return dict(
        tenant_id=prov.tenant_id,
        created_by_user_id=prov.created_by_user_id,
        job_id=prov.job_id,
    )


class Neo4jExtractionSink:
    """:class:`~scinr.newton.graph.sinks.ExtractionSink` over one async Neo4j session.

    Each method is one autocommit statement with the Cypher the extraction
    writers always ran; the ones that were retried on transient errors still
    are. Built by :meth:`Neo4jExtractionOps.sink`.
    """

    def __init__(self, session) -> None:
        self._session = session

    # ── Shared ────────────────────────────────────────────────────────────

    @neo4j_errors()
    async def read_node_provenance(self, node_full_id: str) -> RawProvenance | None:
        result = await self._session.run(
            "MATCH (n:StructureNode {id: $nid}) "
            "RETURN count(n) AS cnt, n.tenant_id AS tenant_id, "
            "n.created_by_user_id AS created_by_user_id, n.job_id AS job_id",
            nid=node_full_id,
        )
        rec = await result.single()
        cnt = rec["cnt"] if rec else 0
        if cnt == 0:
            return None
        return RawProvenance(
            tenant_id=rec["tenant_id"],
            created_by_user_id=rec["created_by_user_id"],
            job_id=rec["job_id"],
        )

    # ── write_extraction_subgraph ─────────────────────────────────────────

    @neo4j_errors()
    async def delete_stale_extraction_result(self, node_full_id: str) -> None:
        await delete_stale_extraction_result(self._session, node_full_id)

    @neo4j_errors()
    async def create_extraction_result(
        self,
        node_full_id: str,
        uid: str,
        document_name: str,
        model_class: str,
        timestamp: str,
        prov: Provenance,
    ) -> None:
        await self._session.run(
            """
            MATCH (n:StructureNode {id: $nid})
            CREATE (er:ExtractionResult {
                uid:           $uid,
                node_full_id:  $nid,
                document_name: $doc_name,
                model_class:   $model_class,
                timestamp:     $timestamp,
                tenant_id:            $tenant_id,
                created_by_user_id:   $created_by_user_id,
                job_id:               $job_id
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=node_full_id,
            uid=uid,
            doc_name=document_name,
            model_class=model_class,
            timestamp=timestamp,
            **_prov_params(prov),
        )

    @neo4j_errors()
    async def link_catalog_model(self, er_uid: str, name: str, *, primary: bool) -> None:
        params = dict(uid=er_uid, name=name)
        if primary:
            query = """
            MATCH (er:ExtractionResult {uid: $uid})
            MERGE (cm:CatalogModel {name: $name})
            MERGE (er)-[:USES_PRIMARY_MODEL]->(cm)
            """
        else:
            query = """
                MATCH (er:ExtractionResult {uid: $uid})
                MERGE (cm:CatalogModel {name: $name})
                MERGE (er)-[:USES_COMPLEMENTARY_MODEL]->(cm)
                """
        await with_neo4j_retry(lambda: self._session.run(query, **params))

    @neo4j_errors()
    async def merge_keyed_instance(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        model_class: str,
        key_fields: dict[str, str],
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        _check_label(parent_label)
        key_props_set = ", ".join(f"child.`{k}` = ${k}" for k in key_fields)
        prov_set = provenance_set_clause("child")
        if list_index is None:
            rel = f"[:`{rel_name}`]"
            extra: dict[str, Any] = {}
        else:
            rel = f"[:`{rel_name}` {{index: $idx}}]"
            extra = {"idx": list_index}
        await self._session.run(
            f"""
                    MATCH (parent:{parent_label} {{uid: $parent_uid}})
                    MERGE (child:ModelInstance {{uid: $child_uid}})
                    ON CREATE SET child.model_class = $model_class, {key_props_set}, {prov_set}
                    ON MATCH  SET {key_props_set}, {prov_set}
                    MERGE (parent)-{rel}->(child)
                    """,
            parent_uid=parent_uid,
            child_uid=child_uid,
            model_class=model_class,
            **extra,
            **_prov_params(prov),
            **key_fields,
        )

    @neo4j_errors()
    async def create_instance(
        self,
        parent_label: str,
        parent_uid: str,
        rel_name: str,
        child_uid: str,
        model_class: str,
        prov: Provenance,
        list_index: int | None,
    ) -> None:
        _check_label(parent_label)
        if list_index is None:
            rel = f"[:`{rel_name}`]"
            extra: dict[str, Any] = {}
        else:
            rel = f"[:`{rel_name}` {{index: $idx}}]"
            extra = {"idx": list_index}
        await self._session.run(
            f"""
                    MATCH (parent:{parent_label} {{uid: $parent_uid}})
                    CREATE (child:ModelInstance {{
                        uid:         $child_uid,
                        model_class: $model_class,
                        tenant_id:   $tenant_id,
                        created_by_user_ids: CASE WHEN $created_by_user_id IS NULL THEN [] ELSE [$created_by_user_id] END,
                        job_ids:             CASE WHEN $job_id IS NULL THEN [] ELSE [$job_id] END
                    }})
                    CREATE (parent)-{rel}->(child)
                    """,
            parent_uid=parent_uid,
            child_uid=child_uid,
            model_class=model_class,
            **extra,
            **_prov_params(prov),
        )

    @neo4j_errors()
    async def merge_labeled_entity(
        self, uid: str, label: str, value: str, normalized_value: str, prov: Provenance
    ) -> None:
        prov_set = provenance_set_clause("le")
        params = dict(
            label=label,
            normalized_value=normalized_value,
            uid=uid,
            value=value,
            **_prov_params(prov),
        )
        await with_neo4j_retry(
            lambda: self._session.run(
                f"""
        MERGE (le:LabeledEntity {{uid: $uid}})
        ON CREATE SET le.label = $label, le.normalized_value = $normalized_value,
                      le.value = $value, {prov_set}
        ON MATCH  SET {prov_set}
        """,
                **params,
            )
        )

    @neo4j_errors()
    async def link_reference(
        self,
        parent_label: str,
        parent_uid: str,
        le_uid: str,
        field_name: str,
        list_index: int | None,
    ) -> None:
        _check_label(parent_label)
        if list_index is None:
            await self._session.run(
                f"""
                MATCH (parent:{parent_label} {{uid: $parent_uid}})
                MATCH (le:LabeledEntity {{uid: $le_uid}})
                MERGE (parent)-[:REFERENCES {{field_name: $field_name}}]->(le)
                """,
                parent_uid=parent_uid,
                le_uid=le_uid,
                field_name=field_name,
            )
        else:
            await self._session.run(
                f"""
                            MATCH (parent:{parent_label} {{uid: $parent_uid}})
                            MATCH (le:LabeledEntity {{uid: $le_uid}})
                            MERGE (parent)-[:REFERENCES {{field_name: $field_name, list_index: $idx}}]->(le)
                            """,
                parent_uid=parent_uid,
                le_uid=le_uid,
                field_name=field_name,
                idx=list_index,
            )

    @neo4j_errors()
    async def merge_instance_shell(
        self, tgt_uid: str, model_class: str, key_fields: dict[str, str], prov: Provenance
    ) -> None:
        key_set = ", ".join(f"tgt.`{k}` = ${k}" for k in key_fields)
        prov_set = provenance_set_clause("tgt")
        params: dict[str, Any] = {
            "tgt_uid": tgt_uid,
            "model_class": model_class,
            **_prov_params(prov),
            **key_fields,
        }
        await with_neo4j_retry(
            lambda: self._session.run(
                f"""
                    MERGE (tgt:ModelInstance {{uid: $tgt_uid}})
                    ON CREATE SET tgt.model_class = $model_class, {key_set}, {prov_set}
                    ON MATCH  SET {key_set}, {prov_set}
                    """,
                **params,
            )
        )

    @neo4j_errors()
    async def link_instance_relationship(
        self, src_label: str, src_uid: str, tgt_uid: str, rel_type: str
    ) -> None:
        _check_label(src_label)
        params = {"src_uid": src_uid, "tgt_uid": tgt_uid}
        await with_neo4j_retry(
            lambda: self._session.run(
                f"""
                    MATCH (src:{src_label} {{uid: $src_uid}})
                    MATCH (tgt:ModelInstance {{uid: $tgt_uid}})
                    MERGE (src)-[:`{rel_type}`]->(tgt)
                    """,
                **params,
            )
        )

    @neo4j_errors()
    async def set_instance_properties(
        self, parent_label: str, parent_uid: str, props: dict[str, Any]
    ) -> None:
        _check_label(parent_label)
        set_clause = ", ".join(f"parent.`{k}` = ${k}" for k in props)
        await self._session.run(
            f"MATCH (parent:{parent_label} {{uid: $parent_uid}}) SET {set_clause}",
            parent_uid=parent_uid,
            **props,
        )

    @neo4j_errors()
    async def link_field_relationship(self, src_uid: str, tgt_uid: str, rel_type: str) -> None:
        await self._session.run(
            f"""
                            MATCH (src:LabeledEntity {{uid: $src_uid}})
                            MATCH (tgt:LabeledEntity {{uid: $tgt_uid}})
                            MERGE (src)-[:`{rel_type}`]->(tgt)
                            """,
            src_uid=src_uid,
            tgt_uid=tgt_uid,
        )

    # ── write_triple_subgraph ─────────────────────────────────────────────

    @neo4j_errors()
    async def delete_triple_extraction_result(self, node_full_id: str) -> None:
        # DETACH DELETE removes HAS_EXTRACTION and HAS_ENTITY relationships
        # but leaves :Entity nodes intact (they are deduplicated within a tenant).
        await self._session.run(
            """
            MATCH (n:StructureNode {id: $nid})-[:HAS_EXTRACTION]->(er:ExtractionResult)
            DETACH DELETE er
            """,
            nid=node_full_id,
        )

    @neo4j_errors()
    async def create_triple_extraction_result(
        self, node_full_id: str, uid: str, document_name: str, timestamp: str, prov: Provenance
    ) -> None:
        await self._session.run(
            """
            MATCH (n:StructureNode {id: $nid})
            CREATE (er:ExtractionResult {
                uid:           $uid,
                node_full_id:  $nid,
                document_name: $doc_name,
                model_class:   'Triple',
                timestamp:     $timestamp,
                tenant_id:            $tenant_id,
                created_by_user_id:   $created_by_user_id,
                job_id:               $job_id
            })
            CREATE (n)-[:HAS_EXTRACTION]->(er)
            """,
            nid=node_full_id,
            uid=uid,
            doc_name=document_name,
            timestamp=timestamp,
            **_prov_params(prov),
        )

    @neo4j_errors()
    async def merge_entity(
        self, uid: str, value: str, normalized_value: str, prov: Provenance
    ) -> None:
        entity_prov_set = provenance_set_clause("e")
        params = dict(
            normalized_value=normalized_value,
            uid=uid,
            value=value,
            **_prov_params(prov),
        )
        await with_neo4j_retry(
            lambda: self._session.run(
                f"""
                MERGE (e:Entity {{uid: $uid}})
                ON CREATE SET e.normalized_value = $normalized_value,
                              e.value = $value, {entity_prov_set}
                ON MATCH  SET {entity_prov_set}
                """,
                **params,
            )
        )

    @neo4j_errors()
    async def link_triple_entity(
        self, er_uid: str, entity_uid: str, role: Literal["subject", "object"]
    ) -> None:
        if role == "subject":
            await self._session.run(
                """
                MATCH (er:ExtractionResult {uid: $er_uid})
                MATCH (subj:Entity {uid: $subj_uid})
                MERGE (er)-[:HAS_ENTITY {role: 'subject'}]->(subj)
                """,
                er_uid=er_uid,
                subj_uid=entity_uid,
            )
        elif role == "object":
            await self._session.run(
                """
                MATCH (er:ExtractionResult {uid: $er_uid})
                MATCH (obj:Entity {uid: $obj_uid})
                MERGE (er)-[:HAS_ENTITY {role: 'object'}]->(obj)
                """,
                er_uid=er_uid,
                obj_uid=entity_uid,
            )
        else:
            raise ValueError(f"role must be 'subject' or 'object', got {role!r}")

    @neo4j_errors()
    async def merge_predicate(
        self, subj_uid: str, obj_uid: str, rel_type: str, predicate_raw: str
    ) -> None:
        await self._session.run(
            f"""
                MATCH (subj:Entity {{uid: $subj_uid}})
                MATCH (obj:Entity {{uid: $obj_uid}})
                MERGE (subj)-[r:`{rel_type}`]->(obj)
                ON CREATE SET r.predicate_raw = $predicate_raw
                """,
            subj_uid=subj_uid,
            obj_uid=obj_uid,
            predicate_raw=predicate_raw,
        )


@asynccontextmanager
async def _open_extraction_sink() -> AsyncIterator[Neo4jExtractionSink]:
    """One session on the shared async driver, closed when the block ends.

    Driver errors raised while acquiring or closing the session are translated.
    """
    cfg = get_config()
    with neo4j_errors():
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            yield Neo4jExtractionSink(session)


class Neo4jExtractionOps(ExtractionOps):
    """Extraction operations over Neo4j (asynchronous, shared async driver)."""

    def sink(self) -> AbstractAsyncContextManager[ExtractionSink]:
        """Open the extraction sink: one session on the shared async driver, autocommit statements."""
        return _open_extraction_sink()

    @neo4j_errors()
    async def fetch_targets(
        self,
        *,
        tenant_id: str | None,
        doc_path: str,
        only_unextracted: bool = False,
    ) -> list[ExtractionTarget]:
        """
        Fetch all StructureNodes that:
          - belong to *tenant_id*'s (``None`` = public) latest document at
            *doc_path* (never selected by name, which is not unique)
          - have a ModelDecision (matched or unmatched)
          - have at least one InfoUnit

        When *only_unextracted* is True, nodes that already have a
        :HAS_EXTRACTION->(:ExtractionResult) relationship are excluded.

        Nodes with matched_model_class IS NULL will have model_class=None in the
        returned target, signalling the pipeline to use the default Triple extraction.

        Returns targets ordered by StructureNode.appearance_order with their
        InfoUnits ordered by iu.order (ascending). Re-running always re-fetches
        all nodes regardless of prior extraction status.
        """
        extra_filter = "AND NOT EXISTS { MATCH (n)-[:HAS_EXTRACTION]->(:ExtractionResult {source: 'tabular_raw'}) }"
        if only_unextracted:
            extra_filter += "\nAND NOT EXISTS { MATCH (n)-[:HAS_EXTRACTION]->(:ExtractionResult) }"

        query = f"""
        MATCH (d:Document {{tenant_id: $tenant_id, path: $doc_path, latest: true}})
              -[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
        MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
        WHERE EXISTS {{
          MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
        }}
        {extra_filter}
        OPTIONAL MATCH (md)-[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
        OPTIONAL MATCH (md)-[:HAS_SUPPLEMENTARY_FIELD]->(sf:SupplementaryField)
        WITH n, md,
             collect(DISTINCT CASE WHEN cm IS NOT NULL
                     THEN {{model_class: cm.model_class, coverage_note: cm.coverage_note}}
                     ELSE null END) AS complementary_raw,
             collect(DISTINCT CASE WHEN sf IS NOT NULL
                     THEN {{field_name: sf.field_name, field_type: sf.field_type,
                           description: sf.description, required: sf.required}}
                     ELSE null END) AS supplementary_raw
        MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
        WITH n, md,
             [c IN complementary_raw WHERE c IS NOT NULL] AS complementary,
             [s IN supplementary_raw WHERE s IS NOT NULL] AS supplementary,
             collect(iu {{.uid, .title, .description, .order}}) AS info_units_unsorted
        WITH n, md, complementary, supplementary,
             apoc.coll.sortMaps(info_units_unsorted, '^order') AS info_units_sorted
        RETURN n.id          AS node_full_id,
               n.node_id     AS node_id,
               n.title       AS node_title,
               md.matched_model_class AS model_class,
               complementary,
               supplementary,
               info_units_sorted AS info_units
        ORDER BY n.appearance_order
        """
        # Fallback query without apoc if needed
        fallback_query = f"""
        MATCH (d:Document {{tenant_id: $tenant_id, path: $doc_path, latest: true}})
              -[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode)
        MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
        WHERE EXISTS {{
          MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
        }}
        {extra_filter}
        OPTIONAL MATCH (md)-[:HAS_COMPLEMENTARY_MATCH]->(cm:ComplementaryMatch)
        OPTIONAL MATCH (md)-[:HAS_SUPPLEMENTARY_FIELD]->(sf:SupplementaryField)
        WITH n, md,
             collect(DISTINCT CASE WHEN cm IS NOT NULL
                     THEN {{model_class: cm.model_class, coverage_note: cm.coverage_note}}
                     ELSE null END) AS complementary_raw,
             collect(DISTINCT CASE WHEN sf IS NOT NULL
                     THEN {{field_name: sf.field_name, field_type: sf.field_type,
                           description: sf.description, required: sf.required}}
                     ELSE null END) AS supplementary_raw
        MATCH (n)-[:HAS_INFO_UNIT]->(iu:InfoUnit)
        WITH n, md,
             [c IN complementary_raw WHERE c IS NOT NULL] AS complementary,
             [s IN supplementary_raw WHERE s IS NOT NULL] AS supplementary,
             collect(iu {{.uid, .title, .description, .order}}) AS info_units
        RETURN n.id          AS node_full_id,
               n.node_id     AS node_id,
               n.title       AS node_title,
               md.matched_model_class AS model_class,
               complementary,
               supplementary,
               info_units
        ORDER BY n.appearance_order
        """
        params = {"tenant_id": tenant_key(tenant_id), "doc_path": doc_path}
        cfg = get_config()
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            try:
                result = await session.run(query, **params)
                rows = await result.data()
            except Exception:
                log.warning(
                    "fetch_targets: apoc.coll.sortMaps unavailable, "
                    "falling back to unsorted query"
                )
                result = await session.run(fallback_query, **params)
                rows = await result.data()

            targets: list[ExtractionTarget] = []
            for row in rows:
                # Sort info_units by order in Python as a reliable fallback
                ius = list(row.get("info_units") or [])
                ius.sort(key=lambda x: (x.get("order") or 0))
                targets.append(ExtractionTarget(
                    node_full_id=row["node_full_id"],
                    node_id=row.get("node_id"),
                    node_title=row.get("node_title"),
                    model_class=row["model_class"],
                    complementary_models=list(row.get("complementary") or []),
                    supplementary_fields=list(row.get("supplementary") or []),
                    info_units=ius,
                ))

        log.info(
            "fetch_targets: found %d targets for document %r (tenant=%r)",
            len(targets),
            doc_path,
            tenant_id,
        )
        return targets

    @neo4j_errors()
    async def mark_info_units_extracted(self, node_full_id: str) -> None:
        """Set ``extracted`` timestamp on all unextracted InfoUnits of the given StructureNode.

        Idempotent: only updates
        InfoUnits where ``extracted IS NULL``.
        """
        timestamp = datetime.now(UTC).isoformat()
        cfg = get_config()
        driver = _driver.get_async_driver()
        async with driver.session(database=cfg.neo4j_database) as session:
            result = await session.run(
                """
                MATCH (n:StructureNode {id: $nid})-[:HAS_INFO_UNIT]->(iu:InfoUnit)
                WHERE iu.extracted IS NULL
                SET iu.extracted = $timestamp
                RETURN count(iu) AS updated
                """,
                nid=node_full_id,
                timestamp=timestamp,
            )
            record = await result.single()
            updated = record["updated"] if record else 0
        log.info(
            "mark_info_units_extracted: marked %d InfoUnits as extracted for node %r",
            updated, node_full_id,
        )
