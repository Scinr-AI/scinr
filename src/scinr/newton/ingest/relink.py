"""
ingest/relink.py — Repair ``instance_relationships`` written before
same-extraction sibling resolution existed.

Before it, every ``instance_relationships`` entry was resolved by key alone: a
shell ModelInstance addressed by ``make_instance_uid(target, join keys,
tenant)`` was MERGE'd and the relationship pointed at it. When the target was
extracted in the same row (same :ExtractionResult) but could not be reached
by that key — a key-less target (random uid), or a join value built
differently on each side — the shell never filled and the link was dead.

:func:`relink_same_row_instances` finds, for one tenant, every relationship
of a declared ``instance_relationships`` type whose target is such a shell,
and applies the rule the writer now applies
(:func:`~scinr.newton.entity_extraction.instance_links.match_sibling`): if a
same-extraction sibling of the target class matches, the relationship is
moved to it (MERGE the new one, delete the old one, in the same transaction).
Shells left with no relationship at all are then deleted.

What it reads from the graph only (no Python models are needed, so it works
for any deployment):

- the declared relationships: the ``(:CatalogModel)-[:REL {via_field,
  join_via}]->(:CatalogModel)`` edges ``ensure_catalog_models()`` writes;
- the ``instance_key`` fields: ``(:ModelField {is_instance_key: true})``.

Definitions used by the queries:

- **composition** — the path from an :ExtractionResult down to the
  ModelInstances it was written with, over relationships whose type is not a
  declared ``instance_relationships`` type (``HAS_<FIELD>``), at most
  :data:`COMPOSITION_MAX_HOPS` deep (the writer's recursion cap);
- **shell** — a ModelInstance with no incoming composition relationship: it
  was only ever created as the target of a reference, never written as a
  real instance;
- **siblings** of a source — the ModelInstances reached by composition from
  the :ExtractionResult(s) that reach the source.

Idempotent: a relationship moved to its sibling no longer points at a shell,
so a second run finds nothing to do. Tenant-scoped: every node read or
written carries ``tenant_id = $tenant_id``. Bounded: the moves go
:data:`~scinr.newton.ingest._cascade.DELETE_BATCH_ROWS` per write
transaction, the shell deletion runs ``CALL { ... } IN TRANSACTIONS``.
Relationships whose source is an :ExtractionResult (a root-level
declaration) are not considered.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scinr.newton.entity_extraction.instance_links import SiblingCandidate, match_sibling
from scinr.newton.ingest._cascade import DELETE_BATCH_ROWS
from scinr.newton.results import RelinkResult
from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync
from scinr.newton.utils.tenancy import tenant_key

logger = logging.getLogger(__name__)

COMPOSITION_MAX_HOPS = 10
"""Depth bound of a composition path (the writer stops recursing at depth 10)."""

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

DECLARATIONS_QUERY = """
MATCH (s:CatalogModel)-[r]->(t:CatalogModel)
WHERE r.via_field IS NOT NULL AND r.join_via IS NOT NULL
RETURN DISTINCT s.name AS src_class, type(r) AS rel_type, t.name AS tgt_class,
       r.via_field AS via_field, r.join_via AS join_via
"""

INSTANCE_KEYS_QUERY = """
MATCH (mf:ModelField)
WHERE mf.is_instance_key = true
RETURN mf.model AS model, collect(mf.name) AS fields
"""

_NOT_INSTANCE_REL = "none(x IN {rels} WHERE type(x) IN $irel_types)"


def _shell_edges_query(rel_type: str) -> str:
    """Relationships ``(src {src_class})-[:rel_type]->(shell {tgt_class})`` of
    the tenant, each with the source's :ExtractionResult and the siblings of
    the target class it was written with (compact: uid, join values, keyed)."""
    return f"""
MATCH (src:ModelInstance)-[r:`{rel_type}`]->(tgt:ModelInstance)
WHERE src.tenant_id = $tenant_id AND src.model_class = $src_class
  AND tgt.tenant_id = $tenant_id AND tgt.model_class = $tgt_class
  AND NOT EXISTS {{ MATCH (tgt)<-[x]-() WHERE NOT type(x) IN $irel_types }}
MATCH (er:ExtractionResult)-[rels*1..{COMPOSITION_MAX_HOPS}]->(src)
WHERE er.tenant_id = $tenant_id AND {_NOT_INSTANCE_REL.format(rels="rels")}
WITH DISTINCT src, r, tgt, er
RETURN elementId(r) AS rel_id,
       src.uid AS src_uid,
       src[$via_field] AS via_value,
       elementId(tgt) AS shell_id,
       COUNT {{ (tgt)--() }} AS shell_degree,
       [f IN $remote_fields | tgt[f]] AS shell_values,
       er.uid AS er_uid,
       COLLECT {{
         MATCH (er)-[rels2*1..{COMPOSITION_MAX_HOPS}]->(sib:ModelInstance)
         WHERE sib.model_class = $tgt_class AND sib.tenant_id = $tenant_id
           AND {_NOT_INSTANCE_REL.format(rels="rels2")}
         WITH DISTINCT sib
         RETURN {{uid: sib.uid,
                 values: [f IN $remote_fields | sib[f]],
                 keyed: any(k IN $key_fields WHERE sib[k] IS NOT NULL)}}
       }} AS siblings
"""


def _move_query(rel_type: str) -> str:
    """Move a batch of relationships to their sibling: MERGE the new one (with
    ``resolution = 'sibling'``), delete the old one. Every match is tenant-checked."""
    return f"""
UNWIND $rows AS row
MATCH ()-[r_old]->() WHERE elementId(r_old) = row.rel_id
MATCH (src:ModelInstance {{uid: row.src_uid}}) WHERE src.tenant_id = $tenant_id
MATCH (sib:ModelInstance {{uid: row.sib_uid}}) WHERE sib.tenant_id = $tenant_id
MERGE (src)-[r_new:`{rel_type}`]->(sib)
ON CREATE SET r_new.resolution = 'sibling'
DELETE r_old
RETURN count(*) AS moved
"""


# Shells of the target classes left with no relationship at all. A real
# instance always has its composition parent, so it can never match.
DELETE_ISOLATED_SHELLS_QUERY = f"""
MATCH (s:ModelInstance)
WHERE s.tenant_id = $tenant_id AND s.model_class IN $classes AND NOT EXISTS {{ (s)--() }}
CALL (s) {{
  DELETE s
}} IN TRANSACTIONS OF {DELETE_BATCH_ROWS} ROWS
RETURN count(*) AS deleted
"""

COUNT_ISOLATED_SHELLS_QUERY = """
MATCH (s:ModelInstance)
WHERE s.tenant_id = $tenant_id AND s.model_class IN $classes AND NOT EXISTS { (s)--() }
RETURN count(s) AS n
"""


# ---------------------------------------------------------------------------
# Planning (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Declaration:
    """One ``instance_relationships`` declaration, as stored on the catalog."""

    src_class: str
    rel_type: str
    tgt_class: str
    via_field: str
    join_via: Mapping[str, str]

    @property
    def remote_fields(self) -> list[str]:
        return list(self.join_via.values())


@dataclass(frozen=True)
class Move:
    """One relationship to move from a shell to a sibling."""

    rel_id: str
    src_uid: str
    sib_uid: str
    shell_id: str
    outcome: str


def parse_declarations(records: Iterable[Mapping[str, Any]]) -> list[Declaration]:
    """Build :class:`Declaration` objects from :data:`DECLARATIONS_QUERY` rows."""
    out: list[Declaration] = []
    for rec in records:
        join_via = rec["join_via"]
        if isinstance(join_via, str):
            join_via = json.loads(join_via)
        if not join_via:
            continue
        out.append(
            Declaration(
                src_class=rec["src_class"],
                rel_type=rec["rel_type"],
                tgt_class=rec["tgt_class"],
                via_field=rec["via_field"],
                join_via=dict(join_via),
            )
        )
    return out


def plan_moves(
    declaration: Declaration, rows: Iterable[Mapping[str, Any]]
) -> tuple[list[Move], Counter]:
    """Decide, for each shell relationship of *declaration*, whether a sibling
    takes it over.

    *rows* are :func:`_shell_edges_query` records (one per relationship and
    :ExtractionResult of its source). A relationship whose source hangs from
    several :ExtractionResults is moved only if they all agree on the sibling.

    Returns the moves and a counter of outcomes per relationship:
    ``relinked``, ``ambiguous`` (several equally good siblings),
    ``unmatched`` (no sibling matches: kept, a cross-document reference).
    """
    remote = declaration.remote_fields
    by_rel: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rel[row["rel_id"]].append(row)

    moves: list[Move] = []
    counts: Counter = Counter()
    for rel_id, rel_rows in by_rel.items():
        first = rel_rows[0]
        shell_key = {
            field: str(value)
            for field, value in zip(remote, first["shell_values"], strict=True)
            if value is not None
        }
        via_value = first["via_value"]
        single_target = not (isinstance(via_value, list) and len(via_value) > 1)

        picked: set[str] = set()
        outcomes: list[str] = []
        for row in rel_rows:
            candidates = [
                SiblingCandidate(
                    uid=sib["uid"],
                    model_class=declaration.tgt_class,
                    values=dict(zip(remote, sib["values"], strict=True)),
                    keyed=bool(sib["keyed"]),
                )
                for sib in row["siblings"]
            ]
            match = match_sibling(
                declaration.tgt_class,
                shell_key,
                candidates,
                exclude_uid=row["src_uid"],
                single_target=single_target,
            )
            outcomes.append(match.outcome)
            if match.linked:
                picked.add(match.uid)  # type: ignore[arg-type]

        if "ambiguous" in outcomes or len(picked) > 1:
            counts["ambiguous"] += 1
        elif len(picked) == 1 and all(o != "none" for o in outcomes):
            counts["relinked"] += 1
            moves.append(
                Move(
                    rel_id=rel_id,
                    src_uid=first["src_uid"],
                    sib_uid=next(iter(picked)),
                    shell_id=first["shell_id"],
                    outcome=outcomes[0],
                )
            )
        else:
            counts["unmatched"] += 1
    return moves, counts


# ---------------------------------------------------------------------------
# Execution (sync driver, like the GC passes)
# ---------------------------------------------------------------------------


def _read(driver, database: str, query: str, **params) -> list[dict[str, Any]]:
    def _do() -> list[dict[str, Any]]:
        with driver.session(database=database) as session:
            return [record.data() for record in session.run(query, **params)]

    return with_neo4j_retry_sync(_do)


def _irel_types(declarations: Sequence[Declaration]) -> list[str]:
    return sorted({d.rel_type for d in declarations})


def run_relink(
    driver, *, tenant_id: str, database: str, dry_run: bool = False
) -> dict[str, Any]:
    """Synchronous body of :func:`relink_same_row_instances` (stored tenant
    key, open sync driver). Returns the :class:`RelinkResult` fields."""
    declarations = parse_declarations(_read(driver, database, DECLARATIONS_QUERY))
    key_fields: dict[str, list[str]] = {
        rec["model"]: list(rec["fields"]) for rec in _read(driver, database, INSTANCE_KEYS_QUERY)
    }
    irel_types = _irel_types(declarations)

    totals: Counter = Counter()
    by_relationship: dict[str, dict[str, int]] = {}
    moved_per_shell: Counter = Counter()
    shell_degree: dict[str, int] = {}
    for decl in declarations:
        if not _is_valid_rel_type(decl.rel_type):
            logger.warning("relink: skipping invalid relationship type %r", decl.rel_type)
            continue
        rows = _read(
            driver,
            database,
            _shell_edges_query(decl.rel_type),
            tenant_id=tenant_id,
            src_class=decl.src_class,
            tgt_class=decl.tgt_class,
            via_field=decl.via_field,
            remote_fields=decl.remote_fields,
            key_fields=key_fields.get(decl.tgt_class, []),
            irel_types=irel_types,
        )
        moves, counts = plan_moves(decl, rows)
        for row in rows:
            shell_degree[row["shell_id"]] = row["shell_degree"]
        moved_per_shell.update(m.shell_id for m in moves)
        counts["shell_relationships"] = sum(
            counts[k] for k in ("relinked", "ambiguous", "unmatched")
        )
        label = f"{decl.src_class}-[{decl.rel_type}]->{decl.tgt_class}"
        by_relationship[label] = dict(counts)
        totals.update(counts)
        logger.info("relink: tenant %r %s: %s", tenant_id, label, dict(counts))

        if dry_run or not moves:
            continue
        query = _move_query(decl.rel_type)
        for start in range(0, len(moves), DELETE_BATCH_ROWS):
            batch = [
                {"rel_id": m.rel_id, "src_uid": m.src_uid, "sib_uid": m.sib_uid}
                for m in moves[start : start + DELETE_BATCH_ROWS]
            ]

            def _do_move(batch: list[dict[str, str]] = batch, query: str = query) -> int:
                with driver.session(database=database) as session:
                    return session.execute_write(
                        lambda tx: tx.run(query, rows=batch, tenant_id=tenant_id).single()[
                            "moved"
                        ]
                    )

            with_neo4j_retry_sync(_do_move)

    classes = sorted({d.tgt_class for d in declarations})
    if dry_run:
        # Already isolated + isolated once every planned move is applied.
        shells = _read(
            driver, database, COUNT_ISOLATED_SHELLS_QUERY, tenant_id=tenant_id, classes=classes
        )
        shells_deleted = (shells[0]["n"] if shells else 0) + sum(
            1 for shell, n in moved_per_shell.items() if n >= shell_degree.get(shell, 0)
        )
    else:

        def _do_delete() -> int:
            with driver.session(database=database) as session:
                return session.run(
                    DELETE_ISOLATED_SHELLS_QUERY, tenant_id=tenant_id, classes=classes
                ).single()["deleted"]

        shells_deleted = with_neo4j_retry_sync(_do_delete)

    return {
        "dry_run": dry_run,
        "declarations": len(declarations),
        "shell_relationships": totals["shell_relationships"],
        "relinked": totals["relinked"],
        "ambiguous": totals["ambiguous"],
        "unmatched": totals["unmatched"],
        "shells_deleted": shells_deleted,
        "by_relationship": by_relationship,
    }


def _is_valid_rel_type(rel_type: str) -> bool:
    # Interpolated into Cypher: same rule as annotation.neo4j_ops._validate_rel_type.
    import re

    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel_type or ""))


async def relink_same_row_instances(
    *, tenant_id: str | None, dry_run: bool = False
) -> RelinkResult:
    """Move ``instance_relationships`` from dead shells to the same-row sibling
    they meant, and delete the shells left isolated — for one tenant.

    Applies to graphs written before same-extraction sibling resolution (see
    the module docstring for the exact rule). Idempotent: run it again and it
    finds nothing left to move. Do not run it while the same tenant is being
    ingested.

    *tenant_id* is mandatory (keyword-only, no default); ``None`` and
    ``"__public__"`` both mean the public documents. With ``dry_run=True``
    nothing is written and the counts say what would be moved. Opens and
    closes its own Neo4j driver.

    Raises
    ------
    TypeError
        If *tenant_id* is not passed.
    ValueError
        If *tenant_id* is empty.
    """
    from scinr.newton.config import get_config
    from scinr.newton.ingest.config import get_driver

    tenant = tenant_key(tenant_id)
    driver = get_driver()
    try:
        counts = await asyncio.to_thread(
            run_relink,
            driver,
            tenant_id=tenant,
            database=get_config().neo4j_database,
            dry_run=dry_run,
        )
    finally:
        driver.close()
    logger.info(
        "relink_same_row_instances: tenant %r%s — relinked=%d ambiguous=%d unmatched=%d "
        "shells_deleted=%d",
        tenant,
        " (dry run)" if dry_run else "",
        counts["relinked"],
        counts["ambiguous"],
        counts["unmatched"],
        counts["shells_deleted"],
    )
    return RelinkResult(tenant_id=tenant_id, **counts)
