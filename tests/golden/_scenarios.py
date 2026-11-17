"""
tests/golden/_scenarios.py — The scenarios of the golden harness.

Each scenario drives the public API against real Neo4j and MongoDB with the scripted LLM
(``_llm.py``). Whatever it returns as ``extras`` (result objects, snapshots, caught error
messages) is stored next to the graph dump, so a refactor that changes a result without
changing the graph is also seen.

The extraction stage (the one stage whose LLM output is a whole document) is replaced by
``_fake_extraction``: it builds the document structure that the ingestion then writes.
Everything else is the real code path.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from scinr.newton.models.document_structure import Document, InfoUnit, NodeRole, StructureNode

from ._llm import NO_MODEL, Script, decision, set_script

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

TENANT_A = "golden-a"
TENANT_B = "golden-b"
USER = "golden-user"

_TRIPLES = {
    "triples": [
        {"subject": "Widget", "predicate": "is made by", "object": "Acme Corp"},
        {"subject": "Widget", "predicate": "ships with", "object": "Manual"},
    ]
}
_PRODUCT_A = {
    "product_name": "Widget A",
    "manufacturer": "Acme Corp",
    "category": "tools",
    "tags": ["steel", "small"],
    "supplier_codes": ["S-1", "S-2"],
    "spec": {"material": "steel", "weight": "2 kg"},
    "primary_part": {"part_code": "P-0", "description": "Main part"},
    "parts": [
        {"part_code": "P-1", "description": "Bolt"},
        {"part_code": "P-2", "description": "Nut"},
    ],
}
_PRODUCT_B = {
    "product_name": "Widget B",
    "manufacturer": "Acme Corp",
    "category": "tools",
    "tags": ["steel"],
    "supplier_codes": ["S-1"],
    "parts": [{"part_code": "P-1", "description": "Bolt (revised)"}],
}

PRODUCT_SCRIPT = Script(
    annotations={
        "sec_1": NO_MODEL,
        "prod_a": decision("GoldenProduct"),
        "combo": decision("GoldenProduct", complementary=("GoldenSupplier",), supplementary=True),
        "misc": NO_MODEL,
    },
    extractions={
        "sec_1": _TRIPLES,
        "prod_a": {"golden_product": _PRODUCT_A},
        "combo": {
            "golden_product": _PRODUCT_B,
            "golden_supplier": {"supplier_code": "S-1", "supplier_name": "Supply Co"},
            "warranty_period": "2 years",
        },
        "misc": _TRIPLES,
    },
    tabular={
        "norm.csv": decision("GoldenNormRow"),
        "plain.csv": decision("GoldenRow"),
        "b.csv": decision("GoldenRow"),
        "raw.csv": {**NO_MODEL},
    },
)


def node(
    node_id: str,
    title: str,
    order: int,
    infos: list[tuple[str, str]] | None = None,
    children: list[StructureNode] | None = None,
) -> StructureNode:
    return StructureNode(
        node_id=node_id,
        title=title,
        role=NodeRole.SECTION,
        appearance_order=order,
        info_units=[
            InfoUnit(title=t, order=i, description=d) for i, (t, d) in enumerate(infos or [])
        ],
        children=children or [],
    )


def product_structure(revision: int = 1) -> list[StructureNode]:
    return [
        node(
            "sec_1",
            "1 Products",
            1,
            [("Overview", f"Catalogue of widgets, revision {revision}.")],
            [
                node("prod_a", "1.1 Widget A", 1, [("Widget A", "Steel widget made by Acme.")]),
                node("combo", "1.2 Widget B", 2, [("Widget B", "Widget with supplier S-1.")]),
            ],
        ),
        node("misc", "2 Miscellaneous", 2, [("Notes", "Unrelated remarks.")]),
    ]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class Env:
    tmp: Path

    async def store_raw(
        self,
        name: str,
        folder: str = "",
        tenant: str | None = None,
        created_by: str | None = None,
        job_id: str | None = None,
    ) -> str:
        from scinr.newton.storage.factory import get_storage

        path = self.tmp / "raw" / (folder or ".") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"raw content of {name}", encoding="utf-8")
        raw_repo, _ = get_storage()
        return await raw_repo.store_file(
            path=path,
            filename=name,
            content_type="text/plain",
            folder_path=folder or None,
            tenant_id=tenant,
            created_by_user_id=created_by,
            job_id=job_id,
        )

    async def ingest(
        self,
        name: str,
        *,
        folder: str = "",
        tenant: str | None = None,
        structure: list[StructureNode] | None = None,
        update_mode: bool = False,
        created_by: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Store the raw file and ingest a document; return its ``doc_path``."""
        from scinr.newton.stages import run_ingestion

        raw_id = await self.store_raw(name, folder, tenant, created_by, job_id)
        doc_path = f"{folder}/{name}" if folder else name
        document = Document(
            document_name=name,
            doc_path=doc_path,
            raw_file_id=raw_id,
            document_structure=structure if structure is not None else product_structure(),
        )
        result = await run_ingestion(
            documents=[document],
            update_mode=update_mode,
            tenant_id=tenant,
            created_by_user_id=created_by,
            job_id=job_id,
        )
        assert result.success, [d.errors for d in result.documents]
        return doc_path

    async def annotate(self, name: str, doc_path: str, tenant: str | None, **kwargs: Any) -> dict:
        from scinr.newton.annotation.agent import run_annotation_agent

        out = await run_annotation_agent(name, tenant_id=tenant, doc_path=doc_path, **kwargs)
        assert not out["errors"], out["errors"]
        return out

    async def extract(self, name: str, doc_path: str, tenant: str | None, **kwargs: Any) -> dict:
        from scinr.newton.entity_extraction.agent import run_entity_extraction_agent

        out = await run_entity_extraction_agent(
            name, tenant_id=tenant, doc_path=doc_path, **kwargs
        )
        assert not out["errors"], out["errors"]
        return out

    async def full(self, name: str, **kwargs: Any) -> str:
        """Ingest + annotate + extract a product document; return its ``doc_path``."""
        tenant = kwargs.get("tenant")
        doc_path = await self.ingest(name, **kwargs)
        await self.annotate(name, doc_path, tenant)
        await self.extract(name, doc_path, tenant)
        return doc_path


def as_extras(value: Any) -> Any:
    """Result objects to plain data for the artifacts."""
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def _fake_extraction() -> AsyncMock:
    """Replacement of the LLM-bound extraction stage: a fixed structure per document."""

    async def _extract(doc, output_path, fast_extraction=False, tenant_id=None, **kwargs):
        folder = doc.folder_path
        return Document(
            document_name=doc.document_name,
            doc_path=f"{folder}/{doc.document_name}" if folder else doc.document_name,
            raw_file_id=doc.raw_file_id or "",
            tenant_id=tenant_id if tenant_id is not None else doc.tenant_id,
            created_by_user_id=kwargs.get("created_by_user_id") or doc.created_by_user_id,
            job_id=kwargs.get("job_id") or doc.job_id,
            document_structure=product_structure(),
        )

    return AsyncMock(side_effect=_extract)


def canonical_snapshot(snapshot: dict) -> dict:
    """A snapshot with its lists in a canonical order.

    The order in which Neo4j returns the nodes and relationships of a subtree is not
    specified by the export queries and changes between runs; the content is what matters.
    """

    def _key(item: Any) -> str:
        return json.dumps(item, sort_keys=True, ensure_ascii=False)

    out = dict(snapshot)
    documents = []
    for entry in snapshot["documents"]:
        entry = dict(entry)
        for field_name in ("nodes", "relationships"):
            entry[field_name] = sorted(entry[field_name], key=_key)
        documents.append(entry)
    out["documents"] = sorted(documents, key=_key)
    return out


def _csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def s1_ingest_versions(env: Env) -> dict:
    """New document, re-ingestion in update_mode, new version."""
    extras: dict[str, Any] = {}
    extras["v1"] = await env.ingest("alpha.txt", tenant=TENANT_A, created_by=USER, job_id="job-1")
    await env.ingest("alpha.txt", tenant=TENANT_A, update_mode=True, structure=product_structure(2))
    await env.ingest("alpha.txt", tenant=TENANT_A, structure=product_structure(3))
    await env.ingest("beta.txt", folder="Docs/Sub", tenant=TENANT_A)
    await env.ingest("beta.txt", folder="Docs/Sub", tenant=TENANT_A)
    return extras


async def s2_replace_mixed(env: Env) -> dict:
    """Replacement (``replaces``) of a root document by a folder run, and a mixed folder
    (document + CSV in the same subfolder) run twice."""
    from scinr.newton.pipeline import run_pipeline

    extras: dict[str, Any] = {}
    await env.ingest("old.txt", tenant=TENANT_A)

    incoming = env.tmp / "incoming"
    (incoming / "Sub").mkdir(parents=True)
    (incoming / "new.txt").write_text("new text", encoding="utf-8")
    (incoming / "Sub" / "inner.txt").write_text("inner text", encoding="utf-8")
    mixed = env.tmp / "mixed"
    (mixed / "A").mkdir(parents=True)
    (mixed / "A" / "a.txt").write_text("hello", encoding="utf-8")
    _csv(mixed / "A" / "b.csv", "sku,item,qty\nS1,Bolt,10\nS2,Nut,20\n")

    with patch("scinr.newton.stages.extraction.extract_one_intermediate", _fake_extraction()):
        res = await run_pipeline(
            stages=["preprocess", "extraction", "ingestion"],
            input_raw=str(incoming),
            replaces="old.txt",
            tenant_id=TENANT_A,
            parallel_docs=1,
        )
        assert res.ingestion.success, res.ingestion.documents[0].errors
        for _ in range(2):
            res = await run_pipeline(
                stages=["preprocess", "extraction", "ingestion"],
                input_raw=str(mixed),
                tenant_id=TENANT_A,
                parallel_docs=1,
            )
            assert res.ingestion.success, res.ingestion.documents[0].errors
    return extras


async def s3_annotation(env: Env) -> dict:
    """Automatic annotation (with model, without, with complementary), re-run and manual."""
    from scinr.newton.annotation.agent import run_manual_annotation

    extras: dict[str, Any] = {}
    doc_path = await env.ingest("annot.txt", tenant=TENANT_A)
    out = await env.annotate("annot.txt", doc_path, TENANT_A)
    extras["first"] = {"errors": out["errors"], "nodes": len(out["nodes_to_annotate"])}
    out = await env.annotate("annot.txt", doc_path, TENANT_A, only_unannotated=True)
    extras["only_unannotated"] = {"nodes": len(out["nodes_to_annotate"])}
    out = await env.annotate(
        "annot.txt", doc_path, TENANT_A, context_instructions_override="Focus on suppliers."
    )
    extras["rerun"] = {"nodes": len(out["nodes_to_annotate"])}
    extras["manual"] = await run_manual_annotation(
        "annot.txt", "GoldenSupplier", tenant_id=TENANT_A, doc_path=doc_path
    )
    return extras


async def s4_extraction(env: Env) -> dict:
    """Entity extraction (entities, nested, lists, relationships) and triples; re-run."""
    extras: dict[str, Any] = {}
    doc_path = await env.ingest("extract.txt", tenant=TENANT_A)
    await env.annotate("extract.txt", doc_path, TENANT_A)
    out = await env.extract("extract.txt", doc_path, TENANT_A)
    extras["first"] = {"targets": len(out["targets"])}
    out = await env.extract("extract.txt", doc_path, TENANT_A)
    extras["rerun"] = {"targets": len(out["targets"])}
    second = await env.ingest("extract2.txt", tenant=TENANT_A)
    await env.annotate("extract2.txt", second, TENANT_A)
    await env.extract("extract2.txt", second, TENANT_A, only_unextracted=True)
    return extras


async def s5_tabular(env: Env) -> dict:
    """Tabular ingestion with normalization, without it and with raw rows; then update_mode."""
    from scinr.newton.stages import run_tabular_pipeline

    extras: dict[str, Any] = {}
    root = env.tmp / "tab"
    _csv(root / "norm.csv", "name,city\nAcme,Madrid\nBeta,Paris\nAcme,Madrid\nGamma,Rome\n")
    _csv(root / "plain.csv", "sku,item,qty,x_note\nS1,Bolt,10,a\nS2,Nut,20,b\n")
    _csv(root / "Sub" / "raw.csv", "alpha,beta\n1,2\n3,4\n5,6\n")
    for update_mode in (False, True):
        res = await run_tabular_pipeline(
            input_raw=str(root),
            update_mode=update_mode,
            tenant_id=TENANT_A,
            created_by_user_id=USER,
            job_id="job-tab",
        )
        assert res.success, [d.errors for d in res.documents]
        extras[f"update_{update_mode}"] = [
            (d.document_name, d.nodes_processed, d.nodes_failed) for d in res.documents
        ]
    return extras


async def s6_delete_gc(env: Env) -> dict:
    """Deletion by every kind of filter, and collect_orphans."""
    from scinr.newton.ingest.deletion import collect_orphans, delete_document

    extras: dict[str, Any] = {}
    await env.full("del_path.txt", tenant=TENANT_A, created_by=USER, job_id="job-1")
    await env.full("del_version.txt", tenant=TENANT_A, created_by=USER, job_id="job-2")
    await env.full("del_version.txt", tenant=TENANT_A, created_by=USER, job_id="job-2")
    await env.full("del_job.txt", folder="Jobs", tenant=TENANT_A, created_by=USER, job_id="job-3")
    await env.full("del_job2.txt", folder="Jobs", tenant=TENANT_A, created_by=USER, job_id="job-3")
    await env.full("del_user.txt", tenant=TENANT_A, created_by="other-user", job_id="job-4")
    await env.full("keep.txt", tenant=TENANT_A, created_by=USER, job_id="job-5")
    # Re-ingestion in update_mode leaves extraction results without collecting orphans.
    doc = await env.full("orphans.txt", tenant=TENANT_A)
    await env.ingest("orphans.txt", tenant=TENANT_A, update_mode=True)

    extras["by_path"] = as_extras(await delete_document("del_path.txt", tenant_id=TENANT_A))
    extras["by_version"] = as_extras(
        await delete_document("del_version.txt", version=1, tenant_id=TENANT_A)
    )
    extras["by_job"] = as_extras(await delete_document(job_id="job-3", tenant_id=TENANT_A))
    extras["by_path_and_user"] = as_extras(
        await delete_document("del_user.txt", tenant_id=TENANT_A, created_by_user_id="other-user")
    )
    extras["missing"] = as_extras(await delete_document("nope.txt", tenant_id=TENANT_A))
    extras["orphans_before"] = doc
    extras["collect"] = as_extras(await collect_orphans(tenant_id=TENANT_A))
    return extras


async def s7_freeze_restore(env: Env) -> dict:
    """Freeze, backup, export (dict and file), restore (stubs and snapshot)."""
    from scinr.newton.ingest.deletion import delete_document
    from scinr.newton.ingest.freeze import export_document_snapshot, freeze_document
    from scinr.newton.ingest.restore import restore_document

    extras: dict[str, Any] = {}
    await env.full("frz_a.txt", folder="Frz", tenant=TENANT_A, created_by=USER, job_id="job-f")
    await env.full("frz_b.txt", folder="Frz", tenant=TENANT_A, created_by=USER, job_id="job-f")
    await env.full("frz_c.txt", tenant=TENANT_A)
    await env.full("frz_d.txt", tenant=TENANT_A)

    extras["export_dict"] = canonical_snapshot(
        await export_document_snapshot("frz_c.txt", tenant_id=TENANT_A)
    )
    out_file = env.tmp / "export.json"
    await export_document_snapshot(
        "frz_c.txt", tenant_id=TENANT_A, destination="file", file_path=out_file
    )
    extras["export_file"] = canonical_snapshot(json.loads(out_file.read_text(encoding="utf-8")))
    extras["backup"] = as_extras(
        await freeze_document("frz_c.txt", tenant_id=TENANT_A, delete_after_export=False)
    )
    extras["freeze"] = as_extras(await freeze_document("frz_c.txt", tenant_id=TENANT_A))
    extras["restore_stubs"] = as_extras(
        await restore_document("frz_c.txt", tenant_id=TENANT_A, concurrency=1)
    )
    extras["freeze_keep"] = as_extras(
        await freeze_document(
            job_id="job-f",
            tenant_id=TENANT_A,
            keep_structure_nodes=True,
            keep_annotations=True,
            keep_extraction_results=True,
        )
    )
    extras["restore_folder"] = as_extras(
        await restore_document(job_id="job-f", tenant_id=TENANT_A, concurrency=1)
    )
    await freeze_document("frz_d.txt", tenant_id=TENANT_A, delete_after_export=False)
    await delete_document("frz_d.txt", tenant_id=TENANT_A)
    extras["restore_snapshot"] = as_extras(
        await restore_document("frz_d.txt", tenant_id=TENANT_A, concurrency=1)
    )
    return extras


async def s8_tenants(env: Env) -> dict:
    """Two tenants and a public document with the same path: ingest, extract, delete."""
    from scinr.newton.ingest.deletion import collect_orphans, delete_document

    extras: dict[str, Any] = {}
    for tenant in (TENANT_A, TENANT_B, None):
        await env.full("shared.txt", folder="Shared", tenant=tenant)
    extras["delete_a"] = as_extras(await delete_document("Shared/shared.txt", tenant_id=TENANT_A))
    extras["collect_b"] = as_extras(await collect_orphans(tenant_id=TENANT_B))
    extras["delete_public"] = as_extras(await delete_document("Shared/shared.txt", tenant_id=None))
    return extras


async def s9_preconditions(env: Env) -> dict:
    """Pre-checks of the annotation and extraction agents."""
    from scinr.newton.annotation.agent import run_annotation_agent
    from scinr.newton.entity_extraction.agent import run_entity_extraction_agent
    from scinr.newton.exceptions import PreconditionError

    extras: dict[str, Any] = {}
    cases = {
        "annotation_missing": lambda: run_annotation_agent("ghost.txt", tenant_id=TENANT_A),
        "extraction_missing": lambda: run_entity_extraction_agent("ghost.txt", tenant_id=TENANT_A),
        "extraction_unannotated": lambda: run_entity_extraction_agent(
            "plain.txt", tenant_id=TENANT_A, doc_path="plain.txt"
        ),
    }
    await env.ingest("plain.txt", tenant=TENANT_A)
    for label, call in cases.items():
        try:
            await call()
        except PreconditionError as exc:
            extras[label] = str(exc)
        else:
            raise AssertionError(f"{label}: PreconditionError expected")
    return extras


SCENARIOS: dict[str, Callable[[Env], Awaitable[dict]]] = {
    fn.__name__: fn
    for fn in (
        s1_ingest_versions,
        s2_replace_mixed,
        s3_annotation,
        s4_extraction,
        s5_tabular,
        s6_delete_gc,
        s7_freeze_restore,
        s8_tenants,
        s9_preconditions,
    )
}


def prepare_script() -> None:
    set_script(PRODUCT_SCRIPT)
