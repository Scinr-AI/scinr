"""Deterministic synthetic corpus written THROUGH ``GraphStore`` (never raw SQL / Cypher).

Every document is a pure function of ``(SEED, index, scale)``, so any document can be rebuilt
on its own (the run needs that to pick samples without keeping the corpus in memory).

Shape of one document (``scale=1``), as the plan (WP7.2) asks for::

    1    Document             (no folder: the path is the file name, no folder Documents)
    200  StructureNode        (1 root, 9 sections, 190 leaves; one tree, so a "structure tree"
                               read of the document returns all of them)
    400  InfoUnit             (2 per structure node)
    50   ModelInstance        (class ``BenchModel``, keyed by ``code``) under 25 ExtractionResults,
                               with 2 relations per instance: ``SUPPLIED_BY`` (i -> i+1) and
                               ``RELATED_TO`` (i -> i+5), indices modulo 50
    20   Entity               (10 triples on one more ExtractionResult)

Documents are spread over :data:`TENANTS` tenants; one in ten is public (``tenant_id=None``).

``python scripts/sql_bench/corpus.py --nodes 1000000`` prints the plan (documents, nodes per
label) without touching any database.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from scinr.newton.ingest.nodes import make_structure_node_id
from scinr.newton.models.document_structure import Document, InfoUnit, NodeRole, StructureNode
from scinr.newton.utils.tenancy import tenant_key
from scinr.newton.utils.uid import make_instance_uid, make_uid, normalize_key

SEED = 20261119
TENANTS = tuple(f"bench-t{i}" for i in range(10))
PUBLIC_EVERY = 10  # documents whose index % 10 == 9 are public: 10 %
MODEL_CLASS = "BenchModel"
SUPPLIED_BY = "SUPPLIED_BY"
RELATED_TO = "RELATED_TO"
VOCABULARY_SIZE = 800
SEARCH_VOCABULARY = 300  # search terms come from the most frequent words, so they hit

BASE_SECTIONS = 9
BASE_STRUCTURE_NODES = 200
BASE_INFO_UNITS_PER_NODE = 2
BASE_INSTANCES = 50
BASE_EXTRACTION_RESULTS = 25  # model extractions: 2 instances each
BASE_TRIPLES = 10  # 20 distinct entities

_SYLLABLES = (
    "ba",
    "be",
    "bo",
    "ca",
    "co",
    "da",
    "de",
    "fi",
    "ga",
    "ke",
    "la",
    "le",
    "lo",
    "ma",
    "me",
    "mi",
    "na",
    "no",
    "pa",
    "re",
    "ri",
    "sa",
    "se",
    "ta",
    "te",
    "to",
    "va",
    "ve",
    "xo",
    "zi",
)


def _vocabulary() -> list[str]:
    rng = random.Random(SEED)
    words: set[str] = set()
    while len(words) < VOCABULARY_SIZE:
        words.add("".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


VOCABULARY = _vocabulary()
# Zipf-like weights: word i is drawn ~1/(i+1) as often as word 0.
_WEIGHTS = [1.0 / (i + 1) for i in range(VOCABULARY_SIZE)]


def search_terms(count: int, seed: int) -> list[str]:
    """*count* single-word search terms from the frequent end of the vocabulary."""
    rng = random.Random(seed)
    return [VOCABULARY[rng.randrange(SEARCH_VOCABULARY)] for _ in range(count)]


@dataclass(frozen=True)
class DocSpec:
    """One document ready to write, with the sample handles the run needs."""

    index: int
    scale: int
    tenant: str | None  # public-API form (None = public)
    stored_tenant: str  # stored form ("__public__" for public)
    doc_path: str
    document: Document
    leaf_paths: list[str]  # node_path of each leaf, in creation order
    node_paths: list[str]  # node_path of every structure node
    root_id: str  # full id of the root StructureNode

    def node_full_id(self, node_path: str) -> str:
        return make_structure_node_id(self.stored_tenant, self.doc_path, 1, node_path)

    def instance_code(self, k: int) -> str:
        return f"d{self.index}-m{k}"

    def instance_uid(self, k: int) -> str:
        return make_instance_uid(
            MODEL_CLASS, {"code": normalize_key(self.instance_code(k))}, self.stored_tenant
        )

    @property
    def instances(self) -> int:
        return BASE_INSTANCES * self.scale


def tenant_of(index: int) -> str | None:
    """Tenant (public-API form) of document *index*: deterministic, 10 % public."""
    if index % PUBLIC_EVERY == PUBLIC_EVERY - 1:
        return None
    return TENANTS[index % len(TENANTS)]


def _text(rng: random.Random, words: int) -> str:
    return " ".join(rng.choices(VOCABULARY, weights=_WEIGHTS, k=words))


def build_document(index: int, scale: int = 1) -> DocSpec:
    """The document *index* (``scale`` multiplies every count: used for the large document)."""
    rng = random.Random(f"{SEED}:{index}:{scale}")
    tenant = tenant_of(index)
    stored = tenant_key(tenant)
    doc_path = f"d{index:08d}.txt"
    total = BASE_STRUCTURE_NODES * scale
    sections = BASE_SECTIONS * scale
    leaves = total - 1 - sections

    node_paths: list[str] = []
    leaf_paths: list[str] = []

    def make_node(node_id: str, role: NodeRole, order: int) -> StructureNode:
        return StructureNode(
            node_id=node_id,
            title=_text(rng, rng.randint(2, 5)),
            role=role,
            appearance_order=order,
            info_units=[
                InfoUnit(title=_text(rng, rng.randint(3, 6)), order=i, description=_text(rng, 28))
                for i in range(BASE_INFO_UNITS_PER_NODE)
            ],
        )

    root = make_node("n0", NodeRole.SECTION, 1)
    node_paths.append("n0")
    section_nodes: list[StructureNode] = []
    for s in range(sections):
        section = make_node(f"s{s}", NodeRole.SECTION, s + 1)
        section_nodes.append(section)
        node_paths.append(f"n0/s{s}")
        root.children.append(section)
    for leaf in range(leaves):
        s = leaf % sections
        node = make_node(f"l{leaf}", NodeRole.SUBSECTION, len(section_nodes[s].children) + 1)
        section_nodes[s].children.append(node)
        path = f"n0/s{s}/l{leaf}"
        node_paths.append(path)
        leaf_paths.append(path)

    document = Document(
        document_name=doc_path,
        doc_path=doc_path,
        raw_file_id=f"raw-{doc_path}",
        tenant_id=tenant,
        document_structure=[root],
    )
    return DocSpec(
        index=index,
        scale=scale,
        tenant=tenant,
        stored_tenant=stored,
        doc_path=doc_path,
        document=document,
        leaf_paths=leaf_paths,
        node_paths=node_paths,
        root_id=make_structure_node_id(stored, doc_path, 1, "n0"),
    )


def nodes_per_document(scale: int = 1) -> int:
    """Graph nodes one document adds (every label: the figure ``--nodes`` is divided by)."""
    structure = BASE_STRUCTURE_NODES * scale
    info_units = structure * BASE_INFO_UNITS_PER_NODE
    instances = BASE_INSTANCES * scale
    entities = 2 * BASE_TRIPLES * scale
    results = BASE_EXTRACTION_RESULTS * scale + scale  # model extractions + one triple extraction
    return 1 + structure + info_units + instances + entities + results


def documents_for(nodes: int) -> int:
    """Number of documents whose nodes add up to about *nodes* (at least 1)."""
    return max(1, round(nodes / nodes_per_document()))


async def write_catalog(store: Any) -> None:
    """The one ``CatalogModel`` the extractions link to (so ``USES_PRIMARY_MODEL`` exists)."""
    async with store.annotation.catalog_sink() as sink:
        await sink.merge_catalog_model(MODEL_CLASS, "Synthetic benchmark model", True)


async def write_extractions(store: Any, spec: DocSpec) -> None:
    """ExtractionResults, ModelInstances (+ relations) and Entities of one document.

    Follows the order of emission of ``entity_extraction/graph_mapper.py``.
    """
    from scinr.newton.graph.types import Provenance

    stamp = datetime.now(UTC).isoformat()
    scale = spec.scale
    instances = spec.instances
    per_result = instances // (BASE_EXTRACTION_RESULTS * scale)
    async with store.extraction.sink() as sink:
        first = spec.node_full_id(spec.leaf_paths[0])
        raw = await sink.read_node_provenance(first)
        if raw is None:
            raise RuntimeError(f"structure node not found: {first!r} (id format changed?)")
        prov = Provenance(raw.tenant_id, raw.created_by_user_id, raw.job_id)
        for r in range(BASE_EXTRACTION_RESULTS * scale):
            node_full_id = spec.node_full_id(spec.leaf_paths[r])
            er_uid = make_uid("er", spec.stored_tenant, spec.doc_path, str(r))
            await sink.create_extraction_result(
                node_full_id, er_uid, spec.doc_path, MODEL_CLASS, stamp, prov
            )
            await sink.link_catalog_model(er_uid, MODEL_CLASS, primary=True)
            for j in range(per_result):
                k = r * per_result + j
                code = spec.instance_code(k)
                await sink.merge_keyed_instance(
                    "ExtractionResult",
                    er_uid,
                    "HAS_ITEM",
                    spec.instance_uid(k),
                    MODEL_CLASS,
                    {"code": normalize_key(code)},
                    prov,
                    None,
                )
        for k in range(instances):
            await sink.set_instance_properties(
                "ModelInstance", spec.instance_uid(k), {"code": spec.instance_code(k), "seq": k}
            )
            for rel_type, step in ((SUPPLIED_BY, 1), (RELATED_TO, 5)):
                await sink.link_instance_relationship(
                    "ModelInstance",
                    spec.instance_uid(k),
                    spec.instance_uid((k + step) % instances),
                    rel_type,
                )
        # Triples: one more ExtractionResult, on the node after the last model extraction.
        triple_node = spec.node_full_id(spec.leaf_paths[BASE_EXTRACTION_RESULTS * scale])
        triple_uid = make_uid("er-triple", spec.stored_tenant, spec.doc_path)
        await sink.create_triple_extraction_result(
            triple_node, triple_uid, spec.doc_path, stamp, prov
        )
        for t in range(BASE_TRIPLES * scale):
            subject, obj = f"subject d{spec.index} t{t}", f"object d{spec.index} t{t}"
            subj_uid = make_uid("entity", spec.stored_tenant, normalize_key(subject))
            obj_uid = make_uid("entity", spec.stored_tenant, normalize_key(obj))
            await sink.merge_entity(subj_uid, subject, normalize_key(subject), prov)
            await sink.merge_entity(obj_uid, obj, normalize_key(obj), prov)
            await sink.link_triple_entity(triple_uid, subj_uid, "subject")
            await sink.link_triple_entity(triple_uid, obj_uid, "object")
            await sink.merge_predicate(subj_uid, obj_uid, "SELLS", "sells")


def write_structure(store: Any, spec: DocSpec) -> None:
    """Document + StructureNodes + InfoUnits, in one ``insert_document_graph`` transaction."""
    store.documents.insert_document_graph(spec.document, 1, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the plan of the synthetic corpus.")
    parser.add_argument("--nodes", type=int, required=True, help="target number of graph nodes")
    args = parser.parse_args()
    docs = documents_for(args.nodes)
    per_doc = nodes_per_document()
    print(f"seed={SEED} tenants={len(TENANTS)} public=1/{PUBLIC_EVERY}")
    print(f"nodes per document: {per_doc}")
    print(f"documents: {docs} -> {docs * per_doc} nodes (target {args.nodes})")


if __name__ == "__main__":
    main()
