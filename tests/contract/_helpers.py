"""
tests/contract/_helpers.py -- Small helpers shared by the contract tests.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

from scinr.newton.models.document_structure import Document, StructureNode
from tests.golden._llm import deterministic_uuid4
from tests.golden._scenarios import product_structure

Dump = Callable[[], dict[str, Any]]


def make_document(
    name: str,
    *,
    folder: str = "",
    tenant: str | None = None,
    structure: list[StructureNode] | None = None,
    job_id: str | None = None,
    created_by: str | None = None,
) -> Document:
    """A document ready for ``insert_document_graph`` (no raw file needed)."""
    return Document(
        document_name=name,
        doc_path=f"{folder}/{name}" if folder else name,
        raw_file_id=f"raw-{name}",
        tenant_id=tenant,
        job_id=job_id,
        created_by_user_id=created_by,
        document_structure=structure if structure is not None else product_structure(),
    )


def labels(graph: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Nodes of the dump that carry *label*."""
    return [n for n in graph["nodes"] if label in n["labels"]]


def rels(graph: dict[str, Any], rel_type: str) -> list[dict[str, Any]]:
    """Relationships of the dump of type *rel_type*."""
    return [r for r in graph["relationships"] if r["type"] == rel_type]


def documents(graph: dict[str, Any], *, tenant: str | None = None) -> list[dict[str, Any]]:
    """``:Document`` nodes of the dump, optionally of one stored tenant."""
    docs = labels(graph, "Document")
    if tenant is None:
        return docs
    return [d for d in docs if d["props"].get("tenant_id") == tenant]


def diff(graph_a: dict[str, Any], graph_b: dict[str, Any]) -> str:
    """What is in one dump and not in the other (first few of each)."""

    def items(graph: dict[str, Any]) -> set[str]:
        return {json.dumps(x, sort_keys=True) for x in graph["nodes"] + graph["relationships"]}

    a, b = items(graph_a), items(graph_b)
    return f"only in first: {sorted(a - b)[:3]}\nonly in second: {sorted(b - a)[:3]}"


def same(graph_a: dict[str, Any], graph_b: dict[str, Any]) -> bool:
    return json.dumps(graph_a, sort_keys=True) == json.dumps(graph_b, sort_keys=True)


def without(graph: dict[str, Any], *drop: str) -> dict[str, Any]:
    """The dump minus the nodes that carry any label of *drop* and the relationships on them."""
    gone = {n["key"] for n in graph["nodes"] if set(n["labels"]) & set(drop)}
    return {
        "nodes": [n for n in graph["nodes"] if n["key"] not in gone],
        "relationships": [
            r for r in graph["relationships"] if r["from"] not in gone and r["to"] not in gone
        ],
    }


async def twice(
    dump: Dump,
    action: Callable[[], Awaitable[Any]],
    *,
    ignore: tuple[str, ...] = (),
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run *action* twice and return the dump, asserting the second run changed nothing.

    Each run has its own deterministic ``uuid4`` (the same sequence), so a write that
    draws a random uid is still comparable. Nodes with a label in *ignore* (and their
    relationships) are left out of the comparison and of the returned dump. With
    *deterministic* off the real ``uuid4`` is used (needed when a previous run leaves
    nodes behind whose uid a deterministic run would draw again).
    """
    scope = deterministic_uuid4 if deterministic else nullcontext
    with scope():
        await action()
    first = without(dump(), *ignore)
    with scope():
        await action()
    second = without(dump(), *ignore)
    assert same(first, second), "second run changed the graph"
    return second
