# utils/tenancy.py
"""Single normalization point for the multi-tenant ``tenant_id``.

The tenant is part of every document's identity: two tenants ingesting the
same path get two independent documents (their own ``:Document``, folders,
versions, structure, annotations and extractions). A document ingested with
no tenant is **public** — readable by every tenant, but still its own upload
whose content is never merged with any tenant's.

"Public" is stored in the graph as the reserved value :data:`PUBLIC_TENANT`,
never as ``null``:

- Neo4j does not index nulls, so a ``tenant_id = $t OR tenant_id IS NULL``
  read filter scans the whole label, while ``tenant_id IN [$t, '__public__']``
  is an index seek.
- ``MERGE`` cannot use a null property in its pattern, and a uniqueness
  constraint does not apply to nodes where a key property is null — a
  non-null value lets ``(tenant_id, path, version)`` be both the ``MERGE``
  key and a real constraint for public documents too.

On the write side ``None`` and ``"__public__"`` are the same thing (public):
:func:`tenant_key` maps both to :data:`PUBLIC_TENANT`. On the read side the
stored value is exposed as is (``"__public__"`` included), so that ``None`` is
free to mean "do not filter by tenant" (see ``navigation.scope``).
"""
from __future__ import annotations

PUBLIC_TENANT = "__public__"
"""Stored ``tenant_id`` of a public (tenant-less) document and of everything derived from it."""


def tenant_key(tenant_id: str | None) -> str:
    """Return the value stored in the graph for *tenant_id*: the tenant, or
    :data:`PUBLIC_TENANT` when it is ``None``.

    Idempotent: ``"__public__"`` (the stored value) is accepted and means public,
    the same as ``None``.

    Raises:
        ValueError: If *tenant_id* is empty (almost certainly a caller bug).

    Examples:
        >>> tenant_key(None)
        '__public__'
        >>> tenant_key("__public__")
        '__public__'
        >>> tenant_key("acme")
        'acme'
    """
    if tenant_id is None:
        return PUBLIC_TENANT
    if tenant_id == "":
        raise ValueError("tenant_id must be a non-empty string, or None for public documents.")
    return tenant_id


def require_stored_tenant(stored: str | None, node_full_id: str) -> str:
    """Return the stored tenant key read off a StructureNode, refusing a
    missing one: hashing a null tenant into a uid would silently create a
    bucket shared by every such node, outside any tenant.

    Raises:
        RuntimeError: If *stored* is ``None`` (a node ingested before tenancy).
    """
    if stored is None:
        raise RuntimeError(
            f"StructureNode {node_full_id!r} has no tenant_id; re-ingest its document."
        )
    return stored
